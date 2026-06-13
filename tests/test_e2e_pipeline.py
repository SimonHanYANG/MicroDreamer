"""End-to-end pipeline test for MicroDreamer.

Tests the full workflow:
1. Generate mock training data
2. Load dataset (MicroDreamerDataset)
3. Train action prediction model (2 epochs)
4. Train video prediction model (2 epochs)
5. Evaluate both models
6. Run inference (predict actions + future frames)

Run standalone: python tests/test_e2e_pipeline.py
Run via runner: python tests/run_all_tests.py
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset, create_dummy_dataset
from models.action.action_model import ActionPredictionModel
from models.action.metrics import ActionMetrics
from models.video.video_model import VideoPredictionModel
from models.video.losses import VideoLoss
from models.video.metrics import VideoMetrics
from models.language.encoder import SimpleLanguageEncoder, encode_text_simple


# ============================================================
# Test 1: Mock data generation
# ============================================================
def test_mock_data_generation():
    """Generate mock data and verify structure."""
    print("\n" + "=" * 60)
    print("TEST 1: Mock Data Generation")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=3, frames_per_episode=30)

        # Verify directory structure
        episodes = sorted([d for d in tmpdir.iterdir() if d.is_dir()])
        assert len(episodes) == 3, f"Expected 3 episodes, got {len(episodes)}"

        for ep_dir in episodes:
            assert (ep_dir / "data.npz").exists(), f"Missing data.npz in {ep_dir}"
            assert (ep_dir / "metadata.json").exists(), f"Missing metadata.json in {ep_dir}"

            data = np.load(ep_dir / "data.npz")
            assert data["frames"].shape == (30, 1200, 1600), f"Wrong frame shape: {data['frames'].shape}"
            assert data["stage_positions"].shape == (30, 2), f"Wrong stage shape: {data['stage_positions'].shape}"
            assert data["pipette_positions"].shape == (30, 3), f"Wrong pipette shape: {data['pipette_positions'].shape}"
            assert data["timestamps"].shape == (30,), f"Wrong timestamp shape: {data['timestamps'].shape}"

            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "task_description" in meta, "Missing task_description"
            assert "subgoals" in meta, "Missing subgoals"

        print("[PASS] Mock data generation: structure verified")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 2: Dataset loading
# ============================================================
def test_dataset_loading():
    """Load dataset and verify sample format."""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset Loading")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=2, frames_per_episode=30)

        dataset = MicroDreamerDataset(
            data_dir=str(tmpdir),
            action_horizon=4,
            low_res=(128, 96),
            normalize_actions=True,
        )

        assert len(dataset) > 0, "Dataset is empty"
        sample = dataset[0]

        # Check all expected keys
        expected_keys = ["high_res_tiles", "low_res_frames", "actions", "task_description"]
        for k in expected_keys:
            assert k in sample, f"Missing key: {k}"

        # Check tensor shapes
        tiles = sample["high_res_tiles"]
        assert tiles.dim() == 4, f"Expected 4D tiles, got {tiles.dim()}D"
        assert tiles.shape[1] == 1, f"Expected 1 channel, got {tiles.shape[1]}"
        assert tiles.shape[2] == 448 and tiles.shape[3] == 448, f"Expected 448x448 tiles, got {tiles.shape[2:]}"

        lr_frames = sample["low_res_frames"]
        assert lr_frames.dim() == 4, f"Expected 4D low_res, got {lr_frames.dim()}D"
        assert lr_frames.shape[1] == 1, f"Expected 1 channel, got {lr_frames.shape[1]}"
        assert lr_frames.shape[2] == 96 and lr_frames.shape[3] == 128, f"Expected 96x128 low_res, got {lr_frames.shape[2:]}"

        actions = sample["actions"]
        assert actions.dim() == 2, f"Expected 2D actions, got {actions.dim()}D"
        assert actions.shape[1] == 5, f"Expected 5 action dims, got {actions.shape[1]}"

        # Check normalizer
        assert dataset.normalizer is not None, "Normalizer should be fitted"

        print(f"[PASS] Dataset loading: {len(dataset)} samples, shapes verified")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 3: Action model training (2 epochs)
# ============================================================
def test_action_model_training():
    """Train action model for 2 epochs and verify loss decreases."""
    print("\n" + "=" * 60)
    print("TEST 3: Action Model Training")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=2, frames_per_episode=30)

        device = torch.device("cpu")
        dataset = MicroDreamerDataset(
            data_dir=str(tmpdir),
            action_horizon=4,
            low_res=(128, 96),
            normalize_actions=True,
            use_simple_lang=True,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        model = ActionPredictionModel(
            hidden_dim=128,
            visual_layers=2,
            visual_heads=4,
            action_dim=5,
            action_horizon=4,
            action_layers=2,
            action_heads=4,
            use_simple_lang=True,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Action model: {total_params / 1e6:.2f}M params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        model.train()
        for epoch in range(2):
            epoch_loss = 0
            for batch in dataloader:
                tiles = batch["high_res_tiles"].to(device)
                actions = batch["actions"].to(device)
                lang_text = batch.get("task_description")

                loss = model.training_loss(tiles, actions, lang_text=lang_text)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            losses.append(avg_loss)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt = {"model": model.state_dict(), "epoch": 1}
        torch.save(ckpt, ckpt_dir / "action_best.pt")

        # Verify checkpoint loads
        model2 = ActionPredictionModel(
            hidden_dim=128, visual_layers=2, visual_heads=4,
            action_dim=5, action_horizon=4, action_layers=2, action_heads=4,
            use_simple_lang=True,
        )
        model2.load_state_dict(torch.load(ckpt_dir / "action_best.pt", map_location="cpu", weights_only=False)["model"])
        print(f"  Checkpoint saved and loaded: {ckpt_dir / 'action_best.pt'}")

        print(f"[PASS] Action model training: loss {losses[0]:.4f} → {losses[-1]:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 4: Video model training (2 epochs)
# ============================================================
def test_video_model_training():
    """Train video model for 2 epochs and verify loss decreases."""
    print("\n" + "=" * 60)
    print("TEST 4: Video Model Training")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=2, frames_per_episode=30)

        device = torch.device("cpu")
        dataset = MicroDreamerDataset(
            data_dir=str(tmpdir),
            action_horizon=8,
            low_res=(128, 96),
            normalize_actions=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        # Video model (H, W) = (96, 128)
        model = VideoPredictionModel(
            in_channels=1,
            hidden_dim=128,
            num_frames=8,
            resolution=(96, 128),
            num_layers=2,
            num_heads=4,
            lora_rank=8,
            lora_alpha=16,
            context_dim=256,
        ).to(device)

        lang_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=256).to(device)
        criterion = VideoLoss()

        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for p in model.get_lora_params())
        print(f"  Video model: {total_params / 1e6:.2f}M total, {lora_params / 1e3:.1f}K LoRA")

        optimizer = torch.optim.AdamW([
            {"params": model.get_non_lora_params(), "lr": 1e-4},
            {"params": model.get_lora_params(), "lr": 5e-4},
            {"params": lang_encoder.parameters(), "lr": 1e-4},
        ])

        losses = []
        model.train()
        lang_encoder.train()
        for epoch in range(2):
            epoch_loss = 0
            for batch in dataloader:
                frames = batch["low_res_frames"].to(device)
                if frames.shape[1] < 8:
                    continue

                input_frames = frames[:, :4]
                target_frames = frames[:, 4:8]
                B = frames.shape[0]
                lang_ids = torch.randint(0, 100, (B, 16), device=device)
                lang_ctx = lang_encoder(lang_ids)

                pred_frames = model(input_frames, lang_context=lang_ctx, num_pred=4)
                loss_dict = criterion(pred_frames, target_frames)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            losses.append(avg_loss)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt = {
            "model": model.state_dict(),
            "lang_encoder": lang_encoder.state_dict(),
            "epoch": 1,
        }
        torch.save(ckpt, ckpt_dir / "video_best.pt")
        print(f"  Checkpoint saved: {ckpt_dir / 'video_best.pt'}")

        # Test LoRA merge
        model.merge_lora()
        print("  LoRA weights merged successfully")

        print(f"[PASS] Video model training: loss {losses[0]:.4f} → {losses[-1]:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 5: Action evaluation
# ============================================================
def test_action_evaluation():
    """Evaluate a trained action model on test data."""
    print("\n" + "=" * 60)
    print("TEST 5: Action Model Evaluation")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=2, frames_per_episode=30)

        device = torch.device("cpu")
        dataset = MicroDreamerDataset(
            data_dir=str(tmpdir),
            action_horizon=4,
            low_res=(128, 96),
            normalize_actions=True,
            use_simple_lang=True,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        model = ActionPredictionModel(
            hidden_dim=128, visual_layers=2, visual_heads=4,
            action_dim=5, action_horizon=4, action_layers=2, action_heads=4,
            use_simple_lang=True,
        ).to(device)
        model.eval()

        metrics = ActionMetrics()
        with torch.no_grad():
            for batch in dataloader:
                tiles = batch["high_res_tiles"].to(device)
                actions = batch["actions"].to(device)
                lang_text = batch.get("task_description")

                pred = model.predict_actions(tiles, lang_text=lang_text)

                pred_np = pred.cpu().numpy()
                gt_np = actions.cpu().numpy()
                if dataset.normalizer is not None:
                    pred_np = dataset.normalizer.denormalize(pred_np)
                    gt_np = dataset.normalizer.denormalize(gt_np)
                metrics.update(pred_np, gt_np)

        results = metrics.compute()
        print(f"  Action MSE:      {results['action_mse']:.4f}")
        print(f"  Action MAE:      {results['action_mae']:.4f}")
        print(f"  Endpoint Error:  {results['endpoint_error']:.4f}")

        for k, v in results.items():
            assert np.isfinite(v), f"Non-finite metric {k}: {v}"

        print("[PASS] Action evaluation: all metrics finite")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 6: Video evaluation
# ============================================================
def test_video_evaluation():
    """Evaluate a trained video model on test data."""
    print("\n" + "=" * 60)
    print("TEST 6: Video Model Evaluation")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="md_e2e_"))
    try:
        create_dummy_dataset(str(tmpdir), num_episodes=2, frames_per_episode=30)

        device = torch.device("cpu")
        dataset = MicroDreamerDataset(
            data_dir=str(tmpdir),
            action_horizon=8,
            low_res=(128, 96),
            normalize_actions=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        model = VideoPredictionModel(
            in_channels=1, hidden_dim=128, num_frames=8,
            resolution=(96, 128), num_layers=2, num_heads=4,
            context_dim=256,
        ).to(device)
        model.eval()

        video_metrics = VideoMetrics()
        with torch.no_grad():
            for batch in dataloader:
                frames = batch["low_res_frames"].to(device)
                if frames.shape[1] < 8:
                    continue
                input_frames = frames[:, :4]
                target_frames = frames[:, 4:8]
                pred_frames = model(input_frames, num_pred=4)
                video_metrics.update(pred_frames.cpu().numpy(), target_frames.cpu().numpy())

        results = video_metrics.compute()
        print(f"  Pixel MSE:           {results['pixel_mse']:.4f}")
        print(f"  PSNR:                {results['psnr']:.2f} dB")
        print(f"  Temporal Consistency:{results['temporal_consistency']:.4f}")

        for k, v in results.items():
            assert np.isfinite(v), f"Non-finite metric {k}: {v}"

        print("[PASS] Video evaluation: all metrics finite")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Test 7: Inference (predict actions + future frames)
# ============================================================
def test_inference():
    """Run inference with random weights and verify output shapes."""
    print("\n" + "=" * 60)
    print("TEST 7: Inference")
    print("=" * 60)

    device = torch.device("cpu")

    # Create models with small config
    action_model = ActionPredictionModel(
        hidden_dim=128, visual_layers=2, visual_heads=4,
        action_dim=5, action_horizon=4, action_layers=2, action_heads=4,
        use_simple_lang=True,
    ).to(device)
    action_model.eval()

    video_model = VideoPredictionModel(
        in_channels=1, hidden_dim=128, num_frames=8,
        resolution=(96, 128), num_layers=2, num_heads=4,
        context_dim=256,
    ).to(device)
    video_model.eval()

    lang_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=256).to(device)

    # Simulate a camera frame (200x160 grayscale)
    frame = np.random.randint(0, 255, (160, 200), dtype=np.uint8)
    task = "Aspirate the target cell"

    # Prepare inputs (same as predict.py)
    from data.preprocessor.frame_processor import prepare_high_res, prepare_low_res

    tiles = prepare_high_res(frame)
    tiles_tensor = torch.tensor(tiles, dtype=torch.float32).unsqueeze(1)
    tiles_batch = tiles_tensor.unsqueeze(0).to(device)
    print(f"  Tiles shape: {tiles_batch.shape}")

    low_res = prepare_low_res(frame, (128, 96))
    low_res_tensor = torch.tensor(low_res, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    print(f"  Low-res shape: {low_res_tensor.shape}")

    lang_ids = encode_text_simple([task]).to(device)
    lang_ctx = lang_encoder(lang_ids)
    print(f"  Language IDs: {lang_ids.shape}, Context: {lang_ctx.shape}")

    # Action prediction
    with torch.no_grad():
        actions = action_model.predict_actions(tiles_batch, lang_input_ids=lang_ids)
    print(f"  Predicted actions: {actions.shape}")
    assert actions.shape == (1, 4, 5), f"Expected (1, 4, 5), got {actions.shape}"

    # Video prediction
    with torch.no_grad():
        pred_frames = video_model(low_res_tensor, lang_context=lang_ctx, num_pred=4)
    print(f"  Predicted frames: {pred_frames.shape}")
    assert pred_frames.shape[0] == 1 and pred_frames.shape[1] == 4, f"Expected (1, 4, ...), got {pred_frames.shape}"
    assert pred_frames.shape[2] == 1, f"Expected 1 channel, got {pred_frames.shape[2]}"

    print("[PASS] Inference: correct output shapes")


if __name__ == "__main__":
    print("=" * 60)
    print("  MicroDreamer End-to-End Pipeline Test Suite")
    print("=" * 60)

    tests = [
        ("Mock Data Generation", test_mock_data_generation),
        ("Dataset Loading", test_dataset_loading),
        ("Action Model Training", test_action_model_training),
        ("Video Model Training", test_video_model_training),
        ("Action Evaluation", test_action_evaluation),
        ("Video Evaluation", test_video_evaluation),
        ("Inference", test_inference),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
