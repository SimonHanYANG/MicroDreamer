"""End-to-end integration test: collect → train → evaluate."""

import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import load_config
from hardware.factory import create_camera, create_stage, create_pipette
from data.collector.synchronizer import DataSynchronizer
from data.collector.recorder import DataRecorder
from data.dataset import MicroDreamerDataset, create_dummy_dataset
from models.action.action_model import ActionPredictionModel
from models.action.metrics import ActionMetrics
from models.video.video_model import VideoPredictionModel
from models.video.losses import VideoLoss
from models.language.encoder import SimpleLanguageEncoder
import torch
import numpy as np


def test_e2e_pipeline():
    """Full pipeline: data → dataset → model → loss → metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        output_dir = tmpdir / "outputs"

        # Step 1: Create dummy dataset
        print("  [1/5] Creating dataset...")
        create_dummy_dataset(str(data_dir), num_episodes=3, frames_per_episode=20)
        assert data_dir.exists()

        # Step 2: Load dataset
        print("  [2/5] Loading dataset...")
        ds = MicroDreamerDataset(
            str(data_dir), action_horizon=8, low_res=(128, 96), use_simple_lang=True
        )
        assert len(ds) > 0
        sample = ds[0]

        # Step 3: Action model forward + backward
        print("  [3/5] Action model train step...")
        action_model = ActionPredictionModel(
            hidden_dim=128, visual_layers=2, visual_heads=4,
            action_layers=2, action_heads=4, use_simple_lang=True, num_tiles=12,
            action_horizon=8,
        )
        tiles = sample["high_res_tiles"].unsqueeze(0)
        actions = sample["actions"].unsqueeze(0)
        loss = action_model.training_loss(tiles, actions, lang_text=["test task"])
        loss.backward()
        assert loss.item() > 0

        # Step 4: Video model forward + backward
        print("  [4/5] Video model train step...")
        video_model = VideoPredictionModel(
            in_channels=1, hidden_dim=128, num_frames=8,
            resolution=(96, 128), num_layers=2, num_heads=4, context_dim=128,
        )
        lang_enc = SimpleLanguageEncoder(vocab_size=100, hidden_dim=128)
        frames = sample["low_res_frames"].unsqueeze(0)
        if frames.shape[1] >= 8:
            input_f = frames[:, :4]
            target_f = frames[:, 4:8]
            lang_ctx = lang_enc(torch.randint(0, 100, (1, 8)))
            pred_f = video_model(input_f, lang_context=lang_ctx, num_pred=4)
            video_loss = VideoLoss()(pred_f, target_f)
            video_loss["loss"].backward()
            assert video_loss["loss"].item() > 0

        # Step 5: Metrics
        print("  [5/5] Computing metrics...")
        pred_actions = action_model.predict_actions(tiles, lang_text=["test task"])
        action_metrics = ActionMetrics()
        action_metrics.update(pred_actions.detach().numpy(), actions.numpy())
        results = action_metrics.compute()
        assert "action_mse" in results
        assert "endpoint_error" in results

        print("  [PASS] End-to-end pipeline")


def test_virtual_collection():
    """Test virtual device collection."""
    cfg = load_config()
    cfg._data["stage"]["type"] = "virtual"
    cfg._data["pipette"]["type"] = "virtual"

    camera = create_camera(cfg)
    stage = create_stage(cfg)
    pipette = create_pipette(cfg)

    sync = DataSynchronizer(camera, stage, pipette, camera_fps=10)
    sync.start()

    # Move and collect
    stage.move_relative(10, 5)
    pipette.move_relative(0, 0, -5)

    import time
    time.sleep(0.5)

    sample = sync.get_latest_sample()
    sync.stop()

    assert sample is not None
    assert sample.frame.image.shape == (1200, 1600)
    print("  [PASS] Virtual collection")


if __name__ == "__main__":
    print("Running integration tests...")
    test_virtual_collection()
    test_e2e_pipeline()
    print("All integration tests passed!")
