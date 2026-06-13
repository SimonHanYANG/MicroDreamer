"""Training script for video prediction model."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset
from models.video.video_model import VideoPredictionModel
from models.video.losses import VideoLoss
from models.language.encoder import SimpleLanguageEncoder
from utils.logger import setup_logger

logger = setup_logger("train_video")


def train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dataset
    dataset = MicroDreamerDataset(
        data_dir=args.data_dir,
        action_horizon=cfg.video_model.num_frames,
        low_res=tuple(cfg.video_model.resolution),
        normalize_actions=False,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    # Model
    model = VideoPredictionModel(
        in_channels=1,
        hidden_dim=256,
        num_frames=cfg.video_model.num_frames,
        resolution=tuple(cfg.video_model.resolution),
        num_layers=4,
        num_heads=8,
        lora_rank=cfg.video_model.lora_rank,
        lora_alpha=cfg.video_model.lora_alpha,
        context_dim=cfg.language.cross_attn_dim,
    ).to(device)

    logger.info(f"Video model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Language encoder (simple for testing)
    lang_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=256).to(device)

    # Loss and optimizer
    criterion = VideoLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(lang_encoder.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Training loop
    model.train()
    lang_encoder.train()

    for epoch in range(cfg.training.max_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            frames = batch["low_res_frames"].to(device)  # (B, T, 1, H, W)

            if frames.shape[1] < 8:
                continue

            # Split into input and target
            input_frames = frames[:, :4]
            target_frames = frames[:, 4:8]

            # Language context (dummy for testing)
            B = frames.shape[0]
            lang_ids = torch.randint(0, 100, (B, 16), device=device)
            lang_ctx = lang_encoder(lang_ids)

            # Predict
            pred_frames = model(input_frames, lang_context=lang_ctx, num_pred=4)

            # Loss
            losses = criterion(pred_frames, target_frames)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            total_loss += losses["loss"].item()

            if batch_idx % cfg.training.log_interval == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx}: loss={losses['loss'].item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"video_model_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    train(args)
