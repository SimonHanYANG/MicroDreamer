"""Training script for video prediction model.

Features:
- TensorBoard logging (loss, predicted frames)
- Gradient clipping, cosine LR scheduler
- Checkpoint save/resume
- Mixed precision
- LoRA parameter groups

Usage:
    python scripts/train_video.py --data_dir ./data/raw --output_dir ./outputs
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset
from models.video.video_model import VideoPredictionModel
from models.video.losses import VideoLoss
from models.video.metrics import VideoMetrics
from models.language.encoder import SimpleLanguageEncoder
from utils.logger import setup_logger

logger = setup_logger("train_video")


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(args.output_dir) / "tb_logs" / "video"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        logger.warning("TensorBoard not installed")

    # Dataset
    dataset = MicroDreamerDataset(
        data_dir=args.data_dir,
        action_horizon=cfg.video_model.num_frames,
        low_res=tuple(cfg.video_model.resolution),
        normalize_actions=False,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    # Model - resolution is (H, W) in model, config is [W, H]
    model_res = (cfg.video_model.resolution[1], cfg.video_model.resolution[0])
    model = VideoPredictionModel(
        in_channels=1,
        hidden_dim=256,
        num_frames=cfg.video_model.num_frames,
        resolution=model_res,
        num_layers=4,
        num_heads=8,
        lora_rank=cfg.video_model.lora_rank,
        lora_alpha=cfg.video_model.lora_alpha,
        context_dim=cfg.language.cross_attn_dim,
    ).to(device)
    logger.info(f"Video model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Language encoder (hidden_dim must match video model context_dim)
    lang_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=cfg.language.cross_attn_dim).to(device)

    # Optimizer with separate LR for LoRA params
    lora_params = model.get_lora_params()
    non_lora_params = model.get_non_lora_params()
    optimizer = torch.optim.AdamW([
        {"params": non_lora_params, "lr": cfg.training.learning_rate},
        {"params": lora_params, "lr": cfg.training.learning_rate * 5},  # higher LR for LoRA
        {"params": lang_encoder.parameters(), "lr": cfg.training.learning_rate},
    ], weight_decay=cfg.training.weight_decay)

    total_steps = cfg.training.max_epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.training.warmup_steps, total_steps)

    # Loss
    criterion = VideoLoss()

    # Mixed precision
    scaler = GradScaler(enabled=cfg.training.fp16 and device.type == "cuda")

    # Resume
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        lang_encoder.load_state_dict(ckpt["lang_encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    model.train()
    lang_encoder.train()
    patience = args.patience
    patience_counter = 0

    for epoch in range(start_epoch, cfg.training.max_epochs):
        epoch_loss = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            frames = batch["low_res_frames"].to(device)  # (B, T, 1, H, W)
            if frames.shape[1] < 8:
                continue

            input_frames = frames[:, :4]
            target_frames = frames[:, 4:8]

            B = frames.shape[0]
            lang_ids = torch.randint(0, 100, (B, 16), device=device)
            lang_ctx = lang_encoder(lang_ids)

            with autocast(enabled=cfg.training.fp16 and device.type == "cuda"):
                pred_frames = model(input_frames, lang_context=lang_ctx, num_pred=4)
                losses = criterion(pred_frames, target_frames)

            optimizer.zero_grad()
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses["loss"].item()
            global_step += 1

            if batch_idx % cfg.training.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Batch {batch_idx}: loss={losses['loss'].item():.4f} lr={lr:.2e}")
                if writer:
                    writer.add_scalar("train/loss", losses["loss"].item(), global_step)
                    writer.add_scalar("train/l1", losses["l1"].item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

            # Log sample images periodically
            if writer and batch_idx == 0 and epoch % 5 == 0:
                # Log first frame of first sample in batch
                writer.add_image("input/frame0", input_frames[0, 0], epoch)
                writer.add_image("pred/frame0", pred_frames[0, 0], epoch)
                writer.add_image("target/frame0", target_frames[0, 0], epoch)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f} lr={lr:.2e} time={epoch_time:.1f}s")
        scheduler.step()

        if writer:
            writer.add_scalar("epoch/loss", avg_loss, epoch)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "lang_encoder": lang_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(ckpt, ckpt_dir / f"video_ckpt_epoch{epoch}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, ckpt_dir / "video_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs")
            break

    if writer:
        writer.close()
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)
