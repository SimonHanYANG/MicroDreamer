"""Training script for action prediction model.

Features:
- TensorBoard logging (loss curves, learning rate)
- Gradient clipping
- Cosine LR scheduler with warmup
- Checkpoint save/resume
- Mixed precision training
- Evaluation metrics

Usage:
    python scripts/train_action.py --data_dir ./data/raw --output_dir ./outputs --simple_lang
    python scripts/train_action.py --data_dir ./data/raw --resume ./outputs/checkpoints/ckpt_epoch5.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset
from models.action.action_model import ActionPredictionModel
from models.action.metrics import ActionMetrics
from utils.logger import setup_logger

logger = setup_logger("train_action")


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
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
        log_dir = Path(args.output_dir) / "tb_logs" / "action"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logging to {log_dir}")
    except ImportError:
        logger.warning("TensorBoard not installed, skipping logging")

    # Dataset
    dataset = MicroDreamerDataset(
        data_dir=args.data_dir,
        action_horizon=cfg.action_model.action_horizon,
        low_res=tuple(cfg.preprocessing.low_res),
        normalize_actions=cfg.preprocessing.normalize_actions,
        use_simple_lang=args.simple_lang,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    # Model
    model = ActionPredictionModel(
        tile_size=448,
        num_tiles=cfg.action_model.num_tiles,
        hidden_dim=cfg.action_model.hidden_dim,
        visual_layers=cfg.action_model.num_layers,
        visual_heads=cfg.action_model.num_heads,
        action_dim=cfg.action_model.action_dim,
        action_horizon=cfg.action_model.action_horizon,
        action_layers=cfg.action_model.num_layers,
        action_heads=cfg.action_model.num_heads,
        dropout=cfg.action_model.dropout,
        use_simple_lang=args.simple_lang,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler
    total_steps = cfg.training.max_epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.training.warmup_steps, total_steps)

    # Mixed precision
    scaler = GradScaler(enabled=cfg.training.fp16 and device.type == "cuda")

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # Checkpoint dir
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    patience = args.patience
    patience_counter = 0

    for epoch in range(start_epoch, cfg.training.max_epochs):
        epoch_loss = 0
        epoch_metrics = ActionMetrics()
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            tiles = batch["high_res_tiles"].to(device)  # (B, num_tiles, 1, 448, 448)
            actions = batch["actions"].to(device)

            lang_text = batch.get("task_description") if args.simple_lang else None
            lang_ids = None
            if not args.simple_lang:
                lang_ids = batch.get("lang_input_ids")
                if lang_ids is not None:
                    lang_ids = lang_ids.to(device)

            with autocast(enabled=cfg.training.fp16 and device.type == "cuda"):
                loss = model.training_loss(tiles, actions, lang_input_ids=lang_ids, lang_text=lang_text)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # Get predictions for metrics (every N steps)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    pred_actions = model.predict_actions(tiles, lang_input_ids=lang_ids, lang_text=lang_text)
                pred_np = pred_actions.cpu().numpy()
                gt_np = actions.cpu().numpy()
                if dataset.normalizer is not None:
                    pred_np = dataset.normalizer.denormalize(pred_np)
                    gt_np = dataset.normalizer.denormalize(gt_np)
                epoch_metrics.update(pred_np, gt_np)

            # Logging
            if batch_idx % cfg.training.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f} lr={lr:.2e}")
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(len(dataloader), 1)
        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f} lr={lr:.2e} time={epoch_time:.1f}s")
        scheduler.step()

        if writer:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            metrics = epoch_metrics.compute()
            for k, v in metrics.items():
                writer.add_scalar(f"epoch/{k}", v, epoch)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(ckpt, ckpt_dir / f"action_ckpt_epoch{epoch}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, ckpt_dir / "action_best.pt")
            patience_counter = 0
            logger.info(f"  New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
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
    parser.add_argument("--simple_lang", action="store_true")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()
    train(args)
