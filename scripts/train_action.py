"""Training script for action prediction model."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset
from models.action.action_model import ActionPredictionModel
from utils.logger import setup_logger

logger = setup_logger("train_action")


def train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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
        context_dim=cfg.language.cross_attn_dim,
        dropout=cfg.action_model.dropout,
        use_simple_lang=args.simple_lang,
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Training loop
    model.train()
    for epoch in range(cfg.training.max_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            tiles = batch["high_res_tiles"].to(device)
            actions = batch["actions"].to(device)

            B, T_tiles, C, H, W = tiles.shape
            tiles = tiles.reshape(B * T_tiles, C, H, W).unsqueeze(1)  # flatten tiles

            # Use simple language if specified
            lang_text = batch.get("task_description") if args.simple_lang else None
            lang_ids = None
            if not args.simple_lang:
                lang_ids = batch.get("lang_input_ids")
                if lang_ids is not None:
                    lang_ids = lang_ids.to(device)

            loss = model.training_loss(
                tiles, actions,
                lang_input_ids=lang_ids,
                lang_text=lang_text,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % cfg.training.log_interval == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"action_model_epoch{epoch}.pt")
            logger.info(f"Saved checkpoint: action_model_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--simple_lang", action="store_true", help="Use simple language encoder")
    args = parser.parse_args()
    train(args)
