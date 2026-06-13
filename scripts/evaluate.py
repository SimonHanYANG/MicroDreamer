"""Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --data_dir ./data/raw --action_ckpt ./outputs/checkpoints/action_best.pt
    python scripts/evaluate.py --data_dir ./data/raw --video_ckpt ./outputs/checkpoints/video_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from data.dataset import MicroDreamerDataset
from models.action.action_model import ActionPredictionModel
from models.action.metrics import ActionMetrics
from models.video.video_model import VideoPredictionModel
from models.video.metrics import VideoMetrics
from utils.logger import setup_logger

logger = setup_logger("evaluate")


def evaluate_action(args, cfg, device):
    """Evaluate action prediction model."""
    dataset = MicroDreamerDataset(
        data_dir=args.data_dir,
        action_horizon=cfg.action_model.action_horizon,
        low_res=tuple(cfg.preprocessing.low_res),
        normalize_actions=cfg.preprocessing.normalize_actions,
        use_simple_lang=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ActionPredictionModel(
        hidden_dim=cfg.action_model.hidden_dim,
        visual_layers=cfg.action_model.num_layers,
        visual_heads=cfg.action_model.num_heads,
        action_dim=cfg.action_model.action_dim,
        action_horizon=cfg.action_model.action_horizon,
        action_layers=cfg.action_model.num_layers,
        action_heads=cfg.action_model.num_heads,
        use_simple_lang=True,
    ).to(device)

    if args.action_ckpt:
        ckpt = torch.load(args.action_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        logger.info(f"Loaded action checkpoint: {args.action_ckpt}")

    model.eval()
    metrics = ActionMetrics()

    with torch.no_grad():
        for batch in dataloader:
            tiles = batch["high_res_tiles"].to(device)
            actions = batch["actions"].to(device)

            B, T_tiles, C, H, W = tiles.shape
            tiles = tiles.reshape(B * T_tiles, C, H, W).unsqueeze(1)

            pred = model.predict_actions(tiles, lang_text=batch.get("task_description"))

            pred_np = pred.cpu().numpy()
            gt_np = actions.cpu().numpy()
            if dataset.normalizer is not None:
                pred_np = dataset.normalizer.denormalize(pred_np)
                gt_np = dataset.normalizer.denormalize(gt_np)

            metrics.update(pred_np, gt_np)

    results = metrics.compute()
    logger.info("Action Evaluation Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_action.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_video(args, cfg, device):
    """Evaluate video prediction model."""
    dataset = MicroDreamerDataset(
        data_dir=args.data_dir,
        action_horizon=cfg.video_model.num_frames,
        low_res=tuple(cfg.video_model.resolution),
        normalize_actions=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = VideoPredictionModel(
        in_channels=1,
        hidden_dim=256,
        num_frames=cfg.video_model.num_frames,
        resolution=tuple(cfg.video_model.resolution),
        num_layers=4,
        num_heads=8,
        context_dim=cfg.language.cross_attn_dim,
    ).to(device)

    if args.video_ckpt:
        ckpt = torch.load(args.video_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        logger.info(f"Loaded video checkpoint: {args.video_ckpt}")

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
    logger.info("Video Evaluation Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_video.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--action_ckpt", type=str, default=None)
    parser.add_argument("--video_ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action_ckpt:
        evaluate_action(args, cfg, device)
    if args.video_ckpt:
        evaluate_video(args, cfg, device)
    if not args.action_ckpt and not args.video_ckpt:
        logger.warning("No checkpoints specified. Use --action_ckpt or --video_ckpt")
