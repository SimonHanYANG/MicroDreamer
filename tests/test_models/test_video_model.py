"""Tests for video prediction model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from models.video.video_model import VideoPredictionModel
from models.video.losses import VideoLoss


def test_video_model_forward():
    model = VideoPredictionModel(
        in_channels=1, hidden_dim=256, num_frames=8,
        resolution=(96, 128), num_layers=2, num_heads=4, context_dim=256,
    )
    input_frames = torch.randn(1, 4, 1, 96, 128)
    lang_ctx = torch.randn(1, 8, 256)
    pred = model(input_frames, lang_context=lang_ctx, num_pred=4)
    assert pred.shape == (1, 4, 1, 96, 128)
    print("  [PASS] video_model_forward")


def test_video_model_no_language():
    model = VideoPredictionModel(
        in_channels=1, hidden_dim=128, num_frames=4,
        resolution=(48, 64), num_layers=2, num_heads=4,
    )
    input_frames = torch.randn(1, 2, 1, 48, 64)
    pred = model(input_frames, num_pred=2)
    assert pred.shape == (1, 2, 1, 48, 64)
    print("  [PASS] video_model_no_language")


def test_video_loss():
    criterion = VideoLoss()
    pred = torch.randn(2, 4, 1, 96, 128)
    target = torch.randn(2, 4, 1, 96, 128)
    losses = criterion(pred, target)
    assert "loss" in losses
    assert "l1" in losses
    assert "temporal" in losses
    assert losses["loss"].dim() == 0
    assert losses["loss"].item() > 0
    print("  [PASS] video_loss")


def test_video_model_params():
    model = VideoPredictionModel(
        in_channels=1, hidden_dim=256, num_frames=8,
        resolution=(96, 128), num_layers=2, num_heads=4,
    )
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    print(f"  Video model params: {num_params / 1e6:.1f}M")
    print("  [PASS] video_model_params")


if __name__ == "__main__":
    print("Running video model tests...")
    test_video_model_forward()
    test_video_model_no_language()
    test_video_loss()
    test_video_model_params()
    print("All video model tests passed!")
