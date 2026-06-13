"""Metrics for video prediction evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict


def pixel_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Pixel-wise MSE between predicted and target frames. (B, T, C, H, W)"""
    return float(np.mean((pred - target) ** 2))


def pixel_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Pixel-wise MAE. (B, T, C, H, W)"""
    return float(np.mean(np.abs(pred - target)))


def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """Simplified SSIM (structural similarity) without external dependencies.

    Computes SSIM per frame and averages. Uses numpy only.
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Flatten batch and time dims
    pred_flat = pred.reshape(-1, *pred.shape[-3:])  # (N, C, H, W)
    target_flat = target.reshape(-1, *target.shape[-3:])

    ssim_vals = []
    for i in range(min(len(pred_flat), 100)):  # limit for speed
        p = pred_flat[i].astype(np.float64)
        t = target_flat[i].astype(np.float64)

        mu_p = np.mean(p)
        mu_t = np.mean(t)
        sigma_p2 = np.var(p)
        sigma_t2 = np.var(t)
        sigma_pt = np.mean((p - mu_p) * (t - mu_t))

        num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p2 + sigma_t2 + C2)
        ssim_vals.append(float(num / den))

    return float(np.mean(ssim_vals))


def temporal_consistency(pred: np.ndarray) -> float:
    """Average frame-to-frame difference. Lower = more consistent. (B, T, C, H, W)"""
    if pred.shape[1] < 2:
        return 0.0
    diffs = np.abs(pred[:, 1:] - pred[:, :-1])
    return float(np.mean(diffs))


def frechet_video_distance(pred_features: np.ndarray, target_features: np.ndarray) -> float:
    """Simplified FVD using feature statistics.

    Args:
        pred_features: (N, D) feature vectors from predicted frames
        target_features: (N, D) feature vectors from target frames

    Returns:
        FVD score (lower = better)
    """
    mu_p = np.mean(pred_features, axis=0)
    mu_t = np.mean(target_features, axis=0)
    sigma_p = np.cov(pred_features.T)
    sigma_t = np.cov(target_features.T)

    diff = mu_p - mu_t
    # Simplified: use trace instead of matrix sqrt
    covmean = sigma_p @ sigma_t
    # Eigenvalue-based sqrt approximation
    eigvals = np.linalg.eigvalsh(covmean)
    covmean_sqrt = np.sum(np.sqrt(np.maximum(eigvals, 0)))

    fvd = float(np.sum(diff ** 2) + np.trace(sigma_p) + np.trace(sigma_t) - 2 * covmean_sqrt)
    return max(fvd, 0.0)


class VideoMetrics:
    """Collects and aggregates video prediction metrics."""

    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Add batch of predicted and target frames."""
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)

        return {
            "pixel_mse": pixel_mse(preds, targets),
            "pixel_mae": pixel_mae(preds, targets),
            "psnr": psnr(preds, targets),
            "ssim": ssim_simple(preds, targets),
            "temporal_consistency": temporal_consistency(preds),
        }

    def reset(self):
        self.preds.clear()
        self.targets.clear()


if __name__ == "__main__":
    np.random.seed(42)
    pred = np.random.rand(5, 4, 1, 96, 128).astype(np.float32)
    target = pred + np.random.randn(5, 4, 1, 96, 128).astype(np.float32) * 0.05

    metrics = VideoMetrics()
    metrics.update(pred, target)
    results = metrics.compute()

    print("Video Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
