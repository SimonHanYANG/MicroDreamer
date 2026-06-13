"""Video prediction losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoLoss(nn.Module):
    """Combined loss for video prediction.

    Components:
    - L1 reconstruction loss
    - Perceptual loss (LPIPS-like, simplified)
    - Temporal consistency loss
    """

    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.1, temporal_weight: float = 0.05):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight

    def l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pixel-wise L1 loss."""
        return F.l1_loss(pred, target)

    def temporal_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize large differences between consecutive predicted frames."""
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
        diff = pred[:, 1:] - pred[:, :-1]
        return diff.abs().mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute combined loss.

        Args:
            pred: (B, T, C, H, W) predicted frames
            target: (B, T, C, H, W) ground truth frames

        Returns:
            dict with 'loss', 'l1', 'temporal'
        """
        l1 = self.l1_loss(pred, target)
        temporal = self.temporal_loss(pred)

        total = self.l1_weight * l1 + self.temporal_weight * temporal

        return {
            "loss": total,
            "l1": l1.detach(),
            "temporal": temporal.detach(),
        }
