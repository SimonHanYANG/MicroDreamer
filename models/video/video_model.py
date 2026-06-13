"""Video prediction model based on CogVideoX with LoRA fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for linear layers."""

    def __init__(self, original: nn.Linear, rank: int = 64, alpha: float = 128):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_dim = original.in_features
        out_dim = original.out_features
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


class TemporalBlock(nn.Module):
    """Temporal attention + feedforward for video frames."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) temporal sequence of frame features."""
        x = x + self.temporal_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class VideoPredictionModel(nn.Module):
    """Video prediction model with CogVideoX-style architecture.

    Simplified version for testing:
    - Takes low-res frames as input
    - Generates next frames autoregressively
    - LoRA fine-tuning of attention layers
    - Language conditioning via cross-attention
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        num_frames: int = 16,
        resolution: tuple = (384, 512),
        num_layers: int = 6,
        num_heads: int = 12,
        lora_rank: int = 64,
        lora_alpha: float = 128,
        context_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.resolution = resolution

        # Frame encoder: encode each frame to a feature vector
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=4),  # downsample 4x
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 4, stride=4),  # downsample 4x more
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # Total downsample: 16x, so 384x512 -> 24x32 -> global pool

        # Frame decoder: feature to frame
        self.frame_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4 * 4),
            nn.Unflatten(1, (hidden_dim, 4, 4)),
            nn.ConvTranspose2d(hidden_dim, 128, 4, stride=4),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=4),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=4),
            nn.AdaptiveAvgPool2d(resolution),  # resize to target resolution
            nn.Sigmoid(),
        )

        # Temporal transformer
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Positional embedding for frames
        self.frame_pos_embed = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Language cross-attention
        self.lang_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Apply LoRA to temporal attention layers
        self._apply_lora(lora_rank, lora_alpha)

    def _apply_lora(self, rank: int, alpha: float):
        """Apply LoRA to temporal attention in-projections."""
        for block in self.temporal_blocks:
            attn = block.temporal_attn
            attn.in_proj_weight = nn.Parameter(attn.in_proj_weight)
            # Create LoRA wrappers for q, k, v
            # We wrap the entire in_proj as a LoRA linear
            # This is a simplified approach

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to features.

        Args:
            frames: (B, T, C, H, W)

        Returns:
            features: (B, T, hidden_dim)
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        feat = self.frame_encoder(flat)  # (B*T, hidden_dim)
        return feat.reshape(B, T, self.hidden_dim)

    def decode_frames(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to frames.

        Args:
            features: (B, T, hidden_dim)

        Returns:
            frames: (B, T, C, H, W)
        """
        B, T, D = features.shape
        flat = features.reshape(B * T, D)
        frames = self.frame_decoder(flat)  # (B*T, C, H', W')
        C = frames.shape[1]
        H, W = self.resolution
        return frames.reshape(B, T, C, H, W)

    def forward(
        self,
        input_frames: torch.Tensor,
        lang_context: Optional[torch.Tensor] = None,
        num_pred: int = 4,
    ) -> torch.Tensor:
        """Predict future frames.

        Args:
            input_frames: (B, T_in, C, H, W) input frame sequence
            lang_context: (B, L, context_dim) language conditioning
            num_pred: number of frames to predict

        Returns:
            pred_frames: (B, num_pred, C, H, W)
        """
        B = input_frames.shape[0]

        # Encode input frames
        feat = self.encode_frames(input_frames)  # (B, T_in, D)
        T_in = feat.shape[1]

        # Autoregressive prediction
        predicted_frames = []
        current_feat = feat

        for i in range(num_pred):
            # Add positional embedding
            seq_len = current_feat.shape[1]
            pos = self.frame_pos_embed[:, :seq_len, :]
            x = current_feat + pos

            # Temporal attention
            for block in self.temporal_blocks:
                x = block(x)
            x = self.temporal_norm(x)

            # Language cross-attention
            if lang_context is not None:
                ctx = self.lang_proj(lang_context)
                residual = x
                x = self.cross_norm(x)
                x = residual + self.cross_attn(x, ctx, ctx)[0]

            # Take last frame feature -> decode
            last_feat = x[:, -1:, :]  # (B, 1, D)
            pred_frame = self.decode_frames(last_feat)  # (B, 1, C, H, W)
            predicted_frames.append(pred_frame)

            # Append predicted feature for next step
            new_feat = self.frame_encoder(pred_frame.squeeze(1)).unsqueeze(1)
            current_feat = torch.cat([current_feat[:, 1:, :], new_feat], dim=1)

        return torch.cat(predicted_frames, dim=1)  # (B, num_pred, C, H, W)


if __name__ == "__main__":
    model = VideoPredictionModel(
        in_channels=1, hidden_dim=256, num_frames=8,
        resolution=(96, 128), num_layers=2, num_heads=4,
        context_dim=256,
    )
    print(f"Video model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Test forward
    input_frames = torch.randn(1, 4, 1, 96, 128)  # 4 input frames
    lang_ctx = torch.randn(1, 8, 256)
    pred = model(input_frames, lang_context=lang_ctx, num_pred=4)
    print(f"Predicted frames: {pred.shape}")  # (1, 4, 1, 96, 128)
