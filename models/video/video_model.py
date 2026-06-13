"""Video prediction model based on CogVideoX with LoRA fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for linear layers."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 64, alpha: float = 128, bias: bool = True):
        super().__init__()
        self.original = nn.Linear(in_dim, out_dim, bias=bias)
        self.original.weight.requires_grad = False
        if bias:
            self.original.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 64, alpha: float = 128) -> "LoRALinear":
        """Create LoRALinear from existing nn.Linear, copying weights."""
        lora = cls(linear.in_features, linear.out_features, rank, alpha, bias=linear.bias is not None)
        lora.original.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora.original.bias.data.copy_(linear.bias.data)
        return lora


class LoRAMultiheadAttention(nn.Module):
    """Multi-head attention with LoRA on Q, K, V projections."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, lora_rank: int = 64, lora_alpha: float = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # LoRA-wrapped Q, K, V
        self.q_proj = LoRALinear(dim, dim, lora_rank, lora_alpha)
        self.k_proj = LoRALinear(dim, dim, lora_rank, lora_alpha)
        self.v_proj = LoRALinear(dim, dim, lora_rank, lora_alpha)
        self.out_proj = nn.Linear(dim, dim)  # output projection (trainable)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TemporalBlock(nn.Module):
    """Temporal attention + feedforward for video frames."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 lora_rank: int = 64, lora_alpha: float = 128, use_lora: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if use_lora:
            self.temporal_attn = LoRAMultiheadAttention(dim, num_heads, dropout, lora_rank, lora_alpha)
        else:
            self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) temporal sequence of frame features."""
        normed = self.norm1(x)
        if isinstance(self.temporal_attn, LoRAMultiheadAttention):
            x = x + self.temporal_attn(normed)
        else:
            x = x + self.temporal_attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x


class VideoPredictionModel(nn.Module):
    """Video prediction model with CogVideoX-style architecture.

    Architecture:
    - Frame encoder: Conv → feature vector per frame
    - Temporal transformer: LoRA fine-tuned attention over frame sequence
    - Language cross-attention: condition on task description
    - Frame decoder: feature → predicted frame
    - Autoregressive: predict one frame at a time, append to sequence
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

        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=4),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 4, stride=4),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Frame decoder
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
            nn.AdaptiveAvgPool2d(resolution),
            nn.Sigmoid(),
        )

        # Temporal transformer with LoRA
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, num_heads, dropout, lora_rank, lora_alpha, use_lora=True)
            for _ in range(num_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Positional embedding
        self.frame_pos_embed = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Language cross-attention
        self.lang_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to features. (B, T, C, H, W) → (B, T, D)"""
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        feat = self.frame_encoder(flat)
        return feat.reshape(B, T, self.hidden_dim)

    def decode_frames(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to frames. (B, T, D) → (B, T, C, H, W)"""
        B, T, D = features.shape
        flat = features.reshape(B * T, D)
        frames = self.frame_decoder(flat)
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
            input_frames: (B, T_in, C, H, W)
            lang_context: (B, L, context_dim)
            num_pred: number of frames to predict

        Returns:
            pred_frames: (B, num_pred, C, H, W)
        """
        B = input_frames.shape[0]
        feat = self.encode_frames(input_frames)
        predicted_frames = []
        current_feat = feat

        for i in range(num_pred):
            seq_len = current_feat.shape[1]
            pos = self.frame_pos_embed[:, :seq_len, :]
            x = current_feat + pos

            for block in self.temporal_blocks:
                x = block(x)
            x = self.temporal_norm(x)

            if lang_context is not None:
                ctx = self.lang_proj(lang_context)
                residual = x
                x_normed = self.cross_norm(x)
                x = residual + self.cross_attn(x_normed, ctx, ctx)[0]

            last_feat = x[:, -1:, :]
            pred_frame = self.decode_frames(last_feat)
            predicted_frames.append(pred_frame)

            new_feat = self.frame_encoder(pred_frame.squeeze(1)).unsqueeze(1)
            current_feat = torch.cat([current_feat[:, 1:, :], new_feat], dim=1)

        return torch.cat(predicted_frames, dim=1)

    def get_lora_params(self) -> list:
        """Get all LoRA parameters for separate optimizer groups."""
        params = []
        for block in self.temporal_blocks:
            if isinstance(block.temporal_attn, LoRAMultiheadAttention):
                params.extend(block.temporal_attn.q_proj.parameters())
                params.extend(block.temporal_attn.k_proj.parameters())
                params.extend(block.temporal_attn.v_proj.parameters())
        return params

    def get_non_lora_params(self) -> list:
        """Get all non-LoRA trainable parameters."""
        lora_ids = {id(p) for p in self.get_lora_params()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in lora_ids]

    def merge_lora(self):
        """Merge LoRA weights into original weights for faster inference."""
        for block in self.temporal_blocks:
            if isinstance(block.temporal_attn, LoRAMultiheadAttention):
                for proj in [block.temporal_attn.q_proj, block.temporal_attn.k_proj, block.temporal_attn.v_proj]:
                    merged = proj.original.weight.data + (proj.lora_A @ proj.lora_B * proj.scaling).T
                    proj.original.weight.data.copy_(merged)
                    proj.lora_A.data.zero_()


if __name__ == "__main__":
    model = VideoPredictionModel(
        in_channels=1, hidden_dim=256, num_frames=8,
        resolution=(96, 128), num_layers=2, num_heads=4,
        lora_rank=16, lora_alpha=32, context_dim=256,
    )

    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for p in model.get_lora_params())
    print(f"Total params: {total / 1e6:.1f}M, LoRA params: {lora / 1e3:.1f}K ({100 * lora / total:.2f}%)")

    input_frames = torch.randn(1, 4, 1, 96, 128)
    lang_ctx = torch.randn(1, 8, 256)
    pred = model(input_frames, lang_context=lang_ctx, num_pred=4)
    print(f"Predicted frames: {pred.shape}")

    # Test merge
    model.merge_lora()
    pred2 = model(input_frames, lang_context=lang_ctx, num_pred=4)
    print(f"After merge: {pred2.shape}")
