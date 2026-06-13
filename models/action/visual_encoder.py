"""Visual encoder for action prediction using InternViT-style tiling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """Convert image tiles to patch embeddings."""

    def __init__(self, tile_size: int = 448, patch_size: int = 14, in_channels: int = 1, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        # Support non-square: tile_size can be (H, W) or int
        if isinstance(tile_size, int):
            self.h_patches = tile_size // patch_size
            self.w_patches = tile_size // patch_size
        else:
            self.h_patches = tile_size[0] // patch_size
            self.w_patches = tile_size[1] // patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, num_patches+1, embed_dim)"""
        B = x.shape[0]
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TileAttentionEncoder(nn.Module):
    """InternViT-style visual encoder with tile-based processing.

    Handles high-resolution (1600x1200) images by:
    1. Splitting into 448x448 tiles (12 tiles for 4x3 grid)
    2. Processing each tile through ViT
    3. Aggregating tile features
    """

    def __init__(
        self,
        tile_size: int = 448,
        patch_size: int = 14,
        in_channels: int = 1,
        embed_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        num_tiles: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(tile_size, patch_size, in_channels, embed_dim)
        self.tile_embed = nn.Parameter(torch.randn(1, num_tiles, 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Tile aggregation: learnable query tokens
        self.tile_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.tile_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Process tiled image.

        Args:
            tiles: (B, num_tiles, C, tile_size, tile_size)

        Returns:
            features: (B, embed_dim)
        """
        B, T, C, H, W = tiles.shape

        # Process each tile through patch embedding
        tiles_flat = tiles.reshape(B * T, C, H, W)
        tokens = self.patch_embed(tiles_flat)  # (B*T, P+1, D)

        # Add tile position embedding
        tile_pos = self.tile_embed[:, :T, :, :].expand(B, -1, -1, -1)
        tile_pos = tile_pos.reshape(B * T, 1, self.embed_dim)
        tokens[:, :1, :] = tokens[:, :1, :] + tile_pos

        # Transformer
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Aggregate: use CLS token from each tile
        tile_features = tokens[:, 0, :]  # (B*T, D)
        tile_features = tile_features.reshape(B, T, self.embed_dim)  # (B, T, D)

        # Cross-attention aggregation
        query = self.tile_query.expand(B, -1, -1)  # (B, 1, D)
        aggregated, _ = self.tile_attn(query, tile_features, tile_features)  # (B, 1, D)

        return aggregated.squeeze(1)  # (B, D)


class LowResEncoder(nn.Module):
    """Lightweight encoder for low-res (512x384) frames (video prediction path)."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 768, num_layers: int = 6, num_heads: int = 12):
        super().__init__()
        self.patch_embed = PatchEmbedding((384, 512), 16, in_channels, embed_dim)  # 24x32=768 patches
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 384, 512) -> (B, embed_dim)"""
        tokens = self.patch_embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 0, :]  # CLS token


if __name__ == "__main__":
    # Test tile encoder
    encoder = TileAttentionEncoder(tile_size=448, patch_size=14, embed_dim=256, num_layers=2, num_heads=4, num_tiles=12)
    tiles = torch.randn(1, 12, 1, 448, 448)  # batch=1, 12 tiles, 1 channel, 448x448
    out = encoder(tiles)
    print(f"Tile encoder output: {out.shape}")  # (1, 256)

    # Test low-res encoder
    low_encoder = LowResEncoder(embed_dim=256, num_layers=2, num_heads=4)
    low_input = torch.randn(1, 1, 384, 512)
    out_low = low_encoder(low_input)
    print(f"Low-res encoder output: {out_low.shape}")  # (1, 256)
