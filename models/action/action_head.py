"""Diffusion-based action prediction head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CrossAttentionBlock(nn.Module):
    """Cross-attention between action tokens and language/visual features."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.context_proj = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        ctx = self.context_proj(context)
        x = x + self.cross_attn(self.norm2(x), ctx, ctx)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class DiffusionActionHead(nn.Module):
    """Predict action sequences using diffusion.

    Architecture:
    - Input: noisy actions (B, T, action_dim) + timestep
    - Conditioning: visual features + language embeddings (cross-attention)
    - Output: predicted noise (B, T, action_dim)
    """

    def __init__(
        self,
        action_dim: int = 5,
        action_horizon: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        context_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Action embedding
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, context_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for diffusion denoising.

        Args:
            noisy_actions: (B, T, action_dim) noisy action sequence
            timestep: (B,) diffusion timestep
            context: (B, L, context_dim) conditioning features (visual + language)

        Returns:
            predicted_noise: (B, T, action_dim)
        """
        B, T, _ = noisy_actions.shape

        # Embed actions + timestep
        x = self.action_proj(noisy_actions) + self.pos_embed[:, :T, :]
        t_emb = self.time_embed(timestep)  # (B, hidden_dim)
        x = x + t_emb.unsqueeze(1)

        # Cross-attention layers
        for layer in self.layers:
            x = layer(x, context)

        x = self.norm(x)
        return self.output_proj(x)


class ActionDiffusion:
    """DDPM diffusion process for action sequences."""

    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x0.device)[t, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion: one denoising step."""
        betas_t = self.betas.to(x_t.device)[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        sqrt_recip = self.sqrt_recip_alphas.to(x_t.device)[t]
        model_mean = sqrt_recip * (x_t - betas_t / sqrt_one_minus * model_output)
        if t == 0:
            return model_mean
        posterior_var = self.posterior_variance.to(x_t.device)[t]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var) * noise

    def sample(self, model: DiffusionActionHead, context: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Full reverse diffusion sampling."""
        device = context.device
        B = shape[0]
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch, context)
            x = self.p_sample(pred_noise, x, t)

        return x


if __name__ == "__main__":
    # Test action head
    head = DiffusionActionHead(action_dim=5, action_horizon=16, hidden_dim=256, num_layers=2, num_heads=4, context_dim=256)
    noisy = torch.randn(2, 16, 5)
    t = torch.randint(0, 100, (2,))
    ctx = torch.randn(2, 32, 256)
    out = head(noisy, t, ctx)
    print(f"Action head output: {out.shape}")  # (2, 16, 5)

    # Test diffusion
    diff = ActionDiffusion(num_timesteps=100)
    x0 = torch.randn(2, 16, 5)
    x_noisy = diff.q_sample(x0, torch.tensor([50, 50]))
    print(f"Noisy actions: {x_noisy.shape}")
