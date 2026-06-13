"""Complete action prediction model combining visual encoder + language encoder + diffusion head."""

import torch
import torch.nn as nn
from typing import Optional

from models.action.visual_encoder import TileAttentionEncoder
from models.language.encoder import LanguageEncoder, SimpleLanguageEncoder
from models.action.action_head import DiffusionActionHead, ActionDiffusion


class ActionPredictionModel(nn.Module):
    """Full action prediction model.

    Pipeline:
    1. High-res image tiles -> TileAttentionEncoder -> visual features
    2. Language instruction -> LanguageEncoder -> language features
    3. Concatenate visual + language as context
    4. DiffusionActionHead predicts action sequence conditioned on context
    """

    def __init__(
        self,
        # Visual encoder params
        tile_size: int = 448,
        patch_size: int = 14,
        num_tiles: int = 12,
        visual_layers: int = 12,
        visual_heads: int = 16,
        # Language encoder params
        lang_model: str = "google/flan-t5-xl",
        lang_max_length: int = 128,
        # Action head params
        action_dim: int = 5,
        action_horizon: int = 16,
        hidden_dim: int = 1024,
        action_layers: int = 8,
        action_heads: int = 16,
        # General
        dropout: float = 0.1,
        use_simple_lang: bool = False,  # True for testing without T5
    ):
        super().__init__()

        # Visual encoder
        self.visual_encoder = TileAttentionEncoder(
            tile_size=tile_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=hidden_dim,
            num_layers=visual_layers,
            num_heads=visual_heads,
            num_tiles=num_tiles,
            dropout=dropout,
        )

        # Language encoder
        if use_simple_lang:
            self.language_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=hidden_dim)
        else:
            self.language_encoder = LanguageEncoder(
                model_name=lang_model, hidden_dim=hidden_dim, max_length=lang_max_length
            )

        # Context fusion: project concatenated features
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

        # Action diffusion head
        self.action_head = DiffusionActionHead(
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_layers=action_layers,
            num_heads=action_heads,
            context_dim=hidden_dim,
            dropout=dropout,
        )

        # Diffusion process
        self.diffusion = ActionDiffusion(num_timesteps=100)

    def encode_context(
        self,
        tiles: torch.Tensor,
        lang_input_ids: torch.Tensor,
        lang_attention_mask: Optional[torch.Tensor] = None,
        lang_text: Optional[list] = None,
    ) -> torch.Tensor:
        """Encode visual + language context.

        Returns:
            context: (B, L_v + L_l, hidden_dim)
        """
        # Visual features: (B, hidden_dim) -> (B, 1, hidden_dim)
        vis_feat = self.visual_encoder(tiles)
        vis_feat = self.visual_proj(vis_feat).unsqueeze(1)

        # Language features: (B, L, hidden_dim)
        if lang_text is not None:
            lang_feat = self.language_encoder(text=lang_text)
        else:
            lang_feat = self.language_encoder(input_ids=lang_input_ids, attention_mask=lang_attention_mask)

        # Concatenate: (B, 1 + L, hidden_dim)
        context = torch.cat([vis_feat, lang_feat], dim=1)
        return self.context_norm(context)

    def training_loss(
        self,
        tiles: torch.Tensor,
        gt_actions: torch.Tensor,
        lang_input_ids: Optional[torch.Tensor] = None,
        lang_attention_mask: Optional[torch.Tensor] = None,
        lang_text: Optional[list] = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            tiles: (B, T, C, H, W) high-res image tiles
            gt_actions: (B, action_horizon, action_dim) ground truth actions
            lang_input_ids: (B, L) language token ids

        Returns:
            loss: scalar
        """
        context = self.encode_context(tiles, lang_input_ids, lang_attention_mask, lang_text)

        # Sample random timesteps
        B = tiles.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=tiles.device)

        # Add noise to ground truth actions
        noise = torch.randn_like(gt_actions)
        noisy_actions = self.diffusion.q_sample(gt_actions, t, noise)

        # Predict noise
        pred_noise = self.action_head(noisy_actions, t, context)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict_actions(
        self,
        tiles: torch.Tensor,
        lang_input_ids: Optional[torch.Tensor] = None,
        lang_attention_mask: Optional[torch.Tensor] = None,
        lang_text: Optional[list] = None,
    ) -> torch.Tensor:
        """Predict action sequence via reverse diffusion.

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        context = self.encode_context(tiles, lang_input_ids, lang_attention_mask, lang_text)
        B = tiles.shape[0]
        shape = (B, self.action_head.action_horizon, self.action_head.action_dim)
        return self.diffusion.sample(self.action_head, context, shape)


import torch.nn.functional as F


if __name__ == "__main__":
    # Test with simple language encoder (no T5 download needed)
    model = ActionPredictionModel(
        hidden_dim=256, visual_layers=2, visual_heads=4,
        action_layers=2, action_heads=4, use_simple_lang=True,
        tile_size=448, num_tiles=12,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Dummy inputs
    tiles = torch.randn(1, 12, 1, 448, 448)
    gt_actions = torch.randn(1, 16, 5)
    lang_ids = torch.randint(0, 100, (1, 16))

    # Training loss
    loss = model.training_loss(tiles, gt_actions, lang_input_ids=lang_ids)
    print(f"Training loss: {loss.item():.4f}")

    # Inference
    pred = model.predict_actions(tiles, lang_input_ids=lang_ids)
    print(f"Predicted actions: {pred.shape}")
