"""Tests for action prediction model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from models.action.visual_encoder import TileAttentionEncoder, LowResEncoder
from models.action.action_head import DiffusionActionHead, ActionDiffusion
from models.action.action_model import ActionPredictionModel
from models.language.encoder import SimpleLanguageEncoder, encode_text_simple


def test_tile_encoder():
    enc = TileAttentionEncoder(tile_size=448, patch_size=14, embed_dim=256, num_layers=2, num_heads=4, num_tiles=12)
    tiles = torch.randn(1, 12, 1, 448, 448)
    out = enc(tiles)
    assert out.shape == (1, 256)
    print("  [PASS] tile_encoder")


def test_low_res_encoder():
    enc = LowResEncoder(embed_dim=256, num_layers=2, num_heads=4)
    x = torch.randn(1, 1, 384, 512)
    out = enc(x)
    assert out.shape == (1, 256)
    print("  [PASS] low_res_encoder")


def test_action_head():
    head = DiffusionActionHead(action_dim=5, action_horizon=16, hidden_dim=256, num_layers=2, num_heads=4, context_dim=256)
    noisy = torch.randn(2, 16, 5)
    t = torch.randint(0, 100, (2,))
    ctx = torch.randn(2, 32, 256)
    out = head(noisy, t, ctx)
    assert out.shape == (2, 16, 5)
    print("  [PASS] action_head")


def test_diffusion():
    diff = ActionDiffusion(num_timesteps=100)
    x0 = torch.randn(2, 16, 5)
    t = torch.tensor([50, 30])
    x_noisy = diff.q_sample(x0, t)
    assert x_noisy.shape == x0.shape

    # One denoising step
    model_out = torch.randn_like(x_noisy)
    x_denoised = diff.p_sample(model_out, x_noisy, 50)
    assert x_denoised.shape == x0.shape
    print("  [PASS] diffusion")


def test_full_action_model():
    model = ActionPredictionModel(
        hidden_dim=256, visual_layers=2, visual_heads=4,
        action_layers=2, action_heads=4, use_simple_lang=True,
        tile_size=448, num_tiles=12,
    )

    tiles = torch.randn(1, 12, 1, 448, 448)
    gt_actions = torch.randn(1, 16, 5)
    lang_ids = torch.randint(0, 100, (1, 16))

    # Training loss
    loss = model.training_loss(tiles, gt_actions, lang_input_ids=lang_ids)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0

    # Prediction
    pred = model.predict_actions(tiles, lang_input_ids=lang_ids)
    assert pred.shape == (1, 16, 5)
    print("  [PASS] full_action_model")


def test_language_encoder():
    enc = SimpleLanguageEncoder(vocab_size=100, hidden_dim=256)
    ids = torch.randint(0, 100, (2, 16))
    out = enc(ids)
    assert out.shape == (2, 16, 256)

    # Simple text encoding
    encoded = encode_text_simple(["hello world", "test"], max_length=20)
    assert encoded.shape == (2, 20)
    print("  [PASS] language_encoder")


if __name__ == "__main__":
    print("Running action model tests...")
    test_tile_encoder()
    test_low_res_encoder()
    test_action_head()
    test_diffusion()
    test_language_encoder()
    test_full_action_model()
    print("All action model tests passed!")
