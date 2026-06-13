"""Inference: predict actions and future video frames."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from models.action.action_model import ActionPredictionModel
from models.video.video_model import VideoPredictionModel
from models.language.encoder import SimpleLanguageEncoder, encode_text_simple
from data.preprocessor.frame_processor import prepare_high_res, prepare_low_res
from data.preprocessor.action_converter import ActionNormalizer


class MicroDreamerPredictor:
    """Unified predictor for both action and video prediction."""

    def __init__(
        self,
        action_ckpt: Optional[str] = None,
        video_ckpt: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.cfg = load_config(config_path)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Action model
        self.action_model = ActionPredictionModel(
            hidden_dim=self.cfg.action_model.hidden_dim,
            visual_layers=self.cfg.action_model.num_layers,
            visual_heads=self.cfg.action_model.num_heads,
            action_dim=self.cfg.action_model.action_dim,
            action_horizon=self.cfg.action_model.action_horizon,
            action_layers=self.cfg.action_model.num_layers,
            action_heads=self.cfg.action_model.num_heads,
            use_simple_lang=True,
        ).to(self.device)

        if action_ckpt and Path(action_ckpt).exists():
            self.action_model.load_state_dict(torch.load(action_ckpt, map_location=self.device))
            self.action_model.eval()

        # Video model
        self.video_model = VideoPredictionModel(
            hidden_dim=256,
            num_frames=self.cfg.video_model.num_frames,
            resolution=tuple(self.cfg.video_model.resolution),
            num_layers=4,
            num_heads=8,
            context_dim=self.cfg.language.cross_attn_dim,
        ).to(self.device)

        if video_ckpt and Path(video_ckpt).exists():
            self.video_model.load_state_dict(torch.load(video_ckpt, map_location=self.device))
            self.video_model.eval()

        # Language encoder
        self.lang_encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=self.cfg.action_model.hidden_dim).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        frame: np.ndarray,
        task_description: str = "",
    ) -> dict:
        """Run prediction on a single frame.

        Args:
            frame: (H, W) grayscale frame
            task_description: language instruction

        Returns:
            dict with 'actions' (T, 5) and 'predicted_frames' (T, C, H, W)
        """
        # Prepare inputs
        tiles = prepare_high_res(frame)  # (num_tiles, 448, 448)
        tiles_tensor = torch.tensor(tiles, dtype=torch.float32).unsqueeze(1)  # (T, 1, 448, 448)
        tiles_batch = tiles_tensor.unsqueeze(0).to(self.device)  # (1, T, 1, 448, 448)

        low_res = prepare_low_res(frame, tuple(self.cfg.preprocessing.low_res))
        low_res_tensor = torch.tensor(low_res, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1, H, W)

        # Language
        lang_ids = encode_text_simple([task_description]).to(self.device)
        lang_ctx = self.lang_encoder(lang_ids)

        # Action prediction
        actions = self.action_model.predict_actions(tiles_batch, lang_input_ids=lang_ids)
        actions_np = actions.cpu().numpy()[0]  # (horizon, 5)

        # Video prediction
        pred_frames = self.video_model(low_res_tensor, lang_context=lang_ctx, num_pred=4)
        pred_frames_np = pred_frames.cpu().numpy()[0]  # (4, 1, H, W)

        return {
            "actions": actions_np,
            "predicted_frames": pred_frames_np,
        }


if __name__ == "__main__":
    # Test with random input
    predictor = MicroDreamerPredictor(device="cpu")
    frame = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
    result = predictor.predict(frame, task_description="move to cell")
    print(f"Actions: {result['actions'].shape}")
    print(f"Frames: {result['predicted_frames'].shape}")
