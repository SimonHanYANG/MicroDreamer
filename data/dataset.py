"""PyTorch Dataset for MicroDreamer training data."""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessor.action_converter import positions_to_deltas, ActionNormalizer
from data.preprocessor.frame_processor import prepare_high_res, prepare_low_res

logger = logging.getLogger(__name__)


class MicroDreamerDataset(Dataset):
    """Dataset for paired video frames + action sequences + language.

    Each episode contains:
    - frames: (T, H, W) grayscale frames
    - stage_positions: (T, 2) absolute XY
    - pipette_positions: (T, 3) absolute XYZ
    - metadata: task description, subgoals
    """

    def __init__(
        self,
        data_dir: str,
        action_horizon: int = 16,
        frame_skip: int = 1,
        high_res: tuple = (1600, 1200),
        low_res: tuple = (512, 384),
        normalize_actions: bool = True,
        use_simple_lang: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon
        self.frame_skip = frame_skip
        self.high_res = high_res
        self.low_res = low_res
        self.normalize_actions = normalize_actions
        self.use_simple_lang = use_simple_lang

        # Discover episodes
        self.episodes = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "data.npz").exists()
        ])

        if not self.episodes:
            logger.warning(f"No episodes found in {data_dir}")
            self.episodes = []

        # Build flat index: (episode_idx, start_frame)
        self.index = []
        for ep_idx, ep_dir in enumerate(self.episodes):
            data = np.load(ep_dir / "data.npz")
            num_frames = len(data["frames"])
            for start in range(0, num_frames - action_horizon * frame_skip, action_horizon):
                self.index.append((ep_idx, start))

        # Load all metadata
        self.metadata = []
        for ep_dir in self.episodes:
            meta_path = ep_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata.append(json.load(f))
            else:
                self.metadata.append({"task_description": "", "subgoals": []})

        # Action normalizer
        self.normalizer = None
        if normalize_actions:
            self._fit_normalizer()

        logger.info(f"Loaded {len(self.index)} samples from {len(self.episodes)} episodes")

    def _fit_normalizer(self):
        """Compute action statistics across all episodes."""
        all_actions = []
        for ep_dir in self.episodes:
            data = np.load(ep_dir / "data.npz")
            stage = data["stage_positions"]
            pip = data["pipette_positions"]
            actions = positions_to_deltas(stage, pip)
            all_actions.append(actions)
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            self.normalizer = ActionNormalizer(all_actions)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.index[idx]
        ep_dir = self.episodes[ep_idx]
        meta = self.metadata[ep_idx]
        data = np.load(ep_dir / "data.npz")

        frames = data["frames"]
        stage_pos = data["stage_positions"]
        pip_pos = data["pipette_positions"]

        # Extract frame sequence
        end = start + self.action_horizon * self.frame_skip
        frame_indices = list(range(start, min(end, len(frames)), self.frame_skip))

        # High-res tiles for action prediction (use first frame)
        high_res_tiles = prepare_high_res(frames[frame_indices[0]])  # (num_tiles, H, W)
        high_res_tiles = torch.tensor(high_res_tiles, dtype=torch.float32).unsqueeze(1)  # (T, 1, H, W)

        # Low-res frames for video prediction
        low_res_frames = []
        for fi in frame_indices:
            lr = prepare_low_res(frames[fi], self.low_res)
            low_res_frames.append(lr)
        low_res_frames = np.array(low_res_frames)  # (T, H', W')
        low_res_tensor = torch.tensor(low_res_frames, dtype=torch.float32).unsqueeze(1)  # (T, 1, H', W')

        # Actions
        all_actions = positions_to_deltas(stage_pos, pip_pos)
        action_start = start
        action_end = action_start + self.action_horizon
        if action_end > len(all_actions):
            action_end = len(all_actions)
            action_start = action_end - self.action_horizon
        actions = all_actions[action_start:action_end]  # (horizon, 5)
        actions = torch.tensor(actions, dtype=torch.float32)

        if self.normalizer is not None:
            actions = torch.tensor(self.normalizer.normalize(actions.numpy()), dtype=torch.float32)

        # Language
        task_desc = meta.get("task_description", "")

        return {
            "high_res_tiles": high_res_tiles,  # (num_tiles, 1, 448, 448)
            "low_res_frames": low_res_tensor,  # (T, 1, H', W')
            "actions": actions,  # (horizon, 5)
            "task_description": task_desc,
            "episode_idx": ep_idx,
            "start_frame": start,
        }


def create_dummy_dataset(output_dir: str, num_episodes: int = 5, frames_per_episode: int = 50):
    """Create a dummy dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(num_episodes):
        ep_dir = output_dir / f"episode_{ep:04d}"
        ep_dir.mkdir(exist_ok=True)

        T = frames_per_episode
        frames = np.random.randint(0, 255, (T, 1200, 1600), dtype=np.uint8)
        # Add moving circle
        for t in range(T):
            cx = int(800 + 200 * np.sin(t * 0.1))
            cy = int(600 + 150 * np.cos(t * 0.1))
            yy, xx = np.ogrid[:1200, :1600]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30**2
            frames[t][mask] = 180

        stage_pos = np.cumsum(np.random.randn(T, 2) * 0.5, axis=0)
        pip_pos = np.column_stack([
            np.cumsum(np.random.randn(T) * 0.3),
            np.cumsum(np.random.randn(T) * 0.3),
            100 - np.arange(T) * 0.5,  # Z decreasing
        ])

        np.savez_compressed(
            ep_dir / "data.npz",
            frames=frames,
            stage_positions=stage_pos.astype(np.float32),
            pipette_positions=pip_pos.astype(np.float32),
            timestamps=np.linspace(0, T / 30, T),
        )

        meta = {
            "episode_id": f"episode_{ep:04d}",
            "task_description": f"Approach and aspirate cell {ep}",
            "subgoals": [
                {"description": "Locate target cell", "action_type": "observe"},
                {"description": "Move pipette to cell", "action_type": "move_stage"},
            ],
            "num_frames": T,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Created {num_episodes} episodes in {output_dir}")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_episodes=3, frames_per_episode=30)
        ds = MicroDreamerDataset(tmpdir, action_horizon=8, low_res=(128, 96))
        print(f"Dataset size: {len(ds)}")
        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
