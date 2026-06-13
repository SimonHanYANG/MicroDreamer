"""Data recorder: saves synchronized samples to disk."""

import json
import time
import logging
from pathlib import Path
from typing import List

import numpy as np

from data.collector.synchronizer import SyncedSample

logger = logging.getLogger(__name__)


class DataRecorder:
    """Records synchronized samples to disk as .npz + metadata."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_dir: Path = None

    def start_episode(self, task_description: str = "", subgoals: list = None) -> str:
        """Create a new episode directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        episode_id = f"episode_{timestamp}"
        self._episode_dir = self.output_dir / episode_id
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "episode_id": episode_id,
            "task_description": task_description,
            "subgoals": subgoals or [],
            "timestamp": timestamp,
        }
        with open(self._episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Started episode: {episode_id}")
        return episode_id

    def save_samples(self, samples: List[SyncedSample]) -> None:
        """Save a list of synchronized samples."""
        if not self._episode_dir:
            raise RuntimeError("No active episode. Call start_episode first.")

        frames = []
        stage_positions = []
        pipette_positions = []
        timestamps = []

        for s in samples:
            frames.append(s.frame.image)
            stage_positions.append([s.stage_pos.x, s.stage_pos.y])
            pipette_positions.append([s.pipette_pos.x, s.pipette_pos.y, s.pipette_pos.z])
            timestamps.append(s.timestamp)

        np.savez_compressed(
            self._episode_dir / "data.npz",
            frames=np.array(frames),
            stage_positions=np.array(stage_positions),
            pipette_positions=np.array(pipette_positions),
            timestamps=np.array(timestamps),
        )

        logger.info(f"Saved {len(samples)} samples to {self._episode_dir}")
