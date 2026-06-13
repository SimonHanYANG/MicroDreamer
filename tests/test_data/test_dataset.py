"""Tests for dataset."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.dataset import MicroDreamerDataset, create_dummy_dataset


def test_dummy_dataset_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_episodes=2, frames_per_episode=20)
        episodes = list(Path(tmpdir).iterdir())
        assert len(episodes) == 2
        for ep in episodes:
            assert (ep / "data.npz").exists()
            assert (ep / "metadata.json").exists()
    print("  [PASS] dummy_dataset_creation")


def test_dataset_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_episodes=2, frames_per_episode=30)
        ds = MicroDreamerDataset(
            tmpdir, action_horizon=8, low_res=(128, 96),
            normalize_actions=True, use_simple_lang=True,
        )
        assert len(ds) > 0

        sample = ds[0]
        assert "high_res_tiles" in sample
        assert "low_res_frames" in sample
        assert "actions" in sample
        assert "task_description" in sample

        # Check shapes
        tiles = sample["high_res_tiles"]
        assert tiles.shape[1] == 1  # 1 channel
        assert tiles.shape[2] == 448 and tiles.shape[3] == 448

        frames = sample["low_res_frames"]
        assert frames.shape[2] == 96  # low_res height
        assert frames.shape[3] == 128  # low_res width

        actions = sample["actions"]
        assert actions.shape[1] == 5  # action_dim
    print("  [PASS] dataset_loading")


if __name__ == "__main__":
    print("Running dataset tests...")
    test_dummy_dataset_creation()
    test_dataset_loading()
    print("All dataset tests passed!")
