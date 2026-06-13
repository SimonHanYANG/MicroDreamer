"""Data collection CLI for MicroDreamer.

Collects synchronized camera/stage/pipette data with language annotations.
Supports virtual devices for testing and real hardware for production.

Usage:
    python scripts/collect_data.py --mode virtual --num_episodes 10 --output_dir ./data/raw
    python scripts/collect_data.py --mode real --config config/default.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from hardware.factory import create_camera, create_stage, create_pipette
from data.collector.synchronizer import DataSynchronizer
from data.collector.recorder import DataRecorder
from data.annotation.format import TASK_TEMPLATES, EpisodeAnnotation, Subgoal
from utils.logger import setup_logger

logger = setup_logger("collect_data")


def collect_virtual_episode(
    synchronizer: DataSynchronizer,
    recorder: DataRecorder,
    task_type: str,
    num_frames: int = 50,
    move_pattern: str = "linear",
) -> str:
    """Collect one episode with virtual devices using scripted movements."""
    task_info = TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["icsi"])
    episode_id = recorder.start_episode(
        task_description=task_info["description"],
        subgoals=[sg["description"] for sg in task_info["subgoals"]],
    )

    # Scripted movement for virtual devices
    stage = synchronizer.stage
    pipette = synchronizer.pipette

    samples = []
    for i in range(num_frames):
        # Move stage in a pattern
        if move_pattern == "linear":
            stage.move_relative(2.0, 1.0)
        elif move_pattern == "circular":
            angle = i * 0.1
            stage.move_relative(3.0 * np.cos(angle), 3.0 * np.sin(angle))
        elif move_pattern == "zigzag":
            if i % 20 < 10:
                stage.move_relative(3.0, 0.0)
            else:
                stage.move_relative(-3.0, 2.0)

        # Move pipette Z down gradually
        if i < num_frames // 2:
            pipette.move_relative(0.0, 0.0, -0.5)
        else:
            pipette.move_relative(0.0, 0.0, 0.5)

        # Collect synchronized sample
        sample = synchronizer.get_latest_sample()
        if sample:
            samples.append(sample)

        time.sleep(0.033)  # ~30fps

    recorder.save_samples(samples)
    logger.info(f"Collected episode {episode_id}: {len(samples)} samples")
    return episode_id


def collect_interactive_episode(
    synchronizer: DataSynchronizer,
    recorder: DataRecorder,
    task_description: str,
    subgoals: list = None,
) -> str:
    """Collect one episode with manual control (for real hardware)."""
    episode_id = recorder.start_episode(task_description, subgoals)
    logger.info(f"Started episode {episode_id}. Press Ctrl+C to stop.")

    samples = []
    try:
        while True:
            sample = synchronizer.get_latest_sample()
            if sample:
                samples.append(sample)
            time.sleep(0.033)
    except KeyboardInterrupt:
        pass

    recorder.save_samples(samples)
    logger.info(f"Episode {episode_id}: {len(samples)} samples collected")
    return episode_id


def run_collection(args):
    """Main collection loop."""
    cfg = load_config(args.config)
    mode = args.mode

    # Create devices
    if mode == "virtual":
        cfg._data["stage"]["type"] = "virtual"
        cfg._data["pipette"]["type"] = "virtual"

    camera = create_camera(cfg)
    stage = create_stage(cfg)
    pipette = create_pipette(cfg)

    # Create synchronizer and recorder
    sync = DataSynchronizer(
        camera=camera, stage=stage, pipette=pipette,
        camera_fps=cfg.collection.camera_fps,
        sync_tolerance_ms=cfg.collection.sync_tolerance_ms,
    )
    recorder = DataRecorder(args.output_dir)

    sync.start()
    logger.info(f"Data collection started (mode={mode})")

    try:
        if mode == "virtual":
            # Automated collection
            task_types = list(TASK_TEMPLATES.keys())
            patterns = ["linear", "circular", "zigzag"]

            for ep in range(args.num_episodes):
                task_type = task_types[ep % len(task_types)]
                pattern = patterns[ep % len(patterns)]
                logger.info(f"Episode {ep + 1}/{args.num_episodes}: {task_type} / {pattern}")

                collect_virtual_episode(
                    sync, recorder, task_type,
                    num_frames=args.frames_per_episode,
                    move_pattern=pattern,
                )
                time.sleep(0.5)

        elif mode == "real":
            # Interactive collection
            for ep in range(args.num_episodes):
                logger.info(f"Episode {ep + 1}/{args.num_episodes}")
                input("Press Enter to start recording (Ctrl+C to stop)...")
                collect_interactive_episode(
                    sync, recorder,
                    task_description=args.task_description,
                )

    finally:
        sync.stop()
        logger.info("Data collection finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroDreamer Data Collection")
    parser.add_argument("--mode", choices=["virtual", "real"], default="virtual")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--frames_per_episode", type=int, default=50)
    parser.add_argument("--task_description", type=str, default="micro manipulation")
    args = parser.parse_args()
    run_collection(args)
