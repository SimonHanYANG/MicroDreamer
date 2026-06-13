"""Visualize mock training data.

Generates mock data and displays:
1. Sample frames with moving objects
2. Stage XY trajectory
3. Pipette XYZ trajectory
4. Frame montage (every N-th frame)

Usage:
    python scripts/visualize_mock_data.py
    python scripts/visualize_mock_data.py --num_episodes 3 --output_dir ./data/viz_mock
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_episode(ep_dir: Path, save_path: Path = None):
    """Visualize one episode: frames + trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    data = np.load(ep_dir / "data.npz")
    frames = data["frames"]
    stage_pos = data["stage_positions"]
    pip_pos = data["pipette_positions"]
    timestamps = data["timestamps"]

    with open(ep_dir / "metadata.json") as f:
        meta = json.load(f)

    T, H, W = frames.shape
    episode_id = meta.get("episode_id", ep_dir.name)
    task = meta.get("task_description", "N/A")

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Episode: {episode_id}\nTask: {task}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3)

    # --- Row 1: Sample frames (4 frames evenly spaced) ---
    frame_indices = np.linspace(0, T - 1, 4, dtype=int)
    for i, fi in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[fi], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Frame {fi}/{T}", fontsize=10)
        ax.axis("off")

    # --- Row 2: Stage trajectory + Pipette XY ---
    ax_stage = fig.add_subplot(gs[1, 0:2])
    scatter = ax_stage.scatter(
        stage_pos[:, 0], stage_pos[:, 1],
        c=np.arange(T), cmap="viridis", s=8, alpha=0.8
    )
    ax_stage.plot(stage_pos[0, 0], stage_pos[0, 1], "g^", markersize=10, label="Start")
    ax_stage.plot(stage_pos[-1, 0], stage_pos[-1, 1], "rv", markersize=10, label="End")
    ax_stage.set_xlabel("X (µm)")
    ax_stage.set_ylabel("Y (µm)")
    ax_stage.set_title("Stage XY Trajectory", fontsize=11)
    ax_stage.legend(fontsize=9)
    ax_stage.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax_stage, label="Time step")

    ax_pip_xy = fig.add_subplot(gs[1, 2:4])
    scatter2 = ax_pip_xy.scatter(
        pip_pos[:, 0], pip_pos[:, 1],
        c=np.arange(T), cmap="plasma", s=8, alpha=0.8
    )
    ax_pip_xy.plot(pip_pos[0, 0], pip_pos[0, 1], "g^", markersize=10, label="Start")
    ax_pip_xy.plot(pip_pos[-1, 0], pip_pos[-1, 1], "rv", markersize=10, label="End")
    ax_pip_xy.set_xlabel("X (µm)")
    ax_pip_xy.set_ylabel("Y (µm)")
    ax_pip_xy.set_title("Pipette XY Trajectory", fontsize=11)
    ax_pip_xy.legend(fontsize=9)
    ax_pip_xy.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax_pip_xy, label="Time step")

    # --- Row 3: Pipette Z + All positions over time ---
    ax_z = fig.add_subplot(gs[2, 0:2])
    ax_z.plot(timestamps, pip_pos[:, 2], "b-", linewidth=1.5, label="Pipette Z")
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Z (µm)")
    ax_z.set_title("Pipette Z over Time", fontsize=11)
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(fontsize=9)

    ax_all = fig.add_subplot(gs[2, 2:4])
    t_norm = np.arange(T)
    ax_all.plot(t_norm, stage_pos[:, 0], "r-", alpha=0.7, label="Stage X")
    ax_all.plot(t_norm, stage_pos[:, 1], "g-", alpha=0.7, label="Stage Y")
    ax_all.plot(t_norm, pip_pos[:, 0], "b--", alpha=0.7, label="Pip X")
    ax_all.plot(t_norm, pip_pos[:, 1], "m--", alpha=0.7, label="Pip Y")
    ax_all.plot(t_norm, pip_pos[:, 2], "c-", alpha=0.7, label="Pip Z")
    ax_all.set_xlabel("Time step")
    ax_all.set_ylabel("Position (µm)")
    ax_all.set_title("All Positions over Time", fontsize=11)
    ax_all.legend(fontsize=8, ncol=3)
    ax_all.grid(True, alpha=0.3)

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_frame_montage(ep_dir: Path, save_path: Path, num_frames: int = 16):
    """Create a montage of evenly spaced frames."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    data = np.load(ep_dir / "data.npz")
    frames = data["frames"]
    T = len(frames)

    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    cols = 4
    rows = (num_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(f"Frame Montage — {ep_dir.name}", fontsize=13, fontweight="bold")

    for i, (ax, fi) in enumerate(zip(axes.flat, indices)):
        ax.imshow(frames[fi], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t={fi}", fontsize=9)
        ax.axis("off")

    # Hide extra axes
    for i in range(len(indices), len(axes.flat)):
        axes.flat[i].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize mock training data")
    parser.add_argument("--output_dir", type=str, default="./data/viz_mock",
                        help="Directory to generate mock data into")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to generate")
    parser.add_argument("--frames", type=int, default=50,
                        help="Frames per episode")
    parser.add_argument("--resolution", type=str, default="200,160",
                        help="Frame resolution as WIDTH,HEIGHT")
    parser.add_argument("--save_dir", type=str, default="./outputs/viz",
                        help="Directory to save visualization plots")
    args = parser.parse_args()

    from scripts.generate_test_data import generate_episode
    import time

    width, height = [int(x) for x in args.resolution.split(",")]
    np.random.seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate episodes
    print(f"Generating {args.num_episodes} episodes ({width}x{height}, {args.frames} frames)...")
    episode_ids = []
    for i in range(args.num_episodes):
        ep_id = generate_episode(output_dir, i, num_frames=args.frames, height=height, width=width)
        episode_ids.append(ep_id)

    # Visualize each episode
    print(f"\nCreating visualizations...")
    for i, ep_id in enumerate(episode_ids):
        ep_dir = output_dir / ep_id
        print(f"[{i+1}/{len(episode_ids)}] {ep_id}")

        # Full visualization (frames + trajectories)
        visualize_episode(ep_dir, save_path=save_dir / f"episode_{i:02d}_overview.png")

        # Frame montage
        create_frame_montage(ep_dir, save_path=save_dir / f"episode_{i:02d}_montage.png", num_frames=16)

    print(f"\nDone! Visualizations saved to: {save_dir}")
    print(f"  - episode_XX_overview.png  : Frames + trajectory plots")
    print(f"  - episode_XX_montage.png   : Frame montage (16 frames)")
    print(f"\nMock data at: {output_dir}")
    print(f"\nClean up when done:")
    print(f"  rmdir /s /q {output_dir}")
    print(f"  rmdir /s /q {save_dir}")


if __name__ == "__main__":
    main()
