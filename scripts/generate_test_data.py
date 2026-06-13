"""Generate mock training data for end-to-end pipeline testing.

Creates synthetic episodes with realistic structure:
- Moving circle patterns in frames
- Smooth stage/pipette trajectories
- Proper metadata.json with task descriptions and subgoals
- Compatible with MicroDreamerDataset and all training scripts

Usage:
    python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10
    python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --resolution 200,160 --frames 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


TASK_TEMPLATES = [
    {
        "task": "Aspirate the target cell using the pipette",
        "subgoals": [
            {"description": "Locate target cell in field of view", "action_type": "observe"},
            {"description": "Move stage to center cell", "action_type": "move_stage"},
            {"description": "Lower pipette to cell surface", "action_type": "lower_pipette"},
            {"description": "Apply gentle suction to aspirate cell", "action_type": "aspire"},
        ],
    },
    {
        "task": "Inject sperm into oocyte (ICSI procedure)",
        "subgoals": [
            {"description": "Locate and immobilize sperm cell", "action_type": "observe"},
            {"description": "Aspirate sperm into pipette", "action_type": "aspire"},
            {"description": "Move to oocyte position", "action_type": "move_stage"},
            {"description": "Penetrate zona pellucida and inject", "action_type": "inject"},
        ],
    },
    {
        "task": "Sort healthy cells from debris",
        "subgoals": [
            {"description": "Scan field of view for cells", "action_type": "observe"},
            {"description": "Identify healthy cell morphology", "action_type": "classify"},
            {"description": "Move pipette to target cell", "action_type": "move_stage"},
            {"description": "Aspirate and relocate cell", "action_type": "aspire"},
        ],
    },
    {
        "task": "Transfer embryo to culture dish",
        "subgoals": [
            {"description": "Locate embryo in source dish", "action_type": "observe"},
            {"description": "Carefully aspirate embryo", "action_type": "aspire"},
            {"description": "Move to target culture dish", "action_type": "move_stage"},
            {"description": "Gently release embryo", "action_type": "release"},
        ],
    },
    {
        "task": "Perform zona drilling on embryo",
        "subgoals": [
            {"description": "Focus on embryo zona pellucida", "action_type": "observe"},
            {"description": "Position pipette at drilling angle", "action_type": "move_stage"},
            {"description": "Apply laser or acid to create opening", "action_type": "drill"},
            {"description": "Verify opening integrity", "action_type": "observe"},
        ],
    },
]


def generate_episode(
    output_dir: Path,
    episode_idx: int,
    num_frames: int = 50,
    height: int = 160,
    width: int = 200,
    fps: float = 30.0,
) -> str:
    """Generate one synthetic episode with realistic patterns and explicit targets."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    episode_id = f"episode_{timestamp}_{episode_idx:04d}"
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(episode_idx * 1000)

    # --- Target definition ---
    # Target cell position in stage coordinates (µm)
    target_stage_x = np.random.uniform(-30.0, 30.0)
    target_stage_y = np.random.uniform(-30.0, 30.0)
    # Target pipette end position (µm)
    target_pip_x = np.random.uniform(-5.0, 5.0)
    target_pip_y = np.random.uniform(-5.0, 5.0)
    target_pip_z = np.random.uniform(50.0, 60.0)  # near the cell surface

    # --- Generate frames with moving circle + noise ---
    frames = np.random.randint(10, 40, (num_frames, height, width), dtype=np.uint8)

    # Target cell: fixed bright circle at a known pixel location
    # Place target near center with some offset
    target_pixel_x = width // 2 + np.random.randint(-20, 20)
    target_pixel_y = height // 2 + np.random.randint(-15, 15)
    target_radius = np.random.randint(8, 14)
    target_brightness = np.random.randint(180, 240)

    # Background objects (moving)
    num_objects = np.random.randint(2, 4)
    object_params = []
    for _ in range(num_objects):
        cx_start = np.random.randint(width // 4, 3 * width // 4)
        cy_start = np.random.randint(height // 4, 3 * height // 4)
        radius = np.random.randint(4, 10)
        speed_x = np.random.uniform(-1.0, 1.0)
        speed_y = np.random.uniform(-0.8, 0.8)
        brightness = np.random.randint(80, 150)
        freq_x = np.random.uniform(0.05, 0.15)
        freq_y = np.random.uniform(0.03, 0.12)
        object_params.append((cx_start, cy_start, radius, speed_x, speed_y, brightness, freq_x, freq_y))

    yy, xx = np.ogrid[:height, :width]
    target_dist_sq = (xx - target_pixel_x) ** 2 + (yy - target_pixel_y) ** 2
    target_mask = target_dist_sq < target_radius ** 2
    target_intensity = target_brightness * np.exp(-target_dist_sq / (2 * (target_radius * 0.7) ** 2))

    for t in range(num_frames):
        # Draw target cell (static)
        frames[t] = np.clip(
            frames[t].astype(np.float32) + target_intensity * target_mask, 0, 255
        ).astype(np.uint8)
        # Draw background objects (moving)
        for cx_start, cy_start, radius, speed_x, speed_y, brightness, freq_x, freq_y in object_params:
            cx = int(cx_start + speed_x * t + 8 * np.sin(freq_x * t))
            cy = int(cy_start + speed_y * t + 6 * np.cos(freq_y * t))
            cx = max(radius, min(width - radius, cx))
            cy = max(radius, min(height - radius, cy))
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = dist_sq < radius ** 2
            intensity = brightness * np.exp(-dist_sq / (2 * (radius * 0.7) ** 2))
            frames[t] = np.clip(frames[t].astype(np.float32) + intensity * mask, 0, 255).astype(np.uint8)

    # --- Stage trajectory: start far, approach target ---
    start_stage_x = target_stage_x + np.random.uniform(40.0, 80.0) * np.random.choice([-1, 1])
    start_stage_y = target_stage_y + np.random.uniform(40.0, 80.0) * np.random.choice([-1, 1])
    # Smooth approach with noise
    t_norm = np.linspace(0, 1, num_frames)
    approach_x = start_stage_x + (target_stage_x - start_stage_x) * (1 - np.exp(-3 * t_norm))
    approach_y = start_stage_y + (target_stage_y - start_stage_y) * (1 - np.exp(-3 * t_norm))
    noise_x = np.cumsum(np.random.randn(num_frames) * 0.15)
    noise_y = np.cumsum(np.random.randn(num_frames) * 0.15)
    stage_pos = np.column_stack([approach_x + noise_x, approach_y + noise_y]).astype(np.float32)

    # --- Pipette trajectory: start at rest, approach target, dip Z ---
    start_pip_x = target_pip_x + np.random.uniform(15.0, 30.0) * np.random.choice([-1, 1])
    start_pip_y = target_pip_y + np.random.uniform(15.0, 30.0) * np.random.choice([-1, 1])
    pip_approach_x = start_pip_x + (target_pip_x - start_pip_x) * (1 - np.exp(-2.5 * t_norm))
    pip_approach_y = start_pip_y + (target_pip_y - start_pip_y) * (1 - np.exp(-2.5 * t_norm))
    pip_noise_x = np.cumsum(np.random.randn(num_frames) * 0.1)
    pip_noise_y = np.cumsum(np.random.randn(num_frames) * 0.1)
    # Z: descend to target, hold, then retract
    z_descend = np.linspace(100, target_pip_z, num_frames // 3)
    z_hold = np.linspace(target_pip_z, target_pip_z - 2, num_frames // 3)
    z_retract = np.linspace(target_pip_z - 2, 100, num_frames - 2 * (num_frames // 3))
    z_trajectory = np.concatenate([z_descend, z_hold, z_retract])
    pipette_pos = np.column_stack([
        pip_approach_x + pip_noise_x,
        pip_approach_y + pip_noise_y,
        z_trajectory,
    ]).astype(np.float32)

    # Timestamps
    timestamps = np.linspace(0, num_frames / fps, num_frames)

    # Save data.npz
    np.savez_compressed(
        ep_dir / "data.npz",
        frames=frames,
        stage_positions=stage_pos,
        pipette_positions=pipette_pos,
        timestamps=timestamps,
    )

    # Select task template
    template = TASK_TEMPLATES[episode_idx % len(TASK_TEMPLATES)]

    # Build subgoals with frame ranges and positions
    n_sg = len(template["subgoals"])
    seg_len = num_frames // n_sg
    subgoals = []
    for i, sg in enumerate(template["subgoals"]):
        s = i * seg_len
        e = (i + 1) * seg_len if i < n_sg - 1 else num_frames
        subgoals.append({
            "description": sg["description"],
            "action_type": sg["action_type"],
            "start_frame": s,
            "end_frame": e,
        })

    # Save metadata.json with targets
    metadata = {
        "episode_id": episode_id,
        "task_description": template["task"],
        "subgoals": subgoals,
        "num_frames": num_frames,
        "timestamp": timestamp,
        "resolution": [width, height],
        # ── Target info ──
        "target": {
            "stage_position": [float(target_stage_x), float(target_stage_y)],
            "pipette_position": [float(target_pip_x), float(target_pip_y), float(target_pip_z)],
            "pixel_position": [int(target_pixel_x), int(target_pixel_y)],
            "pixel_radius": int(target_radius),
        },
        # ── Start positions ──
        "start_position": {
            "stage": [float(start_stage_x), float(start_stage_y)],
            "pipette": [float(start_pip_x), float(start_pip_y), 100.0],
        },
    }
    with open(ep_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return episode_id


def main():
    parser = argparse.ArgumentParser(description="Generate mock training data for testing")
    parser.add_argument("--output_dir", type=str, default="./data/test_raw",
                        help="Output directory for episodes")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to generate")
    parser.add_argument("--frames", type=int, default=50,
                        help="Frames per episode")
    parser.add_argument("--resolution", type=str, default="200,160",
                        help="Frame resolution as WIDTH,HEIGHT (e.g., 200,160)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    width, height = [int(x) for x in args.resolution.split(",")]
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_episodes} episodes...")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames per episode: {args.frames}")
    print(f"  Output: {output_dir}")

    episode_ids = []
    for i in range(args.num_episodes):
        ep_id = generate_episode(
            output_dir, i,
            num_frames=args.frames,
            height=height, width=width,
        )
        episode_ids.append(ep_id)
        print(f"  [{i+1}/{args.num_episodes}] {ep_id}")

    # Summary
    print(f"\nDone! Generated {len(episode_ids)} episodes in {output_dir}")
    print(f"\nVerify with:")
    print(f"  python -c \"import numpy as np; d=np.load('{output_dir}/{episode_ids[0]}/data.npz'); print('frames:', d['frames'].shape, 'stage:', d['stage_positions'].shape, 'pipette:', d['pipette_positions'].shape)\"")
    print(f"\nUse for training:")
    print(f"  python scripts/train_action.py --data_dir {output_dir} --simple_lang --config config/test.yaml")
    print(f"  python scripts/train_video.py --data_dir {output_dir} --config config/test.yaml")


if __name__ == "__main__":
    main()
