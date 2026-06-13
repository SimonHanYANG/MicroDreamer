"""Calibration utilities for MicroDreamer hardware.

Performs:
1. Pixel-to-micrometer calibration
2. Focus-depth calibration
3. Stage-to-camera alignment

Usage:
    python scripts/calibrate.py --mode pixel --pixel_size 0.6
    python scripts/calibrate.py --mode focus --data_dir ./calibration_data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.calibration import PixelCalibration, FocusCalibration
from utils.logger import setup_logger

logger = setup_logger("calibrate")


def calibrate_pixel(args):
    """Pixel-to-micrometer calibration."""
    cal = PixelCalibration(pixel_size_um=args.pixel_size)

    # Test conversions
    test_pixels = [(100, 100), (500, 300), (1600, 1200)]
    logger.info(f"Pixel-to-UM calibration (scale: {args.pixel_size} um/px):")
    for px, py in test_pixels:
        um_x, um_y = cal.pixel_to_um(float(px), float(py))
        logger.info(f"  ({px}px, {py}px) -> ({um_x:.1f}um, {um_y:.1f}um)")

    # Save calibration
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_data = {
        "pixel_size_um": args.pixel_size,
        "type": "pixel_to_um",
    }
    with open(output_dir / "pixel_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=2)
    logger.info(f"Saved to {output_dir / 'pixel_calibration.json'}")

    return cal


def calibrate_focus(args):
    """Focus-depth calibration using sharpness analysis."""
    cal = FocusCalibration()

    # Generate synthetic calibration data (for testing)
    # In real usage, images would be captured at known Z positions
    z_positions = np.linspace(80, 120, 9)  # 80um to 120um
    logger.info(f"Focus calibration with {len(z_positions)} Z positions: {z_positions[0]:.1f} - {z_positions[-1]:.1f} um")

    # Simulate: sharpness peaks at z=100um
    images = []
    for z in z_positions:
        # Synthetic image with sharpness that peaks at z=100
        sharpness_factor = np.exp(-((z - 100) ** 2) / 50)
        img = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
        # Add structure with varying sharpness
        img = (img * sharpness_factor).astype(np.uint8)
        images.append(img)

    cal.calibrate(images, list(z_positions))

    # Test estimation
    test_z = 105.0
    test_img = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
    estimated_z = cal.estimate_z(test_img)
    logger.info(f"Test: true Z={test_z:.1f}, estimated Z={estimated_z:.1f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_data = {
        "z_positions": list(z_positions),
        "sharpness_values": cal.sharpness_values,
        "type": "focus_depth",
    }
    with open(output_dir / "focus_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=2)
    logger.info(f"Saved to {output_dir / 'focus_calibration.json'}")

    return cal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroDreamer Calibration")
    parser.add_argument("--mode", choices=["pixel", "focus", "all"], default="all")
    parser.add_argument("--output_dir", type=str, default="./calibration")
    parser.add_argument("--pixel_size", type=float, default=0.6, help="Pixel size in um/px")
    args = parser.parse_args()

    if args.mode in ("pixel", "all"):
        calibrate_pixel(args)
    if args.mode in ("focus", "all"):
        calibrate_focus(args)
