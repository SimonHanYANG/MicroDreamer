"""Calibration utilities: pixel-to-um conversion, focus depth."""

import numpy as np
from typing import Tuple, Optional


class PixelCalibration:
    """Convert between pixel coordinates and micrometer coordinates."""

    def __init__(self, pixel_size_um: float = 0.6):
        self.pixel_size_um = pixel_size_um

    def pixel_to_um(self, dx_px: float, dy_px: float) -> Tuple[float, float]:
        """Convert pixel displacement to micrometer displacement."""
        return dx_px * self.pixel_size_um, dy_px * self.pixel_size_um

    def um_to_pixel(self, dx_um: float, dy_um: float) -> Tuple[float, float]:
        """Convert micrometer displacement to pixel displacement."""
        return dx_um / self.pixel_size_um, dy_um / self.pixel_size_um

    def pixel_to_um_array(self, arr: np.ndarray) -> np.ndarray:
        """Convert array of pixel displacements to um. Shape: (..., 2)"""
        return arr * self.pixel_size_um

    def um_to_pixel_array(self, arr: np.ndarray) -> np.ndarray:
        """Convert array of um displacements to pixels. Shape: (..., 2)"""
        return arr / self.pixel_size_um


class FocusCalibration:
    """Estimate Z depth from image focus/sharpness."""

    def __init__(self, z_positions: Optional[list] = None):
        self.z_positions = z_positions or []
        self.sharpness_values: list = []

    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        """Compute Laplacian variance as sharpness metric."""
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        image = image.astype(np.float64)
        # 3x3 Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.signal import convolve2d
        laplacian = convolve2d(image, kernel, mode="valid")
        return float(np.var(laplacian))

    def calibrate(self, images: list, z_positions: list) -> None:
        """Store sharpness profiles for known Z positions."""
        self.z_positions = list(z_positions)
        self.sharpness_values = [self.compute_sharpness(img) for img in images]

    def estimate_z(self, image: np.ndarray) -> Optional[float]:
        """Estimate Z position by interpolating sharpness profile."""
        if not self.z_positions or not self.sharpness_values:
            return None
        sharpness = self.compute_sharpness(image)
        idx = int(np.argmax(self.sharpness_values))
        # Simple: return Z of sharpest calibration image
        # Advanced: parabolic interpolation
        return self.z_positions[idx]


if __name__ == "__main__":
    cal = PixelCalibration(pixel_size_um=0.6)
    dx, dy = cal.pixel_to_um(100.0, 50.0)
    print(f"100px, 50px -> {dx}um, {dy}um")

    # Test sharpness on synthetic image
    img = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
    s = FocusCalibration.compute_sharpness(img)
    print(f"Random image sharpness: {s:.2f}")
