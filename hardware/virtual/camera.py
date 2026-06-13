"""Virtual camera for testing without hardware."""

import time
import numpy as np
from hardware.base import CameraBase, Frame


class VirtualCamera(CameraBase):
    """Generates synthetic frames for testing."""

    def __init__(self, width: int = 1600, height: int = 1200, fps: float = 30.0):
        self._width = width
        self._height = height
        self._fps = fps
        self._open = False
        self._frame_id = 0

    def open(self) -> None:
        self._open = True
        self._frame_id = 0

    def capture(self) -> Frame:
        if not self._open:
            raise RuntimeError("Camera not open")
        # Synthetic frame: gradient + noise + moving circle
        img = np.zeros((self._height, self._width), dtype=np.uint8)
        # Background gradient
        for y in range(self._height):
            img[y, :] = int(20 + 30 * y / self._height)
        # Moving circle (simulates a cell)
        cx = int(self._width / 2 + 200 * np.sin(self._frame_id * 0.05))
        cy = int(self._height / 2 + 150 * np.cos(self._frame_id * 0.05))
        yy, xx = np.ogrid[:self._height, :self._width]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30**2
        img[mask] = 180
        # Add noise
        noise = np.random.randint(0, 10, (self._height, self._width), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self._frame_id += 1
        return Frame(image=img, timestamp=time.time(), frame_id=self._frame_id)

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        return self._fps
