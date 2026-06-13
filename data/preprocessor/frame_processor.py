"""Frame preprocessing: resize, normalize, tile for InternViT."""

import numpy as np
from typing import Tuple


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame using bilinear interpolation (no OpenCV dependency)."""
    from scipy.ndimage import zoom
    h, w = frame.shape[:2]
    th, tw = target_size[1], target_size[0]
    factors = (th / h, tw / w)
    if frame.ndim == 3:
        factors = factors + (1,)
    return zoom(frame, factors, order=1).astype(frame.dtype)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame to [0, 1] float32."""
    return frame.astype(np.float32) / 255.0


def tile_frame(frame: np.ndarray, tile_size: int = 448) -> np.ndarray:
    """Tile a high-res frame into tiles for InternViT.

    Args:
        frame: (H, W) or (H, W, C) input frame
        tile_size: tile size (default 448 for InternViT)

    Returns:
        tiles: (num_tiles, tile_size, tile_size[, C])
    """
    h, w = frame.shape[:2]
    # Pad to multiple of tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if frame.ndim == 3:
        padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    else:
        padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="constant")

    # Reshape into tiles
    nh = (h + pad_h) // tile_size
    nw = (w + pad_w) // tile_size
    if frame.ndim == 3:
        tiles = padded.reshape(nh, tile_size, nw, tile_size, frame.shape[2])
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, frame.shape[2])
    else:
        tiles = padded.reshape(nh, tile_size, nw, tile_size)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)

    return tiles


def prepare_high_res(frame: np.ndarray) -> np.ndarray:
    """Prepare frame for action prediction: tile into 448x448 patches."""
    return tile_frame(frame, tile_size=448)


def prepare_low_res(frame: np.ndarray, target_size: Tuple[int, int] = (512, 384)) -> np.ndarray:
    """Prepare frame for video prediction: resize to low resolution."""
    return resize_frame(frame, target_size)


if __name__ == "__main__":
    # Test with synthetic 1600x1200 frame
    frame = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)

    # High-res tiling
    tiles = tile_frame(frame, tile_size=448)
    print(f"Input: {frame.shape}, Tiles: {tiles.shape}")
    # 1200/448=3 tiles high, 1600/448=4 tiles wide -> 12 tiles

    # Low-res resize
    low = resize_frame(frame, (512, 384))
    print(f"Low-res: {low.shape}")
