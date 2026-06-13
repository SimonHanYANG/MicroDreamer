"""Tests for data preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from data.preprocessor.action_converter import positions_to_deltas, deltas_to_positions, ActionNormalizer
from data.preprocessor.frame_processor import resize_frame, tile_frame, normalize_frame


def test_action_conversion():
    stage = np.array([[0, 0], [10, 5], [20, 10]], dtype=np.float32)
    pip = np.array([[0, 0, 100], [1, 0, 98], [2, 1, 95]], dtype=np.float32)

    actions = positions_to_deltas(stage, pip)
    assert actions.shape == (2, 5)
    assert np.allclose(actions[0], [10, 5, 1, 0, -2])

    # Round-trip
    s, p = deltas_to_positions(stage[0], pip[0], actions)
    assert np.allclose(s, stage)
    assert np.allclose(p, pip)
    print("  [PASS] action_conversion")


def test_action_normalizer():
    actions = np.random.randn(100, 5).astype(np.float32)
    norm = ActionNormalizer(actions)

    normalized = norm.normalize(actions)
    assert normalized.shape == actions.shape
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-5)

    denormalized = norm.denormalize(normalized)
    assert np.allclose(denormalized, actions, atol=1e-5)

    # Save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        norm.save(f.name)
        norm2 = ActionNormalizer()
        norm2.load(f.name)
        assert np.allclose(norm.mean, norm2.mean)
    print("  [PASS] action_normalizer")


def test_frame_resize():
    frame = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
    resized = resize_frame(frame, (512, 384))
    assert resized.shape == (384, 512)
    print("  [PASS] frame_resize")


def test_frame_tiling():
    frame = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
    tiles = tile_frame(frame, tile_size=448)
    # 1200/448 ceil=3, 1600/448 ceil=4 -> 12 tiles
    assert tiles.shape[0] == 12
    assert tiles.shape[1:] == (448, 448)
    print("  [PASS] frame_tiling")


def test_normalize_frame():
    frame = np.array([[0, 128, 255]], dtype=np.uint8)
    norm = normalize_frame(frame)
    assert norm.dtype == np.float32
    assert norm[0, 0] == 0.0
    assert norm[0, 2] == 1.0
    print("  [PASS] normalize_frame")


if __name__ == "__main__":
    print("Running preprocessor tests...")
    test_action_conversion()
    test_action_normalizer()
    test_frame_resize()
    test_frame_tiling()
    test_normalize_frame()
    print("All preprocessor tests passed!")
