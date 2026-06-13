"""Convert absolute positions to relative action deltas."""

import numpy as np
from typing import Tuple


def positions_to_deltas(
    stage_positions: np.ndarray,
    pipette_positions: np.ndarray,
) -> np.ndarray:
    """Convert absolute position sequences to relative deltas.

    Args:
        stage_positions: (T, 2) absolute XY stage positions in um
        pipette_positions: (T, 3) absolute XYZ pipette positions in um

    Returns:
        actions: (T-1, 5) deltas [stage_dx, stage_dy, pip_dx, pip_dy, pip_dz]
    """
    stage_deltas = np.diff(stage_positions, axis=0)  # (T-1, 2)
    pip_deltas = np.diff(pipette_positions, axis=0)  # (T-1, 3)
    actions = np.concatenate([stage_deltas, pip_deltas], axis=1)  # (T-1, 5)
    return actions


def deltas_to_positions(
    initial_stage: np.ndarray,
    initial_pipette: np.ndarray,
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert action deltas back to absolute positions.

    Args:
        initial_stage: (2,) initial stage XY
        initial_pipette: (3,) initial pipette XYZ
        actions: (T, 5) action deltas

    Returns:
        stage_positions: (T+1, 2)
        pipette_positions: (T+1, 3)
    """
    stage_positions = [initial_stage.copy()]
    pipette_positions = [initial_pipette.copy()]

    for a in actions:
        new_stage = stage_positions[-1] + a[:2]
        new_pip = pipette_positions[-1] + a[2:5]
        stage_positions.append(new_stage)
        pipette_positions.append(new_pip)

    return np.array(stage_positions), np.array(pipette_positions)


class ActionNormalizer:
    """Normalize/denormalize actions for training."""

    def __init__(self, actions: np.ndarray = None):
        self.mean = None
        self.std = None
        if actions is not None:
            self.fit(actions)

    def fit(self, actions: np.ndarray) -> None:
        """Compute mean and std from action array (N, 5)."""
        self.mean = actions.mean(axis=0)
        self.std = actions.std(axis=0) + 1e-8

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.mean) / self.std

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


if __name__ == "__main__":
    # Test conversion
    stage_pos = np.array([[0, 0], [10, 5], [20, 10], [30, 15]], dtype=np.float32)
    pip_pos = np.array([[0, 0, 100], [1, 0, 98], [2, 1, 95], [3, 1, 90]], dtype=np.float32)

    actions = positions_to_deltas(stage_pos, pip_pos)
    print(f"Actions shape: {actions.shape}")
    print(f"Actions:\n{actions}")

    # Round-trip
    s, p = deltas_to_positions(stage_pos[0], pip_pos[0], actions)
    print(f"Reconstructed stage:\n{s}")
    print(f"Reconstructed pipette:\n{p}")

    # Normalizer
    norm = ActionNormalizer(actions)
    normalized = norm.normalize(actions)
    denormalized = norm.denormalize(normalized)
    print(f"Round-trip error: {np.abs(actions - denormalized).max()}")
