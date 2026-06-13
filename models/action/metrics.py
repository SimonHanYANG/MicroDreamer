"""Metrics for action prediction evaluation."""

import numpy as np
import torch
from typing import Dict


def action_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error for action sequences. (N, T, 5)"""
    return float(np.mean((pred - target) ** 2))


def action_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error for action sequences. (N, T, 5)"""
    return float(np.mean(np.abs(pred - target)))


def action_mse_per_dim(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Per-dimension MSE. pred/target: (N, T, 5)"""
    dim_names = ["stage_dx", "stage_dy", "pip_dx", "pip_dy", "pip_dz"]
    mse_per_dim = np.mean((pred - target) ** 2, axis=(0, 1))
    return {name: float(mse) for name, mse in zip(dim_names, mse_per_dim)}


def endpoint_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Cumulative displacement error at final timestep.

    pred/target: (N, T, 5) where last 3 dims are pipette xyz deltas.
    Computes ||sum(pred_deltas) - sum(target_deltas)|| for pipette.
    """
    pred_cumsum = np.cumsum(pred[:, :, 2:5], axis=1)  # (N, T, 3)
    target_cumsum = np.cumsum(target[:, :, 2:5], axis=1)
    error = np.linalg.norm(pred_cumsum[:, -1, :] - target_cumsum[:, -1, :], axis=1)
    return float(np.mean(error))


def trajectory_length(actions: np.ndarray) -> float:
    """Average trajectory length in action space. (N, T, 5)"""
    diffs = np.diff(actions, axis=1)  # (N, T-1, 5)
    lengths = np.sum(np.linalg.norm(diffs, axis=2), axis=1)  # (N,)
    return float(np.mean(lengths))


def action_consistency(actions: np.ndarray) -> float:
    """Measure smoothness: average magnitude of action differences between timesteps.

    Lower = smoother trajectory. (N, T, 5)
    """
    diffs = np.diff(actions, axis=1)
    return float(np.mean(np.linalg.norm(diffs, axis=2)))


class ActionMetrics:
    """Collects and aggregates action prediction metrics."""

    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Add a batch of predictions and targets."""
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics over accumulated data."""
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)

        results = {
            "action_mse": action_mse(preds, targets),
            "action_mae": action_mae(preds, targets),
            "endpoint_error": endpoint_error(preds, targets),
            "pred_trajectory_length": trajectory_length(preds),
            "target_trajectory_length": trajectory_length(targets),
            "pred_consistency": action_consistency(preds),
            "target_consistency": action_consistency(targets),
        }
        results.update({f"mse_{k}": v for k, v in action_mse_per_dim(preds, targets).items()})
        return results

    def reset(self):
        self.preds.clear()
        self.targets.clear()


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    pred = np.random.randn(10, 16, 5) * 0.5
    target = pred + np.random.randn(10, 16, 5) * 0.1

    metrics = ActionMetrics()
    metrics.update(pred, target)
    results = metrics.compute()

    print("Action Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
