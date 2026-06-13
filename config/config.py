"""Configuration management for MicroDreamer."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Hierarchical configuration with YAML loading and dot-access."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """Load configuration from YAML file with optional overrides."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "default.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if overrides:
        _deep_update(data, overrides)

    return Config(data)


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update nested dict."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    cfg = load_config()
    print(f"Project: {cfg.project.name}")
    print(f"Camera resolution: {cfg.camera.resolution}")
    print(f"Stage type: {cfg.stage.type}")
    print(f"Action dim: {cfg.preprocessing.action_dim}")
