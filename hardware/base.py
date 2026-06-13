"""Abstract base classes for hardware devices."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Position2D:
    x: float = 0.0
    y: float = 0.0

    def to_tuple(self):
        return (self.x, self.y)


@dataclass
class Position3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self):
        return (self.x, self.y, self.z)


@dataclass
class Frame:
    """A captured camera frame with metadata."""
    image: any  # np.ndarray, H x W or H x W x C
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0


class CameraBase(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def capture(self) -> Frame: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_open(self) -> bool: ...

    @property
    @abstractmethod
    def resolution(self) -> tuple: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...


class StageBase(ABC):
    """Abstract XY stage interface."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_position(self) -> Position2D: ...

    @abstractmethod
    def move_absolute(self, x: float, y: float) -> None: ...

    @abstractmethod
    def move_relative(self, dx: float, dy: float) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


class PipetteBase(ABC):
    """Abstract pipette (3-DOF: X, Y, Z) interface."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_position(self) -> Position3D: ...

    @abstractmethod
    def move_absolute(self, x: float, y: float, z: float) -> None: ...

    @abstractmethod
    def move_relative(self, dx: float, dy: float, dz: float) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...
