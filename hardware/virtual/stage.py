"""Virtual XY stage for testing without hardware."""

from hardware.base import StageBase, Position2D


class VirtualStage(StageBase):
    """Simulated XY stage with position tracking."""

    def __init__(self):
        self._pos = Position2D(0.0, 0.0)
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        self._pos = Position2D(0.0, 0.0)

    def disconnect(self) -> None:
        self._connected = False

    def get_position(self) -> Position2D:
        if not self._connected:
            raise RuntimeError("Stage not connected")
        return Position2D(self._pos.x, self._pos.y)

    def move_absolute(self, x: float, y: float) -> None:
        if not self._connected:
            raise RuntimeError("Stage not connected")
        self._pos.x = x
        self._pos.y = y

    def move_relative(self, dx: float, dy: float) -> None:
        if not self._connected:
            raise RuntimeError("Stage not connected")
        self._pos.x += dx
        self._pos.y += dy

    def is_connected(self) -> bool:
        return self._connected
