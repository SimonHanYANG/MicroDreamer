"""Virtual pipette for testing without hardware."""

from hardware.base import PipetteBase, Position3D


class VirtualPipette(PipetteBase):
    """Simulated 3-DOF pipette with position tracking."""

    def __init__(self):
        self._pos = Position3D(0.0, 0.0, 100.0)  # Z starts at 100um
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        self._pos = Position3D(0.0, 0.0, 100.0)

    def disconnect(self) -> None:
        self._connected = False

    def get_position(self) -> Position3D:
        if not self._connected:
            raise RuntimeError("Pipette not connected")
        return Position3D(self._pos.x, self._pos.y, self._pos.z)

    def move_absolute(self, x: float, y: float, z: float) -> None:
        if not self._connected:
            raise RuntimeError("Pipette not connected")
        self._pos.x = x
        self._pos.y = y
        self._pos.z = z

    def move_relative(self, dx: float, dy: float, dz: float) -> None:
        if not self._connected:
            raise RuntimeError("Pipette not connected")
        self._pos.x += dx
        self._pos.y += dy
        self._pos.z += dz

    def is_connected(self) -> bool:
        return self._connected
