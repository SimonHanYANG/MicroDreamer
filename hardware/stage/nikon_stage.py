"""Nikon Ti2E stage controller via StageCPP.dll (ctypes)."""

import ctypes
import os
import logging
from hardware.base import StageBase, Position2D

logger = logging.getLogger(__name__)


class NikonStage(StageBase):
    """Nikon Ti2E microscope XY stage via DLL."""

    def __init__(self, dll_dir: str = r"D:\Aojun\Ti2E_API\StageCPP\x64\Release"):
        self._dll_dir = dll_dir
        self._dll = None
        self._stage = None
        self._connected = False

    def connect(self) -> None:
        os.environ["PATH"] = self._dll_dir + os.pathsep + os.environ["PATH"]
        dll_path = os.path.join(self._dll_dir, "StageCPP.dll")
        self._dll = ctypes.WinDLL(dll_path)

        self._dll.CreateStage.restype = ctypes.c_void_p
        self._dll.DisposeStage.argtypes = [ctypes.c_void_p]
        self._dll.ConnectStage.argtypes = [ctypes.c_void_p]
        self._dll.MoveXYtoAbsolute.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self._dll.MoveXYRelative.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self._dll.MoveZtoAbsolute.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self._dll.GetZPositionCurrent.restype = ctypes.c_double
        self._dll.GetZPositionCurrent.argtypes = [ctypes.c_void_p]
        self._dll.GetPositionCurrent.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]

        self._stage = self._dll.CreateStage()
        self._dll.ConnectStage(self._stage)
        self._connected = True
        logger.info("Nikon Ti2E stage connected")

    def disconnect(self) -> None:
        if self._stage and self._dll:
            self._dll.DisposeStage(self._stage)
        self._connected = False

    def get_position(self) -> Position2D:
        x = ctypes.c_double(0.0)
        y = ctypes.c_double(0.0)
        self._dll.GetPositionCurrent(self._stage, ctypes.byref(x), ctypes.byref(y))
        return Position2D(x.value, y.value)

    def move_absolute(self, x: float, y: float) -> None:
        self._dll.MoveXYtoAbsolute(self._stage, ctypes.c_double(x), ctypes.c_double(y))

    def move_relative(self, dx: float, dy: float) -> None:
        self._dll.MoveXYRelative(self._stage, ctypes.c_double(dx), ctypes.c_double(dy))

    def is_connected(self) -> bool:
        return self._connected
