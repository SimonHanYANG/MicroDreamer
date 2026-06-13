"""Basler camera driver using pypylon."""

import logging
import numpy as np
from hardware.base import CameraBase, Frame

logger = logging.getLogger(__name__)


class BaslerCamera(CameraBase):
    """Basler camera via pypylon (GigE/USB3)."""

    def __init__(self, width: int = 1600, height: int = 1200, fps: float = 30.0):
        self._width = width
        self._height = height
        self._fps = fps
        self._camera = None
        self._open = False
        self._frame_id = 0

    def open(self) -> None:
        try:
            import pypylon.pylon as pylon
            from pypylon import genicam
        except ImportError:
            raise ImportError("pypylon not installed. Use VirtualCamera for testing.")

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler cameras found")

        self._camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        self._camera.Open()
        logger.info(f"Opened camera: {self._camera.GetDeviceInfo().GetModelName()}")

        self._configure()
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._open = True
        self._frame_id = 0

    def _configure(self) -> None:
        from pypylon import genicam
        if genicam.IsAvailable(self._camera.TriggerMode):
            self._camera.TriggerMode.SetValue("Off")
        if genicam.IsAvailable(self._camera.AcquisitionFrameRateEnable):
            self._camera.AcquisitionFrameRateEnable.SetValue(True)
            if genicam.IsAvailable(self._camera.AcquisitionFrameRate):
                self._camera.AcquisitionFrameRate.SetValue(self._fps)
        if genicam.IsAvailable(self._camera.PixelFormat):
            self._camera.PixelFormat.SetValue("Mono8")
        if genicam.IsAvailable(self._camera.Width):
            self._camera.Width.SetValue(self._width)
            self._camera.Height.SetValue(self._height)

    def capture(self) -> Frame:
        import pypylon.pylon as pylon
        if not self._open or not self._camera.IsGrabbing():
            raise RuntimeError("Camera not open")
        grab = self._camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            raise RuntimeError(f"Grab failed: {grab.ErrorDescription}")
        img = grab.Array.copy()
        grab.Release()
        self._frame_id += 1
        import time
        return Frame(image=img, timestamp=time.time(), frame_id=self._frame_id)

    def close(self) -> None:
        if self._camera:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            if self._camera.IsOpen():
                self._camera.Close()
        self._open = False

    def is_open(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        return self._fps
