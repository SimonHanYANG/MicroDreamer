"""Multi-device data synchronizer with timestamp alignment."""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hardware.base import CameraBase, StageBase, PipetteBase, Frame, Position2D, Position3D

logger = logging.getLogger(__name__)


@dataclass
class SyncedSample:
    """A synchronized sample from all devices."""
    frame: Frame
    stage_pos: Position2D
    pipette_pos: Position3D
    timestamp: float


class DataSynchronizer:
    """Collects and synchronizes data from camera, stage, and pipette."""

    def __init__(
        self,
        camera: CameraBase,
        stage: StageBase,
        pipette: PipetteBase,
        camera_fps: float = 30.0,
        stage_hz: float = 100.0,
        pipette_hz: float = 100.0,
        sync_tolerance_ms: float = 50.0,
        buffer_size: int = 1000,
    ):
        self.camera = camera
        self.stage = stage
        self.pipette = pipette
        self.camera_fps = camera_fps
        self.sync_tolerance_ms = sync_tolerance_ms

        self._frame_buffer: deque = deque(maxlen=buffer_size)
        self._stage_buffer: deque = deque(maxlen=buffer_size)
        self._pipette_buffer: deque = deque(maxlen=buffer_size)
        self._samples: list = []
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start all collection threads."""
        self.camera.open()
        self.stage.connect()
        self.pipette.connect()
        self._running = True

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._stage_thread = threading.Thread(target=self._stage_loop, daemon=True)
        self._pip_thread = threading.Thread(target=self._pipette_loop, daemon=True)

        self._cam_thread.start()
        self._stage_thread.start()
        self._pip_thread.start()
        logger.info("Data synchronizer started")

    def stop(self) -> None:
        """Stop all collection threads."""
        self._running = False
        self._cam_thread.join(timeout=3)
        self._stage_thread.join(timeout=3)
        self._pip_thread.join(timeout=3)
        self.camera.close()
        self.stage.disconnect()
        self.pipette.disconnect()
        logger.info("Data synchronizer stopped")

    def _camera_loop(self):
        interval = 1.0 / self.camera_fps
        while self._running:
            try:
                frame = self.camera.capture()
                with self._lock:
                    self._frame_buffer.append(frame)
            except Exception as e:
                logger.error(f"Camera error: {e}")
            time.sleep(interval)

    def _stage_loop(self):
        while self._running:
            try:
                pos = self.stage.get_position()
                with self._lock:
                    self._stage_buffer.append((time.time(), pos))
            except Exception as e:
                logger.error(f"Stage error: {e}")
            time.sleep(0.01)

    def _pipette_loop(self):
        while self._running:
            try:
                pos = self.pipette.get_position()
                with self._lock:
                    self._pipette_buffer.append((time.time(), pos))
            except Exception as e:
                logger.error(f"Pipette error: {e}")
            time.sleep(0.01)

    def get_latest_sample(self) -> Optional[SyncedSample]:
        """Get the latest synchronized sample."""
        with self._lock:
            if not self._frame_buffer:
                return None
            frame = self._frame_buffer[-1]

            # Find closest stage position
            stage_pos = self._find_closest(self._stage_buffer, frame.timestamp)
            pip_pos = self._find_closest(self._pipette_buffer, frame.timestamp)

        if stage_pos is None:
            stage_pos = Position2D(0.0, 0.0)
        if pip_pos is None:
            pip_pos = Position3D(0.0, 0.0, 0.0)

        return SyncedSample(
            frame=frame, stage_pos=stage_pos, pipette_pos=pip_pos, timestamp=frame.timestamp
        )

    def _find_closest(self, buffer: deque, target_time: float):
        """Find the buffer entry closest to target_time within tolerance."""
        if not buffer:
            return None
        best = None
        best_dt = float("inf")
        for ts, pos in buffer:
            dt = abs(ts - target_time)
            if dt < best_dt:
                best_dt = dt
                best = pos
        if best_dt * 1000 > self.sync_tolerance_ms:
            return None
        return best

    def collect_samples(self, num_samples: int, interval_s: float = 0.1) -> list:
        """Collect a fixed number of synchronized samples."""
        samples = []
        for _ in range(num_samples):
            sample = self.get_latest_sample()
            if sample:
                samples.append(sample)
            time.sleep(interval_s)
        return samples
