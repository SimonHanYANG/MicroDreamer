"""Pipette controller via HTTP REST API (MMS Motor API)."""

import logging
import requests
from hardware.base import PipetteBase, Position3D

logger = logging.getLogger(__name__)


class HttpPipette(PipetteBase):
    """3-DOF pipette via HTTP API (localhost:5000)."""

    def __init__(self, api_url: str = "http://localhost:5000", arm_id: int = 1):
        self._api_url = api_url.rstrip("/")
        self._arm_id = arm_id
        self._session = requests.Session()
        self._connected = False

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self._api_url}/api/MMSMotor/{endpoint}"
        if params is None:
            params = {}
        params["mmsType"] = self._arm_id
        resp = self._session.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("successed", False):
            raise RuntimeError(f"API error: {data}")
        return data.get("resultData", {})

    def _post(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self._api_url}/api/MMSMotor/{endpoint}"
        if params is None:
            params = {}
        params["mmsType"] = self._arm_id
        resp = self._session.post(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("successed", False):
            raise RuntimeError(f"API error: {data}")
        return data.get("resultData", {})

    def connect(self) -> None:
        result = self._get("ConnectedState")
        x_ok = result.get("xMotorIsConnected", False)
        y_ok = result.get("yMotorIsConnected", False)
        z_ok = result.get("zMotorIsConnected", False)
        if not (x_ok and y_ok and z_ok):
            raise RuntimeError(f"Arm not fully connected: x={x_ok} y={y_ok} z={z_ok}")
        self._connected = True
        logger.info(f"Pipette arm {self._arm_id} connected")

    def disconnect(self) -> None:
        self._connected = False

    def get_position(self) -> Position3D:
        result = self._get("GetMotorPosition")
        return Position3D(
            x=float(result.get("xSteps", 0)),
            y=float(result.get("ySteps", 0)),
            z=float(result.get("zSteps", 0)),
        )

    def move_absolute(self, x: float, y: float, z: float) -> None:
        # Move each axis sequentially
        for motor_type, target in [(1, x), (2, y), (3, z)]:
            self._post("MotorMoveByDisplacementMode", {
                "motorType": motor_type, "speed": 10, "step": target
            })

    def move_relative(self, dx: float, dy: float, dz: float) -> None:
        for motor_type, step in [(1, dx), (2, dy), (3, dz)]:
            if step != 0:
                self._post("MotorMoveByDisplacementMode", {
                    "motorType": motor_type, "speed": 10, "step": step
                })

    def is_connected(self) -> bool:
        return self._connected
