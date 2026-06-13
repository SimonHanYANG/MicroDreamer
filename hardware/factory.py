"""Factory for creating hardware instances from config."""

from config.config import Config
from hardware.base import CameraBase, StageBase, PipetteBase


def create_camera(cfg: Config) -> CameraBase:
    """Create camera instance based on config."""
    cam_type = "virtual"  # default to virtual for testing
    w, h = cfg.camera.resolution[0], cfg.camera.resolution[1]
    fps = cfg.camera.fps

    if cam_type == "virtual":
        from hardware.virtual.camera import VirtualCamera
        return VirtualCamera(width=w, height=h, fps=fps)
    else:
        from hardware.camera.basler_camera import BaslerCamera
        return BaslerCamera(width=w, height=h, fps=fps)


def create_stage(cfg: Config) -> StageBase:
    """Create stage instance based on config."""
    stage_type = cfg.stage.type

    if stage_type == "virtual":
        from hardware.virtual.stage import VirtualStage
        return VirtualStage()
    elif stage_type == "nikon_ti2":
        from hardware.stage.nikon_stage import NikonStage
        return NikonStage(dll_dir=cfg.stage.dll_path)
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def create_pipette(cfg: Config) -> PipetteBase:
    """Create pipette instance based on config."""
    pip_type = cfg.pipette.type

    if pip_type == "virtual":
        from hardware.virtual.pipette import VirtualPipette
        return VirtualPipette()
    elif pip_type == "http_api":
        from hardware.pipette.http_pipette import HttpPipette
        return HttpPipette(api_url=cfg.pipette.api_url, arm_id=cfg.pipette.arm_id)
    else:
        raise ValueError(f"Unknown pipette type: {pip_type}")
