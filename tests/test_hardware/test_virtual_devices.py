"""Tests for virtual hardware devices."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hardware.virtual.camera import VirtualCamera
from hardware.virtual.stage import VirtualStage
from hardware.virtual.pipette import VirtualPipette


def test_virtual_camera():
    cam = VirtualCamera(width=1600, height=1200, fps=30)
    assert not cam.is_open()

    cam.open()
    assert cam.is_open()
    assert cam.resolution == (1600, 1200)
    assert cam.fps == 30.0

    frame = cam.capture()
    assert frame.image.shape == (1200, 1600)
    assert frame.frame_id == 1

    frame2 = cam.capture()
    assert frame2.frame_id == 2

    cam.close()
    assert not cam.is_open()
    print("  [PASS] VirtualCamera")


def test_virtual_stage():
    stage = VirtualStage()
    assert not stage.is_connected()

    stage.connect()
    assert stage.is_connected()

    pos = stage.get_position()
    assert pos.x == 0.0 and pos.y == 0.0

    stage.move_relative(10.0, 20.0)
    pos = stage.get_position()
    assert pos.x == 10.0 and pos.y == 20.0

    stage.move_absolute(100.0, 200.0)
    pos = stage.get_position()
    assert pos.x == 100.0 and pos.y == 200.0

    stage.disconnect()
    assert not stage.is_connected()
    print("  [PASS] VirtualStage")


def test_virtual_pipette():
    pip = VirtualPipette()
    assert not pip.is_connected()

    pip.connect()
    assert pip.is_connected()

    pos = pip.get_position()
    assert pos.z == 100.0  # default Z

    pip.move_relative(5.0, -3.0, -10.0)
    pos = pip.get_position()
    assert pos.x == 5.0 and pos.y == -3.0 and pos.z == 90.0

    pip.move_absolute(0.0, 0.0, 50.0)
    pos = pip.get_position()
    assert pos.x == 0.0 and pos.y == 0.0 and pos.z == 50.0

    pip.disconnect()
    assert not pip.is_connected()
    print("  [PASS] VirtualPipette")


if __name__ == "__main__":
    print("Running virtual device tests...")
    test_virtual_camera()
    test_virtual_stage()
    test_virtual_pipette()
    print("All virtual device tests passed!")
