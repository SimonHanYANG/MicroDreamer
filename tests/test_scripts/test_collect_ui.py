"""Tests for data collection UI components."""

import sys
import time
import json
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scripts.collect_ui import PIDController, DataCollector


def test_pid_controller():
    # Pure proportional test
    pid = PIDController(kp=0.5, ki=0.0, kd=0.0, output_limit=100.0)
    output1 = pid.update(10.0)
    assert output1 > 0, "PID should produce positive output for positive error"
    assert abs(output1 - 5.0) < 0.01, f"P-only: kp*error = 0.5*10 = 5.0, got {output1}"
    assert output1 <= 100.0, "PID output should respect limit"

    # Same error should give same output (no integral/derivative)
    output2 = pid.update(10.0)
    assert abs(output2 - output1) < 0.01, "P-only with same error should give same output"

    # Negative error
    output_neg = pid.update(-5.0)
    assert output_neg < 0, "PID should produce negative output for negative error"

    # Output limit
    pid2 = PIDController(kp=100.0, ki=0, kd=0, output_limit=10.0)
    output_limited = pid2.update(100.0)
    assert output_limited == 10.0, "PID should clamp output to limit"

    # Derivative response
    pid3 = PIDController(kp=0.0, ki=0.0, kd=0.1, output_limit=100.0)
    pid3._last_time = time.time() - 0.1  # simulate 100ms gap
    d_output = pid3.update(10.0)  # error changes from 0 to 10
    assert d_output > 0, "Derivative should respond to error change"

    # Integral accumulation (simulate real time)
    pid4 = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limit=100.0)
    pid4._last_time = time.time() - 1.0  # simulate 1 second of accumulation
    i_output = pid4.update(10.0)  # integral = 10 * 1.0 = 10
    assert i_output > 0, "Integral should accumulate over time"

    # Anti-windup: integral clamped
    pid5 = PIDController(kp=0.0, ki=10.0, kd=0, output_limit=50.0)
    for i in range(100):
        pid5._last_time = time.time() - 0.1  # 100ms each step
        pid5.update(10.0)
    assert abs(pid5._integral) <= 50.0, "Integral should be clamped by anti-windup"

    # Reset
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0
    assert pid._last_time is None

    print("  [PASS] pid_controller")


def test_data_collector():
    tmpdir = Path("_test_collector_tmp")
    tmpdir.mkdir(exist_ok=True)
    try:
        collector = DataCollector(output_dir=str(tmpdir))

        # Not recording initially
        assert not collector.recording
        assert collector.frame_count == 0

        # Start recording
        collector.start_recording("test task")
        assert collector.recording

        # Add samples
        for i in range(20):
            frame = np.random.randint(0, 255, (1200, 1600), dtype=np.uint8)
            collector.add_sample(frame, (i * 1.0, i * 2.0), (i * 0.5, i * 0.3, 50.0))
        assert collector.frame_count == 20

        # Stop and save
        episode_path = collector.stop_recording()
        assert episode_path is not None
        assert not collector.recording

        # Verify files
        ep_dir = Path(episode_path)
        assert (ep_dir / "data.npz").exists()
        assert (ep_dir / "metadata.json").exists()

        # Verify data
        data = np.load(ep_dir / "data.npz")
        assert data["frames"].shape == (20, 1200, 1600)
        assert data["frames"].dtype == np.uint8
        assert data["stage_positions"].shape == (20, 2)
        assert data["pipette_positions"].shape == (20, 3)
        del data

        # Verify metadata
        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["num_frames"] == 20
        assert meta["task_description"] == "test task"
        assert "timestamp" in meta
        assert "episode_id" in meta

        print("  [PASS] data_collector")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_data_collector_empty():
    """Test that stopping without data returns None."""
    tmpdir = Path("_test_collector_empty")
    tmpdir.mkdir(exist_ok=True)
    try:
        collector = DataCollector(output_dir=str(tmpdir))
        collector.start_recording("empty test")
        result = collector.stop_recording()
        assert result is None, "Should return None when no frames collected"
        print("  [PASS] data_collector_empty")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("Running collect UI tests...")
    test_pid_controller()
    test_data_collector()
    test_data_collector_empty()
    print("All collect UI tests passed!")
