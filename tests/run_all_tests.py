"""Run all MicroDreamer tests."""

import sys
import subprocess
from pathlib import Path

TESTS = [
    "tests/test_hardware/test_virtual_devices.py",
    "tests/test_data/test_preprocessor.py",
    "tests/test_data/test_dataset.py",
    "tests/test_models/test_action_model.py",
    "tests/test_models/test_video_model.py",
    "tests/test_inference/test_e2e.py",
    "tests/test_scripts/test_collect_ui.py",
    "tests/test_e2e_pipeline.py",
]

def main():
    project_root = Path(__file__).parent.parent
    passed = 0
    failed = 0

    for test in TESTS:
        print(f"\n{'='*60}")
        print(f"Running: {test}")
        print('='*60)
        result = subprocess.run(
            [sys.executable, test],
            cwd=str(project_root),
            capture_output=False,
        )
        if result.returncode == 0:
            passed += 1
        else:
            failed += 1
            print(f"  FAILED: {test}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print('='*60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
