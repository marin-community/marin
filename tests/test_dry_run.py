import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest
import ray


@pytest.fixture
def ray_start():
    # (nothing to do for setup)
    yield
    ray.shutdown()  # teardown


def create_temp_dir():
    with tempfile.TemporaryDirectory(prefix="executor-", delete=False) as temp_dir:
        return temp_dir


def test_run_dry_runs():
    """Test the dry runs of experiment scripts"""
    experiments_dir = Path("experiments")
    if experiments_dir.exists() and experiments_dir.is_dir():
        for script in experiments_dir.glob("*.py"):
            temp_dir = create_temp_dir()
            result = subprocess.run(
                ["python", script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Dry run failed for {script}"
            print(f"Output for {script}: {result.stdout}")
            os.unlink(temp_dir)
    else:
        print("No experiments directory found")


if __name__ == "__main__":
    unittest.main()
