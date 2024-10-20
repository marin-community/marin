import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest
import ray


@pytest.fixture
def ray_start():
    # (nothing to do for setup)
    yield
    ray.shutdown()  # teardown


def test_run_dry_runs():
    """Test the dry runs of experiment scripts"""
    experiments_dir = Path("experiments")
    skip_marker = "# SKIP_DRY_RUN_TEST"
    if experiments_dir.exists() and experiments_dir.is_dir():

        for script in experiments_dir.glob("*.py"):
            start = time.time()  # Start time for performance measurement
            # Check if the script contains the skip marker
            with open(script, "r") as file:
                content = file.read()
                if skip_marker in content:
                    print(f"Skipping {script} (contains skip marker)")
                    continue  # Skip this file

            with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
                result = subprocess.run(
                    ["python", script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir],
                    capture_output=True,
                    text=True,
                )

            assert result.returncode == 0, f"Dry run failed for {script}"
            print(f"Execution time for {script}: {time.time() - start} seconds")
    else:
        print("No experiments directory found")


if __name__ == "__main__":
    unittest.main()
