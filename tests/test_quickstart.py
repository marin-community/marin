import os
import subprocess
import unittest

import pytest
import ray


@pytest.fixture
def ray_start():
    # (nothing to do for setup)
    yield
    ray.shutdown()  # teardown


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping in GitHub CI")  # Skip in CI
def test_quickstart_run():
    """Test the dry runs of experiment scripts"""
    # Emulate running experiments/quickstart.py on the cmdline
    result = subprocess.run(
        ["python", "experiments/quickstart.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Quickstart run failed: {result.stderr}"


if __name__ == "__main__":
    unittest.main()
