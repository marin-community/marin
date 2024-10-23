import subprocess
import tempfile

import pytest
import ray


@pytest.fixture
def ray_start():
    ray.init("local", ignore_reinit_error=True)  # setup
    yield
    ray.shutdown()  # teardown


def test_quickstart_run():
    """Test the dry runs of experiment scripts"""
    # Emulate running experiments/quickstart.py on the cmdline
    with tempfile.TemporaryDirectory(prefix="executor-quickstart") as temp_dir:
        result = subprocess.run(
            ["python", "experiments/quickstart.py", "--prefix", temp_dir],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, f"Quickstart run failed: {result.stderr}"
