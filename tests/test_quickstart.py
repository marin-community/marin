import os
import subprocess
import tempfile

import pytest
import ray


@pytest.fixture
def ray_start():
    ray.init(namespace="marin")  # setup
    yield
    ray.shutdown()  # teardown


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip this test in CI, since we run it as a separate worflow.")
def test_quickstart_run():
    MARIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    """Test the dry runs of experiment scripts"""
    # Emulate running experiments/quickstart.py on the cmdline
    with tempfile.TemporaryDirectory(prefix="executor-quickstart") as temp_dir:
        result = subprocess.run(
            ["python", os.path.join(MARIN_ROOT, "experiments/quickstart.py"), "--prefix", temp_dir],
            capture_output=False,
            text=True,
        )
    assert result.returncode == 0, f"Quickstart run failed: {result.stderr}"
