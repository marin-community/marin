import logging
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import ray

logger = logging.getLogger(__name__)


@pytest.fixture
def ray_start():
    ray.init(namespace="marin")
    yield
    ray.shutdown()  # teardown


def test_run_dry_runs():
    """Test the dry runs of experiment scripts"""
    experiments_dir = Path("experiments")
    if experiments_dir.is_dir():
        for script in experiments_dir.glob("*.py"):
            start = time.time()  # Start time for performance measurement
            # Check if the script contains the skip marker
            with open(script, "r") as file:
                content = file.read()
                # Hack: Skip files that don't seem to call executor_main
                if "executor_main(" in content:
                    logger.info(f"Skipping {script} (no executor_main)")
                    continue

            with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
                result = subprocess.run(
                    ["python", script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir],
                    capture_output=True,
                    text=True,
                )

            assert result.returncode == 0, f"Dry run failed for {script}"
            logger.info(f"Execution time for {script}: {time.time() - start} seconds")
    else:
        logger.info("No experiments directory found")
