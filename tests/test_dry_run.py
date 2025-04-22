import logging
import runpy
import sys
import tempfile
import time
from pathlib import Path

import pytest
import ray

from tests.test_utils import parameterize_with_configs

logger = logging.getLogger(__name__)


@pytest.fixture
def ray_start():
    ray.init(namespace="marin")
    yield
    ray.shutdown()  # teardown


marin_root = Path(__file__).parent.parent
experiments_dir = marin_root / "experiments"


@parameterize_with_configs(pattern="**/*.py", config_path=str(experiments_dir))
def test_run_dry_runs(config_file, monkeypatch):
    """Test the dry runs of experiment scripts"""
    script = config_file
    # first get this script path
    print(f"Running dry run for {script}")
    start = time.time()  # Start time for performance measurement
    # Check if the script contains the skip marker
    with open(script, "r") as file:
        content = file.read()
        # Hack: Skip files that don't seem to call executor_main
        if "executor_main(" not in content:
            pytest.skip(f"Skipping {script} (no executor_main)")

        if "nodryrun" in content:
            pytest.skip(f"Skipping {script} (contains nodryrun marker)")

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        monkeypatch.setattr(
            sys, "argv", [script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir]
        )
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as e:
            # Some scripts may call `sys.exit()`, so we catch it to treat `exit(0)` as success
            assert e.code == 0, f"Dry run failed with exit code {e.code} for {script}"
        except Exception as e:
            # pytest.fail(f"Dry run exception in {script}: {e}")
            # raise Failed(f"Dry run exception in {script}: {e}") from e
            raise e
        finally:
            print(f"Execution time for {script}: {time.time() - start:.2f} seconds")

        # result = subprocess.run(
        #     ["python", script, "--dry_run", "True", "--executor_info_base_path", temp_dir, "--prefix", temp_dir],
        #     # capture_output=True,
        #     text=True,
        #     cwd=str(marin_root),
        # )

    # logger.info(f"Execution time for {script}: {time.time() - start} seconds")
    print(f"Execution time for {script}: {time.time() - start} seconds")
