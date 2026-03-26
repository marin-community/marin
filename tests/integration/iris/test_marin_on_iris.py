# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full marin integration pipeline running on Iris.

Ports the pipeline from tests/integration_test.py to dispatch through an Iris
cluster via FrayIrisClient instead of Ray.

Unlike the local integration test, this dispatches Zephyr coordinator/worker
jobs to a remote cluster. All paths (prefix, input data) must be on
S3-compatible storage so the remote pods can access them.
"""

import logging
import uuid
from pathlib import Path

import fsspec
import pytest
from fray import set_current_client
from fray.v2.iris_backend import FrayIrisClient
from marin.execution.executor import ExecutorMainConfig, executor_main
from tests.integration_test import create_steps

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SYNTH_DATA = REPO_ROOT / "tests" / "quickstart-data"

# S3 prefix for CI artifacts. Cleaned up after each test run.
S3_CI_BASE = "s3://marin-na/temp/ci"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _upload_tree(local_root: Path, s3_dest: str) -> None:
    """Upload a local directory tree to S3."""
    fs, _ = fsspec.core.url_to_fs(s3_dest)
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root)
        fs.put(str(path), f"{s3_dest}/{rel}")


def _rm_tree(s3_prefix: str) -> None:
    """Remove all objects under an S3 prefix (best-effort)."""
    fs, _ = fsspec.core.url_to_fs(s3_prefix)
    try:
        fs.rm(s3_prefix, recursive=True)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(600)
def test_marin_pipeline_on_iris(integration_cluster, monkeypatch):
    """Run the full marin data pipeline dispatched through Iris."""
    run_id = f"marin-itest-{uuid.uuid4().hex[:8]}"
    prefix = f"{S3_CI_BASE}/{run_id}"

    try:
        monkeypatch.setenv("MARIN_PREFIX", prefix)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setenv("WANDB_API_KEY", "")
        monkeypatch.setenv("JAX_TRACEBACK_FILTERING", "off")

        # Upload local test fixtures so remote Zephyr pods can read them.
        synth_data = f"{prefix}/quickstart-data"
        _upload_tree(LOCAL_SYNTH_DATA, synth_data)

        iris_client = FrayIrisClient(
            controller_address=integration_cluster.url,
            workspace=REPO_ROOT,
        )

        config = ExecutorMainConfig(
            prefix=prefix,
            executor_info_base_path=f"{prefix}/experiments",
        )

        experiment_prefix = "quickstart-tests"
        steps = create_steps(experiment_prefix, synth_data)

        with set_current_client(iris_client):
            executor_main(config, steps=steps)
    finally:
        _rm_tree(prefix)
