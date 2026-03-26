# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full marin integration pipeline running on Iris.

Ports the pipeline from tests/integration_test.py to dispatch through an Iris
cluster via FrayIrisClient instead of Ray.

When MARIN_CI_S3_PREFIX is set (e.g. on CoreWeave CI), all paths use S3 so
remote Zephyr pods can access them. The executor is submitted as an Iris job
so that child jobs (Zephyr coordinator/workers) inherit env vars — including
S3 credentials — via Iris auto-propagation.

Otherwise falls back to running in-process with a local tmpdir (works when
the Iris cluster is local / in-process).
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import fsspec
import pytest
from fray import set_current_client
from fray.v2.iris_backend import FrayIrisClient
from fray.v2.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from marin.execution.executor import ExecutorMainConfig, executor_main
from tests.integration_test import create_steps

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SYNTH_DATA = REPO_ROOT / "tests" / "quickstart-data"

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# S3/R2 env vars that must be forwarded to Iris jobs for object storage access.
_S3_ENV_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "FSSPEC_S3"]


def _upload_tree(local_root: Path, s3_dest: str) -> None:
    """Upload a local directory tree to S3."""
    fs, _ = fsspec.core.url_to_fs(s3_dest)
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root)
        fs.put(str(path), f"{s3_dest}/{rel}")


def _rm_s3(s3_prefix: str) -> None:
    """Remove all objects under an S3 prefix (best-effort)."""
    fs, _ = fsspec.core.url_to_fs(s3_prefix)
    try:
        fs.rm(s3_prefix, recursive=True)
    except FileNotFoundError:
        pass


def _s3_env_vars() -> dict[str, str]:
    """Collect S3/R2 env vars from the current process."""
    return {k: os.environ[k] for k in _S3_ENV_KEYS if k in os.environ}


def _run_executor(prefix: str, synth_data: str) -> None:
    """Entry point for the Iris job that runs the full executor pipeline."""
    config = ExecutorMainConfig(
        prefix=prefix,
        executor_info_base_path=f"{prefix}/experiments",
    )
    steps = create_steps("quickstart-tests", synth_data)
    executor_main(config, steps=steps)


@pytest.mark.timeout(600)
def test_marin_pipeline_on_iris(integration_cluster, monkeypatch):
    """Run the full marin data pipeline dispatched through Iris."""
    s3_base = os.environ.get("MARIN_CI_S3_PREFIX")

    if s3_base:
        # Remote cluster: use S3 so Zephyr coordinator pods can access data.
        run_id = f"marin-itest-{uuid.uuid4().hex[:8]}"
        prefix = f"{s3_base}/{run_id}"
        synth_data = f"{prefix}/quickstart-data"
        _upload_tree(LOCAL_SYNTH_DATA, synth_data)
        cleanup = lambda: _rm_s3(prefix)  # noqa: E731
    else:
        # Local cluster: local filesystem is fine.
        prefix = tempfile.mkdtemp(prefix="iris-marin-itest-")
        synth_data = str(LOCAL_SYNTH_DATA)
        cleanup = lambda: shutil.rmtree(prefix, ignore_errors=True)  # noqa: E731

    try:
        monkeypatch.setenv("MARIN_PREFIX", prefix)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setenv("WANDB_API_KEY", "")
        monkeypatch.setenv("JAX_TRACEBACK_FILTERING", "off")

        iris_client = FrayIrisClient(
            controller_address=integration_cluster.url,
            workspace=REPO_ROOT,
        )

        config = ExecutorMainConfig(
            prefix=prefix,
            executor_info_base_path=f"{prefix}/experiments",
        )
        steps = create_steps("quickstart-tests", synth_data)

        if s3_base:
            # Remote cluster: submit the executor as an Iris job so child jobs
            # (Zephyr coordinator/workers) inherit S3 env vars automatically.
            env_vars = {
                "MARIN_PREFIX": prefix,
                "WANDB_MODE": "disabled",
                "WANDB_API_KEY": "",
                "JAX_TRACEBACK_FILTERING": "off",
                **_s3_env_vars(),
            }

            with set_current_client(iris_client):
                handle = iris_client.submit(
                    JobRequest(
                        name=f"marin-itest-{uuid.uuid4().hex[:8]}",
                        entrypoint=Entrypoint.from_callable(
                            _run_executor,
                            args=(prefix, synth_data),
                        ),
                        resources=ResourceConfig.with_cpu(),
                        environment=create_environment(env_vars=env_vars),
                    )
                )
                handle.wait(raise_on_failure=True)
        else:
            # Local cluster: run in-process (local filesystem is accessible).
            with set_current_client(iris_client):
                executor_main(config, steps=steps)
    finally:
        cleanup()
