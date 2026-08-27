# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify schema-2 GrugMoE through the ordinary Marin serving path.

Run one case at a time and retain the complete streamed log::

    uv run pytest tests/cluster/vllm/test_grugmoe_schema2_serving.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s -k real_d1536
"""

import subprocess
import uuid
from pathlib import Path

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.testing.inference.grugmoe_schema2_serving import (
    REAL_MODEL_KEY,
    run_grugmoe_schema2_qualification,
)

from experiments.evaluation.models import models
from tests.cluster.conftest import MARIN_GPU_CLUSTER

PENDING_TIMEOUT_SECONDS = 45 * 60.0
RUNTIME_TIMEOUT_SECONDS = 75 * 60.0
ORDINARY_CATALOG_COMMAND = (
    "uv run python -m experiments.evaluation.cli launch --model rav-ladder-d1536 "
    "--evals smoke --limit 1 --platform gpu --accelerator H100x8 --federated_cluster cw-us-east-02a "
    "--priority interactive"
)

pytestmark = [
    pytest.mark.cluster,
    pytest.mark.slow,
    pytest.mark.timeout(PENDING_TIMEOUT_SECONDS + RUNTIME_TIMEOUT_SECONDS + 300),
]


def _clean_git_head() -> str:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise AssertionError("live qualification requires a clean committed Marin worktree")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _qualification_command() -> str:
    return (
        "uv run pytest tests/cluster/vllm/test_grugmoe_schema2_serving.py "
        "-m cluster -o addopts= --import-mode=importlib -vv -s -k real_d1536"
    )


def test_grugmoe_schema2_real_d1536_whole_node(
    marin_gpu_client: IrisClient,
    run_test_job,
) -> None:
    model = models()[REAL_MODEL_KEY]
    request = JobRequest(
        name=f"grugmoe-schema2-real-d1536-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            run_grugmoe_schema2_qualification,
            args=(model, _clean_git_head(), _qualification_command(), ORDINARY_CATALOG_COMMAND),
        ),
        resources=ResourceConfig.with_cpu(
            cpu=4,
            ram="32g",
            disk="32g",
            target_cluster=MARIN_GPU_CLUSTER,
            preemptible=False,
        ),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
        ),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    run_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT_SECONDS,
        runtime_timeout=RUNTIME_TIMEOUT_SECONDS,
    )
