# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BF16 Hugging Face export regressions for the June 67B checkpoint.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_hf_bf16_export.py -o addopts= -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem import StoragePath

from .export_model import export_checkpoint_bf16
from .iris import run_remote_test_job
from .june_67b import (
    CHECKPOINT_PATH,
    EXECUTOR_INFO_PATH,
    GCS_CHECKPOINT_PATH,
    GCS_EXECUTOR_INFO_PATH,
)
from .reference import CHECKPOINT_NAME, GCS_MODEL_COMPLETION_URI, GCS_MODEL_URI

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 60 * 60.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60),
]


def assert_gpu_checkpoint_bf16_export() -> None:
    export_checkpoint_bf16(
        executor_info_path=EXECUTOR_INFO_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )


def export_and_publish_tpu_host_checkpoint_bf16() -> None:
    staging_uri = f"gs://marin-us-east5/tmp/ttl=14d/vllm-export/june-67b-a2b-{CHECKPOINT_NAME}/{uuid.uuid4().hex}/"
    staging_path = StoragePath(staging_uri)
    try:
        export_checkpoint_bf16(
            executor_info_path=GCS_EXECUTOR_INFO_PATH,
            checkpoint_path=GCS_CHECKPOINT_PATH,
            staging_uri=staging_uri,
            publish_uri=GCS_MODEL_URI,
            completion_uri=GCS_MODEL_COMPLETION_URI,
        )
    finally:
        if staging_path.exists():
            staging_path.rmtree()


def test_h100_node_exports_checkpoint_as_vllm_bf16(marin_gpu_client: IrisClient) -> None:
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"june-67b-bf16-export-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(assert_gpu_checkpoint_bf16_export),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="512g", disk="256g"),
            environment=create_environment(extras=["gpu"], sync_packages=["marin-levanter"]),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )


@pytest.mark.tpu_ci
def test_v6e_8_host_exports_and_publishes_checkpoint(marin_tpu_client: IrisClient) -> None:
    run_remote_test_job(
        marin_tpu_client,
        JobRequest(
            name=f"june-67b-bf16-tpu-host-export-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(export_and_publish_tpu_host_checkpoint_bf16),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=160,
                ram="640g",
                disk="32g",
                zone="us-east5-b",
            ),
            environment=create_environment(
                extras=["tpu"],
                sync_packages=["marin-levanter"],
                env_vars={"JAX_PLATFORMS": "cpu", "TMPDIR": "/dev/shm"},
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
