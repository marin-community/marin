# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Snowball BF16 export on GPU and TPU.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/cluster/vllm/test_snowball_hf_bf16_export.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2

from tests.cluster.vllm.snowball import (
    SNOWBALL_GPU,
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    SNOWBALL_TPU,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_export import export_snowball_bf16

PENDING_TIMEOUT = 20 * 60.0
RUNTIME_TIMEOUT = 60 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


def assert_checkpoint_reproduces_bf16_export(cell, goldens, scratch_root: str, report_uri: str) -> None:
    export_snowball_bf16(
        cell,
        scratch_root=scratch_root,
        report_uri=report_uri,
        goldens=goldens,
    )


def test_snowball_checkpoint_reproduces_gpu_vllm_bf16_export(
    marin_gpu_client: IrisClient,
    run_test_job,
) -> None:
    run_id = uuid.uuid4().hex[:8]
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-bf16-export-gpu-{run_id}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_reproduces_bf16_export,
                args=[
                    SNOWBALL_NATIVE_GPU,
                    read_representative_goldens(),
                    "/tmp",
                    f"s3://marin-us-east-02a/tmp/ttl=30d/snowball-parity/export-gpu-{run_id}.json",
                ],
            ),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="512g", disk="256g"),
            environment=create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_GPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                },
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )


def test_snowball_checkpoint_reproduces_tpu_vllm_bf16_export(
    iris_client: IrisClient,
    smoke_region: str,
    run_test_job,
) -> None:
    run_id = uuid.uuid4().hex[:8]
    run_test_job(
        iris_client,
        JobRequest(
            name=f"snowball-bf16-export-tpu-{run_id}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_reproduces_bf16_export,
                args=[
                    SNOWBALL_NATIVE_TPU,
                    read_representative_goldens(),
                    "/dev/shm",
                    f"gs://marin-us-east5/tmp/ttl=30d/snowball-parity/export-tpu-{run_id}.json",
                ],
            ),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=160,
                ram="640g",
                disk="100g",
                regions=(smoke_region,),
            ),
            environment=create_environment(
                extras=["tpu"],
                sync_packages=["marin-levanter"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_TPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                },
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
