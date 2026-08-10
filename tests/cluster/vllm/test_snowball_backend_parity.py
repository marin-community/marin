# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Snowball parity implementations as remote Iris jobs.

Run the complete standing-cluster gate from the repository root only after
interactive H100 validation::

    uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2

from tests.cluster.vllm.snowball import RepresentativeGolden, read_representative_goldens
from tests.cluster.vllm.snowball_backend_parity_jobs import (
    GPU_COUNT,
    score_levanter_against_goldens,
    score_vllm_against_goldens,
)

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)]


def _levanter_job(goldens: tuple[RepresentativeGolden, ...]) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-levanter-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_levanter_against_goldens, args=[goldens]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="256g", disk="128g"),
        environment=create_environment(
            extras=["gpu"],
            sync_packages=["marin-levanter", "marin-core"],
            env_vars={
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def _vllm_job(
    goldens: tuple[RepresentativeGolden, ...],
    attention_backend: str,
    pipeline_parallel_size: int,
) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-vllm-pp{pipeline_parallel_size}-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            score_vllm_against_goldens,
            args=[goldens, attention_backend, pipeline_parallel_size],
        ),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g"),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        replicas=pipeline_parallel_size,
        max_retries_failure=2,
        max_task_failures=2,
    )


@pytest.mark.parametrize("backend", ["levanter-gpu", "vllm-gpu-pp1", "vllm-gpu-pp2"])
def test_snowball_export_matches_representative_goldens(
    marin_gpu_client: IrisClient,
    backend: str,
    vllm_attention_backend: str,
    run_test_job,
) -> None:
    goldens = read_representative_goldens()
    if backend == "levanter-gpu":
        request = _levanter_job(goldens)
    else:
        pipeline_parallel_size = int(backend.removeprefix("vllm-gpu-pp"))
        request = _vllm_job(goldens, vllm_attention_backend, pipeline_parallel_size)
    run_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
