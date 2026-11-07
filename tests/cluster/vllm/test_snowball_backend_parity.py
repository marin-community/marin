# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Snowball parity implementations as remote Iris jobs.

Run the complete standing-cluster gate from the repository root only after
interactive H100 validation::

    uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import inspect
import uuid

import cloudpickle
import pytest
from fray.iris_backend import convert_entrypoint
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.testing.inference.snowball import (
    SNOWBALL,
    TPU_REGION,
    TPU_TYPE,
    RepresentativeGolden,
    read_representative_goldens,
)
from marin.testing.inference.snowball_backend_parity_jobs import (
    GPU_COUNT,
    score_levanter_against_goldens,
    score_tpu_vllm_against_goldens,
    score_vllm_against_goldens,
)

from tests.cluster.conftest import MARIN_GPU_CLUSTER

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0


def _levanter_job(goldens: tuple[RepresentativeGolden, ...]) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-levanter-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_levanter_against_goldens, args=[goldens]),
        resources=ResourceConfig.with_gpu(
            "H100", count=GPU_COUNT, cpu=64, ram="256g", disk="128g", target_cluster=MARIN_GPU_CLUSTER
        ),
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
        resources=ResourceConfig.with_gpu(
            "H100",
            count=GPU_COUNT,
            cpu=64,
            ram="512g",
            disk="128g",
            target_cluster=None if get_job_info() is not None else MARIN_GPU_CLUSTER,
        ),
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


def _tpu_vllm_job(goldens: tuple[RepresentativeGolden, ...]) -> JobRequest:
    resources = ResourceConfig.with_tpu(
        TPU_TYPE,
        cpu=32,
        ram="360g",
        disk="100g",
        regions=[TPU_REGION],
        zone=f"{TPU_REGION}-b",
        preemptible=False,
    )
    return JobRequest(
        name=f"snowball-parity-vllm-v4-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            score_tpu_vllm_against_goldens,
            args=[goldens, SNOWBALL.tpu_export_uri, resources.cpu],
        ),
        resources=resources,
        environment=create_environment(extras=["cpu"], env_vars={"JAX_PLATFORMS": "cpu"}),
        ports=RAY_PORTS,
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        max_retries_failure=2,
        max_task_failures=2,
    )


@pytest.mark.cluster
@pytest.mark.slow
@pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)
@pytest.mark.parametrize("backend", ["levanter-gpu", "vllm-gpu-pp1", "vllm-gpu-pp2", "vllm-tpu-v4-tp16"])
def test_snowball_export_matches_representative_goldens(
    marin_gpu_client: IrisClient,
    backend: str,
    vllm_attention_backend: str,
    run_test_job,
) -> None:
    goldens = read_representative_goldens()
    if backend == "levanter-gpu":
        request = _levanter_job(goldens)
    elif backend == "vllm-tpu-v4-tp16":
        request = _tpu_vllm_job(goldens)
    else:
        pipeline_parallel_size = int(backend.removeprefix("vllm-gpu-pp"))
        request = _vllm_job(goldens, vllm_attention_backend, pipeline_parallel_size)
    run_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )


@pytest.mark.parametrize("backend", ["vllm-gpu-pp1", "vllm-gpu-pp2", "vllm-tpu-v4-tp16"])
def test_vllm_job_payload_survives_worker_serialization(backend):
    goldens = read_representative_goldens()
    request = (
        _tpu_vllm_job(goldens) if backend == "vllm-tpu-v4-tp16" else _vllm_job(goldens, "FLASH_ATTN", int(backend[-1]))
    )
    assert request.priority == job_pb2.PRIORITY_BAND_PRODUCTION
    assert request.max_retries_failure == 2
    assert request.max_task_failures == 2
    entrypoint = convert_entrypoint(request.entrypoint)
    function, args, kwargs = cloudpickle.loads(entrypoint.workdir_files["_callable.pkl"])
    bound = inspect.signature(function).bind(*args, **kwargs)
    assert bound.arguments["goldens"] == goldens
    if backend == "vllm-tpu-v4-tp16":
        assert bound.arguments["cpus_per_host"] == request.resources.cpu
        assert bound.arguments["weights"] == SNOWBALL.tpu_export_uri
    else:
        assert bound.arguments["pipeline_parallel_size"] == int(backend[-1])


@pytest.mark.parametrize("nested", [False, True])
def test_gpu_job_routes_only_root_submissions_to_the_peer(monkeypatch, nested):
    monkeypatch.setattr(
        "tests.cluster.vllm.test_snowball_backend_parity.get_job_info", lambda: object() if nested else None
    )
    request = _vllm_job(read_representative_goldens(), "FLASH_ATTN", 1)
    assert request.resources.target_cluster == (None if nested else MARIN_GPU_CLUSTER)
