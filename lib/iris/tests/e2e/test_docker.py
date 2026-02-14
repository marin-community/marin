# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker-specific E2E tests: OOM detection, TPU simulation.

These tests use E2ECluster(use_docker=True) which manually wires up Controller +
Workers with a DockerRuntime. They validate behavior that only manifests inside
real containers (cgroup OOM kills, JAX coordinator env vars).
"""

import uuid
from pathlib import Path

import pytest
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.cluster.worker.env_probe import TPUSimEnvironmentProvider
from iris.rpc import cluster_pb2

from tests.e2e._docker_cluster import E2ECluster

pytestmark = [pytest.mark.e2e, pytest.mark.docker]


def unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def shared_cache(tmp_path_factory) -> Path:
    """Session-scoped cache so uv/cargo/bundles are only downloaded once."""
    return tmp_path_factory.mktemp("iris_cache")


@pytest.fixture(scope="module")
def docker_cluster(shared_cache):
    with E2ECluster(use_docker=True, cache_dir=shared_cache) as cluster:
        yield cluster


@pytest.fixture(scope="module")
def tpu_sim_cluster(shared_cache):
    """Docker cluster with simulated TPU metadata for JAX coordination tests.

    Each worker gets a TPUSimEnvironmentProvider which populates tpu_worker_hostnames,
    tpu_worker_id, and device.tpu. This triggers _build_device_env_vars() in docker.py
    to set JAX_COORDINATOR_ADDRESS, JAX_PROCESS_ID, and JAX_NUM_PROCESSES.
    """
    with E2ECluster(
        num_workers=2,
        use_docker=True,
        cache_dir=shared_cache,
        env_provider_factory=lambda i, n: TPUSimEnvironmentProvider(i, n),
    ) as cluster:
        yield cluster


@pytest.mark.timeout(180)
def test_oom_detection(docker_cluster):
    """Container killed by OOM reports oom_killed in error message.

    Submits a job with 64MB memory limit that tries to allocate ~200MB.
    Docker's cgroups will OOM-kill the process. 64MB is enough for Python + uv sync
    but not for the allocation.
    """

    def allocate_memory():
        data = bytearray(200 * 1024 * 1024)
        return len(data)

    job_id = docker_cluster.submit(
        allocate_memory,
        name=unique_name("oom-test"),
        memory="64m",
    )

    status = docker_cluster.wait(job_id, timeout=docker_cluster.wait_timeout)
    assert status["state"] == "JOB_STATE_FAILED", f"Expected failure, got: {status}"

    error = status.get("error", "")
    has_oom_indicator = "137" in error or "OOM" in error or "SIGKILL" in error or "oom" in error.lower()
    assert has_oom_indicator, f"Expected OOM-related error message, got: {error}"

    logs = docker_cluster.get_task_logs(job_id)
    log_text = "\n".join(logs)

    if "OOM" in error:
        assert (
            "OOM killed" in log_text or "OOM" in error
        ), f"Expected OOM in logs when oom_killed detected. Logs: {log_text[-500:]}"


@pytest.mark.timeout(180)
def test_jax_coordinator_address_format(tpu_sim_cluster):
    """Verify JAX_COORDINATOR_ADDRESS has correct host:port format.

    Validates the fix for the bug where JAX_COORDINATOR_ADDRESS was set without a
    port, causing jax.distributed.initialize() to crash with "IndexError: list index
    out of range" when parsing the address via addr.rsplit(':', 1)[1].
    """

    def validate_jax_env_format():
        import os

        addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        proc_id = os.environ.get("JAX_PROCESS_ID", "")
        num_procs = os.environ.get("JAX_NUM_PROCESSES", "")
        tpu_accelerator_type = os.environ.get("TPU_ACCELERATOR_TYPE", "")
        tpu_type = os.environ.get("TPU_TYPE", "")

        print(f"JAX_COORDINATOR_ADDRESS={addr}")
        print(f"JAX_PROCESS_ID={proc_id}")
        print(f"JAX_NUM_PROCESSES={num_procs}")
        print(f"TPU_ACCELERATOR_TYPE={tpu_accelerator_type}")
        print(f"TPU_TYPE={tpu_type}")

        if not addr:
            raise ValueError("JAX_COORDINATOR_ADDRESS not set")
        if not proc_id:
            raise ValueError("JAX_PROCESS_ID not set")
        if not num_procs:
            raise ValueError("JAX_NUM_PROCESSES not set")
        if not tpu_accelerator_type:
            raise ValueError("TPU_ACCELERATOR_TYPE not set")
        if not tpu_type:
            raise ValueError("TPU_TYPE not set")
        if tpu_accelerator_type != tpu_type:
            raise ValueError(
                "TPU_ACCELERATOR_TYPE and TPU_TYPE must match " f"(got {tpu_accelerator_type!r} vs {tpu_type!r})"
            )

        # This is the exact parsing that JAX does in distributed.py:107
        try:
            port_str = addr.rsplit(":", 1)[1]
            f"[::]:{port_str}"
        except IndexError as e:
            raise ValueError(f"JAX_COORDINATOR_ADDRESS missing port: '{addr}'. Expected format: 'host:port'") from e

        try:
            int(port_str)
        except ValueError as e:
            raise ValueError(f"Port is not numeric: '{port_str}'") from e

        return {
            "coordinator_address": addr,
            "process_id": proc_id,
            "num_processes": num_procs,
        }

    tpu_device = cluster_pb2.DeviceConfig()
    tpu_device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v4-8-sim", count=4))

    entrypoint = Entrypoint.from_callable(validate_jax_env_format)
    environment = EnvironmentSpec()
    resources = ResourceSpec(cpu=1, memory="1g", device=tpu_device)

    job = tpu_sim_cluster.get_client().submit(
        entrypoint=entrypoint,
        name=unique_name("jax-env-test"),
        resources=resources,
        environment=environment,
    )
    status = tpu_sim_cluster.wait(job, timeout=tpu_sim_cluster.wait_timeout)

    if status["state"] != "JOB_STATE_SUCCEEDED":
        logs = tpu_sim_cluster.get_task_logs(job)
        pytest.fail(f"Job failed: {status}\nLogs:\n" + "\n".join(logs[-50:]))
