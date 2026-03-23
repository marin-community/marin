# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris integration tests exercising job lifecycle, scheduling, and cluster features.

These tests replace the "cloud smoke test" and run against either a local in-process
cluster (default) or an existing controller via --controller-url. No dashboard
screenshots are taken here; those remain in lib/iris/tests/e2e/test_smoke.py.
"""

import logging
import os
import subprocess
import time
import uuid
from unittest.mock import patch

import pytest
from iris.client.client import IrisClient
from iris.cluster.config import connect_cluster, load_config, make_local_config
from iris.cluster.constraints import region_constraint
from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.types import (
    Entrypoint,
    ReservationEntry,
    ResourceSpec,
    gpu_device,
)
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2, config_pb2, logging_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff

from .conftest import (
    DEFAULT_CONFIG,
    IRIS_ROOT,
    IrisIntegrationCluster,
)
from .jobs import IntegrationJobs

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ============================================================================
# Job lifecycle
# ============================================================================


def test_submit_and_succeed(integration_cluster):
    """Submit a simple job and verify it succeeds."""
    job = integration_cluster.submit(IntegrationJobs.quick, "itest-simple")
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_submit_and_fail(integration_cluster):
    """Submit a failing job and verify it reports failure."""
    job = integration_cluster.submit(IntegrationJobs.fail, "itest-fail")
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_cancel_job_releases_resources(integration_cluster):
    """Cancelling a running job decommits worker resources so new jobs can schedule.

    Regression test for #3553.
    """
    heavy_cpu = 8 if integration_cluster.is_remote else 900
    job = integration_cluster.submit(IntegrationJobs.sleep, "itest-cancel-heavy", 30, cpu=heavy_cpu)
    integration_cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=integration_cluster.job_timeout)

    integration_cluster.kill(job)
    killed_status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert killed_status.state == cluster_pb2.JOB_STATE_KILLED

    followup = integration_cluster.submit(IntegrationJobs.quick, "itest-cancel-followup", cpu=heavy_cpu)
    followup_status = integration_cluster.wait(followup, timeout=integration_cluster.job_timeout)
    assert followup_status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Scheduling & endpoint verification
# ============================================================================


def test_endpoint_registration(integration_cluster):
    """Endpoint registered from inside job via RPC."""
    prefix = f"itest-ep-{uuid.uuid4().hex[:8]}"
    job = integration_cluster.submit(IntegrationJobs.register_endpoint, "itest-endpoint", prefix)
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_port_allocation(integration_cluster, capabilities):
    """Port allocation job succeeded."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider does not inject port allocations into task pods yet")
    job = integration_cluster.submit(IntegrationJobs.validate_ports, "itest-ports", ports=["http", "grpc"])
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_reservation_gates_scheduling(integration_cluster):
    """Unsatisfiable reservation blocks scheduling; regular jobs proceed."""
    with integration_cluster.launched_job(
        IntegrationJobs.quick,
        "itest-reserved",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("NONEXISTENT-GPU-9999", 99)))
        ],
    ) as reserved:
        reserved_status = integration_cluster.status(reserved)
        assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

        regular = integration_cluster.submit(IntegrationJobs.quick, "itest-regular-while-reserved")
        status = integration_cluster.wait(regular, timeout=integration_cluster.job_timeout)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Log level verification
# ============================================================================


@pytest.fixture(scope="module")
def verbose_job(integration_cluster):
    """Shared verbose log job used by log-related tests."""
    job = integration_cluster.submit(IntegrationJobs.log_verbose, "itest-verbose")
    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    return job


def test_log_levels_populated(integration_cluster, verbose_job, capabilities):
    """Task logs have level field (INFO, WARNING, ERROR)."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider log collection does not parse structured levels yet")

    task_id = verbose_job.job_id.task(0).to_wire()

    deadline = time.monotonic() + integration_cluster.job_timeout
    entries = []
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = integration_cluster.controller_client.get_task_logs(request)
        entries = []
        for batch in response.task_logs:
            entries.extend(batch.logs)
        if any("info-marker" in e.data for e in entries):
            break
        time.sleep(0.5)

    markers_found = {}
    for entry in entries:
        for marker in ("info-marker", "warning-marker", "error-marker"):
            if marker in entry.data:
                markers_found[marker] = entry.level

    assert "info-marker" in markers_found, f"info-marker not found. Got {len(entries)} entries"
    assert markers_found["info-marker"] == logging_pb2.LOG_LEVEL_INFO
    assert markers_found.get("warning-marker") == logging_pb2.LOG_LEVEL_WARNING
    assert markers_found.get("error-marker") == logging_pb2.LOG_LEVEL_ERROR


def test_log_level_filter(integration_cluster, verbose_job, capabilities):
    """min_level=WARNING excludes INFO."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider log collection does not parse structured levels yet")

    task_id = verbose_job.job_id.task(0).to_wire()

    request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id, min_level="WARNING")
    response = integration_cluster.controller_client.get_task_logs(request)
    filtered = []
    for batch in response.task_logs:
        filtered.extend(batch.logs)

    filtered_data = [e.data for e in filtered]
    assert any("warning-marker" in d for d in filtered_data), f"warning-marker missing: {filtered_data}"
    assert any("error-marker" in d for d in filtered_data), f"error-marker missing: {filtered_data}"
    assert not any("info-marker" in d for d in filtered_data if d), "info-marker should be filtered out"


# ============================================================================
# Multi-region routing
# ============================================================================


def test_region_constrained_routing(integration_cluster, capabilities):
    """Job with region constraint lands on correct worker."""
    if not capabilities.has_multi_region:
        pytest.skip("No multi-region workers in cluster")

    target_region = capabilities.regions[0]
    job = integration_cluster.submit(
        IntegrationJobs.noop,
        "itest-region",
        constraints=[region_constraint([target_region])],
    )
    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)

    task = integration_cluster.task_status(job, task_index=0)
    assert task.worker_id

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = integration_cluster.controller_client.list_workers(request)
    from iris.cluster.constraints import WellKnownAttribute

    worker = next(
        (w for w in response.workers if w.worker_id == task.worker_id or w.address == task.worker_id),
        None,
    )
    assert worker is not None
    region_attr = worker.metadata.attributes.get(WellKnownAttribute.REGION)
    if region_attr and region_attr.HasField("string_value"):
        assert region_attr.string_value == target_region, f"Expected {target_region}, got {region_attr.string_value}"


# ============================================================================
# Profiling
# ============================================================================


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="py-spy ptrace can segfault worker threads in CI")
def test_profile_running_task(integration_cluster):
    """Profile a running task, verify data returned."""
    if integration_cluster.is_remote:
        pytest.skip("py-spy races with short-lived containers in remote mode")
    job = integration_cluster.submit(IntegrationJobs.busy_loop, name="itest-profile")

    last_state = "unknown"

    def _is_running():
        nonlocal last_state
        task = integration_cluster.task_status(job, task_index=0)
        last_state = task.state
        return last_state == cluster_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(integration_cluster.job_timeout),
        error_message=f"Task did not reach RUNNING within {integration_cluster.job_timeout}s, last state: {last_state}",
    )
    task_id = integration_cluster.task_status(job, task_index=0).task_id

    request = cluster_pb2.ProfileTaskRequest(
        target=task_id,
        duration_seconds=1,
        profile_type=cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH)),
    )
    response = integration_cluster.controller_client.profile_task(request, timeout_ms=3000)
    assert len(response.profile_data) > 0
    assert not response.error

    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)


# ============================================================================
# Exec in container
# ============================================================================


@pytest.mark.timeout(300)
def test_exec_in_container(integration_cluster):
    """Exec a command in a running task's container."""
    job = integration_cluster.submit(IntegrationJobs.sleep, "itest-exec", 120)
    integration_cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=integration_cluster.job_timeout)

    task_id = integration_cluster.task_status(job, task_index=0).task_id
    deadline = time.monotonic() + integration_cluster.job_timeout
    task = integration_cluster.task_status(job, task_index=0)
    while time.monotonic() < deadline:
        task = integration_cluster.task_status(job, task_index=0)
        if task.state == cluster_pb2.TASK_STATE_RUNNING:
            break
        time.sleep(0.5)
    assert task.state == cluster_pb2.TASK_STATE_RUNNING, f"Task stuck in {cluster_pb2.TaskState.Name(task.state)}"

    request = cluster_pb2.Controller.ExecInContainerRequest(
        task_id=task_id,
        command=["echo", "hello"],
    )
    response = integration_cluster.controller_client.exec_in_container(request)
    assert not response.error, f"exec failed: {response.error}"
    assert response.exit_code == 0
    assert "hello" in response.stdout

    integration_cluster.kill(job)


# ============================================================================
# Checkpoint / restore
# ============================================================================


def test_checkpoint_restore():
    """Controller restart resumes from checkpoint: completed jobs visible, cluster functional."""
    from iris.cluster.local_cluster import LocalCluster

    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)

    cluster = LocalCluster(config)
    url = cluster.start()
    try:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisIntegrationCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(1, timeout=30)

        job = tc.submit(IntegrationJobs.quick, "pre-restart")
        tc.wait(job, timeout=30)
        saved_job_id = job.job_id.to_wire()

        ckpt = controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
        assert ckpt.checkpoint_path, "begin_checkpoint returned empty path"
        assert ckpt.job_count >= 1
        controller_client.close()

        url = cluster.restart()

        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisIntegrationCluster(
            url=url, client=IrisClient.remote(url, workspace=IRIS_ROOT), controller_client=controller_client
        )

        resp = controller_client.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=saved_job_id))
        assert (
            resp.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
        ), f"Pre-restart job has state {resp.job.state} after restore"

        tc.wait_for_workers(1, timeout=30)
        post_job = tc.submit(IntegrationJobs.quick, "post-restart")
        status = tc.wait(post_job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

        controller_client.close()
    finally:
        cluster.close()


# ============================================================================
# Stress test
# ============================================================================


@pytest.mark.timeout(600)
def test_stress_50_tasks(integration_cluster):
    """50 concurrent tasks exercises scheduler concurrency and bin-packing."""
    job = integration_cluster.submit(
        IntegrationJobs.quick,
        "itest-stress-50",
        cpu=0,
        replicas=50,
    )
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout * 2)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# GPU metadata (local-only, creates standalone cluster with mocked nvidia-smi)
# ============================================================================

_NVIDIA_SMI_H100_8X = "\n".join(["NVIDIA H100 80GB HBM3, 81559"] * 8)


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["placeholder"]
    sg.name = "placeholder"
    sg.num_vms = 1
    sg.min_slices = 0
    sg.max_slices = 0
    sg.resources.cpu_millicores = 1000
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


def test_gpu_worker_metadata(tmp_path):
    """Mocked nvidia-smi registers GPU metadata on worker."""
    config = _make_controller_only_config()

    with connect_cluster(config) as url:
        original_run = subprocess.run
        with patch(
            "iris.cluster.worker.env_probe.subprocess.run",
            side_effect=lambda cmd, *a, **kw: (
                subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_NVIDIA_SMI_H100_8X, stderr="")
                if isinstance(cmd, list) and cmd and cmd[0] == "nvidia-smi"
                else original_run(cmd, *a, **kw)
            ),
        ):
            env_provider = DefaultEnvironmentProvider()
            threads = ThreadContainer(name="test-gpu-worker")
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=0,
                cache_dir=cache_dir,
                controller_address=url,
                worker_id=f"test-gpu-worker-{uuid.uuid4().hex[:8]}",
                poll_interval=Duration.from_seconds(0.1),
            )
            worker = Worker(
                worker_config,
                container_runtime=ProcessRuntime(cache_dir=cache_dir),
                environment_provider=env_provider,
                threads=threads,
            )
            worker.start()

            try:
                controller_client = ControllerServiceClientSync(address=url, timeout_ms=10000)
                deadline = time.monotonic() + 15.0
                workers = []
                while time.monotonic() < deadline:
                    request = cluster_pb2.Controller.ListWorkersRequest()
                    response = controller_client.list_workers(request)
                    workers = [w for w in response.workers if w.healthy]
                    if workers:
                        break
                    time.sleep(0.5)

                assert workers, "Worker did not register within timeout"
                w = workers[0]
                meta = w.metadata
                assert meta.gpu_count == 8
                assert "H100" in meta.gpu_name
                assert meta.gpu_memory_mb == 81559
                assert meta.device.gpu.count == 8
                assert "H100" in meta.device.gpu.variant

                from iris.cluster.constraints import WellKnownAttribute

                attrs = meta.attributes
                assert WellKnownAttribute.GPU_VARIANT in attrs
                assert "H100" in attrs[WellKnownAttribute.GPU_VARIANT].string_value
                assert WellKnownAttribute.GPU_COUNT in attrs
                assert attrs[WellKnownAttribute.GPU_COUNT].int_value == 8

                controller_client.close()
            finally:
                worker.stop()
                threads.stop(timeout=Duration.from_seconds(5.0))


# ============================================================================
# Authentication (standalone clusters with auth enabled)
# ============================================================================

_AUTH_TOKEN = "e2e-test-token"
_AUTH_USER = "test-user"


def _login_for_jwt(url: str, identity_token: str) -> str:
    """Exchange a raw identity token for a JWT via the Login RPC."""
    client = ControllerServiceClientSync(address=url, timeout_ms=10000)
    try:
        resp = client.login(cluster_pb2.LoginRequest(identity_token=identity_token))
        return resp.token
    finally:
        client.close()


def test_static_auth_rpc_access():
    """Static auth rejects unauthenticated and wrong-token RPCs, accepts valid JWT."""
    from connectrpc.errors import ConnectError
    from iris.cluster.local_cluster import LocalCluster
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        list_req = cluster_pb2.Controller.ListWorkersRequest()

        unauth_client = ControllerServiceClientSync(address=url, timeout_ms=5000)
        with pytest.raises(ConnectError, match=r"(?i)authorization"):
            unauth_client.list_workers(list_req)
        unauth_client.close()

        wrong_injector = AuthTokenInjector(StaticTokenProvider("wrong-token"))
        wrong_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[wrong_injector])
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            wrong_client.list_workers(list_req)
        wrong_client.close()

        jwt_token = _login_for_jwt(url, _AUTH_TOKEN)
        valid_injector = AuthTokenInjector(StaticTokenProvider(jwt_token))
        valid_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[valid_injector])
        response = valid_client.list_workers(list_req)
        assert response is not None
        valid_client.close()
    finally:
        controller.close()


def test_static_auth_job_ownership():
    """Job ownership: user A cannot terminate user B's job."""
    from connectrpc.errors import ConnectError
    from iris.cluster.local_cluster import LocalCluster
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    _TOKEN_A = "token-user-a"
    _TOKEN_B = "token-user-b"

    config = _make_controller_only_config()
    config.auth.static.tokens[_TOKEN_A] = "user-a"
    config.auth.static.tokens[_TOKEN_B] = "user-b"
    controller = LocalCluster(config)
    url = controller.start()

    try:
        jwt_a = _login_for_jwt(url, _TOKEN_A)
        jwt_b = _login_for_jwt(url, _TOKEN_B)

        injector_a = AuthTokenInjector(StaticTokenProvider(jwt_a))
        client_a = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_a])

        entrypoint = Entrypoint.from_callable(IntegrationJobs.quick)
        launch_req = cluster_pb2.Controller.LaunchJobRequest(
            name="/user-a/auth-owned-job",
            entrypoint=entrypoint.to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
        )
        resp = client_a.launch_job(launch_req)
        job_id = resp.job_id

        injector_b = AuthTokenInjector(StaticTokenProvider(jwt_b))
        client_b = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_b])
        with pytest.raises(ConnectError, match="cannot access resources owned by"):
            client_b.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.close()
        client_b.close()
    finally:
        controller.close()
