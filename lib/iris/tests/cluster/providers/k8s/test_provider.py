# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8sTaskProvider: sync lifecycle, logs, capacity, scheduling, profiling."""

from __future__ import annotations

import time

import pytest
from finelog.rpc import logging_pb2
from finelog.server import LogServiceImpl
from iris.cluster.controller.transitions import ClusterCapacity, RunningTaskEntry, SchedulingEvent
from iris.cluster.log_store_helpers import task_log_key
from iris.cluster.providers.k8s.tasks import (
    _GC_MAX_AGE_SECONDS,
    _LABEL_JOB_ID,
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _LABEL_TASK_HASH,
    _MANAGED_POD_LABELS,
    _POD_NOT_FOUND_GRACE_CYCLES,
    _RUNTIME_LABEL_VALUE,
    K8sTaskProvider,
    _LogPod,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
)
from iris.cluster.providers.k8s.types import ExecResult, K8sResource, PodResourceUsage
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import job_pb2

from .conftest import make_batch, make_run_req, populate_node, populate_pod, populate_running_pod_resource


def _fetch_logs(log_service: LogServiceImpl, key: str, max_lines: int = 100) -> list[logging_pb2.LogEntry]:
    resp = log_service.fetch_logs(logging_pb2.FetchLogsRequest(source=key, max_lines=max_lines), ctx=None)
    return list(resp.entries)


# ---------------------------------------------------------------------------
# sync(): tasks_to_run
# ---------------------------------------------------------------------------


def test_sync_applies_pods_for_tasks_to_run(provider, k8s):
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    pods = k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})
    assert len(pods) == 1
    assert pods[0]["kind"] == "Pod"
    assert result.updates == []


def test_sync_propagates_non_kubectl_failure(provider, k8s):
    k8s.inject_failure("apply_json", RuntimeError("kubectl down"))
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    with pytest.raises(RuntimeError, match="kubectl down"):
        provider.sync(batch)


def test_sync_catches_kubectl_error_and_returns_task_failure(provider, k8s):
    from iris.cluster.providers.k8s.types import KubectlError

    k8s.inject_failure(
        "apply_json",
        KubectlError("kubectl apply failed: Error from server (RequestEntityTooLarge): limit is 3145728"),
    )
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    assert len(result.updates) == 1
    update = result.updates[0]
    assert update.new_state == job_pb2.TASK_STATE_FAILED
    assert "RequestEntityTooLarge" in update.error


# ---------------------------------------------------------------------------
# sync(): tasks_to_kill
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_for_tasks_to_kill(provider, k8s):
    task_id = "/test-job/0"
    populate_pod(
        k8s,
        "iris-test-job-0-0",
        "Running",
        labels={_LABEL_TASK_HASH: _task_hash(task_id), _LABEL_JOB_ID: _sanitize_label_value("/test-job")},
    )
    batch = make_batch(tasks_to_kill=[task_id])

    result = provider.sync(batch)

    assert k8s.get_json(K8sResource.PODS, "iris-test-job-0-0") is None
    assert result.updates == []


def test_delete_pods_uses_task_hash_label(provider, k8s):
    """_delete_pods_by_task_id must filter by _LABEL_TASK_HASH, not sanitized task_id."""
    task_id = "/test-job/0"
    task_hash = _task_hash(task_id)

    populate_pod(k8s, "iris-test-pod", "Running", labels={_LABEL_TASK_HASH: task_hash})
    populate_pod(k8s, "iris-other-pod", "Running", labels={_LABEL_TASK_HASH: "wrong-hash"})

    provider._delete_pods_by_task_id(task_id)

    assert k8s.get_json(K8sResource.PODS, "iris-test-pod") is None
    assert k8s.get_json(K8sResource.PODS, "iris-other-pod") is not None


def test_delete_pods_does_not_delete_colliding_task(provider, k8s):
    """Two task IDs with the same sanitized label must not share hash-based pod deletion."""
    base = "a" * 63
    task_id_a = base + "X"
    task_id_b = base + "Y"
    assert _sanitize_label_value(task_id_a) == _sanitize_label_value(task_id_b)

    hash_a = _task_hash(task_id_a)
    hash_b = _task_hash(task_id_b)
    assert hash_a != hash_b, "distinct task IDs must use distinct hash labels for deletion"

    populate_pod(k8s, "pod-a", "Running", labels={_LABEL_TASK_HASH: hash_a})
    populate_pod(k8s, "pod-b", "Running", labels={_LABEL_TASK_HASH: hash_b})

    provider._delete_pods_by_task_id(task_id_a)

    assert k8s.get_json(K8sResource.PODS, "pod-a") is None
    assert k8s.get_json(K8sResource.PODS, "pod-b") is not None


# ---------------------------------------------------------------------------
# sync(): running_tasks polling
# ---------------------------------------------------------------------------


def test_sync_running_task_returns_running_state(provider, k8s):
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING


def test_sync_pod_not_found_marks_failed(provider, k8s):
    """Pod must be missing for _POD_NOT_FOUND_GRACE_CYCLES consecutive syncs before FAILED."""
    task_id = JobName.from_wire("/job/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    batch = make_batch(running_tasks=[entry])

    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert len(result.updates) == 1
        assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert len(result.updates) == 1
    assert result.updates[0].new_state == job_pb2.TASK_STATE_FAILED
    assert result.updates[0].error == "Pod not found"


def test_pod_not_found_grace_period(provider, k8s):
    """A single missing-pod sync returns RUNNING, not FAILED."""
    task_id = JobName.from_wire("/job/grace")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates) == 1
    assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING


def test_pod_not_found_grace_resets_when_pod_reappears(provider, k8s):
    """If the pod reappears after a transient miss, the grace counter resets."""
    task_id = JobName.from_wire("/job/reset")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)
    batch = make_batch(running_tasks=[entry])

    # Miss for (grace - 1) cycles.
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING

    # Pod reappears — counter should reset.
    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, None)
    result = provider.sync(batch)
    assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING

    # Now disappear again: need full grace cycles again before failure.
    k8s.delete(K8sResource.PODS, pod_name)
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert result.updates[0].new_state == job_pb2.TASK_STATE_FAILED


def test_sync_succeeded_pod_fetches_logs(provider, k8s, log_service: LogServiceImpl):
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Succeeded")
    k8s.set_logs(pod_name, "task complete\n")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert result.updates[0].new_state == job_pb2.TASK_STATE_SUCCEEDED
    # Logs are pushed through the LogService via LogCollector (set_pods removal does a final fetch).
    key = task_log_key(TaskAttempt(task_id=task_id, attempt_id=attempt_id))
    logs = _fetch_logs(log_service, key)
    assert any(e.data == "task complete" for e in logs)


def test_sync_empty_batch(provider):
    batch = make_batch()
    result = provider.sync(batch)
    assert result.updates == []


# ---------------------------------------------------------------------------
# Incremental log polling
# ---------------------------------------------------------------------------


def test_poll_fetches_incremental_logs_for_running_pods(provider, k8s, log_service: LogServiceImpl):
    """Running pods get incremental logs via the background LogCollector."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_logs(pod_name, "hello from running pod\n")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == job_pb2.TASK_STATE_RUNNING
    # Logs are collected by the background LogCollector thread.
    # Give it time to run one cycle.
    time.sleep(3)
    key = task_log_key(TaskAttempt(task_id=task_id, attempt_id=attempt_id))
    logs = _fetch_logs(log_service, key)
    assert len(logs) >= 1
    assert logs[0].data == "hello from running pod"


def test_log_cursors_advance_across_sync_cycles(provider, k8s, log_service: LogServiceImpl):
    """LogCollector advances byte offsets: repeated fetches don't duplicate."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_logs(pod_name, "line 1\n")

    # First sync: LogCollector starts tracking the pod.
    provider.sync(make_batch(running_tasks=[entry]))
    time.sleep(3)
    key = task_log_key(TaskAttempt(task_id=task_id, attempt_id=attempt_id))
    logs = _fetch_logs(log_service, key)
    assert len(logs) == 1
    assert logs[0].data == "line 1"

    # Append new content and let collector run again.
    k8s.set_logs(pod_name, "line 1\nline 2\n")
    provider.sync(make_batch(running_tasks=[entry]))
    time.sleep(3)
    logs = _fetch_logs(log_service, key)
    assert len(logs) == 2
    assert logs[1].data == "line 2"


def test_final_log_fetch_on_pod_completion(provider, k8s, log_service: LogServiceImpl):
    """Completed pods get a final log fetch when removed from the collector's tracked set."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Succeeded")
    k8s.set_logs(pod_name, "line 1\nline 2\nline 3\n")

    result = provider.sync(make_batch(running_tasks=[entry]))

    assert result.updates[0].new_state == job_pb2.TASK_STATE_SUCCEEDED
    # set_pods() removal does a synchronous final fetch — logs should be in the service.
    key = task_log_key(TaskAttempt(task_id=task_id, attempt_id=attempt_id))
    logs = _fetch_logs(log_service, key)
    assert len(logs) == 3
    assert logs[0].data == "line 1"


# ---------------------------------------------------------------------------
# ClusterState.capacity (via sync)
# ---------------------------------------------------------------------------


def test_capacity_returns_cluster_capacity(provider, k8s):
    """Capacity reports total and available resources as ClusterCapacity."""
    populate_node(k8s, "node-1", cpu="4", memory="8Gi")
    populate_node(k8s, "node-2", cpu="4", memory="8Gi")
    populate_running_pod_resource(k8s, "running-pod-1", cpu_limits="1000m", memory_limits=str(2 * 1024**3))

    provider.sync(make_batch())
    cap = provider._cluster_state.capacity()

    assert cap is not None
    assert isinstance(cap, ClusterCapacity)
    assert cap.schedulable_nodes == 2
    assert cap.total_cpu_millicores == 8000
    assert cap.total_memory_bytes == 2 * 8 * 1024**3
    assert cap.available_cpu_millicores == 7000
    assert cap.available_memory_bytes == (2 * 8 - 2) * 1024**3


def test_capacity_skips_tainted_nodes(provider, k8s):
    populate_node(
        k8s,
        "tainted-node",
        cpu="8",
        memory="16Gi",
        taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}],
    )
    populate_node(k8s, "clean-node", cpu="4", memory="8Gi")

    provider.sync(make_batch())
    cap = provider._cluster_state.capacity()

    assert cap is not None
    assert cap.schedulable_nodes == 1
    assert cap.total_memory_bytes == 8 * 1024**3


def test_capacity_returns_none_when_all_tainted(provider, k8s):
    populate_node(k8s, "tainted-only", cpu="4", memory="8Gi", taints=[{"effect": "NoSchedule"}])

    provider.sync(make_batch())
    cap = provider._cluster_state.capacity()

    assert cap is None


# ---------------------------------------------------------------------------
# _fetch_scheduling_events
# ---------------------------------------------------------------------------


def test_fetch_scheduling_events_returns_events(provider, k8s):
    pod_name = _pod_name(JobName.from_wire("/test-job/0"), 1)
    populate_pod(
        k8s,
        pod_name,
        "Pending",
        labels={
            "iris.task_id": "test-job.0",
            "iris.attempt_id": "1",
        },
    )

    event = {
        "kind": "Event",
        "metadata": {"name": "evt-1"},
        "involvedObject": {"kind": "Pod", "name": pod_name},
        "type": "Warning",
        "reason": "FailedScheduling",
        "message": "0/3 nodes available",
    }
    k8s.seed_resource(K8sResource.EVENTS, "evt-1", event)

    events = provider._fetch_scheduling_events(k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS))
    assert len(events) == 1
    assert isinstance(events[0], SchedulingEvent)
    assert events[0].task_id == "test-job.0"
    assert events[0].attempt_id == 1
    assert events[0].reason == "FailedScheduling"


def test_fetch_scheduling_events_ignores_non_iris_events(provider, k8s):
    event = {
        "kind": "Event",
        "metadata": {"name": "evt-non-iris"},
        "involvedObject": {"kind": "Pod", "name": "some-other-pod"},
        "type": "Warning",
        "reason": "FailedScheduling",
        "message": "0/3 nodes available",
    }
    k8s.seed_resource(K8sResource.EVENTS, "evt-non-iris", event)

    events = provider._fetch_scheduling_events(k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS))
    assert events == []


def test_fetch_scheduling_events_returns_empty_on_failure(provider, k8s):
    """Events fetch failure returns empty list (pods are pre-cached, only events call can fail)."""
    # Seed a pod so pod_names is non-empty, then fail the events list call.
    populate_pod(k8s, "iris-test-0", "Running")
    cached_pods = k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
    k8s.inject_failure("list_json", RuntimeError("events API unavailable"))
    events = provider._fetch_scheduling_events(cached_pods)
    assert events == []


# ---------------------------------------------------------------------------
# get_cluster_status
# ---------------------------------------------------------------------------


def test_get_cluster_status_basic(k8s):
    """get_cluster_status returns namespace, node counts, and pod statuses after sync."""
    populate_node(k8s, "node-1", cpu="4", memory="8Gi")
    node_tainted = {
        "kind": "Node",
        "metadata": {"name": "node-2"},
        "spec": {"taints": [{"effect": "NoSchedule", "key": "k"}]},
        "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
    }
    k8s.seed_resource(K8sResource.NODES, "node-2", node_tainted)

    populate_pod(
        k8s,
        "iris-task-0",
        "Running",
        labels={
            "iris.task_id": "job-0",
            "iris.attempt_id": "0",
        },
    )
    pod = k8s.get_json(K8sResource.PODS, "iris-task-0")
    pod["status"]["conditions"] = []

    p = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest")
    try:
        p.sync(make_batch())
        resp = p.get_cluster_status()

        assert resp.namespace == "iris"
        assert resp.total_nodes == 2
        assert resp.schedulable_nodes == 1
        assert "cores" in resp.allocatable_cpu
        assert "GiB" in resp.allocatable_memory
        assert len(resp.pod_statuses) == 1
        assert resp.pod_statuses[0].pod_name == "iris-task-0"
        assert resp.pod_statuses[0].phase == "Running"
    finally:
        p.close()


def test_get_cluster_status_node_failure(k8s):
    """Node list failure during sync is handled gracefully; status reports 0 nodes."""
    k8s.inject_failure("list_json:node", RuntimeError("kubectl error"))
    p = K8sTaskProvider(kubectl=k8s, namespace="test-ns", default_image="img:latest")
    try:
        p.sync(make_batch())
        resp = p.get_cluster_status()
        assert resp.namespace == "test-ns"
        assert resp.total_nodes == 0
        assert resp.schedulable_nodes == 0
    finally:
        p.close()


def test_get_cluster_status_excludes_terminal_pods(k8s):
    """After sync, only active pods appear; Succeeded/Failed are excluded by the field selector."""
    populate_node(k8s, "node-1", cpu="4", memory="8Gi")
    populate_pod(k8s, "iris-running", "Running")
    populate_pod(k8s, "iris-succeeded", "Succeeded")
    populate_pod(k8s, "iris-failed", "Failed")

    p = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest")
    try:
        p.sync(make_batch())
        resp = p.get_cluster_status()

        phases = {ps.pod_name: ps.phase for ps in resp.pod_statuses}
        assert "iris-running" in phases
        assert "iris-succeeded" not in phases
        assert "iris-failed" not in phases
    finally:
        p.close()


def test_get_cluster_status_uses_sync_cache(provider, k8s):
    """After sync(), pod data is served from cache even if the pod is deleted from k8s."""
    populate_pod(k8s, "iris-task-0", "Running")

    provider.sync(make_batch())

    # Delete the pod from the fake k8s store. A fresh kubectl call would return 0 pods.
    k8s.delete(K8sResource.PODS, "iris-task-0")

    resp = provider.get_cluster_status()

    # Pod statuses reflect the sync() cache (pod still visible), not a fresh kubectl call.
    assert len(resp.pod_statuses) == 1
    assert resp.pod_statuses[0].pod_name == "iris-task-0"


def test_sync_cache_excludes_terminal_pods(provider, k8s):
    """sync() caches only active pods; get_cluster_status reflects the field-selector filter."""
    # sync() uses _ACTIVE_PODS_FIELD_SELECTOR which excludes Succeeded/Failed.
    populate_pod(k8s, "iris-running", "Running")
    populate_pod(k8s, "iris-succeeded", "Succeeded")

    batch = make_batch()
    provider.sync(batch)

    resp = provider.get_cluster_status()
    phases = {ps.pod_name: ps.phase for ps in resp.pod_statuses}
    assert "iris-running" in phases
    assert "iris-succeeded" not in phases


def test_get_cluster_status_includes_node_pools(provider, k8s):
    """Node pools fetched during sync() are included in get_cluster_status() response."""
    k8s.seed_resource(
        K8sResource.NODE_POOLS,
        "gpu-pool",
        {
            "kind": "NodePool",
            "metadata": {"name": "gpu-pool", "labels": {}},
            "spec": {"instanceType": "H100", "targetNodes": 4},
            "status": {"currentNodes": 3},
        },
    )
    provider.sync(make_batch())
    resp = provider.get_cluster_status()
    assert any(np.name == "gpu-pool" for np in resp.node_pools)


def test_sync_node_failure_yields_no_capacity(provider, k8s):
    """When node list fails during sync, capacity is None but sync still returns."""
    populate_pod(k8s, "iris-running", "Running")
    k8s.inject_failure("list_json:node", RuntimeError("nodes unavailable"))

    result = provider.sync(make_batch())

    assert result.capacity is None
    # Pod statuses are still populated from the successful pod list.
    resp = provider.get_cluster_status()
    assert any(ps.pod_name == "iris-running" for ps in resp.pod_statuses)


# ---------------------------------------------------------------------------
# Resource stats from kubectl top
# ---------------------------------------------------------------------------


def test_resource_stats_from_kubectl_top(provider, k8s):
    """Running pods get resource_usage populated by background ResourceCollector."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, PodResourceUsage(cpu_millicores=500, memory_bytes=1024 * 1024 * 1024))

    batch = make_batch(running_tasks=[entry])
    # First sync registers the pod with the ResourceCollector.
    provider.sync(batch)
    # Wait for background collector to fetch.
    time.sleep(6)
    # Second sync reads the cached resource usage.
    result = provider.sync(batch)

    assert len(result.updates) == 1
    usage = result.updates[0].resource_usage
    assert usage is not None
    assert usage.cpu_millicores == 500
    assert usage.memory_mb == 1024


def test_resource_stats_none_when_metrics_unavailable(provider, k8s):
    """resource_usage stays None when kubectl top returns None."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, None)

    batch = make_batch(running_tasks=[entry])
    # Even after background collector runs, None top_pod stays None.
    provider.sync(batch)
    time.sleep(6)
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_none_when_top_pod_raises(provider, k8s):
    """resource_usage stays None when kubectl top raises an exception."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.inject_failure("top_pod", RuntimeError("metrics-server unavailable"))

    batch = make_batch(running_tasks=[entry])
    provider.sync(batch)
    time.sleep(6)
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_none_for_non_running_pods(provider, k8s):
    """resource_usage is None for pods in terminal phases."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Succeeded")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


# ---------------------------------------------------------------------------
# Profiling via kubectl exec
# ---------------------------------------------------------------------------


def _success_cp(stdout: str = "", stderr: str = "") -> ExecResult:
    return ExecResult(returncode=0, stdout=stdout, stderr=stderr)


def _failure_cp(stderr: str = "", stdout: str = "") -> ExecResult:
    return ExecResult(returncode=1, stdout=stdout, stderr=stderr)


def test_profile_threads_via_kubectl_exec(provider, k8s):
    """profile_task with threads type calls py-spy dump via kubectl exec."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _success_cp(stdout="Thread 0x7f00 (idle)\n  main.py:42"))

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            threads=job_pb2.ThreadsProfile(locals=False),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    assert b"Thread 0x7f00" in resp.profile_data


def test_profile_threads_with_locals(provider, k8s):
    """profile_task with threads.locals=True passes --locals to py-spy dump."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _success_cp(stdout="Thread 0x7f00\n  x = 42"))

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            threads=job_pb2.ThreadsProfile(locals=True),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    assert b"Thread 0x7f00" in resp.profile_data


def test_profile_cpu_via_kubectl_exec(provider, k8s):
    """profile_task with cpu type calls py-spy record, reads file, cleans up."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 1)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _success_cp())
    k8s.set_file_content(pod_name, "/tmp/iris-profile.svg", b"<svg>flamegraph</svg>")

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=3,
        profile_type=job_pb2.ProfileType(
            cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH),
        ),
    )
    resp = provider.profile_task("/job/0", 1, request)

    assert not resp.error
    assert resp.profile_data == b"<svg>flamegraph</svg>"
    assert len(k8s._rm_files_calls) == 1


def test_profile_memory_flamegraph_via_kubectl_exec(provider, k8s):
    """profile_task with memory flamegraph attaches memray, transforms, reads file."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    # Two exec calls: attach + transform
    k8s.set_exec_response(pod_name, _success_cp())
    k8s.set_exec_response(pod_name, _success_cp())
    k8s.set_file_content(pod_name, "/tmp/iris-memray.html", b"<html>flamegraph</html>")

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    assert resp.profile_data == b"<html>flamegraph</html>"
    assert len(k8s._rm_files_calls) == 1


def test_profile_memory_table_returns_stdout(provider, k8s):
    """Memory table format returns stdout instead of reading a file."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _success_cp())  # attach
    k8s.set_exec_response(pod_name, _success_cp(stdout="ALLOC  SIZE  FILE\n100  1KB  main.py"))  # table transform

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.TABLE),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    assert b"ALLOC" in resp.profile_data
    assert len(k8s._rm_files_calls) >= 1


def test_profile_unknown_type_returns_error(provider, k8s):
    """An empty ProfileType (no profiler selected) returns an error."""
    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert resp.error == "Unknown profile type"
    assert not resp.profile_data


def test_profile_kubectl_exec_failure_returns_error(provider, k8s):
    """When kubectl exec fails, the error is captured in the response."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _failure_cp(stderr="container not running"))

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            threads=job_pb2.ThreadsProfile(),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert resp.error
    assert "container not running" in resp.error


# ---------------------------------------------------------------------------
# ConfigMap lifecycle for workdir files
# ---------------------------------------------------------------------------


def test_configmap_created_for_workdir_files(provider, k8s):
    """_apply_pod creates a ConfigMap when workdir_files are present."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["script.py"] = b"print('hello')"

    provider._apply_pod(req)

    configmaps = k8s.list_json(K8sResource.CONFIGMAPS)
    pods = k8s.list_json(K8sResource.PODS)
    assert len(configmaps) == 1
    assert configmaps[0]["kind"] == "ConfigMap"
    assert configmaps[0]["metadata"]["namespace"] == "iris"
    assert _LABEL_MANAGED in configmaps[0]["metadata"]["labels"]
    assert "f0000" in configmaps[0]["binaryData"]

    assert len(pods) == 1
    assert pods[0]["kind"] == "Pod"
    assert "initContainers" in pods[0]["spec"]


def test_no_configmap_when_no_workdir_files(provider, k8s):
    """_apply_pod does not create a ConfigMap when no workdir_files are set."""
    req = make_run_req("/my-job/task-0")

    provider._apply_pod(req)

    configmaps = k8s.list_json(K8sResource.CONFIGMAPS)
    pods = k8s.list_json(K8sResource.PODS)
    assert len(configmaps) == 0
    assert len(pods) == 1
    assert pods[0]["kind"] == "Pod"


def test_configmap_cleaned_up_on_delete(provider, k8s):
    """_delete_pods_by_task_id also deletes associated ConfigMaps."""
    task_id = "/my-job/task-0"
    task_hash = _task_hash(task_id)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: task_hash,
    }

    populate_pod(k8s, "iris-pod-1", "Running", labels={_LABEL_TASK_HASH: task_hash})
    cm = {
        "kind": "ConfigMap",
        "metadata": {"name": "iris-pod-1-wf", "labels": labels},
    }
    k8s.seed_resource(K8sResource.CONFIGMAPS, "iris-pod-1-wf", cm)

    provider._delete_pods_by_task_id(task_id)

    assert k8s.get_json(K8sResource.PODS, "iris-pod-1") is None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-pod-1-wf") is None


# ---------------------------------------------------------------------------
# PodDisruptionBudget for coordinator tasks
# ---------------------------------------------------------------------------


def test_sync_creates_pdb_for_coordinator_task(provider, k8s):
    """Coordinator tasks (single-task, no accelerator) get a PDB."""
    req = make_run_req("/coord-job/0")
    req.num_tasks = 1
    batch = make_batch(tasks_to_run=[req])

    provider.sync(batch)

    pdbs = k8s.list_json(K8sResource.PDBS)
    assert len(pdbs) == 1
    pdb = pdbs[0]
    assert pdb["spec"]["minAvailable"] == 1
    assert pdb["metadata"]["labels"][_LABEL_TASK_HASH] == _task_hash("/coord-job/0")


def test_bulk_delete_defers_pdb_cleanup_to_gc(provider, k8s):
    """_bulk_delete_task_pods deletes pods immediately but defers PDB/CM cleanup to GC."""
    task_id = "/coord-job/0"
    task_hash = _task_hash(task_id)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: task_hash,
    }

    populate_pod(k8s, "iris-coord-pod", "Running", labels={_LABEL_TASK_HASH: task_hash})
    pdb = {
        "kind": "PodDisruptionBudget",
        "metadata": {"name": "iris-coord-pod-pdb", "labels": labels},
        "spec": {"minAvailable": 1},
    }
    k8s.seed_resource(K8sResource.PDBS, "iris-coord-pod-pdb", pdb)

    cached_pods = k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})
    provider._bulk_delete_task_pods([task_id], cached_pods)

    # Pod deleted immediately.
    assert k8s.get_json(K8sResource.PODS, "iris-coord-pod") is None
    # PDB still exists — deferred to GC.
    assert k8s.get_json(K8sResource.PDBS, "iris-coord-pod-pdb") is not None

    # GC pass cleans up the deferred PDB.
    provider._gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.PDBS, "iris-coord-pod-pdb") is None


# ---------------------------------------------------------------------------
# GC: terminal pod and resource cleanup
# ---------------------------------------------------------------------------


def _seed_terminal_pod(k8s, name: str, phase: str, task_hash: str, created: str) -> None:
    """Insert a terminal pod with a creationTimestamp into the fake k8s store."""
    pod = {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "creationTimestamp": created,
            "labels": {
                _LABEL_MANAGED: "true",
                _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
                _LABEL_TASK_HASH: task_hash,
            },
        },
        "status": {"phase": phase},
    }
    k8s.seed_resource(K8sResource.PODS, name, pod)


def _seed_configmap(k8s, name: str, task_hash: str, created: str) -> None:
    cm = {
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "creationTimestamp": created,
            "labels": {
                _LABEL_MANAGED: "true",
                _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
                _LABEL_TASK_HASH: task_hash,
            },
        },
    }
    k8s.seed_resource(K8sResource.CONFIGMAPS, name, cm)


def test_gc_deletes_old_terminal_pods_and_configmaps(provider, k8s):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(seconds=_GC_MAX_AGE_SECONDS + 600)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_ts = (now - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")

    hash_old = "aabbccdd11223344"
    hash_recent = "eeff001122334455"

    # Old succeeded pod + its configmap — should be GC'd.
    _seed_terminal_pod(k8s, "old-succeeded-pod", "Succeeded", hash_old, old_ts)
    _seed_configmap(k8s, "old-succeeded-pod-wf", hash_old, old_ts)

    # Recent succeeded pod + its configmap — should survive.
    _seed_terminal_pod(k8s, "recent-succeeded-pod", "Succeeded", hash_recent, recent_ts)
    _seed_configmap(k8s, "recent-succeeded-pod-wf", hash_recent, recent_ts)

    # Old failed pod — should be GC'd.
    _seed_terminal_pod(k8s, "old-failed-pod", "Failed", "ffaa112233445566", old_ts)

    provider._gc_terminal_resources(active_pods=[])

    # Old resources deleted.
    assert k8s.get_json(K8sResource.PODS, "old-succeeded-pod") is None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "old-succeeded-pod-wf") is None
    assert k8s.get_json(K8sResource.PODS, "old-failed-pod") is None

    # Recent resources preserved.
    assert k8s.get_json(K8sResource.PODS, "recent-succeeded-pod") is not None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "recent-succeeded-pod-wf") is not None


def test_gc_respects_interval(provider, k8s):
    """_maybe_gc_terminal_resources should only run every _GC_INTERVAL_SECONDS."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(seconds=_GC_MAX_AGE_SECONDS + 600)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Trigger GC once to set _last_gc_time to now.
    provider._maybe_gc_terminal_resources(active_pods=[])

    # Seed an old pod. An immediate second call should NOT trigger GC (interval not elapsed).
    _seed_terminal_pod(k8s, "gc-pod-1", "Succeeded", "aaaa111122223333", old_ts)
    provider._maybe_gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.PODS, "gc-pod-1") is not None  # Still exists — interval gate held


def test_gc_cleans_up_deferred_configmaps(provider, k8s):
    """GC deletes configmaps for task hashes enqueued by _bulk_delete_task_pods."""
    task_id = "/deferred-job/0"
    task_hash = _task_hash(task_id)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: task_hash,
    }

    # Seed a configmap (no pod needed — the hash is what matters).
    cm = {
        "kind": "ConfigMap",
        "metadata": {"name": "deferred-cm", "labels": labels},
    }
    k8s.seed_resource(K8sResource.CONFIGMAPS, "deferred-cm", cm)

    # Simulate _bulk_delete_task_pods enqueuing the hash.
    provider._pending_gc_hashes.add(task_hash)

    # GC picks it up and deletes the configmap.
    provider._gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.CONFIGMAPS, "deferred-cm") is None


def test_gc_retains_pending_hash_when_pod_still_in_snapshot(provider, k8s):
    """Deferred hashes must not be dropped when the killed pod is still in the
    pre-delete managed_pods snapshot (the common tasks_to_kill path).

    Reproduces: sync fetches managed_pods, _bulk_delete_task_pods deletes the pod
    and enqueues hash, then _maybe_gc sees the hash as "active" from the stale
    snapshot. The hash must be retained for the next GC cycle.
    """
    task_id = "/kill-me/0"
    task_hash = _task_hash(task_id)
    labels = {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, _LABEL_TASK_HASH: task_hash}

    # Seed the pod and its configmap.
    populate_pod(k8s, "iris-kill-me-0-0", "Running", labels={_LABEL_TASK_HASH: task_hash})
    cm = {"kind": "ConfigMap", "metadata": {"name": "iris-kill-me-0-0-wf", "labels": labels}}
    k8s.seed_resource(K8sResource.CONFIGMAPS, "iris-kill-me-0-0-wf", cm)

    # Snapshot managed pods BEFORE delete (as sync() does).
    pre_delete_pods = k8s.list_json(
        K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE}
    )

    # Kill the pod — hash goes into _pending_gc_hashes.
    provider._bulk_delete_task_pods([task_id], pre_delete_pods)
    assert k8s.get_json(K8sResource.PODS, "iris-kill-me-0-0") is None
    assert task_hash in provider._pending_gc_hashes

    # GC with the stale snapshot — hash should be skipped but NOT discarded.
    provider._gc_terminal_resources(active_pods=pre_delete_pods)
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-kill-me-0-0-wf") is not None  # Not yet cleaned
    assert task_hash in provider._pending_gc_hashes  # Retained for next cycle

    # Next GC cycle with empty active pods — now the CM is cleaned up.
    provider._gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-kill-me-0-0-wf") is None
    assert task_hash not in provider._pending_gc_hashes


def test_gc_skips_hashes_with_active_pods(provider, k8s):
    """GC must not delete configmaps/PDBs for task hashes that have active retry pods.

    task_hash is shared across all attempts of the same task_id. If attempt 0 is
    terminal (old) and attempt 1 is still Running, deleting by task_hash would
    remove the active attempt's configmap and PDB protection.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(seconds=_GC_MAX_AGE_SECONDS + 600)).strftime("%Y-%m-%dT%H:%M:%SZ")

    shared_hash = "shared_hash_12345"

    # Old terminal pod for attempt 0.
    _seed_terminal_pod(k8s, "old-attempt-0", "Succeeded", shared_hash, old_ts)

    # Configmap and PDB for the active retry (attempt 1).
    active_labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: shared_hash,
    }
    cm = {"kind": "ConfigMap", "metadata": {"name": "active-retry-cm", "labels": active_labels}}
    k8s.seed_resource(K8sResource.CONFIGMAPS, "active-retry-cm", cm)
    pdb = {
        "kind": "PodDisruptionBudget",
        "metadata": {"name": "active-retry-pdb", "labels": active_labels},
        "spec": {"minAvailable": 1},
    }
    k8s.seed_resource(K8sResource.PDBS, "active-retry-pdb", pdb)

    # Simulate the active pod (from the sync loop's managed_pods list).
    active_pod = {
        "metadata": {"name": "active-attempt-1", "labels": {_LABEL_TASK_HASH: shared_hash}},
        "status": {"phase": "Running"},
    }

    provider._gc_terminal_resources(active_pods=[active_pod])

    # Terminal pod is deleted (by name, not by hash).
    assert k8s.get_json(K8sResource.PODS, "old-attempt-0") is None
    # But configmap and PDB are preserved because the hash is still active.
    assert k8s.get_json(K8sResource.CONFIGMAPS, "active-retry-cm") is not None
    assert k8s.get_json(K8sResource.PDBS, "active-retry-pdb") is not None


# ---------------------------------------------------------------------------
# Collector set_pods
# ---------------------------------------------------------------------------


def test_log_collector_set_pods_adds_and_removes(k8s, log_client):
    """LogCollector.set_pods() adds new pods and removes absent ones."""
    from iris.cluster.providers.k8s.tasks import LogCollector
    from iris.cluster.types import JobName

    collector = LogCollector(k8s, log_client, concurrency=1)
    task_a = JobName.from_wire("/job/0")
    task_b = JobName.from_wire("/job/1")
    key_a = f"{task_a.to_wire()}:0"
    key_b = f"{task_b.to_wire()}:0"

    collector.set_pods(
        {
            key_a: _LogPod(pod_name="pod-a", task_id=task_a, attempt_id=0),
            key_b: _LogPod(pod_name="pod-b", task_id=task_b, attempt_id=0),
        }
    )
    with collector._lock:
        assert key_a in collector._pods
        assert key_b in collector._pods

    # Remove pod A, keep pod B.
    collector.set_pods(
        {
            key_b: _LogPod(pod_name="pod-b", task_id=task_b, attempt_id=0),
        }
    )
    with collector._lock:
        assert key_a not in collector._pods
        assert key_b in collector._pods

    # Clear all.
    collector.set_pods({})
    with collector._lock:
        assert len(collector._pods) == 0

    collector.close()


def test_log_collector_set_pods_preserves_cursor_state(k8s, log_client):
    """set_pods() preserves last_timestamp for pods that remain tracked."""
    from datetime import datetime, timezone

    from iris.cluster.providers.k8s.tasks import LogCollector
    from iris.cluster.types import JobName

    collector = LogCollector(k8s, log_client, concurrency=1)
    task_id = JobName.from_wire("/job/0")
    key = f"{task_id.to_wire()}:0"

    collector.set_pods(
        {
            key: _LogPod(pod_name="pod-0", task_id=task_id, attempt_id=0),
        }
    )

    # Simulate the collector having advanced the cursor.
    marker = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with collector._lock:
        collector._pods[key].last_timestamp = marker

    # Re-declare the same pod — cursor should be preserved.
    collector.set_pods(
        {
            key: _LogPod(pod_name="pod-0", task_id=task_id, attempt_id=0),
        }
    )
    with collector._lock:
        assert collector._pods[key].last_timestamp == marker

    collector.close()


def test_resource_collector_set_pods_adds_and_removes(k8s):
    """ResourceCollector.set_pods() adds new pods and cleans up removed ones."""
    from iris.cluster.providers.k8s.tasks import ResourceCollector

    collector = ResourceCollector(k8s, concurrency=1)
    key_a = "/job/0:0"
    key_b = "/job/1:0"

    collector.set_pods({key_a: "pod-a", key_b: "pod-b"})
    with collector._lock:
        assert key_a in collector._pods
        assert key_b in collector._pods

    # Inject a cached result for pod A.
    with collector._lock:
        collector._results[key_a] = job_pb2.ResourceUsage(cpu_millicores=100, memory_mb=512)

    # Remove pod A — its cached result should also be cleaned up.
    collector.set_pods({key_b: "pod-b"})
    with collector._lock:
        assert key_a not in collector._pods
        assert key_a not in collector._results
        assert key_b in collector._pods

    collector.close()
