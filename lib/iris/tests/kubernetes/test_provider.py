# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider: sync lifecycle, logs, capacity, scheduling, profiling."""

from __future__ import annotations

import subprocess

import pytest

from iris.cluster.controller.transitions import ClusterCapacity, RunningTaskEntry, SchedulingEvent
from iris.cluster.k8s.provider import (
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _LABEL_TASK_HASH,
    _POD_NOT_FOUND_GRACE_CYCLES,
    _RUNTIME_LABEL_VALUE,
    KubernetesProvider,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
)
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2

from .conftest import make_batch, make_run_req, populate_node, populate_pod, populate_running_pod_resource

# ---------------------------------------------------------------------------
# sync(): tasks_to_run
# ---------------------------------------------------------------------------


def test_sync_applies_pods_for_tasks_to_run(provider, k8s):
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    pods = k8s.list_json("pods", labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})
    assert len(pods) == 1
    assert pods[0]["kind"] == "Pod"
    assert result.updates == []


def test_sync_propagates_kubectl_failure(provider, k8s):
    k8s.inject_failure("apply_json", RuntimeError("kubectl down"))
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    with pytest.raises(RuntimeError, match="kubectl down"):
        provider.sync(batch)


# ---------------------------------------------------------------------------
# sync(): tasks_to_kill
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_for_tasks_to_kill(provider, k8s):
    task_id = "/test-job/0"
    populate_pod(
        k8s,
        "iris-test-job-0-0",
        "Running",
        labels={_LABEL_TASK_HASH: _task_hash(task_id)},
    )
    batch = make_batch(tasks_to_kill=[task_id])

    result = provider.sync(batch)

    assert k8s.get_json("pod", "iris-test-job-0-0") is None
    assert result.updates == []


def test_delete_pods_uses_task_hash_label(provider, k8s):
    """_delete_pods_by_task_id must filter by _LABEL_TASK_HASH, not sanitized task_id."""
    task_id = "/test-job/0"
    task_hash = _task_hash(task_id)

    populate_pod(k8s, "iris-test-pod", "Running", labels={_LABEL_TASK_HASH: task_hash})
    populate_pod(k8s, "iris-other-pod", "Running", labels={_LABEL_TASK_HASH: "wrong-hash"})

    provider._delete_pods_by_task_id(task_id)

    assert k8s.get_json("pod", "iris-test-pod") is None
    assert k8s.get_json("pod", "iris-other-pod") is not None


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

    assert k8s.get_json("pod", "pod-a") is None
    assert k8s.get_json("pod", "pod-b") is not None


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
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING


def test_sync_pod_not_found_marks_failed(provider, k8s):
    """Pod must be missing for _POD_NOT_FOUND_GRACE_CYCLES consecutive syncs before FAILED."""
    task_id = JobName.from_wire("/job/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    batch = make_batch(running_tasks=[entry])

    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert len(result.updates) == 1
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED
    assert result.updates[0].error == "Pod not found"


def test_pod_not_found_grace_period(provider, k8s):
    """A single missing-pod sync returns RUNNING, not FAILED."""
    task_id = JobName.from_wire("/job/grace")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING


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
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Pod reappears — counter should reset.
    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, None)
    result = provider.sync(batch)
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Now disappear again: need full grace cycles again before failure.
    k8s.delete("pod", pod_name)
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED


def test_sync_succeeded_pod_fetches_logs(provider, k8s):
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Succeeded")
    k8s.set_logs(pod_name, "task complete\n")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert len(result.updates[0].log_entries) >= 1
    assert any(e.data == "task complete" for e in result.updates[0].log_entries)


def test_sync_empty_batch(provider):
    batch = make_batch()
    result = provider.sync(batch)
    assert result.updates == []


# ---------------------------------------------------------------------------
# fetch_live_logs
# ---------------------------------------------------------------------------


def test_fetch_live_logs_returns_entries_with_cursor(provider, k8s):
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    k8s.set_logs(pod_name, "line 1\nline 2\n")

    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=10)

    assert len(entries) == 2
    assert entries[0].data == "line 1"
    assert next_cursor > 0


def test_fetch_live_logs_respects_max_lines(provider, k8s):
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    k8s.set_logs(pod_name, "\n".join(f"line {i}" for i in range(10)) + "\n")

    entries, _ = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=3)
    assert len(entries) == 3


def test_fetch_live_logs_falls_back_to_previous_when_empty(provider, k8s):
    """When stream_logs returns nothing, fall back to kubectl.logs(previous=True)."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    k8s.set_logs(pod_name, "line a\nline b\nline c\n")

    entries, _next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=10)
    assert len(entries) >= 1
    assert any(e.data == "line a" for e in entries)


def test_fetch_live_logs_fallback_replays_all_with_nonzero_cursor(provider, k8s):
    """With a nonzero cursor beyond log length, fallback replays from logs(previous=True)."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    k8s.set_logs(pod_name, "line a\nline b\nline c\n")

    # Cursor beyond log bytes: stream_logs returns empty, falls back to logs(previous=True)
    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=1024, max_lines=100)

    assert len(entries) == 3
    assert entries[0].data == "line a"
    assert next_cursor == 3


# ---------------------------------------------------------------------------
# Incremental log polling
# ---------------------------------------------------------------------------


def test_poll_fetches_incremental_logs_for_running_pods(provider, k8s):
    """Running pods get incremental logs via stream_logs each sync cycle."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_logs(pod_name, "hello from running pod\n")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "hello from running pod"


def test_log_cursors_advance_across_sync_cycles(provider, k8s):
    """Byte offset from stream_logs advances: second sync returns no new logs if unchanged."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_logs(pod_name, "line 1\n")

    # First sync: should get "line 1"
    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "line 1"

    # Second sync with same logs: cursor advanced, so no new entries
    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates[0].log_entries) == 0

    # Append new content: should get "line 2" on next sync
    k8s.set_logs(pod_name, "line 1\nline 2\n")
    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "line 2"


def test_final_log_fetch_on_pod_completion(provider, k8s):
    """Completed pods get a final full-log fetch; longer result replaces incremental logs."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Succeeded")
    k8s.set_logs(pod_name, "line 1\nline 2\nline 3\n")

    result = provider.sync(make_batch(running_tasks=[entry]))

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert len(result.updates[0].log_entries) == 3
    assert result.updates[0].log_entries[0].data == "line 1"


# ---------------------------------------------------------------------------
# _query_capacity
# ---------------------------------------------------------------------------


def test_query_capacity_returns_cluster_capacity(provider, k8s):
    """Capacity reports total and available resources as ClusterCapacity."""
    populate_node(k8s, "node-1", cpu="4", memory="8Gi")
    populate_node(k8s, "node-2", cpu="4", memory="8Gi")
    populate_running_pod_resource(k8s, "running-pod-1", cpu_limits="1000m", memory_limits=str(2 * 1024**3))

    cap = provider._query_capacity()
    assert cap is not None
    assert isinstance(cap, ClusterCapacity)
    assert cap.schedulable_nodes == 2
    assert cap.total_cpu_millicores == 8000
    assert cap.total_memory_bytes == 2 * 8 * 1024**3
    assert cap.available_cpu_millicores == 7000
    assert cap.available_memory_bytes == (2 * 8 - 2) * 1024**3


def test_query_capacity_skips_tainted_nodes(provider, k8s):
    populate_node(
        k8s,
        "tainted-node",
        cpu="8",
        memory="16Gi",
        taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}],
    )
    populate_node(k8s, "clean-node", cpu="4", memory="8Gi")

    cap = provider._query_capacity()
    assert cap is not None
    assert cap.schedulable_nodes == 1
    assert cap.total_memory_bytes == 8 * 1024**3


def test_query_capacity_returns_none_when_all_tainted(provider, k8s):
    populate_node(k8s, "tainted-only", cpu="4", memory="8Gi", taints=[{"effect": "NoSchedule"}])

    cap = provider._query_capacity()
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
    k8s._resources[("event", "evt-1")] = event

    events = provider._fetch_scheduling_events()
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
    k8s._resources[("event", "evt-non-iris")] = event

    events = provider._fetch_scheduling_events()
    assert events == []


def test_fetch_scheduling_events_returns_empty_on_failure(provider, k8s):
    k8s.inject_failure("list_json", RuntimeError("events API unavailable"))
    events = provider._fetch_scheduling_events()
    assert events == []


# ---------------------------------------------------------------------------
# get_cluster_status
# ---------------------------------------------------------------------------


def test_get_cluster_status_basic(k8s):
    """get_cluster_status returns namespace, node counts, and pod statuses."""
    populate_node(k8s, "node-1", cpu="4", memory="8Gi")
    node_tainted = {
        "kind": "Node",
        "metadata": {"name": "node-2"},
        "spec": {"taints": [{"effect": "NoSchedule", "key": "k"}]},
        "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
    }
    k8s._resources[("node", "node-2")] = node_tainted

    populate_pod(
        k8s,
        "iris-task-0",
        "Running",
        labels={
            "iris.task_id": "job-0",
            "iris.attempt_id": "0",
        },
    )
    pod = k8s._resources[("pod", "iris-task-0")]
    pod["status"]["conditions"] = []

    provider = KubernetesProvider(kubectl=k8s, namespace="iris", default_image="img:latest")
    resp = provider.get_cluster_status()

    assert resp.namespace == "iris"
    assert resp.total_nodes == 2
    assert resp.schedulable_nodes == 1
    assert "cores" in resp.allocatable_cpu
    assert "GiB" in resp.allocatable_memory
    assert len(resp.pod_statuses) == 1
    assert resp.pod_statuses[0].pod_name == "iris-task-0"
    assert resp.pod_statuses[0].phase == "Running"


def test_get_cluster_status_node_failure(k8s):
    """get_cluster_status handles node query failure gracefully."""
    k8s.inject_failure("list_json", RuntimeError("kubectl error"))
    provider = KubernetesProvider(kubectl=k8s, namespace="test-ns", default_image="img:latest")
    resp = provider.get_cluster_status()

    assert resp.namespace == "test-ns"
    assert resp.total_nodes == 0
    assert resp.schedulable_nodes == 0


# ---------------------------------------------------------------------------
# Resource stats from kubectl top
# ---------------------------------------------------------------------------


def test_resource_stats_from_kubectl_top(provider, k8s):
    """Running pods get resource_usage populated from kubectl top pod."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, (500, 1024 * 1024 * 1024))

    batch = make_batch(running_tasks=[entry])
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


def _success_cp(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _failure_cp(stderr: str = "", stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr=stderr)


def test_profile_threads_via_kubectl_exec(provider, k8s):
    """profile_task with threads type calls py-spy dump via kubectl exec."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _success_cp(stdout="Thread 0x7f00 (idle)\n  main.py:42"))

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            threads=cluster_pb2.ThreadsProfile(locals=False),
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

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            threads=cluster_pb2.ThreadsProfile(locals=True),
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

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=3,
        profile_type=cluster_pb2.ProfileType(
            cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH),
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

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            memory=cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH),
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

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            memory=cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    assert b"ALLOC" in resp.profile_data
    assert len(k8s._rm_files_calls) >= 1


def test_profile_unknown_type_returns_error(provider, k8s):
    """An empty ProfileType (no profiler selected) returns an error."""
    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert resp.error == "Unknown profile type"
    assert not resp.profile_data


def test_profile_kubectl_exec_failure_returns_error(provider, k8s):
    """When kubectl exec fails, the error is captured in the response."""
    pod_name = _pod_name(JobName.from_wire("/job/0"), 0)
    populate_pod(k8s, pod_name, "Running")
    k8s.set_exec_response(pod_name, _failure_cp(stderr="container not running"))

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            threads=cluster_pb2.ThreadsProfile(),
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

    configmaps = k8s.list_json("configmap")
    pods = k8s.list_json("pod")
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

    configmaps = k8s.list_json("configmap")
    pods = k8s.list_json("pod")
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
    k8s._resources[("configmap", "iris-pod-1-wf")] = cm

    provider._delete_pods_by_task_id(task_id)

    assert k8s.get_json("pod", "iris-pod-1") is None
    assert k8s.get_json("configmap", "iris-pod-1-wf") is None
