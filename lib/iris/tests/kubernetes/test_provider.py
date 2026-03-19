# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider: sync lifecycle, logs, capacity, scheduling, profiling."""

from datetime import datetime, timezone

import pytest

from iris.cluster.controller.transitions import ClusterCapacity, SchedulingEvent
from iris.cluster.k8s.provider import (
    _LABEL_MANAGED,
    _LABEL_TASK_HASH,
    _POD_NOT_FOUND_GRACE_CYCLES,
    KubernetesProvider,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
)
from iris.cluster.controller.transitions import RunningTaskEntry
from iris.cluster.k8s.kubectl import KubectlLogLine, KubectlLogResult
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2

from .conftest import completed_process, make_batch, make_pod, make_run_req

# ---------------------------------------------------------------------------
# sync(): tasks_to_run
# ---------------------------------------------------------------------------


def test_sync_applies_pods_for_tasks_to_run(provider, mock_kubectl):
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    mock_kubectl.apply_json.assert_called_once()
    manifest = mock_kubectl.apply_json.call_args[0][0]
    assert manifest["kind"] == "Pod"
    assert result.updates == []


def test_sync_propagates_kubectl_failure(provider, mock_kubectl):
    mock_kubectl.apply_json.side_effect = RuntimeError("kubectl down")
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    with pytest.raises(RuntimeError, match="kubectl down"):
        provider.sync(batch)


# ---------------------------------------------------------------------------
# sync(): tasks_to_kill
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_for_tasks_to_kill(provider, mock_kubectl):
    mock_kubectl.list_json.side_effect = [
        [{"metadata": {"name": "iris-test-job-0-0", "labels": {}}}],  # pods
        [],  # configmaps
        [],  # capacity nodes
        [],  # capacity pods
    ]
    batch = make_batch(tasks_to_kill=["/test-job/0"])

    result = provider.sync(batch)

    mock_kubectl.delete.assert_called_once_with("pod", "iris-test-job-0-0")
    assert result.updates == []


def test_delete_pods_uses_task_hash_label(provider, mock_kubectl):
    """_delete_pods_by_task_id must filter by _LABEL_TASK_HASH, not sanitized task_id."""
    task_id = "/test-job/0"
    mock_kubectl.list_json.return_value = []

    provider._delete_pods_by_task_id(task_id)

    call_kwargs = mock_kubectl.list_json.call_args.kwargs
    labels_used = call_kwargs.get("labels", {})
    assert _LABEL_TASK_HASH in labels_used
    assert labels_used[_LABEL_TASK_HASH] == _task_hash(task_id)


def test_delete_pods_does_not_delete_colliding_task(provider, mock_kubectl):
    """Two task IDs with the same sanitized label must not share hash-based pod deletion."""
    base = "a" * 63
    task_id_a = base + "X"
    task_id_b = base + "Y"
    assert _sanitize_label_value(task_id_a) == _sanitize_label_value(task_id_b)

    pod_calls: list[dict] = []

    def capture_list_json(resource, **kwargs):
        labels = kwargs.get("labels", {})
        if resource == "pods":
            pod_calls.append({"resource": resource, "labels": labels})
        return []

    mock_kubectl.list_json.side_effect = capture_list_json

    provider._delete_pods_by_task_id(task_id_a)
    provider._delete_pods_by_task_id(task_id_b)

    hash_a = pod_calls[0]["labels"][_LABEL_TASK_HASH]
    hash_b = pod_calls[1]["labels"][_LABEL_TASK_HASH]
    assert hash_a != hash_b, "distinct task IDs must use distinct hash labels for deletion"


# ---------------------------------------------------------------------------
# sync(): running_tasks polling
# ---------------------------------------------------------------------------


def test_sync_running_task_returns_running_state(provider, mock_kubectl):
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING


def test_sync_pod_not_found_marks_failed(provider, mock_kubectl):
    """Pod must be missing for _POD_NOT_FOUND_GRACE_CYCLES consecutive syncs before FAILED.

    Pod disappearance is treated as application failure (FAILED) rather than
    infrastructure failure (WORKER_FAILED) to prevent runaway retries via
    max_retries_preemption.
    """
    task_id = JobName.from_wire("/job/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)
    mock_kubectl.list_json.return_value = []

    batch = make_batch(running_tasks=[entry])

    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert len(result.updates) == 1
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED
    assert result.updates[0].error == "Pod not found"


def test_pod_not_found_grace_period(provider, mock_kubectl):
    """A single missing-pod sync returns RUNNING, not FAILED."""
    task_id = JobName.from_wire("/job/grace")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)
    mock_kubectl.list_json.return_value = []

    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING


def test_pod_not_found_grace_resets_when_pod_reappears(provider, mock_kubectl):
    """If the pod reappears after a transient miss, the grace counter resets."""
    task_id = JobName.from_wire("/job/reset")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)
    batch = make_batch(running_tasks=[entry])

    # Miss for (grace - 1) cycles.
    mock_kubectl.list_json.return_value = []
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Pod reappears — counter should reset.
    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.return_value = None
    result = provider.sync(batch)
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Now disappear again: need full grace cycles again before failure.
    mock_kubectl.list_json.return_value = []
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED


def test_sync_succeeded_pod_fetches_logs(provider, mock_kubectl):
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Succeeded")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[
            KubectlLogLine(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                stream="stdout",
                data="task complete",
            )
        ],
        byte_offset=100,
    )

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "task complete"


def test_sync_empty_batch(provider):
    batch = make_batch()
    result = provider.sync(batch)
    assert result.updates == []


# ---------------------------------------------------------------------------
# fetch_live_logs
# ---------------------------------------------------------------------------


def test_fetch_live_logs_returns_entries_with_cursor(provider, mock_kubectl):
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[
            KubectlLogLine(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                stream="stdout",
                data="line 1",
            ),
            KubectlLogLine(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                stream="stdout",
                data="line 2",
            ),
        ],
        byte_offset=200,
    )

    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=10)

    assert len(entries) == 2
    assert entries[0].data == "line 1"
    assert next_cursor == 200


def test_fetch_live_logs_respects_max_lines(provider, mock_kubectl):
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[
            KubectlLogLine(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                stream="stdout",
                data=f"line {i}",
            )
            for i in range(10)
        ],
        byte_offset=500,
    )

    entries, _ = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=3)
    assert len(entries) == 3


def test_fetch_live_logs_falls_back_to_previous_when_empty(provider, mock_kubectl):
    """When stream_logs returns nothing, fall back to kubectl.logs(previous=True)."""
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.logs.return_value = "line a\nline b\nline c\n"

    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=0, max_lines=10)

    mock_kubectl.logs.assert_called_once()
    assert any(e.data == "line a" for e in entries)
    assert next_cursor == 3


def test_fetch_live_logs_fallback_replays_all_with_nonzero_cursor(provider, mock_kubectl):
    """Fallback always replays all lines regardless of cursor (byte offset != line index)."""
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.logs.return_value = "line a\nline b\nline c\n"

    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=1024, max_lines=100)

    assert len(entries) == 3
    assert entries[0].data == "line a"
    assert next_cursor == 3


# ---------------------------------------------------------------------------
# Incremental log polling
# ---------------------------------------------------------------------------


def test_poll_fetches_incremental_logs_for_running_pods(provider, mock_kubectl):
    """Running pods get incremental logs via stream_logs each sync cycle."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[
            KubectlLogLine(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                stream="stdout",
                data="hello from running pod",
            ),
        ],
        byte_offset=128,
    )

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "hello from running pod"
    mock_kubectl.stream_logs.assert_called_once_with(pod_name, container="task", byte_offset=0)


def test_log_cursors_advance_across_sync_cycles(provider, mock_kubectl):
    """Byte offset from stream_logs is used as the cursor in the next sync cycle."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]

    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 1")],
        byte_offset=128,
    )
    provider.sync(make_batch(running_tasks=[entry]))

    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 2")],
        byte_offset=256,
    )
    provider.sync(make_batch(running_tasks=[entry]))

    calls = mock_kubectl.stream_logs.call_args_list
    assert calls[0].kwargs["byte_offset"] == 0
    assert calls[1].kwargs["byte_offset"] == 128


def test_final_log_fetch_on_pod_completion(provider, mock_kubectl):
    """Completed pods get a final full-log fetch; longer result replaces incremental logs."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Succeeded")]

    call_count = 0

    def stream_logs_side_effect(pod, *, container, byte_offset):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return KubectlLogResult(
                lines=[
                    KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 3")
                ],
                byte_offset=192,
            )
        return KubectlLogResult(
            lines=[
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 1"),
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 2"),
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 3"),
            ],
            byte_offset=192,
        )

    mock_kubectl.stream_logs.side_effect = stream_logs_side_effect

    result = provider.sync(make_batch(running_tasks=[entry]))

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert len(result.updates[0].log_entries) == 3
    assert result.updates[0].log_entries[0].data == "line 1"


# ---------------------------------------------------------------------------
# _query_capacity
# ---------------------------------------------------------------------------


def test_query_capacity_returns_cluster_capacity(provider, mock_kubectl):
    """Capacity reports total and available resources as ClusterCapacity."""

    def list_json_side_effect(resource, **kwargs):
        if resource == "nodes":
            return [
                {
                    "spec": {"taints": []},
                    "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
                },
                {
                    "spec": {"taints": []},
                    "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
                },
            ]
        if resource == "pods":
            return [
                {
                    "status": {"phase": "Running"},
                    "spec": {
                        "containers": [
                            {
                                "resources": {
                                    "limits": {"cpu": "1000m", "memory": str(2 * 1024**3)},
                                }
                            }
                        ]
                    },
                }
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect

    cap = provider._query_capacity()
    assert cap is not None
    assert isinstance(cap, ClusterCapacity)
    assert cap.schedulable_nodes == 2
    assert cap.total_cpu_millicores == 8000
    assert cap.total_memory_bytes == 2 * 8 * 1024**3
    assert cap.available_cpu_millicores == 7000
    assert cap.available_memory_bytes == (2 * 8 - 2) * 1024**3


def test_query_capacity_skips_tainted_nodes(provider, mock_kubectl):
    def list_json_side_effect(resource, **kwargs):
        if resource == "nodes":
            return [
                {
                    "spec": {"taints": [{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]},
                    "status": {"allocatable": {"cpu": "8", "memory": "16Gi"}},
                },
                {
                    "spec": {"taints": []},
                    "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
                },
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect

    cap = provider._query_capacity()
    assert cap is not None
    assert cap.schedulable_nodes == 1
    assert cap.total_memory_bytes == 8 * 1024**3


def test_query_capacity_returns_none_when_all_tainted(provider, mock_kubectl):
    def list_json_side_effect(resource, **kwargs):
        if resource == "nodes":
            return [
                {
                    "spec": {"taints": [{"effect": "NoSchedule"}]},
                    "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
                },
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect
    cap = provider._query_capacity()
    assert cap is None


# ---------------------------------------------------------------------------
# _fetch_scheduling_events
# ---------------------------------------------------------------------------


def test_fetch_scheduling_events_returns_events(provider, mock_kubectl):
    pod_name = _pod_name(JobName.from_wire("/test-job/0"), 1)

    def list_json_side_effect(resource, **kwargs):
        if resource == "pods":
            return [
                {
                    "metadata": {
                        "name": pod_name,
                        "labels": {
                            "iris.managed": "true",
                            "iris.runtime": "iris-kubernetes",
                            "iris.task_id": "test-job.0",
                            "iris.attempt_id": "1",
                        },
                    },
                    "status": {"phase": "Pending"},
                }
            ]
        if resource == "events":
            return [
                {
                    "involvedObject": {"kind": "Pod", "name": pod_name},
                    "type": "Warning",
                    "reason": "FailedScheduling",
                    "message": "0/3 nodes available",
                }
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect

    events = provider._fetch_scheduling_events()
    assert len(events) == 1
    assert isinstance(events[0], SchedulingEvent)
    assert events[0].task_id == "test-job.0"
    assert events[0].attempt_id == 1
    assert events[0].reason == "FailedScheduling"


def test_fetch_scheduling_events_ignores_non_iris_events(provider, mock_kubectl):
    def list_json_side_effect(resource, **kwargs):
        if resource == "pods":
            return []
        if resource == "events":
            return [
                {
                    "involvedObject": {"kind": "Pod", "name": "some-other-pod"},
                    "type": "Warning",
                    "reason": "FailedScheduling",
                    "message": "0/3 nodes available",
                }
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect

    events = provider._fetch_scheduling_events()
    assert events == []


def test_fetch_scheduling_events_returns_empty_on_failure(provider, mock_kubectl):
    mock_kubectl.list_json.side_effect = RuntimeError("events API unavailable")
    events = provider._fetch_scheduling_events()
    assert events == []


# ---------------------------------------------------------------------------
# get_cluster_status
# ---------------------------------------------------------------------------


def test_get_cluster_status_basic(mock_kubectl):
    """get_cluster_status returns namespace, node counts, and pod statuses."""

    def list_json_side_effect(resource, **kwargs):
        if resource == "nodes":
            return [
                {
                    "metadata": {"name": "node-1"},
                    "spec": {},
                    "status": {
                        "allocatable": {"cpu": "4", "memory": "8Gi"},
                    },
                },
                {
                    "metadata": {"name": "node-2"},
                    "spec": {"taints": [{"effect": "NoSchedule", "key": "k"}]},
                    "status": {"allocatable": {"cpu": "4", "memory": "8Gi"}},
                },
            ]
        if resource == "pods":
            return [
                {
                    "metadata": {
                        "name": "iris-task-0",
                        "labels": {
                            "iris.managed": "true",
                            "iris.runtime": "iris-kubernetes",
                            "iris.task_id": "job-0",
                            "iris.attempt_id": "0",
                        },
                    },
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [],
                        "conditions": [],
                    },
                }
            ]
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect
    provider = KubernetesProvider(kubectl=mock_kubectl, namespace="iris", default_image="img:latest")
    resp = provider.get_cluster_status()

    assert resp.namespace == "iris"
    assert resp.total_nodes == 2
    assert resp.schedulable_nodes == 1
    assert "cores" in resp.allocatable_cpu
    assert "GiB" in resp.allocatable_memory
    assert len(resp.pod_statuses) == 1
    assert resp.pod_statuses[0].pod_name == "iris-task-0"
    assert resp.pod_statuses[0].phase == "Running"


def test_get_cluster_status_node_failure(mock_kubectl):
    """get_cluster_status handles node query failure gracefully."""

    def list_json_side_effect(resource, **kwargs):
        if resource == "nodes":
            raise RuntimeError("kubectl error")
        return []

    mock_kubectl.list_json.side_effect = list_json_side_effect
    provider = KubernetesProvider(kubectl=mock_kubectl, namespace="test-ns", default_image="img:latest")
    resp = provider.get_cluster_status()

    assert resp.namespace == "test-ns"
    assert resp.total_nodes == 0
    assert resp.schedulable_nodes == 0


# ---------------------------------------------------------------------------
# Resource stats from kubectl top
# ---------------------------------------------------------------------------


def test_resource_stats_from_kubectl_top(provider, mock_kubectl):
    """Running pods get resource_usage populated from kubectl top pod."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.return_value = (500, 1024 * 1024 * 1024)

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    usage = result.updates[0].resource_usage
    assert usage is not None
    assert usage.cpu_millicores == 500
    assert usage.memory_mb == 1024

    mock_kubectl.top_pod.assert_called_once_with(pod_name)


def test_resource_stats_none_when_metrics_unavailable(provider, mock_kubectl):
    """resource_usage stays None when kubectl top returns None."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.return_value = None

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_none_when_top_pod_raises(provider, mock_kubectl):
    """resource_usage stays None when kubectl top raises an exception."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.side_effect = RuntimeError("metrics-server unavailable")

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_not_fetched_for_non_running_pods(provider, mock_kubectl):
    """kubectl top is not called for pods in terminal phases."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [make_pod(pod_name, "Succeeded")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)

    batch = make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None
    mock_kubectl.top_pod.assert_not_called()


# ---------------------------------------------------------------------------
# Profiling via kubectl exec
# ---------------------------------------------------------------------------


def test_profile_threads_via_kubectl_exec(provider, mock_kubectl):
    """profile_task with threads type calls py-spy dump via kubectl exec."""
    mock_kubectl.exec.return_value = completed_process(stdout="Thread 0x7f00 (idle)\n  main.py:42")

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

    exec_call = mock_kubectl.exec.call_args
    shell_cmd = exec_call[0][1]
    joined = " ".join(shell_cmd)
    assert "py-spy" in joined
    assert "dump" in joined
    assert "--pid" in joined


def test_profile_threads_with_locals(provider, mock_kubectl):
    """profile_task with threads.locals=True passes --locals to py-spy dump."""
    mock_kubectl.exec.return_value = completed_process(stdout="Thread 0x7f00\n  x = 42")

    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(
            threads=cluster_pb2.ThreadsProfile(locals=True),
        ),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert not resp.error
    shell_cmd_str = " ".join(mock_kubectl.exec.call_args[0][1])
    assert "--locals" in shell_cmd_str


def test_profile_cpu_via_kubectl_exec(provider, mock_kubectl):
    """profile_task with cpu type calls py-spy record, reads file, cleans up."""
    mock_kubectl.exec.return_value = completed_process(stdout="")
    mock_kubectl.read_file.return_value = b"<svg>flamegraph</svg>"

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

    exec_call = mock_kubectl.exec.call_args
    shell_cmd_str = " ".join(exec_call[0][1])
    assert "py-spy" in shell_cmd_str
    assert "record" in shell_cmd_str

    mock_kubectl.read_file.assert_called_once()
    mock_kubectl.rm_files.assert_called_once()


def test_profile_memory_flamegraph_via_kubectl_exec(provider, mock_kubectl):
    """profile_task with memory flamegraph attaches memray, transforms, reads file."""
    mock_kubectl.exec.return_value = completed_process(stdout="")
    mock_kubectl.read_file.return_value = b"<html>flamegraph</html>"

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

    assert mock_kubectl.exec.call_count == 2
    mock_kubectl.read_file.assert_called_once()
    mock_kubectl.rm_files.assert_called_once()


def test_profile_memory_table_returns_stdout(provider, mock_kubectl):
    """Memory table format returns stdout instead of reading a file."""
    mock_kubectl.exec.side_effect = [
        completed_process(stdout=""),  # attach
        completed_process(stdout="ALLOC  SIZE  FILE\n100  1KB  main.py"),  # table transform
    ]

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
    mock_kubectl.read_file.assert_not_called()


def test_profile_unknown_type_returns_error(provider, mock_kubectl):
    """An empty ProfileType (no profiler selected) returns an error."""
    request = cluster_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=cluster_pb2.ProfileType(),
    )
    resp = provider.profile_task("/job/0", 0, request)

    assert resp.error == "Unknown profile type"
    assert not resp.profile_data


def test_profile_kubectl_exec_failure_returns_error(provider, mock_kubectl):
    """When kubectl exec fails, the error is captured in the response."""
    mock_kubectl.exec.return_value = completed_process(stdout="", stderr="container not running", returncode=1)

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


def test_configmap_created_for_workdir_files(provider, mock_kubectl):
    """_apply_pod creates a ConfigMap when workdir_files are present."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["script.py"] = b"print('hello')"

    provider._apply_pod(req)

    assert mock_kubectl.apply_json.call_count == 2
    cm_call = mock_kubectl.apply_json.call_args_list[0][0][0]
    assert cm_call["kind"] == "ConfigMap"
    assert cm_call["metadata"]["namespace"] == "iris"
    assert _LABEL_MANAGED in cm_call["metadata"]["labels"]
    assert "f0000" in cm_call["binaryData"]

    pod_call = mock_kubectl.apply_json.call_args_list[1][0][0]
    assert pod_call["kind"] == "Pod"
    assert "initContainers" in pod_call["spec"]


def test_no_configmap_when_no_workdir_files(provider, mock_kubectl):
    """_apply_pod does not create a ConfigMap when no workdir_files are set."""
    req = make_run_req("/my-job/task-0")

    provider._apply_pod(req)

    assert mock_kubectl.apply_json.call_count == 1
    pod_call = mock_kubectl.apply_json.call_args_list[0][0][0]
    assert pod_call["kind"] == "Pod"


def test_configmap_cleaned_up_on_delete(provider, mock_kubectl):
    """_delete_pods_by_task_id also deletes associated ConfigMaps."""
    task_id = "/my-job/task-0"

    mock_kubectl.list_json.side_effect = [
        [{"metadata": {"name": "iris-pod-1"}}],  # pods
        [{"metadata": {"name": "iris-pod-1-wf"}}],  # configmaps
    ]

    provider._delete_pods_by_task_id(task_id)

    assert mock_kubectl.delete.call_count == 2
    mock_kubectl.delete.assert_any_call("pod", "iris-pod-1")
    mock_kubectl.delete.assert_any_call("configmap", "iris-pod-1-wf")
