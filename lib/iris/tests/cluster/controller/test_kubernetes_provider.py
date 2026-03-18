# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider: DirectTaskProvider interface, pod lifecycle, log fetch, capacity."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.direct_provider import ClusterCapacity, SchedulingEvent
from iris.cluster.controller.kubernetes_provider import (
    _INFRASTRUCTURE_FAILURE_REASONS,
    _LABEL_JOB_ID,
    _LABEL_MANAGED,
    _LABEL_TASK_HASH,
    _POD_NOT_FOUND_GRACE_CYCLES,
    KubernetesProvider,
    PodConfig,
    _build_init_container_spec,
    _build_pod_manifest,
    _build_task_script,
    _build_volumes_and_mounts,
    _constraints_to_node_selector,
    _is_infrastructure_failure,
    _job_id_from_task,
    _parse_k8s_quantity,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
    _task_update_from_pod,
)
from iris.cluster.runtime.env import build_common_iris_env
from iris.cluster.controller.transitions import DirectProviderBatch, RunningTaskEntry
from iris.cluster.k8s.kubectl import KubectlLogLine, KubectlLogResult
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kubectl():
    kubectl = MagicMock()
    kubectl.namespace = "iris"
    kubectl.list_json.return_value = []
    kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    kubectl.logs.return_value = ""
    return kubectl


@pytest.fixture
def provider(mock_kubectl):
    return KubernetesProvider(
        kubectl=mock_kubectl,
        namespace="iris",
        default_image="myrepo/iris:latest",
        cache_dir="/cache",
    )


def _pod_config(
    namespace: str = "iris",
    default_image: str = "myrepo/iris:latest",
    **kwargs,
) -> PodConfig:
    return PodConfig(namespace=namespace, default_image=default_image, **kwargs)


def _make_run_req(task_id: str, attempt_id: int = 0, cpu_mc: int = 1000) -> cluster_pb2.Worker.RunTaskRequest:
    req = cluster_pb2.Worker.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.entrypoint.run_command.argv.extend(["python", "train.py"])
    req.environment.env_vars["IRIS_JOB_ID"] = "test-job"
    req.resources.cpu_millicores = cpu_mc
    req.resources.memory_bytes = 4 * 1024**3
    return req


def _make_batch(
    tasks_to_run=None,
    tasks_to_kill=None,
    running_tasks=None,
) -> DirectProviderBatch:
    return DirectProviderBatch(
        running_tasks=running_tasks or [],
        tasks_to_run=tasks_to_run or [],
        tasks_to_kill=tasks_to_kill or [],
    )


def _make_pod(name: str, phase: str, exit_code: int | None = None, reason: str = "") -> dict:
    pod: dict = {
        "metadata": {"name": name},
        "status": {"phase": phase, "containerStatuses": []},
    }
    if exit_code is not None:
        pod["status"]["containerStatuses"] = [
            {
                "state": {
                    "terminated": {
                        "exitCode": exit_code,
                        "reason": reason,
                    }
                }
            }
        ]
    return pod


# ---------------------------------------------------------------------------
# Pod naming
# ---------------------------------------------------------------------------


def test_pod_name_sanitizes_slashes():
    name = _pod_name(JobName.from_wire("/smoke-job/0"), 1)
    assert "/" not in name
    assert name.startswith("iris-")
    assert name.islower()


def test_pod_name_length_limit():
    long_task = "/a" * 50
    name = _pod_name(JobName.from_wire(long_task), 0)
    assert len(name) <= 63


def test_pod_name_deterministic():
    task = JobName.from_wire("/test-job/42")
    assert _pod_name(task, 0) == _pod_name(task, 0)
    assert _pod_name(task, 0) != _pod_name(task, 1)


def test_pod_name_preserves_attempt_suffix_with_long_task_id():
    # A task ID long enough to push the suffix off the end under old truncation behavior.
    long_task = JobName.from_wire("/a" * 40)
    name_0 = _pod_name(long_task, 0)
    name_1 = _pod_name(long_task, 1)
    name_999 = _pod_name(long_task, 999)
    assert len(name_0) <= 63
    assert len(name_1) <= 63
    assert len(name_999) <= 63
    assert name_0 != name_1, "different attempts must produce different pod names"
    # Suffix format is {hash8}-{attempt_id}
    assert name_0.endswith("-0")
    assert name_1.endswith("-1")
    assert name_999.endswith("-999")


def test_pod_name_different_tasks_never_collide():
    # Different task IDs must produce different pod names even when their sanitized
    # prefixes are identical after truncation.
    task_a = JobName.from_wire("/a" * 40 + "-suffix-1")
    task_b = JobName.from_wire("/a" * 40 + "-suffix-2")
    assert _pod_name(task_a, 1) != _pod_name(
        task_b, 1
    ), "sibling tasks with the same long prefix must have different pod names"


# ---------------------------------------------------------------------------
# Pod manifest building
# ---------------------------------------------------------------------------


def test_build_pod_manifest_fields():
    req = _make_run_req("/test-job/0", attempt_id=2)
    manifest = _build_pod_manifest(req, _pod_config())

    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["namespace"] == "iris"
    assert manifest["spec"]["restartPolicy"] == "Never"

    container = manifest["spec"]["containers"][0]
    assert container["image"] == "myrepo/iris:latest"
    # Command is now a bash task script wrapping setup + run
    assert container["command"][0] == "bash"
    assert container["command"][1] == "-lc"
    assert "exec python train.py" in container["command"][2]

    resources = container["resources"]["limits"]
    assert resources["cpu"] == "1000m"
    assert resources["memory"] == str(4 * 1024**3)


def test_build_pod_manifest_env_vars():
    req = _make_run_req("/test-job/0")
    req.environment.env_vars["MY_VAR"] = "hello"
    manifest = _build_pod_manifest(req, _pod_config())
    env_names = {e["name"] for e in manifest["spec"]["containers"][0]["env"]}
    # User vars are present
    assert "MY_VAR" in env_names
    assert "IRIS_JOB_ID" in env_names
    # Iris system vars are injected
    assert "IRIS_TASK_ID" in env_names
    assert "IRIS_NUM_TASKS" in env_names
    assert "IRIS_BIND_HOST" in env_names
    assert "IRIS_WORKDIR" in env_names
    # Downward API for pod IP
    assert "IRIS_ADVERTISE_HOST" in env_names


def test_build_pod_manifest_gpu():
    req = _make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, _pod_config())
    limits = manifest["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "4"


def test_build_pod_manifest_runtime_label():
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, _pod_config())
    assert manifest["metadata"]["labels"]["iris.runtime"] == "iris-kubernetes"


def test_build_pod_manifest_task_hash_label():
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, _pod_config())
    labels = manifest["metadata"]["labels"]
    assert _LABEL_TASK_HASH in labels
    assert labels[_LABEL_TASK_HASH] == _task_hash("/test-job/0")
    # Hash must be a valid k8s label value (alphanumeric, ≤63 chars).
    assert len(labels[_LABEL_TASK_HASH]) <= 63
    assert labels[_LABEL_TASK_HASH].isalnum()


def test_task_hash_distinct_for_sanitization_collisions():
    # Two task IDs that map to the same _sanitize_label_value output must have different hashes.
    # Craft two IDs that differ only beyond the 63-char truncation boundary.
    base = "a" * 63
    id_a = base + "X"
    id_b = base + "Y"
    assert _sanitize_label_value(id_a) == _sanitize_label_value(id_b), "precondition: same sanitized value"
    assert _task_hash(id_a) != _task_hash(id_b), "hashes must be distinct"


# ---------------------------------------------------------------------------
# Phase -> state mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase,expected_state",
    [
        ("Pending", cluster_pb2.TASK_STATE_RUNNING),
        ("Running", cluster_pb2.TASK_STATE_RUNNING),
        ("Succeeded", cluster_pb2.TASK_STATE_SUCCEEDED),
        ("Failed", cluster_pb2.TASK_STATE_FAILED),
        ("Unknown", cluster_pb2.TASK_STATE_FAILED),
    ],
)
def test_task_update_from_pod_phases(phase, expected_state):
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = _make_pod("iris-job-0-0", phase, exit_code=1 if phase == "Failed" else None)
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == expected_state


def test_task_update_failed_has_exit_code():
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = _make_pod("iris-job-0-0", "Failed", exit_code=42, reason="Error")
    update = _task_update_from_pod(entry, pod)
    assert update.exit_code == 42
    assert update.new_state == cluster_pb2.TASK_STATE_FAILED


@pytest.mark.parametrize("reason", sorted(_INFRASTRUCTURE_FAILURE_REASONS))
def test_task_update_infrastructure_failure_is_worker_failed(reason):
    """OOMKilled, Evicted, etc. should be WORKER_FAILED, not FAILED."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = _make_pod("iris-job-0-0", "Failed", exit_code=137, reason=reason)
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert update.exit_code == 137


def test_task_update_application_error_is_failed():
    """Non-zero exit with reason 'Error' is an application failure, not infrastructure."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = _make_pod("iris-job-0-0", "Failed", exit_code=1, reason="Error")
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == cluster_pb2.TASK_STATE_FAILED
    assert update.exit_code == 1


def test_is_infrastructure_failure_with_pod_level_reason():
    """Pod-level eviction (no container statuses) is detected as infrastructure failure."""
    pod: dict = {
        "metadata": {"name": "test"},
        "status": {"phase": "Failed", "reason": "Evicted", "containerStatuses": []},
    }
    assert _is_infrastructure_failure(pod)


def test_is_infrastructure_failure_false_for_application_error():
    pod = _make_pod("test", "Failed", exit_code=1, reason="Error")
    assert not _is_infrastructure_failure(pod)


# ---------------------------------------------------------------------------
# sync(): tasks_to_run
# ---------------------------------------------------------------------------


def test_sync_applies_pods_for_tasks_to_run(provider, mock_kubectl):
    req = _make_run_req("/test-job/0")
    batch = _make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    mock_kubectl.apply_json.assert_called_once()
    manifest = mock_kubectl.apply_json.call_args[0][0]
    assert manifest["kind"] == "Pod"
    assert result.updates == []


def test_sync_propagates_kubectl_failure(provider, mock_kubectl):
    mock_kubectl.apply_json.side_effect = RuntimeError("kubectl down")
    req = _make_run_req("/test-job/0")
    batch = _make_batch(tasks_to_run=[req])

    with pytest.raises(RuntimeError, match="kubectl down"):
        provider.sync(batch)


# ---------------------------------------------------------------------------
# sync(): tasks_to_kill
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_for_tasks_to_kill(provider, mock_kubectl):
    # list_json is called for pods, then configmaps, then capacity pods.
    mock_kubectl.list_json.side_effect = [
        [{"metadata": {"name": "iris-test-job-0-0", "labels": {}}}],  # pods
        [],  # configmaps
        [],  # capacity nodes
        [],  # capacity pods
    ]
    batch = _make_batch(tasks_to_kill=["/test-job/0"])

    result = provider.sync(batch)

    mock_kubectl.delete.assert_called_once_with("pod", "iris-test-job-0-0")
    # Result is a DirectProviderSyncResult; updates should be empty since no running tasks.
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

    # Each call produces one "pods" list_json; compare the task hashes.
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

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]

    batch = _make_batch(running_tasks=[entry])
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

    batch = _make_batch(running_tasks=[entry])

    # First (grace - 1) syncs report RUNNING.
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert len(result.updates) == 1
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Next sync exhausts grace — FAILED (not WORKER_FAILED).
    result = provider.sync(batch)
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED
    assert result.updates[0].error == "Pod not found"


def test_pod_not_found_grace_period(provider, mock_kubectl):
    """A single missing-pod sync returns RUNNING, not FAILED."""
    task_id = JobName.from_wire("/job/grace")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)
    mock_kubectl.list_json.return_value = []

    result = provider.sync(_make_batch(running_tasks=[entry]))
    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING


def test_pod_not_found_grace_resets_when_pod_reappears(provider, mock_kubectl):
    """If the pod reappears after a transient miss, the grace counter resets."""
    task_id = JobName.from_wire("/job/reset")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)
    batch = _make_batch(running_tasks=[entry])

    # Miss for (grace - 1) cycles.
    mock_kubectl.list_json.return_value = []
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING

    # Pod reappears — counter should reset.
    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
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

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Succeeded")]
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

    batch = _make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert len(result.updates[0].log_entries) == 1
    assert result.updates[0].log_entries[0].data == "task complete"


def test_sync_empty_batch(provider):
    batch = _make_batch()
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
    assert next_cursor == 3  # len(lines) — marks all lines consumed


def test_fetch_live_logs_fallback_replays_all_with_nonzero_cursor(provider, mock_kubectl):
    """Fallback always replays all lines regardless of cursor (byte offset != line index)."""
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.logs.return_value = "line a\nline b\nline c\n"

    # Cursor=1024 (byte offset from primary path) must NOT skip 1024 lines
    entries, next_cursor = provider.fetch_live_logs("/job/0", 0, cursor=1024, max_lines=100)

    assert len(entries) == 3
    assert entries[0].data == "line a"
    assert next_cursor == 3


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_is_noop(provider):
    provider.close()  # Should not raise


# ---------------------------------------------------------------------------
# Node resource parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2", 2),
        ("500m", 500),
        ("4Gi", 4 * 1024**3),
        ("1024Mi", 1024 * 1024**2),
        ("100Ki", 100 * 1024),
        ("2G", 2 * 10**9),
        ("0", 0),
        ("", 0),
    ],
)
def test_parse_k8s_quantity(value, expected):
    assert _parse_k8s_quantity(value) == expected


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
    # 1 running pod using 1000m CPU and 2Gi memory
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
    call_count = 0

    def list_json_side_effect(resource, **kwargs):
        nonlocal call_count
        call_count += 1
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
    call_count = 0

    def list_json_side_effect(resource, **kwargs):
        nonlocal call_count
        call_count += 1
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
# Constraint -> nodeSelector mapping
# ---------------------------------------------------------------------------


def _add_eq_constraint(req: cluster_pb2.Worker.RunTaskRequest, key: str, value: str) -> None:
    """Helper: add an EQ string constraint to a RunTaskRequest."""
    c = req.constraints.add()
    c.key = key
    c.op = cluster_pb2.CONSTRAINT_OP_EQ
    c.value.string_value = value


def test_constraints_to_node_selector_pool():
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    _add_eq_constraint(req, "pool", "h100-8x")

    manifest = _build_pod_manifest(req, _pod_config())
    assert manifest["spec"]["nodeSelector"] == {"iris.pool": "h100-8x"}


def test_constraints_to_node_selector_region():
    req = _make_run_req("/my-job/task-0")
    _add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, _pod_config())
    assert manifest["spec"]["nodeSelector"] == {"iris.region": "US-WEST-04A"}


def test_constraints_to_node_selector_multiple():
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    _add_eq_constraint(req, "pool", "h100-8x")
    _add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, _pod_config())
    assert manifest["spec"]["nodeSelector"] == {
        "iris.pool": "h100-8x",
        "iris.region": "US-WEST-04A",
    }


def test_constraints_unknown_key_ignored():
    req = _make_run_req("/my-job/task-0")
    _add_eq_constraint(req, "custom_key", "foo")

    manifest = _build_pod_manifest(req, _pod_config())
    assert "nodeSelector" not in manifest["spec"]


def test_constraints_non_eq_op_ignored():
    req = _make_run_req("/my-job/task-0")
    c = req.constraints.add()
    c.key = "pool"
    c.op = cluster_pb2.CONSTRAINT_OP_NE
    c.value.string_value = "h100-8x"

    manifest = _build_pod_manifest(req, _pod_config())
    assert "nodeSelector" not in manifest["spec"]


def test_constraints_to_node_selector_function_directly():
    """Unit test the helper in isolation."""
    c = cluster_pb2.Constraint(key="pool", op=cluster_pb2.CONSTRAINT_OP_EQ)
    c.value.string_value = "a100-4x"
    assert _constraints_to_node_selector([c]) == {"iris.pool": "a100-4x"}


def test_constraints_to_node_selector_empty():
    assert _constraints_to_node_selector([]) == {}


# ---------------------------------------------------------------------------
# GPU toleration
# ---------------------------------------------------------------------------


def test_build_pod_manifest_gpu_adds_cw_toleration():
    req = _make_run_req("/my-job/task-0")
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="A100", count=4))

    manifest = _build_pod_manifest(req, _pod_config())
    tolerations = manifest["spec"].get("tolerations", [])
    assert any(t.get("key") == "qos.coreweave.cloud/interruptable" for t in tolerations)


def test_build_pod_manifest_no_gpu_no_toleration():
    req = _make_run_req("/my-job/task-0")

    manifest = _build_pod_manifest(req, _pod_config())
    assert "tolerations" not in manifest["spec"]


# ---------------------------------------------------------------------------
# End-to-end: coreweave-style constraints
# ---------------------------------------------------------------------------


def test_coreweave_constraints_end_to_end():
    """Constraints from a coreweave h100-8x scale group map to correct nodeSelector."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="H100", count=8))
    _add_eq_constraint(req, "pool", "h100-8x")
    _add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, _pod_config(default_image="ghcr.io/marin-community/iris-task:latest"))
    spec = manifest["spec"]

    assert spec["nodeSelector"]["iris.pool"] == "h100-8x"
    assert spec["nodeSelector"]["iris.region"] == "US-WEST-04A"
    # GPU requests should also add the CoreWeave toleration.
    assert any(t.get("key") == "qos.coreweave.cloud/interruptable" for t in spec["tolerations"])


# ---------------------------------------------------------------------------
# Rack-level colocation (pod affinity for multi-task jobs)
# ---------------------------------------------------------------------------


def test_build_pod_manifest_single_task_no_affinity():
    """Single-task jobs get no podAffinity (no IB colocation needed)."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 1
    manifest = _build_pod_manifest(
        req, _pod_config(default_image="img:latest", colocation_topology_key="coreweave.cloud/spine")
    )
    assert "affinity" not in manifest["spec"]


def test_build_pod_manifest_multi_task_adds_pod_affinity():
    """Multi-task jobs get podAffinity for IB colocation on same spine."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 2
    manifest = _build_pod_manifest(
        req, _pod_config(default_image="img:latest", colocation_topology_key="coreweave.cloud/spine")
    )
    affinity = manifest["spec"]["affinity"]
    pod_affinity = affinity["podAffinity"]
    terms = pod_affinity["preferredDuringSchedulingIgnoredDuringExecution"]
    assert len(terms) == 1
    term = terms[0]
    assert term["weight"] == 100
    assert term["podAffinityTerm"]["topologyKey"] == "coreweave.cloud/spine"
    labels = term["podAffinityTerm"]["labelSelector"]["matchLabels"]
    assert _LABEL_JOB_ID in labels


def test_build_pod_manifest_multi_task_no_topology_key_no_affinity():
    """Empty colocation_topology_key disables affinity even for multi-task jobs."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 2
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest", colocation_topology_key=""))
    assert "affinity" not in manifest["spec"]


def test_job_id_label_on_pod():
    """Pod metadata includes iris.job_id label."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest"))
    assert _LABEL_JOB_ID in manifest["metadata"]["labels"]


def test_job_id_from_task_strips_task_suffix():
    """_job_id_from_task extracts the parent path from a task wire ID."""
    task_id = JobName.from_wire("/my-job/task-0")
    job_id = _job_id_from_task(task_id)
    assert "task-0" not in job_id
    assert "my-job" in job_id


def test_job_id_shared_across_sibling_tasks():
    """Sibling tasks from the same job produce the same job_id label."""
    task_0 = JobName.from_wire("/training-run/task-0")
    task_1 = JobName.from_wire("/training-run/task-1")
    assert _job_id_from_task(task_0) == _job_id_from_task(task_1)


# ---------------------------------------------------------------------------
# Timeout -> activeDeadlineSeconds
# ---------------------------------------------------------------------------


def test_timeout_sets_active_deadline_seconds():
    req = _make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 3600_000  # 1 hour
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest"))
    assert manifest["spec"]["activeDeadlineSeconds"] == 3600


def test_timeout_rounds_down_to_at_least_one_second():
    req = _make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 500  # sub-second
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest"))
    assert manifest["spec"]["activeDeadlineSeconds"] == 1


def test_no_timeout_no_deadline():
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest"))
    assert "activeDeadlineSeconds" not in manifest["spec"]


def test_zero_timeout_no_deadline():
    req = _make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 0
    manifest = _build_pod_manifest(req, _pod_config(default_image="img:latest"))
    assert "activeDeadlineSeconds" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Volumes and mounts
# ---------------------------------------------------------------------------


def test_build_pod_manifest_includes_standard_volumes():
    """Pod manifest includes all 5 standard volumes plus dshm (6 total)."""
    req = _make_run_req("/test-job/0", attempt_id=1)
    manifest = _build_pod_manifest(req, _pod_config())

    spec = manifest["spec"]
    container = spec["containers"][0]

    volume_names = {v["name"] for v in spec["volumes"]}
    mount_names = {m["name"] for m in container["volumeMounts"]}
    expected_names = {"workdir", "tmpfs", "uv-cache", "cargo-registry", "cargo-target", "dshm"}
    assert volume_names == expected_names
    assert mount_names == expected_names

    mount_paths = {m["mountPath"] for m in container["volumeMounts"]}
    assert "/app" in mount_paths
    assert "/tmp" in mount_paths
    assert "/uv/cache" in mount_paths
    assert "/dev/shm" in mount_paths

    assert container["workingDir"] == "/app"


def test_build_pod_manifest_shm_size_limit_with_gpu():
    """dshm volume gets sizeLimit=100Gi when GPU resources are requested."""
    req = _make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, _pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["medium"] == "Memory"
    assert dshm_volumes[0]["emptyDir"]["sizeLimit"] == "100Gi"


def test_build_pod_manifest_shm_no_size_limit_without_gpu():
    """dshm volume has no sizeLimit when no GPU is requested."""
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, _pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["medium"] == "Memory"
    assert "sizeLimit" not in dshm_volumes[0]["emptyDir"]


def test_build_pod_manifest_shm_size_limit_with_tpu():
    """dshm volume gets sizeLimit=100Gi when TPU resources are requested."""
    req = _make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, _pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["sizeLimit"] == "100Gi"


def test_tpu_adds_sys_resource_capability():
    """TPU pods get SYS_RESOURCE capability for memlock ulimits."""
    req = _make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, _pod_config())

    caps = manifest["spec"]["containers"][0]["securityContext"]["capabilities"]["add"]
    assert "SYS_PTRACE" in caps
    assert "SYS_RESOURCE" in caps


def test_build_volumes_and_mounts_cache_uses_host_path():
    """Cache volumes use hostPath with DirectoryOrCreate under the given cache_dir."""
    volumes, _mounts = _build_volumes_and_mounts("/my-cache", has_accelerator=False)
    cache_volumes = [v for v in volumes if "hostPath" in v]
    assert len(cache_volumes) == 3
    for v in cache_volumes:
        assert v["hostPath"]["path"].startswith("/my-cache/")
        assert v["hostPath"]["type"] == "DirectoryOrCreate"


# ---------------------------------------------------------------------------
# NVIDIA GPU toleration
# ---------------------------------------------------------------------------


def test_nvidia_gpu_toleration_added():
    """GPU pods get both CW interruptable and NVIDIA GPU tolerations."""
    req = _make_run_req("/my-job/task-0")
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="A100", count=4))

    manifest = _build_pod_manifest(req, _pod_config())
    tolerations = manifest["spec"].get("tolerations", [])
    toleration_keys = {t.get("key") for t in tolerations}
    assert "nvidia.com/gpu" in toleration_keys
    assert "qos.coreweave.cloud/interruptable" in toleration_keys


# ---------------------------------------------------------------------------
# SYS_PTRACE security context
# ---------------------------------------------------------------------------


def test_sys_ptrace_capability():
    """Container gets SYS_PTRACE capability for profiling."""
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config())
    container = manifest["spec"]["containers"][0]
    assert "SYS_PTRACE" in container["securityContext"]["capabilities"]["add"]


# ---------------------------------------------------------------------------
# Service account
# ---------------------------------------------------------------------------


def test_service_account_set():
    """serviceAccountName is set in spec when service_account is provided."""
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config(service_account="my-sa"))
    assert manifest["spec"]["serviceAccountName"] == "my-sa"


def test_service_account_omitted_when_empty():
    """serviceAccountName is absent from spec when service_account is empty."""
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config(service_account=""))
    assert "serviceAccountName" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Host networking
# ---------------------------------------------------------------------------


def test_host_network_mode():
    """hostNetwork and dnsPolicy are set when host_network is enabled."""
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config(host_network=True))
    assert manifest["spec"]["hostNetwork"] is True
    assert manifest["spec"]["dnsPolicy"] == "ClusterFirstWithHostNet"


def test_host_network_omitted_when_disabled():
    """hostNetwork and dnsPolicy are absent when host_network is False."""
    req = _make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, _pod_config(host_network=False))
    assert "hostNetwork" not in manifest["spec"]
    assert "dnsPolicy" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Iris env vars and task script
# ---------------------------------------------------------------------------


def test_iris_env_vars_injected():
    """Pod manifest includes IRIS_TASK_ID, IRIS_NUM_TASKS, and other system vars."""
    req = _make_run_req("/test-job/0")
    req.num_tasks = 4
    req.bundle_id = "bundle-abc"
    manifest = _build_pod_manifest(req, _pod_config(controller_address="http://ctrl:8080"))

    env_by_name = {e["name"]: e for e in manifest["spec"]["containers"][0]["env"]}
    # attempt_id=0 omits the suffix (matches task_attempt.py wire format)
    assert env_by_name["IRIS_TASK_ID"]["value"] == "/test-job/0"
    assert env_by_name["IRIS_NUM_TASKS"]["value"] == "4"
    assert env_by_name["IRIS_BUNDLE_ID"]["value"] == "bundle-abc"
    assert env_by_name["IRIS_CONTROLLER_ADDRESS"]["value"] == "http://ctrl:8080"
    assert env_by_name["IRIS_CONTROLLER_URL"]["value"] == "http://ctrl:8080"
    assert env_by_name["IRIS_BIND_HOST"]["value"] == "0.0.0.0"
    assert env_by_name["IRIS_WORKDIR"]["value"] == "/app"
    assert env_by_name["IRIS_PYTHON"]["value"] == "python"
    assert env_by_name["UV_PYTHON_INSTALL_DIR"]["value"] == "/uv/cache/python"
    assert env_by_name["CARGO_TARGET_DIR"]["value"] == "/root/.cargo/target"


def test_advertise_host_uses_downward_api():
    """IRIS_ADVERTISE_HOST is populated via the k8s downward API (status.podIP)."""
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, _pod_config())

    env_by_name = {e["name"]: e for e in manifest["spec"]["containers"][0]["env"]}
    adv = env_by_name["IRIS_ADVERTISE_HOST"]
    assert "valueFrom" in adv
    assert adv["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"


def test_device_env_vars_tpu():
    """TPU device resources inject JAX_PLATFORMS, PJRT_DEVICE, JAX_FORCE_TPU_INIT."""
    req = _make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v4-8", count=4))
    manifest = _build_pod_manifest(req, _pod_config())

    env_by_name = {e["name"]: e.get("value") for e in manifest["spec"]["containers"][0]["env"]}
    assert env_by_name["JAX_PLATFORMS"] == "tpu,cpu"
    assert env_by_name["PJRT_DEVICE"] == "TPU"
    assert env_by_name["JAX_FORCE_TPU_INIT"] == "1"


def test_iris_env_overrides_user_env():
    """Iris system vars override user-supplied vars with the same key."""
    req = _make_run_req("/test-job/0")
    # User sets IRIS_TASK_ID to something wrong; iris env should override
    req.environment.env_vars["IRIS_TASK_ID"] = "wrong-value"
    manifest = _build_pod_manifest(req, _pod_config())

    env_by_name = {e["name"]: e.get("value") for e in manifest["spec"]["containers"][0]["env"]}
    assert env_by_name["IRIS_TASK_ID"] == "/test-job/0"


def test_task_script_includes_setup_commands():
    """Setup commands appear in the task script before the run command."""
    req = _make_run_req("/test-job/0")
    req.entrypoint.setup_commands.extend(["pip install foo", "export BAR=1"])
    script = _build_task_script(req)
    lines = script.split("\n")
    assert "set -e" in lines
    assert "pip install foo" in lines
    assert "export BAR=1" in lines
    # Setup commands come before exec
    setup_idx = lines.index("pip install foo")
    exec_idx = next(i for i, l in enumerate(lines) if l.startswith("exec "))
    assert setup_idx < exec_idx


def test_task_script_exec_run_command():
    """Run command is exec'd as the last line of the task script."""
    req = _make_run_req("/test-job/0")
    script = _build_task_script(req)
    lines = script.split("\n")
    assert lines[-1] == "exec python train.py"


def _common_env_from_req(
    req: cluster_pb2.Worker.RunTaskRequest,
    controller_address: str | None = None,
) -> dict[str, str]:
    """Helper: call build_common_iris_env with fields extracted from a RunTaskRequest."""
    return build_common_iris_env(
        task_id=req.task_id,
        attempt_id=req.attempt_id,
        num_tasks=req.num_tasks,
        bundle_id=req.bundle_id,
        controller_address=controller_address,
        environment=req.environment,
        constraints=req.constraints,
        ports=req.ports,
        resources=req.resources if req.HasField("resources") else None,
    )


def test_build_common_iris_env_no_controller_address():
    """Controller address env vars are omitted when controller_address is None."""
    req = _make_run_req("/test-job/0")
    env = _common_env_from_req(req, controller_address=None)
    assert "IRIS_CONTROLLER_ADDRESS" not in env
    assert "IRIS_CONTROLLER_URL" not in env
    assert "IRIS_TASK_ID" in env


def test_build_common_iris_env_serializes_user_env_as_iris_job_env():
    """User env vars are serialized into IRIS_JOB_ENV for child job inheritance."""
    req = _make_run_req("/test-job/0")
    env = _common_env_from_req(req, controller_address=None)
    import json

    job_env = json.loads(env["IRIS_JOB_ENV"])
    assert job_env["IRIS_JOB_ID"] == "test-job"


def test_build_common_iris_env_includes_attempt_suffix_on_retry():
    """IRIS_TASK_ID includes :attempt_id suffix for retried tasks."""
    req = _make_run_req("/test-job/0", attempt_id=3)
    env = _common_env_from_req(req, controller_address=None)
    assert env["IRIS_TASK_ID"] == "/test-job/0:3"


def test_build_common_iris_env_no_attempt_suffix_for_first_attempt():
    """IRIS_TASK_ID has no suffix when attempt_id is 0."""
    req = _make_run_req("/test-job/0", attempt_id=0)
    env = _common_env_from_req(req, controller_address=None)
    assert env["IRIS_TASK_ID"] == "/test-job/0"


def test_parse_k8s_quantity_decimal():
    """Decimal quantities like '1.5' are parsed correctly."""
    assert _parse_k8s_quantity("1.5") == 1
    assert _parse_k8s_quantity("0.5Gi") == 0.5 * 1024**3


# ---------------------------------------------------------------------------
# Incremental log polling
# ---------------------------------------------------------------------------


def test_poll_fetches_incremental_logs_for_running_pods(provider, mock_kubectl):
    """Running pods get incremental logs via stream_logs each sync cycle."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
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

    batch = _make_batch(running_tasks=[entry])
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

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]

    # First cycle: return 128 bytes of logs.
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 1")],
        byte_offset=128,
    )
    provider.sync(_make_batch(running_tasks=[entry]))

    # Second cycle: stream_logs should be called with byte_offset=128.
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 2")],
        byte_offset=256,
    )
    provider.sync(_make_batch(running_tasks=[entry]))

    calls = mock_kubectl.stream_logs.call_args_list
    assert calls[0].kwargs["byte_offset"] == 0
    assert calls[1].kwargs["byte_offset"] == 128


def test_log_cursors_cleaned_on_terminal_state(provider, mock_kubectl):
    """Cursor is removed from _log_cursors when the pod reaches a terminal phase."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    # First cycle: running pod, cursor gets set.
    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="running")],
        byte_offset=64,
    )
    provider.sync(_make_batch(running_tasks=[entry]))
    cursor_key = f"{task_id.to_wire()}:{attempt_id}"
    assert cursor_key in provider._log_cursors

    # Second cycle: pod succeeded, cursor should be cleaned up.
    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Succeeded")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(
        lines=[KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="done")],
        byte_offset=128,
    )
    provider.sync(_make_batch(running_tasks=[entry]))
    assert cursor_key not in provider._log_cursors


def test_final_log_fetch_on_pod_completion(provider, mock_kubectl):
    """Completed pods get a final full-log fetch; longer result replaces incremental logs."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Succeeded")]

    # stream_logs returns the incremental chunk (1 line at offset 64).
    # _fetch_completed_pod_logs calls stream_logs with byte_offset=0, returning all 3 lines.
    call_count = 0

    def stream_logs_side_effect(pod, *, container, byte_offset):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Incremental fetch (from cursor)
            return KubectlLogResult(
                lines=[
                    KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 3")
                ],
                byte_offset=192,
            )
        # Full fetch (from byte_offset=0)
        return KubectlLogResult(
            lines=[
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 1"),
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 2"),
                KubectlLogLine(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), stream="stdout", data="line 3"),
            ],
            byte_offset=192,
        )

    mock_kubectl.stream_logs.side_effect = stream_logs_side_effect

    result = provider.sync(_make_batch(running_tasks=[entry]))

    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_SUCCEEDED
    # The final full fetch (3 lines) should replace the incremental fetch (1 line).
    assert len(result.updates[0].log_entries) == 3
    assert result.updates[0].log_entries[0].data == "line 1"


def test_log_cursors_cleaned_on_pod_not_found(provider, mock_kubectl):
    """Cursor is removed when the pod disappears after grace period (FAILED)."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    cursor_key = f"{task_id.to_wire()}:{attempt_id}"
    provider._log_cursors[cursor_key] = 64

    mock_kubectl.list_json.return_value = []  # Pod gone

    # Cursor should survive the grace period.
    batch = _make_batch(running_tasks=[entry])
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result.updates[0].new_state == cluster_pb2.TASK_STATE_RUNNING
        assert cursor_key in provider._log_cursors

    # Final sync exhausts grace — cursor cleaned up.
    result = provider.sync(batch)
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_FAILED
    assert cursor_key not in provider._log_cursors


# ---------------------------------------------------------------------------
# Resource stats from kubectl top
# ---------------------------------------------------------------------------


def test_resource_stats_from_kubectl_top(provider, mock_kubectl):
    """Running pods get resource_usage populated from kubectl top pod."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.return_value = (500, 1024 * 1024 * 1024)  # 500m CPU, 1 GiB memory

    batch = _make_batch(running_tasks=[entry])
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

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.return_value = None

    batch = _make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_none_when_top_pod_raises(provider, mock_kubectl):
    """resource_usage stays None when kubectl top raises an exception."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Running")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    mock_kubectl.top_pod.side_effect = RuntimeError("metrics-server unavailable")

    batch = _make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None


def test_resource_stats_not_fetched_for_non_running_pods(provider, mock_kubectl):
    """kubectl top is not called for pods in terminal phases."""
    task_id = JobName.from_wire("/job/0")
    attempt_id = 0
    pod_name = _pod_name(task_id, attempt_id)
    entry = RunningTaskEntry(task_id=task_id, attempt_id=attempt_id)

    mock_kubectl.list_json.return_value = [_make_pod(pod_name, "Succeeded")]
    mock_kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)

    batch = _make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].resource_usage is None
    mock_kubectl.top_pod.assert_not_called()


# ---------------------------------------------------------------------------
# Profiling via kubectl exec
# ---------------------------------------------------------------------------


def _completed_process(stdout="", stderr="", returncode=0):
    """Build a fake subprocess.CompletedProcess for mocking kubectl.exec."""
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_profile_threads_via_kubectl_exec(provider, mock_kubectl):
    """profile_task with threads type calls py-spy dump via kubectl exec."""
    mock_kubectl.exec.return_value = _completed_process(stdout="Thread 0x7f00 (idle)\n  main.py:42")

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

    # Verify kubectl.exec was called with py-spy dump
    exec_call = mock_kubectl.exec.call_args
    shell_cmd = exec_call[0][1]  # second positional arg is the command list
    joined = " ".join(shell_cmd)
    assert "py-spy" in joined
    assert "dump" in joined
    assert "--pid" in joined


def test_profile_threads_with_locals(provider, mock_kubectl):
    """profile_task with threads.locals=True passes --locals to py-spy dump."""
    mock_kubectl.exec.return_value = _completed_process(stdout="Thread 0x7f00\n  x = 42")

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
    mock_kubectl.exec.return_value = _completed_process(stdout="")
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

    # Verify py-spy record was called
    exec_call = mock_kubectl.exec.call_args
    shell_cmd_str = " ".join(exec_call[0][1])
    assert "py-spy" in shell_cmd_str
    assert "record" in shell_cmd_str

    # Verify file was read and cleaned up
    mock_kubectl.read_file.assert_called_once()
    mock_kubectl.rm_files.assert_called_once()


def test_profile_memory_flamegraph_via_kubectl_exec(provider, mock_kubectl):
    """profile_task with memory flamegraph attaches memray, transforms, reads file."""
    # Two exec calls: attach then transform
    mock_kubectl.exec.return_value = _completed_process(stdout="")
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

    # Two exec calls: attach + transform
    assert mock_kubectl.exec.call_count == 2
    mock_kubectl.read_file.assert_called_once()
    mock_kubectl.rm_files.assert_called_once()


def test_profile_memory_table_returns_stdout(provider, mock_kubectl):
    """Memory table format returns stdout instead of reading a file."""
    mock_kubectl.exec.side_effect = [
        _completed_process(stdout=""),  # attach
        _completed_process(stdout="ALLOC  SIZE  FILE\n100  1KB  main.py"),  # table transform
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
    # Table format should not read a file
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
    mock_kubectl.exec.return_value = _completed_process(stdout="", stderr="container not running", returncode=1)

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
# Init containers: bundle fetch and workdir files
# ---------------------------------------------------------------------------


def test_init_container_created_when_bundle_id_present():
    """Setting bundle_id + controller_address produces an init container."""
    req = _make_run_req("/my-job/task-0")
    req.bundle_id = "bundle-abc"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-my-job-task-0-abcd1234-0",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    assert ic["name"] == "stage-workdir"
    assert ic["image"] == "myrepo/iris:latest"
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert env_by_name["IRIS_BUNDLE_ID"] == "bundle-abc"
    assert env_by_name["IRIS_CONTROLLER_URL"] == "http://ctrl:8080"
    assert env_by_name["IRIS_WORKDIR"] == "/app"
    assert configmap_name is None
    assert extra_volumes == []


def test_no_init_container_when_no_bundle_or_files():
    """No init containers when neither bundle_id nor workdir_files are set."""
    req = _make_run_req("/my-job/task-0")
    req.bundle_id = ""

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert init_containers == []
    assert extra_volumes == []
    assert configmap_name is None


def test_init_container_for_workdir_files():
    """Workdir files produce a ConfigMap volume and init container with IRIS_WORKDIR_FILES_SRC."""
    req = _make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["config.yaml"] = b"key: value"
    req.entrypoint.workdir_files["sub/data.txt"] = b"hello"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        None,
    )

    assert len(init_containers) == 1
    assert configmap_name == "iris-pod-name-wf"
    assert len(extra_volumes) == 1
    assert extra_volumes[0]["name"] == "workdir-files"
    assert extra_volumes[0]["configMap"]["name"] == configmap_name

    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert env_by_name["IRIS_WORKDIR_FILES_SRC"] == "/iris/staged-workdir-files"

    mount_by_name = {m["name"]: m for m in ic["volumeMounts"]}
    assert "workdir-files" in mount_by_name
    assert mount_by_name["workdir-files"]["readOnly"] is True


def test_init_container_bundle_and_workdir_files():
    """Both bundle and workdir files produce a single init container with all env vars."""
    req = _make_run_req("/my-job/task-0")
    req.bundle_id = "bundle-xyz"
    req.entrypoint.workdir_files["run.sh"] = b"#!/bin/bash"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert "IRIS_BUNDLE_ID" in env_by_name
    assert "IRIS_WORKDIR_FILES_SRC" in env_by_name
    assert configmap_name is not None
    assert len(extra_volumes) == 1


def test_configmap_created_for_workdir_files(provider, mock_kubectl):
    """_apply_pod creates a ConfigMap when workdir_files are present."""
    req = _make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["script.py"] = b"print('hello')"

    provider._apply_pod(req)

    # Two apply_json calls: one for ConfigMap, one for Pod.
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
    req = _make_run_req("/my-job/task-0")

    provider._apply_pod(req)

    # Only one apply_json call for the Pod.
    assert mock_kubectl.apply_json.call_count == 1
    pod_call = mock_kubectl.apply_json.call_args_list[0][0][0]
    assert pod_call["kind"] == "Pod"


def test_configmap_cleaned_up_on_delete(provider, mock_kubectl):
    """_delete_pods_by_task_id also deletes associated ConfigMaps."""
    task_id = "/my-job/task-0"

    mock_kubectl.list_json.side_effect = [
        # First call: pods
        [{"metadata": {"name": "iris-pod-1"}}],
        # Second call: configmaps
        [{"metadata": {"name": "iris-pod-1-wf"}}],
    ]

    provider._delete_pods_by_task_id(task_id)

    assert mock_kubectl.delete.call_count == 2
    mock_kubectl.delete.assert_any_call("pod", "iris-pod-1")
    mock_kubectl.delete.assert_any_call("configmap", "iris-pod-1-wf")


def test_bundle_fetch_script_exists_and_compiles():
    """The bundle fetch script exists and is valid Python."""
    from pathlib import Path

    script_path = Path(__file__).parents[3] / "src" / "iris" / "cluster" / "controller" / "kubernetes_bundle_fetch.py"
    assert script_path.exists(), f"Bundle fetch script not found at {script_path}"
    source = script_path.read_text()
    compile(source, str(script_path), "exec")
