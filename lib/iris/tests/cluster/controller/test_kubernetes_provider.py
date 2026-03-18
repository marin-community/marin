# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider: DirectTaskProvider interface, pod lifecycle, log fetch, capacity."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.direct_provider import ClusterCapacity, SchedulingEvent
from iris.cluster.controller.kubernetes_provider import (
    _LABEL_JOB_ID,
    _LABEL_TASK_HASH,
    KubernetesProvider,
    _build_pod_manifest,
    _constraints_to_node_selector,
    _job_id_from_task,
    _parse_k8s_quantity,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
    _task_update_from_pod,
)
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
    )


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
    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")

    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["namespace"] == "iris"
    assert manifest["spec"]["restartPolicy"] == "Never"

    container = manifest["spec"]["containers"][0]
    assert container["image"] == "myrepo/iris:latest"
    assert container["command"] == ["python", "train.py"]

    resources = container["resources"]["limits"]
    assert resources["cpu"] == "1000m"
    assert resources["memory"] == str(4 * 1024**3)


def test_build_pod_manifest_env_vars():
    req = _make_run_req("/test-job/0")
    req.environment.env_vars["MY_VAR"] = "hello"
    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    env_names = {e["name"] for e in manifest["spec"]["containers"][0]["env"]}
    assert "MY_VAR" in env_names
    assert "IRIS_JOB_ID" in env_names


def test_build_pod_manifest_gpu():
    req = _make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    limits = manifest["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "4"


def test_build_pod_manifest_runtime_label():
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    assert manifest["metadata"]["labels"]["iris.runtime"] == "iris-kubernetes"


def test_build_pod_manifest_task_hash_label():
    req = _make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
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


def test_sync_returns_empty_result_on_kubectl_failure(provider, mock_kubectl):
    mock_kubectl.apply_json.side_effect = RuntimeError("kubectl down")
    req = _make_run_req("/test-job/0")
    batch = _make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    assert result.updates == []
    assert result.capacity is None


# ---------------------------------------------------------------------------
# sync(): tasks_to_kill
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_for_tasks_to_kill(provider, mock_kubectl):
    mock_kubectl.list_json.return_value = [
        {"metadata": {"name": "iris-test-job-0-0", "labels": {}}},
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

    calls: list[dict] = []

    def capture_list_json(resource, **kwargs):
        calls.append({"resource": resource, "labels": kwargs.get("labels", {})})
        return []

    mock_kubectl.list_json.side_effect = capture_list_json

    provider._delete_pods_by_task_id(task_id_a)
    provider._delete_pods_by_task_id(task_id_b)

    hash_a = calls[0]["labels"][_LABEL_TASK_HASH]
    hash_b = calls[1]["labels"][_LABEL_TASK_HASH]
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


def test_sync_pod_not_found_marks_worker_failed(provider, mock_kubectl):
    task_id = JobName.from_wire("/job/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)
    mock_kubectl.list_json.return_value = []

    batch = _make_batch(running_tasks=[entry])
    result = provider.sync(batch)

    assert len(result.updates) == 1
    assert result.updates[0].new_state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert result.updates[0].error == "Pod not found"


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
    mock_kubectl.list_json.return_value = [
        {
            "metadata": {
                "labels": {
                    "iris.task_id": "test-job.0",
                    "iris.attempt_id": "1",
                },
            },
            "type": "Warning",
            "reason": "FailedScheduling",
            "message": "0/3 nodes available",
        }
    ]

    events = provider._fetch_scheduling_events()
    assert len(events) == 1
    assert isinstance(events[0], SchedulingEvent)
    assert events[0].task_id == "test-job.0"
    assert events[0].attempt_id == 1
    assert events[0].reason == "FailedScheduling"


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

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    assert manifest["spec"]["nodeSelector"] == {"iris.pool": "h100-8x"}


def test_constraints_to_node_selector_region():
    req = _make_run_req("/my-job/task-0")
    _add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    assert manifest["spec"]["nodeSelector"] == {"iris.region": "US-WEST-04A"}


def test_constraints_to_node_selector_multiple():
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    _add_eq_constraint(req, "pool", "h100-8x")
    _add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    assert manifest["spec"]["nodeSelector"] == {
        "iris.pool": "h100-8x",
        "iris.region": "US-WEST-04A",
    }


def test_constraints_unknown_key_ignored():
    req = _make_run_req("/my-job/task-0")
    _add_eq_constraint(req, "custom_key", "foo")

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    assert "nodeSelector" not in manifest["spec"]


def test_constraints_non_eq_op_ignored():
    req = _make_run_req("/my-job/task-0")
    c = req.constraints.add()
    c.key = "pool"
    c.op = cluster_pb2.CONSTRAINT_OP_NE
    c.value.string_value = "h100-8x"

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
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

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
    tolerations = manifest["spec"].get("tolerations", [])
    assert any(t.get("key") == "qos.coreweave.cloud/interruptable" for t in tolerations)


def test_build_pod_manifest_no_gpu_no_toleration():
    req = _make_run_req("/my-job/task-0")

    manifest = _build_pod_manifest(req, "iris", "myrepo/iris:latest")
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

    manifest = _build_pod_manifest(req, "iris", "ghcr.io/marin-community/iris-task:latest")
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
    manifest = _build_pod_manifest(req, "iris", "img:latest", colocation_topology_key="coreweave.cloud/spine")
    assert "affinity" not in manifest["spec"]


def test_build_pod_manifest_multi_task_adds_pod_affinity():
    """Multi-task jobs get podAffinity for IB colocation on same spine."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 2
    manifest = _build_pod_manifest(req, "iris", "img:latest", colocation_topology_key="coreweave.cloud/spine")
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
    manifest = _build_pod_manifest(req, "iris", "img:latest", colocation_topology_key="")
    assert "affinity" not in manifest["spec"]


def test_job_id_label_on_pod():
    """Pod metadata includes iris.job_id label."""
    req = _make_run_req("/my-job/task-0", attempt_id=1)
    manifest = _build_pod_manifest(req, "iris", "img:latest")
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
