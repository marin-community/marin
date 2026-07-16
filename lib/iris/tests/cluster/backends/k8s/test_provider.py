# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8sTaskProvider: sync lifecycle, capacity, scheduling, profiling."""

from datetime import UTC, datetime, timedelta

import pytest
from iris.cluster.backends.k8s.tasks import (
    _GANG_GC_MAX_AGE_SECONDS,
    _GC_MAX_AGE_SECONDS,
    _KUEUE_MANAGED_FINALIZER,
    _KUEUE_POD_GROUP_NAME,
    _KUEUE_POD_GROUP_TOTAL,
    _LABEL_ATTEMPT_ID,
    _LABEL_JOB_ID,
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _LABEL_TASK_HASH,
    _LABEL_TASK_ID,
    _MANAGED_POD_LABELS,
    _POD_NOT_FOUND_GRACE_CYCLES,
    _RUNTIME_LABEL_VALUE,
    K8sTaskProvider,
    ResourceCollector,
    _pod_name,
    _sanitize_label_value,
    _task_hash,
)
from iris.cluster.controller.backend import TaskTarget
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.coreweave_topology import RACK_SIZE
from iris.cluster.platforms.k8s.types import ExecResult, K8sResource, KubectlError, PodResourceUsage
from iris.cluster.types import JobName
from iris.cluster.worker.stats import IrisTaskStat
from iris.rpc import job_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration

from .conftest import make_batch, make_kueue_provider, make_run_req, populate_node, populate_pod

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
    assert result == []


def test_sync_propagates_non_kubectl_failure(provider, k8s):
    k8s.inject_failure("apply_json", RuntimeError("kubectl down"))
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    with pytest.raises(RuntimeError, match="kubectl down"):
        provider.sync(batch)


def test_sync_apply_error_yields_worker_failed(provider, k8s):
    """A pod-apply KubectlError -> WORKER_FAILED (retryable worker loss).

    The pod was never created, so there is no k8s verdict to track and nothing
    ran. Any apply failure is treated as worker loss so the task retries on the
    next sync rather than permanently failing the job.
    """
    k8s.inject_failure(
        "apply_json",
        KubectlError("apply Pod/x failed: apiserver unavailable"),
    )
    req = make_run_req("/test-job/0")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    assert len(result) == 1
    assert result[0].new_state == job_pb2.TASK_STATE_WORKER_FAILED


def test_sync_invalid_manifest_fails_task_terminally(provider, k8s):
    """An unbuildable manifest -> terminal FAILED, not retryable WORKER_FAILED.

    A programmatic client can stamp a required nvlink.domain gang larger than one
    rack (the CLI caps it, direct RunTaskRequests do not). The manifest can never
    be built, so retrying would rebuild the same broken request every tick and
    wedge the reconcile loop. sync must fail the task terminally and keep going.
    """
    req = make_run_req("/test-job/0", num_tasks=RACK_SIZE + 1, coscheduling_group_by="nvlink.domain")
    batch = make_batch(tasks_to_run=[req])

    result = provider.sync(batch)

    assert len(result) == 1
    assert result[0].new_state == job_pb2.TASK_STATE_FAILED
    assert "NVLink domain size" in result[0].error
    # Nothing was created, and the tick was not aborted by the escaping error.
    assert k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true"}) == []


def test_redrive_does_not_recreate_running_pod(provider, k8s):
    """Re-applying a still-ASSIGNED task's RunTaskRequest never recreates its pod.

    A task stays ASSIGNED until a (slower) poll observes it Running, so the apply
    loop redrives the same RunTaskRequest every reconcile tick. That redrive must
    be create-if-absent: delete-and-recreating the live pod destroyed the running
    task and raced its own deletion (409 AlreadyExists), churning the task
    through attempts until it failed. Here the pod is Running, so the second
    apply is a no-op — the pod keeps its phase and the task is not failed.
    """
    req = make_run_req("/test-job/0")

    provider.sync(make_batch(tasks_to_run=[req]))
    pods = k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})
    assert len(pods) == 1
    pod_name = pods[0]["metadata"]["name"]
    k8s.transition_pod(pod_name, "Running")

    # Redrive: same RunTaskRequest while the task is still ASSIGNED.
    result = provider.sync(make_batch(tasks_to_run=[req]))

    pod_after = k8s.get_json(K8sResource.PODS, pod_name)
    assert pod_after is not None
    assert pod_after["status"]["phase"] == "Running"  # not reset by a recreate
    assert all(u.new_state != job_pb2.TASK_STATE_WORKER_FAILED for u in result)


# ---------------------------------------------------------------------------
# sync(): stray pod deletion (kill via desired-set diff)
# ---------------------------------------------------------------------------


def test_sync_deletes_pods_not_in_desired_set(provider, k8s):
    """A managed pod whose (task_hash, attempt_id) is not in tasks_to_run|running_tasks
    is considered a stray and gets deleted."""
    task_id = "/test-job/0"
    populate_pod(
        k8s,
        "iris-test-job-0-0",
        "Running",
        labels={
            _LABEL_TASK_HASH: _task_hash(task_id),
            _LABEL_ATTEMPT_ID: "0",
            _LABEL_JOB_ID: _sanitize_label_value("/test-job"),
        },
    )
    # Empty batch: nothing desired → existing pod is stray.
    batch = make_batch()

    result = provider.sync(batch)

    assert k8s.get_json(K8sResource.PODS, "iris-test-job-0-0") is None
    assert result == []


def test_sync_keeps_pods_in_desired_running_set(provider, k8s):
    """A managed pod for a desired (task_hash, attempt_id) is kept across the diff."""
    task_id = JobName.from_wire("/test-job/0")
    pod_name = _pod_name(task_id, 0)
    populate_pod(
        k8s,
        pod_name,
        "Running",
        labels={
            _LABEL_TASK_HASH: _task_hash(task_id.to_wire()),
            _LABEL_ATTEMPT_ID: "0",
        },
    )
    batch = make_batch(running_tasks=[RunningTaskEntry(task_id=task_id, attempt_id=0)])

    provider.sync(batch)

    assert k8s.get_json(K8sResource.PODS, pod_name) is not None


def test_sync_deletes_pod_for_stale_attempt(provider, k8s):
    """A pod for an older attempt of a still-active task is a stray (attempt_id mismatch)."""
    task_id = JobName.from_wire("/test-job/0")
    old_pod = _pod_name(task_id, 0)
    populate_pod(
        k8s,
        old_pod,
        "Running",
        labels={
            _LABEL_TASK_HASH: _task_hash(task_id.to_wire()),
            _LABEL_ATTEMPT_ID: "0",
        },
    )
    # Desired = attempt 1 (task was preempted and re-promoted).
    batch = make_batch(running_tasks=[RunningTaskEntry(task_id=task_id, attempt_id=1)])

    provider.sync(batch)

    assert k8s.get_json(K8sResource.PODS, old_pod) is None


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

    assert len(result) == 1
    assert result[0].new_state == job_pb2.TASK_STATE_RUNNING


def test_sync_pod_not_found_marks_failed(provider, k8s):
    """Pod must be missing for _POD_NOT_FOUND_GRACE_CYCLES consecutive syncs before FAILED."""
    task_id = JobName.from_wire("/job/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    batch = make_batch(running_tasks=[entry])

    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert len(result) == 1
        assert result[0].new_state == job_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert len(result) == 1
    assert result[0].new_state == job_pb2.TASK_STATE_FAILED


def test_sync_coscheduled_pod_not_found_is_worker_failed(provider, k8s):
    """A vanished pod for a coscheduled task is billed as WORKER_FAILED (gang preemption),
    not FAILED — Kueue deletes every pod in a preempted group, leaving only the absence."""
    task_id = JobName.from_wire("/gang/task/0")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0, coscheduled=True)
    batch = make_batch(running_tasks=[entry])

    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result[0].new_state == job_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert result[0].new_state == job_pb2.TASK_STATE_WORKER_FAILED


def test_pod_not_found_grace_period(provider, k8s):
    """A single missing-pod sync returns RUNNING, not FAILED."""
    task_id = JobName.from_wire("/job/grace")
    entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

    result = provider.sync(make_batch(running_tasks=[entry]))
    assert len(result) == 1
    assert result[0].new_state == job_pb2.TASK_STATE_RUNNING


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
        assert result[0].new_state == job_pb2.TASK_STATE_RUNNING

    # Pod reappears — counter should reset.
    populate_pod(k8s, pod_name, "Running")
    k8s.set_top_pod(pod_name, None)
    result = provider.sync(batch)
    assert result[0].new_state == job_pb2.TASK_STATE_RUNNING

    # Now disappear again: need full grace cycles again before failure.
    k8s.delete(K8sResource.PODS, pod_name)
    for _ in range(_POD_NOT_FOUND_GRACE_CYCLES - 1):
        result = provider.sync(batch)
        assert result[0].new_state == job_pb2.TASK_STATE_RUNNING

    result = provider.sync(batch)
    assert result[0].new_state == job_pb2.TASK_STATE_FAILED


def test_sync_empty_batch(provider):
    batch = make_batch()
    result = provider.sync(batch)
    assert result == []


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
            _LABEL_TASK_ID: "job-0",
            _LABEL_ATTEMPT_ID: "0",
        },
    )
    pod = k8s.get_json(K8sResource.PODS, "iris-task-0")
    pod["status"]["conditions"] = []

    p = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest", cluster_scan_interval=0.0)
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
    p = K8sTaskProvider(kubectl=k8s, namespace="test-ns", default_image="img:latest", cluster_scan_interval=0.0)
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

    p = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest", cluster_scan_interval=0.0)
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


def test_sync_survives_node_list_failure(provider, k8s):
    """When the node list fails during sync, reconcile still returns and pod statuses populate from the pod list."""
    populate_pod(k8s, "iris-running", "Running")
    k8s.inject_failure("list_json:node", RuntimeError("nodes unavailable"))

    provider.sync(make_batch())

    # Pod statuses are still populated from the successful pod list.
    resp = provider.get_cluster_status()
    assert any(ps.pod_name == "iris-running" for ps in resp.pod_statuses)


# ---------------------------------------------------------------------------
# Resource stats from kubectl top
# ---------------------------------------------------------------------------


def test_resource_stats_only_for_running_tasks(provider, k8s, task_stats_table):
    """reconcile registers running pods (not terminal ones) so the background
    collector emits IrisTaskStat rows only for running tasks."""
    running = RunningTaskEntry(task_id=JobName.from_wire("/job/run"), attempt_id=0)
    terminal = RunningTaskEntry(task_id=JobName.from_wire("/job/done"), attempt_id=0)
    running_pod = _pod_name(running.task_id, running.attempt_id)
    terminal_pod = _pod_name(terminal.task_id, terminal.attempt_id)

    populate_pod(k8s, running_pod, "Running")
    populate_pod(k8s, terminal_pod, "Succeeded")
    k8s.set_top_pod(running_pod, PodResourceUsage(cpu_millicores=500, memory_bytes=1024 * 1024 * 1024))
    k8s.set_top_pod(terminal_pod, PodResourceUsage(cpu_millicores=999, memory_bytes=1024))

    provider.sync(make_batch(running_tasks=[running, terminal]))
    # The collector samples all tracked pods in one pass, so once the running
    # pod's row lands a full cycle has run — the terminal pod's absence is real.
    wait_for_condition(lambda: bool(task_stats_table.writes), timeout=Duration.from_seconds(5.0))

    rows = [row for batch_rows in list(task_stats_table.writes) for row in batch_rows]
    assert all(isinstance(r, IrisTaskStat) for r in rows)
    assert {r.worker_id for r in rows} == {running_pod}, "only the running pod should be sampled"
    row = next(r for r in rows if r.worker_id == running_pod)
    assert row.task_id == running.task_id.to_wire()
    assert row.cpu_millicores == 500
    assert row.memory_mb == 1024


def test_resource_collector_skips_pod_without_metrics_sample(k8s, task_stats_table):
    """A tracked pod with no metrics sample produces no row."""
    k8s.set_top_pod("pod-a", None)

    collector = ResourceCollector(k8s, task_stats_table, poll_interval=60.0)
    collector.close()  # stop the background loop; drive one collection synchronously
    collector.set_pods({("/job/0", 0): "pod-a"})
    collector.collect_once()

    assert task_stats_table.writes == []


def test_resource_collector_swallows_metrics_query_failure(k8s, task_stats_table):
    """A raising bulk metrics query is swallowed; no row is written."""
    k8s.inject_persistent_failure("top_pods", RuntimeError("metrics-server unavailable"))

    collector = ResourceCollector(k8s, task_stats_table, poll_interval=60.0)
    collector.close()
    collector.set_pods({("/job/0", 0): "pod-a"})
    collector.collect_once()

    assert task_stats_table.writes == []


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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=1, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    k8s.set_file_content(pod_name, "/tmp/iris-profile.html", b"<html>flamegraph</html>")

    request = job_pb2.ProfileTaskRequest(
        target="/job/0",
        duration_seconds=5,
        profile_type=job_pb2.ProfileType(
            memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH),
        ),
    )
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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
    resp = provider.profile_task(
        TaskTarget(task_id="/job/0", attempt_id=0, worker_id=None, address=None), request, timeout_ms=30000
    )

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


def test_stray_delete_defers_pdb_cleanup_to_gc(provider, k8s):
    """_delete_stray_pods deletes pods immediately but defers PDB/CM cleanup to GC."""
    task_id = "/coord-job/0"
    task_hash = _task_hash(task_id)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: task_hash,
    }

    populate_pod(
        k8s,
        "iris-coord-pod",
        "Running",
        labels={_LABEL_TASK_HASH: task_hash, _LABEL_ATTEMPT_ID: "0"},
    )
    pdb = {
        "kind": "PodDisruptionBudget",
        "metadata": {"name": "iris-coord-pod-pdb", "labels": labels},
        "spec": {"minAvailable": 1},
    }
    k8s.seed_resource(K8sResource.PDBS, "iris-coord-pod-pdb", pdb)

    cached_pods = k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})
    # Empty desired set → pod is stray.
    provider._delete_stray_pods(cached_pods, desired_keys=set())

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
    now = datetime.now(UTC)
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

    now = datetime.now(UTC)
    old_ts = (now - timedelta(seconds=_GC_MAX_AGE_SECONDS + 600)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Trigger GC once to set _last_gc_time to now.
    provider._maybe_gc_terminal_resources(active_pods=[])

    # Seed an old pod. An immediate second call should NOT trigger GC (interval not elapsed).
    _seed_terminal_pod(k8s, "gc-pod-1", "Succeeded", "aaaa111122223333", old_ts)
    provider._maybe_gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.PODS, "gc-pod-1") is not None  # Still exists — interval gate held


def test_gc_cleans_up_deferred_configmaps(provider, k8s):
    """GC deletes configmaps for task hashes enqueued by _delete_stray_pods."""
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

    # Simulate _delete_stray_pods enqueuing the hash.
    provider._pending_gc_hashes.add(task_hash)

    # GC picks it up and deletes the configmap.
    provider._gc_terminal_resources(active_pods=[])
    assert k8s.get_json(K8sResource.CONFIGMAPS, "deferred-cm") is None


def test_gc_retains_pending_hash_when_pod_still_in_snapshot(provider, k8s):
    """Deferred hashes must not be dropped when the killed pod is still in the
    pre-delete managed_pods snapshot.

    Reproduces: sync fetches managed_pods, _delete_stray_pods deletes the pod
    and enqueues hash, then _maybe_gc sees the hash as "active" from the stale
    snapshot. The hash must be retained for the next GC cycle.
    """
    task_id = "/kill-me/0"
    task_hash = _task_hash(task_id)
    labels = {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, _LABEL_TASK_HASH: task_hash}

    # Seed the pod and its configmap.
    populate_pod(k8s, "iris-kill-me-0-0", "Running", labels={_LABEL_TASK_HASH: task_hash, _LABEL_ATTEMPT_ID: "0"})
    cm = {"kind": "ConfigMap", "metadata": {"name": "iris-kill-me-0-0-wf", "labels": labels}}
    k8s.seed_resource(K8sResource.CONFIGMAPS, "iris-kill-me-0-0-wf", cm)

    # Snapshot managed pods BEFORE delete (as sync() does).
    pre_delete_pods = k8s.list_json(
        K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE}
    )

    # Kill the pod via stray-set diff (empty desired) — hash goes into _pending_gc_hashes.
    provider._delete_stray_pods(pre_delete_pods, desired_keys=set())
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

    now = datetime.now(UTC)
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


def test_resource_collector_set_pods_replaces_active_set(k8s, task_stats_table):
    """set_pods() replaces the tracked pod set wholesale: a pod dropped from the
    set stops being sampled on the next collection."""
    k8s.set_top_pod("pod-a", PodResourceUsage(cpu_millicores=100, memory_bytes=128 * 1024 * 1024))
    k8s.set_top_pod("pod-b", PodResourceUsage(cpu_millicores=100, memory_bytes=128 * 1024 * 1024))

    collector = ResourceCollector(k8s, task_stats_table, poll_interval=60.0)
    collector.close()  # stop the background loop; drive collections synchronously

    collector.set_pods({("/job/0", 0): "pod-a", ("/job/1", 0): "pod-b"})
    collector.collect_once()
    assert {r.worker_id for r in task_stats_table.writes[-1]} == {"pod-a", "pod-b"}

    collector.set_pods({("/job/1", 0): "pod-b"})
    collector.collect_once()
    assert {r.worker_id for r in task_stats_table.writes[-1]} == {"pod-b"}


def test_resource_collector_writes_iris_task_rows(k8s, task_stats_table):
    """A successful bulk metrics read appends one IrisTaskStat row to the Table."""

    k8s.set_top_pod("pod-a", PodResourceUsage(cpu_millicores=750, memory_bytes=2 * 1024 * 1024 * 1024))

    collector = ResourceCollector(k8s, task_stats_table, poll_interval=60.0)
    # Stop the background loop so we drive a single collection deterministically.
    collector.close()
    collector.set_pods({("/job/0", 3): "pod-a"})
    collector.collect_once()

    rows = [row for batch_rows in task_stats_table.writes for row in batch_rows]
    assert rows, "no rows emitted"
    row = rows[-1]
    assert isinstance(row, IrisTaskStat)
    assert row.task_id == "/job/0"
    assert row.attempt_id == 3
    assert row.worker_id == "pod-a"
    assert row.cpu_millicores == 750
    assert row.memory_mb == 2048


# ---------------------------------------------------------------------------
# Kueue gang admission: sync() applies one pod-group per gang generation
# ---------------------------------------------------------------------------


def test_coscheduled_gang_pods_share_pod_group_name(kueue_provider, k8s):
    """sync() applies one Kueue pod-group-name across all sibling pods of a gang,
    each annotated with the full pod-group-total-count."""
    reqs = [
        make_run_req(f"/gang/task/{i}", attempt_id=0, num_tasks=4, coscheduling_group_by="leafgroup") for i in range(4)
    ]
    kueue_provider.sync(make_batch(tasks_to_run=reqs))

    pods = k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
    assert len(pods) == 4
    group_names = {p["metadata"]["labels"][_KUEUE_POD_GROUP_NAME] for p in pods}
    assert len(group_names) == 1, "all siblings must share one pod-group-name"
    for p in pods:
        assert p["metadata"]["annotations"][_KUEUE_POD_GROUP_TOTAL] == "4"


def test_coscheduled_sibling_failure_bumps_pod_group_generation(kueue_provider, k8s):
    """A full-gang requeue (new attempt) yields a fresh pod-group-name so Kueue
    forms a new Workload and re-admits the gang atomically."""
    gen0 = [
        make_run_req(f"/run/task/{i}", attempt_id=0, num_tasks=2, coscheduling_group_by="leafgroup") for i in range(2)
    ]
    kueue_provider.sync(make_batch(tasks_to_run=gen0))
    gen0_names = {
        p["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]
        for p in k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
    }
    assert len(gen0_names) == 1

    # Requeue: every sibling moves to the next attempt in lockstep.
    gen1 = [
        make_run_req(f"/run/task/{i}", attempt_id=1, num_tasks=2, coscheduling_group_by="leafgroup") for i in range(2)
    ]
    kueue_provider.sync(make_batch(tasks_to_run=gen1))
    gen1_names = {
        p["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]
        for p in k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
        if p["metadata"]["labels"][_LABEL_ATTEMPT_ID] == "1"
    }
    assert len(gen1_names) == 1
    assert gen0_names.isdisjoint(gen1_names), "new generation must use a fresh pod-group-name"


def test_gang_teardown_deletes_kueue_workload(kueue_provider, k8s):
    """Tearing down a coscheduled gang deletes its Kueue Workload, releasing the
    reserved quota.

    Kueue parks a coscheduled Workload in WaitingForReplacementPods when its
    pods are deleted (it expects replacement pods per the plain-pod-group
    contract), holding the quota until the Workload itself is removed. Without
    deleting it, a gang requeue — which bumps to a fresh pod-group generation —
    would deadlock behind the old generation's still-reserved quota.
    """
    reqs = [
        make_run_req(f"/gang/task/{i}", attempt_id=0, num_tasks=3, coscheduling_group_by="leafgroup") for i in range(3)
    ]
    kueue_provider.sync(make_batch(tasks_to_run=reqs))

    pods = k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
    group_name = pods[0]["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]
    # Kueue names the Workload exactly after the pod-group-name; seed it as the
    # controller would observe it on a live cluster once Kueue admits the gang.
    k8s.seed_resource(K8sResource.WORKLOADS, group_name, {"kind": "Workload", "metadata": {"name": group_name}})

    # Empty desired set: the whole gang is now stray and gets torn down.
    kueue_provider.sync(make_batch())

    assert k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS) == []
    assert k8s.get_json(K8sResource.WORKLOADS, group_name) is None, "stray-pod teardown must release the Kueue Workload"


def test_gang_teardown_strips_kueue_finalizer(kueue_provider, k8s):
    """Tearing down a gang whose pods hold the Kueue pod finalizer removes the
    finalizer so the pod objects actually disappear.

    The fake honors finalizers: a plain delete leaves a finalizer-held pod
    parked with a deletionTimestamp. The pods being fully gone after teardown
    proves the provider stripped the finalizer; without that, Kueue rebuilds
    the pod-group Workload from the surviving labeled pods and re-holds the
    quota/TAS reservation.
    """
    reqs = [
        make_run_req(f"/gang/task/{i}", attempt_id=0, num_tasks=2, coscheduling_group_by="leafgroup") for i in range(2)
    ]
    kueue_provider.sync(make_batch(tasks_to_run=reqs))

    pods = k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)
    assert len(pods) == 2
    pod_names = [p["metadata"]["name"] for p in pods]
    group_name = pods[0]["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]
    # Kueue's webhook stamps its finalizer on every admitted gang pod.
    for pod in pods:
        pod["metadata"]["finalizers"] = [_KUEUE_MANAGED_FINALIZER]
    k8s.seed_resource(K8sResource.WORKLOADS, group_name, {"kind": "Workload", "metadata": {"name": group_name}})

    # Empty desired set: the whole gang is stray.
    kueue_provider.sync(make_batch())

    for name in pod_names:
        assert k8s.get_json(K8sResource.PODS, name) is None, "finalizer-held pod must be fully removed"
    assert k8s.get_json(K8sResource.WORKLOADS, group_name) is None


def _seed_gang_pod(
    k8s,
    name: str,
    pod_group: str,
    created: str,
    *,
    task_hash: str = "feedfacecafebeef",
    finalizers: list[str] | None = None,
    deletion_timestamp: str | None = None,
) -> None:
    """Insert a Failed gang pod (Kueue pod-group label) into the fake k8s store."""
    metadata: dict = {
        "name": name,
        "creationTimestamp": created,
        "labels": {
            _LABEL_MANAGED: "true",
            _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
            _LABEL_TASK_HASH: task_hash,
            _KUEUE_POD_GROUP_NAME: pod_group,
        },
    }
    if finalizers:
        metadata["finalizers"] = finalizers
    if deletion_timestamp:
        metadata["deletionTimestamp"] = deletion_timestamp
    k8s.seed_resource(K8sResource.PODS, name, {"kind": "Pod", "metadata": metadata, "status": {"phase": "Failed"}})


def test_gc_sweeps_finalizer_wedged_gang_pod(provider, k8s):
    """A Failed gang pod wedged in deletion on the Kueue finalizer is swept by GC
    (finalizer stripped, pod removed, Workload deleted) regardless of age."""
    now = datetime.now(UTC)
    recent_ts = (now - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    group = "wedged-gang-group"
    _seed_gang_pod(
        k8s,
        "wedged-gang-pod",
        group,
        recent_ts,
        finalizers=[_KUEUE_MANAGED_FINALIZER],
        deletion_timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    k8s.seed_resource(K8sResource.WORKLOADS, group, {"kind": "Workload", "metadata": {"name": group}})

    provider._gc_terminal_resources(active_pods=[])

    assert k8s.get_json(K8sResource.PODS, "wedged-gang-pod") is None
    assert k8s.get_json(K8sResource.WORKLOADS, group) is None


def test_gc_sweeps_crashed_gang_pods_on_short_retention(provider, k8s):
    """A Failed gang pod older than the gang retention (but younger than the 1h
    plain-pod retention) is swept along with its Workload; a non-gang Failed pod
    of the same age keeps the 1h debugging window."""
    now = datetime.now(UTC)
    age_ts = (now - timedelta(seconds=_GANG_GC_MAX_AGE_SECONDS + 60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    group = "crashed-gang-group"
    _seed_gang_pod(k8s, "crashed-gang-pod", group, age_ts)
    k8s.seed_resource(K8sResource.WORKLOADS, group, {"kind": "Workload", "metadata": {"name": group}})
    _seed_terminal_pod(k8s, "plain-failed-pod", "Failed", "1122334455667788", age_ts)

    provider._gc_terminal_resources(active_pods=[])

    assert k8s.get_json(K8sResource.PODS, "crashed-gang-pod") is None
    assert k8s.get_json(K8sResource.WORKLOADS, group) is None
    assert k8s.get_json(K8sResource.PODS, "plain-failed-pod") is not None, "1h retention for plain pods must hold"


def test_gc_skips_gang_with_active_sibling(provider, k8s):
    """A terminal gang pod past the gang retention is NOT swept while a
    Pending/Running sibling of the same pod group exists: releasing the shared
    Workload would evict the live siblings. Once the gang has no live members,
    the next GC pass sweeps it."""
    now = datetime.now(UTC)
    age_ts = (now - timedelta(seconds=_GANG_GC_MAX_AGE_SECONDS + 60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    group = "skewed-gang-group"
    _seed_gang_pod(k8s, "early-failed-gang-pod", group, age_ts, finalizers=[_KUEUE_MANAGED_FINALIZER])
    k8s.seed_resource(K8sResource.WORKLOADS, group, {"kind": "Workload", "metadata": {"name": group}})
    running_sibling = {
        "kind": "Pod",
        "metadata": {
            "name": "running-gang-pod",
            "labels": {_LABEL_TASK_HASH: "feedfacecafebeef", _KUEUE_POD_GROUP_NAME: group},
        },
        "status": {"phase": "Running"},
    }

    provider._gc_terminal_resources(active_pods=[running_sibling])

    assert k8s.get_json(K8sResource.PODS, "early-failed-gang-pod") is not None, "gang with live sibling must be kept"
    assert k8s.get_json(K8sResource.WORKLOADS, group) is not None, "shared Workload must survive while gang is live"

    provider._gc_terminal_resources(active_pods=[])

    assert k8s.get_json(K8sResource.PODS, "early-failed-gang-pod") is None
    assert k8s.get_json(K8sResource.WORKLOADS, group) is None


# ---------------------------------------------------------------------------
# Preemptible blocker eviction (preempt_namespaces)
# ---------------------------------------------------------------------------

_PREEMPT_NS = "verify-ns"


@pytest.fixture
def preempt_provider(k8s):
    """Kueue provider with blocker eviction enabled for _PREEMPT_NS."""
    p = make_kueue_provider(k8s, preempt_namespaces=[_PREEMPT_NS])
    yield p
    p.close()


def _seed_blocker_pod(
    k8s,
    namespace: str,
    name: str,
    *,
    priority: int = -1,
    gpus: int = 8,
    phase: str = "Running",
) -> None:
    """Insert a health-check-style pod into a foreign namespace of the fake."""
    resources = {"requests": {"nvidia.com/gpu": str(gpus)}} if gpus else {}
    pod = {
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "priority": priority,
            "containers": [{"name": "verify", "resources": resources}],
        },
        "status": {"phase": phase},
    }
    k8s.seed_namespaced_pod(namespace, name, pod)


def _gang_reqs(num_tasks: int = 2) -> list[job_pb2.RunTaskRequest]:
    reqs = []
    for i in range(num_tasks):
        req = make_run_req(f"/gang/task/{i}", attempt_id=0, num_tasks=num_tasks, coscheduling_group_by="leafgroup")
        # Gangs are GPU workloads; the GPU request is what makes blocker eviction fire.
        req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
        reqs.append(req)
    return reqs


def test_gang_submit_evicts_preemptible_gpu_blocker(preempt_provider, k8s):
    """Submitting a gang deletes negative-priority GPU pods in configured namespaces,
    freeing the node capacity Kueue TAS counts against the gang."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "nhc-verify-0")

    preempt_provider.sync(make_batch(tasks_to_run=_gang_reqs()))

    assert k8s.list_pods_in_namespace(_PREEMPT_NS) == []
    # The gang's own pods were still created.
    assert len(k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS)) == 2


def test_gang_submit_spares_non_blocker_pods(preempt_provider, k8s):
    """The hard guards hold regardless of config: normal-priority pods, non-GPU
    pods, and pods in unconfigured namespaces are never deleted."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "normal-priority", priority=0)
    _seed_blocker_pod(k8s, _PREEMPT_NS, "no-gpu", gpus=0)
    _seed_blocker_pod(k8s, "other-ns", "unconfigured-ns")
    _seed_blocker_pod(k8s, _PREEMPT_NS, "real-blocker")

    preempt_provider.sync(make_batch(tasks_to_run=_gang_reqs()))

    survivors = {p["metadata"]["name"] for p in k8s.list_pods_in_namespace(_PREEMPT_NS)}
    assert survivors == {"normal-priority", "no-gpu"}
    assert [p["metadata"]["name"] for p in k8s.list_pods_in_namespace("other-ns")] == ["unconfigured-ns"]


def test_cpu_submit_does_not_evict(preempt_provider, k8s):
    """A CPU-only submission needs no GPU capacity, so it triggers no blocker eviction."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "blocker")

    preempt_provider.sync(make_batch(tasks_to_run=[make_run_req("/plain-job/0")]))

    assert [p["metadata"]["name"] for p in k8s.list_pods_in_namespace(_PREEMPT_NS)] == ["blocker"]


def test_single_pod_gpu_submit_evicts_blocker(preempt_provider, k8s):
    """A non-coscheduled GPU job now routes through Kueue and is gated too, so it also
    frees the GPU capacity blockers hold — eviction is not gated on coscheduling."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "nhc-verify-0")

    req = make_run_req("/gpu-job/0", num_tasks=1)
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
    preempt_provider.sync(make_batch(tasks_to_run=[req]))

    assert k8s.list_pods_in_namespace(_PREEMPT_NS) == []


def test_reconcile_evicts_blockers_while_gang_gated(preempt_provider, k8s):
    """A blocker that lands AFTER gang submission is evicted by the reconcile
    loop while the gang's pods remain SchedulingGated, and the sweep is
    debounced so back-to-back reconciles don't re-list the namespace."""
    entries = [
        RunningTaskEntry(task_id=JobName.from_wire(f"/gang/task/{i}"), attempt_id=0, coscheduled=True) for i in range(2)
    ]
    preempt_provider.sync(make_batch(tasks_to_run=_gang_reqs(), running_tasks=entries))

    # Kueue's webhook gates gang pods until the pod-group Workload is admitted.
    for pod in k8s.list_json(K8sResource.PODS, labels=_MANAGED_POD_LABELS):
        pod["spec"]["schedulingGates"] = [{"name": "kueue.x-k8s.io/admission"}]
        pod["status"] = {"phase": "Pending"}

    # Health-check pod lands after submission, on capacity the gang needs.
    _seed_blocker_pod(k8s, _PREEMPT_NS, "late-blocker")
    preempt_provider._last_preempt_time = 0.0  # clear the submit-time debounce
    preempt_provider.sync(make_batch(running_tasks=entries))
    assert k8s.list_pods_in_namespace(_PREEMPT_NS) == []

    # Debounce: a blocker appearing immediately after survives this cycle.
    _seed_blocker_pod(k8s, _PREEMPT_NS, "back-to-back-blocker")
    preempt_provider.sync(make_batch(running_tasks=entries))
    assert [p["metadata"]["name"] for p in k8s.list_pods_in_namespace(_PREEMPT_NS)] == ["back-to-back-blocker"]


def test_reconcile_without_gated_gang_pods_does_not_evict(preempt_provider, k8s):
    """No gang work waiting on admission -> no eviction, even with a blocker present."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "blocker")
    preempt_provider._last_preempt_time = 0.0

    preempt_provider.sync(make_batch())

    assert [p["metadata"]["name"] for p in k8s.list_pods_in_namespace(_PREEMPT_NS)] == ["blocker"]


def test_preemption_disabled_makes_no_foreign_namespace_calls(kueue_provider, k8s):
    """With preempt_namespaces unset (the default), gang submission never lists
    or deletes pods outside iris's own namespace."""
    _seed_blocker_pod(k8s, _PREEMPT_NS, "blocker")

    kueue_provider.sync(make_batch(tasks_to_run=_gang_reqs()))

    assert k8s.namespaced_pod_calls == []
    assert [p["metadata"]["name"] for p in k8s.list_pods_in_namespace(_PREEMPT_NS)] == ["blocker"]


def test_preemption_never_touches_own_namespace(k8s):
    """Even if misconfigured to include iris's own namespace, eviction skips it."""
    provider = make_kueue_provider(k8s, preempt_namespaces=["iris"])
    try:
        victim = {
            "kind": "Pod",
            "metadata": {"name": "own-ns-victim"},
            "spec": {
                "priority": -1,
                "containers": [{"name": "x", "resources": {"requests": {"nvidia.com/gpu": "8"}}}],
            },
            "status": {"phase": "Running"},
        }
        k8s.seed_resource(K8sResource.PODS, "own-ns-victim", victim)

        provider.sync(make_batch(tasks_to_run=_gang_reqs()))

        assert k8s.get_json(K8sResource.PODS, "own-ns-victim") is not None
    finally:
        provider.close()
