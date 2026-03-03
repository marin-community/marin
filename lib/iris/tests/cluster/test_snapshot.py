# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller snapshot/restore.

Exercises roundtrip behavior: create state, snapshot it, restore to a fresh
ControllerState, and verify that observable properties are preserved.
"""

import tempfile

import pytest

from iris.cluster.controller.snapshot import (
    SCHEMA_VERSION,
    create_snapshot,
    read_latest_snapshot,
    restore_snapshot,
    write_snapshot,
)
from iris.cluster.controller.state import (
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerTaskAttempt,
    ControllerWorker,
)
from iris.cluster.types import AttributeValue, JobName, WorkerId
from iris.rpc import cluster_pb2, snapshot_pb2
from iris.time_utils import Deadline, Duration, Timestamp


def _make_request(name: str = "test-job", replicas: int = 1) -> cluster_pb2.Controller.LaunchJobRequest:
    return cluster_pb2.Controller.LaunchJobRequest(
        name=name,
        replicas=replicas,
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024 * 1024 * 1024,
        ),
    )


def _make_worker(worker_id: str = "w-1", address: str = "10.0.0.1:10001") -> ControllerWorker:
    metadata = cluster_pb2.WorkerMetadata(
        hostname="host-1",
        cpu_count=8,
        memory_bytes=32 * 1024 * 1024 * 1024,
    )
    return ControllerWorker(
        worker_id=WorkerId(worker_id),
        address=address,
        metadata=metadata,
        last_heartbeat=Timestamp.now(),
        attributes={"region": AttributeValue("us-central1")},
    )


def _populated_state() -> ControllerState:
    """Build a ControllerState with jobs in various states for testing."""
    state = ControllerState()

    # A pending job with 2 tasks
    job1_id = JobName.from_string("/alice/pending-job")
    job1 = ControllerJob(
        job_id=job1_id,
        request=_make_request("pending-job", replicas=2),
        submitted_at=Timestamp.from_ms(1000),
    )
    state.add_job(job1)

    # A running job with 1 task assigned to a worker
    worker = _make_worker("w-1")
    state.add_worker(worker)

    job2_id = JobName.from_string("/alice/running-job")
    job2 = ControllerJob(
        job_id=job2_id,
        request=_make_request("running-job"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at=Timestamp.from_ms(2000),
        started_at=Timestamp.from_ms(2100),
    )
    task = ControllerTask(
        task_id=job2_id.task(0),
        job_id=job2_id,
        state=cluster_pb2.TASK_STATE_RUNNING,
        started_at=Timestamp.from_ms(2100),
        submitted_at=Timestamp.from_ms(2000),
        attempts=[
            ControllerTaskAttempt(
                attempt_id=0,
                worker_id=WorkerId("w-1"),
                state=cluster_pb2.TASK_STATE_RUNNING,
                created_at=Timestamp.from_ms(2050),
                started_at=Timestamp.from_ms(2100),
            ),
        ],
    )
    state.add_job(job2, [task])

    # Assign the running task to the worker so running_tasks is populated
    worker.assign_task(task.task_id, job2.request.resources)

    return state


def test_snapshot_roundtrip_preserves_jobs():
    state = _populated_state()
    result = create_snapshot(state)

    assert result.job_count == 2
    assert result.task_count == 3  # 2 from pending-job + 1 from running-job

    # Restore into a fresh state
    fresh_state = ControllerState()
    restore_result = restore_snapshot(result.proto, fresh_state)

    assert restore_result.job_count == 2
    assert restore_result.task_count == 3

    # Verify jobs exist with correct state
    pending = fresh_state.get_job(JobName.from_string("/alice/pending-job"))
    assert pending is not None
    assert pending.state == cluster_pb2.JOB_STATE_PENDING
    assert pending.num_tasks == 2

    running = fresh_state.get_job(JobName.from_string("/alice/running-job"))
    assert running is not None
    assert running.state == cluster_pb2.JOB_STATE_RUNNING

    # Verify tasks are restored
    pending_tasks = fresh_state.get_job_tasks(JobName.from_string("/alice/pending-job"))
    assert len(pending_tasks) == 2
    assert all(t.state == cluster_pb2.TASK_STATE_PENDING for t in pending_tasks)

    running_tasks = fresh_state.get_job_tasks(JobName.from_string("/alice/running-job"))
    assert len(running_tasks) == 1
    assert running_tasks[0].state == cluster_pb2.TASK_STATE_RUNNING
    assert len(running_tasks[0].attempts) == 1


def test_snapshot_roundtrip_preserves_workers():
    state = ControllerState()
    worker = _make_worker("w-42", "10.0.0.42:10001")
    worker.consecutive_failures = 5
    worker.healthy = False
    state.add_worker(worker)

    result = create_snapshot(state)
    assert result.worker_count == 1

    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    restored_worker = fresh_state.get_worker(WorkerId("w-42"))
    assert restored_worker is not None

    # Identity preserved
    assert restored_worker.worker_id == WorkerId("w-42")
    assert restored_worker.address == "10.0.0.42:10001"
    assert restored_worker.attributes["region"].value == "us-central1"

    # Health reset to fresh
    assert restored_worker.healthy is True
    assert restored_worker.consecutive_failures == 0
    assert restored_worker.last_heartbeat.epoch_ms() > 0


def test_restore_rebuilds_running_tasks_from_task_state():
    """Worker.running_tasks and committed resources are rebuilt from task state."""
    state = _populated_state()
    result = create_snapshot(state)

    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    worker = fresh_state.get_worker(WorkerId("w-1"))
    assert worker is not None

    # The running task should be in the worker's running_tasks
    running_task_id = JobName.from_string("/alice/running-job/0")
    assert running_task_id in worker.running_tasks

    # Committed resources should be rebuilt
    assert worker.committed_cpu_millicores == 1000
    assert worker.committed_mem == 1024 * 1024 * 1024


def test_restore_converts_wall_clock_deadlines():
    """Scheduling deadlines stored as wall-clock epoch_ms are converted back to monotonic."""
    state = ControllerState()
    job_id = JobName.from_string("/alice/deadline-job")
    job = ControllerJob(
        job_id=job_id,
        request=_make_request("deadline-job"),
        submitted_at=Timestamp.from_ms(1000),
    )
    # Set a deadline 60 seconds from now
    job.scheduling_deadline = Deadline.from_now(Duration.from_seconds(60.0))
    state.add_job(job)

    result = create_snapshot(state)

    # The proto should have a non-zero deadline
    job_snap = result.proto.jobs[0]
    assert job_snap.scheduling_deadline_epoch_ms > 0

    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    restored_job = fresh_state.get_job(job_id)
    assert restored_job is not None
    assert restored_job.scheduling_deadline is not None
    # The restored deadline should not be expired yet (we set 60s, roundtrip is fast)
    assert not restored_job.scheduling_deadline.expired()


def test_snapshot_write_read_roundtrip():
    """Write snapshot to local FS, read back, verify contents match."""
    state = _populated_state()
    result = create_snapshot(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_prefix = f"{tmpdir}/bundles"
        path = write_snapshot(result.proto, storage_prefix)
        assert "snapshot-" in path

        loaded = read_latest_snapshot(storage_prefix)
        assert loaded is not None
        assert loaded.schema_version == SCHEMA_VERSION
        assert len(loaded.jobs) == 2
        assert len(loaded.workers) == 1

        # Full restore from loaded snapshot should work
        fresh_state = ControllerState()
        restore_result = restore_snapshot(loaded, fresh_state)
        assert restore_result.job_count == 2


def test_schema_version_mismatch_raises():
    proto = snapshot_pb2.ControllerSnapshot(schema_version=999)
    proto.created_at.CopyFrom(Timestamp.now().to_proto())
    state = ControllerState()

    with pytest.raises(ValueError, match="Incompatible snapshot schema version"):
        restore_snapshot(proto, state)


def test_snapshot_preserves_endpoints():
    state = _populated_state()
    endpoint = ControllerEndpoint(
        endpoint_id="ep-1",
        name="/alice/running-job/my-actor",
        address="10.0.0.1:8080",
        job_id=JobName.from_string("/alice/running-job"),
        metadata={"role": "primary"},
        registered_at=Timestamp.from_ms(3000),
    )
    state.add_endpoint(endpoint)

    result = create_snapshot(state)
    fresh_state = ControllerState()
    restore_result = restore_snapshot(result.proto, fresh_state)

    assert restore_result.endpoint_count == 1
    endpoints = fresh_state.list_all_endpoints()
    assert len(endpoints) == 1
    assert endpoints[0].endpoint_id == "ep-1"
    assert endpoints[0].name == "/alice/running-job/my-actor"
    assert endpoints[0].metadata == {"role": "primary"}


def test_snapshot_preserves_task_retry_state():
    """Tasks with failure/preemption counts roundtrip correctly."""
    state = ControllerState()
    job_id = JobName.from_string("/alice/retry-job")
    job = ControllerJob(
        job_id=job_id,
        request=_make_request("retry-job", replicas=1),
        submitted_at=Timestamp.from_ms(1000),
    )
    task = ControllerTask(
        task_id=job_id.task(0),
        job_id=job_id,
        state=cluster_pb2.TASK_STATE_WORKER_FAILED,
        failure_count=2,
        preemption_count=3,
        max_retries_failure=5,
        max_retries_preemption=10,
        submitted_at=Timestamp.from_ms(1000),
    )
    state.add_job(job, [task])

    result = create_snapshot(state)
    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    restored_task = fresh_state.get_task(job_id.task(0))
    assert restored_task is not None
    assert restored_task.failure_count == 2
    assert restored_task.preemption_count == 3
    assert restored_task.max_retries_failure == 5
    assert restored_task.max_retries_preemption == 10


def test_snapshot_preserves_job_hierarchy():
    """Parent/child job relationships and root_submitted_at survive restore.

    The snapshot must preserve the hierarchical job naming so that after restore,
    the controller can still enumerate child jobs and priority ordering (via
    root_submitted_at) remains consistent.
    """
    state = ControllerState()

    # Parent job
    parent_id = JobName.from_string("/alice/train")
    parent_job = ControllerJob(
        job_id=parent_id,
        request=_make_request("train", replicas=1),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at=Timestamp.from_ms(1000),
        started_at=Timestamp.from_ms(1100),
    )
    state.add_job(parent_job)

    # Child job inherits root_submitted_at from parent
    child_id = JobName.from_string("/alice/train/eval")
    child_job = ControllerJob(
        job_id=child_id,
        request=_make_request("eval", replicas=2),
        state=cluster_pb2.JOB_STATE_PENDING,
        submitted_at=Timestamp.from_ms(2000),
    )
    state.add_job(child_job)

    # Verify root_submitted_at inheritance happened
    assert state.get_job(child_id).root_submitted_at.epoch_ms() == 1000

    result = create_snapshot(state)
    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    restored_parent = fresh_state.get_job(parent_id)
    restored_child = fresh_state.get_job(child_id)
    assert restored_parent is not None
    assert restored_child is not None

    # Hierarchy preserved: child's parent is recoverable from JobName
    assert restored_child.job_id.parent == parent_id

    # root_submitted_at preserved for both
    assert restored_parent.root_submitted_at.epoch_ms() == 1000
    assert restored_child.root_submitted_at.epoch_ms() == 1000

    # Child tasks exist
    child_tasks = fresh_state.get_job_tasks(child_id)
    assert len(child_tasks) == 2


def test_snapshot_restore_worker_with_tasks_then_worker_disappears():
    """After restore, if a checkpointed worker never re-registers, its tasks become retryable.

    This simulates the "worker dies during restart window" scenario: the snapshot
    includes the worker and its running tasks, but after restore the worker's
    heartbeats will fail (it's dead), eventually triggering WORKER_FAILED and
    task retry.
    """
    state = ControllerState()

    worker = _make_worker("w-dead", "10.0.0.99:10001")
    state.add_worker(worker)

    job_id = JobName.from_string("/alice/will-retry")
    job = ControllerJob(
        job_id=job_id,
        request=_make_request("will-retry"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at=Timestamp.from_ms(1000),
        started_at=Timestamp.from_ms(1100),
    )
    task = ControllerTask(
        task_id=job_id.task(0),
        job_id=job_id,
        state=cluster_pb2.TASK_STATE_RUNNING,
        started_at=Timestamp.from_ms(1100),
        submitted_at=Timestamp.from_ms(1000),
        max_retries_preemption=5,
        attempts=[
            ControllerTaskAttempt(
                attempt_id=0,
                worker_id=WorkerId("w-dead"),
                state=cluster_pb2.TASK_STATE_RUNNING,
                created_at=Timestamp.from_ms(1050),
                started_at=Timestamp.from_ms(1100),
            ),
        ],
    )
    state.add_job(job, [task])
    worker.assign_task(task.task_id, job.request.resources)

    result = create_snapshot(state)
    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    # Worker is restored with fresh health and the task assigned to it
    restored_worker = fresh_state.get_worker(WorkerId("w-dead"))
    assert restored_worker is not None
    assert restored_worker.healthy is True
    assert job_id.task(0) in restored_worker.running_tasks

    # Task is still RUNNING, assigned to the worker
    restored_task = fresh_state.get_task(job_id.task(0))
    assert restored_task.state == cluster_pb2.TASK_STATE_RUNNING
    assert restored_task.max_retries_preemption == 5


def test_snapshot_roundtrip_preserves_reservation_claims():
    """Reservation claims survive a snapshot/restore cycle."""
    from iris.cluster.controller.state import ReservationClaim

    state = _populated_state()
    claims = {
        WorkerId("w-1"): ReservationClaim(job_id="/alice/reserved-job", entry_idx=0),
        WorkerId("w-2"): ReservationClaim(job_id="/alice/reserved-job", entry_idx=1),
    }

    result = create_snapshot(state, reservation_claims=claims)
    assert len(result.proto.reservation_claims) == 2

    # Verify proto fields
    claim_by_worker = {c.worker_id: c for c in result.proto.reservation_claims}
    assert claim_by_worker["w-1"].job_id == "/alice/reserved-job"
    assert claim_by_worker["w-1"].entry_idx == 0
    assert claim_by_worker["w-2"].entry_idx == 1

    # Verify restore (controller.restore_from_snapshot reads these from the proto)
    restored_claims: dict[WorkerId, ReservationClaim] = {}
    for claim_snap in result.proto.reservation_claims:
        wid = WorkerId(claim_snap.worker_id)
        restored_claims[wid] = ReservationClaim(
            job_id=claim_snap.job_id,
            entry_idx=claim_snap.entry_idx,
        )

    assert restored_claims[WorkerId("w-1")] == claims[WorkerId("w-1")]
    assert restored_claims[WorkerId("w-2")] == claims[WorkerId("w-2")]


def test_task_state_counts_not_doubled_after_restore():
    """task_state_counts must reflect actual task states, not double-count.

    This verifies the fix for a bug where _restore_job() pre-populated
    task_state_counts and then state.add_job() incremented them again.
    """
    state = ControllerState()
    job_id = JobName.from_string("/alice/mixed-job")
    job = ControllerJob(
        job_id=job_id,
        request=_make_request("mixed-job", replicas=3),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at=Timestamp.from_ms(1000),
        started_at=Timestamp.from_ms(1100),
    )
    tasks = [
        ControllerTask(
            task_id=job_id.task(0),
            job_id=job_id,
            state=cluster_pb2.TASK_STATE_PENDING,
            submitted_at=Timestamp.from_ms(1000),
        ),
        ControllerTask(
            task_id=job_id.task(1),
            job_id=job_id,
            state=cluster_pb2.TASK_STATE_PENDING,
            submitted_at=Timestamp.from_ms(1000),
        ),
        ControllerTask(
            task_id=job_id.task(2),
            job_id=job_id,
            state=cluster_pb2.TASK_STATE_RUNNING,
            started_at=Timestamp.from_ms(1100),
            submitted_at=Timestamp.from_ms(1000),
        ),
    ]
    state.add_job(job, tasks)

    result = create_snapshot(state)
    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    restored_job = fresh_state.get_job(job_id)
    assert restored_job is not None
    assert restored_job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] == 2
    assert restored_job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] == 1
    assert restored_job.num_tasks == 3


def test_endpoint_task_association_survives_roundtrip():
    """Endpoint-task mapping persisted in the snapshot is restored correctly.

    After restore, _endpoints_by_task should be populated so that
    _remove_endpoints_for_task can find endpoints for a given task.
    """
    state = ControllerState()
    worker = _make_worker("w-1")
    state.add_worker(worker)

    job_id = JobName.from_string("/alice/serve-job")
    job = ControllerJob(
        job_id=job_id,
        request=_make_request("serve-job"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at=Timestamp.from_ms(1000),
        started_at=Timestamp.from_ms(1100),
    )
    task = ControllerTask(
        task_id=job_id.task(0),
        job_id=job_id,
        state=cluster_pb2.TASK_STATE_RUNNING,
        started_at=Timestamp.from_ms(1100),
        submitted_at=Timestamp.from_ms(1000),
    )
    state.add_job(job, [task])

    endpoint = ControllerEndpoint(
        endpoint_id="ep-1",
        name="/alice/serve-job/my-actor",
        address="10.0.0.1:8080",
        job_id=job_id,
        metadata={"role": "primary"},
        registered_at=Timestamp.from_ms(2000),
    )
    state.add_endpoint(endpoint, task_id=task.task_id)

    result = create_snapshot(state)

    # Verify the proto includes the task_id
    assert len(result.proto.endpoints) == 1
    assert result.proto.endpoints[0].task_id == str(task.task_id)

    # Restore into a fresh state
    fresh_state = ControllerState()
    restore_snapshot(result.proto, fresh_state)

    # The endpoint-task mapping should be restored
    ep_task_map = fresh_state.get_endpoint_task_mapping()
    assert "ep-1" in ep_task_map
    assert ep_task_map["ep-1"] == task.task_id
