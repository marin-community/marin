# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KubernetesProvider integration with controller and transitions."""

import threading

from finelog.rpc import logging_pb2
from iris.cluster.controller import ops
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    ProviderUnsupportedError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.schema import tasks_table
from iris.cluster.controller.writes import stamp_backend
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from sqlalchemy import update as sa_update
from tests.cluster.controller.transition_driver import commit_dispatch_updates

from .conftest import (
    make_direct_job_request,
    query_attempt,
    query_task,
    query_tasks_for_job,
    submit_direct_job,
)


class FakeDirectProvider:
    """Minimal cluster-view TaskBackend (K8s-like) for testing."""

    name = "kubernetes"
    capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    autoscaler = None
    health = None

    def __init__(self):
        self.sync_calls: list[ReconcileRequest] = []
        self.sync_result = ReconcileResult()
        self.closed = False
        self.advertised: dict[str, set[str]] = {}
        self.allowed_users: frozenset[str] = frozenset({"*"})

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def admits(self, user: str) -> bool:
        return "*" in self.allowed_users or user in self.allowed_users

    def configure_routing(self, advertised: dict[str, set[str]], allowed_users: frozenset[str]) -> None:
        self.advertised = advertised
        self.allowed_users = allowed_users

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        self.sync_calls.append(request)
        return self.sync_result

    def run_teardown(self) -> None:
        """No-op: a cluster-view backend tracks no Iris workers to reap."""

    def teardown(self, dead_workers, *, reason: str) -> None:
        """No-op: a cluster-view backend tracks no Iris workers to reap."""

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """No-op: a cluster-view backend tracks no Iris workers to garbage-collect."""
        return 0

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return ScheduleResult()

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult()

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        """No-op: a cluster-view backend tracks no Iris workers, so it builds no worker source."""

    def seed_liveness(self) -> None:
        """No-op: a cluster-view backend tracks no Iris worker liveness."""

    def get_process_status(self, target: TaskTarget, request):
        raise ProviderUnsupportedError("fake k8s")

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return [], cursor

    def close(self) -> None:
        self.closed = True


# =============================================================================
# Transition-level tests: drain_for_dispatch
# =============================================================================


def test_drain_pending_creates_attempt_rows(state):
    """Pending tasks are promoted to ASSIGNED with NULL worker_id and an attempt row is created."""
    [task_id] = submit_direct_job(state, "drain-pending")

    task_before = query_task(state, task_id)
    assert task_before.state == job_pb2.TASK_STATE_PENDING

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch.tasks_to_run[0].attempt_id == 0

    task_after = query_task(state, task_id)
    assert task_after.state == job_pb2.TASK_STATE_ASSIGNED
    assert task_after.current_attempt_id == 0

    attempt = query_attempt(state, task_id, 0)
    assert attempt is not None
    assert attempt.worker_id is None


def test_drain_propagates_task_image(state):
    """task_image set on the LaunchJobRequest is copied into RunTaskRequest."""
    [task_id] = submit_direct_job(state, "drain-task-image", task_image="custom/swetrace:dev")

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch.tasks_to_run[0].task_image == "custom/swetrace:dev"


def test_drain_default_task_image_is_empty(state):
    """When the LaunchJobRequest omits task_image, the dispatched RunTaskRequest is empty."""
    submit_direct_job(state, "drain-default-image")

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_image == ""


def test_drain_includes_workdir_files(state):
    """Workdir files stored in job_workdir_files are included in the RunTaskRequest."""

    job_name = JobName.from_wire("/test-user/drain-workdir")
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "_callable_runner.py"]
    entrypoint.workdir_files["_callable_runner.py"] = b"print('hello')"
    req = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=entrypoint,
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=job_name, request=req, ts=Timestamp.now())

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 1
    run_req = batch.tasks_to_run[0]
    assert "_callable_runner.py" in run_req.entrypoint.workdir_files
    assert run_req.entrypoint.workdir_files["_callable_runner.py"] == b"print('hello')"


def test_drain_redrives_assigned_null_worker(state):
    """ASSIGNED+null-worker rows are redriven into ``tasks_to_run`` on each
    cycle (idempotent ``kubectl apply``), so a controller crash between the
    promote-commit and the pod-apply still recovers. They are *also* in
    ``running_tasks`` so the same-cycle poll observes the freshly-applied
    pod's phase and transitions the row out of ASSIGNED."""
    [task_id] = submit_direct_job(state, "drain-redrive")

    # First drain promotes PENDING -> ASSIGNED, builds a RunTaskRequest, and
    # also includes the row in running_tasks so the post-apply poll picks up
    # the new pod's phase on the same cycle.
    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur)
    assert len(batch1.tasks_to_run) == 1
    assert batch1.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch1.tasks_to_run[0].attempt_id == 0
    assert [(e.task_id, e.attempt_id) for e in batch1.running_tasks] == [(task_id, 0)]

    # Second drain (simulates a crash between assign-commit and provider.sync,
    # or a transient apply failure): task is still ASSIGNED+null-worker, so it
    # is redriven in tasks_to_run with the same attempt_id and stays in
    # running_tasks.
    with state._db.transaction() as cur:
        batch2 = dispatch.drain_for_dispatch(cur)
    assert len(batch2.tasks_to_run) == 1
    assert batch2.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch2.tasks_to_run[0].attempt_id == 0
    assert [(e.task_id, e.attempt_id) for e in batch2.running_tasks] == [(task_id, 0)]


def test_drain_scopes_running_tasks_to_backend(state):
    """A CLUSTER_VIEW backend's drain scopes ``running_tasks`` (the poll set) to
    its own backend_id. Without it two K8s backends each poll the other's
    running pods and, after the pod-not-found grace, mark them FAILED."""
    [task_a] = submit_direct_job(state, "backend-a")
    submit_direct_job(state, "backend-b")  # the other backend's task must not leak into a's poll set
    with state._db.transaction() as cur:
        stamp_backend(
            cur,
            [
                (JobName.root("test-user", "backend-a"), "a"),
                (JobName.root("test-user", "backend-b"), "b"),
            ],
        )

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur, backend_id="a")

    assert [r.task_id for r in batch.tasks_to_run] == [task_a.to_wire()]
    assert [e.task_id for e in batch.running_tasks] == [task_a]


def test_drain_executing_goes_to_running_tasks(state):
    """BUILDING/RUNNING rows with null worker land in running_tasks (poll set),
    not tasks_to_run."""
    [task_id] = submit_direct_job(state, "drain-running")

    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur)
    attempt_id = batch1.tasks_to_run[0].attempt_id

    # Provider reports the pod has reached RUNNING.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING)],
            now=Timestamp.now(),
        )

    with state._db.transaction() as cur:
        batch2 = dispatch.drain_for_dispatch(cur)

    assert len(batch2.tasks_to_run) == 0
    assert len(batch2.running_tasks) == 1
    assert batch2.running_tasks[0].task_id == task_id
    assert batch2.running_tasks[0].attempt_id == attempt_id


# =============================================================================
# Transition-level tests: apply_dispatch_updates
# =============================================================================


def test_apply_running(state):
    """ASSIGNED -> RUNNING via direct provider update."""
    [task_id] = submit_direct_job(state, "apply-running")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_RUNNING


def test_apply_succeeded(state):
    """RUNNING -> SUCCEEDED via direct provider update."""
    [task_id] = submit_direct_job(state, "apply-succeeded")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    # First move to RUNNING.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )

    # Then to SUCCEEDED.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_SUCCEEDED),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert task.exit_code == 0


def test_apply_failed_with_retry(state):
    """FAILED with retries remaining returns task to PENDING."""
    jid = JobName.root("test-user", "retry-job")
    req = make_direct_job_request("retry-job")
    req.max_retries_failure = 2
    req.max_task_failures = 2
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now())
    task_id = query_tasks_for_job(state, jid)[0].task_id

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    # Should be back to PENDING because failure_count(1) <= max_retries_failure(2).
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.failure_count == 1


def test_apply_failed_no_retry(state):
    """FAILED with no retries remaining stays terminal."""
    jid = JobName.root("test-user", "no-retry-job")
    req = make_direct_job_request("no-retry-job")
    req.max_retries_failure = 0
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now())
    task_id = query_tasks_for_job(state, jid)[0].task_id

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED, error="fatal"),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.failure_count == 1


def test_apply_failed_directly_from_assigned(state):
    """ASSIGNED -> FAILED without going through RUNNING (e.g. ConfigMap too large)."""
    [task_id] = submit_direct_job(state, "fail-on-apply")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    new_state=job_pb2.TASK_STATE_FAILED,
                    error="kubectl apply failed: RequestEntityTooLarge",
                ),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.failure_count == 1


def test_apply_worker_failed_from_running_retries(state):
    """WORKER_FAILED from RUNNING with retries remaining returns to PENDING."""
    jid = JobName.root("test-user", "wf-retry")
    req = make_direct_job_request("wf-retry")
    req.max_retries_preemption = 5
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now())
    task_id = query_tasks_for_job(state, jid)[0].task_id

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_WORKER_FAILED),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 1


def test_apply_worker_failed_from_assigned(state):
    """WORKER_FAILED from ASSIGNED returns to PENDING without incrementing preemption_count."""
    [task_id] = submit_direct_job(state, "wf-assigned")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Task is ASSIGNED after drain (not yet RUNNING).
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_WORKER_FAILED),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 0


# =============================================================================
# Controller-level tests
# =============================================================================


def test_drain_multiple_tasks(state):
    """Multiple pending tasks are all promoted in a single drain call."""
    task_ids = submit_direct_job(state, "multi-task", replicas=3)
    assert len(task_ids) == 3

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    assert len(batch.tasks_to_run) == 3

    promoted_ids = {req.task_id for req in batch.tasks_to_run}
    expected_ids = {tid.to_wire() for tid in task_ids}
    assert promoted_ids == expected_ids


def test_apply_ignores_stale_attempt(state):
    """Updates with a mismatched attempt_id are silently skipped."""
    [task_id] = submit_direct_job(state, "stale-attempt")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Apply with wrong attempt_id.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id + 99, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    # Should still be ASSIGNED (the update was skipped).
    assert task.state == job_pb2.TASK_STATE_ASSIGNED


# =============================================================================
# Gang-atomic promotion (Kueue coscheduled jobs)
# =============================================================================

_GROUP = "tpu-name"


def _submit_cosched(state, name, replicas, *, max_retries_preemption=0, band=0):
    """Submit a coscheduled direct job and return its task ids."""
    jid = JobName.root("test-user", name)
    req = make_direct_job_request(name, replicas=replicas, coscheduling_group_by=_GROUP, priority_band=band)
    req.max_retries_preemption = max_retries_preemption
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now())
    return jid, [t.task_id for t in query_tasks_for_job(state, jid)]


def _states(state, task_ids):
    return [query_task(state, t).state for t in task_ids]


def test_drain_promotes_coscheduled_gang_atomically(state):
    """A coscheduled gang is promoted whole in one drain; every RunTaskRequest carries
    the same attempt_id (the pod-group generation) and the coscheduling + priority fields."""
    _jid, task_ids = _submit_cosched(state, "gang", replicas=4, band=job_pb2.PRIORITY_BAND_BATCH)

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 4
    assert {r.task_id for r in batch.tasks_to_run} == {t.to_wire() for t in task_ids}
    assert {r.attempt_id for r in batch.tasks_to_run} == {0}, "siblings must share the generation"
    for r in batch.tasks_to_run:
        assert r.HasField("coscheduling")
        assert r.coscheduling.group_by == _GROUP
        assert r.priority == job_pb2.PRIORITY_BAND_BATCH
    assert all(s == job_pb2.TASK_STATE_ASSIGNED for s in _states(state, task_ids))


def test_drain_unprioritized_gang_defaults_to_interactive(state):
    """A coscheduled gang submitted without an explicit priority drains at the EFFECTIVE
    INTERACTIVE band. UNSPECIFIED is normalized to INTERACTIVE at submit and persisted in
    tasks.priority_band (the column the dispatch query reads), so the Kueue path can stamp a
    real WorkloadPriorityClass instead of dropping to Kueue's cluster default."""
    _submit_cosched(state, "gang-default-prio", replicas=3)  # band defaults to UNSPECIFIED

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 3
    assert {r.priority for r in batch.tasks_to_run} == {job_pb2.PRIORITY_BAND_INTERACTIVE}


def test_drain_oversized_gang_promoted_whole_despite_cap(state):
    """A gang larger than the per-cycle cap is still promoted whole (the cap only bounds
    API-server pressure; a partial gang would deadlock Kueue)."""
    _jid, task_ids = _submit_cosched(state, "big-gang", replicas=5)

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur, max_promotions=2)

    assert len(batch.tasks_to_run) == 5
    assert all(s == job_pb2.TASK_STATE_ASSIGNED for s in _states(state, task_ids))


def test_drain_defers_gang_over_remaining_budget(state):
    """When a gang fits the per-cycle cap but not the remaining budget, it is deferred whole
    rather than split. The next cycle promotes it."""
    _a, a_tasks = _submit_cosched(state, "gang-a", replicas=3)
    _b, b_tasks = _submit_cosched(state, "gang-b", replicas=3)

    # Cap = 4: one gang of 3 fits, the second (3 > remaining 1, 3 <= 4 cap) is deferred.
    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur, max_promotions=4)
    assert len(batch1.tasks_to_run) == 3

    all_states = _states(state, a_tasks) + _states(state, b_tasks)
    assert all_states.count(job_pb2.TASK_STATE_ASSIGNED) == 3
    assert all_states.count(job_pb2.TASK_STATE_PENDING) == 3

    # Next cycle: deferred gang promoted (the already-ASSIGNED gang is redriven, not re-promoted).
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur, max_promotions=4)
    after = _states(state, a_tasks) + _states(state, b_tasks)
    assert all(s == job_pb2.TASK_STATE_ASSIGNED for s in after)


def test_drain_does_not_promote_partial_gang(state):
    """A gang is promoted only when every sibling is PENDING together; a lone PENDING
    sibling (siblings still in flight) is held until the gang reconverges."""
    _jid, task_ids = _submit_cosched(state, "partial", replicas=3)
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)  # all -> ASSIGNED @0

    # Force a partial state: one sibling back to PENDING, two still ASSIGNED.
    with state._db.transaction() as cur:
        cur.execute(
            sa_update(tasks_table).where(tasks_table.c.task_id == task_ids[0]).values(state=job_pb2.TASK_STATE_PENDING)
        )

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    # The lone PENDING sibling must NOT be promoted (still attempt 0, still PENDING).
    promoted_to_attempt1 = [r for r in batch.tasks_to_run if r.attempt_id == 1]
    assert promoted_to_attempt1 == []
    assert query_task(state, task_ids[0]).state == job_pb2.TASK_STATE_PENDING
    assert query_task(state, task_ids[0]).current_attempt_id == 0


def test_coscheduled_gang_requeue_keeps_siblings_in_lockstep(state):
    """End-to-end lockstep invariant: a transient failure bounces the whole gang to PENDING,
    and the next drain re-promotes every sibling to the SAME next attempt_id — which is what
    keeps the per-generation pod-group-name uniform across the gang."""
    _jid, task_ids = _submit_cosched(state, "lockstep", replicas=3, max_retries_preemption=5)

    with state._db.transaction() as cur:
        batch0 = dispatch.drain_for_dispatch(cur)
    assert {r.attempt_id for r in batch0.tasks_to_run} == {0}

    # All siblings reach RUNNING.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=t, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING) for t in task_ids],
            now=Timestamp.now(),
        )

    # One sibling hits a transient (preemption) failure -> whole gang bounced to PENDING.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=task_ids[0], attempt_id=0, new_state=job_pb2.TASK_STATE_WORKER_FAILED)],
            now=Timestamp.now(),
        )
    assert all(s == job_pb2.TASK_STATE_PENDING for s in _states(state, task_ids))

    # Re-drain: the entire gang re-promotes to attempt 1 in lockstep.
    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur)
    assert len(batch1.tasks_to_run) == 3
    assert {r.attempt_id for r in batch1.tasks_to_run} == {1}, "all siblings share the new generation"
    assert all(r.coscheduling.group_by == _GROUP for r in batch1.tasks_to_run)


def test_gang_requeue_bounces_assigned_sibling_off_old_generation(state):
    """A still-ASSIGNED (pod not yet landed / redrive-pending) sibling is bounced to
    PENDING when another sibling fails, so the next drain re-promotes the WHOLE gang on
    one new attempt_id.

    Guards against a mixed-generation gang: if the ASSIGNED sibling stayed on attempt 0
    (redriven on the old pod-group-name) while its siblings advanced to attempt 1, Kueue
    would see two partial Workloads and never admit either. The fix hinges on ASSIGNED
    being an active state, so the requeue cascade catches the not-yet-running sibling.
    """
    _jid, task_ids = _submit_cosched(state, "assigned-bounce", replicas=3, max_retries_preemption=5)

    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)  # all -> ASSIGNED @0
    assert all(s == job_pb2.TASK_STATE_ASSIGNED for s in _states(state, task_ids))

    # Two siblings reach RUNNING; task_ids[0] stays ASSIGNED+null-worker (its pod has
    # not landed yet — it is a redrive candidate this whole time).
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=t, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING) for t in task_ids[1:]],
            now=Timestamp.now(),
        )
    assert query_task(state, task_ids[0]).state == job_pb2.TASK_STATE_ASSIGNED

    # A running sibling hits a transient failure -> the whole gang, including the
    # still-ASSIGNED sibling, must bounce to PENDING.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=task_ids[1], attempt_id=0, new_state=job_pb2.TASK_STATE_WORKER_FAILED)],
            now=Timestamp.now(),
        )
    assert all(
        s == job_pb2.TASK_STATE_PENDING for s in _states(state, task_ids)
    ), "the ASSIGNED sibling must not be stranded on the old generation"

    # Re-drain: every sibling re-promotes to attempt 1 together; nothing is redriven on
    # attempt 0 (which would mean a split pod-group generation).
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    assert {r.task_id for r in batch.tasks_to_run} == {t.to_wire() for t in task_ids}
    assert {r.attempt_id for r in batch.tasks_to_run} == {1}, "no sibling left on the old pod-group generation"


def test_drain_gang_and_noncoscheduled_coexist(state):
    """A coscheduled gang promotes whole; non-coscheduled tasks fill the remaining budget."""
    _jid, gang_tasks = _submit_cosched(state, "mixed-gang", replicas=2)
    single = submit_direct_job(state, "mixed-single")

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    promoted = {r.task_id for r in batch.tasks_to_run}
    assert {t.to_wire() for t in gang_tasks} <= promoted
    assert single[0].to_wire() in promoted


def test_apply_ignores_finished_task(state):
    """Updates to already-finished tasks are silently skipped."""
    [task_id] = submit_direct_job(state, "finished-task")
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    attempt_id = batch.tasks_to_run[0].attempt_id

    # Move to SUCCEEDED.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_SUCCEEDED),
            ],
            now=Timestamp.now(),
        )

    # Try to move to FAILED after already succeeded.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_FAILED),
            ],
            now=Timestamp.now(),
        )

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
