# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for priority bands, per-user fairness, and scheduling caps."""

from collections import defaultdict

from iris.cluster.controller.budget import UserTask, compute_effective_band, interleave_by_user
from iris.cluster.controller.controller import SchedulingOutcome, _schedulable_tasks
from iris.cluster.controller.schema import TASK_DETAIL_PROJECTION
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Timestamp

from .conftest import (
    inject_device_constraints,
    make_controller_state,
    make_job_request,
    make_worker_metadata,
    query_task,
    query_tasks_for_job,
    submit_job,
)


def _submit_user_job(state, user: str, name: str, replicas: int = 1, band: int | None = None) -> list:
    """Submit a job for a specific user, optionally overriding band."""
    req = make_job_request(name=f"/{user}/{name}", cpu=1, replicas=replicas, priority_band=band or 0)
    return submit_job(state, f"/{user}/{name}", req)


def test_production_scheduled_before_interactive():
    """PRODUCTION band tasks appear before INTERACTIVE in schedulable order."""
    with make_controller_state() as state:
        # Submit interactive tasks first
        interactive_tasks = _submit_user_job(
            state, "alice", "interactive-job", replicas=3, band=job_pb2.PRIORITY_BAND_INTERACTIVE
        )
        # Submit production tasks second
        prod_tasks = _submit_user_job(state, "bob", "prod-job", replicas=2, band=job_pb2.PRIORITY_BAND_PRODUCTION)

        schedulable = _schedulable_tasks(state._db)
        task_ids = [t.task_id for t in schedulable]

        # All production tasks should come before all interactive tasks
        prod_task_ids = {t.task_id for t in prod_tasks}
        interactive_task_ids = {t.task_id for t in interactive_tasks}

        prod_indices = [i for i, tid in enumerate(task_ids) if tid in prod_task_ids]
        interactive_indices = [i for i, tid in enumerate(task_ids) if tid in interactive_task_ids]

        assert prod_indices, "Production tasks should be schedulable"
        assert interactive_indices, "Interactive tasks should be schedulable"
        assert max(prod_indices) < min(interactive_indices), (
            f"All production tasks (indices {prod_indices}) must come before "
            f"interactive tasks (indices {interactive_indices})"
        )


def test_batch_scheduled_after_interactive():
    """BATCH band tasks appear after INTERACTIVE in schedulable order."""
    with make_controller_state() as state:
        batch_tasks = _submit_user_job(state, "alice", "batch-job", replicas=2, band=job_pb2.PRIORITY_BAND_BATCH)
        interactive_tasks = _submit_user_job(
            state, "bob", "interactive-job", replicas=2, band=job_pb2.PRIORITY_BAND_INTERACTIVE
        )

        schedulable = _schedulable_tasks(state._db)
        task_ids = [t.task_id for t in schedulable]

        batch_ids = {t.task_id for t in batch_tasks}
        interactive_ids = {t.task_id for t in interactive_tasks}

        batch_indices = [i for i, tid in enumerate(task_ids) if tid in batch_ids]
        interactive_indices = [i for i, tid in enumerate(task_ids) if tid in interactive_ids]

        assert max(interactive_indices) < min(batch_indices)


def test_single_task_user_beats_hundred_task_user():
    """User A (1 task, lower spend) gets interleaved before User B (many tasks, higher spend).

    When User B has higher budget spend, A's task should come first in the
    interleaved order since interleave_by_user sorts users by ascending spend.
    """
    with make_controller_state() as state:
        # User B submits 10 tasks first
        _submit_user_job(state, "user-b", "big-batch", replicas=10)
        # User A submits 1 task second
        a_tasks = _submit_user_job(state, "user-a", "small-job", replicas=1)

        schedulable = _schedulable_tasks(state._db)

        # Simulate user-b having higher spend (e.g. from running other tasks)
        user_spend = {"user-b": 5000, "user-a": 0}

        # Group by band and interleave
        tasks_by_band: dict[int, list[JobName]] = defaultdict(list)
        for task in schedulable:
            tasks_by_band[task.priority_band].append(task)

        interleaved: list[JobName] = []
        for band_key in sorted(tasks_by_band.keys()):
            band_tasks = tasks_by_band[band_key]
            user_tasks = [UserTask(user_id=t.task_id.user, task=t.task_id) for t in band_tasks]
            interleaved.extend(interleave_by_user(user_tasks, user_spend))

        # User A (lower spend) should have their task first
        a_task_ids = {t.task_id for t in a_tasks}
        first_task = interleaved[0]
        assert first_task in a_task_ids, f"Expected user-a's task first, got {first_task} (user={first_task.user})"
        # User A's single task should appear in position 0, User B's first in position 1
        assert interleaved[1].user == "user-b"


def test_depth_boost_within_band():
    """Deeper tasks (child jobs) are still prioritized within the same band."""
    with make_controller_state() as state:
        # Submit parent (shallow) job
        parent_id = JobName.root("alice", "parent")
        parent_req = make_job_request(name="/alice/parent", cpu=1, replicas=1)
        parent_tasks = submit_job(state, "/alice/parent", parent_req)

        # Submit child (deeper) job
        child_id = parent_id.child("child")
        child_req = controller_pb2.Controller.LaunchJobRequest(
            name=child_id.to_wire(),
            entrypoint=parent_req.entrypoint,
            resources=parent_req.resources,
            environment=parent_req.environment,
            replicas=1,
        )
        state.submit_job(child_id, child_req, Timestamp.now())
        child_tasks = query_tasks_for_job(state, child_id)

        schedulable = _schedulable_tasks(state._db)
        task_ids = [t.task_id for t in schedulable]

        child_task_ids = {t.task_id for t in child_tasks}
        parent_task_ids = {t.task_id for t in parent_tasks}

        child_indices = [i for i, tid in enumerate(task_ids) if tid in child_task_ids]
        parent_indices = [i for i, tid in enumerate(task_ids) if tid in parent_task_ids]

        # Deeper (child) tasks should come before shallower (parent) tasks
        # because priority_neg_depth is more negative for deeper jobs
        assert child_indices and parent_indices
        assert max(child_indices) < min(parent_indices), (
            f"Child tasks (depth={child_id.depth}, indices={child_indices}) should come "
            f"before parent tasks (depth={parent_id.depth}, indices={parent_indices})"
        )


def test_child_inherits_parent_band():
    """Child job inherits parent's priority band."""
    with make_controller_state() as state:
        # Submit parent as PRODUCTION
        parent_id = JobName.root("alice", "parent-prod")
        parent_req = make_job_request(
            name="/alice/parent-prod", cpu=1, replicas=1, priority_band=job_pb2.PRIORITY_BAND_PRODUCTION
        )
        submit_job(state, "/alice/parent-prod", parent_req)

        # Submit child job
        child_id = parent_id.child("child")
        child_req = controller_pb2.Controller.LaunchJobRequest(
            name=child_id.to_wire(),
            entrypoint=parent_req.entrypoint,
            resources=parent_req.resources,
            environment=parent_req.environment,
            replicas=1,
        )
        state.submit_job(child_id, child_req, Timestamp.now())
        child_tasks = query_tasks_for_job(state, child_id)

        # Child should have inherited PRODUCTION band
        for ct in child_tasks:
            task = query_task(state, ct.task_id)
            assert task.priority_band == job_pb2.PRIORITY_BAND_PRODUCTION, (
                f"Child task {ct.task_id} has band {task.priority_band}, "
                f"expected {job_pb2.PRIORITY_BAND_PRODUCTION} (PRODUCTION)"
            )


def test_user_budget_row_created_on_submit():
    """Submitting a job creates a user_budgets row with defaults."""
    with make_controller_state() as state:
        _submit_user_job(state, "newuser", "first-job")

        row = state._db.fetchone(
            "SELECT budget_limit, max_band FROM user_budgets WHERE user_id = ?",
            ("newuser",),
        )
        assert row is not None, "user_budgets row should be created on first job submission"
        assert row["budget_limit"] == 0  # default unlimited
        assert row["max_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE  # default


def test_default_band_is_interactive():
    """Tasks submitted without explicit band get INTERACTIVE (band=2)."""
    with make_controller_state() as state:
        tasks = _submit_user_job(state, "alice", "default-band")
        for t in tasks:
            task = query_task(state, t.task_id)
            assert task.priority_band == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_user_over_budget_tasks_become_batch():
    """User exceeding budget has INTERACTIVE tasks treated as BATCH in scheduling order."""
    with make_controller_state() as state:
        # Submit interactive tasks for alice (over budget) and bob (within budget)
        alice_tasks = _submit_user_job(state, "alice", "alice-job", replicas=2, band=job_pb2.PRIORITY_BAND_INTERACTIVE)
        bob_tasks = _submit_user_job(state, "bob", "bob-job", replicas=2, band=job_pb2.PRIORITY_BAND_INTERACTIVE)

        schedulable = _schedulable_tasks(state._db)

        # Simulate alice being over budget
        user_spend = {"alice": 10000, "bob": 1000}
        user_budget_limits = {"alice": 5000, "bob": 50000}

        # Compute effective bands — alice's tasks should become BATCH
        tasks_by_band: dict[int, list[JobName]] = defaultdict(list)
        for task in schedulable:
            band = compute_effective_band(task.priority_band, task.task_id.user, user_spend, user_budget_limits)
            tasks_by_band[band].append(task.task_id)

        alice_ids = {t.task_id for t in alice_tasks}
        bob_ids = {t.task_id for t in bob_tasks}

        # Bob's tasks should be INTERACTIVE, alice's should be BATCH
        interactive_ids = set(tasks_by_band.get(job_pb2.PRIORITY_BAND_INTERACTIVE, []))
        batch_ids = set(tasks_by_band.get(job_pb2.PRIORITY_BAND_BATCH, []))
        assert bob_ids <= interactive_ids, "Bob's tasks should remain INTERACTIVE"
        assert alice_ids <= batch_ids, "Alice's tasks should be downgraded to BATCH"


def test_user_within_budget_keeps_interactive():
    """User within budget keeps INTERACTIVE band."""
    with make_controller_state() as state:
        _submit_user_job(state, "alice", "within-budget", replicas=2, band=job_pb2.PRIORITY_BAND_INTERACTIVE)

        schedulable = _schedulable_tasks(state._db)
        user_spend = {"alice": 3000}
        user_budget_limits = {"alice": 50000}

        for task in schedulable:
            band = compute_effective_band(task.priority_band, task.task_id.user, user_spend, user_budget_limits)
            assert band == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_production_never_downgraded_by_budget():
    """PRODUCTION tasks are never downgraded even when user exceeds budget."""
    with make_controller_state() as state:
        _submit_user_job(state, "alice", "prod-job", replicas=1, band=job_pb2.PRIORITY_BAND_PRODUCTION)

        schedulable = _schedulable_tasks(state._db)
        user_spend = {"alice": 999999}
        user_budget_limits = {"alice": 100}

        for task in schedulable:
            band = compute_effective_band(task.priority_band, task.task_id.user, user_spend, user_budget_limits)
            assert band == job_pb2.PRIORITY_BAND_PRODUCTION


def test_zero_budget_means_unlimited():
    """budget_limit=0 means no down-weighting regardless of spend."""
    with make_controller_state() as state:
        _submit_user_job(state, "alice", "unlimited", replicas=1, band=job_pb2.PRIORITY_BAND_INTERACTIVE)

        schedulable = _schedulable_tasks(state._db)
        user_spend = {"alice": 999999}
        user_budget_limits = {"alice": 0}

        for task in schedulable:
            band = compute_effective_band(task.priority_band, task.task_id.user, user_spend, user_budget_limits)
            assert band == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_unplaceable_tasks_do_not_starve_placeable_tasks(make_controller, tmp_path):
    """A user's CPU task is scheduled even when they have many unplaceable TPU tasks.

    Regression test: a per-user input cap (max_tasks_per_user_per_cycle) applied before
    scheduling let unplaceable TPU tasks consume all per-user slots, permanently blocking
    CPU tasks for the same user that had available workers.  The cap must only apply to
    actual assignments, not scheduling candidates.
    """
    OLD_CAP = 8  # historical default — must exceed this many TPU tasks
    ctrl = make_controller(local_state_dir=tmp_path / "local")

    # Submit OLD_CAP+2 unplaceable TPU tasks for alice (no TPU workers will be registered)
    for i in range(OLD_CAP + 2):
        tpu_req = controller_pb2.Controller.LaunchJobRequest(
            name=f"/alice/tpu-job-{i}",
            entrypoint=make_job_request().entrypoint,
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        tpu_req.resources.device.tpu.variant = "v5p-8"
        inject_device_constraints(tpu_req)
        jid = JobName.from_string(f"/alice/tpu-job-{i}")
        ctrl._transitions.submit_job(jid, tpu_req, Timestamp.now())

    # Submit 1 CPU task for alice — this should be placeable on the CPU worker
    cpu_jid = JobName.from_string("/alice/cpu-job")
    cpu_req = make_job_request(name="/alice/cpu-job", cpu=1, replicas=1)
    inject_device_constraints(cpu_req)
    ctrl._transitions.submit_job(cpu_jid, cpu_req, Timestamp.now())

    # Register exactly 1 CPU worker — no TPU workers
    ctrl._transitions.register_or_refresh_worker(
        worker_id=WorkerId("cpu-worker"),
        address="cpu-worker:8080",
        metadata=make_worker_metadata(cpu=4, memory_bytes=8 * 1024**3),
        ts=Timestamp.now(),
    )

    outcome = ctrl._run_scheduling()

    assert outcome == SchedulingOutcome.ASSIGNMENTS_MADE, f"Expected ASSIGNMENTS_MADE, got {outcome}"

    with ctrl._db.snapshot() as q:
        cpu_tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (cpu_jid.to_wire(),))
        )

    assert len(cpu_tasks) == 1
    assert (
        cpu_tasks[0].state == job_pb2.TASK_STATE_ASSIGNED
    ), f"CPU task state={cpu_tasks[0].state}; unplaceable TPU tasks may be blocking it"


def test_submit_with_explicit_band_stores_band():
    """Submitting a job with an explicit priority_band stores it in task rows."""
    with make_controller_state() as state:
        req = make_job_request(name="/alice/batch-job", cpu=1, replicas=2, priority_band=job_pb2.PRIORITY_BAND_BATCH)
        tasks = submit_job(state, "/alice/batch-job", req)

        assert len(tasks) == 2
        for t in tasks:
            task = query_task(state, t.task_id)
            assert (
                task.priority_band == job_pb2.PRIORITY_BAND_BATCH
            ), f"Task {t.task_id} has band {task.priority_band}, expected PRIORITY_BAND_BATCH"
