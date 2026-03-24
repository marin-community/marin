# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for priority bands, per-user fairness, and scheduling caps."""

from collections import defaultdict


from iris.cluster.controller.budget import BAND_SORT_KEY, PriorityBand
from iris.cluster.controller.controller import _schedulable_tasks
from iris.cluster.controller.db import ControllerDB
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp

from .conftest import (
    make_controller_state,
    make_job_request,
    query_task,
    query_tasks_for_job,
    submit_job,
)


def _set_task_band(db: ControllerDB, task_id: JobName, band: int) -> None:
    """Directly set priority_band on a task row for testing."""
    db.execute("UPDATE tasks SET priority_band = ? WHERE task_id = ?", (band, task_id.to_wire()))


def _submit_user_job(state, user: str, name: str, replicas: int = 1, band: int | None = None) -> list:
    """Submit a job for a specific user, optionally overriding band."""
    req = make_job_request(name=f"/{user}/{name}", cpu=1, replicas=replicas)
    tasks = submit_job(state, f"/{user}/{name}", req)
    if band is not None:
        for t in tasks:
            _set_task_band(state._db, t.task_id, band)
    return tasks


def test_production_scheduled_before_interactive():
    """PRODUCTION band tasks appear before INTERACTIVE in schedulable order."""
    with make_controller_state() as state:
        # Submit interactive tasks first
        interactive_tasks = _submit_user_job(
            state, "alice", "interactive-job", replicas=3, band=BAND_SORT_KEY[PriorityBand.INTERACTIVE]
        )
        # Submit production tasks second
        prod_tasks = _submit_user_job(state, "bob", "prod-job", replicas=2, band=BAND_SORT_KEY[PriorityBand.PRODUCTION])

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
        batch_tasks = _submit_user_job(state, "alice", "batch-job", replicas=2, band=BAND_SORT_KEY[PriorityBand.BATCH])
        interactive_tasks = _submit_user_job(
            state, "bob", "interactive-job", replicas=2, band=BAND_SORT_KEY[PriorityBand.INTERACTIVE]
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
    from iris.cluster.controller.budget import interleave_by_user

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
            user_task_pairs = [(t.task_id.user, t.task_id) for t in band_tasks]
            interleaved.extend(interleave_by_user(user_task_pairs, user_spend))

        # User A (lower spend) should have their task first
        a_task_ids = {t.task_id for t in a_tasks}
        first_task = interleaved[0]
        assert first_task in a_task_ids, f"Expected user-a's task first, got {first_task} (user={first_task.user})"
        # User A's single task should appear in position 0, User B's first in position 1
        assert interleaved[1].user == "user-b"


def test_per_user_cap():
    """max_tasks_per_user_per_cycle limits how many tasks per user are scheduled."""
    from iris.cluster.controller.budget import compute_user_spend, interleave_by_user

    with make_controller_state() as state:
        cap = 3
        # User submits 10 tasks
        _submit_user_job(state, "greedy", "many-tasks", replicas=10)

        schedulable = _schedulable_tasks(state._db)
        with state._db.snapshot() as snap:
            user_spend = compute_user_spend(snap)

        # Interleave
        tasks_by_band: dict[int, list[JobName]] = defaultdict(list)
        for task in schedulable:
            tasks_by_band[task.priority_band].append(task)
        interleaved: list[JobName] = []
        for band_key in sorted(tasks_by_band.keys()):
            band_tasks = tasks_by_band[band_key]
            user_task_pairs = [(t.task_id.user, t.task_id) for t in band_tasks]
            interleaved.extend(interleave_by_user(user_task_pairs, user_spend))

        # Apply cap
        tasks_per_user: dict[str, int] = defaultdict(int)
        capped: list[JobName] = []
        for task_id in interleaved:
            if tasks_per_user[task_id.user] < cap:
                capped.append(task_id)
                tasks_per_user[task_id.user] += 1

        assert len(capped) == cap
        assert all(tid.user == "greedy" for tid in capped)


def test_depth_boost_within_band():
    """Deeper tasks (child jobs) are still prioritized within the same band."""
    with make_controller_state() as state:
        # Submit parent (shallow) job
        parent_id = JobName.root("alice", "parent")
        parent_req = make_job_request(name="/alice/parent", cpu=1, replicas=1)
        parent_tasks = submit_job(state, "/alice/parent", parent_req)

        # Submit child (deeper) job
        child_id = parent_id.child("child")
        child_req = cluster_pb2.Controller.LaunchJobRequest(
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
        parent_req = make_job_request(name="/alice/parent-prod", cpu=1, replicas=1)
        parent_tasks = submit_job(state, "/alice/parent-prod", parent_req)

        # Set parent's band to PRODUCTION
        for t in parent_tasks:
            _set_task_band(state._db, t.task_id, BAND_SORT_KEY[PriorityBand.PRODUCTION])

        # Submit child job
        child_id = parent_id.child("child")
        child_req = cluster_pb2.Controller.LaunchJobRequest(
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
            assert task.priority_band == BAND_SORT_KEY[PriorityBand.PRODUCTION], (
                f"Child task {ct.task_id} has band {task.priority_band}, "
                f"expected {BAND_SORT_KEY[PriorityBand.PRODUCTION]} (PRODUCTION)"
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
        assert row["max_band"] == BAND_SORT_KEY[PriorityBand.INTERACTIVE]  # default


def test_default_band_is_interactive():
    """Tasks submitted without explicit band get INTERACTIVE (band=2)."""
    with make_controller_state() as state:
        tasks = _submit_user_job(state, "alice", "default-band")
        for t in tasks:
            task = query_task(state, t.task_id)
            assert task.priority_band == BAND_SORT_KEY[PriorityBand.INTERACTIVE]
