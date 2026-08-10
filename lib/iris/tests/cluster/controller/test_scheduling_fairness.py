# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for priority bands, per-user fairness, and scheduling caps."""

from iris.cluster.controller import reads
from iris.cluster.controller.budget import (
    compute_user_spend,
)
from iris.cluster.controller.scheduling.policy import (
    _sort_pending_tasks_by_resolved_band,
)
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp

from ._test_support import set_task_state_for_test, submit_job_in_tx
from .conftest import (
    make_controller_state,
    make_job_request,
    query_tasks_for_job,
    submit_job,
)


def _submit_user_job(state, user: str, name: str, replicas: int = 1, band: int | None = None) -> list:
    """Submit a job for a specific user, optionally overriding band."""
    req = make_job_request(name=f"/{user}/{name}", cpu=1, replicas=replicas, priority_band=band or 0)
    return submit_job(state, f"/{user}/{name}", req)


def _pending(state):
    """Test helper: read pending tasks within a fresh snapshot."""
    with state._db.read_snapshot() as tx:
        return reads.pending_tasks_with_jobs(tx)


def _pending_sorted(state):
    """Test helper: pending tasks resorted by resolved priority band."""
    with state._db.read_snapshot() as tx:
        tasks = reads.pending_tasks_with_jobs(tx)
        bands = reads.get_priority_bands(tx, {t.job_id for t in tasks})
    return _sort_pending_tasks_by_resolved_band(tasks, bands)


def test_production_scheduled_before_interactive():
    """PRODUCTION band tasks appear before INTERACTIVE in schedulable order."""
    with make_controller_state() as state:
        # Submit interactive tasks first
        interactive_tasks = _submit_user_job(
            state, "alice", "interactive-job", replicas=3, band=job_pb2.PRIORITY_BAND_INTERACTIVE
        )
        # Submit production tasks second
        prod_tasks = _submit_user_job(state, "bob", "prod-job", replicas=2, band=job_pb2.PRIORITY_BAND_PRODUCTION)

        schedulable = _pending_sorted(state)
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

        schedulable = _pending_sorted(state)
        task_ids = [t.task_id for t in schedulable]

        batch_ids = {t.task_id for t in batch_tasks}
        interactive_ids = {t.task_id for t in interactive_tasks}

        batch_indices = [i for i, tid in enumerate(task_ids) if tid in batch_ids]
        interactive_indices = [i for i, tid in enumerate(task_ids) if tid in interactive_ids]

        assert max(interactive_indices) < min(batch_indices)


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
        with state._db.transaction() as cur:
            submit_job_in_tx(cur, job_id=child_id, request=child_req, ts=Timestamp.now())
        child_tasks = query_tasks_for_job(state, child_id)

        schedulable = _pending(state)
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


def _submit_child(
    state, parent_id: JobName, parent_req, name: str = "child", band: int = job_pb2.PRIORITY_BAND_INHERIT
) -> JobName:
    """Submit a child reusing the parent's shape; INHERIT is what an inheriting child sends."""
    child_id = parent_id.child(name)
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_id.to_wire(),
        entrypoint=parent_req.entrypoint,
        resources=parent_req.resources,
        environment=parent_req.environment,
        replicas=1,
        priority_band=band,
    )
    with state._db.transaction() as cur:
        submit_job_in_tx(cur, job_id=child_id, request=child_req, ts=Timestamp.now())
    return child_id


def _activate_job(state, job_id: JobName) -> None:
    """Move a job's tasks to RUNNING; ``compute_user_spend`` only reads active rows."""
    for row in query_tasks_for_job(state, job_id):
        set_task_state_for_test(state, row.task_id, job_pb2.TASK_STATE_RUNNING)


def test_batch_childs_spend_does_not_count_against_the_budget():
    """A child of a BATCH job is excluded from user spend, like its parent."""
    with make_controller_state() as state:
        parent_id = JobName.root("alice", "parent-batch")
        parent_req = make_job_request(
            name="/alice/parent-batch", cpu=1, replicas=1, priority_band=job_pb2.PRIORITY_BAND_BATCH
        )
        submit_job(state, "/alice/parent-batch", parent_req)
        child_id = _submit_child(state, parent_id, parent_req)
        _activate_job(state, parent_id)
        _activate_job(state, child_id)

        # Bob's INTERACTIVE job is the control: it proves the spend query sees
        # active tasks at all, so alice's absence is exclusion and not an empty read.
        _submit_user_job(state, "bob", "interactive-job", band=job_pb2.PRIORITY_BAND_INTERACTIVE)
        _activate_job(state, JobName.root("bob", "interactive-job"))

        with state._db.read_snapshot() as snap:
            assert compute_user_spend(snap).keys() == {"bob"}
