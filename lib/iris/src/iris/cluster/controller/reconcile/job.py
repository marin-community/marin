# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure rules for the job aggregate: state recomputation.

job.py owns the job-aggregate primitives. Cross-aggregate
orchestration that needs to kill tasks (_cascade_to_children, _finalize_terminal_job)
lives in ``batches.py`` because it needs task-aggregate calls. job.py imports
nothing from ``task``, ``worker``, ``peers``, ``batches``.
"""

from iris.cluster.controller.reconcile.effects import JobRowDelta
from iris.cluster.controller.reconcile.policy import ERROR_STATES
from iris.cluster.controller.reconcile.working_state import WorkingState
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
)
from iris.rpc import job_pb2


def recompute_state(state: WorkingState, job_id: JobName) -> int | None:
    """Recompute job state from the prospective task histogram.

    Returns the new state (which may equal current). Returns ``None`` when
    the job basis is not in the snapshot (out-of-slice). Records a job-state
    delta when the state changes.
    """
    basis = state.job_basis(job_id)
    if basis is None:
        return None
    current_state = basis.state
    max_task_failures = basis.max_task_failures
    if current_state in TERMINAL_JOB_STATES:
        return current_state
    counts = basis.task_state_counts
    total = sum(counts.values())
    new_state = current_state
    now = state.now
    if total > 0 and counts.get(job_pb2.TASK_STATE_SUCCEEDED, 0) == total:
        new_state = job_pb2.JOB_STATE_SUCCEEDED
    elif counts.get(job_pb2.TASK_STATE_FAILED, 0) > max_task_failures:
        new_state = job_pb2.JOB_STATE_FAILED
    elif counts.get(job_pb2.TASK_STATE_UNSCHEDULABLE, 0) > 0:
        new_state = job_pb2.JOB_STATE_UNSCHEDULABLE
    elif counts.get(job_pb2.TASK_STATE_KILLED, 0) > 0:
        new_state = job_pb2.JOB_STATE_KILLED
    elif (
        total > 0
        and (
            counts.get(job_pb2.TASK_STATE_WORKER_FAILED, 0)
            + counts.get(job_pb2.TASK_STATE_PREEMPTED, 0)
            + counts.get(job_pb2.TASK_STATE_COSCHED_FAILED, 0)
        )
        > 0
        and all(s in TERMINAL_TASK_STATES for s in counts)
    ):
        new_state = job_pb2.JOB_STATE_WORKER_FAILED
    elif (
        counts.get(job_pb2.TASK_STATE_ASSIGNED, 0) > 0
        or counts.get(job_pb2.TASK_STATE_BUILDING, 0) > 0
        or counts.get(job_pb2.TASK_STATE_RUNNING, 0) > 0
    ):
        new_state = job_pb2.JOB_STATE_RUNNING
    elif basis.started_at is not None:
        new_state = job_pb2.JOB_STATE_RUNNING
    elif total > 0:
        new_state = job_pb2.JOB_STATE_PENDING
    if new_state == current_state:
        return new_state
    error = basis.first_task_error
    state.record_job_state(
        JobRowDelta(
            job_id=job_id,
            state=new_state,
            started_at=now if new_state == job_pb2.JOB_STATE_RUNNING else None,
            finished_at=now if new_state in TERMINAL_JOB_STATES else None,
            error=error if new_state in ERROR_STATES else None,
        )
    )
    return new_state
