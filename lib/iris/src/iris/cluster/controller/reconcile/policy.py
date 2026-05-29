# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constants and predicate sets shared across the reconcile kernel."""

from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.types import TERMINAL_JOB_STATES
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Limits and well-known names
# ---------------------------------------------------------------------------

MAX_REPLICAS_PER_JOB = 10000
"""Maximum replicas allowed per job to prevent resource exhaustion."""

DEFAULT_MAX_RETRIES_PREEMPTION = 100
"""Default preemption retries. High because worker failures are typically transient."""

RESERVATION_HOLDER_JOB_NAME = ":reservation:"
"""Well-known name component for synthetic reservation holder child jobs.

Uses colons to clearly distinguish from user-created jobs and avoid
accidental collision with normal job names."""


# ---------------------------------------------------------------------------
# Predicate sets
# ---------------------------------------------------------------------------

# Failure states that trigger coscheduled sibling cascades.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)

# Non-terminal task states (ACTIVE plus PENDING). Used as the snapshot's
# per-job ``active_tasks_by_job`` state filter so a single read covers both
# ACTIVE-only and NON_TERMINAL readers.
NON_TERMINAL_TASK_STATES: frozenset[int] = ACTIVE_TASK_STATES | {job_pb2.TASK_STATE_PENDING}

# Cancel intentionally overwrites the transient WORKER_FAILED terminal so
# operator intent is recorded; other real terminals are still protected.
CANCEL_GUARD_STATES: frozenset[int] = frozenset(TERMINAL_JOB_STATES - {job_pb2.JOB_STATE_WORKER_FAILED})

# Job states that warrant recording an error message on finished_at_ms.
ERROR_STATES: frozenset[int] = frozenset(
    [
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_UNSCHEDULABLE,
        job_pb2.JOB_STATE_WORKER_FAILED,
    ]
)

TERMINAL_STATE_REASONS: dict[int, str] = {
    job_pb2.JOB_STATE_FAILED: "Job exceeded max_task_failures",
    job_pb2.JOB_STATE_KILLED: "Job was terminated.",
    job_pb2.JOB_STATE_UNSCHEDULABLE: "Job could not be scheduled.",
    job_pb2.JOB_STATE_WORKER_FAILED: "Worker failed",
}
