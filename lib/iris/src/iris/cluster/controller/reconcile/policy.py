# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constants and predicate sets shared across the reconcile kernel."""

from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.resources.state import JobState, TaskState
from iris.cluster.types import TERMINAL_JOB_STATES

# ---------------------------------------------------------------------------
# Limits and well-known names
# ---------------------------------------------------------------------------

MAX_REPLICAS_PER_JOB = 10000
"""Maximum replicas allowed per job to prevent resource exhaustion."""

MAX_ACTIVE_TASKS_PER_USER = 16000
"""Maximum non-terminal tasks a single user may have admitted at once.

A submission that would push a user past this is rejected up front, before its
tasks are materialized, to keep one user's burst from OOMing the controller. A
launcher job that admits tasks gradually stays under the cap as earlier tasks
finish and free budget. Raise as controller capacity improves (#6411)."""

# ---------------------------------------------------------------------------
# Predicate sets
# ---------------------------------------------------------------------------

# Failure states that trigger coscheduled sibling cascades. Also reused to pick
# the cascade *direction* from a transition's resolved task state: a member here
# tears the gang down, a non-member (PENDING) requeues it.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        TaskState.FAILED,
        TaskState.WORKER_FAILED,
        TaskState.PREEMPTED,
    }
)

# Worker-reported states that, on a coscheduled task, must fan out to the gang.
# KILLED joins the failure states because a worker only reports it on an
# out-of-band container stop (slice reclaimed for a higher-priority job, node
# drain, spot/preemptible reclaim) — an infra event the whole gang must react to,
# exactly like WORKER_FAILED. This gates *whether* to cascade; the resolved task
# state (via FAILURE_TASK_STATES) still decides requeue-vs-terminate downstream,
# so KILLED is deliberately kept out of FAILURE_TASK_STATES itself.
PEER_CASCADE_TRIGGER_STATES: frozenset[int] = FAILURE_TASK_STATES | {TaskState.KILLED}

# Non-terminal task states (ACTIVE plus PENDING). Used as the snapshot's
# per-job ``active_tasks_by_job`` state filter so a single read covers both
# ACTIVE-only and NON_TERMINAL readers.
NON_TERMINAL_TASK_STATES: frozenset[int] = ACTIVE_TASK_STATES | {TaskState.PENDING}

# Cancel intentionally overwrites the transient WORKER_FAILED terminal so
# operator intent is recorded; other real terminals are still protected.
CANCEL_GUARD_STATES: frozenset[int] = frozenset(TERMINAL_JOB_STATES - {JobState.WORKER_FAILED})

# Job states that warrant recording an error message on finished_at_ms.
ERROR_STATES: frozenset[int] = frozenset(
    [
        JobState.FAILED,
        JobState.KILLED,
        JobState.UNSCHEDULABLE,
        JobState.WORKER_FAILED,
    ]
)

TERMINAL_STATE_REASONS: dict[int, str] = {
    JobState.FAILED: "Job exceeded max_task_failures",
    JobState.KILLED: "Job was terminated.",
    JobState.UNSCHEDULABLE: "Job could not be scheduled.",
    JobState.WORKER_FAILED: "Worker failed",
}
