# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource lifecycle values independent of generated RPC modules."""

from enum import IntEnum, StrEnum


class JobState(IntEnum):
    UNSPECIFIED = 0
    PENDING = 1
    BUILDING = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    KILLED = 6
    WORKER_FAILED = 7
    UNSCHEDULABLE = 8


class TaskState(IntEnum):
    UNSPECIFIED = 0
    PENDING = 1
    BUILDING = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    KILLED = 6
    WORKER_FAILED = 7
    UNSCHEDULABLE = 8
    ASSIGNED = 9
    PREEMPTED = 10
    COSCHED_FAILED = 11
    MISSING = 12


class FederationState(StrEnum):
    """Federation state exposed by the legacy workload snapshot."""

    LOCAL = "local"
    PENDING = "pending"
    ASSIGNED = "assigned"
    SYNCED = "synced"
    REJECTED = "rejected"


class PriorityBand(IntEnum):
    """Scheduling priority shared by Job records and controller defaults."""

    INHERIT = 0
    PRODUCTION = 1
    INTERACTIVE = 2
    BATCH = 3


TERMINAL_JOB_STATES: frozenset[JobState] = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.KILLED,
        JobState.WORKER_FAILED,
        JobState.UNSCHEDULABLE,
    }
)

TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.KILLED,
        TaskState.UNSCHEDULABLE,
        TaskState.WORKER_FAILED,
        TaskState.PREEMPTED,
        TaskState.COSCHED_FAILED,
    }
)


def is_job_finished(state: int | JobState) -> bool:
    return state in TERMINAL_JOB_STATES


def is_task_finished(state: int | TaskState) -> bool:
    return state in TERMINAL_TASK_STATES
