# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workload lifecycle values independent of the RPC transport."""

from enum import IntEnum


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


class FederationState(IntEnum):
    LOCAL = 0
    PENDING = 1
    ASSIGNED = 2
    SYNCED = 3
    REJECTED = 4


TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.KILLED,
        JobState.WORKER_FAILED,
        JobState.UNSCHEDULABLE,
    }
)

TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.KILLED,
        TaskState.WORKER_FAILED,
        TaskState.UNSCHEDULABLE,
        TaskState.PREEMPTED,
        TaskState.COSCHED_FAILED,
    }
)


def is_job_finished(state: JobState) -> bool:
    return state in TERMINAL_JOB_STATES


def is_task_finished(state: TaskState) -> bool:
    return state in TERMINAL_TASK_STATES
