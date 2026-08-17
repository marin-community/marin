# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workload lifecycle values independent of the RPC transport."""

from enum import StrEnum


class JobState(StrEnum):
    UNSPECIFIED = "unspecified"
    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"
    WORKER_FAILED = "worker_failed"
    UNSCHEDULABLE = "unschedulable"


class TaskState(StrEnum):
    UNSPECIFIED = "unspecified"
    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"
    WORKER_FAILED = "worker_failed"
    UNSCHEDULABLE = "unschedulable"
    ASSIGNED = "assigned"
    PREEMPTED = "preempted"
    COSCHED_FAILED = "cosched_failed"
    MISSING = "missing"


class FederationState(StrEnum):
    LOCAL = "local"
    PENDING = "pending"
    ASSIGNED = "assigned"
    SYNCED = "synced"
    REJECTED = "rejected"


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
