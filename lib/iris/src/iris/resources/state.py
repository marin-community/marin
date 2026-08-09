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


class PriorityBand(IntEnum):
    """Scheduling priority shared by Job records and controller defaults."""

    INHERIT = 0
    PRODUCTION = 1
    INTERACTIVE = 2
    BATCH = 3


class EndpointAccess(StrEnum):
    PRIVATE = "private"
    LINK = "link"

    @classmethod
    def from_storage(cls, value: int | None) -> "EndpointAccess":
        """Decode the stable controller persistence representation."""
        if value in (None, 0):
            return cls.PRIVATE
        if value == 2:
            return cls.LINK
        raise ValueError(f"Unknown stored endpoint access value: {value!r}")
