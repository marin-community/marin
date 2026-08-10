# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-level resource summaries for operator views."""

from dataclasses import dataclass

from iris.resources.state import JobState, TaskState


@dataclass(frozen=True, slots=True)
class UserSummary:
    """Active Job and Task counts for one observed owner."""

    user_id: str
    task_state_counts: tuple[tuple[TaskState, int], ...]
    job_state_counts: tuple[tuple[JobState, int], ...]
    role: str
