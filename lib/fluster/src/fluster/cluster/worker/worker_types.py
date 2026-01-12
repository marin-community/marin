# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal worker types for job tracking."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from fluster import cluster_pb2
from fluster.cluster.types import is_job_finished
from fluster.cluster_pb2 import JobState


@dataclass(kw_only=True)
class Job:
    """Internal job tracking state."""

    job_id: str
    request: cluster_pb2.RunJobRequest
    status: JobState = cluster_pb2.JOB_STATE_PENDING
    exit_code: int | None = None
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    ports: dict[str, int] = field(default_factory=dict)
    status_message: str = ""

    # Internals
    container_id: str | None = None
    workdir: Path | None = None  # Job working directory with logs
    task: asyncio.Task | None = None
    cleanup_done: bool = False

    def transition_to(
        self,
        state: JobState,
        *,
        message: str = "",
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Transition to a new state with appropriate side effects.

        Args:
            state: Target state
            message: Progress message (only retained in BUILDING state)
            error: Error message (for FAILED state)
            exit_code: Process exit code (for terminal states)
        """
        self.status = state
        self.status_message = message
        if is_job_finished(state):
            self.finished_at_ms = int(time.time() * 1000)
            if error:
                self.error = error
            if exit_code is not None:
                self.exit_code = exit_code

    def to_proto(self) -> cluster_pb2.JobStatus:
        """Convert job to JobStatus proto."""
        return cluster_pb2.JobStatus(
            job_id=self.job_id,
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            started_at_ms=self.started_at_ms or 0,
            finished_at_ms=self.finished_at_ms or 0,
            ports=self.ports,
            status_message=self.status_message,
        )
