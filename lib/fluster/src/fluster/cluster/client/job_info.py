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

"""Job metadata available to cluster operations.

This module provides a lightweight JobInfo container with just the essential
metadata about the current job execution. This is a "dumb" metadata holder -
it doesn't contain client instances or context logic.

For the full FlusterContext with client/registry/resolver, use fluster.client.
"""

import os
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class JobInfo:
    """Minimal job metadata available in cluster operations.

    This is a "dumb" metadata container - just facts about the current job.
    For full context with client/registry/resolver, use fluster.client.FlusterContext.

    Attributes:
        job_id: Full hierarchical job ID (e.g., "root/worker-0")
        attempt_id: Attempt number for this job execution (0-based)
        worker_id: Identifier for the worker executing this job
        controller_address: Controller URL (e.g., "http://localhost:8080")
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: str
    attempt_id: int = 0
    worker_id: str | None = None
    controller_address: str | None = None
    ports: dict[str, int] = field(default_factory=dict)


# Module-level ContextVar for job metadata
_job_info: ContextVar[JobInfo | None] = ContextVar("job_info", default=None)


def get_job_info() -> JobInfo | None:
    """Get current job info from contextvar or environment.

    First checks the contextvar (set by local execution), then falls back
    to environment variables (set by remote workers).

    Returns:
        JobInfo if available, None otherwise
    """
    info = _job_info.get()
    if info is not None:
        return info

    # Fall back to environment variables
    job_id = os.environ.get("FLUSTER_JOB_ID")
    if job_id:
        return JobInfo(
            job_id=job_id,
            attempt_id=int(os.environ.get("FLUSTER_ATTEMPT_ID", "0")),
            worker_id=os.environ.get("FLUSTER_WORKER_ID"),
            controller_address=os.environ.get("FLUSTER_CONTROLLER_ADDRESS"),
            ports=_parse_ports_from_env(),
        )
    return None


def set_job_info(info: JobInfo) -> None:
    """Set job info in contextvar (used by local execution).

    Args:
        info: Job metadata to set
    """
    _job_info.set(info)


def _parse_ports_from_env(env: dict[str, str] | None = None) -> dict[str, int]:
    """Parse port allocations from FLUSTER_PORT_* variables.

    Args:
        env: Dict to parse from. Defaults to os.environ.
    """
    source = env if env is not None else os.environ
    ports = {}
    for key, value in source.items():
        if key.startswith("FLUSTER_PORT_"):
            port_name = key[len("FLUSTER_PORT_") :].lower()
            ports[port_name] = int(value)
    return ports
