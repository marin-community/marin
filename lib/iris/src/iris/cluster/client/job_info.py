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

"""Lightweight job metadata container without client instances or context logic.

For the full IrisContext with client/registry/resolver, use iris.client.
"""

import os
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class JobInfo:
    """Minimal job metadata available in cluster operations.

    For full context with client/registry/resolver, use iris.client.IrisContext.
    """

    job_id: str
    task_id: str | None = None
    task_index: int = 0
    num_tasks: int = 1
    attempt_id: int = 0
    worker_id: str | None = None
    controller_address: str | None = None
    advertise_host: str = "127.0.0.1"
    ports: dict[str, int] = field(default_factory=dict)


# Module-level ContextVar for job metadata
_job_info: ContextVar[JobInfo | None] = ContextVar("job_info", default=None)


def get_job_info() -> JobInfo | None:
    """Get current job info from contextvar or environment.

    Returns:
        JobInfo if available, None otherwise
    """
    info = _job_info.get()
    if info is not None:
        return info

    # Fall back to environment variables
    job_id = os.environ.get("IRIS_JOB_ID")
    if job_id:
        info = JobInfo(
            job_id=job_id,
            task_id=os.environ.get("IRIS_TASK_ID"),
            task_index=int(os.environ.get("IRIS_TASK_INDEX", "0")),
            num_tasks=int(os.environ.get("IRIS_NUM_TASKS", "1")),
            attempt_id=int(os.environ.get("IRIS_ATTEMPT_ID", "0")),
            worker_id=os.environ.get("IRIS_WORKER_ID"),
            controller_address=os.environ.get("IRIS_CONTROLLER_ADDRESS"),
            advertise_host=os.environ.get("IRIS_ADVERTISE_HOST", "127.0.0.1"),
            ports=_parse_ports_from_env(),
        )
        _job_info.set(info)
        return info
    return None


def set_job_info(info: JobInfo) -> None:
    _job_info.set(info)


def _parse_ports_from_env(env: dict[str, str] | None = None) -> dict[str, int]:
    source = env if env is not None else os.environ
    ports = {}
    for key, value in source.items():
        if key.startswith("IRIS_PORT_"):
            port_name = key[len("IRIS_PORT_") :].lower()
            ports[port_name] = int(value)
    return ports
