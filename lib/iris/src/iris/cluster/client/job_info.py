# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight job metadata container without client instances or context logic.

For the full IrisContext with client/registry/resolver, use iris.client.
"""

import json
import os
import getpass
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

from google.protobuf import json_format

from iris.cluster.types import Constraint, JobName
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


@dataclass
class JobInfo:
    """Information about the currently running job."""

    task_id: JobName
    num_tasks: int = 1
    attempt_id: int = 0
    worker_id: str | None = None
    bundle_gcs_path: str | None = None

    controller_address: str | None = None
    """Address of the controller that started this job, if any."""

    advertise_host: str = "127.0.0.1"
    """The externally visible host name to use when advertising services."""

    extras: list[str] = field(default_factory=list)
    """Extras from parent job, for child job inheritance."""

    pip_packages: list[str] = field(default_factory=list)
    """Pip packages from parent job, for child job inheritance."""

    ports: dict[str, int] = field(default_factory=dict)
    """Name to port number mapping for this task."""

    env: dict[str, str] = field(default_factory=dict)
    """Explicit env vars from the job's EnvironmentConfig, for child job inheritance."""

    constraints: list[Constraint] = field(default_factory=list)
    """Explicit job constraints for child job inheritance."""

    @property
    def job_id(self) -> JobName:
        return self.task_id.parent or self.task_id

    @property
    def user(self) -> str:
        return self.task_id.user

    @property
    def task_index(self) -> int:
        return self.task_id.require_task()[1]


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
    raw_task_id = os.environ.get("IRIS_JOB_ID")
    if raw_task_id:
        try:
            task_id = JobName.from_wire(raw_task_id)
            task_id.require_task()
        except ValueError:
            return None
        job_env_json = os.environ.get("IRIS_JOB_ENV", "")
        job_env = json.loads(job_env_json) if job_env_json else {}
        constraints_json = os.environ.get("IRIS_JOB_CONSTRAINTS", "")
        constraints: list[Constraint] = []
        if constraints_json:
            for item in json.loads(constraints_json):
                constraints.append(Constraint.from_proto(json_format.ParseDict(item, cluster_pb2.Constraint())))

        info = JobInfo(
            task_id=task_id,
            num_tasks=int(os.environ.get("IRIS_NUM_TASKS", "1")),
            attempt_id=int(os.environ.get("IRIS_ATTEMPT_ID", "0")),
            worker_id=os.environ.get("IRIS_WORKER_ID"),
            controller_address=os.environ.get("IRIS_CONTROLLER_ADDRESS"),
            advertise_host=os.environ.get("IRIS_ADVERTISE_HOST", "127.0.0.1"),
            extras=json.loads(os.environ.get("IRIS_JOB_EXTRAS", "[]")),
            pip_packages=json.loads(os.environ.get("IRIS_JOB_PIP_PACKAGES", "[]")),
            bundle_gcs_path=os.environ.get("IRIS_BUNDLE_GCS_PATH"),
            ports=_parse_ports_from_env(),
            env=job_env,
            constraints=constraints,
        )
        _job_info.set(info)
        return info
    return None


def set_job_info(info: JobInfo | None) -> None:
    _job_info.set(info)


def resolve_job_user(explicit_user: str | None = None) -> str:
    """Resolve the submitting user for a new top-level job."""
    if explicit_user is not None:
        if not explicit_user.strip():
            raise ValueError("Job user must not be empty")
        return explicit_user

    info = get_job_info()
    if info is not None:
        return info.user

    try:
        resolved = getpass.getuser()
    except (OSError, KeyError, ImportError) as exc:
        logger.warning("Falling back to default Iris job user 'root': could not resolve local user (%s)", exc)
        return "root"

    if not resolved or not resolved.strip():
        logger.warning("Falling back to default Iris job user 'root': local user was empty")
        return "root"
    return resolved


def _parse_ports_from_env(env: dict[str, str] | None = None) -> dict[str, int]:
    source = env if env is not None else os.environ
    ports = {}
    for key, value in source.items():
        if key.startswith("IRIS_PORT_"):
            port_name = key[len("IRIS_PORT_") :].lower()
            ports[port_name] = int(value)
    return ports
