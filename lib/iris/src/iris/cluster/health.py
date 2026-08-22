# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Application health configuration and task-side health port publication."""

import os
from dataclasses import dataclass
from pathlib import Path

from rigging.timing import Duration

from iris.rpc import job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto

HEALTH_PORT_NAME = "healthz"
HEALTH_PATH = "/healthz"
HEALTH_PORT_ENV = "IRIS_PORT_HEALTHZ"
HEALTH_PORT_FILE_ENV = "IRIS_HEALTH_PORT_FILE"
HEALTH_FAILURE_COUNT_FILE_ENV = "IRIS_HEALTH_FAILURE_COUNT_FILE"
HEALTH_TERMINATION_FILE_ENV = "IRIS_HEALTH_TERMINATION_FILE"
HEALTH_PORT_FILE = "/tmp/iris/health-port"
HEALTH_FAILURE_COUNT_FILE = "/tmp/iris/health-failures"
HEALTH_TERMINATION_FILE = "/tmp/iris/health-termination-log"


def _positive_whole_seconds(name: str, duration: Duration) -> None:
    milliseconds = duration.to_ms()
    if milliseconds <= 0:
        raise ValueError(f"{name} must be positive")
    if milliseconds % 1000:
        raise ValueError(f"{name} must use whole seconds")


@dataclass(frozen=True, slots=True)
class TaskHealthCheck:
    """Health policy that Iris applies to each task attempt."""

    startup_timeout: Duration
    period: Duration
    request_timeout: Duration
    failure_threshold: int

    def __post_init__(self) -> None:
        _positive_whole_seconds("startup_timeout", self.startup_timeout)
        _positive_whole_seconds("period", self.period)
        _positive_whole_seconds("request_timeout", self.request_timeout)
        if self.request_timeout.to_ms() >= self.period.to_ms():
            raise ValueError("request_timeout must be less than period")
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")

    def to_proto(self) -> job_pb2.TaskHealthCheck:
        return job_pb2.TaskHealthCheck(
            startup_timeout=duration_to_proto(self.startup_timeout),
            period=duration_to_proto(self.period),
            request_timeout=duration_to_proto(self.request_timeout),
            failure_threshold=self.failure_threshold,
        )


def validate_task_health_check(health_check: job_pb2.TaskHealthCheck) -> None:
    """Validate a health policy received through the Iris API."""
    TaskHealthCheck(
        startup_timeout=duration_from_proto(health_check.startup_timeout),
        period=duration_from_proto(health_check.period),
        request_timeout=duration_from_proto(health_check.request_timeout),
        failure_threshold=health_check.failure_threshold,
    )


def task_health_enabled() -> bool:
    """Return whether Iris asked this task to publish a health port."""
    return HEALTH_PORT_FILE_ENV in os.environ


def task_health_port() -> int:
    """Return the port that an application health server must bind."""
    if not task_health_enabled():
        raise RuntimeError("Iris task health is not enabled")
    raw_port = os.environ.get(HEALTH_PORT_ENV, "0")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError(f"{HEALTH_PORT_ENV} must be an integer") from error
    if not 0 <= port <= 65535:
        raise RuntimeError(f"{HEALTH_PORT_ENV} must be between 0 and 65535")
    return port


def publish_task_health(port: int) -> None:
    """Publish the bound application health port for the backend probe."""
    requested_port = task_health_port()
    if not 1 <= port <= 65535:
        raise ValueError("published health port must be between 1 and 65535")
    if requested_port and port != requested_port:
        raise ValueError(f"health server bound port {port}, expected {requested_port}")

    path = Path(os.environ[HEALTH_PORT_FILE_ENV])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{port}\n", encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o644)
