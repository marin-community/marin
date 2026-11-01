# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shutdown budgets shared by the controller, task launchers, and runtimes."""

from rigging.timing import Duration

TERMINATION_GRACE_ENV = "IRIS_TERMINATION_GRACE_MS"
DEFAULT_TERMINATION_GRACE = Duration.from_seconds(10)
MAX_TERMINATION_GRACE = Duration.from_seconds(110)
SHUTDOWN_MARGIN = Duration.from_seconds(5)
DEFAULT_WORKER_STOP_TIMEOUT_SECONDS = 5
WORKER_CONTAINER_STOP_TIMEOUT_SECONDS = int((MAX_TERMINATION_GRACE + SHUTDOWN_MARGIN * 2).to_seconds())


def resolve_termination_grace(grace: Duration | None) -> Duration:
    """Resolve and validate the job's SIGTERM grace before submission."""
    grace = DEFAULT_TERMINATION_GRACE if grace is None else grace
    if not 1000 <= grace.to_ms() <= MAX_TERMINATION_GRACE.to_ms():
        raise ValueError("termination_grace must be between 1 and 110 seconds")
    return grace
