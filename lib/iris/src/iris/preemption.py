# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shutdown budget shared by task launchers and runtimes."""

from collections.abc import Mapping

PREEMPTION_GRACE_ENV = "IRIS_PREEMPTION_GRACE_SECONDS"
DEFAULT_PREEMPTION_GRACE_SECONDS = 10
DEFAULT_WORKER_STOP_TIMEOUT_SECONDS = 5
SHUTDOWN_MARGIN_SECONDS = 5
MAX_PREEMPTION_GRACE_SECONDS = 110
WORKER_CONTAINER_STOP_TIMEOUT_SECONDS = MAX_PREEMPTION_GRACE_SECONDS + 2 * SHUTDOWN_MARGIN_SECONDS


def preemption_grace_seconds(env: Mapping[str, str]) -> int:
    """Return the task's configured SIGTERM grace in seconds."""
    try:
        value = int(env.get(PREEMPTION_GRACE_ENV, DEFAULT_PREEMPTION_GRACE_SECONDS))
    except ValueError as exc:
        raise ValueError(f"{PREEMPTION_GRACE_ENV} must be an integer") from exc
    if not 1 <= value <= MAX_PREEMPTION_GRACE_SECONDS:
        raise ValueError(f"{PREEMPTION_GRACE_ENV} must be between 1 and {MAX_PREEMPTION_GRACE_SECONDS}")
    return value
