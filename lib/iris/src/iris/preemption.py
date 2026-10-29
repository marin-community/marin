# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shutdown budget shared by task launchers and runtimes."""

from collections.abc import Mapping

PREEMPTION_GRACE_ENV = "IRIS_PREEMPTION_GRACE_SECONDS"
DEFAULT_PREEMPTION_GRACE_SECONDS = 10
SHUTDOWN_MARGIN_SECONDS = 5


def preemption_grace_seconds(env: Mapping[str, str]) -> int:
    """Return the task's configured SIGTERM grace in seconds."""
    value = int(env.get(PREEMPTION_GRACE_ENV, DEFAULT_PREEMPTION_GRACE_SECONDS))
    if value < 1:
        raise ValueError(f"{PREEMPTION_GRACE_ENV} must be positive")
    return value
