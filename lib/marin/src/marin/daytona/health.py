# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral timing records for Daytona sandbox probes."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HealthProbeResult:
    """Timings and output from a create/exec/delete sandbox health probe."""

    create_seconds: float
    exec_seconds: float
    delete_seconds: float
    exit_code: int
    output: str


def run_health_probe(
    *,
    create: Callable[[], Any],
    command: str,
    execute: Callable[[Any, str], tuple[int, str]],
    delete: Callable[[Any], None],
) -> HealthProbeResult:
    """Run an owned sandbox probe and always attempt its cleanup."""

    started = time.monotonic()
    sandbox = create()
    create_seconds = time.monotonic() - started
    exec_started = time.monotonic()
    try:
        exit_code, output = execute(sandbox, command)
        exec_seconds = time.monotonic() - exec_started
    finally:
        delete_started = time.monotonic()
        delete(sandbox)
        delete_seconds = time.monotonic() - delete_started
    return HealthProbeResult(create_seconds, exec_seconds, delete_seconds, exit_code, output)
