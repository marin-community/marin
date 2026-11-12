# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional cgroup evidence for oracle resource calibration."""

import json
import logging
import time
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import TaskOS

RESOURCE_COUNTERS_COMMAND = (
    "for name in cpu.stat memory.events; do "
    "if test -r /sys/fs/cgroup/$name; then cat /sys/fs/cgroup/$name; fi; done"
)


async def resource_counters(environment: BaseEnvironment) -> dict[str, int]:
    if environment.os != TaskOS.LINUX:
        return {}
    result = await environment.exec(command=RESOURCE_COUNTERS_COMMAND, timeout_sec=5, user="root")
    return {key: int(value) for line in result.stdout.splitlines() if len(parts := line.split()) == 2
            for key, value in [parts]}


async def write_resource_usage(
    environment: BaseEnvironment, path: Path, initial: dict[str, int], started: float
) -> None:
    """Persist counter deltas without replacing the trial's primary failure if a sandbox died."""
    elapsed = time.monotonic() - started
    try:
        final = await resource_counters(environment)
        usage = {"elapsed_seconds": elapsed}
        if "usage_usec" in initial and "usage_usec" in final:
            usage["cpu_seconds"] = max(0, final["usage_usec"] - initial["usage_usec"]) / 1e6
        if "oom_kill" in initial and "oom_kill" in final:
            usage["oom_kills"] = max(0, final["oom_kill"] - initial["oom_kill"])
        path.write_text(json.dumps(usage))
    except Exception:
        logging.getLogger(__name__).exception("Oracle resource counters unavailable after execution")
