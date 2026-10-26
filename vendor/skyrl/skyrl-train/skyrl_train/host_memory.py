# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host (CPU) memory of a worker process, logged at named points.

Each record reads ``/proc/self/status``: ``VmRSS`` is the resident set now and
``VmHWM`` the peak since the last ``reset_peak`` (or process start); ``MemAvailable``
from ``/proc/meminfo`` is what the node has left. Lines are tagged ``HOST_MEMORY`` so a
job's log can be grepped into a per-rank timeline.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_KIB_PER_GIB = 1024**2


def parse_status_kib(text: str, field: str) -> int:
    """A ``kB`` field of ``/proc/self/status`` or ``/proc/meminfo`` text."""
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1])
    raise ValueError(f"{field} is missing")


def host_memory_gib() -> dict[str, float]:
    status = Path("/proc/self/status").read_text()
    meminfo = Path("/proc/meminfo").read_text()
    return {
        "rss": parse_status_kib(status, "VmRSS") / _KIB_PER_GIB,
        "peak": parse_status_kib(status, "VmHWM") / _KIB_PER_GIB,
        "node_available": parse_status_kib(meminfo, "MemAvailable") / _KIB_PER_GIB,
    }


def reset_peak() -> None:
    """Restart ``VmHWM`` from the current resident set (Linux ``clear_refs`` mode 5)."""
    Path("/proc/self/clear_refs").write_text("5")


def log_host_memory(label: str, rank: int) -> None:
    memory = host_memory_gib()
    logger.info(
        f"HOST_MEMORY {label} rank={rank} rss_gib={memory['rss']:.1f} peak_gib={memory['peak']:.1f} "
        f"node_available_gib={memory['node_available']:.1f}"
    )
