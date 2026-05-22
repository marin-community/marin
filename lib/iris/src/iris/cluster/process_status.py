# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared process status collection for controller and worker.

Collects local process info (PID, memory, CPU, threads, etc.)
into a GetProcessStatusResponse. Used identically by both services.
"""

import os
import platform
import resource
import sys
import threading

from rigging.timing import Timer

from iris.cluster.runtime.process import _read_proc_cpu_millicores
from iris.rpc import job_pb2

# Persistent CPU sampling state so delta-based measurement works across requests.
_prev_cpu_total: float = 0.0
_prev_cpu_utime: float = 0.0


def _memory_bytes() -> tuple[int, int]:
    """Return (rss_bytes, vms_bytes).

    On Linux, reads current RSS and VMS from /proc/self/statm.
    Falls back to ru_maxrss (peak RSS) for both values when /proc is unavailable.
    On macOS, ru_maxrss (in bytes) is used as a rough approximation for both.
    """
    # Fallback: ru_maxrss is peak RSS (kilobytes on Linux, bytes on macOS).
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024 if sys.platform == "linux" else 1
    peak_rss = usage.ru_maxrss * scale
    rss = peak_rss
    vms = peak_rss

    if sys.platform == "linux":
        try:
            with open("/proc/self/statm") as f:
                parts = f.read().split()
                page_size = os.sysconf("SC_PAGE_SIZE")
                # statm fields: size(vms) resident shared text lib data dt
                vms = int(parts[0]) * page_size
                rss = int(parts[1]) * page_size
        except (OSError, ValueError, IndexError):
            pass

    return rss, vms


def _total_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def _open_fd_count() -> int:
    if sys.platform == "linux":
        try:
            return len(os.listdir("/proc/self/fd"))
        except OSError:
            pass
    return 0


def collect_process_info(timer: Timer) -> job_pb2.ProcessInfo:
    """Collect information about the current process and host."""
    global _prev_cpu_total, _prev_cpu_utime

    rss, vms = _memory_bytes()
    cpu_millicores, _prev_cpu_total, _prev_cpu_utime = _read_proc_cpu_millicores(
        os.getpid(), _prev_cpu_total, _prev_cpu_utime
    )

    return job_pb2.ProcessInfo(
        hostname=platform.node(),
        pid=os.getpid(),
        python_version=sys.version.split()[0],
        uptime_ms=timer.elapsed_ms(),
        memory_rss_bytes=rss,
        memory_vms_bytes=vms,
        cpu_millicores=cpu_millicores,
        thread_count=threading.active_count(),
        open_fd_count=_open_fd_count(),
        memory_total_bytes=_total_memory_bytes(),
        cpu_count=os.cpu_count() or 0,
        git_hash=os.environ.get("IRIS_GIT_HASH", "unknown"),
    )


def get_process_status(
    timer: Timer,
) -> job_pb2.GetProcessStatusResponse:
    """Build a GetProcessStatusResponse with local process info.

    This is the shared implementation used by both controller and worker services.
    Log fetching is handled separately via FetchLogs.
    """
    return job_pb2.GetProcessStatusResponse(
        process_info=collect_process_info(timer),
    )
