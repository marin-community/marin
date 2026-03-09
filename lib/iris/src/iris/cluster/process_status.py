# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared process status collection for controller and worker.

Collects local process info (PID, memory, CPU, threads, etc.) and recent
logs into a GetProcessStatusResponse. Used identically by both services.
"""

import os
import platform
import resource
import sys
import threading

from iris.cluster.log_store import PROCESS_LOG_KEY, LogStore
from iris.cluster.runtime.process import _read_proc_cpu_percent
from iris.rpc import cluster_pb2
from iris.time_utils import Timer

# Persistent CPU sampling state so delta-based measurement works across requests.
_prev_cpu_total: float = 0.0
_prev_cpu_utime: float = 0.0


def _memory_bytes() -> tuple[int, int]:
    """Return (rss_bytes, vms_bytes). vms is approximated as ru_maxrss.

    Linux reports ru_rss and ru_maxrss in kilobytes; macOS reports bytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024 if sys.platform == "linux" else 1
    rss = usage.ru_rss * scale
    vms = usage.ru_maxrss * scale
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


def collect_process_info(timer: Timer) -> cluster_pb2.ProcessInfo:
    """Collect information about the current process and host."""
    global _prev_cpu_total, _prev_cpu_utime

    rss, vms = _memory_bytes()
    cpu_pct, _prev_cpu_total, _prev_cpu_utime = _read_proc_cpu_percent(os.getpid(), _prev_cpu_total, _prev_cpu_utime)

    return cluster_pb2.ProcessInfo(
        hostname=platform.node(),
        pid=os.getpid(),
        python_version=sys.version.split()[0],
        uptime_ms=timer.elapsed_ms(),
        memory_rss_bytes=rss,
        memory_vms_bytes=vms,
        cpu_percent=cpu_pct,
        thread_count=threading.active_count(),
        open_fd_count=_open_fd_count(),
        memory_total_bytes=_total_memory_bytes(),
        cpu_count=os.cpu_count() or 0,
        git_hash=os.environ.get("IRIS_GIT_HASH", "unknown"),
    )


def get_process_status(
    request: cluster_pb2.GetProcessStatusRequest,
    log_store: LogStore | None,
    timer: Timer,
) -> cluster_pb2.GetProcessStatusResponse:
    """Build a GetProcessStatusResponse with local process info and recent logs.

    This is the shared implementation used by both controller and worker services.
    """
    process_info = collect_process_info(timer)

    # Fetch recent process logs
    log_entries = []
    if log_store:
        max_lines = request.max_log_lines if request.max_log_lines > 0 else 200
        result = log_store.get_logs(
            PROCESS_LOG_KEY,
            substring_filter=request.log_substring or "",
            max_lines=max_lines,
            tail=True,
            min_level=request.min_log_level or "",
        )
        log_entries = list(result.entries)

    return cluster_pb2.GetProcessStatusResponse(
        process_info=process_info,
        log_entries=log_entries,
    )
