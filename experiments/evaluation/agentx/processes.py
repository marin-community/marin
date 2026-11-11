# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run isolated CLIs with retained logs and process-group cleanup."""

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import psutil


def sample_client(process: subprocess.Popen, output: Path, stopped: threading.Event) -> None:
    """Retain client-tree CPU and memory samples across the measured window."""
    with output.open("w") as stream:
        while not stopped.is_set():
            cpu = 0.0
            rss = 0
            with contextlib.suppress(psutil.NoSuchProcess):
                parent = psutil.Process(process.pid)
                for child in [parent, *parent.children(recursive=True)]:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        usage = child.cpu_times()
                        cpu += usage.user + usage.system
                        rss += child.memory_info().rss
            record = {
                "unix": time.time(),
                "cpu_seconds": cpu,
                "rss_bytes": rss,
                "available_cpus": len(os.sched_getaffinity(0)),
            }
            cgroup = Path("/sys/fs/cgroup/cpu.stat")
            if cgroup.exists():
                record["cgroup_cpu_stat"] = cgroup.read_text()
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            stopped.wait(10)


def logged_process(command: list[str], environment: dict[str, str], log: Path) -> subprocess.Popen:
    """Keep file artifacts and native Iris logs, including interrupted attempts."""
    return subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", '"$@" 2>&1 | tee "$AGENTX_LOG_FILE"', "agentx-log", *command],
        env=environment | {"AGENTX_LOG_FILE": str(log)},
        start_new_session=True,
    )


def stop_group(process: subprocess.Popen) -> None:
    """Reap the command and stop its descendants before releasing the allocation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_logged(
    command: list[str], environment: dict[str, str], log: Path, timeout: int, *, sample: Path | None = None
) -> None:
    with logged_process(command, environment, log) as process:
        stopped = threading.Event()
        sampler = threading.Thread(target=sample_client, args=(process, sample, stopped)) if sample is not None else None
        if sampler is not None:
            sampler.start()
        try:
            result = process.wait(timeout=timeout)
            if result:
                raise subprocess.CalledProcessError(result, command)
        finally:
            stop_group(process)
            stopped.set()
            if sampler is not None:
                sampler.join()
