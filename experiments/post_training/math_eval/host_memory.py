# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded cgroup-v2 host memory evidence around one native inference task."""

import json
import threading
from contextlib import contextmanager
from pathlib import Path

EVENTS = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")


def cgroup_memory(root, expected_limit_bytes):
    root = Path(root)
    values = {name: int((root / f"memory.{name}").read_text().strip()) for name in ("current", "peak", "max")}
    if values["max"] != expected_limit_bytes or any(value < 0 for value in values.values()):
        raise ValueError("Native cgroup memory limit differs from the reviewed task")
    events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
    return {
        "current_bytes": values["current"],
        "kernel_peak_bytes": values["peak"],
        "limit_bytes": values["max"],
        "events": {key: int(events[key]) for key in EVENTS if key in events},
    }


def _emit(snapshot):
    print("RATING_HOST_MEMORY " + json.dumps(snapshot, sort_keys=True), flush=True)


@contextmanager
def host_memory_monitor(*, expected_limit_bytes, root=Path("/sys/fs/cgroup"), interval=5.0):
    """Keep before/after and at most 360 periodic samples in a 30-minute task.

    The caller persists the yielded receipt in its finally block. Kernel OOM
    kills can prevent all Python finally blocks; periodic native log evidence
    remains partial evidence in that case, never a fabricated final snapshot.
    """
    if interval < 5:
        raise ValueError("Memory sampling interval must be at least five seconds")
    before = cgroup_memory(root, expected_limit_bytes)
    result = {
        "schema": "math_eval_host_memory_v1",
        "before": before,
        "after": None,
        "observed_current_peak_bytes": before["current_bytes"],
        "samples": 1,
        "status": "running",
        "sample_error": None,
        "interval_seconds": interval,
    }
    _emit(before)
    stop = threading.Event()

    def observe():
        try:
            for _ in range(360):
                if stop.wait(interval):
                    return
                current = cgroup_memory(root, expected_limit_bytes)
                result["observed_current_peak_bytes"] = max(
                    result["observed_current_peak_bytes"], current["current_bytes"]
                )
                result["samples"] += 1
                _emit(current)
        except Exception as error:
            result["sample_error"] = f"{type(error).__name__}: {error}"

    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    task_failed = False
    try:
        yield result
        result["status"] = "success"
    except BaseException:
        task_failed = True
        result["status"] = "error"
        raise
    finally:
        stop.set()
        worker.join()
        try:
            result["after"] = cgroup_memory(root, expected_limit_bytes)
            result["observed_current_peak_bytes"] = max(
                result["observed_current_peak_bytes"], result["after"]["current_bytes"]
            )
            result["samples"] += 1
            _emit(result["after"])
        except Exception as error:
            result["sample_error"] = f"{type(error).__name__}: {error}"
        if result["sample_error"] is not None:
            result["status"] = "error"
            if not task_failed:
                raise RuntimeError(result["sample_error"])
