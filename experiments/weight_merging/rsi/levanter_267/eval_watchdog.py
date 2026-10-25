# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound evaluator lifetime when its serving endpoint disappears."""

import json
import os
import signal
import subprocess
import time
import urllib.request

from iris.resources.state import TERMINAL_TASK_STATES


def models_healthy(url):
    """Check the serving route without persisting its private address."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return bool(json.load(response).get("data"))
    except (OSError, ValueError):
        return False


def gang_healthy(client, job):
    """Read one serving-gang snapshot; terminal peers invalidate HTTP liveness."""
    try:
        tasks = client.list_tasks(job)
    except Exception:
        return False
    if any(task.state in TERMINAL_TASK_STATES for task in tasks):
        raise RuntimeError("Serving gang has a terminal task; stopping evaluator")
    return bool(tasks)


def run_watched(command, env, models_url, timeout, *, interval=60, failure_seconds=300, check_gang=None):
    """Stop the client process group after continuous endpoint failure or its runtime bound."""
    started = time.monotonic()
    failed_since = None
    with subprocess.Popen(command, env=env, start_new_session=True) as process:
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Evaluator exceeded its runtime bound")
                try:
                    return process.wait(timeout=min(interval, remaining))
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                gang_ok = check_gang() if check_gang is not None else True
                if gang_ok and models_healthy(models_url):
                    failed_since = None
                else:
                    failed_since = now if failed_since is None else failed_since
                    print("Evaluator serving health check failed", flush=True)
                    if now - failed_since >= failure_seconds:
                        raise RuntimeError("Serving endpoint unavailable for five minutes; stopping evaluator")
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # The session leader can exit while a descendant ignores TERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
