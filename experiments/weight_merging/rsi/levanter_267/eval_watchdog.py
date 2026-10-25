# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound evaluator lifetime when its serving endpoint disappears."""

import json
import os
import signal
import subprocess
import time
import urllib.request

from iris.rpc import controller_pb2, job_pb2


def models_healthy(url):
    """Check the serving route without persisting its private address."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return bool(json.load(response).get("data"))
    except (OSError, ValueError):
        return False


def gang_healthy(rpc, job, *, timeout_ms=10000):
    """Issue one no-retry RPC with a short deadline; require every serving rank."""
    terminal = {
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    }
    try:
        tasks = rpc.list_tasks(
            controller_pb2.Controller.ListTasksRequest(job_id=job.to_wire()), timeout_ms=timeout_ms
        ).tasks
    except Exception:
        return False
    if any(task.state in terminal for task in tasks):
        raise RuntimeError("Serving gang has a terminal task; stopping evaluator")
    expected = {f"{job.to_wire()}/{rank}" for rank in range(4)}
    return (
        len(tasks) == 4
        and {task.task_id for task in tasks} == expected
        and all(task.state == job_pb2.TASK_STATE_RUNNING for task in tasks)
    )


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
