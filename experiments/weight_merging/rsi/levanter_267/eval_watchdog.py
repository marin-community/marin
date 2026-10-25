# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound evaluator lifetime when its serving endpoint disappears."""

import json
import os
import signal
import subprocess
import time
import urllib.request


def models_healthy(url):
    """Check the serving route without persisting its private address."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return bool(json.load(response).get("data"))
    except (OSError, ValueError):
        return False


def run_watched(command, env, models_url, timeout, *, interval=60, failure_seconds=300):
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
                if models_healthy(models_url):
                    failed_since = None
                else:
                    failed_since = now if failed_since is None else failed_since
                    print("Evaluator endpoint health check failed", flush=True)
                    if now - failed_since >= failure_seconds:
                        raise RuntimeError("Serving endpoint unavailable for five minutes; stopping evaluator")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
