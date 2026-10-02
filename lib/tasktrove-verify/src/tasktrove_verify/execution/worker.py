# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run trusted Python grading functions in a subprocess."""

import base64
import contextlib
import math
import os
import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import TypeVar

from tasktrove_verify.execution.command import run_command

T = TypeVar("T")


def call_bounded(function: Callable[..., T], *args: object, timeout: float, **kwargs: object) -> T:
    """Call one trusted top-level function; kill its process group on every exit path."""
    if not isinstance(function, FunctionType) or function.__qualname__ != function.__name__:
        raise TypeError("bounded grading requires an importable top-level function")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("bounded grading timeout must be finite and positive")
    started = time.monotonic()
    payload = base64.b64encode(pickle.dumps((function, args, kwargs))).decode("ascii")
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("grading exceeded its deadline")
    result = run_command(
        [sys.executable, "-m", "tasktrove_verify.execution.worker"],
        Path.cwd(),
        remaining,
        stdin_text=payload,
        env={"PYTHONPATH": os.pathsep.join(sys.path)},
    )
    if result.timed_out or time.monotonic() - started >= timeout:
        raise TimeoutError("grading exceeded its deadline")
    if result.returncode:
        raise RuntimeError("grading worker exited without a result")
    try:
        succeeded, value = pickle.loads(base64.b64decode(result.stdout, validate=True))
    except Exception as error:
        raise RuntimeError("grading worker returned an unreadable result") from error
    if time.monotonic() - started >= timeout:
        raise TimeoutError("grading exceeded its deadline")
    if succeeded:
        return value
    raise value


def _main() -> None:
    try:
        function, args, kwargs = pickle.loads(base64.b64decode(sys.stdin.read(), validate=True))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = function(*args, **kwargs)
            result = (True, value)
        except Exception as error:
            result = (False, error)
        encoded = base64.b64encode(pickle.dumps(result)).decode("ascii")
    except BaseException:
        # Do not print exception text: task data or credentials may be embedded in it.
        raise SystemExit(1) from None
    sys.stdout.write(encoded)


if __name__ == "__main__":
    _main()
