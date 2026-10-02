# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Importable trusted scorers for real subprocess grading tests."""

import os
import subprocess
import sys
import time
from pathlib import Path

from tasktrove_verify.grade import InvalidTask, Reward, Status, scored


def grading_with_child(receipt: str, behavior: str):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(receipt).write_text(str(child.pid))
    if behavior == "invalid":
        raise InvalidTask("trusted reference malformed")
    if behavior == "crash":
        os._exit(7)
    if behavior == "unserializable":
        return lambda: None
    if behavior in {"timeout", "interrupt"}:
        time.sleep(30)
    return {"score": 0.5}


def grading_with_large_result(private_output: str):
    print(private_output)
    print(private_output, file=sys.stderr)
    return b"x" * (1024 * 1024)


def trusted_script(behavior):
    if behavior == "invalid_reference":
        raise InvalidTask("missing trusted reference")
    if behavior == "runtime_failure":
        raise RuntimeError("scorer dependency unavailable")
    if behavior == "timeout":
        time.sleep(10)
    if behavior == "unscored_positive":
        return Reward(1, Status.INFRA_ERROR)
    if behavior == "nonfinite":
        return Reward(float("nan"), Status.SCORED)
    if behavior == "malformed":
        return {"reward": 1, "status": "scored"}
    if behavior == "invalid_detail":
        return Reward(1, Status.SCORED, {"value": float("inf")})
    print("ordinary grader output")
    return scored(0.25, observations=[True, False, False, False])
