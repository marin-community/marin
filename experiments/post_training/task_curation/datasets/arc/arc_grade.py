# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a Nemotron Ultra inductive ARC reply with the scorer the grader image installs.

The NVARC scorer runs the submitted transform through the NeMo Skills sandbox server.

Usage: ``arc_grade.py CONFIG ANSWER SCORE``. Reads the row contract from ``CONFIG`` and the reply from
``ANSWER``; writes the reward JSON to ``SCORE``.
"""

import importlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

nvarc = importlib.import_module("skyrl_gym.envs.nemotron_ultra.nvarc")
sandbox = importlib.import_module("skyrl_gym.envs.nemotron_ultra.sandbox")

TESTS = Path("/tests")
LOGS = Path("/logs/verifier")
SERVER = "/opt/nemo_skills/local_sandbox_server.py"


def wait_for_server(process, worker_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Original NeMo Skills sandbox exited: {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:6000/health", timeout=0.2) as response:
                if response.status == 200 and json.load(response)["worker"] == worker_id:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise TimeoutError("Original NeMo Skills sandbox did not become ready")


def grade_inductive(answer, contract):
    # The original scorer runs submitted Python through its HTTP sandbox. Keep
    # the server under a separate UID so it cannot read held-out grader files.
    TESTS.chmod(0o700)
    LOGS.chmod(0o700)
    worker_id = uuid4().hex
    process = subprocess.Popen(
        (
            "setpriv",
            "--reuid",
            "65534",
            "--regid",
            "65534",
            "--clear-groups",
            "--no-new-privs",
            "python3",
            SERVER,
        ),
        cwd="/tmp",
        env={**os.environ, "WORKER_NUM": worker_id},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_server(process, worker_id)
        return nvarc.grade_inductive_arc(answer, contract, sandbox=sandbox.SandboxClient())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    config_path, answer_path, score_path = map(Path, sys.argv[1:4])
    contract = json.loads(config_path.read_text())["contract"]
    reward, diagnostics = grade_inductive(answer_path.read_text(), contract)
    score_path.write_text(json.dumps({"reward": reward, "detail": diagnostics}))


if __name__ == "__main__":
    main()
