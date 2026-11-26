# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the RL gate's native SkyRL programs in one long-lived process during tests.

``rl_gate._run_native`` starts a fresh SkyRL CPU process for every composition, and
about half of each call is interpreter start-up and imports. The tests compose about a
hundred distinct launch documents, so they pay that start-up every time. This worker
runs the same program text in one SkyRL process and returns the same stdout, or raises
the same subprocess errors, so ``rl_gate._native_json`` maps results exactly as in
production.
"""

import json
import select
import subprocess

from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training import rl_gate as gate

_WORKER_PROGRAM = """
import contextlib, io, json, sys, traceback
for line in sys.stdin:
    request = json.loads(line)
    sys.argv = ["-c", *request["args"]]
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(request["program"], "<string>", "exec"), {"__name__": "__main__"})
        reply = {"stdout": stdout.getvalue()}
    except BaseException:
        reply = {"stdout": stdout.getvalue(), "stderr": traceback.format_exc()}
    sys.__stdout__.write(json.dumps(reply) + "\\n")
    sys.__stdout__.flush()
"""


class NativeComposer:
    """Serve ``rl_gate._run_native`` calls from one SkyRL process.

    Calls that pass uv options, or that run while a test has replaced ``subprocess.run``
    to simulate a failing composer, keep the original subprocess path.
    """

    def __init__(self):
        self._run_native = gate._run_native
        self._subprocess_run = subprocess.run
        self._process: subprocess.Popen[str] | None = None

    def run_native(self, *command: str, timeout: float) -> str:
        if command[:2] != ("python", "-c") or subprocess.run is not self._subprocess_run:
            return self._run_native(*command, timeout=timeout)
        program, *args = command[2:]
        process = self._worker()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps({"program": program, "args": args}) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise subprocess.TimeoutExpired(command, timeout)
        line = process.stdout.readline()
        if not line:
            self.close()
            raise subprocess.CalledProcessError(1, command, "", "native composer worker exited")
        reply = json.loads(line)
        if "stderr" in reply:
            raise subprocess.CalledProcessError(1, command, reply["stdout"], reply["stderr"])
        return reply["stdout"]

    def _worker(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                skyrl_cpu_command("python", "-c", _WORKER_PROGRAM),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        return self._process

    def close(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process = None
