# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the RL gate's native SkyRL programs in one long-lived process during tests.

``rl_gate._native_json`` starts a fresh SkyRL CPU process for every composition, and
about half of each call is interpreter start-up and imports. The tests compose about a
hundred distinct launch documents, so they pay that start-up every time. This worker
runs the same program text in one SkyRL process and reports the same JSON and errors.
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
    """Serve ``rl_gate._native_json`` calls from one SkyRL process.

    Calls that pass uv options, or that run while a test has replaced ``subprocess.run``
    to simulate a failing composer, keep the original subprocess path.
    """

    def __init__(self):
        self._native_json = gate._native_json
        self._subprocess_run = subprocess.run
        self._process: subprocess.Popen[str] | None = None

    def native_json(self, *command: str, timeout: int = 120):
        if command[:2] != ("python", "-c") or subprocess.run is not self._subprocess_run:
            return self._native_json(*command, timeout=timeout)
        program, *args = command[2:]
        try:
            reply = self._request(program, args, timeout)
            if "stderr" in reply:
                raise subprocess.CalledProcessError(1, command, reply["stdout"], reply["stderr"])
            return json.loads(reply["stdout"].splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
            # Match rl_gate._native_json's diagnostics for each failure.
            if isinstance(error, subprocess.TimeoutExpired):
                cause = f"timed out after {error.timeout:g} s"
            elif isinstance(error, subprocess.CalledProcessError) and error.stderr and error.stderr.strip():
                cause = error.stderr.strip().splitlines()[-1]
            else:
                cause = str(error)
            raise gate.InvalidComparison(f"native configuration cannot be composed: {cause}") from error

    def _request(self, program: str, args: list[str], timeout: int) -> dict:
        process = self._worker()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps({"program": program, "args": args}) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise subprocess.TimeoutExpired(["python", "-c", program], timeout)
        line = process.stdout.readline()
        if not line:
            self.close()
            raise subprocess.CalledProcessError(1, ["python", "-c", program], "", "native composer worker exited")
        return json.loads(line)

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
