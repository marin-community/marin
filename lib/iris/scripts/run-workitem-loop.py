#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs claude in a loop, each iteration picking up the next unchecked task
from the workitem checklist. Stops when the agent reports 'ALL DONE'."""

import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKITEM = Path("lib/iris/docs/tasks/workitem.md")
LOGDIR = Path("lib/iris/logs/workitem-runs")
MAX_ITERATIONS = 50

PROMPT = f"Read and follow the instructions in {WORKITEM}. Do exactly one task, commit, and report back."

# Strip ANSI escape sequences for signal detection.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def find_signal(text: str) -> str | None:
    """Scan the last few non-blank lines for a completion signal.

    Returns "ALL DONE", "Labor numquam deest", or None.
    """
    lines = [ANSI_RE.sub("", line).strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-10:]):
        if "ALL DONE" in line:
            return "ALL DONE"
        if "Labor numquam deest" in line:
            return "Labor numquam deest"
    return None


def run_iteration(iteration: int) -> bool:
    """Run one claude invocation. Returns True if all tasks are done."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = LOGDIR / f"run-{iteration}-{timestamp}.log"

    print(f"=== Iteration {iteration} ({datetime.now()}) ===")
    print(f"    Log: {logfile}")

    # Stream output live to terminal while capturing to log file.
    # Start in a new process group so we can kill the entire tree on exit.
    # Strip ANTHROPIC_API_KEY so claude uses OAuth instead of credit mode.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with open(logfile, "w") as log:
        proc = subprocess.Popen(
            ["claude", "--dangerously-skip-permissions", "--verbose", "-p", PROMPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        assert proc.stdout is not None
        output_lines: list[str] = []
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                output_lines.append(line)
            proc.wait()
        except KeyboardInterrupt:
            print("\n=== Interrupted. Killing subprocess tree... ===")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    output = "".join(output_lines)

    if proc.returncode != 0:
        print(f"\n=== claude exited with code {proc.returncode}. Check {logfile}. ===")

    completion_signal = find_signal(output)

    if completion_signal == "ALL DONE":
        print(f"\n=== All tasks complete after {iteration} iteration(s). ===")
        return True

    if completion_signal == "Labor numquam deest":
        print("\n=== Tasks remain. Continuing to next iteration... ===\n")
        return False

    print("\n=== No completion signal found in output. ===")
    print(f"=== Agent may have errored. Check {logfile}. Continuing anyway... ===\n")
    return False


def main():
    if not WORKITEM.exists():
        print(f"Error: {WORKITEM} not found. Run this script from the repo root.", file=sys.stderr)
        sys.exit(1)

    LOGDIR.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, MAX_ITERATIONS + 1):
        if run_iteration(iteration):
            sys.exit(0)

    print(f"=== Hit max iterations ({MAX_ITERATIONS}). Stopping. ===")
    sys.exit(1)


if __name__ == "__main__":
    main()
