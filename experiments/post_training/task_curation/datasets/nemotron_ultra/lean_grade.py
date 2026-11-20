# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a Nemotron Ultra Lean reply in the grader's pinned Mathlib environment."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from local_lean_sandbox import LocalLeanSandbox
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt

LEAN_COMPILE_TIMEOUT = 90.0
TOOLCHAIN_PROBE = "import Mathlib\nexample : True := by trivial\n"


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    sandbox = LocalLeanSandbox(user=65534)
    health = sandbox.execute(TOOLCHAIN_PROBE, language="lean4", timeout_seconds=LEAN_COMPILE_TIMEOUT)
    if health["process_status"] != "completed" or health.get("output_truncated"):
        raise RuntimeError(f"Lean toolchain could not compile its trusted health probe: {health}")
    reward, detail, _ = verify_lean_attempt(
        answer,
        config["contract"],
        sandbox=sandbox,
        timeout_seconds=LEAN_COMPILE_TIMEOUT,
    )
    compiler = detail.get("compiler_output", {})
    if compiler.get("process_status") == "timeout":
        raise RuntimeError(f"Lean compiler timed out: {compiler}")
    if compiler.get("process_status") == "failed" and not re.search(
        r"/lean4/my_project/scratch/[^\n]+:\d+:\d+: error:",
        compiler.get("stdout", "") + "\n" + compiler.get("stderr", ""),
    ):
        raise RuntimeError(f"Lean compiler failed without a proof diagnostic: {compiler}")
    print(json.dumps(detail, ensure_ascii=False, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
