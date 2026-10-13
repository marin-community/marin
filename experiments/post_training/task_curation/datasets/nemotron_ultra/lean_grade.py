# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a Nemotron Ultra Lean reply in the grader's pinned Mathlib environment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from local_lean_sandbox import LocalLeanSandbox
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt

LEAN_COMPILE_TIMEOUT = 90.0


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    reward, detail, _ = verify_lean_attempt(
        answer,
        config["contract"],
        sandbox=LocalLeanSandbox(user=65534),
        timeout_seconds=LEAN_COMPILE_TIMEOUT,
    )
    print(json.dumps(detail, ensure_ascii=False, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
