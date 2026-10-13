# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a Nemotron Ultra Lean reply with the source proof builder and sandbox client."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    sandbox = SandboxClient(host=config["sandbox_host"], port=config["sandbox_port"])
    reward, detail, _ = verify_lean_attempt(answer, config["contract"], sandbox=sandbox)
    print(json.dumps(detail, ensure_ascii=False, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
