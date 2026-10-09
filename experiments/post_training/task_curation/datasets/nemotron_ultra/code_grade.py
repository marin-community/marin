# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra competitive-programming reply with the vendored NeMo Gym scorer and print the reward.

The scorer runs the final reply's last fenced program against the row's hidden unit tests with the
vendored LiveCodeBench evaluator, and reads the reply as the chat message in ``/tests/conversation.json``.
The evaluator's child process shares this script's stdout, so stdout points at stderr while it runs and
carries only the reward.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from conversation import terminal_message
from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    message = terminal_message()
    stdout = os.dup(1)
    os.dup2(2, 1)
    reward, detail = grade_code(message["content"], contract, assistant_message=message)
    sys.stdout.flush()
    os.dup2(stdout, 1)
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
