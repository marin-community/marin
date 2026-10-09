# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra next-action reply with the vendored NeMo Gym scorer and print the reward.

The scorer compares the reply, read as the chat message in ``/tests/conversation.json``, with the row's
``expected_action``: one call with the expected name and arguments, or any text without calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from conversation import terminal_message
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    reward, category = grade_expected_action(contract["expected_action"], terminal_message())
    print(category, file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
