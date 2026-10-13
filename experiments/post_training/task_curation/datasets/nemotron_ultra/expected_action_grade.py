# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one Nemotron Ultra function call against its expected name and arguments."""

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
