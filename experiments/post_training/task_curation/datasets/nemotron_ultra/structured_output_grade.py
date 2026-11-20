# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra structured-output reply with the vendored NeMo Gym scorer and print the reward.

The scorer validates the final answer text, or the payload of the reply's single typed tool call, against
the row's schema. It reads the reply as the chat message in ``/tests/conversation.json``, where tool calls
carry their arguments as a JSON string.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from conversation import terminal_message
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    message = terminal_message()
    reward, detail = grade_structured_output(message["content"], contract, message)
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
