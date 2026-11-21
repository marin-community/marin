# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra formatting reply with the vendored NeMo Gym scorer and print the reward.

The scorer applies the row's ``verifier`` (line patterns or citation markers) to the final answer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.format_verification import grade_format


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    reward, detail = grade_format(answer, contract["verifier"])
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
