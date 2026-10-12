# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra instruction-following reply with the vendored NeMo Gym scorer and print the reward.

The scorer checks every instruction in the row's ``instruction_id_list`` with the grader packages'
``verifiable_instructions`` registry.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.instruction_following import grade_instruction_following


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    reward, detail = grade_instruction_following(answer, contract)
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
