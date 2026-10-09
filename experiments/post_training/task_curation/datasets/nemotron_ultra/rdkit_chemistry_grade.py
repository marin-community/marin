# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra chemistry reply with the vendored NeMo Gym scorer and print the reward.

The scorer compares the rounded number in the row's answer wrapper with the stored property target.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.rdkit_chemistry import grade_rdkit_chemistry


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    answer = final_answer_text(Path("/app/answer.txt").read_text())
    reward, detail = grade_rdkit_chemistry(answer, contract)
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
