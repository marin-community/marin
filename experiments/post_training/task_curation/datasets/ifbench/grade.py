# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned Open-Instruct IFEvalVerifier scoring loop."""

import json
from pathlib import Path

from open_instruct.IFEvalG import instructions_registry


def score(prediction: str, contract: dict) -> float:
    answer = prediction.replace("<|assistant|>", "").strip().split("</think>")[-1]
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip()
    if not prediction or not answer:
        return 0.0
    rewards = []
    for name, arguments in zip(contract["instruction_id"], contract["kwargs"], strict=True):
        instruction = instructions_registry.INSTRUCTION_DICT[name](name)
        instruction.build_description(**{key: value for key, value in (arguments or {}).items() if value is not None})
        rewards.append(float(bool(prediction.strip()) and instruction.check_following(answer)))
    return sum(rewards) / max(len(rewards), 1)


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())
    print(score(Path("/app/answer.txt").read_text(), contract))


if __name__ == "__main__":
    main()
