# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra Reasoning Gym reply with the puzzle task's own scorer and print the reward.

The candidate is the final answer's last ``<answer>`` block, else its last boxed answer, else the whole
final answer. The grader packages' ``reasoning_gym`` scores it against the row's question, answer and
metadata; the metadata's ``source_dataset`` names the task.
"""

import json
import re
import sys
from pathlib import Path

import reasoning_gym

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer


def candidate(reply: str) -> str:
    text = final_answer_text(reply)
    blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return blocks[-1].strip() if blocks else last_boxed_answer(text) or text.strip()


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    entry = {"question": contract["question"], "answer": contract.get("answer"), "metadata": contract["metadata"]}
    scorer = reasoning_gym.get_score_answer_fn(contract["metadata"]["source_dataset"])
    print(float(scorer(candidate(Path("/app/answer.txt").read_text()), entry)))


if __name__ == "__main__":
    main()
