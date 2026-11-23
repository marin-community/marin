# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra Reasoning Gym reply with the pinned scorer and targeted repairs.

The candidate is the final answer's last ``<answer>`` block, else its last boxed answer, else the whole
final answer. The grader packages' ``reasoning_gym`` scores it against the row's question, answer and
metadata; the metadata's ``source_dataset`` names the task.
"""

import json
import re
import sys
from pathlib import Path

from verifyit.modes.extract import extract_markdown_answer
from verifyit.modes.grade_reasoning_gym import score_reasoning_gym_answer

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer


def candidate(reply: str) -> str:
    """Prefer explicit final-answer wrappers, then preserve the source scorer's text fallback."""
    text = final_answer_text(reply)
    blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    boxed = last_boxed_answer(text)
    if boxed is not None:
        return boxed
    section = extract_markdown_answer(text)
    if section is not None:
        return section
    return text.strip()


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    entry = {"question": contract["question"], "answer": contract.get("answer"), "metadata": contract["metadata"]}
    print(
        score_reasoning_gym_answer(
            contract["metadata"]["source_dataset"], entry, candidate(Path("/app/answer.txt").read_text())
        )
    )


if __name__ == "__main__":
    main()
