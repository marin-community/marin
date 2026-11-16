# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic path of POLARIS's DeepScaleR reward, with its native format gate."""

import json
from pathlib import Path

from polaris_math import extract_answer, grade_answer_mathd, grade_answer_sympy


def score(answer: str, contract: dict[str, str]) -> float:
    # Upstream RewardManager grades the decoded public prompt plus response. Its
    # first-</think> split also applies if the public problem contains the delimiter.
    sequence = contract["problem"] + answer
    if "</think>" not in sequence:
        return 0.0
    candidate = extract_answer(sequence.split("</think>")[1])
    if candidate is None:
        return 0.0
    reference = contract["answer"]
    if "\\boxed" in reference:
        reference = extract_answer(reference)
        if reference is None:
            return 0.0
    return float(grade_answer_mathd(candidate, reference) or grade_answer_sympy(candidate, reference))


def main() -> None:
    print(score(Path("/app/answer.txt").read_text(), json.loads(Path("/tests/config.json").read_text())))


if __name__ == "__main__":
    main()
