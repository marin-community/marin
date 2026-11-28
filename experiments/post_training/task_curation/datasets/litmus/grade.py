# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from litmus_score import compute_reward, extract_predicted_value, resolve_answer_type


def score(answer: str, contract: dict) -> float:
    answer_type = resolve_answer_type(contract["answer_type"])
    predicted = extract_predicted_value(
        answer, answer_type, output_regex=contract.get("output_regex"), answer_format=contract.get("answer_format")
    )
    return compute_reward(predicted, contract["expected_answer"], answer_type, match=contract.get("match"))


if __name__ == "__main__":
    print(score(Path("/app/answer.txt").read_text(), json.loads(Path("/tests/config.json").read_text())))
