# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    generated_payloads_to_rows,
)


@pytest.mark.parametrize(
    "question",
    [
        "Revenue is 80 and operating cost is 100. What is the gross profit?",
        "Revenue is 100 and operating cost is 80. What is twice the gross profit?",
    ],
)
def test_sft_question_and_answer_use_verified_facts(question):
    payload = {
        "task_id": "example",
        "issuer": "Fictional issuer",
        "question": question,
        "facts": {"revenue": 100, "operating_cost": 80},
        "answer": {"gross_profit": 20, "margin_bps": 2000},
        "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
    }
    cell = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, accepted_examples=1)
    [row] = generated_payloads_to_rows(cell, [payload])

    user_message = row["messages"][1]["content"]
    assert "revenue of 100 (evidence: disclosure.revenue)" in user_message
    assert "operating cost of 80 (evidence: disclosure.operating_cost)" in user_message
    assert "revenue minus operating cost" in user_message
    assert "gross profit divided by revenue times 10000" in user_message
    assert question not in user_message
    assert json.loads(row["messages"][2]["content"]) == {
        "result": {"gross_profit": 20, "margin_bps": 2000},
        "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
    }
