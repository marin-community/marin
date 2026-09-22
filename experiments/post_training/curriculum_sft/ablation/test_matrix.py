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

    assert row["messages"][1]["content"] == (
        "A fictional issuer reports revenue of 100 (evidence: disclosure.revenue) "
        "and operating cost of 80 (evidence: disclosure.operating_cost). "
        "Calculate gross profit as revenue minus operating cost "
        "and gross margin in basis points as gross profit divided by revenue times 10000."
    )
    assert json.loads(row["messages"][2]["content"]) == {
        "result": {"gross_profit": 20, "margin_bps": 2000},
        "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
    }
