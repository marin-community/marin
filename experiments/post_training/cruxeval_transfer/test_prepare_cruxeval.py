# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.cruxeval_transfer.prepare_cruxeval import selected_rows


def test_selects_correct_reasoning_traces_without_rewriting_prompts_or_answers():
    requests = []
    responses = []
    grades = []
    for identity, direction, correct, reasoning in (
        ("input-good", "input", True, "Check the input."),
        ("output-good", "output", True, "Compute the result."),
        ("incorrect", "input", False, "Wrong computation."),
        ("no-reasoning", "output", True, ""),
    ):
        requests.append(
            {
                "custom_id": identity,
                "task": direction,
                "source_id": "sample_1",
                "b_request": {"messages": [{"role": "user", "content": identity + "\n[THOUGHT]\n"}]},
            }
        )
        responses.append(
            {
                "custom_id": identity,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "reasoning_content": reasoning,
                                    "content": "[THOUGHT]Work.[/THOUGHT][ANSWER]3[/ANSWER]",
                                },
                            }
                        ]
                    }
                },
            }
        )
        grades.append({"custom_id": identity, "task": direction, "source_id": "sample_1", "correct": correct})
    original = json.dumps([requests, responses, grades])
    rows = selected_rows(requests, responses, grades)
    assert [row["id"] for row in rows] == ["input-good", "output-good"]
    for row, reasoning in zip(rows, ("Check the input.", "Compute the result."), strict=True):
        assert row["messages"][0]["content"] == row["id"] + "\n[THOUGHT]\n"
        assert row["messages"][1]["content"] == (
            reasoning + "\n<|end_think|>\n\n[THOUGHT]Work.[/THOUGHT][ANSWER]3[/ANSWER]"
        )
    assert json.dumps([requests, responses, grades]) == original
