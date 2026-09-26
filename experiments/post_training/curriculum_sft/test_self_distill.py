# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig

from experiments.post_training.curriculum_sft.self_distill import (
    AnswerCheck,
    SelfDistillConfig,
    answer_matches,
    grade_samples,
)


def _character_count(row: dict) -> int:
    return sum(len(message["content"]) + len(message["reasoning_content"] or "") for message in row["messages"])


def test_grade_samples_keeps_first_closed_correct_sample_that_fits():
    config = SelfDistillConfig(
        problems_paths={},
        output_path="unused",
        answer_check=AnswerCheck.MATH,
        model=ModelConfig(name="unused", location="unused"),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        samples_per_problem=6,
        solutions_per_problem=1,
        temperature=0.7,
        max_completion_tokens=1024,
        max_sequence_tokens=200,
        seed=17,
    )
    problem = {"request_id": "problem-00000", "problem": "Compute 1/2.", "answer": "\\frac{1}{2}"}
    outputs = [
        ("length", "<|start_think|>Half of one is"),
        ("stop", "<|start_think|>Half of one. The answer is \\boxed{0.5}."),
        ("stop", "<|start_think|>Guess.<|end_think|>So \\boxed{2}."),
        ("stop", "<|start_think|>" + "x" * 300 + "<|end_think|>So \\boxed{1/2}."),
        ("stop", "<|start_think|>One half.<|end_think|>Halve 1: \\boxed{0.5}."),
        ("stop", "<|start_think|>Half.<|end_think|>So \\boxed{\\tfrac12}."),
    ]

    records, chat_rows = grade_samples(problem, outputs, config, _character_count)

    assert [record["rejection_reason"] for record in records] == [
        "truncated",
        "unclosed_think",
        "wrong_answer",
        "too_long",
        None,
        None,
    ]
    assert [record["selected"] for record in records] == [False, False, False, False, True, False]
    assert chat_rows == [
        {
            "id": "problem-00000-self04",
            "messages": [
                {"role": "user", "content": "Compute 1/2.", "reasoning_content": None},
                {"role": "assistant", "content": "Halve 1: \\boxed{0.5}.", "reasoning_content": "One half."},
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]


@pytest.mark.parametrize(
    ("check", "reference", "candidate", "expected"),
    [
        (AnswerCheck.NUMERIC, "93.86", "93.9 days", True),
        (AnswerCheck.NUMERIC, "1,250", "$1250", True),
        (AnswerCheck.NUMERIC, "0.83", "0.80", False),
        (AnswerCheck.NUMERIC, "12.5%", "12.4", True),
        (AnswerCheck.CHOICE, "C", "(c)", True),
        (AnswerCheck.CHOICE, "C", "B", False),
    ],
)
def test_answer_matches_numeric_and_choice(check, reference, candidate, expected):
    assert answer_matches(check, reference, candidate) is expected
