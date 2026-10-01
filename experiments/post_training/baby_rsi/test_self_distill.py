# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig

from experiments.post_training.baby_rsi.self_distill import (
    AnswerCheck,
    SelfDistillConfig,
    answer_matches,
    grade_samples,
)


def _character_count(row: dict) -> int:
    return sum(len(message["content"]) + len(message["reasoning_content"] or "") for message in row["messages"])


def _config(answer_check: AnswerCheck, max_sequence_tokens: int = 10_000) -> SelfDistillConfig:
    return SelfDistillConfig(
        problems_paths={},
        output_path="unused",
        answer_check=answer_check,
        model=ModelConfig(name="unused", location="unused"),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        samples_per_problem=6,
        solutions_per_problem=1,
        temperature=0.7,
        max_completion_tokens=1024,
        max_sequence_tokens=max_sequence_tokens,
        seed=17,
    )


def test_grade_samples_keeps_first_closed_correct_sample_that_fits():
    config = _config(AnswerCheck.MATH, max_sequence_tokens=200)
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
        (AnswerCheck.PYTHON_LITERAL, "['a', (1, 2.0)]", "[ 'a',(1,2.0) ]", True),
        (AnswerCheck.PYTHON_LITERAL, "[1, 2]", "[1.0, 2]", False),
        (AnswerCheck.PYTHON_LITERAL, "'abc'", "abc", False),
    ],
)
def test_answer_matches_numeric_choice_and_literal(check, reference, candidate, expected):
    assert answer_matches(check, reference, candidate) is expected


def test_python_tests_check_grades_the_last_code_block_by_running_the_tests():
    tests = ["assert double(2) == 4", "assert double(-3) == -6", "assert double(0) == 0"]
    problem = {"request_id": "task-00000", "problem": "Implement double.", "answer": json.dumps(tests)}
    wrong = "```python\ndef double(x):\n    return x + 2\n```"
    right = "```python\ndef double(x):\n    return 2 * x\n```"
    outputs = [
        ("stop", "<|start_think|>Easy.<|end_think|>def double(x): return 2 * x"),
        ("stop", f"<|start_think|>Add two?<|end_think|>{wrong}"),
        ("stop", f"<|start_think|>First try, then fix.<|end_think|>{wrong}\nFixed:\n{right}"),
    ]

    records, chat_rows = grade_samples(problem, outputs, _config(AnswerCheck.PYTHON_TESTS), _character_count)

    assert [record["rejection_reason"] for record in records] == ["no_code_block", "wrong_answer", None]
    assert records[2]["extracted"] == "def double(x):\n    return 2 * x\n"
    assert [row["id"] for row in chat_rows] == ["task-00000-self02"]


def test_constrained_problem_rejects_a_correct_answer_that_breaks_a_constraint():
    constraints = [{"kind": "no_commas", "kwargs": {}}, {"kind": "bullet_count", "kwargs": {"count": 2}}]
    problem = {
        "request_id": "problem-00000",
        "problem": "What is 1,000 + 250?",
        "answer": "1250",
        "constraints": json.dumps(constraints),
    }
    outputs = [
        ("stop", "<|start_think|>Add.<|end_think|>* Sum is 1,250.\n* So \\boxed{1250}"),
        ("stop", "<|start_think|>Add.<|end_think|>* Sum is 1250.\n* So \\boxed{1350}"),
        ("stop", "<|start_think|>Add, then 1,250.<|end_think|>* Sum is 1250.\n* So \\boxed{1250}"),
    ]

    records, chat_rows = grade_samples(problem, outputs, _config(AnswerCheck.NUMERIC), _character_count)

    assert [record["rejection_reason"] for record in records] == ["violated_constraint", "wrong_answer", None]
    assert [record["correct"] for record in records] == [False, False, True]
    assert [row["id"] for row in chat_rows] == ["problem-00000-self02"]
