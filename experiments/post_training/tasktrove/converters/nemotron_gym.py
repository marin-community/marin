# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converters for the Nemotron-Gym adapter templates (``verifier_data.json`` driven graders).

These are the worked examples for agents writing further converters: read the per-task data
file, map its fields onto a :class:`VerifierSpec`, and pass the instruction through.
"""

import json

from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask
from experiments.post_training.tasktrove.taskbinary import INSTRUCTION, TaskFiles
from experiments.post_training.tasktrove.verifier_spec import (
    AnswerSpec,
    AnswerType,
    ImageTier,
    MathType,
    VerifierKind,
    VerifierSpec,
)

VERIFIER_DATA = "tests/verifier_data.json"


def _verifier_data(task: TaskFiles) -> dict:
    return json.loads(task.text(VERIFIER_DATA))


def _metadata(task: TaskFiles) -> dict:
    raw = task.get_text("metadata.json")
    return json.loads(raw) if raw else {}


def convert_mcqa(task: TaskFiles) -> ConvertedTask:
    """Knowledge MCQA: ``{"expected_answer": "C", "output_regex": ...}``.

    The output regex captured one alphanumeric character, so a multi-character gold answer was
    never matchable; those rows are rejected here rather than converted.
    """
    data = _verifier_data(task)
    expected = str(data["expected_answer"]).strip()
    if len(expected) != 1 or not expected.isalnum():
        raise ValueError(f"mcqa gold answer is not a single option letter: {expected!r}")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        verifier=VerifierSpec(
            kind=VerifierKind.ANSWER,
            answer=AnswerSpec(type=AnswerType.MCQ, expected=expected.upper()),
        ),
        tier=ImageTier.ANSWER,
        metadata=_metadata(task),
    )


def convert_math_boxed(task: TaskFiles) -> ConvertedTask:
    """Typed math answers: ``{"expected_answer": "...", "answer_type": "scalar" | "equation" | ...}``."""
    data = _verifier_data(task)
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        verifier=VerifierSpec(
            kind=VerifierKind.ANSWER,
            answer=AnswerSpec(
                type=AnswerType.MATH,
                expected=str(data["expected_answer"]),
                math_type=MathType(data.get("answer_type", "scalar")),
            ),
        ),
        tier=ImageTier.ANSWER,
        solution_files=task.under("solution/"),
        metadata=_metadata(task),
    )


# Template ids from templates.json (fingerprint run over the v4.15 tree, 2026-09-09).
CONVERTERS = {
    "c814af4f124d": convert_mcqa,  # laion__nemotron-gym-knowledge-mcqa-v2 (616,888)
    "5ee94cf985a9": (
        convert_math_boxed
    ),  # nemotron math family: stack-overflow, oracle-filtered, openmathreasoning, nemo-prism
}
