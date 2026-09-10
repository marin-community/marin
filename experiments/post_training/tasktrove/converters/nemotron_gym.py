# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converters for the Nemotron-Gym adapter templates (``verifier_data.json`` driven graders).

These are the worked examples for agents writing further converters: read the per-task data
file, map its fields onto one spec, pass the instruction and Dockerfile through, set the tags.
"""

import re

from tasktrove_verify.spec import MathSpec, MathType, McqSpec

from experiments.post_training.tasktrove.converters.answer_solution import answer_solution
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata, verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

_OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[\.\):]\s", re.MULTILINE)
_MAX_OPTIONS = 10


def _option_count(instruction: str) -> int:
    letters = {m.group(1) for m in _OPTION_LINE.finditer(instruction)}
    return max((ord(letter) - ord("A") + 1 for letter in letters), default=_MAX_OPTIONS)


def convert_mcqa(task: TaskFiles) -> ConvertedTask | Rejected:
    """Knowledge MCQA: ``{"expected_answer": "C", "output_regex": ...}``.

    The output regex captured one alphanumeric character, so a multi-character gold answer was never
    matchable; those rows are rejected rather than converted.
    """
    data = verifier_data(task)
    expected = str(data["expected_answer"]).strip()
    if len(expected) != 1 or not expected.isalpha():
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"mcqa gold answer is not one option letter: {expected!r}")
    instruction = task.text(INSTRUCTION)
    spec = McqSpec(expected=expected.upper(), options=_option_count(instruction))
    return ConvertedTask(
        instruction=instruction,
        spec=spec,
        dockerfile=task.text(DOCKERFILE),
        tags=("qa", "mcq", "nemotron"),
        solution_files=answer_solution(spec),
        metadata=metadata(task),
    )


def convert_math_boxed(task: TaskFiles) -> ConvertedTask | Rejected:
    """Typed math answers: ``{"expected_answer": "...", "answer_type": "scalar" | "equation" | ...}``."""
    data = verifier_data(task)
    expected = str(data["expected_answer"]).strip()
    if not expected:
        return Rejected(ConvertStatus.NULL_GRADER, "empty expected_answer")
    spec = MathSpec(expected=expected, math_type=MathType(data.get("answer_type", "scalar")))
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=task.text(DOCKERFILE),
        tags=("math", "nemotron"),
        solution_files=task.under("solution/") or answer_solution(spec),
        metadata=metadata(task),
    )


CONVERTERS = (
    Converter(
        name="nemotron_mcqa",
        keys=(
            ConverterKey(
                "qa-short-answer",
                frozenset({"tests/test.sh", "tests/validate_verifier_data.py", "tests/verifier.py"}),
            ),
        ),
        convert=convert_mcqa,
    ),
    Converter(
        name="nemotron_math",
        keys=(ConverterKey("math-answer", frozenset({"tests/test.sh", "tests/verifier.py"})),),
        convert=convert_math_boxed,
    ),
)
