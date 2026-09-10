# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron open-ended QA: an exact gate over reference answers, then an LLM judge.

The old grader (``tests/exact_gate`` plus ``rewardkit``) took one reference answer field that
varies by source: ``science-so-openq`` ships a single ``reference_answer`` string,
``knowledge-openqa`` ships a list under ``expected_answers``. Both feed the same gate: normalize
the candidate's boxed answer and compare it against the normalized references, falling back to an
LLM judge on a miss. This maps onto :class:`JudgeSpec` directly, with ``exact_gate=True``
reproducing the gate and the tool's own judge replacing ``rewardkit``.
"""


from tasktrove_verify.spec import JudgeSpec

from experiments.post_training.tasktrove.contract import OLD_GRADER_LINE, drop_dockerfile_lines
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata, verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

RESPONSE_OUTPUT = "/app/response.txt"
"""Every task in this template tells the agent to write here, not the tool's ``answer.txt`` default."""


def _references(data: dict) -> list[str]:
    """Reference answers, from ``expected_answers`` (a list) or ``reference_answer`` (a string)."""
    expected_answers = data.get("expected_answers")
    if isinstance(expected_answers, list):
        return [answer for answer in expected_answers if isinstance(answer, str) and answer.strip()]
    reference_answer = data.get("reference_answer")
    return [reference_answer] if isinstance(reference_answer, str) and reference_answer.strip() else []


def _subject_tag(data: dict) -> str:
    """``knowledge`` for the ``expected_answers`` shape, ``science`` for ``reference_answer``."""
    if isinstance(data.get("expected_answers"), list):
        return "knowledge"
    if isinstance(data.get("reference_answer"), str):
        return "science"
    return "openqa"


def convert_nemotron_openqa(task: TaskFiles) -> ConvertedTask | Rejected:
    data = verifier_data(task)
    references = _references(data)
    if not references:
        return Rejected(ConvertStatus.NULL_GRADER, "no non-empty reference answers")
    tags = tuple(dict.fromkeys(("qa", "openqa", "nemotron", _subject_tag(data))))
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=JudgeSpec(
            references=tuple(references),
            question=str(data.get("instruction", "")).strip(),
            exact_gate=True,
            output=RESPONSE_OUTPUT,
        ),
        dockerfile=drop_dockerfile_lines(task.text(DOCKERFILE), OLD_GRADER_LINE),
        tags=tags,
        metadata=metadata(task),
    )


CONVERTER = Converter(
    name="nemotron_openqa",
    keys=(
        ConverterKey("llm-judge-freeform", frozenset({"tests/test.sh", "tests/exact_gate", "tests/sitecustomize.py"})),
        ConverterKey("qa-short-answer", frozenset({"tests/test.sh", "tests/exact_gate", "tests/sitecustomize.py"})),
    ),
    convert=convert_nemotron_openqa,
)
