# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import bounded, answer-only NeMo Gym rows with shared verifier contracts."""

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from tasktrove_verify.spec import RUBRIC_REFERENCE, MathType, Mode

from taskcompendium.models import (
    AnswerRequirements,
    Embedded,
    JudgeConfig,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskTroveVerifier,
)


class StaticCorpus(StrEnum):
    """The answer-only NeMo source families reviewed for shared verification."""

    MCQA = "mcqa"
    OPEN_MATH = "open_math"
    STACK_MATH = "stack_math"
    OPEN_QA = "open_qa"
    SCIENCE = "science"
    REASONING_GYM = "reasoning_gym"


_DATASETS = {
    StaticCorpus.MCQA: ("nvidia/Nemotron-RL-knowledge-mcqa", "62a1eec1f952723eab2ee3832222f533b8138067"),
    StaticCorpus.OPEN_MATH: ("nvidia/Nemotron-RL-math-OpenMathReasoning", "5ef69384b65f13f08c35b73ddd5e7bf5e4621043"),
    StaticCorpus.STACK_MATH: ("nvidia/Nemotron-RL-math-stack_overflow", "be489b25f36ef92864546a7ace22edec4e053ac3"),
    StaticCorpus.OPEN_QA: ("nvidia/Nemotron-RL-knowledge-openqa", "3604d4119623f2961c9cd0a3a5365e0cff0dd393"),
    StaticCorpus.SCIENCE: ("nvidia/Nemotron-RL-Science-v1", "a7f55756f14bdd16c6469b94601d86be15e4c4fc"),
    StaticCorpus.REASONING_GYM: ("nvidia/Nemotron-RL-ReasoningGym-v1", "ad2c929b2dfd64ca30afb4d60e2f69a5a1919c4d"),
}
IMPORTER_REVISION = "taskcompendium-nemo-static-v0.1"
_SOURCE_ROW_PATH = "source-row.json"
_SOURCE_PROVENANCE_PATH = "source-provenance.json"
_MCQA_PREFIX = "Answer the following multiple choice question."
_MATH_PREFIX = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."
_OPEN_QA_PREFIX = "Solve the following problem step by step. Put your answer inside \\boxed{}."
_OPEN_QA_SUFFIX = re.compile(r"\nRemember to put your answer inside \\boxed\{\}\.?\s*\Z")
_SCIENCE_PREFIX = "Provide a precise and accurate response based on your knowledge."
_SCIENCE_SUFFIX = re.compile(
    r"\nPlace the final answer at the end of your response in the format \*\*X\*\*, where X is the answer\.\s*\Z"
)
_OPTION = re.compile(r"(?m)^([A-Z])(?:\)|:)\s+")


def _row(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("source record must be a JSON object")
    return value


def _source(corpus: StaticCorpus, offset: int) -> Source:
    dataset, revision = _DATASETS[corpus]
    return Source(dataset, revision, str(offset), IMPORTER_REVISION)


def _prompt(row: Mapping[str, Any]) -> str:
    request = row.get("responses_create_params")
    if not isinstance(request, Mapping):
        raise ValueError("source record has no responses request")
    messages = request.get("input")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("static importer requires one source input message")
    message = messages[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise ValueError("static importer requires one user message")
    prompt = message.get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("source user message is empty")
    if request.get("tools") not in (None, []):
        raise ValueError("static importer does not accept source tool declarations")
    return prompt


def _remove_prefix(prompt: str, prefix: str) -> str:
    if not prompt.startswith(prefix):
        raise ValueError("unsupported source delivery template")
    _, separator, cleaned = prompt.partition("\n\n")
    if not separator or not cleaned.strip():
        raise ValueError("source delivery template has no task text")
    return cleaned.strip()


def _mcqa_prompt(prompt: str) -> tuple[str, int]:
    cleaned = _remove_prefix(prompt, _MCQA_PREFIX)
    choices = _OPTION.findall(cleaned)
    if choices != [chr(ord("A") + index) for index in range(len(choices))] or len(choices) < 2:
        raise ValueError("MCQA prompt does not contain a contiguous option alphabet")
    return cleaned, len(choices)


def _open_qa_prompt(prompt: str) -> str:
    cleaned = _remove_prefix(prompt, _OPEN_QA_PREFIX)
    cleaned = _OPEN_QA_SUFFIX.sub("", cleaned).strip()
    if not cleaned:
        raise ValueError("open QA prompt has no task text")
    return cleaned


def _science_prompt(prompt: str) -> str:
    cleaned = _remove_prefix(prompt, _SCIENCE_PREFIX)
    cleaned = _SCIENCE_SUFFIX.sub("", cleaned).strip()
    if not cleaned:
        raise ValueError("science prompt has no task text")
    return cleaned


def _expected(row: Mapping[str, Any], key: str) -> str:
    expected = row.get(key)
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"source record has no nonempty {key}")
    return expected.strip()


def _resources(data: bytes, corpus: StaticCorpus, split: str, offset: int) -> tuple[Resource, ...]:
    dataset, revision = _DATASETS[corpus]
    provenance = {"dataset": dataset, "revision": revision, "split": split, "offset": str(offset)}
    return (
        Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data)),
        Resource(
            _SOURCE_PROVENANCE_PATH,
            (ResourceRole.VERIFIER,),
            Embedded(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()),
        ),
    )


def _identity(row: Mapping[str, Any], offset: int) -> str:
    identifier = row.get("uuid")
    return identifier if isinstance(identifier, str) and identifier else str(offset)


def _reference_verifier(question: str, reference: str, judge: JudgeConfig | None) -> TaskTroveVerifier:
    if judge is None:
        raise ValueError("reference-answer rows require an explicit judge configuration")
    return TaskTroveVerifier(
        Mode.JUDGE,
        {"references": (reference,), "question": question, "rubric": RUBRIC_REFERENCE, "exact_gate": True},
        judge,
    )


def import_hub_row(
    data: bytes,
    *,
    corpus: StaticCorpus,
    split: str,
    offset: int,
    judge: JudgeConfig | None = None,
) -> TaskSpecification | Rejected:
    """Convert one reviewed static NeMo row while retaining its source privately."""
    source = _source(corpus, offset)
    try:
        row = _row(data)
        prompt = _prompt(row)
        task_id = f"nemo/{corpus.value}/{_identity(row, offset)}"
        if corpus is StaticCorpus.MCQA:
            instructions, options = _mcqa_prompt(prompt)
            expected = _expected(row, "expected_answer").upper()
            if len(expected) != 1 or not "A" <= expected <= chr(ord("A") + options - 1):
                raise ValueError("MCQA expected option is outside the visible option alphabet")
            verifier = TaskTroveVerifier(Mode.MCQ, {"expected": expected, "options": options})
            competencies = ("multiple-choice-reasoning",)
        elif corpus in {StaticCorpus.OPEN_MATH, StaticCorpus.STACK_MATH}:
            instructions = _remove_prefix(prompt, _MATH_PREFIX)
            verifier = TaskTroveVerifier(
                Mode.MATH, {"expected": _expected(row, "expected_answer"), "math_type": MathType.SCALAR}
            )
            competencies = ("math",)
        elif corpus is StaticCorpus.OPEN_QA:
            instructions = _open_qa_prompt(prompt)
            verifier = _reference_verifier(instructions, _expected(row, "expected_answer"), judge)
            competencies = ("open-domain-question-answering",)
        elif corpus is StaticCorpus.SCIENCE:
            instructions = _science_prompt(prompt)
            verifier = _reference_verifier(instructions, _expected(row, "expected_answer"), judge)
            competencies = ("science-reasoning",)
        elif corpus is StaticCorpus.REASONING_GYM:
            instructions = prompt.strip()
            expected = _expected(row, "answer")
            if offset == 1:
                verifier = TaskTroveVerifier(Mode.MATH, {"expected": expected, "math_type": MathType.SCALAR})
                competencies = ("math", "reasoning")
            elif offset == 0:
                verifier = _reference_verifier(instructions, expected, judge)
                competencies = ("constraint-reasoning",)
            else:
                raise ValueError("ReasoningGym acceptance is limited to reviewed answer-only offsets")
        else:
            raise ValueError(f"unsupported static corpus {corpus.value}")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    return TaskSpecification(
        id=task_id,
        requirements=TaskRequirements(),
        resources=_resources(data, corpus, split, offset),
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=instructions,
                verifier=verifier,
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )
