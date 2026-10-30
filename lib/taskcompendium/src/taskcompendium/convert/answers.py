# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversation tasks whose final answer a verifyit mode grades in process.

Each helper takes the public prompt and reference values that a declaration extracts from its row,
checks that they are usable, and returns a task with its grader set, or a rejection naming the
defect. ``evidence`` is source material kept with the grader for review, never shown to the solver.
"""

import json
from collections.abc import Mapping
from typing import Any

from verifyit.modes.extract import extract_boxed
from verifyit.numeric import numeric_literal
from verifyit.spec import (
    Constraint,
    ExactSpec,
    IfevalSpec,
    JsonSchemaSpec,
    MathSpec,
    MathType,
    McqSpec,
    NumericSpec,
    SchemaFormat,
    Spec,
)

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import inline_resource

EVIDENCE_PATH = "reference/source-evidence.json"


def source_defect(reason: str, detail: str) -> ImportRejection:
    return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason=reason, detail=detail)


def unsupported(reason: str, detail: str) -> ImportRejection:
    return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason=reason, detail=detail)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def evidence_resource(evidence: Mapping[str, Any]) -> TaskResource:
    """Source fields retained beside the grader for review and audit."""
    return inline_resource(EVIDENCE_PATH, json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True).encode())


def answer_task(
    row: RawRow,
    *,
    prompt: str,
    spec: Spec,
    answer_type: AnswerType = AnswerType.TEXT,
    evidence: Mapping[str, Any] | None = None,
    resources: tuple[TaskResource, ...] = (),
) -> TaskSpec:
    """A one-message conversation task graded in process by ``spec``."""
    verifier = (*resources, *((evidence_resource(evidence),) if evidence else ()))
    package = verifyit_package(spec, verifier)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(verifier=package.resources),
        answer_type=answer_type,
        answer_format=PlainText(),
        grader=package.grader,
    )


def math_type(expected: str) -> MathType:
    """Keep structured mathematical answers distinct from scalar values."""
    if expected.startswith("[") and expected.endswith("]"):
        return MathType.LIST
    if expected.startswith(("[", "(")) and expected.endswith(("]", ")")) and "," in expected:
        return MathType.TUPLE if expected.startswith("(") and expected.endswith(")") else MathType.INTERVAL
    if expected.startswith((r"\{", "{")):
        return MathType.SET
    if "=" in expected or r"\approx" in expected:
        return MathType.EQUATION
    return MathType.SCALAR


def math_answer_task(
    row: RawRow, *, prompt: Any, answer: Any, evidence: Mapping[str, Any] | None = None
) -> TaskSpec | ImportRejection:
    """Grade a symbolic math answer; a boxed reference is unwrapped."""
    problem, reference = _text(prompt), _text(answer)
    if problem is None:
        return source_defect("missing_prompt", "The problem must be a nonempty string")
    if reference is None:
        return source_defect("invalid_reference", "The reference answer must be a nonempty string")
    expected = extract_boxed(reference) or reference
    return answer_task(
        row, prompt=problem, spec=MathSpec(expected=expected, math_type=math_type(expected)), evidence=evidence
    )


def numeric_answer_task(
    row: RawRow,
    *,
    prompt: Any,
    answer: Any,
    tolerance_abs: float,
    tolerance_rel: float,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec | ImportRejection:
    """Grade a numeric literal within absolute and relative tolerances."""
    problem = _text(prompt)
    if problem is None:
        return source_defect("missing_prompt", "The problem must be a nonempty string")
    reference = answer.strip() if isinstance(answer, str) else str(answer) if isinstance(answer, int | float) else None
    try:
        if reference is None:
            raise ValueError("missing")
        numeric_literal(reference)
    except ValueError:
        return source_defect("invalid_reference", f"The reference answer is not numeric: {answer!r}")
    return answer_task(
        row,
        prompt=problem,
        spec=NumericSpec(reference, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel),
        answer_type=AnswerType.NUMBER,
        evidence=evidence,
    )


def mcq_task(
    row: RawRow, *, prompt: Any, answer: Any, options: int, evidence: Mapping[str, Any] | None = None
) -> TaskSpec | ImportRejection:
    """Grade an option letter; ``prompt`` already lists the ``options`` labeled choices."""
    question, letter = _text(prompt), _text(answer)
    if question is None:
        return source_defect("missing_prompt", "The question must be a nonempty string")
    if letter is None or len(letter) != 1 or not "A" <= letter.upper() < chr(65 + options):
        return source_defect("invalid_reference", f"The key must be one of {options} option letters: {answer!r}")
    return answer_task(row, prompt=question, spec=McqSpec(letter.upper(), options=options), evidence=evidence)


def exact_answer_task(
    row: RawRow,
    *,
    prompt: Any,
    answers: tuple[str, ...],
    ignore_case: bool,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec | ImportRejection:
    """Grade an answer equal to one of ``answers`` after whitespace normalization."""
    question = _text(prompt)
    if question is None:
        return source_defect("missing_prompt", "The question must be a nonempty string")
    if not answers or not all(_text(item) for item in answers):
        return source_defect("invalid_reference", "Accepted answers must be nonempty strings")
    return answer_task(row, prompt=question, spec=ExactSpec(tuple(answers), ignore_case=ignore_case), evidence=evidence)


def ifeval_task(
    row: RawRow,
    *,
    prompt: Any,
    constraints: tuple[Constraint, ...],
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec | ImportRejection:
    """Grade the fraction of instruction-following constraints the reply satisfies."""
    instruction = _text(prompt)
    if instruction is None:
        return source_defect("missing_prompt", "The instruction must be a nonempty string")
    if not constraints:
        return unsupported("invalid_constraints", "At least one constraint is required")
    return answer_task(row, prompt=instruction, spec=IfevalSpec(constraints=constraints), evidence=evidence)


def json_schema_task(
    row: RawRow,
    *,
    prompt: Any,
    schema: Any,
    schema_format: SchemaFormat,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec | ImportRejection:
    """Grade a reply against ``schema`` (serialized schema text), shipped as the grader's ``schema.json``."""
    instruction, document = _text(prompt), _text(schema)
    if instruction is None:
        return source_defect("missing_prompt", "The instruction must be a nonempty string")
    if document is None:
        return source_defect("invalid_schema", "The schema must be nonempty text")
    return answer_task(
        row,
        prompt=instruction,
        spec=JsonSchemaSpec(schema="schema.json", format=schema_format),
        evidence=evidence,
        resources=(inline_resource("schema.json", document.encode()),),
    )
