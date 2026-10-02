# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned training math records scored by the existing typed math comparator."""

import json
from typing import Literal

from verifyit.modes.extract import extract_boxed

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.models import ImportRejection, RawRow, VerificationReport
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.verifiers.atlas_answers import MathAnswerVerifier


def answer_type(expected: str) -> Literal["scalar", "equation", "interval", "set", "tuple", "list"]:
    """Keep structured mathematical answers distinct from scalar values."""
    if expected.startswith("[") and expected.endswith("]"):
        return "list"
    if expected.startswith(("[", "(")) and expected.endswith(("]", ")")) and "," in expected:
        return "tuple" if expected.startswith("(") and expected.endswith(")") else "interval"
    if expected.startswith(r"\{") or expected.startswith("{"):
        return "set"
    if "=" in expected or r"\approx" in expected:
        return "equation"
    return "scalar"


def normalize_math(row: RawRow, problem_field: str, reference_field: str) -> TaskSpec | ImportRejection:
    problem, reference = row.data.get(problem_field), row.data.get(reference_field)
    if not isinstance(problem, str) or not problem.strip():
        return ImportRejection(reason="missing_prompt", detail=f"{problem_field} must be a nonempty string")
    if not isinstance(reference, str) or not reference.strip():
        return ImportRejection(reason="invalid_reference", detail=f"{reference_field} must be a nonempty string")
    expected = extract_boxed(reference) or reference.strip()
    verifier = MathAnswerVerifier(expected=expected, math_type=answer_type(expected))
    private = json.dumps(
        {key: row.data[key] for key in ("solution", "answer_type", "extracted_answer", "source") if key in row.data},
        ensure_ascii=False,
    ).encode()
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=problem),)),
        environment_requirements=EnvironmentRequirements(),
        resources=(task_resource("/reference/source-evidence.json", private, ResourceVisibility.VERIFIER),),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.MATH_ANSWER, parameters_json=verifier.model_dump_json()),
    )


def math_controls(task: TaskSpec) -> VerificationReport:
    verifier = MathAnswerVerifier.model_validate_json(task.verifier.parameters_json)
    return VerificationReport(
        checks=verify_witness(task, rf"\boxed{{{verifier.expected}}}", "__incorrect_math_answer__")
    )
