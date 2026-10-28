# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NUPA-Loose's representation-sensitive final-answer and aligned-digit protocol."""

from pathlib import Path

from verifyit.adapters.evalchemy_nupa import grade_nupa_answer
from verifyit.adapters.evalchemy_nupa_preparation import digit_parts, extract_answer, full_answer
from verifyit.grade import InvalidTask, Reward, Status, read_output, scored
from verifyit.spec import NupaAnswerFormat, NupaSpec


def validate_nupa(spec: NupaSpec) -> None:
    if not isinstance(spec.answer_format, NupaAnswerFormat) or full_answer(spec.expected, spec.answer_format) is None:
        raise InvalidTask("NUPA requires a valid reference in its declared representation")


def grade_nupa_candidate(spec: NupaSpec, candidate: str) -> Reward:
    """Retain source exact/digit/length/format metrics without numeric equivalence."""
    validate_nupa(spec)
    verdict = grade_nupa_answer(
        candidate,
        spec.expected,
        spec.answer_format,
        extract_answer=extract_answer,
        full_answer=full_answer,
        digit_parts=digit_parts,
    )
    if verdict.status is not Status.SCORED:
        return verdict
    names = {"format_valid": "format_valid_rate", "no_answer": "no_answer_rate"}
    metrics = {names.get(name, name): value for name, value in verdict.detail["metrics"].items()}
    return scored(verdict.reward, **{**verdict.detail, "metrics": metrics})


def grade(spec: NupaSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_nupa(spec)
    response = read_output(spec, workspace)
    if response is None:
        return scored(0.0, reason="no_output")
    return grade_nupa_candidate(spec, response)
