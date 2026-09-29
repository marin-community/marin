# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt TaskTrove's pinned verifier contract to TaskCompendium submissions."""

from typing import Self

from pydantic import Field, model_validator
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.modes.grade_mcq import grade_mcq_candidate
from tasktrove_verify.spec import McqSpec, Spec, parse_spec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import extract_answer


class TaskTroveVerifier(Verifier):
    """Use a TaskTrove verifier after TaskCompendium submission extraction.

    This direct-answer slice supports the MCQ mode. Executable modes need
    verifier-only resources and an isolated runtime before they can be added.
    """

    verifier_toml: str = Field(repr=False)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        contract = _contract(self.verifier_toml)
        if not isinstance(contract, McqSpec):
            raise ValueError("TaskTrove direct-answer verifier requires MCQ mode")
        try:
            grade_mcq_candidate(contract, contract.expected)
        except InvalidTask as error:
            raise ValueError(f"Invalid TaskTrove MCQ contract: {error}") from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.response, attempt.convention).strip()
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if len(candidate) != 1 or not "A" <= candidate.upper() <= "Z":
            return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
        contract = _contract(self.verifier_toml)
        assert isinstance(contract, McqSpec)
        result = grade_mcq_candidate(contract, candidate)
        return GradeResult(Outcome.GRADED, result.reward)


def tasktrove_verifier(verifier_toml: str) -> VerifierSpec:
    """Construct a TaskTrove verifier descriptor from its source contract."""
    verifier = TaskTroveVerifier(verifier_toml=verifier_toml)
    return VerifierSpec(kind=VerifierKind.TASKTROVE, parameters_json=verifier.model_dump_json())


def _contract(verifier_toml: str) -> Spec:
    try:
        return parse_spec(verifier_toml)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove verifier contract: {error}") from error
