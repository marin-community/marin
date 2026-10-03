# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep TaskTrove's puzzle and Reasoning Gym scoring semantics in direct chat."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import JsonValue, model_validator
from verifyit.grade import Status, grade
from verifyit.modes.grade_puzzle import PuzzleAnswerType, puzzle_spec
from verifyit.spec import ReasoningGymSpec, Spec

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.submission import extract_answer
from taskcompendium.verifiers.base import GradingAttempt, Verifier


def grade_answer(spec: Spec, answer: str, entry_json: str | None = None) -> GradeResult:
    """Run an existing output-file mode on an extracted assistant answer."""
    with TemporaryDirectory(prefix="taskcompendium-grade-") as directory:
        root = Path(directory)
        tests, workspace = root / "tests", root / "app"
        tests.mkdir()
        workspace.mkdir()
        (workspace / "answer.txt").write_text(answer)
        if entry_json is not None:
            (tests / "entry.json").write_text(entry_json)
        try:
            result = grade(spec, tests, workspace)
        except ImportError as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"Missing grader dependency: {error}")
    if result.status != Status.SCORED or result.detail.get("reason") == "scorer_error":
        return GradeResult(Outcome.INFRA_ERROR, None, json.dumps(result.detail))
    return GradeResult(Outcome.GRADED, result.reward)


class ReasoningGymVerifier(Verifier):
    dataset: str
    entry: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_entry(self) -> "ReasoningGymVerifier":
        metadata = self.entry.get("metadata")
        answer = self.entry.get("answer")
        if not isinstance(metadata, dict) or metadata.get("source_dataset") != self.dataset:
            raise ValueError("The entry must name the same Reasoning Gym dataset")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The entry must contain a nonempty reference answer")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            answer = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        return grade_answer(ReasoningGymSpec(dataset=self.dataset), answer, json.dumps(self.entry))


class PuzzleAnswerVerifier(Verifier):
    expected: str
    answer_type: PuzzleAnswerType

    @model_validator(mode="after")
    def validate_expected(self) -> "PuzzleAnswerVerifier":
        if not self.expected.strip():
            raise ValueError("A puzzle reference answer is required")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            answer = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        return grade_answer(puzzle_spec(self.expected, self.answer_type), answer)
