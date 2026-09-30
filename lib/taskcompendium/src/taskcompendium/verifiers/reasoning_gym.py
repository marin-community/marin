# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade text answers with the source Reasoning Gym scorer."""

from importlib import import_module
from math import isfinite

from pydantic import JsonValue, model_validator

from taskcompendium.grading import GradeResult, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission, TextSubmission


class ReasoningGymAnswerVerifier(Verifier):
    """Apply a pinned Reasoning Gym scorer to a private generated entry."""

    dataset: str
    entry: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_entry(self) -> "ReasoningGymAnswerVerifier":
        if not self.dataset:
            raise ValueError("Reasoning Gym dataset is required")
        metadata = self.entry.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("source_dataset") != self.dataset:
            raise ValueError("Reasoning Gym entry dataset differs from its verifier")
        answer = self.entry.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Reasoning Gym entry requires a nonempty answer")
        return self

    async def grade(
        self, submission: Submission, *, specification: VerifierSpec, attempt: GradingAttempt
    ) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("Reasoning Gym verifier requires a text submission")
        candidate = submission.value.strip()
        if not candidate:
            return GradeResult(Outcome.GRADED, 0.0)
        reasoning_gym = import_module("reasoning_gym")
        try:
            score_answer = reasoning_gym.get_score_answer_fn(self.dataset)
        except ValueError as error:
            raise ValueError(f"Unknown Reasoning Gym dataset {self.dataset!r}") from error
        # reasoning-gym annotates this bound scorer as Callable[[], float], but it accepts answer and entry.
        # pyrefly: ignore[bad-argument-count]
        score = score_answer(candidate, self.entry)
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise TypeError(f"Reasoning Gym scorer returned {type(score).__name__}")
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Reasoning Gym scorer returned a score outside [0, 1]: {score!r}")
        return GradeResult(Outcome.GRADED, float(score))


def reasoning_gym_answer(dataset: str, entry: dict[str, JsonValue]) -> VerifierSpec:
    """Create a private Reasoning Gym verifier descriptor."""
    verifier = ReasoningGymAnswerVerifier(dataset=dataset, entry=entry)
    return VerifierSpec(
        kind=VerifierKind.REASONING_GYM_ANSWER,
        parameters_json=verifier.model_dump_json(),
    )
