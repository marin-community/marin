# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain upstream reward contracts whose execution is not bound locally."""

from pydantic import Field, JsonValue

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.verifiers.base import GradingAttempt, Verifier


class SourceContractVerifier(Verifier):
    """Preserve an identified evaluator and its private inputs for later binding."""

    evaluator: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    contract: dict[str, JsonValue]
    runtime_requirements: tuple[str, ...]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        del attempt
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            f"Source evaluator {self.evaluator} at {self.source_revision} is unbound; "
            f"requires {', '.join(self.runtime_requirements)}",
        )
