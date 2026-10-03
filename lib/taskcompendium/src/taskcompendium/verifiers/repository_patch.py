# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository repair contracts awaiting an isolated repository runtime."""

from pydantic import JsonValue

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.verifiers.base import GradingAttempt, Verifier


class RepositoryPatchVerifier(Verifier):
    """Retain the source repository and trusted-test contract without invented rewards."""

    repository: str
    source_ref: str
    workspace: str
    source_config: dict[str, JsonValue]
    source_grader_paths: tuple[str, ...]
    source_environment_sha256: str

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        del attempt
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Repository checkout, source environment, trusted tests, and patch capture are not bound",
        )
