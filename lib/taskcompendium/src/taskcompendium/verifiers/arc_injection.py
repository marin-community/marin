# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskSpec adapters for ARC grids, ARC transforms and indirect injections."""

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator
from verifyit.modes.grade_arc import grade_arc_grid, validated_grid
from verifyit.modes.grade_injection import grade_indirect_injection

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.verifiers.base import GradingAttempt, Verifier, grade_extracted, grade_result


class ArcGridVerifier(Verifier):
    expected_output: list[list[int]]

    @field_validator("expected_output")
    @classmethod
    def validate_grid(cls, grid: list[list[int]]) -> list[list[int]]:
        return validated_grid(grid)

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        return grade_extracted(attempt, lambda text: grade_result(grade_arc_grid(text, self.expected_output)))


class ArcCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    input: list[list[int]]
    output: list[list[int]]

    @field_validator("input", "output")
    @classmethod
    def validate_grid(cls, grid: list[list[int]]) -> list[list[int]]:
        return validated_grid(grid)


class ArcTransformVerifier(Verifier):
    """Retain the true source contract until its isolated Python runtime is bound."""

    test_cases: list[ArcCase]
    source_grader_sha256: str
    submission_paths: tuple[str, ...] = ("/app/solution.py", "/app/answer.txt")

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        del attempt
        return GradeResult(Outcome.INFRA_ERROR, None, "ARC transform requires an isolated source Python grader")


class IndirectInjectionVerifier(Verifier):
    """Match the source's avoided call, including its permissive non-target scoring."""

    target_tool: str
    target_args: dict[str, JsonValue]
    available_tools: tuple[str, ...]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        return grade_extracted(
            attempt, lambda text: grade_result(grade_indirect_injection(text, self.target_tool, self.target_args))
        )
