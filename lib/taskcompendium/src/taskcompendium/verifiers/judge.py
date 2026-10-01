# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt TaskTrove's existing judge rubric to TaskCompendium submissions."""

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path

from pydantic import Field, model_validator
from tasktrove_verify.grade import Reward, Status
from tasktrove_verify.spec import (
    JUDGE_CONTEXT_LIMIT,
    RUBRIC_CHECKLIST,
    RUBRIC_REFERENCE,
    RUBRICS,
    JudgeRuntimeConfig,
    JudgeSpec,
    Spec,
    parse_spec,
    render_spec,
)

from taskcompendium.grading import GradeResult, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission, TextSubmission


class JudgeVerifier(Verifier):
    """Private TaskTrove judge rubric serialized as TOML with optional inline context."""

    verifier_toml: str = Field(min_length=1)
    context: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "JudgeVerifier":
        spec = parse_spec(self.verifier_toml)
        if not isinstance(spec, JudgeSpec):
            raise ValueError("Judge verifier requires a TaskTrove judge spec")
        if spec.rubric not in RUBRICS:
            raise ValueError(f"Unsupported judge rubric: {spec.rubric!r}")
        if spec.rubric == RUBRIC_REFERENCE and not any(reference.strip() for reference in spec.references):
            raise ValueError("Reference judge rubric requires at least one non-empty reference")
        if spec.rubric == RUBRIC_CHECKLIST and not any(criterion.strip() for criterion in spec.criteria):
            raise ValueError("Checklist judge rubric requires at least one non-empty criterion")
        if spec.context and self.context is None:
            raise ValueError("Judge verifier context file must be embedded as private context")
        if self.context is not None and len(self.context) > JUDGE_CONTEXT_LIMIT:
            raise ValueError(f"Judge context exceeds {JUDGE_CONTEXT_LIMIT} characters")
        if spec.context:
            path = Path(spec.context)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Judge context path must be relative to the private tests directory")
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("Judge verifier requires a text submission")
        try:
            return await asyncio.to_thread(_grade, self, submission, attempt.judge_runtime)
        except Exception as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")


def judge_answer(spec: JudgeSpec, *, context: str | None = None) -> VerifierSpec:
    """Build a private judge verifier from the shared TaskTrove spec and inline context."""
    if spec.context and context is None:
        raise ValueError("Judge context file must be supplied as private inline context")
    embedded_spec = replace(spec, context="judge-context.txt" if context is not None else "")
    verifier = JudgeVerifier(verifier_toml=render_spec(embedded_spec), context=context)
    return VerifierSpec(kind=VerifierKind.JUDGE, parameters_json=verifier.model_dump_json())


def _grade(verifier: JudgeVerifier, submission: TextSubmission, runtime: JudgeRuntimeConfig | None) -> GradeResult:
    # The judge module imports the optional OpenAI client, so load it only for judge tasks.
    from tasktrove_verify.modes.grade_judge import JudgeInfrastructureError  # noqa: PLC0415
    from tasktrove_verify.modes.grade_judge import grade as grade_judge  # noqa: PLC0415

    spec = parse_spec(verifier.verifier_toml)
    assert isinstance(spec, JudgeSpec)
    with tempfile.TemporaryDirectory(prefix="taskcompendium-judge-") as temporary_directory:
        root = Path(temporary_directory)
        tests_dir = root / "tests"
        tests_dir.mkdir()
        workspace = root / "workspace"
        workspace.mkdir()
        if spec.context:
            context_path = tests_dir / spec.context
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(verifier.context or "")
        output = workspace / "answer.txt"
        output.write_text(submission.value)
        attempt_spec: Spec = replace(spec, output=str(output))
        try:
            reward: Reward = grade_judge(attempt_spec, tests_dir, workspace, runtime)
        except JudgeInfrastructureError as error:
            return GradeResult(Outcome.INFRA_ERROR, None, str(error), error.evidence)
        except Exception as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")
    if reward.status is Status.SCORED:
        return GradeResult(Outcome.GRADED, reward.reward, evidence=reward.detail)
    return GradeResult(Outcome.INFRA_ERROR, None, str(reward.detail.get("error", reward.status.value)), reward.detail)
