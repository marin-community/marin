# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain-independent Taskforge coordinator for hydrated open-ended bundles."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NoReturn

from taskcompendium.models import VerifyitGrader, verifyit_spec
from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage30_author.author import static_author
from taskforge.stage30_author.program import RLEnvBuildProgram
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import (
    ResearchRequirement,
    TaskDraft,
    build_model_digest,
    run_build,
    step_output,
)
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Reject, RejectKind
from taskforge.stage_coordinator.coordinator import STORE_DIR, StageCoordinator, WorkpieceBudget
from taskforge.stage_coordinator.event_log import JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import JudgeSpec

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_task_design import (
    OpenEndedTaskDesign,
    calibrate_controls,
    open_ended_task_program,
)


def verifyit_control_scorer(
    connection: JudgeConnection,
) -> Callable[[JudgeSpec, str], Awaitable[float]]:
    """Adapt one explicit OpenAI-compatible endpoint to control calibration."""

    async def score(spec: JudgeSpec, candidate: str) -> float:
        reward = await asyncio.to_thread(
            grade_judge_candidate,
            spec,
            candidate,
            connection=connection,
            runtime=None,
        )
        return reward.reward

    return score


async def _no_repair(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("the generic open-ended program has no triage repair stage")


def open_ended_coordinator(
    *,
    root: Path,
    bundle: OpenEndedTaskBundle,
    model: ModelClient,
    sampling: LLMPolicy,
    services: RLEnvBuildServices,
    judge_model: str | None = None,
    control_scorer: Callable[[JudgeSpec, str], Awaitable[float]] | None = None,
    control_concurrency: int = 2,
    budget: WorkpieceBudget = WorkpieceBudget(
        max_triage_repairs=0,
        max_build_revisions=0,
        max_validation_retries=0,
    ),
) -> StageCoordinator:
    """Author, materialize, calibrate, and review tasks without domain-specific code."""
    program = open_ended_task_program()

    async def validate(draft: TaskDraft) -> dict[str, object]:
        design, judge = _validate_draft(draft, bundle, program, root)
        calibration = None
        if control_scorer is not None:
            calibration = await calibrate_controls(
                design,
                lambda answer: control_scorer(judge, answer),
                max_concurrency=control_concurrency,
            )
        return {
            "kind": "open_ended_validation",
            "task_digest": digest(draft.task),
            "design_digest": digest(design),
            "output_budget_tokens": design.output_budget_tokens,
            "control_count": len(design.controls),
            "calibration": None if calibration is None else calibration.model_dump(mode="json"),
        }

    async def review(summary: object) -> Decision:
        if not isinstance(summary, dict) or summary.get("kind") != "open_ended_validation":
            raise ValueError("open-ended review received another validation summary")
        calibration = summary.get("calibration")
        if isinstance(calibration, dict) and calibration.get("passed") is not True:
            failed = [
                result["control_id"]
                for result in calibration.get("results", [])
                if isinstance(result, dict) and result.get("passed") is not True
            ]
            return Reject(kind=RejectKind.TASK, reasons=(f"controls outside expected bands: {failed}",))
        return Accept(band=BandOutcome.IN_BAND)

    stages = StageFunctions(
        triage=preflight,
        repair=_no_repair,
        author=static_author(program),
        load_program=load_program,
        build=run_build,
        validate=validate,
        review=review,
    )
    return StageCoordinator(
        stages=stages,
        budget=budget,
        root=root,
        event_store=JsonEventStore(root),
        model=model,
        sampling=sampling,
        services=services,
        research=ResearchRequirement.OFF,
        model_digest=build_model_digest(model, sampling, services),
        build_input=bundle.build_input().model_copy(update={"judge_model": judge_model}).model_dump(mode="json"),
    )


def _validate_draft(
    draft: TaskDraft,
    bundle: OpenEndedTaskBundle,
    program: RLEnvBuildProgram,
    root: Path,
) -> tuple[OpenEndedTaskDesign, JudgeSpec]:
    task = draft.task
    if draft.provenance.program_digest != program.digest:
        raise ValueError("open-ended draft provenance names another program")
    if task.source != bundle.source or task.tags != bundle.tags:
        raise ValueError("open-ended task loses bundle provenance or tags")
    if task.resources.worker != bundle.solver_resources:
        raise ValueError("open-ended task worker resources differ from the bundle")
    if task.environment_requirements != bundle.environment_requirements:
        raise ValueError("open-ended task environment differs from the bundle")
    if not isinstance(task.grader, VerifyitGrader):
        raise ValueError("open-ended task does not have a verifyit judge grader")
    judge = verifyit_spec(task.grader)
    if not isinstance(judge, JudgeSpec):
        raise ValueError("open-ended task verifier is not an LLM judge")
    design = step_output(draft, str(root / STORE_DIR), "design", OpenEndedTaskDesign)
    if judge.references != (design.reference,):
        raise ValueError("open-ended task judge does not preserve the frozen reference")
    context = task.context.model_dump_json().encode()
    leaked_paths = [resource.path for resource in bundle.author_evidence if resource.path.encode() in context]
    if leaked_paths:
        raise ValueError(f"solver instruction exposes private evidence paths: {leaked_paths}")
    return design, judge
