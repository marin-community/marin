# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Human- or model-authored task designs consumed by the open-ended builder."""

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskcompendium.models import AnswerType, PlainText, TaskSpec
from taskcompendium.runtime.resources import resource_bytes
from taskforge.spec.draft import assemble, judge_grader
from taskforge.spec.rubric import ItemRole, Rubric
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import RLEnvBuildProgram, rl_env_build_program, rl_env_build_sink
from taskforge.stage40_build_with_program.context import RLEnvBuildContext

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedBuildInput, OpenEndedTaskBundle

TASK_DESIGNS_FORMAT = "open-ended-task-designs/1"
WEIGHT_TOLERANCE = 1e-6


class ControlKind(StrEnum):
    """A behavior that pointwise task validation must distinguish."""

    REFERENCE = "reference"
    PARTIAL = "partial"
    SEVERE_NEGATIVE = "severe_negative"
    EVALUATOR_INJECTION = "evaluator_injection"
    VERBOSITY = "verbosity"


class OpenEndedControl(BaseModel):
    """One private calibration answer and its acceptable pointwise reward band."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_id: str = Field(min_length=1)
    kind: ControlKind
    answer: str = Field(min_length=1)
    minimum_reward: float = Field(ge=0, le=1)
    maximum_reward: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_band(self) -> Self:
        if self.minimum_reward > self.maximum_reward:
            raise ValueError(f"control {self.control_id!r} has an inverted reward band")
        return self


class OpenEndedTaskDesign(BaseModel):
    """A frozen instruction, grading contract, and controls for one proposal slot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_slot: int = Field(gt=0)
    instruction: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    rubric: Rubric
    controls: tuple[OpenEndedControl, ...] = Field(min_length=1)
    output_budget_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_grading_contract(self) -> Self:
        scored = [item for item in self.rubric.items if item.role is ItemRole.SCORED]
        weight = sum(item.weight for item in scored)
        if abs(weight - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"scored rubric weights sum to {weight}, not 1")
        nonzero_caps = [item.id for item in self.rubric.items if item.role is not ItemRole.SCORED and item.weight != 0]
        if nonzero_caps:
            raise ValueError(f"gate, cap, and disqualifier rubric items must have zero weight: {nonzero_caps}")
        kinds = [control.kind for control in self.controls]
        missing = sorted(str(kind) for kind in ControlKind if kind not in kinds)
        if missing:
            raise ValueError(f"task design lacks required control kinds: {missing}")
        ids = [control.control_id for control in self.controls]
        if len(set(ids)) != len(ids):
            raise ValueError("task design control IDs must be unique")
        references = [control for control in self.controls if control.kind is ControlKind.REFERENCE]
        if len(references) != 1 or references[0].answer != self.reference:
            raise ValueError("task design needs exactly one reference control whose answer equals reference")
        return self


class ControlResult(BaseModel):
    """One frozen control's observed score and expected acceptance band."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_id: str
    kind: ControlKind
    reward: float = Field(ge=0, le=1)
    minimum_reward: float = Field(ge=0, le=1)
    maximum_reward: float = Field(ge=0, le=1)
    passed: bool


class CalibrationSummary(BaseModel):
    """Complete calibration evidence for one frozen task design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_slot: int
    results: tuple[ControlResult, ...]
    passed: bool


class OpenEndedTaskDesigns(BaseModel):
    """All manually frozen task variants supported by one bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["open-ended-task-designs/1"] = TASK_DESIGNS_FORMAT
    designs: tuple[OpenEndedTaskDesign, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_slots(self) -> Self:
        slots = [design.proposal_slot for design in self.designs]
        if len(set(slots)) != len(slots):
            raise ValueError("task design proposal slots must be unique")
        return self

    def for_slot(self, slot: int) -> OpenEndedTaskDesign:
        """Return the design for ``slot`` or reject an unsupported proposal variant."""
        matches = [design for design in self.designs if design.proposal_slot == slot]
        if len(matches) != 1:
            raise ValueError(f"bundle has no unique frozen task design for proposal slot {slot}")
        return matches[0]


async def calibrate_controls(
    design: OpenEndedTaskDesign,
    score: Callable[[str], Awaitable[float]],
    *,
    max_concurrency: int = 2,
) -> CalibrationSummary:
    """Score every frozen control with bounded concurrency and check its declared band."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(control: OpenEndedControl) -> ControlResult:
        async with semaphore:
            reward = await score(control.answer)
        return ControlResult(
            control_id=control.control_id,
            kind=control.kind,
            reward=reward,
            minimum_reward=control.minimum_reward,
            maximum_reward=control.maximum_reward,
            passed=control.minimum_reward <= reward <= control.maximum_reward,
        )

    results = tuple(await asyncio.gather(*(run(control) for control in design.controls)))
    return CalibrationSummary(
        proposal_slot=design.proposal_slot,
        results=results,
        passed=all(result.passed for result in results),
    )


def load_task_designs(bundle: OpenEndedTaskBundle, resource_path: str) -> OpenEndedTaskDesigns:
    """Load frozen designs from one private bundle resource."""
    matches = [resource for resource in bundle.author_evidence if resource.path == resource_path]
    if len(matches) != 1:
        raise ValueError(f"bundle needs exactly one private task-design resource at {resource_path!r}")
    designs = OpenEndedTaskDesigns.model_validate_json(resource_bytes(matches[0]))
    private_paths = [resource.path for resource in bundle.author_evidence]
    leaked = [path for path in private_paths if any(path in design.instruction for design in designs.designs)]
    if leaked:
        raise ValueError(f"solver instructions expose private resource paths: {leaked}")
    return designs


def manual_task_program(
    bundle: OpenEndedTaskBundle,
    designs: OpenEndedTaskDesigns,
    *,
    judge_model: str | None = None,
) -> RLEnvBuildProgram:
    """Return a generic build program over manually frozen private task designs."""
    expected_bundle_digest = bundle.content_digest

    @rl_env_build_sink
    async def task(context: RLEnvBuildContext, proposal: TaskProposal, build_input: object) -> TaskSpec:
        hydrated = OpenEndedBuildInput.model_validate(build_input)
        if hydrated.bundle_digest != expected_bundle_digest:
            raise ValueError("build input does not belong to the task-design bundle")
        prefix = f"{hydrated.bundle_id}/"
        if not proposal.header.id.startswith(prefix):
            raise ValueError("proposal ID does not belong to the task-design bundle")
        try:
            slot = int(proposal.header.id.removeprefix(prefix))
        except ValueError as error:
            raise ValueError(f"proposal ID has no integer slot: {proposal.header.id!r}") from error
        design = designs.for_slot(slot)
        grader = judge_grader(design.rubric, design.reference, model=judge_model)
        return assemble(
            context.workpiece_id,
            design.instruction,
            AnswerType.TEXT,
            PlainText(),
            grader,
            hydrated.source,
            environment=hydrated.environment_requirements,
            files=hydrated.solver_resources,
            tags=hydrated.tags,
        )

    return rl_env_build_program(sink=task)
