# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stage coordinator runs the catalog entry ``d43.culinary.scaling`` to an accepted task."""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

from collections.abc import Mapping, Sequence

from pydantic import BaseModel
from taskcompendium.models import TaskSpec
from taskforge.capability import capability_idea_record
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.stage20_triage.verdict import accept
from taskforge.stage30_author.author import static_author
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink
from taskforge.stage40_build_with_program.context import RLEnvBuildContext
from taskforge.stage40_build_with_program.run import DRAFT_DIR, TaskDraft, load_draft, run_build
from taskforge.stage40_build_with_program.step_store import model_digest
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision
from taskforge.stage_coordinator.coordinator import StageCoordinator, WorkpieceBudget, run_workpiece
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.catalog import example_catalog_path, load_capability_ideas
from experiments.post_training.capability_driven_envs.example_program import task_spec

CAPABILITY = "d43.culinary.scaling"
WORKPIECE = f"{CAPABILITY}--00"


class UnusedModel:
    model = "unused"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the example program calls no model")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("the example program calls no model")


@rl_env_build_sink
async def task(context: RLEnvBuildContext, proposal: dict[str, object]) -> TaskSpec:
    return task_spec(context.workpiece_id)


async def validate(draft: TaskDraft) -> dict[str, int]:
    return {"graded": 4, "solved": 2}


async def review(summary: object) -> Decision:
    return Accept(band=BandOutcome.IN_BAND)


@pytest.mark.asyncio
async def test_the_catalog_entry_becomes_an_accepted_task_under_the_callers_workpiece_id(tmp_path):
    proposal = capability_idea_record(load_capability_ideas(example_catalog_path())[CAPABILITY])
    stages = StageFunctions(
        triage=accept,
        author=static_author(rl_env_build_program(sink=task)),
        load_program=load_program_source,
        build=run_build,
        validate=validate,
        review=review,
    )
    model = UnusedModel()
    sampling = LLMPolicy()
    coordinator = StageCoordinator(
        stages=stages,
        budget=WorkpieceBudget(max_build_revisions=0, max_validation_retries=0),
        root=tmp_path,
        event_store=JsonEventStore(tmp_path),
        model=model,
        sampling=sampling,
        model_digest=model_digest(model, sampling),
    )

    assert await run_workpiece(coordinator, WORKPIECE, proposal, InMemoryEventLog()) is Terminal.ACCEPTED
    written = load_draft(tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / DRAFT_DIR).task
    assert written.id == WORKPIECE
    assert written.tags == (CAPABILITY,)
