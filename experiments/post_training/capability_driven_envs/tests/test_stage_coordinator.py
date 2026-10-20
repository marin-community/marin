# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stage coordinator runs the catalog entry ``d43.culinary.scaling`` to an accepted task."""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

from collections.abc import Mapping, Sequence

from pydantic import BaseModel
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.models import TaskSpec
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.sandbox.factories import MachineHost
from taskforge.stage10_proposal.model import REQUIRED_HEADINGS, TaskProposal, parse
from taskforge.stage20_triage.program import Repair
from taskforge.stage20_triage.verdict import Verdict, accept
from taskforge.stage30_author.author import static_author
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink
from taskforge.stage40_build_with_program.context import RLEnvBuildContext
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import (
    DRAFT_DIR,
    ResearchRequirement,
    TaskDraft,
    build_model_digest,
    load_draft,
    run_build,
)
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision
from taskforge.stage_coordinator.coordinator import StageCoordinator, WorkpieceBudget, run_workpiece
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    example_catalog_path,
    load_capability_ideas,
)
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


def catalog_proposal(idea: CapabilityIdea) -> TaskProposal:
    """A reasoning proposal for ``idea``, sourced from its catalog record, with the capability's outcome as the task."""
    sections = {heading: "None." for heading in REQUIRED_HEADINGS}
    sections["Task"] = str(idea.capability["outcome"])
    body = "\n\n".join(f"## {heading}\n{text}" for heading, text in sections.items())
    return parse(
        f"""---
id: "{idea.capability_id}/0"
source: {{kind: capability, ref: "{idea.capability_id}", hash: "{idea.capability_hash}"}}
grounding: unverified
null_reason: null
environment: shellsim
solver_machine: shellsim
verification: simple
research_need: not_needed
research_reason: "the example program draws its own recipe"
research: []
build: []
resources: []
---
{body}
"""
    )


@rl_env_build_sink
async def task(context: RLEnvBuildContext, proposal: TaskProposal) -> TaskSpec:
    return task_spec(context.workpiece_id)


async def validate(draft: TaskDraft) -> dict[str, int]:
    return {"graded": 4, "solved": 2}


async def review(summary: object) -> Decision:
    return Accept(band=BandOutcome.IN_BAND)


async def no_repair(proposal: TaskProposal, verdict: Verdict) -> Repair:
    raise AssertionError("accept never asks for a repair")


async def test_the_catalog_entry_becomes_an_accepted_task_under_the_callers_workpiece_id(tmp_path):
    proposal = catalog_proposal(load_capability_ideas(example_catalog_path())[CAPABILITY])
    stages = StageFunctions(
        triage=accept,
        repair=no_repair,
        author=static_author(rl_env_build_program(sink=task)),
        load_program=load_program,
        build=run_build,
        validate=validate,
        review=review,
    )
    model = UnusedModel()
    sampling = LLMPolicy()
    services = RLEnvBuildServices(
        host=MachineHost.LAPTOP,
        factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
        images=None,
        web_tools=(),
        ledger=None,
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=AgentLimits(max_turns=1, max_tokens=1024, max_calls=1),
    )
    coordinator = StageCoordinator(
        stages=stages,
        budget=WorkpieceBudget(max_triage_repairs=0, max_build_revisions=0, max_validation_retries=0),
        root=tmp_path,
        event_store=JsonEventStore(tmp_path),
        model=model,
        sampling=sampling,
        services=services,
        research=ResearchRequirement.OFF,
        model_digest=build_model_digest(model, sampling, services),
    )

    assert await run_workpiece(coordinator, WORKPIECE, proposal, InMemoryEventLog()) is Terminal.ACCEPTED
    written = load_draft(tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / DRAFT_DIR).task
    assert written.id == WORKPIECE
    assert written.tags == (CAPABILITY,)
