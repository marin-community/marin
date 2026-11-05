# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run first-three BioTasks ideas through deterministic Taskforge workpieces."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from pydantic import BaseModel, ConfigDict
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from taskforge.content_hash import digest
from taskforge.defaults import DEFAULT_POLICY
from taskforge.ledger.records import DiscardingLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.sandbox.factories import MachineHost
from taskforge.stage10_proposal.model import (
    BuildItem,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    parse,
    render,
)
from taskforge.stage20_triage.preflight import preflight
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
from taskforge.stage50_validate.round import Lowering, LoweringStatus, ValidationContext, ValidationResult
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision
from taskforge.stage60_review.rules import BandRules
from taskforge.stage_coordinator.coordinator import StageCoordinator, run_workpiece
from taskforge.stage_coordinator.event_log import JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import Terminal, derive_state
from verifyit.spec import StructuredExactSpec

from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import (
    BioTaskIdea,
    ideas,
    task_spec,
)

SAMPLING = LLMPolicy()


class ObjectiveBuildInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idea_id: str


class UnusedFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        del requirements, spec

    async def create(self, spec: MachineSpec) -> Machine:
        del spec
        raise AssertionError("deterministic objective build must not acquire a machine")


class UnusedModel:
    model = "deterministic-biotasks-builder"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("deterministic objective build makes no model calls")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("deterministic objective build makes no model calls")


@rl_env_build_sink
async def objective_task(context: RLEnvBuildContext, build_input: object) -> TaskSpec:
    del context
    hydrated = ObjectiveBuildInput.model_validate(build_input)
    matches = [idea for idea in ideas() if idea.idea_id == hydrated.idea_id]
    if len(matches) != 1:
        raise ValueError(f"idea identity {hydrated.idea_id!r} matched {len(matches)} ideas")
    return task_spec(matches[0])


PROGRAM = rl_env_build_program(sink=objective_task)
SERVICES = RLEnvBuildServices(
    host=MachineHost.LAPTOP,
    factories={Backend.LOCAL.value: UnusedFactory()},
    images=None,
    web_tools=(),
    ledger=DiscardingLedger(),
    reasoning_replay=ReasoningReplay.KEEP,
    agent_limits=AgentLimits(max_turns=1, max_tokens=1024, max_calls=1),
)


def proposal(idea: BioTaskIdea) -> TaskProposal:
    task_text = idea.instruction or f"Null task candidate: {idea.null_reason}"
    body = "\n\n".join(
        (
            f"## Task\n{task_text}",
            "## Realism and workflow\n"
            "The task is adapted from one pinned BioTasks source document and uses only mounted, minimal sufficient "
            "resources. The solver inspects explicit biological tables and a narrow interpretation card, performs the "
            "declared calculation with shell or Python tools, and returns a structured result. This resembles a small "
            "auditable analysis step rather than asking for unsupported recall.",
            "## Research plan\n"
            "No live research is allowed because source interpretation, formulas, thresholds, and all task inputs are "
            "frozen before build. The builder must not browse for replacement facts or silently fill absent records. "
            "The source document supplies the analytical motivation, while the adapted packet supplies every value "
            "needed for the objective calculation and cautious interpretation.",
            "## Build plan\n"
            "Regenerate the idea by stable identity, verify its source digest and parameters, mount the declared "
            "resources, and construct an exact structured objective grader. The deterministic registry must resolve "
            "one idea only. Any change to the source, family, parameters, resources, expected result, or generator "
            "revision creates a distinct idea and final TaskSpec digest.",
            "## Grader design and controls\n"
            "The grader parses one complete JSON object and compares every required value with frozen ground truth. "
            "The reference answer must score one and an empty JSON object must score zero. Numeric rounding, category "
            "vocabulary, key presence, arrays, and interpretation limitations are part of the contract, so omitted or "
            "altered fields cannot receive accidental credit.",
            "## Risks and null conditions\n"
            "Reject missing biological substance, missing source data, answer leakage, ambiguous calculations, "
            "unsupported interpretation, or a task solvable only through outside knowledge. Also reject a source that "
            "offers generic software advice without biological evidence. Simulated or adapted inputs must be labeled, "
            "and model predictions must not be presented as experimental confirmation or biological mechanism.",
        )
    )
    header = ProposalHeader(
        id=idea.idea_id,
        source=SourceRef(kind=SourceKind.REPO, ref=idea.source_document_key, hash=idea.source_sha256),
        environment=Environment.REASONING,
        solver_machine=SolverMachine.PACKAGES,
        verification=Verification.CODE,
        composite=(),
        grounding=Grounding.GROUNDED,
        research_need=ResearchNeed.NOT_NEEDED,
        research_reason="the adapted resources contain every required input",
        research=(),
        build=(BuildItem(name="task", description="deterministically built objective BioTasks task"),),
        resources=tuple(resource.path for resource in idea.resources),
        null_reason=idea.null_reason,
    )
    return parse(render(TaskProposal(header=header, body=body)))


def no_call(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("this deterministic campaign does not repair proposals")


def _attempt(task: TaskSpec, answer: str) -> GradingAttempt:
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )


async def validate_objective(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
    grader = draft.task.grader
    if not isinstance(grader, VerifyitGrader):
        raise ValueError("objective task does not use VerifyIt")
    spec = verifyit_spec(grader)
    if not isinstance(spec, StructuredExactSpec):
        raise ValueError("objective task does not use StructuredExactSpec")
    positive = grade_answer(draft.task, _attempt(draft.task, json.dumps(spec.expected)))
    negative = grade_answer(draft.task, _attempt(draft.task, "{}"))
    if (positive.status, positive.reward) != (Outcome.GRADED, 1.0):
        raise ValueError(f"reference control did not score 1.0: {positive}")
    if (negative.status, negative.reward) != (Outcome.GRADED, 0.0):
        raise ValueError(f"negative control did not score 0.0: {negative}")
    lowering = Lowering(LoweringStatus.LOWERED, "objective reference=1 and negative=0", (), None, digest(draft.task))
    return ValidationResult(context.workpiece_id, context.round, lowering, None, (), (), None)


async def accept_objective(result: ValidationResult, *, band: BandRules) -> Decision:
    del result, band
    return Accept(band=BandOutcome.IN_BAND)


def coordinator(root: Path, idea: BioTaskIdea) -> StageCoordinator:
    stages = StageFunctions(
        triage=preflight,
        repair=no_call,
        author=static_author(PROGRAM),
        load_program=load_program,
        build=run_build,
        validate=validate_objective,
        review=accept_objective,
        evidence_gaps=lambda result: (),
    )
    model = UnusedModel()
    return StageCoordinator(
        stages=stages,
        policy=DEFAULT_POLICY.overlaid(
            {
                "budget.max_triage_repairs": 0,
                "budget.max_build_revisions": 0,
                "budget.max_validation_retries": 0,
                "budget.max_modify_rounds": 0,
            }
        ),
        root=root,
        event_store=JsonEventStore(root),
        model=model,
        sampling=SAMPLING,
        services=SERVICES,
        research=ResearchRequirement.OFF,
        model_digest=build_model_digest(model, SAMPLING, SERVICES),
        build_input=ObjectiveBuildInput(idea_id=idea.idea_id).model_dump(mode="json"),
    )


def _accepted_task(root: Path, idea_id: str) -> TaskSpec:
    events = JsonEventStore(root).load(idea_id) or ()
    state = derive_state(events)
    if state.terminal is not Terminal.ACCEPTED:
        raise ValueError(f"workpiece {idea_id} is {state.terminal}, not accepted")
    draft_dir = root / "workpieces" / idea_id / "rounds" / str(state.round) / DRAFT_DIR
    return load_draft(draft_dir).task


async def run_idea(root: Path, idea: BioTaskIdea) -> dict[str, object]:
    """Run or resume one idea and return its complete structural disposition."""
    terminal = await run_workpiece(coordinator(root, idea), idea.idea_id, proposal(idea))
    event_store = JsonEventStore(root)
    stored_events = json.loads(event_store.path(idea.idea_id).read_text())
    result: dict[str, object] = {
        "idea": idea.model_dump(mode="json"),
        "terminal": terminal.value,
        "events": stored_events["events"],
    }
    if terminal is Terminal.ACCEPTED:
        task = _accepted_task(root, idea.idea_id)
        result["task_id"] = task.id
        result["task_digest"] = digest(task)
        result["task_json"] = task.model_dump_json()
    return result
