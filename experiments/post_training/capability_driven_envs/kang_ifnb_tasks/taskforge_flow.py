# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run deterministic IFN-beta tasks through the Taskforge workpiece lifecycle.

This local path performs proposal triage, immutable-input build, objective grader
canaries, review, event persistence, and accepted-task extraction. It is a
structural campaign: model solver/adversary trials remain a later release gate.
"""

import argparse
import asyncio
import base64
import hashlib
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

from experiments.post_training.capability_driven_envs.kang_ifnb_tasks.generate_tasks import (
    BIOLOGY_CARD_PATH,
    DATA_DICTIONARY_PATH,
    DATA_PATH,
    Candidate,
    candidates,
    read_counts,
    select_candidates,
    task_spec,
)

SAMPLING = LLMPolicy()


class ObjectiveBuildInput(BaseModel):
    """Immutable source data and candidate identity supplied to the build program."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_base64: str
    source_revision: str
    candidate_identity: str

    @property
    def data(self) -> bytes:
        return base64.b64decode(self.data_base64, validate=True)


class UnusedFactory:
    """Declared package-machine capability that deterministic builds never acquire."""

    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        del requirements, spec

    async def create(self, spec: MachineSpec) -> Machine:
        del spec
        raise AssertionError("the deterministic objective build must not acquire a machine")


class UnusedModel:
    """Model client proving that deterministic authoring makes no model calls."""

    model = "deterministic-objective-builder"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the deterministic objective build makes no model calls")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("the deterministic objective build makes no model calls")


@rl_env_build_sink
async def objective_task(context: RLEnvBuildContext, build_input: object) -> TaskSpec:
    """Recompute one TaskSpec from source counts rather than accepting a serialized answer."""
    del context
    hydrated = ObjectiveBuildInput.model_validate(build_input)
    matches = [item for item in candidates(read_counts(hydrated.data)) if item.identity == hydrated.candidate_identity]
    if len(matches) != 1:
        raise ValueError(f"candidate identity {hydrated.candidate_identity!r} matched {len(matches)} candidates")
    return task_spec(matches[0], hydrated.data, hydrated.source_revision)


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


def build_input(candidate: Candidate, data: bytes, source_revision: str) -> dict[str, object]:
    """Return the JSON build input whose digest Taskforge records and caches."""
    return ObjectiveBuildInput(
        data_base64=base64.b64encode(data).decode(),
        source_revision=source_revision,
        candidate_identity=candidate.identity,
    ).model_dump(mode="json")


def proposal(candidate: Candidate, source_revision: str) -> TaskProposal:
    """Create a grounded proposal containing the task contract but no hidden answer."""
    descriptions = {
        "Task": (
            f"{candidate.instruction} The answer must come from the mounted table and reference cards rather than "
            "outside cohort facts. Preserve the requested donor-level analysis and do not substitute individual "
            "cells as independent biological replicates."
        ),
        "Realism and workflow": (
            "The solver inspects donor-paired pseudobulk counts, checks the data dictionary, and applies the supplied "
            "IFN-beta biology card. This mirrors a compact single-cell differential-response workflow in which library "
            "normalization, pairing, and donor replication matter. The solver may use Python or shell tools, but the "
            "final deliverable is the declared JSON object."
        ),
        "Research plan": (
            "No live research is needed because the source counts, field definitions, cited biological background, "
            "and decision policy are pinned inputs. The builder must not browse for replacement values or infer "
            "missing cohort records. Any unsupported or degenerate configuration becomes a null candidate rather "
            "than a task completed from external knowledge."
        ),
        "Build plan": (
            "A deterministic build program decodes the immutable source input, validates count inequalities and "
            "unique donor-condition-cell-type-gene rows, regenerates the complete candidate portfolio, and selects "
            "exactly one stable candidate identity. It mounts the pseudobulk count table and reference cards as worker "
            "resources, computes the hidden answer, and creates an exact structured grader."
        ),
        "Grader design and controls": (
            "The grader parses the assistant response as JSON and compares its complete structure with ground truth "
            "computed during the build. A reference answer must receive reward one, while an empty JSON object and "
            "altered fields must receive zero. Numeric precision, categorical vocabulary, required keys, and extra "
            "keys are therefore part of the objective contract."
        ),
        "Risks and null conditions": (
            "Reject malformed counts, inadequate paired-donor support, duplicate pseudobulk rows, non-finite "
            "calculations, "
            "or a candidate identity that no longer resolves uniquely. Also reject tasks whose public resources do "
            "not contain every required input, whose answer leaks into the instruction, or whose apparent biological "
            "lesson reduces to an unsupported claim rather than the frozen decision policy."
        ),
    }
    body = "\n\n".join(f"## {heading}\n{text}" for heading, text in descriptions.items())
    header = ProposalHeader(
        id=f"kang-ifnb/{candidate.family}/{candidate.identity}",
        source=SourceRef(kind=SourceKind.REPO, ref="GEO:GSE96583", hash=source_revision),
        environment=Environment.REASONING,
        solver_machine=SolverMachine.CPYTHON,
        verification=Verification.CODE,
        composite=(),
        grounding=Grounding.GROUNDED,
        research_need=ResearchNeed.NOT_NEEDED,
        research_reason="the pinned pseudobulk count table and reference cards contain every required input",
        research=(),
        build=(BuildItem(name="task", description="deterministically built objective IFN-beta task"),),
        resources=(DATA_PATH, DATA_DICTIONARY_PATH, BIOLOGY_CARD_PATH),
        null_reason=None,
    )
    return parse(render(TaskProposal(header=header, body=body)))


def no_call(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("this deterministic campaign does not repair proposals")


def _attempt(task: TaskSpec, answer: str) -> GradingAttempt:
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )


async def validate_objective(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
    """Require one positive and one severe-negative objective grader canary."""
    grader = draft.task.grader
    if not isinstance(grader, VerifyitGrader):
        raise ValueError("objective task does not use a VerifyIt grader")
    spec = verifyit_spec(grader)
    if not isinstance(spec, StructuredExactSpec):
        raise ValueError("objective task does not use StructuredExactSpec")
    reference = json.dumps(spec.expected)
    positive = grade_answer(draft.task, _attempt(draft.task, reference))
    negative = grade_answer(draft.task, _attempt(draft.task, "{}"))
    if (positive.status, positive.reward) != (Outcome.GRADED, 1.0):
        raise ValueError(f"reference control did not score 1.0: {positive}")
    if (negative.status, negative.reward) != (Outcome.GRADED, 0.0):
        raise ValueError(f"severe-negative control did not score 0.0: {negative}")
    lowering = Lowering(
        LoweringStatus.LOWERED,
        "objective reference=1.0 and severe-negative=0.0",
        (),
        None,
        digest(draft.task),
    )
    return ValidationResult(context.workpiece_id, context.round, lowering, None, (), (), None)


async def accept_objective(result: ValidationResult, *, band: BandRules) -> Decision:
    """Accept only after ``validate_objective`` completed without raising."""
    del result, band
    return Accept(band=BandOutcome.IN_BAND)


def coordinator(root: Path, candidate: Candidate, data: bytes, source_revision: str) -> StageCoordinator:
    """Construct the local deterministic Taskforge coordinator for one candidate."""
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
        build_input=build_input(candidate, data, source_revision),
    )


def accepted_task(root: Path, workpiece_id: str) -> TaskSpec:
    """Load the accepted draft from a terminal workpiece."""
    events = JsonEventStore(root).load(workpiece_id) or ()
    state = derive_state(events)
    if state.terminal is not Terminal.ACCEPTED:
        raise ValueError(f"workpiece {workpiece_id} is {state.terminal}, not accepted")
    draft_dir = root / "workpieces" / workpiece_id / "rounds" / str(state.round) / DRAFT_DIR
    return load_draft(draft_dir).task


async def run_campaign(data: bytes, source_revision: str, count: int, root: Path) -> list[TaskSpec]:
    """Run or resume ``count`` candidates and return accepted TaskSpecs in stable order."""
    selected = select_candidates(candidates(read_counts(data)), count)
    if len(selected) < count:
        raise ValueError(f"requested {count} tasks but this table supports only {len(selected)} candidates")
    accepted = []
    for candidate in selected:
        expected = task_spec(candidate, data, source_revision)
        terminal = await run_workpiece(
            coordinator(root, candidate, data, source_revision),
            expected.id,
            proposal(candidate, source_revision),
        )
        if terminal is not Terminal.ACCEPTED:
            raise ValueError(f"{expected.id} ended {terminal}")
        accepted.append(accepted_task(root, expected.id))
    return accepted


def write_release(output: Path, tasks: list[TaskSpec], data: bytes, source_revision: str) -> None:
    """Write accepted TaskSpecs and the Taskforge authoring-release manifest."""
    output.mkdir(parents=True, exist_ok=False)
    payload = ("\n".join(task.model_dump_json() for task in tasks) + "\n").encode()
    (output / "accepted_tasks.jsonl").write_bytes(payload)
    manifest = {
        "format": "kang-ifnb-taskforge-authoring-release-v1",
        "status": "structural_only_unreviewed_do_not_publish",
        "source_revision": source_revision,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "accepted_tasks_sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(tasks),
        "task_ids": [task.id for task in tasks],
        "validation": "proposal triage, deterministic build, positive grader canary, severe-negative grader canary",
        "remaining_release_gate": "Taskforge solver and adversary trials plus human design review",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--root", type=Path, required=True, help="Durable local Taskforge event/workpiece root.")
    parser.add_argument("--output", type=Path, required=True, help="New accepted-task release directory.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    data = args.counts.read_bytes()
    tasks = asyncio.run(run_campaign(data, args.source_revision, args.count, args.root))
    write_release(args.output, tasks, data, args.source_revision)
    print(f"wrote {len(tasks)} Taskforge-accepted tasks to {args.output}")


if __name__ == "__main__":
    main()
