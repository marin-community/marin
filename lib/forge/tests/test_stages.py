# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from jsonschema import Draft202012Validator, ValidationError
from lib.forge.tests.stage_fakes import CandidateEngine, Environments, RecordingHarness, SolverEngine, solver_model
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskcompendium.runtime.resources import inline_resource

from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.effects.issues import Hold, Restart
from forge.effects.log import EventLog
from forge.harness.protocol import Returned, Stopped, StopReason, Submitted
from forge.harness.tools.adversary import AdversaryReport
from forge.harness.web import FakeWeb
from forge.lowering import machine_backend, requirements
from forge.model import serialization as wire
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message
from forge.model.proposal import (
    REQUIRED_HEADINGS,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    render,
)
from forge.model.review import (
    Accept,
    AdversaryResult,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    ControlResult,
    Incomplete,
    Modify,
    Reject,
    TrialResult,
    ValidationResult,
)
from forge.model.workpiece import Conversation, Submission, Workpiece
from forge.policy import Policy, load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.bundle import Resources
from forge.resources.clock import Clock
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.ids import Ids
from forge.resources.inference import FakeInference, InferenceRequest, complete
from forge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from forge.resources.sandbox.factories import MachineHost
from forge.resources.sandbox.images import GRADER_BASE_IMAGE
from forge.stages.author import author
from forge.stages.exits import Decided, Triaged, TriageRejected
from forge.stages.review import review
from forge.stages.rules import default_review
from forge.stages.triage import triage
from forge.stages.validate import validate


@pytest.fixture
def workpiece() -> Workpiece:
    header = ProposalHeader(
        "sum",
        SourceRef(SourceKind.CAPABILITY, "arithmetic", "v1"),
        Environment.REASONING,
        SolverMachine.NONE,
        Verification.CODE,
        (),
        Grounding.UNVERIFIED,
        ResearchNeed.NOT_NEEDED,
        "synthetic",
        (),
        (),
        (),
        None,
    )
    body = "\n\n".join(f"## {heading}\n" + "Compute the exact sum using a script. " * 7 for heading in REQUIRED_HEADINGS)
    proposal = TaskProposal(header, body)
    controls = (
        Control(Candidate("12"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),
        Control(Candidate("9"), RewardBand(Outcome.GRADED, reward_max=0), ControlKind.NEGATIVE),
    )
    return Workpiece("test", proposal, submission=Submission(arithmetic_task(), controls, "authored"))


def rendered_files(workpiece: Workpiece, policy: Policy) -> tuple[TaskResource, ...]:
    return (inline_resource("workspace/notes", workpiece.proposal.header.id.encode()),)


def config(harness: RecordingHarness) -> ForgeConfig:
    return ForgeConfig(
        load_policy(DEFAULTS_DIR),
        harness,
        {},
        WorkspaceHydration((inline_resource("workspace/notes", b"static"),), rendered_files),
        {"author": {}},
        lambda submission: (),
        default_review,
        BandRules(0.25, 0.75, frozenset({BandOutcome.IN_BAND})),
        Budget(2, 2),
        {},
    )


def resources(ctx: Context, inference: FakeInference | None = None, restart: int | None = None) -> Resources:
    bundle = Resources(
        Environments(),
        inference or FakeInference(()),
        RolloutResource(SolverEngine(3, restart), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Clock(),
        Ids(),
        {},
        FakeWeb({}, {}),
    )
    for resource in (bundle.inference, bundle.rollouts, bundle.grading):
        ctx.resource(resource, resource.fingerprint)
    ctx.handle(InferenceRequest, partial(complete, notify=ctx.emit), uses=(bundle.inference,))
    ctx.handle(SolverRollout, partial(run_rollout, notify=ctx.emit), uses=(bundle.rollouts,))
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(bundle.grading,))
    return bundle


@pytest.mark.asyncio
async def test_author_hydrates_revision_and_preserves_saved_session(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    task = workpiece.submission.task
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"worker": (inline_resource("workspace/notes", b"revision"),)})
        }
    )
    submission = replace(workpiece.submission, task=task)
    saved = Conversation((Message("user", "Continue with feedback"),), ())
    harness = RecordingHarness(Submitted(submission, saved))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    result = await author(ctx, workpiece.with_(submission=submission, session=saved), config(harness), bundle)
    await ctx.drain()
    assert result == harness.result
    assert harness.session == saved
    assert harness.files[PurePosixPath("/workspace/notes")] == b"revision"
    assert wire.TASK_SPEC.validate_json(harness.files[PurePosixPath("/workspace/task.json")]) == task
    assert wire.CONTROLS.validate_json(harness.files[PurePosixPath("/workspace/controls.json")]) == submission.controls
    assert harness.files[PurePosixPath("/workspace/proposal.json")] == render(workpiece.proposal).encode()
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


@pytest.mark.asyncio
async def test_author_new_session_and_failed_harness_close_environment(workpiece: Workpiece) -> None:
    harness = RecordingHarness(Restart("machine gone"))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    with pytest.raises(Restart):
        await author(ctx, workpiece.with_(submission=None), config(harness), bundle)
    await ctx.drain()
    assert harness.session is not None
    assert harness.session.messages[-1].content == config(harness).policy.prompt("author.user").render(
        proposal=render(workpiece.proposal)
    )
    assert harness.files[PurePosixPath("/workspace/notes")] == b"sum"
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


@pytest.mark.asyncio
async def test_author_reference_describes_valid_wire_values_and_supported_environments(workpiece: Workpiece) -> None:
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    await author(ctx, workpiece, config(harness), resources(ctx))
    await ctx.drain()
    assert harness.session is not None and workpiece.submission is not None
    system = harness.session.messages[0].content
    environments, task_environments, schemas = (
        json.loads(block.split("```", 1)[0]) for block in system.split("```json\n")[1:]
    )
    task = json.loads(workpiece.submission.task.model_dump_json())
    task_schema = Draft202012Validator(schemas["TaskSpec"])
    task_schema.validate(task)
    task["context"]["events"][0]["type"] = "invalid_event_type"
    with pytest.raises(ValidationError):
        task_schema.validate(task)
    control = json.loads(wire.CONTROL.dump_json(workpiece.submission.controls[0]))
    control_schema = Draft202012Validator(schemas["Control"])
    control_schema.validate(control)
    del control["candidate"]
    with pytest.raises(ValidationError):
        control_schema.validate(control)
    resolved = tuple(EnvironmentRequirements.model_validate(value) for value in environments)
    assert resolved[0].docker_image == GRADER_BASE_IMAGE and resolved[1].docker_image is None
    assert tuple(machine_backend(value, MachineHost.IRIS).value for value in resolved) == ("gvisor", "shellsim")
    task_resolved = tuple(EnvironmentRequirements.model_validate(value) for value in task_environments)
    assert all(value.capabilities == ("shell",) for value in task_resolved)
    assert tuple(machine_backend(value, MachineHost.IRIS).value for value in task_resolved) == ("gvisor", "shellsim")


@pytest.mark.asyncio
async def test_validate_collects_controls_and_concurrent_trials_despite_one_restart(workpiece: Workpiece) -> None:
    ctx = Context(log=EventLog())
    bundle = resources(ctx, restart=1)
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    result = (await validate(ctx, workpiece, config(harness), bundle)).result
    await ctx.drain()
    assert workpiece.submission is not None and result.task_digest == workpiece.submission.digest
    assert [(c.index, c.met, c.grade.reward if c.grade else None) for c in result.controls] == [
        (0, True, 1),
        (1, True, 0),
    ]
    assert [(t.trial, t.cause) for t in result.solver] == [(0, None), (1, Cause.MODEL_UNAVAILABLE), (2, None)]
    assert result.solver[1].grade is None
    assert isinstance(default_review(result, config(harness).band), Incomplete)


@pytest.mark.asyncio
async def test_validate_adversary_reports_are_graded_and_environment_closed(workpiece: Workpiece) -> None:
    session = Conversation((), ())
    harness = RecordingHarness(
        Returned("report", AdversaryReport(("wrong answer accepted",), (Candidate("9"),)), session)
    )
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    assert evidence.adversary == (
        AdversaryResult("finder", True, ("wrong answer accepted",), (GradeResult(Outcome.GRADED, 0),)),
    )
    assert isinstance(default_review(evidence, setup.band), Modify)
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


@pytest.mark.parametrize(
    "cause,met,completed,decision",
    [
        (None, True, True, Accept),
        (None, False, True, Modify),
        (Cause.MODEL_TIMEOUT, False, True, Incomplete),
        (None, False, False, Incomplete),
        (Cause.BUDGET_EXHAUSTED, True, True, Reject),
    ],
)
@pytest.mark.asyncio
async def test_review_prioritizes_incomplete_evidence_over_task_defects(
    workpiece: Workpiece, cause: Cause | None, met: bool, completed: bool, decision: type
) -> None:
    assert workpiece.submission is not None
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, GradeResult(Outcome.GRADED, 1), cause), TrialResult(1, GradeResult(Outcome.GRADED, 0))),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), met),),
        (AdversaryResult("finder", completed, ()),),
    )
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    result = await review(ctx, workpiece.with_(evidence=(evidence,)), config(harness), resources(ctx))
    assert isinstance(result, Decided) and isinstance(result.decision, decision)


@pytest.mark.parametrize("repair_budget", [0, 1])
@pytest.mark.asyncio
async def test_triage_repairs_sections_only_when_budget_allows(workpiece: Workpiece, repair_budget: int) -> None:
    inference = FakeInference((Completion(render(workpiece.proposal)),))
    ctx = Context(log=EventLog())
    bundle = resources(ctx, inference)
    damaged = workpiece.with_(proposal=replace(workpiece.proposal, body="## Task\nToo short"))
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    setup = replace(setup, policy=setup.policy.overlaid({"budget.max_triage_repairs": repair_budget}))
    result = await triage(ctx, damaged, setup, bundle)
    await ctx.drain()
    if repair_budget:
        assert isinstance(result, Triaged) and result.proposal.header == workpiece.proposal.header
    else:
        assert isinstance(result, TriageRejected) and result.reasons
    assert len(inference.requests) == repair_budget
    assert isinstance(await triage(ctx, workpiece, setup, bundle), Triaged)
    assert len(inference.requests) == repair_budget
    await ctx.drain()


@pytest.mark.asyncio
async def test_validate_returns_lowering_failure_without_starting_trials(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    task = workpiece.submission.task.model_copy(
        update={"environment_requirements": requirements(image="example.org/task@sha256:" + "0" * 64)}
    )
    workpiece = workpiece.with_(submission=replace(workpiece.submission, task=task))
    ctx = Context(log=EventLog())
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    result = (await validate(ctx, workpiece, config(harness), resources(ctx))).result
    assert result.lowering_error is not None
    assert not result.solver and not result.controls
    assert isinstance(default_review(result, config(harness).band), Incomplete)


@pytest.mark.asyncio
async def test_validate_observer_holds_for_grader_outage_but_preserves_authored_defect(workpiece: Workpiece) -> None:
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    for failure in (GradingFailure.MISSING_REWARD, GradingFailure.TIMEOUT, None):
        ctx = Context(log=EventLog())
        bundle = resources(ctx)
        status = Outcome.UNAVAILABLE if failure is None else Outcome.INFRA_ERROR
        grader = GradingResource(CandidateEngine(GradeResult(status, None, failure=failure)), "broken")
        ctx.resource(grader, grader.fingerprint)
        ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grader,))
        if failure is GradingFailure.TIMEOUT:
            with pytest.raises(ExceptionGroup) as caught:
                await validate(ctx, workpiece, setup, bundle)
            assert caught.value.subgroup(Hold) is not None
        else:
            result = (await validate(ctx, workpiece, setup, bundle)).result
            cause = Cause.NO_GRADE if failure is None else Cause.GRADER_MISSING_REWARD
            assert all(control.cause is cause for control in result.controls)
            assert isinstance(default_review(result, setup.band), Incomplete if failure is None else Modify)
        await ctx.drain()
