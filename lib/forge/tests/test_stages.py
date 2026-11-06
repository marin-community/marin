# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from jsonschema import Draft202012Validator, ValidationError
from lib.forge.tests.stage_fakes import CandidateEngine, Environments, RecordingHarness, SolverEngine, solver_model
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskcompendium.runtime.resources import inline_resource

from forge.arms import decided, validated
from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.effects.issues import Hold, Restart
from forge.effects.log import EventLog
from forge.harness.protocol import Returned, Stopped, StopReason, Submitted
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport
from forge.harness.web import FakeWeb
from forge.lowering import machine_backend, requirements
from forge.model import serialization as wire
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, ToolCall
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
    AdversaryFinding,
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
from forge.model.workpiece import Conversation, Submission, Terminal, Workpiece
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
from forge.stages.exits import Decided, Triaged, TriageRejected, Validated
from forge.stages.review import review
from forge.stages.rules import default_review
from forge.stages.triage import triage
from forge.stages.validate import adversary_result, claim_result, control_result, validate


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
    author_client = FakeInference(())
    bundle = replace(resources(ctx), role_inference={"author": author_client})
    with pytest.raises(Restart):
        await author(ctx, workpiece.with_(submission=None), config(harness), bundle)
    await ctx.drain()
    assert harness.session is not None
    assert harness.client is author_client
    assert harness.session.messages[-1].content == config(harness).policy.prompt("author.user").render(
        proposal=render(workpiece.proposal)
    )
    assert harness.files[PurePosixPath("/workspace/notes")] == b"sum"
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


@pytest.mark.asyncio
async def test_author_reference_describes_valid_wire_values_and_supported_environments(workpiece: Workpiece) -> None:
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    setup = config(harness)
    setup = replace(setup, policy=setup.policy.overlaid({"sandbox.memory_mb": 16384}))
    await author(ctx, workpiece, setup, resources(ctx))
    await ctx.drain()
    assert harness.session is not None and workpiece.submission is not None
    system = harness.session.messages[0].content
    assert "memory limit: 16384 MiB" in system and "defaults to 30 seconds per command" in system
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
async def test_validate_propagates_solver_restart_for_stage_replay(workpiece: Workpiece) -> None:
    ctx = Context(log=EventLog())
    bundle = resources(ctx, restart=1)
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    with pytest.raises(Restart):
        await validate(ctx, workpiece, config(harness), bundle)
    await ctx.drain()
    assert isinstance(bundle.rollouts.engine, SolverEngine) and len(bundle.rollouts.engine.started) == 3


@pytest.mark.parametrize("reply,confirmed", [("9", False), ("12", True)])
@pytest.mark.asyncio
async def test_validate_adversary_reports_are_graded_and_environment_closed(
    workpiece: Workpiece, reply: str, confirmed: bool
) -> None:
    session = Conversation((), ())
    harness = RecordingHarness(
        Returned(
            "report",
            AdversaryReport((AdversaryClaim("wrong answer accepted", Candidate(reply), RewardBand(Outcome.GRADED, 1)),)),
            session,
        )
    )
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    adversary_client = FakeInference(())
    bundle = replace(resources(ctx), role_inference={"adversary": adversary_client})
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    finding = evidence.adversary[0].findings[0]
    assert evidence.adversary[0].completed and finding.candidate.reply == reply
    assert harness.client is adversary_client
    assert finding.grade is not None and finding.grade.reward == float(confirmed)
    assert finding.met is confirmed
    assert isinstance(default_review(evidence, setup.band), Modify if confirmed else Accept)
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    assert bundle.envs.requested[0].docker_image == GRADER_BASE_IMAGE


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


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "kind,message",
    [
        (Hold, "environment terminated: cpu_exhausted"),
        (Hold, "provider returned 3 empty completions"),
        (Restart, "model unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_adversary_environment_hold_is_incomplete_evidence(
    workpiece: Workpiece, grouped: bool, kind: type[Hold] | type[Restart], message: str
) -> None:
    failure = kind(message)
    harness = RecordingHarness(ExceptionGroup("observer", [failure]) if grouped else failure)
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    if kind is Restart:
        with pytest.raises((Restart, ExceptionGroup)) as caught:
            await validate(ctx, workpiece, setup, bundle)
        if isinstance(caught.value, ExceptionGroup):
            assert caught.value.subgroup(Restart) is not None
        await ctx.drain()
        return
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    assert evidence.adversary == (AdversaryResult("finder", False, (), (f"{kind.__name__}: {message}",)),)
    assert len(evidence.solver) == 3 and len(evidence.controls) == 2
    decision = default_review(evidence, setup.band)
    assert isinstance(decision, Incomplete)
    assert decision.reasons == (f"finder: {kind.__name__}: {message}",)
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


@pytest.mark.asyncio
async def test_invalid_grader_checks_request_task_repair(workpiece: Workpiece) -> None:
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    invalid = GradeResult(Outcome.GRADED, 1, detail={"checks": {"total": True}})
    grader = GradingResource(CandidateEngine(invalid), "invalid-checks")
    ctx.resource(grader, grader.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grader,))
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    assert all(control.cause is Cause.GRADER_INVALID_CHECKS for control in evidence.controls)
    decision = default_review(evidence, setup.band)
    assert isinstance(decision, Modify)
    assert any("grader_invalid_checks" in finding for finding in decision.findings)


@pytest.mark.parametrize("reason", [StopReason.CONTEXT_LIMIT, StopReason.FINISHED])
@pytest.mark.asyncio
async def test_adversary_stop_reason_survives_review_and_serialization(workpiece: Workpiece, reason: StopReason) -> None:
    harness = RecordingHarness(Stopped(reason, Conversation((), ())))
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    evidence = (await validate(ctx, workpiece, setup, resources(ctx))).result
    await ctx.drain()
    restored = wire.WORKPIECE.validate_json(wire.WORKPIECE.dump_json(workpiece.with_(evidence=(evidence,))))
    assert restored.evidence[0].adversary[0].reasons == (f"{reason} after 0 turns without report",)
    decision = default_review(restored.evidence[0], setup.band)
    assert isinstance(decision, Incomplete)
    assert f"finder: {reason} after 0 turns without report" in decision.reasons


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


@pytest.mark.parametrize("revision", ["in_band", "too_easy", "control", "task_defect", "too_hard", "adversary"])
@pytest.mark.asyncio
async def test_one_hardening_attempt_accepts_or_reverts(workpiece: Workpiece, revision: str) -> None:
    assert workpiece.submission is not None
    original = workpiece.submission
    evidence = ValidationResult(
        original.digest,
        tuple(TrialResult(index, GradeResult(Outcome.GRADED, 1)) for index in range(3)),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), True),),
        (),
    )
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    # An earlier unrelated modification already consumed one round of the shared budget.
    current = workpiece.with_(
        round=1,
        session=Conversation((Message("system", "author"),), ()),
        evidence=(evidence,),
        decisions=(Modify(("fix initial controls",)),),
    )
    first = await review(ctx, current, setup, bundle)
    assert isinstance(first.decision, Modify) and first.decision.placement is BandOutcome.TOO_EASY
    assert "Make the task harder" in first.decision.findings[0]
    pending = decided(current, first, budget=setup.budget, policy=setup.policy)
    assert pending.round == 2 and pending.terminal is None
    assert pending.too_easy_baseline == original
    updated = replace(original, task=original.task.model_copy(update={"labels": {"revision": "harder"}}))
    solved = 2 if revision == "in_band" else 0 if revision == "too_hard" else 3
    controls = evidence.controls
    adversary = ()
    if revision == "control":
        controls = (replace(controls[0], met=False),)
    if revision == "task_defect":
        controls = (replace(controls[0], met=False, cause=Cause.GRADER_INVALID_CHECKS),)
    if revision == "adversary":
        finding = AdversaryFinding(
            "wrong answer accepted",
            Candidate("wrong"),
            RewardBand(Outcome.GRADED, 1),
            GradeResult(Outcome.GRADED, 1),
            True,
        )
        adversary = (AdversaryResult("finder", True, (finding,)),)
    revised = replace(
        evidence,
        task_digest=updated.digest,
        solver=tuple(TrialResult(index, GradeResult(Outcome.GRADED, float(index < solved))) for index in range(3)),
        controls=controls,
        adversary=adversary,
    )
    current = pending.with_(submission=updated, evidence=(*pending.evidence, revised))
    second = await review(ctx, current, setup, bundle)
    final = decided(current, second, budget=setup.budget, policy=setup.policy)
    restored = wire.WORKPIECE.validate_json(wire.WORKPIECE.dump_json(final))
    assert restored.terminal is Terminal.ACCEPTED and restored.round == 2
    assert isinstance(second.decision, Accept)
    assert sum(isinstance(item, Modify) for item in restored.decisions) == 2
    regressed = revision not in ("in_band", "too_easy")
    assert restored.submission == (original if regressed else updated)
    assert second.decision.revert is regressed
    if revision == "in_band":
        assert restored.accepted_with is None and second.decision.band is BandOutcome.IN_BAND
    else:
        assert restored.accepted_with is BandOutcome.TOO_EASY
        assert "accepted_with: too_easy" in second.decision.reasons
    await ctx.drain()


def test_too_easy_does_not_create_an_extra_modification_budget(workpiece: Workpiece) -> None:
    setup = config(RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ()))))
    current = workpiece.with_(round=setup.budget.max_modify_rounds)
    decision = Decided(Modify(("make harder",), placement=BandOutcome.TOO_EASY))
    final = decided(current, decision, budget=setup.budget, policy=setup.policy)
    assert final.terminal is Terminal.ACCEPTED and final.round == current.round
    assert final.submission == workpiece.submission and final.too_easy_baseline is None
    assert final.accepted_with is BandOutcome.TOO_EASY and isinstance(final.decisions[-1], Accept)
    assert "Shared modification budget exhausted" in final.decisions[-1].reasons


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


@pytest.mark.parametrize(
    "status,failure,cause",
    [
        (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD, Cause.GRADER_MISSING_REWARD),
        (Outcome.INFRA_ERROR, GradingFailure.TIMEOUT, Cause.GRADER_TIMEOUT),
        (Outcome.UNAVAILABLE, None, Cause.NO_GRADE),
        (Outcome.INFRA_ERROR, None, Cause.GRADER_INFRA),
    ],
)
@pytest.mark.asyncio
async def test_validate_observer_holds_for_grader_outage_but_preserves_authored_defect(
    workpiece: Workpiece, status: Outcome, failure: GradingFailure | None, cause: Cause
) -> None:
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    grader = GradingResource(
        CandidateEngine(
            GradeResult(
                status,
                None,
                failure=failure,
                diagnostics={"grader_elapsed_seconds": 300.0, "grader_timeout_seconds": 300.0},
            )
        ),
        "broken",
    )
    ctx.resource(grader, grader.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grader,))
    if cause is Cause.GRADER_INFRA:
        with pytest.raises(ExceptionGroup) as caught:
            await validate(ctx, workpiece, setup, bundle)
        assert caught.value.subgroup(Hold) is not None
    else:
        result = (await validate(ctx, workpiece, setup, bundle)).result
        assert all(control.cause is cause for control in result.controls)
        decision = default_review(result, setup.band)
        assert isinstance(decision, Incomplete if cause is Cause.NO_GRADE else Modify)
        if cause is Cause.GRADER_TIMEOUT:
            assert isinstance(decision, Modify) and any("grader_timeout" in finding for finding in decision.findings)
    await ctx.drain()


@pytest.mark.asyncio
async def test_adversary_prompt_uses_manifest_for_large_resources(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    payload = b"large-payload" * 50000
    task = workpiece.submission.task
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"all": (inline_resource("workspace/input.bin", payload),)})
        }
    )
    workpiece = workpiece.with_(submission=replace(workpiece.submission, task=task))
    harness = RecordingHarness(Returned("report", AdversaryReport(()), Conversation((), ())))
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    await adversary_result(ctx, workpiece, setup, resources(ctx), rollout_spec(task), "finder")
    await ctx.drain()
    assert harness.session is not None
    prompt = harness.session.messages[-1].content
    assert "memory limit: 8192 MiB" in harness.session.messages[0].content
    assert len(prompt) < 10000 and "content_base64" not in prompt
    assert f'"bytes":{len(payload)}' in prompt and '"role":"resources.all"' in prompt
    assert harness.files[PurePosixPath("/workspace/input.bin")] == payload
    assert wire.TASK_SPEC.validate_json(harness.files[PurePosixPath("/workspace/task.json")]) == task


@pytest.mark.asyncio
async def test_validate_summary_hydrates_full_evidence_before_author_resumes(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    grade = GradeResult(Outcome.GRADED, 1, diagnostics={"stdout": "first line\n" + "x" * 100000})
    finding = AdversaryFinding(
        "wrong answer accepted",
        Candidate("x", (inline_resource("answer.bin", b"z" * 100000),)),
        RewardBand(Outcome.GRADED, 1),
        grade,
        True,
    )
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, grade),),
        (ControlResult(0, grade, True),),
        (AdversaryResult("finder", True, (finding,)),),
    )
    session = Conversation((Message("assistant", "", (ToolCall("v1", "validate", "{}"),)),), ())
    resumed = validated(workpiece.with_(session=session), Validated(evidence), tool=True, output_chars=3000)
    assert resumed.session is not None
    message = resumed.session.messages[-1]
    assert message.tool_call_id == "v1" and len(message.content) <= 3000
    assert "first line" in message.content and "answer.bin" in message.content
    assert "content_base64" not in message.content and "/workspace/.validation/0.json" in message.content
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    await author(ctx, resumed, config(harness), resources(ctx))
    await ctx.drain()
    saved = harness.files[PurePosixPath("/workspace/.validation/0.json")]
    assert wire.VALIDATION_RESULT.validate_json(saved) == evidence


@pytest.mark.parametrize("claim", [False, True])
@pytest.mark.asyncio
async def test_control_and_claim_preemption_restarts_validation(workpiece: Workpiece, claim: bool) -> None:
    assert workpiece.submission is not None
    ctx = Context(log=EventLog())

    async def preempted(event: GradeCandidate) -> GradeResult:
        raise Restart("verifier preempted")

    ctx.handle(GradeCandidate, preempted)
    control = workpiece.submission.controls[0]
    spec = rollout_spec(workpiece.submission.task)
    with pytest.raises(Restart, match="verifier preempted"):
        if claim:
            await claim_result(ctx, spec, AdversaryClaim("wrong", control.candidate, control.expected))
        else:
            await control_result(ctx, spec, 0, control)
    await ctx.drain()
