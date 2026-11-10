# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator, ValidationError
from lib.forge.tests.stage_fakes import CandidateEngine, Environments, RecordingHarness, SolverEngine, solver_model
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from forge.arms import decided, validated
from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.effects.events import Note, Resolved
from forge.effects.issues import Hold, OutputBudgetExhausted, Restart
from forge.effects.log import EventLog
from forge.harness.loop import AgenticHarness
from forge.harness.protocol import Returned, Stopped, StopReason, Submitted
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport
from forge.lowering import machine_backend, requirements
from forge.model.controls import CONTROL, CONTROLS, Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, Role, ToolCall, Usage
from forge.model.proposal import (
    REQUIRED_HEADINGS,
    ResearchNeed,
    TaskProposal,
    render,
)
from forge.model.review import (
    VALIDATION_RESULT,
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
    Retry,
    TrialResult,
    ValidationResult,
)
from forge.model.workpiece import WORKPIECE, Conversation, Submission, Terminal, Workpiece
from forge.policy import GRADER_BASE_IMAGE, Policy, load_policy, sampling
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.bundle import Resources
from forge.resources.clock import Clock
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.ids import Ids
from forge.resources.inference import ContextExceeded, FakeInference
from forge.resources.rollouts import RolloutResource, SolverRollout
from forge.resources.web import FakeWeb
from forge.stages.author import author
from forge.stages.classify import grade_cause, replayable_grade
from forge.stages.exits import Decided, Triaged, TriageRejected, Validated
from forge.stages.review import review
from forge.stages.rules import default_review
from forge.stages.triage import triage
from forge.stages.validate import adversary_result, claim_result, control_result, solver_result, validate
from forge.wiring import build_config, install

pytest_plugins = ("lib.forge.tests.test_proposal",)


@pytest.fixture
def workpiece(proposal: TaskProposal) -> Workpiece:
    proposal = replace(
        proposal,
        header=replace(proposal.header, id="sum", research=(), research_need=ResearchNeed.NOT_NEEDED),
        body="\n\n".join(
            f"## {heading}\n" + "Compute the exact sum using a script. " * 7 for heading in REQUIRED_HEADINGS
        ),
    )
    controls = tuple(
        Control(Candidate(reply), RewardBand(Outcome.GRADED, reward_min=lo, reward_max=hi), kind)
        for reply, lo, hi, kind in (("12", 1, None, ControlKind.POSITIVE), ("9", None, 0, ControlKind.NEGATIVE))
    )
    return Workpiece("test", proposal, submission=Submission(arithmetic_task(), controls, "authored"))


def rendered_files(workpiece: Workpiece, policy: Policy) -> tuple[TaskResource, ...]:
    return (inline_resource("workspace/notes", workpiece.proposal.header.id.encode()),)


def config(harness: RecordingHarness | None = None) -> ForgeConfig:
    return replace(
        build_config(load_policy(DEFAULTS_DIR), "S", {}),
        author=harness or RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ()))),
        workspace=WorkspaceHydration((inline_resource("workspace/notes", b"static"),), rendered_files),
    )


def adversary_config(harness: RecordingHarness) -> ForgeConfig:
    setup = config(harness)
    return replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )


def resources(ctx: Context, restart: int | None = None, grader: CandidateEngine = CandidateEngine()) -> Resources:
    bundle = Resources(
        Environments(),
        FakeInference(()),
        RolloutResource(SolverEngine(3, restart), solver_model, "solver"),
        GradingResource(grader, "grader"),
        Clock(),
        Ids(),
        {},
        FakeWeb({}, {}),
    )
    install(ctx, bundle)
    return bundle


async def test_author_hydrates_revision_and_preserves_saved_session(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    task = workpiece.submission.task
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"worker": (inline_resource("workspace/notes", b"revision"),)})
        }
    )
    submission = replace(workpiece.submission, task=task)
    saved = Conversation((Message(Role.USER, "Continue with feedback"),), ())
    harness = RecordingHarness(Submitted(submission, saved))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    result = await author(ctx, workpiece.with_(submission=submission, session=saved), config(harness), bundle)
    await ctx.drain()
    assert result == harness.result
    assert harness.session == saved
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/notes")] == b"revision"
    assert TaskSpec.model_validate_json(bundle.envs.created[0].files[PurePosixPath("/workspace/task.json")]) == task
    assert (
        CONTROLS.validate_json(bundle.envs.created[0].files[PurePosixPath("/workspace/controls.json")])
        == submission.controls
    )
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/proposal.json")] == render(workpiece.proposal).encode()
    assert bundle.envs.created[0].closed


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
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/notes")] == b"sum"
    assert bundle.envs.created[0].closed


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
    control = json.loads(CONTROL.dump_json(workpiece.submission.controls[0]))
    control_schema = Draft202012Validator(schemas["Control"])
    control_schema.validate(control)
    del control["candidate"]
    with pytest.raises(ValidationError):
        control_schema.validate(control)
    resolved = tuple(EnvironmentRequirements.model_validate(value) for value in environments)
    assert resolved[0].docker_image == GRADER_BASE_IMAGE and resolved[1].docker_image is None
    assert tuple(machine_backend(value).value for value in resolved) == ("gvisor", "shellsim")
    task_resolved = tuple(EnvironmentRequirements.model_validate(value) for value in task_environments)
    assert all(value.capabilities == ("shell",) for value in task_resolved)
    assert tuple(machine_backend(value).value for value in task_resolved) == ("gvisor", "shellsim")


async def test_validate_restart_joins_adversary_before_replay(
    workpiece: Workpiece, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(log=EventLog())
    bundle = resources(ctx, restart=1)
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    started = asyncio.Event()

    async def blocked(*args: object, **kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    original = bundle.rollouts.engine.run_task

    async def solver(spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        await started.wait()
        return await original(spec, model, run_id=run_id)

    monkeypatch.setattr(harness, "run", blocked)
    monkeypatch.setattr(bundle.rollouts.engine, "run_task", solver)
    setup = adversary_config(harness)
    with pytest.raises(ExceptionGroup) as error:
        await validate(ctx, workpiece, setup, bundle)
    assert error.value.subgroup(Restart) is not None
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    await ctx.drain()
    assert isinstance(bundle.rollouts.engine, SolverEngine) and len(bundle.rollouts.engine.started) == 3


@pytest.mark.parametrize("reply,confirmed", [("9", False), ("12", True)])
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
    setup = adversary_config(harness)
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
    assert isinstance(
        default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Modify if confirmed else Accept
    )
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    assert bundle.envs.requested[0].docker_image == GRADER_BASE_IMAGE


@pytest.mark.parametrize("repair_budget", [0, 1])
async def test_triage_repairs_sections_only_when_budget_allows(workpiece: Workpiece, repair_budget: int) -> None:
    inference = FakeInference((Completion(render(workpiece.proposal)),))
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), role_inference={"author": inference})
    damaged = workpiece.with_(proposal=replace(workpiece.proposal, body="## Task\nToo short"))
    setup = config()
    setup = replace(
        setup, policy=setup.policy.overlaid({"budget.max_triage_repairs": repair_budget, "sampling.temperature": 0.25})
    )
    result = await triage(ctx, damaged, setup, bundle)
    await ctx.drain()
    if repair_budget:
        assert isinstance(result, Triaged) and result.proposal.header == workpiece.proposal.header
        assert inference.requests[0].sampling.temperature == 0.25
    else:
        assert isinstance(result, TriageRejected) and result.reasons
    assert len(inference.requests) == repair_budget
    assert isinstance(await triage(ctx, workpiece, setup, bundle), Triaged)
    assert len(inference.requests) == repair_budget
    await ctx.drain()


# "control" stands for every defect Modify (grader checks, adversary findings): revert keys on placement alone.


# Pilots 20261106i / 20261107j: authored lowering errors held; individual workpiece IDs were not recorded.
@pytest.mark.parametrize("authored", [False, True])
async def test_validate_returns_lowering_failure_without_starting_trials(workpiece: Workpiece, authored: bool) -> None:
    assert workpiece.submission is not None
    task = workpiece.submission.task.model_copy(
        update={"environment_requirements": requirements(image="example.org/task@sha256:" + "0" * 64)}
    )
    if authored:
        task = workpiece.submission.task.model_copy(
            update={
                "resources": workpiece.submission.task.resources.model_copy(
                    update={"worker": (inline_resource("input.txt", b"input"),)}
                )
            }
        )
    workpiece = workpiece.with_(submission=replace(workpiece.submission, task=task))
    ctx = Context(log=EventLog())
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    result = (await validate(ctx, workpiece, config(harness), resources(ctx))).result
    assert result.lowering_error is not None
    assert not result.solver and not result.controls
    assert workpiece.submission is not None
    setup = config(harness)
    if authored:
        assert result.lowering_cause is Cause.INVALID_TASK
        assert setup.submit_gate(workpiece.submission, workpiece.proposal.header.verification) == (
            "Task resources require a task machine",
        )
    else:
        assert result.lowering_cause is Cause.GRADER_INFRA
        with pytest.raises(Hold, match="Lowering failed"):
            setup.submit_gate(workpiece.submission, workpiece.proposal.header.verification)
    assert isinstance(
        default_review(workpiece.with_(evidence=(result,)), config(harness).band, config(harness).budget),
        Modify if authored else Incomplete,
    )


@pytest.mark.parametrize(
    "status,failure,cause",
    [
        (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD, Cause.GRADER_MISSING_REWARD),
        (Outcome.GRADED, None, Cause.GRADER_INVALID_CHECKS),
        (Outcome.INFRA_ERROR, GradingFailure.GRADER_TIMEOUT, Cause.GRADER_TIMEOUT),
        (Outcome.UNAVAILABLE, None, Cause.NO_GRADE),
        (Outcome.INFRA_ERROR, None, Cause.GRADER_INFRA),
    ],
)
async def test_validate_observer_holds_for_grader_outage_but_preserves_authored_defect(
    workpiece: Workpiece, status: Outcome, failure: GradingFailure | None, cause: Cause
) -> None:
    setup = config()
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    grader = GradingResource(
        CandidateEngine(
            GradeResult(
                status,
                1 if status is Outcome.GRADED else None,
                failure=failure,
                detail={"checks": {"total": True}} if cause is Cause.GRADER_INVALID_CHECKS else None,
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
        decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
        assert isinstance(decision, Retry if cause is Cause.NO_GRADE else Modify)
        if cause is Cause.GRADER_TIMEOUT:
            assert isinstance(decision, Modify) and any("grader_timeout" in finding for finding in decision.findings)
    await ctx.drain()


# Pilot 20261106d S/d34.strategy.segmentation/21: base64 resources exhausted the prompt.
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
    setup = adversary_config(harness)
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(task), "finder")
    await ctx.drain()
    assert harness.session is not None
    prompt = harness.session.messages[-1].content
    assert "memory limit: 8192 MiB" in harness.session.messages[0].content
    assert len(prompt) < 10000 and "content_base64" not in prompt
    assert f'"bytes":{len(payload)}' in prompt and '"role":"resources.all"' in prompt
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/input.bin")] == payload
    assert TaskSpec.model_validate_json(bundle.envs.created[0].files[PurePosixPath("/workspace/task.json")]) == task


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
    session = Conversation((Message(Role.ASSISTANT, "", (ToolCall("v1", "validate", "{}"),)),), ())
    resumed = validated(
        workpiece.with_(session=session), Validated(evidence), tool=True, workdir="/workspace", output_chars=3000
    )
    assert resumed.session is not None
    message = resumed.session.messages[-1]
    assert message.tool_call_id == "v1" and len(message.content) <= 3000
    assert "first line" in message.content and "answer.bin" in message.content
    assert "content_base64" not in message.content and "/workspace/.validation/0.json" in message.content
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    await author(ctx, resumed, config(harness), bundle)
    await ctx.drain()
    assert isinstance(bundle.envs, Environments)
    saved = bundle.envs.created[0].files[PurePosixPath("/workspace/.validation/0.json")]
    assert VALIDATION_RESULT.validate_json(saved) == evidence


@pytest.mark.parametrize("claim", [False, True])
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


async def test_custom_review_owns_hardening_policy(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    evidence = ValidationResult(workpiece.submission.digest, (), (), ())
    current = workpiece.with_(too_easy_baseline=workpiece.submission, evidence=(evidence,))
    expected = Modify(("Custom reviewer requires another change",))

    def custom_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Modify:
        assert workpiece is current
        return expected

    setup = replace(config(), review=custom_review)
    ctx = Context(log=EventLog())
    assert (await review(ctx, current, setup, resources(ctx))).decision is expected
    await ctx.drain()


# Pilot o-w6: each output-exhausted agent gets one fresh-machine retry, then failed evidence.
@pytest.mark.parametrize("solver", [False, True])
@pytest.mark.parametrize("failures", [1, 2])
async def test_output_budget_retry(workpiece: Workpiece, solver: bool, failures: int) -> None:
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    success = await CandidateEngine().run_task(rollout_spec(arithmetic_task()), solver_model)
    outcome = success if solver else Returned("report", AdversaryReport(()), Conversation((), ()))
    run = AsyncMock(side_effect=[*[OutputBudgetExhausted("output budget exhausted")] * failures, outcome])
    if solver:
        ctx.handle(SolverRollout, run)
        trial = await solver_result(ctx, rollout_spec(arithmetic_task()), 0, 0)
        assert trial.cause is None
        assert trial.grade == (
            success.grade if failures == 1 else GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted")
        )
    else:
        assert isinstance(outcome, Returned)
        harness = RecordingHarness(outcome)
        harness.run = run
        setup = replace(config(), adversaries={"finder": harness}, tools={"finder": {}})
        result = await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(arithmetic_task()), "finder")
        assert result == AdversaryResult("finder", True, (), ("output budget exhausted",) if failures == 2 else ())
        assert isinstance(bundle.envs, Environments) and len(bundle.envs.created) == 2
        assert all(env.closed for env in bundle.envs.created)
        evidence = ValidationResult(
            "task",
            (TrialResult(0, GradeResult(Outcome.GRADED, 0)),),
            (ControlResult(0, GradeResult(Outcome.GRADED, 1), True),),
            (result,),
        )
        assert not isinstance(
            default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Incomplete
        )
    await ctx.drain()
    assert run.await_count == 2


@pytest.mark.parametrize("primary_failure", [False, True])
async def test_author_cleanup_warns_without_replacing_result(
    workpiece: Workpiece, monkeypatch: pytest.MonkeyPatch, primary_failure: bool
) -> None:
    ctx = Context(log=EventLog())
    notes: list[Note] = []

    async def note(record: Note) -> None:
        notes.append(record)

    ctx.observe(Note, note)
    bundle = resources(ctx)
    assert isinstance(bundle.envs, Environments)
    # Pilots n/o: cleanup after an authored result must not replace it (or its primary failure).
    env = await bundle.envs.create(EnvironmentRequirements())
    monkeypatch.setattr(env, "close", AsyncMock(side_effect=RuntimeError("Machine cleanup failed: ['machine_close']")))
    monkeypatch.setattr(bundle.envs, "create", AsyncMock(return_value=env))
    outcome = Restart("primary failure") if primary_failure else Stopped(StopReason.FINISHED, Conversation((), ()))
    if primary_failure:
        with pytest.raises(Restart, match="primary failure"):
            await author(ctx, workpiece, config(RecordingHarness(outcome)), bundle)
    else:
        assert await author(ctx, workpiece, config(RecordingHarness(outcome)), bundle) == outcome
    await ctx.drain()
    assert len(notes) == 1 and "machine_close" in notes[0].msg


def evidence(solved: int = 1, *, controls_met: bool = True) -> ValidationResult:
    return ValidationResult(
        "task",
        tuple(TrialResult(i, GradeResult(Outcome.GRADED, float(i < solved))) for i in range(3)),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), controls_met),),
        (),
    )


def decide(wp: Workpiece) -> Workpiece:
    wp = wp.with_(session=wp.session or Conversation((Message(Role.SYSTEM, "author"),), ()))
    setup = build_config(load_policy(DEFAULTS_DIR), "S", {})
    return decided(wp, Decided(default_review(wp, setup.band, setup.budget)), budget=setup.budget, policy=setup.policy)


# One hardening attempt uses the shared modification budget, including a prior unrelated repair.
@pytest.mark.parametrize(
    "solved,controls_met,revert", [(2, True, False), (3, True, False), (0, True, True), (3, False, True)]
)
def test_hardening_accepts_or_restores_baseline(
    workpiece: Workpiece, solved: int, controls_met: bool, revert: bool
) -> None:
    original = workpiece.submission
    assert original is not None
    pending = decide(workpiece.with_(round=1, decisions=(Modify(("repair",)),), evidence=(evidence(3),)))
    assert pending.round == 2 and pending.too_easy_baseline == original and pending.terminal is None
    first = pending.decisions[-1]
    assert (
        isinstance(first, Modify)
        and first.placement is BandOutcome.TOO_EASY
        and "Make the task harder" in first.findings[0]
    )
    updated = replace(original, task=original.task.model_copy(update={"labels": {"revision": "harder"}}))
    final = WORKPIECE.validate_json(
        WORKPIECE.dump_json(
            decide(pending.with_(submission=updated, evidence=(evidence(solved, controls_met=controls_met),)))
        )
    )
    assert final.terminal is Terminal.ACCEPTED and final.round == 2
    assert final.submission == (original if revert else updated)
    decision = final.decisions[-1]
    assert isinstance(decision, Accept) and decision.revert is revert
    assert final.accepted_with is (None if solved == 2 else BandOutcome.TOO_EASY)


@pytest.mark.parametrize("exhausted", [False, True])
def test_review_budget_precedence(workpiece: Workpiece, exhausted: bool) -> None:
    current = workpiece.with_(round=2 if exhausted else 0, evidence=(evidence(3),))
    if not exhausted:
        current = current.with_(
            evidence=(replace(evidence(3, controls_met=False), solver=(TrialResult(0, None, Cause.AGENT_TIMEOUT),)),)
        )
    final = decide(current)
    if exhausted:
        assert final.terminal is Terminal.ACCEPTED and final.round == 2 and final.submission == workpiece.submission
        assert final.too_easy_baseline is None and final.accepted_with is BandOutcome.TOO_EASY
        assert (
            isinstance(final.decisions[-1], Accept)
            and "Shared modification budget exhausted" in final.decisions[-1].reasons
        )
    else:
        assert isinstance(final.decisions[-1], Retry) and final.terminal is None


# Pilot 20261106i S/d23.welfare_assessment/03: authored OverflowError was held as infrastructure.
@pytest.mark.parametrize(
    "failure,cause",
    [
        (GradingFailure.GRADER_EXIT, Cause.GRADER_EXECUTION),
        (GradingFailure.GRADER_TIMEOUT, Cause.GRADER_TIMEOUT),
        (GradingFailure.EXECUTION, Cause.GRADER_INFRA),
        (GradingFailure.TIMEOUT, Cause.GRADER_INFRA),
    ],
)
def test_grader_failure_classification_and_repair(workpiece: Workpiece, failure: GradingFailure, cause: Cause) -> None:
    grade = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "grader failed",
        failure=failure,
        diagnostics={
            "exit_code": 1,
            "grader_elapsed_seconds": 1.6779,
            "stderr": "OverflowError: int too large to convert to float",
        },
    )
    assert grade_cause(grade) is cause
    assert replayable_grade(
        Resolved(GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12")), (), grade, False)
    ) is (cause is not Cause.GRADER_INFRA)
    if cause is Cause.GRADER_INFRA:
        return
    finding = AdversaryFinding(
        "Huge integer breaks the grader",
        Candidate('{"target_value":' + "9" * 400 + "}"),
        RewardBand(Outcome.GRADED, 0, 0),
        grade,
        False,
        cause,
    )
    final = decide(
        workpiece.with_(evidence=(replace(evidence(), adversary=(AdversaryResult("adversary", True, (finding,)),)),))
    )
    assert final.round == 1 and final.terminal is None and final.session is not None
    assert all(
        text in final.session.messages[-1].content
        for text in (finding.claim, "grader failed", grade.diagnostics["stderr"])
    )


# Pilot 20261108k T: generation_limit was held; the incident report did not identify its workpiece.
@pytest.mark.parametrize("grade", [None, GradeResult(Outcome.GRADED, 1)])
def test_generation_limit_is_failed_solver_evidence(workpiece: Workpiece, grade: GradeResult | None) -> None:
    result = replace(evidence(), solver=(TrialResult(0, grade, Cause.GENERATION_LIMIT), evidence().solver[0]))
    setup = build_config(load_policy(DEFAULTS_DIR), "S", {})
    band = BandRules(0.5, 0.5, frozenset({BandOutcome.IN_BAND}))
    assert isinstance(default_review(workpiece.with_(evidence=(result,)), band, setup.budget), Accept)
    assert result.solver[0].cause is Cause.GENERATION_LIMIT and result.solver[0].grade is grade
    unavailable = replace(result, controls=(ControlResult(0, None, False, Cause.NO_GRADE),))
    assert isinstance(default_review(workpiece.with_(evidence=(unavailable,)), band, setup.budget), Retry)


@pytest.mark.parametrize("kind,grouped", [(Hold, False), (Hold, True), (Restart, False), (Restart, True)])
async def test_adversary_interruptions_preserve_cause_and_close_environment(
    workpiece: Workpiece,
    kind: type[Hold] | type[Restart],
    grouped: bool,
) -> None:
    failure = kind("unavailable")
    harness = RecordingHarness(
        ExceptionGroup("observer", [failure]) if grouped and isinstance(failure, Exception) else failure
    )
    setup, ctx = adversary_config(harness), Context(log=EventLog())
    bundle = resources(ctx)
    if kind is Restart:
        with pytest.raises((Restart, ExceptionGroup)) as caught:
            await validate(ctx, workpiece, setup, bundle)
        if isinstance(caught.value, ExceptionGroup):
            assert caught.value.subgroup(Restart) is not None
    else:
        result = (await validate(ctx, workpiece, setup, bundle)).result
        restored = WORKPIECE.validate_json(WORKPIECE.dump_json(workpiece.with_(evidence=(result,))))
        reason = "Hold: unavailable"
        assert restored.evidence[0].adversary == (AdversaryResult("finder", False, (), (reason,)),)
        assert len(result.solver) == 3 and len(result.controls) == 2
        decision = default_review(restored, setup.band, setup.budget)
        assert isinstance(decision, Incomplete) and decision.reasons == (f"finder: {reason}",)
    await ctx.drain()
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


# full-s-20261108/S at f929478063: window exhaustion after 20 turns held otherwise reviewable tasks.
@pytest.mark.parametrize("overflow", ["estimate", "provider"])
@pytest.mark.parametrize("verdict", [Accept, Modify, Retry])
async def test_adversary_window_exhaustion_completes_and_explains_review(
    workpiece: Workpiece, overflow: str, verdict: type[Accept] | type[Modify] | type[Retry]
) -> None:
    assert workpiece.submission is not None
    setup = config()
    harness = AgenticHarness(32, sampling(setup.policy), "report findings", 20000)
    setup = replace(setup, adversaries={"finder": harness}, tools={"finder": {}})
    client = FakeInference(
        (
            (
                ContextExceeded("model window exceeded")
                if overflow == "provider"
                else Completion("No report yet", usage=Usage(prompt_tokens=32))
            ),
        )
    )
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    result = await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(arithmetic_task()), "finder")
    assert result == AdversaryResult("finder", True, (), ("adversary exhausted its window without a report",))
    current = replace(evidence(), task_digest=workpiece.submission.digest, adversary=(result,))
    if verdict is Modify:
        current = replace(current, controls=(replace(current.controls[0], met=False),))
    if verdict is Retry:
        current = replace(current, solver=(TrialResult(0, None, Cause.AGENT_TIMEOUT),))
    wp = workpiece.with_(evidence=(current,))
    decision = (await review(ctx, wp, setup, bundle)).decision
    assert isinstance(decision, verdict)
    notes = decision.findings if isinstance(decision, Modify) else decision.reasons
    assert "finder: adversary exhausted its window without a report" in notes
    saved = WORKPIECE.validate_json(WORKPIECE.dump_json(wp.with_(decisions=(decision,))))
    assert saved.decisions[-1] == decision
    assert len(client.requests) == 1
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    await ctx.drain()


def report_call(identity: str, *findings: tuple[str, str, str, float | None, float | None]) -> Completion:
    return Completion(
        "",
        (
            ToolCall(
                identity,
                "report",
                json.dumps(
                    {
                        "findings": [
                            {
                                "defect": defect,
                                "why_the_grade_is_wrong": f"{defect} on {reply}",
                                "candidate": {"reply": reply},
                                "grader_observed_status": status,
                                "grader_observed_reward": observed,
                                "deserved_reward": deserved,
                            }
                            for defect, reply, status, observed, deserved in findings
                        ]
                    }
                ),
            ),
        ),
    )


async def run_scripted_adversary(
    workpiece: Workpiece, script: tuple[Completion, ...], engine: CandidateEngine = CandidateEngine(), revisions: int = 1
) -> tuple[ValidationResult, FakeInference, ForgeConfig]:
    setup = config()
    harness = AgenticHarness(262144, sampling(setup.policy), "report findings", 20000)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"], "adversary.max_revisions": revisions}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
    ctx = Context(log=EventLog())
    client = FakeInference(script)
    bundle = replace(resources(ctx, grader=engine), role_inference={"adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    return result, client, setup


# full-s-20261108 S/d04.structure.periodicity/23: replay 0.0 dropped a false reject reported as expected 0.85-1.0.
async def test_false_reject_with_observed_and_deserved_apart_is_a_finding(workpiece: Workpiece) -> None:
    result, client, setup = await run_scripted_adversary(
        workpiece, (report_call("r1", ("false_reject", "9", "graded", 0.0, 1.0)),)
    )
    (finding,) = result.adversary[0].findings
    assert finding.met and finding.grade is not None and finding.grade.reward == 0.0 and finding.expected is None
    assert len(client.requests) == 1
    decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == ("finder: false_reject on 9",)


async def test_false_accept_reproduces_and_a_correct_grade_does_not(workpiece: Workpiece) -> None:
    result, client, _ = await run_scripted_adversary(
        workpiece,
        (
            report_call("r1", ("false_accept", "12", "graded", 1.0, 0.0), ("false_accept", "9", "graded", 1.0, 0.0)),
            report_call("r2"),
        ),
    )
    accepted, unconfirmed = result.adversary[0].findings
    assert accepted.met and not unconfirmed.met and unconfirmed.grade is not None and unconfirmed.grade.reward == 0
    feedback = client.requests[1].messages[-1]
    assert feedback.role == Role.TOOL and feedback.tool_call_id == "r1"
    assert "1 finding(s) did not reproduce" in feedback.content and "false_accept on 9" in feedback.content
    assert '"reward": 0.0' in feedback.content and "This is your last revision." in feedback.content


# A revision answers the replay in the same session; a restated report of the replayed candidate must match it.
async def test_failed_replay_returns_to_the_session_and_a_revision_recovers_the_finding(workpiece: Workpiece) -> None:
    confused = ("false_accept", "9", "graded", 1.0, 0.0)
    result, client, setup = await run_scripted_adversary(
        workpiece,
        (
            report_call("r1", confused),
            report_call("r2", confused),
            report_call("r3", ("false_reject", "9", "graded", 0.0, 1.0)),
        ),
    )
    first, revised = result.adversary[0].findings
    assert (first.revision, first.met, revised.revision, revised.met) == (0, False, 1, True)
    rejected = client.requests[2].messages[-1]
    assert rejected.tool_call_id == "r2" and "returned graded with reward 0.0" in rejected.content
    assert client.requests[2].conversation[-3].tool_call_id == "r1"
    restored = VALIDATION_RESULT.validate_json(VALIDATION_RESULT.dump_json(result))
    assert restored == result
    decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == ("finder: false_reject on 9",)


@pytest.mark.parametrize("revisions", [0, 1, 2])
async def test_non_reproducing_claims_stay_dropped_after_the_revision_budget(
    workpiece: Workpiece, revisions: int
) -> None:
    script = tuple(report_call(f"r{index}", ("false_accept", f"{9 - index}", "graded", 1.0, 0.0)) for index in range(4))
    result, client, setup = await run_scripted_adversary(workpiece, script, revisions=revisions)
    findings = result.adversary[0].findings
    assert len(client.requests) == revisions + 1
    assert [finding.revision for finding in findings] == list(range(revisions + 1))
    assert not any(finding.met for finding in findings)
    assert isinstance(default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget), Accept)


async def test_grader_crash_finding_reproduces_as_a_modify(workpiece: Workpiece) -> None:
    crash = GradeResult(
        Outcome.INFRA_ERROR, None, "grader exited 1", failure=GradingFailure.GRADER_EXIT, diagnostics={"stderr": "boom"}
    )
    result, client, setup = await run_scripted_adversary(
        workpiece, (report_call("r1", ("grader_crash", "9", "grader_error", None, None)),), CandidateEngine(crash)
    )
    (finding,) = result.adversary[0].findings
    assert finding.met and finding.cause is Cause.GRADER_EXECUTION and len(client.requests) == 1
    decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and "finder: grader_crash on 9" in decision.findings


# full-s-20261108 S/d04.structure.periodicity/23 round 0, as written before findings carried a defect claim.
OLD_FINDING = {
    "claim": "The negation rule in grade.py zeroes a fully correct solution key.",
    "candidate": {"reply": "EA1 of oxygen is exothermic", "files": []},
    "expected": {"status": "graded", "reward_min": 0.85, "reward_max": 1.0},
    "grade": {"status": "graded", "reward": 0.0},
    "met": False,
    "cause": None,
}


def test_old_shape_findings_parse_and_review_as_before(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    old = {**json.loads(VALIDATION_RESULT.dump_json(evidence())), "task_digest": workpiece.submission.digest}
    old["adversary"] = [{"name": "adversary", "completed": True, "findings": [OLD_FINDING], "reasons": []}]
    restored = VALIDATION_RESULT.validate_json(json.dumps(old))
    (finding,) = restored.adversary[0].findings
    assert finding.defect is None and finding.revision == 0 and finding.expected is not None
    assert not finding.met and finding.expected.reward_min == 0.85
    setup = config()
    assert isinstance(default_review(workpiece.with_(evidence=(restored,)), setup.band, setup.budget), Accept)
