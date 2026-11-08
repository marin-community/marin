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
from forge.harness.protocol import Returned, Stopped, StopReason, Submitted
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport
from forge.lowering import machine_backend, requirements
from forge.model.controls import CONTROL, CONTROLS, Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, Role, ToolCall
from forge.model.proposal import (
    REQUIRED_HEADINGS,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
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
from forge.policy import GRADER_BASE_IMAGE, Policy, load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.bundle import Resources
from forge.resources.clock import Clock
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.ids import Ids
from forge.resources.inference import FakeInference
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


@pytest.fixture
def workpiece() -> Workpiece:
    header = ProposalHeader(
        "sum",
        SourceRef(SourceKind.CAPABILITY, "arithmetic", "v1"),
        Environment.REASONING,
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


def config(harness: RecordingHarness | None = None) -> ForgeConfig:
    return replace(
        build_config(load_policy(DEFAULTS_DIR), "S", {}),
        author=harness or RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ()))),
        workspace=WorkspaceHydration((inline_resource("workspace/notes", b"static"),), rendered_files),
    )


def resources(ctx: Context, restart: int | None = None) -> Resources:
    bundle = Resources(
        Environments(),
        FakeInference(()),
        RolloutResource(SolverEngine(3, restart), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
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
    setup = config(harness)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
    )
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
    assert isinstance(
        default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Modify if confirmed else Accept
    )
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    assert bundle.envs.requested[0].docker_image == GRADER_BASE_IMAGE


async def test_review_does_not_repair_from_incomplete_evidence(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, None, Cause.AGENT_TIMEOUT),),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), False),),
        (),
    )
    setup = config()
    assert isinstance(default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Retry)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "kind,message",
    [
        (Hold, "environment terminated: cpu_exhausted"),
        (Restart, "model unavailable"),
    ],
)
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
    decision = default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget)
    assert isinstance(decision, Incomplete)
    assert decision.reasons == (f"finder: {kind.__name__}: {message}",)
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


async def test_invalid_grader_checks_request_task_repair(workpiece: Workpiece) -> None:
    setup = config()
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    invalid = GradeResult(Outcome.GRADED, 1, detail={"checks": {"total": True}})
    grader = GradingResource(CandidateEngine(invalid), "invalid-checks")
    ctx.resource(grader, grader.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grader,))
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    assert all(control.cause is Cause.GRADER_INVALID_CHECKS for control in evidence.controls)
    decision = default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget)
    assert isinstance(decision, Modify)
    assert any("grader_invalid_checks" in finding for finding in decision.findings)


async def test_adversary_stop_reason_survives_review_and_serialization(workpiece: Workpiece) -> None:
    reason = StopReason.CONTEXT_LIMIT
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
    restored = WORKPIECE.validate_json(WORKPIECE.dump_json(workpiece.with_(evidence=(evidence,))))
    assert restored.evidence[0].adversary[0].reasons == (f"{reason} after 0 turns without report",)
    decision = default_review(restored, setup.band, setup.budget)
    assert isinstance(decision, Incomplete)
    assert f"finder: {reason} after 0 turns without report" in decision.reasons


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
@pytest.mark.parametrize("revision", ["in_band", "too_easy", "control", "too_hard"])
async def test_one_hardening_attempt_accepts_or_reverts(workpiece: Workpiece, revision: str) -> None:
    assert workpiece.submission is not None
    original = workpiece.submission
    evidence = ValidationResult(
        original.digest,
        tuple(TrialResult(index, GradeResult(Outcome.GRADED, 1)) for index in range(3)),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), True),),
        (),
    )
    setup = config()
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    # An earlier unrelated modification already consumed one round of the shared budget.
    current = workpiece.with_(
        round=1,
        session=Conversation((Message(Role.SYSTEM, "author"),), ()),
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
    controls = (replace(evidence.controls[0], met=False),) if revision == "control" else evidence.controls
    revised = replace(
        evidence,
        task_digest=updated.digest,
        solver=tuple(TrialResult(index, GradeResult(Outcome.GRADED, float(index < solved))) for index in range(3)),
        controls=controls,
    )
    current = pending.with_(submission=updated, evidence=(*pending.evidence, revised))
    second = await review(ctx, current, setup, bundle)
    final = decided(current, second, budget=setup.budget, policy=setup.policy)
    restored = WORKPIECE.validate_json(WORKPIECE.dump_json(final))
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
    setup = config()
    assert workpiece.submission is not None
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, GradeResult(Outcome.GRADED, 1)),),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1), True),),
        (),
    )
    current = workpiece.with_(round=setup.budget.max_modify_rounds, evidence=(evidence,))
    decision = Decided(default_review(current, setup.band, setup.budget))
    final = decided(current, decision, budget=setup.budget, policy=setup.policy)
    assert final.terminal is Terminal.ACCEPTED and final.round == current.round
    assert final.submission == workpiece.submission and final.too_easy_baseline is None
    assert final.accepted_with is BandOutcome.TOO_EASY and isinstance(final.decisions[-1], Accept)
    assert "Shared modification budget exhausted" in final.decisions[-1].reasons


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
        assert setup.submit_gate(workpiece.submission) == ("Task resources require a task machine",)
    else:
        assert result.lowering_cause is Cause.GRADER_INFRA
        with pytest.raises(Hold, match="Lowering failed"):
            setup.submit_gate(workpiece.submission)
    assert isinstance(
        default_review(workpiece.with_(evidence=(result,)), config(harness).band, config(harness).budget),
        Modify if authored else Incomplete,
    )


@pytest.mark.parametrize(
    "status,failure,cause",
    [
        (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD, Cause.GRADER_MISSING_REWARD),
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
        decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
        assert isinstance(decision, Retry if cause is Cause.NO_GRADE else Modify)
        if cause is Cause.GRADER_TIMEOUT:
            assert isinstance(decision, Modify) and any("grader_timeout" in finding for finding in decision.findings)
    await ctx.drain()


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


@pytest.mark.parametrize(
    "failure,diagnostics,cause",
    [
        (GradingFailure.GRADER_EXIT, {"exit_code": 1}, Cause.GRADER_EXECUTION),
        (GradingFailure.EXECUTION, {"exit_code": 1, "grader_elapsed_seconds": 1.6779}, Cause.GRADER_INFRA),
        (GradingFailure.EXECUTION, {"exit_code": 1}, Cause.GRADER_INFRA),
        (GradingFailure.EXECUTION, {"exit_code": None, "grader_elapsed_seconds": 1}, Cause.GRADER_INFRA),
        (GradingFailure.EXECUTION, {"exit_code": 137, "grader_elapsed_seconds": 1}, Cause.GRADER_INFRA),
        (
            GradingFailure.EXECUTION,
            {"exit_code": 1, "grader_elapsed_seconds": 1, "exit_reason": "exited"},
            Cause.GRADER_INFRA,
        ),
        (GradingFailure.TIMEOUT, {"exit_code": 1, "grader_elapsed_seconds": 1}, Cause.GRADER_INFRA),
        (GradingFailure.GRADER_TIMEOUT, {"exit_code": None}, Cause.GRADER_TIMEOUT),
    ],
)
def test_grader_process_failure_is_distinct_from_infrastructure(
    failure: GradingFailure, diagnostics: dict[str, object], cause: Cause
) -> None:
    grade = GradeResult(Outcome.INFRA_ERROR, None, "OverflowError", diagnostics=diagnostics, failure=failure)
    assert grade_cause(grade) is cause
    event = GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))
    assert replayable_grade(Resolved(event, (), grade, False)) is (
        cause in (Cause.GRADER_EXECUTION, Cause.GRADER_TIMEOUT)
    )


@pytest.mark.parametrize("cause", [Cause.GRADER_EXECUTION, Cause.GRADER_TIMEOUT])
def test_adversary_grader_failure_requests_repair_with_evidence(workpiece: Workpiece, cause: Cause) -> None:
    assert workpiece.submission is not None
    traceback = (
        'Traceback (most recent call last):\n  File "/tests/grader.py", line 105, in is_number\n'
        "OverflowError: int too large to convert to float"
    )
    crash = cause is Cause.GRADER_EXECUTION
    error = "Grader command failed: exited, exit=1" if crash else "Grader command timed out"
    crashed = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        error,
        diagnostics={"stderr": traceback if crash else "", "exit_code": 1 if crash else 124},
        failure=GradingFailure.GRADER_EXIT if crash else GradingFailure.GRADER_TIMEOUT,
    )
    finding = AdversaryFinding(
        "Huge integer breaks the grader",
        Candidate('{"target_value":' + "9" * 400 + "}"),
        RewardBand(Outcome.GRADED, 0, 0),
        crashed,
        False,
        cause,
    )
    good = GradeResult(Outcome.GRADED, 1)
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, good),),
        (ControlResult(0, good, True),),
        (AdversaryResult("adversary", True, (finding,)),),
    )
    current = workpiece.with_(evidence=(evidence,), session=Conversation((Message(Role.SYSTEM, "author"),), ()))
    setup = config()
    decision = default_review(current, setup.band, setup.budget)
    assert isinstance(decision, Modify)
    revised = decided(current, Decided(decision), budget=setup.budget, policy=setup.policy)
    assert revised.session is not None and revised.round == 1 and revised.terminal is None
    assert finding.claim in revised.session.messages[-1].content
    assert error in revised.session.messages[-1].content
    if crash:
        assert traceback in revised.session.messages[-1].content


@pytest.mark.parametrize("grade", [None, GradeResult(Outcome.GRADED, 1)])
def test_generation_limited_solver_is_failed_pass_rate_evidence(workpiece: Workpiece, grade: GradeResult | None) -> None:
    assert workpiece.submission is not None
    good = GradeResult(Outcome.GRADED, 1)
    result = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, grade, Cause.GENERATION_LIMIT), TrialResult(1, good)),
        (ControlResult(0, good, True),),
        (),
    )
    setup = config()
    decision = default_review(
        workpiece.with_(evidence=(result,)), BandRules(0.5, 0.5, frozenset({BandOutcome.IN_BAND})), setup.budget
    )
    assert isinstance(decision, Accept)
    assert result.solver[0].cause is Cause.GENERATION_LIMIT and result.solver[0].grade is grade
    unavailable = replace(result, controls=(ControlResult(0, None, False, Cause.NO_GRADE),))
    assert not isinstance(default_review(workpiece.with_(evidence=(unavailable,)), setup.band, setup.budget), Accept)


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
