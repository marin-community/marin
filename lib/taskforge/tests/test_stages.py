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
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.stage_fakes import (
    OVER_CREDITED,
    CandidateEngine,
    Environments,
    RecordingHarness,
    SolverEngine,
    over_credited,
    solver_model,
)
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from taskforge.config import TaskforgeConfig, WorkspaceHydration
from taskforge.coordinator import apply_decision
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.issues import ContextExceeded, Hold, OutputBudgetExhausted, Restart
from taskforge.effects.log import EventLog
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Returned, Stopped, StopReason, Submitted
from taskforge.harness.tools import adversary_tools
from taskforge.harness.tools.adversary import AdversaryClaim, AdversaryReport
from taskforge.llm.policy import LLMPolicy, replay_arguments
from taskforge.lowering import lowering_settings, machine_backend, requirements
from taskforge.model.content_hash import digest
from taskforge.model.controls import CONTROL, CONTROLS, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, Message, Role, ToolCall, Usage
from taskforge.model.proposal import ResearchNeed, TaskProposal, render
from taskforge.model.review import (
    VALIDATION_RESULT,
    Accept,
    AdversaryFinding,
    AdversaryResult,
    AdversaryStatus,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    Continuation,
    ControlResult,
    Defect,
    DefectClaim,
    Incomplete,
    Modify,
    Observation,
    ObservedStatus,
    Retry,
    SearchEnd,
    SearchLimit,
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import WORKPIECE, Conversation, Fallback, Submission, Terminal, Workpiece
from taskforge.policy import GRADER_BASE_IMAGE, Policy, load_policy, sampling
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.env import WriteFiles
from taskforge.resources.grading import GradeCandidate, GradingResource, grade
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import author
from taskforge.stages.classify import grade_cause, replayable_grade
from taskforge.stages.review import review
from taskforge.stages.rules import default_review, fallback_after
from taskforge.stages.validate import adversary_result, claim_result, control_result, solver_result, validate
from taskforge.wiring import build_config, submit_gate


@pytest.fixture
def workpiece(proposal: TaskProposal) -> Workpiece:
    proposal = replace(
        proposal,
        header=replace(proposal.header, id="sum", research=(), research_need=ResearchNeed.NOT_NEEDED),
        body="## Task\n" + "Compute the exact sum using a script. " * 7,
    )
    controls = tuple(
        Control(Candidate(reply), RewardBand(Outcome.GRADED, reward_min=lo, reward_max=hi), kind)
        for reply, lo, hi, kind in (("12", 1, None, ControlKind.POSITIVE), ("9", None, 0, ControlKind.NEGATIVE))
    )
    return Workpiece("test", proposal, submission=Submission(arithmetic_task(), controls, "authored"))


def config(harness: RecordingHarness | None = None) -> TaskforgeConfig:
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    author = harness or RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    return replace(setup, author=replace(setup.author, harness=author))


def adversary_config(harness: RecordingHarness) -> TaskforgeConfig:
    """One adversary whose recorded harness returns one fixed report; these tests pin that report's reproduction, so
    it gets no revision (``test_adversaries`` pins the revision)."""
    setup = config(harness)
    return replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["false_accept"], "adversary.max_revisions": 0}),
        adversary=replace(setup.adversary, harness=harness, tools={}),
    )


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


async def test_author_workspace_holds_caller_files_on_fresh_and_revision_entries(workpiece: Workpiece) -> None:
    def proposal_id(entry: Workpiece, policy: Policy) -> tuple[TaskResource, ...]:
        return (inline_resource("workspace/reference/id.txt", entry.proposal.header.id.encode()),)

    hydration = WorkspaceHydration((inline_resource("workspace/reference/static.txt", b"static"),), proposal_id)
    setup = replace(config(), workspace=hydration)
    for entry in (workpiece.with_(submission=None), workpiece):
        ctx = Context(log=EventLog())
        bundle = resources(ctx)
        await author(ctx, entry, setup, bundle)
        await ctx.drain()
        assert isinstance(bundle.envs, Environments)
        files = bundle.envs.created[0].files
        assert files[PurePosixPath("/workspace/reference/static.txt")] == b"static"
        assert files[PurePosixPath("/workspace/reference/id.txt")] == workpiece.proposal.header.id.encode()


def unavailable_reference(_workpiece: Workpiece, _policy: Policy) -> tuple[TaskResource, ...]:
    raise Hold("reference data unavailable")


# Review 2026-11-21: the role runner prepared the caller's files and the opening before it opened the machine and scope,
# so a renderer hold created no machine and journalled nothing, and a prompt failure lost the recorded hydration. On n6
# (d510505b69) both failures journal these rows and close the one machine.
@pytest.mark.parametrize(
    ("failure", "rows"),
    [
        ("renderer", [ScopeEntered, ScopeExited]),
        ("prompt", [ScopeEntered, WriteFiles, Resolved, ScopeExited]),
    ],
)
@pytest.mark.parametrize("revision", [False, True])
async def test_an_author_visit_failing_before_its_first_model_call_journals_and_closes_its_machine(
    workpiece: Workpiece, failure: str, rows: list[type[object]], revision: bool
) -> None:
    policy = load_policy(DEFAULTS_DIR)
    setup = build_config(
        policy.overlaid({"author.user": "{proposal} {undeclared}"}) if failure == "prompt" else policy, {}
    )
    if failure == "renderer":
        setup = replace(setup, workspace=WorkspaceHydration(rendered=unavailable_reference))
    log = EventLog()
    ctx = Context(log=log)
    bundle = resources(ctx)
    with pytest.raises(Hold if failure == "renderer" else KeyError):
        await author(ctx, workpiece if revision else workpiece.with_(submission=None), setup, bundle)
    await ctx.drain()
    assert [type(record.event) for record in log.records] == rows
    assert isinstance(bundle.envs, Environments) and [env.closed for env in bundle.envs.created] == [True]


# Review 2026-11-21: the adversary rendered its opening before entering its visit, so a prompt key error created no
# machine and journalled nothing. On n6 (40d35b8bcc) it journals these rows and closes the one machine.
async def test_an_adversary_visit_failing_before_its_first_model_call_journals_and_closes_its_machine(
    workpiece: Workpiece,
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.user": "{name} {absent}"}), {})
    log = EventLog()
    ctx = Context(log=log)
    bundle = resources(ctx)
    with pytest.raises(KeyError, match="absent"):
        await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(arithmetic_task()), "false_accept", 0)
    await ctx.drain()
    assert [type(record.event) for record in log.records] == [ScopeEntered, WriteFiles, Resolved, ScopeExited]
    assert isinstance(bundle.envs, Environments) and [env.closed for env in bundle.envs.created] == [True]


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
            AdversaryReport((AdversaryClaim("wrong answer accepted", Candidate(reply), OVER_CREDITED),)),
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
    assert evidence.adversary[0].status is AdversaryStatus.REPORTED and finding.candidate.reply == reply
    assert harness.client is adversary_client
    assert finding.grade is not None and finding.grade.reward == float(confirmed)
    assert finding.met is confirmed
    assert isinstance(
        default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Modify if confirmed else Accept
    )
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    assert bundle.envs.requested[0].docker_image == GRADER_BASE_IMAGE


# Review 2026-11-20: a reported claim whose reproduction hit the output limit became an exhausted search, and
# review accepted. One of several parallel reproductions failing must not drop the others.
@pytest.mark.parametrize("replies", [("13",), ("13", "9")])
async def test_inconclusive_claim_reproduction_keeps_the_claim_and_cannot_accept(
    workpiece: Workpiece, replies: tuple[str, ...]
) -> None:
    claims = tuple(AdversaryClaim(f"{reply} earns credit", Candidate(reply), OVER_CREDITED) for reply in replies)
    setup = adversary_config(RecordingHarness(Returned("report", AdversaryReport(claims), Conversation((), ()))))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)

    async def output_limited(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            raise OutputBudgetExhausted("output budget exhausted")
        return await grade(event, grading, notify=ctx.notify)

    ctx.handle(GradeCandidate, output_limited, bundle.grading)
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    (adversary,) = evidence.adversary
    assert adversary.status is AdversaryStatus.FAILED
    assert tuple(finding.candidate.reply for finding in adversary.findings) == replies
    inconclusive = adversary.findings[0]
    assert inconclusive.claim == "13 earns credit" and inconclusive.grade is None and not inconclusive.met
    assert all(finding.grade is not None for finding in adversary.findings[1:])
    restored = WORKPIECE.validate_json(WORKPIECE.dump_json(workpiece.with_(evidence=(evidence,))))
    assert isinstance(default_review(restored, setup.band, setup.budget), Incomplete)


# Review 2026-11-20 (follow-up 1): the adversary's grade tool hit the grading machine's output limit; validation
# recorded an exhausted search with no findings and review accepted.
async def test_grading_machine_exhaustion_in_the_adversary_grade_tool_cannot_accept(workpiece: Workpiece) -> None:
    setup = config()
    claim = over_credited("13 earns credit", "13")
    client = FakeInference(
        (
            Completion("", (ToolCall("1", "grade", json.dumps({"reply": "13"})),)),
            Completion("", (ToolCall("2", "report", json.dumps({"findings": [claim]})),)),
        )
    )
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    tools = adversary_tools(
        task_path="/workspace/task.json",
        policy=setup.policy,
        factories=bundle.factories,
        settings=lowering_settings(setup.policy),
    )
    harness = AgenticHarness(262144, sampling(setup.policy, LLMPolicy()), "report findings", 20000, replay_arguments)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["false_accept"]}),
        adversary=replace(setup.adversary, harness=harness, tools=tools),
    )

    async def output_limited(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            raise OutputBudgetExhausted("output budget exhausted")
        return await grade(event, grading, notify=ctx.notify)

    ctx.handle(GradeCandidate, output_limited, bundle.grading)
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    assert client.requests[1].conversation[-1].content.startswith("Grading this candidate failed")
    (adversary,) = evidence.adversary
    assert adversary.status is AdversaryStatus.FAILED and adversary.limit is None
    (finding,) = adversary.findings
    assert (finding.claim, finding.candidate.reply, finding.grade) == ("13 earns credit", "13", None)
    assert isinstance(default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Incomplete)


# Review 2026-11-20 (follow-up 1): an infrastructure grade on one of two reproductions raised Hold at the adversary
# scope's exit, and the outer handler replaced the report with zero findings.
async def test_observed_grader_outage_on_one_claim_keeps_both_claims_and_the_sibling_grade(
    workpiece: Workpiece,
) -> None:
    claims = tuple(AdversaryClaim(f"{reply} earns credit", Candidate(reply), OVER_CREDITED) for reply in ("13", "9"))
    setup = adversary_config(RecordingHarness(Returned("report", AdversaryReport(claims), Conversation((), ()))))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)

    async def outage(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            return GradeResult(Outcome.INFRA_ERROR, None, "grader machine lost", failure=GradingFailure.EXECUTION)
        return await grade(event, grading, notify=ctx.notify)

    ctx.handle(GradeCandidate, outage, bundle.grading)
    evidence = (await validate(ctx, workpiece, setup, bundle)).result
    await ctx.drain()
    (adversary,) = evidence.adversary
    assert adversary.status is AdversaryStatus.FAILED
    failed, sibling = adversary.findings
    assert (failed.claim, failed.candidate.reply, failed.grade, failed.met) == ("13 earns credit", "13", None, False)
    assert sibling.grade is not None and sibling.grade.reward == 0.0 and not sibling.met
    assert any("grader machine lost" in reason for reason in adversary.reasons)
    assert isinstance(default_review(workpiece.with_(evidence=(evidence,)), setup.band, setup.budget), Incomplete)


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
    gate = partial(submit_gate, settings=lowering_settings(load_policy(DEFAULTS_DIR)), factories={})
    if authored:
        assert result.lowering_cause is Cause.INVALID_TASK
        assert gate(workpiece.submission) == ("Task resources require a task machine",)
    else:
        assert result.lowering_cause is Cause.GRADER_INFRA
        with pytest.raises(Hold, match="Lowering failed"):
            gate(workpiece.submission)
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
    setup = replace(setup, policy=setup.policy.overlaid({"adversary.roster": []}))  # controls alone
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    grader = GradingResource(
        CandidateEngine(
            GradeResult(
                status,
                1 if status is Outcome.GRADED else None,
                failure=failure,
                detail={"checks": {"total": True}} if cause is Cause.GRADER_INVALID_CHECKS else None,
            )
        ),
        "broken",
    )
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.notify), grader)
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
    await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(task), "false_accept", 0)
    await ctx.drain()
    assert harness.session is not None
    prompt = harness.session.messages[-1].content
    assert "memory limit: 8192 MiB" in harness.session.messages[0].content
    assert len(prompt) < 10000 and "content_base64" not in prompt
    assert f'"bytes":{len(payload)}' in prompt and '"role":"resources.all"' in prompt
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/input.bin")] == payload
    assert TaskSpec.model_validate_json(bundle.envs.created[0].files[PurePosixPath("/workspace/task.json")]) == task


async def test_author_hydrates_full_validation_evidence(workpiece: Workpiece) -> None:
    """Evidence reaches the author's workspace untruncated: an oversized diagnostic and binary candidate files."""
    assert workpiece.submission is not None
    grade = GradeResult(Outcome.GRADED, 1, diagnostics={"stdout": "first line\n" + "x" * 100000})
    finding = AdversaryFinding(
        "wrong answer accepted",
        Candidate("x", (inline_resource("answer.bin", b"\x00\xffz"),)),
        OVER_CREDITED,
        grade,
        True,
    )
    evidence = ValidationResult(
        workpiece.submission.digest,
        (TrialResult(0, grade),),
        (ControlResult(0, grade, True),),
        (AdversaryResult("false_accept", AdversaryStatus.REPORTED, (finding,)),),
    )
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    await author(ctx, workpiece.with_(evidence=(evidence,)), config(harness), bundle)
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
            await claim_result(ctx, spec, AdversaryClaim("wrong", control.candidate, OVER_CREDITED))
        else:
            await control_result(ctx, spec, 0, control)
    await ctx.drain()


async def test_custom_review_owns_hardening_policy(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    evidence = ValidationResult(workpiece.submission.digest, (), (), ())
    current = workpiece.with_(fallback=Fallback(workpiece.submission, BandOutcome.TOO_EASY), evidence=(evidence,))
    expected = Modify(("Custom reviewer requires another change",))

    def custom_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Modify:
        assert workpiece is current
        return expected

    setup = replace(config(), review=custom_review)
    ctx = Context(log=EventLog())
    assert (await review(ctx, current, setup, resources(ctx))).decision is expected
    await ctx.drain()


OWNER = "validate/0/0/adversary:false_accept/0"
"""The first roster entry's sandbox owner in round 0, less its search retry."""


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
        setup = config()
        setup = replace(setup, adversary=replace(setup.adversary, harness=harness, tools={}))
        result = await adversary_result(
            ctx, workpiece, setup, bundle, rollout_spec(arithmetic_task()), "false_accept", 0
        )
        if failures == 2:
            assert (result.status, result.limit, result.findings) == (
                AdversaryStatus.EXHAUSTED,
                SearchLimit.OUTPUT_BUDGET,
                (),
            )
            assert "ran out" in result.reasons[0]
            assert result.continuation == Continuation(f"{OWNER}/1", SearchEnd.BUDGET_EXHAUSTED)
        else:
            assert (result.status, result.findings) == (AdversaryStatus.REPORTED, ())
            assert result.reasons == (
                "ran out of search budget without a finding (its session ended with a report and no reproduced "
                "finding); this is not a validation of the task",
            )
            assert result.continuation == Continuation(f"{OWNER}/1", SearchEnd.REPORTED, digest(Conversation((), ())), 0)
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


def reviewed(wp: Workpiece) -> Workpiece:
    wp = wp.with_(session=wp.session or Conversation((Message(Role.SYSTEM, "author"),), ()))
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    return apply_decision(wp, default_review(wp, setup.band, setup.budget), setup.budget, setup.policy)


# One hardening attempt uses the shared modification budget, including a prior unrelated repair.
@pytest.mark.parametrize(
    "solved,controls_met,revert", [(2, True, False), (3, True, False), (0, True, True), (3, False, True)]
)
def test_hardening_accepts_or_restores_baseline(
    workpiece: Workpiece, solved: int, controls_met: bool, revert: bool
) -> None:
    original = workpiece.submission
    assert original is not None
    pending = reviewed(workpiece.with_(round=1, decisions=(Modify(("repair",)),), evidence=(evidence(3),)))
    assert pending.round == 2 and pending.fallback == Fallback(original, BandOutcome.TOO_EASY)
    assert pending.terminal is None
    first = pending.decisions[-1]
    assert (
        isinstance(first, Modify)
        and first.placement is BandOutcome.TOO_EASY
        and "Make the task harder" in first.findings[0]
    )
    updated = replace(original, task=original.task.model_copy(update={"labels": {"revision": "harder"}}))
    final = WORKPIECE.validate_json(
        WORKPIECE.dump_json(
            reviewed(pending.with_(submission=updated, evidence=(evidence(solved, controls_met=controls_met),)))
        )
    )
    assert final.terminal is Terminal.ACCEPTED and final.round == 2
    assert final.submission == (original if revert else updated)
    decision = final.decisions[-1]
    assert isinstance(decision, Accept) and decision.restore == (original.digest if revert else None)
    assert final.accepted_with is (None if solved == 2 else BandOutcome.TOO_EASY)


def revised(submission: Submission, label: str) -> Submission:
    return replace(submission, task=submission.task.model_copy(update={"labels": {"revision": label}}))


# Owner's ruling: an out-of-band revision restores the best earlier in-band submission, bound to its digest.
@pytest.mark.parametrize("bad", ["too_hard", "fault"])
def test_a_bad_revision_restores_the_right_earlier_submission_by_its_digest(workpiece: Workpiece, bad: str) -> None:
    hard = workpiece.submission
    assert hard is not None
    first = reviewed(workpiece.with_(evidence=(evidence(0),)))  # too hard: revised, but no fallback
    assert first.round == 1 and first.fallback is None
    easy = revised(hard, "easier")
    second = reviewed(first.with_(submission=easy, evidence=(*first.evidence, evidence(3))))  # too easy: the fallback
    assert second.round == 2 and second.fallback == Fallback(easy, BandOutcome.TOO_EASY)
    assert isinstance(step := second.decisions[-1], Modify) and step.fallback == easy.digest
    regressed = evidence(0) if bad == "too_hard" else evidence(3, controls_met=False)
    final = reviewed(second.with_(submission=revised(easy, "harder"), evidence=(*second.evidence, regressed)))
    decision = final.decisions[-1]
    assert isinstance(decision, Accept) and decision.restore == easy.digest
    assert final.terminal is Terminal.ACCEPTED and final.submission == easy != hard
    assert final.accepted_with is BandOutcome.TOO_EASY and "accepted_with: too_easy" in decision.reasons
    assert f"The revision left the acceptance band; restored submission {easy.digest[:12]}" in decision.reasons
    assert WORKPIECE.validate_json(WORKPIECE.dump_json(final)) == final


def test_the_fallback_is_the_best_acceptable_placement_then_the_latest(workpiece: Workpiece) -> None:
    stored = workpiece.submission
    assert stored is not None
    current = revised(stored, "current")
    both = frozenset({BandOutcome.IN_BAND, BandOutcome.TOO_EASY})
    for kept, placement, accepted, expected in (
        (BandOutcome.IN_BAND, BandOutcome.TOO_EASY, both, stored),
        (BandOutcome.TOO_EASY, BandOutcome.IN_BAND, both, current),
        (BandOutcome.TOO_EASY, BandOutcome.TOO_EASY, both, current),
        (BandOutcome.TOO_EASY, BandOutcome.TOO_HARD, both, stored),
        (BandOutcome.TOO_EASY, None, both, stored),
        (BandOutcome.IN_BAND, BandOutcome.TOO_EASY, frozenset({BandOutcome.IN_BAND}), stored),
    ):
        wp = workpiece.with_(submission=current, fallback=Fallback(stored, kept))
        assert fallback_after(wp, placement, BandRules(0.25, 0.75, accepted)) == expected.digest
    assert fallback_after(workpiece, BandOutcome.TOO_HARD, BandRules(0.25, 0.75, both)) is None


def test_a_decision_naming_another_fallback_digest_is_refused(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    wp = workpiece.with_(session=Conversation((Message(Role.SYSTEM, "author"),), ()))
    with pytest.raises(ValueError, match="Cannot restore"):
        apply_decision(wp, Accept(BandOutcome.TOO_EASY, (), "elsewhere"), setup.budget, setup.policy)
    with pytest.raises(ValueError, match="Cannot keep"):
        apply_decision(wp, Modify(("harder",), BandOutcome.TOO_EASY, "elsewhere"), setup.budget, setup.policy)


@pytest.mark.parametrize("cause", [Cause.HELD, Cause.AGENT_TIMEOUT, Cause.NO_GRADE])
def test_a_solver_slot_without_a_score_holds_by_policy_and_is_never_read_as_a_zero(
    workpiece: Workpiece, cause: Cause
) -> None:
    zero = GradeResult(Outcome.GRADED, 0.0) if cause is Cause.AGENT_TIMEOUT else None  # what a timed-out run graded
    absent = TrialResult(2, zero, cause, "Hold: request rejected" if cause is Cause.HELD else "")
    result = replace(evidence(3), solver=(*evidence(3).solver[:2], absent))
    wp = workpiece.with_(evidence=(result,))
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    assert setup.budget.max_missing_trials == 0  # the default: any missing slot holds the round
    held = default_review(wp, setup.band, setup.budget)
    assert isinstance(held, Retry if cause is not Cause.HELD else Incomplete)
    if cause is Cause.HELD:
        assert "solver trial 2: Hold: request rejected" in held.reasons
    tolerant = build_config(load_policy(DEFAULTS_DIR).overlaid({"budget.max_missing_trials": 1}), {})
    placed = default_review(wp, tolerant.band, tolerant.budget)
    # Placed over the two scored slots, both solved: too easy. Read as a zero, 2 of 3 would be in band.
    assert isinstance(placed, Modify) and placed.placement is BandOutcome.TOO_EASY
    assert placed.findings[0].startswith("2 of 2 trials solved: too_easy")
    assert "1 of 3 solver trials have no score; placed over 2" in placed.findings


# Review 2026-11-21: one solved, one held and one timed-out slot under the zero-missing default chose Retry twice,
# buying every slot three times before the hold.
def test_holds_beyond_the_missing_allowance_take_precedence_over_a_review_retry(workpiece: Workpiece) -> None:
    solver = (
        TrialResult(0, GradeResult(Outcome.GRADED, 1.0)),
        TrialResult(1, None, Cause.HELD, "RequestRejected: refused"),
        TrialResult(2, None, Cause.AGENT_TIMEOUT),
    )
    wp = workpiece.with_(evidence=(replace(evidence(3), solver=solver),))
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    held = default_review(wp, setup.band, setup.budget)
    assert isinstance(held, Incomplete) and "solver trial 1: RequestRejected: refused" in held.reasons
    tolerant = build_config(load_policy(DEFAULTS_DIR).overlaid({"budget.max_missing_trials": 1}), {})
    assert isinstance(default_review(wp, tolerant.band, tolerant.budget), Retry)  # the hold alone is within it


@pytest.mark.parametrize("exhausted", [False, True])
def test_review_budget_precedence(workpiece: Workpiece, exhausted: bool) -> None:
    current = workpiece.with_(round=2 if exhausted else 0, evidence=(evidence(3),))
    if not exhausted:
        current = current.with_(
            evidence=(replace(evidence(3, controls_met=False), solver=(TrialResult(0, None, Cause.AGENT_TIMEOUT),)),)
        )
    final = reviewed(current)
    if exhausted:
        assert final.terminal is Terminal.ACCEPTED and final.round == 2 and final.submission == workpiece.submission
        assert final.fallback is None and final.accepted_with is BandOutcome.TOO_EASY
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
        DefectClaim(Defect.GRADER_CRASH, Observation(ObservedStatus.GRADER_ERROR), None),
        grade,
        True,  # a claimed crash that the replay showed
        cause,
    )
    final = reviewed(
        workpiece.with_(
            evidence=(
                replace(evidence(), adversary=(AdversaryResult("adversary", AdversaryStatus.REPORTED, (finding,)),)),
            )
        )
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
    setup = build_config(load_policy(DEFAULTS_DIR), {})
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
        assert restored.evidence[0].adversary == (
            AdversaryResult(
                "false_accept",
                AdversaryStatus.FAILED,
                (),
                (reason,),
                continuation=Continuation(f"{OWNER}/0", SearchEnd.HELD),
            ),
        )
        assert len(result.solver) == 3 and len(result.controls) == 2
        decision = default_review(restored, setup.band, setup.budget)
        assert isinstance(decision, Incomplete) and decision.reasons == (f"false_accept: {reason}",)
    await ctx.drain()
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed


# full-s-20261108/S at f929478063: window exhaustion after 20 turns held otherwise reviewable tasks.
# An exhausted search may be accepted; its note says the budget ran out, on an Accept and never a Modify.
@pytest.mark.parametrize("overflow", ["estimate", "provider"])
@pytest.mark.parametrize("verdict", [Accept, Modify, Retry])
async def test_adversary_window_exhaustion_passes_and_notes_only_an_accept(
    workpiece: Workpiece, overflow: str, verdict: type[Accept] | type[Modify] | type[Retry]
) -> None:
    assert workpiece.submission is not None
    setup = config()
    harness = AgenticHarness(32, sampling(setup.policy, LLMPolicy()), "report findings", 20000, replay_arguments)
    setup = replace(setup, adversary=replace(setup.adversary, harness=harness, tools={}))
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
    result = await adversary_result(ctx, workpiece, setup, bundle, rollout_spec(arithmetic_task()), "false_accept", 0)
    assert (result.status, result.limit, result.findings) == (AdversaryStatus.EXHAUSTED, SearchLimit.CONTEXT_WINDOW, ())
    (note,) = result.reasons
    assert "ran out" in note and "context window" in note
    current = replace(evidence(), task_digest=workpiece.submission.digest, adversary=(result,))
    if verdict is Modify:
        current = replace(current, controls=(replace(current.controls[0], met=False),))
    if verdict is Retry:
        current = replace(current, solver=(TrialResult(0, None, Cause.AGENT_TIMEOUT),))
    wp = workpiece.with_(evidence=(current,))
    decision = (await review(ctx, wp, setup, bundle)).decision
    assert isinstance(decision, verdict)
    author = Conversation((Message(Role.SYSTEM, "author"),), ())
    after = apply_decision(wp.with_(session=author), decision, setup.budget, setup.policy)
    if isinstance(decision, Accept):
        assert f"false_accept: {note}" in decision.reasons and after.terminal is Terminal.ACCEPTED
    if isinstance(decision, Modify):
        assert after.session is not None and after.session.messages[-1].role == Role.USER
        assert not any(note in text for text in (*decision.findings, after.session.messages[-1].content))
    saved = WORKPIECE.validate_json(WORKPIECE.dump_json(wp.with_(decisions=(decision,))))
    assert saved.decisions[-1] == decision
    assert saved.evidence[-1].adversary[0].status is AdversaryStatus.EXHAUSTED
    assert len(client.requests) == 1
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    await ctx.drain()
