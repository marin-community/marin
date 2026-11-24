# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath
from typing import override
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator, ValidationError
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.stage_fakes import CandidateEngine, Environments, RecordingHarness, SolverEngine, solver_model
from rolloutengine.machines import MachineSetupError
from rolloutengine.models import GradedRollout, Model, Rollout, RolloutInterrupted, RolloutOperation, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, rollout_spec
from shellbox.machine import ExitReason, Result
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from taskforge.arms import decided, validated
from taskforge.config import TaskforgeConfig, WorkspaceHydration
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved
from taskforge.effects.issues import ContextExceeded, EnvironmentSetupFailed, Hold, OutputBudgetExhausted, Restart
from taskforge.effects.log import EventLog
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Returned, Stopped, StopReason, Submitted
from taskforge.harness.tools import adversary_tools
from taskforge.harness.tools.adversary import AdversaryClaim, AdversaryReport
from taskforge.lowering import lowering_settings, machine_backend, requirements
from taskforge.model.controls import CONTROL, CONTROLS, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, Message, Role, ToolCall, Usage
from taskforge.model.proposal import (
    REQUIRED_HEADINGS,
    ResearchNeed,
    TaskProposal,
    render,
)
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
    ControlResult,
    Incomplete,
    Modify,
    Retry,
    SearchLimit,
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import WORKPIECE, Conversation, Submission, Terminal, Workpiece
from taskforge.policy import GRADER_BASE_IMAGE, Policy, load_policy, sampling
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.clock import Clock
from taskforge.resources.env import FakeEnv
from taskforge.resources.grading import GradeCandidate, GradingResource, grade
from taskforge.resources.ids import Ids
from taskforge.resources.inference import FakeInference
from taskforge.resources.rollouts import Engine, RolloutResource, SolverRollout
from taskforge.resources.web import FakeWeb
from taskforge.stages.author import author
from taskforge.stages.classify import grade_cause, replayable_grade
from taskforge.stages.exits import Decided, Validated
from taskforge.stages.review import review
from taskforge.stages.rules import default_review
from taskforge.stages.validate import adversary_result, claim_result, control_result, solver_result, validate
from taskforge.wiring import build_config, install


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


def config(harness: RecordingHarness | None = None) -> TaskforgeConfig:
    return replace(
        build_config(load_policy(DEFAULTS_DIR), "S", {}),
        author=harness or RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ()))),
    )


def adversary_config(harness: RecordingHarness) -> TaskforgeConfig:
    setup = config(harness)
    return replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": {}},
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
    assert workpiece.proposal is not None
    assert bundle.envs.created[0].files[PurePosixPath("/workspace/proposal.json")] == render(workpiece.proposal).encode()
    assert bundle.envs.created[0].closed


async def test_author_workspace_holds_caller_files_on_fresh_and_revision_entries(workpiece: Workpiece) -> None:
    def proposal_id(entry: Workpiece, policy: Policy) -> tuple[TaskResource, ...]:
        return (inline_resource("workspace/reference/id.txt", entry.id.encode()),)

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
        assert files[PurePosixPath("/workspace/reference/id.txt")] == workpiece.id.encode()


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
    assert workpiece.proposal is not None
    assert harness.session.messages[-1].content == config(harness).policy.prompt("author.user").render(
        proposal=render(workpiece.proposal)
    )
    assert isinstance(bundle.envs, Environments)
    assert bundle.envs.created[0].closed


SWE_IMAGE = "docker.io/swerebench/sweb.eval.x86_64.astropy_1776_astropy-12907@sha256:" + "b" * 64


@pytest.mark.parametrize(
    "image,admitted,own",
    [(SWE_IMAGE, [SWE_IMAGE], True), (SWE_IMAGE, [], False), (GRADER_BASE_IMAGE, [], True)],
    ids=["source-image", "unadmitted-image", "pinned-image"],
)
async def test_source_task_author_works_in_the_tasks_own_environment_when_admitted(
    workpiece: Workpiece, image: str, admitted: list[str], own: bool
) -> None:
    assert workpiece.submission is not None
    environment = requirements(image=image, workdir="/testbed", setup=("sh setup.sh",))
    notes = inline_resource("workspace/setup.sh", b"true\n")
    task = workpiece.submission.task.model_copy(
        update={
            "environment_requirements": environment,
            "resources": workpiece.submission.task.resources.model_copy(update={"worker": (notes,)}),
        }
    )
    submission = Submission(task, (), "source task")
    source = Workpiece(
        "source", submission=submission, source=submission, decisions=(Modify(("The task has no controls",)),)
    )
    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    setup = replace(setup, policy=setup.policy.overlaid({"sandbox.source_images": admitted}))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    await author(ctx, source, setup, bundle)
    await ctx.drain()
    assert isinstance(bundle.envs, Environments)
    (requested,) = bundle.envs.requested
    (installed,) = bundle.envs.installed
    if own:
        assert requested == environment
        # The task's own setup may read its files, so they are installed before it runs, as for the solver.
        assert notes in installed
    else:
        assert (requested.docker_image, requested.working_directory) == (GRADER_BASE_IMAGE, "/workspace")
        assert installed == ()
    assert harness.session is not None
    system = harness.session.messages[0].content
    # The prompt describes the machine the author got: its image and the directory its shell starts in.
    assert (f"`{requested.docker_image}`" in system, "`/testbed`" in system) == (True, own)


async def test_validate_returns_a_source_task_whose_id_changed_to_the_author(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    original = workpiece.submission
    renamed = replace(original, task=original.task.model_copy(update={"id": "renamed"}))
    source = Workpiece("source", submission=renamed, source=original)
    ctx = Context(log=EventLog())
    result = (await validate(ctx, source, config(), resources(ctx))).result
    await ctx.drain()
    assert result.lowering_cause is Cause.INVALID_TASK and not result.solver
    assert result.lowering_error is not None and "'renamed'" in result.lowering_error
    decision = default_review(source.with_(evidence=(result,)), config().band, config().budget)
    assert isinstance(decision, Modify)


class SetupFails(Engine):
    """A task whose environment's setup command fails on every machine, for rollouts and grades alike."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        error = RolloutInterrupted(Rollout(run_id or "setup", (), "interrupted"), RolloutOperation.START)
        raise error from MachineSetupError(
            "git clone", Result(128, b"", b"fatal: could not read from remote", False, False, ExitReason.EXITED)
        )


async def test_validate_returns_a_failing_environment_setup_to_the_author(workpiece: Workpiece) -> None:
    ctx = Context(log=EventLog())
    bundle = Resources(
        Environments(),
        FakeInference(()),
        RolloutResource(SetupFails(), solver_model, "solver"),
        GradingResource(SetupFails(), "grader"),
        Clock(),
        Ids(),
        {},
        FakeWeb({}, {}),
    )
    install(ctx, bundle)
    result = (await validate(ctx, workpiece, config(), bundle)).result
    await ctx.drain()
    assert {trial.cause for trial in (*result.solver, *result.controls)} == {Cause.INVALID_TASK}
    decision = default_review(workpiece.with_(evidence=(result,)), config().band, config().budget)
    assert isinstance(decision, Modify) and any("git clone" in finding for finding in decision.findings)


async def test_source_author_works_in_author_image_when_the_task_environment_will_not_start(
    workpiece: Workpiece,
) -> None:
    assert workpiece.submission is not None
    image = "ghcr.io/example/swe-task@sha256:" + "a" * 64
    environment = requirements(image=image, workdir="/testbed", setup=("git clone https://example.invalid/repo .",))
    task = workpiece.submission.task.model_copy(update={"environment_requirements": environment})
    submission = Submission(task, (), "source task")
    source = Workpiece("source", submission=submission, source=submission, decisions=(Modify(("no controls",)),))

    class SourceSetupFails(Environments):
        @override
        async def create(
            self, requirements: EnvironmentRequirements, resources: tuple[TaskResource, ...] = ()
        ) -> FakeEnv:
            if requirements.docker_image == image:
                self.requested.append(requirements)
                raise EnvironmentSetupFailed("Environment setup command failed: exited, exit=128")
            return await super().create(requirements)

    harness = RecordingHarness(Stopped(StopReason.FINISHED, Conversation((), ())))
    setup = config(harness)
    setup = replace(setup, policy=setup.policy.overlaid({"sandbox.source_images": [image]}))
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), envs=SourceSetupFails())
    await author(ctx, source, setup, bundle)
    await ctx.drain()
    assert isinstance(bundle.envs, SourceSetupFails)
    assert [requested.docker_image for requested in bundle.envs.requested] == [image, GRADER_BASE_IMAGE]
    assert harness.session is not None
    assert "exit=128" in harness.session.messages[-1].content


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
    claims = tuple(
        AdversaryClaim(f"{reply} earns credit", Candidate(reply), RewardBand(Outcome.GRADED, 1)) for reply in replies
    )
    setup = adversary_config(RecordingHarness(Returned("report", AdversaryReport(claims), Conversation((), ()))))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)

    async def output_limited(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            raise OutputBudgetExhausted("output budget exhausted")
        return await grade(event, grading, notify=ctx.emit)

    ctx.handle(GradeCandidate, output_limited, uses=(bundle.grading,))
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
    claim = {"claim": "13 earns credit", "candidate": {"reply": "13"}, "expected": {"status": "graded", "reward_min": 1}}
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
    harness = AgenticHarness(262144, sampling(setup.policy), "report findings", 20000)
    setup = replace(
        setup,
        policy=setup.policy.overlaid({"adversary.roster": ["finder"]}),
        adversaries={"finder": harness},
        tools={"finder": tools},
    )

    async def output_limited(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            raise OutputBudgetExhausted("output budget exhausted")
        return await grade(event, grading, notify=ctx.emit)

    ctx.handle(GradeCandidate, output_limited, uses=(bundle.grading,))
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
    claims = tuple(
        AdversaryClaim(f"{reply} earns credit", Candidate(reply), RewardBand(Outcome.GRADED, 1)) for reply in ("13", "9")
    )
    setup = adversary_config(RecordingHarness(Returned("report", AdversaryReport(claims), Conversation((), ()))))
    ctx = Context(log=EventLog())
    bundle = resources(ctx)

    async def outage(event: GradeCandidate, grading: GradingResource) -> GradeResult:
        if event.candidate.reply == "13":
            return GradeResult(Outcome.INFRA_ERROR, None, "grader machine lost", failure=GradingFailure.EXECUTION)
        return await grade(event, grading, notify=ctx.emit)

    ctx.handle(GradeCandidate, outage, uses=(bundle.grading,))
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
        (AdversaryResult("finder", AdversaryStatus.REPORTED, (finding,)),),
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
        if failures == 2:
            assert (result.status, result.limit, result.findings) == (
                AdversaryStatus.EXHAUSTED,
                SearchLimit.OUTPUT_BUDGET,
                (),
            )
            assert "ran out" in result.reasons[0]
        else:
            assert result == AdversaryResult("finder", AdversaryStatus.REPORTED, ())
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
        assert restored.evidence[0].adversary == (AdversaryResult("finder", AdversaryStatus.FAILED, (), (reason,)),)
        assert len(result.solver) == 3 and len(result.controls) == 2
        decision = default_review(restored, setup.band, setup.budget)
        assert isinstance(decision, Incomplete) and decision.reasons == (f"finder: {reason}",)
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
    after = decided(wp.with_(session=author), Decided(decision), budget=setup.budget, policy=setup.policy)
    if isinstance(decision, Accept):
        assert f"finder: {note}" in decision.reasons and after.terminal is Terminal.ACCEPTED
    if isinstance(decision, Modify):
        assert after.session is not None and after.session.messages[-1].role == Role.USER
        assert not any(note in text for text in (*decision.findings, after.session.messages[-1].content))
    saved = WORKPIECE.validate_json(WORKPIECE.dump_json(wp.with_(decisions=(decision,))))
    assert saved.decisions[-1] == decision
    assert saved.evidence[-1].adversary[0].status is AdversaryStatus.EXHAUSTED
    assert len(client.requests) == 1
    assert isinstance(bundle.envs, Environments) and bundle.envs.created[0].closed
    await ctx.drain()
