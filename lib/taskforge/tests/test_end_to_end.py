# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast, override

import pytest
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference, FakeWeb
from lib.taskforge.tests.stage_fakes import CandidateEngine, SolverEngine, solver_model
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, Rollout, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from taskforge import checkpoint, driver
from taskforge.checkpoint import HEAD, CheckpointSink, Published, read_published
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import advance, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import StageCompleted
from taskforge.effects.issues import Hold, LogFailure, Restart, Superseded, Transient
from taskforge.effects.log import EventLog, JsonlSink, fold
from taskforge.harness.protocol import Harness, Stopped, StopReason, Tool
from taskforge.llm.glm_client import EndpointSource
from taskforge.model.controls import CONTROL, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal
from taskforge.model.review import (
    Abandon,
    Accept,
    BandOutcome,
    BandRules,
    Budget,
    Decision,
    Modify,
    Retry,
)
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Terminal, Workpiece
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.env import CommandResult, EnvFactory
from taskforge.resources.grading import GradingResource
from taskforge.resources.ids import Ids
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.resources.rollouts import Engine, RolloutResource
from taskforge.resources.verifier import RoundVerifier
from taskforge.stages.classify import replayable_grade
from taskforge.stages.exits import Decided
from taskforge.wiring import build_config, install


def published(destination: Path) -> Published:
    journal = read_published(str(destination))
    assert journal is not None
    return journal


def published_workpiece(destination: Path) -> Workpiece:
    workpiece = published(destination).head.workpiece
    assert workpiece is not None
    return workpiece


class RecordingEnv(FakeEnv):
    def __init__(self, commands: Mapping[str, FakeCommand], fail: bool) -> None:
        super().__init__("workspace", commands)
        self.executed: list[str] = []
        self.fail = fail

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.executed.append(cmd)
        if self.fail and len(self.executed) == 2:
            raise Restart("fake machine died")
        return await super().run(cmd, timeout=timeout)


class Envs(EnvFactory):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.created: list[RecordingEnv] = []
        controls = (
            Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE),
            Control(Candidate("9"), RewardBand(Outcome.GRADED, 0, 0), ControlKind.NEGATIVE),
        )
        writes = {
            PurePosixPath("/workspace/task.json"): arithmetic_task().model_dump_json().encode(),
            PurePosixPath("/workspace/controls.json"): (
                json.dumps([CONTROL.dump_python(c, mode="json") for c in controls]).encode()
            ),
        }
        result = CommandResult(0, "", "", 0, False)
        self.commands = {"inspect": FakeCommand(result, {}), "write-draft": FakeCommand(result, writes)}

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = RecordingEnv(self.commands, self.fail)
        self.created.append(env)
        return env


def call(name: str, arguments: Mapping[str, object], identity: str) -> Completion:
    return Completion("", (ToolCall(identity, name, json.dumps(arguments)),), FinishReason.TOOL_CALLS)


def script() -> tuple[Completion, ...]:
    return (
        call("shell", {"command": "inspect", "timeout": 5}, "inspect"),
        call("shell", {"command": "write-draft", "timeout": 5}, "write"),
        call("grade", {"reply": "12", "files": []}, "grade"),
        call(
            "submit",
            {"notes": "done"},
            "submit",
        ),
    )


def resources(envs: Envs, responses: Sequence[Completion]) -> Resources:
    return Resources(
        envs,
        FakeInference(responses),
        RolloutResource(SolverEngine(2), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Ids(lambda: "workpiece"),
        {},
        FakeWeb({}, {}),
    )


SETTINGS: Mapping[str, object] = {"solver.k": 2, "adversary.roster": [], "budget.max_modify_rounds": 1}


def config() -> TaskforgeConfig:
    return build_config(load_policy(DEFAULTS_DIR).overlaid(SETTINGS), {})


async def test_workflow_accepts_and_records_stage_sequence(proposal: TaskProposal) -> None:
    log = EventLog()
    wp = await run_workpiece(proposal, config(), resources(Envs(), script()), log, install=install)
    assert wp.terminal == Terminal.ACCEPTED
    stages = [record.event.stage for record in log.records if isinstance(record.event, StageCompleted)]
    assert stages == ["author", "validate", "review"]
    assert fold(tuple(record.event for record in log.records)).terminal == "accepted"


async def test_crash_resume_replays_commands_without_repeating_inference(proposal: TaskProposal, tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    first = EventLog(sink=sink)
    broken = resources(Envs(fail=True), script())
    assert (await run_workpiece(proposal, config(), broken, first, install=install)).terminal == Terminal.HELD
    assert isinstance(broken.inference, FakeInference)
    assert len(broken.inference.requests) == 2
    resumed = resources(Envs(), script()[2:])
    log = EventLog(replay_from=await sink.read(), sink=sink)
    wp = await run_workpiece(proposal, config(), resumed, log, install=install)
    assert wp.terminal == Terminal.ACCEPTED
    assert isinstance(resumed.inference, FakeInference)
    assert isinstance(resumed.envs, Envs)
    assert len(resumed.inference.requests) == 2
    assert resumed.envs.created[0].executed == ["inspect", "write-draft"]


def needs_revision(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    result = workpiece.evidence[-1]
    return Modify((f"Revise {result.task_digest} for band {band.low}-{band.high}",))


async def test_modify_budget_ends_abandoned(proposal: TaskProposal) -> None:
    class RoundPool(RoundVerifier):
        def __init__(self) -> None:
            super().__init__(RolloutEngine({}, {}), 5)
            self.closes = 0

        @override
        async def close(self) -> None:
            self.closes += 1
            await super().close()

    pool = RoundPool()
    settings = replace(config(), review=needs_revision)
    wp = await run_workpiece(
        proposal,
        settings,
        replace(resources(Envs(), (*script(), *script())), verifiers=pool),
        EventLog(),
        install=install,
    )
    assert pool.closes == 2  # modification ends round zero; terminal abandonment ends round one
    assert wp.terminal == Terminal.ABANDONED
    assert wp.round == 1
    assert len(wp.decisions) == 2
    final = WORKPIECE.validate_json(WORKPIECE.dump_json(wp)).decisions[-1]
    assert isinstance(final, Abandon)
    assert final.reasons == (
        "budget.max_modify_rounds exhausted after 1 modify rounds",
        f"Revise {wp.evidence[-1].task_digest} for band {settings.band.low}-{settings.band.high}",
    )


async def test_validation_retry_budget_holds_without_restarting_author(proposal: TaskProposal) -> None:
    settings = config()

    class UngradedEngine(Engine):
        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            assert run_id is not None
            return GradedRollout(Rollout(run_id, (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))

    bundle = resources(Envs(), script())
    bundle = replace(bundle, rollouts=replace(bundle.rollouts, engine=UngradedEngine()))
    log = EventLog()
    wp = await run_workpiece(proposal, settings, bundle, log, install=install)
    assert wp.terminal == Terminal.HELD
    assert len(wp.decisions) == settings.budget.max_validation_retries + 1
    stages = [r.event.stage for r in log.records if isinstance(r.event, StageCompleted)]
    assert stages == ["author", *[stage for _ in wp.decisions for stage in ("validate", "review")]]


async def test_unavailable_control_grade_is_regraded_on_validation_retry(proposal: TaskProposal) -> None:
    @dataclass(slots=True)
    class OnceUnavailable(Engine):
        calls: int = 0

        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            self.calls += 1
            if self.calls == 1:
                return GradedRollout(Rollout(run_id or "grade", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))
            return await CandidateEngine().run_task(spec, model, run_id=run_id)

    without_grade_tool = tuple(c for c in script() if c.tool_calls[0].name != "grade")
    bundle = replace(resources(Envs(), without_grade_tool), grading=GradingResource(OnceUnavailable(), "grader"))
    wp = await run_workpiece(proposal, config(), bundle, EventLog(replayable=replayable_grade), install=install)
    assert [type(decision) for decision in wp.decisions] == [Retry, Accept]
    assert wp.terminal == Terminal.ACCEPTED


@pytest.fixture
def driver_args(proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    monkeypatch.setenv("GLM_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("GLM_API_KEY", "fake")
    monkeypatch.delenv("GLM_RELAY_JOB", raising=False)
    source, overlay = tmp_path / "test.proposal.json", tmp_path / "policy.json"
    source.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    overlay.write_text(json.dumps({"proposals.environments": ["container"], **SETTINGS}))
    return Namespace(
        proposals=str(source),
        policy=str(overlay),
        output=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
        resume_held=False,
        concurrency=2,
        buckets=4096,
        bucket_range=None,
        ranks=(),
    )


async def test_bucket_split_preserves_manifest_order_and_ownership(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"slot-{index}.proposal.json" for index in range(40)]
    for path in paths:
        path.write_bytes(Path(driver_args.proposals).read_bytes())
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(map(str, paths)))
    driver_args.proposals = str(manifest)
    launched: list[str] = []

    def launch(command: list[str]) -> SimpleNamespace:
        launched.append(command[command.index("--proposals") + 1])
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    partitions = []
    for span in ((0, 4096), (0, 3072), (3072, 3584), (3584, 4096)):
        driver_args.bucket_range = span
        launched.clear()
        await driver.run(driver_args)
        partitions.append(tuple(launched))
        assert launched == [str(path) for path in paths if str(path) in launched]
    assert partitions[0] == tuple(map(str, paths))
    assert sorted(path for part in partitions[1:] for path in part) == sorted(partitions[0])


async def test_driver_persists_programming_failure_then_reraises(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def broken_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), resources(Envs(), ())

    monkeypatch.setattr(driver, "runtime", broken_runtime)
    args = driver_args

    with pytest.raises(RuntimeError, match="exhaust"):
        await driver.one(args)
    destination = tmp_path / "out" / driver.proposal_key(args.proposals)
    result = published_workpiece(destination)
    assert result.terminal == Terminal.HELD
    assert "bug:" in str(result.decisions[-1]) and "Traceback" in str(result.decisions[-1])
    assert published(destination).head.status.terminal is Terminal.HELD
    assert published(destination).events

    # An Iris retry has a fresh local directory, but must not rerun this held workpiece.
    head = (destination / HEAD).read_bytes()
    args.work_dir = str(tmp_path / "retry-work")
    await driver.one(args)
    assert (destination / HEAD).read_bytes() == head


async def test_driver_resumes_published_partial_log_without_repeated_inference(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedInference(FakeInference):
        @override
        async def complete(self, request: InferenceRequest) -> Completion:
            if len(self.requests) == 2:
                raise asyncio.CancelledError()
            return await super().complete(request)

    first = replace(resources(Envs(), ()), inference=InterruptedInference(script()))
    second = resources(Envs(), script()[2:])
    attempts = iter((first, second))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    args = driver_args

    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    final_publish = CheckpointSink.finish

    async def preempted(*_args: object) -> None:
        """Simulate preemption: only periodic checkpoints reach storage, not the final publication."""

    monkeypatch.setattr(CheckpointSink, "finish", preempted)
    with pytest.raises(asyncio.CancelledError):
        await driver.one(args)
    monkeypatch.setattr(CheckpointSink, "finish", final_publish)
    destination = tmp_path / "out" / driver.proposal_key(args.proposals)
    assert published(destination).head.status.terminal is None
    args.work_dir = str(tmp_path / "second")
    await driver.one(args)
    assert published_workpiece(destination).terminal == Terminal.ACCEPTED
    assert isinstance(first.inference, FakeInference) and len(first.inference.requests) == 2
    assert isinstance(second.inference, FakeInference) and len(second.inference.requests) == len(script()) - 2
    head = (destination / HEAD).read_bytes()
    # Even with no resources left to construct, another fresh attempt exits successfully.
    args.work_dir = str(tmp_path / "third")
    await driver.one(args)
    assert (destination / HEAD).read_bytes() == head


@pytest.mark.parametrize("reason", tuple(StopReason))
def test_author_abandonment_publishes_stop_cause(proposal: TaskProposal, reason: StopReason) -> None:
    previous = Modify(("prior review",))
    wp = Workpiece(proposal.header.id, proposal).with_(decisions=(previous,))
    session = Conversation((), ())
    stopped, following = advance(wp, Stage.AUTHOR, Stopped(reason, session), config())
    assert following is None
    published = WORKPIECE.validate_json(WORKPIECE.dump_json(stopped))
    assert published.terminal == Terminal.ABANDONED
    assert published.session == session
    assert published.decisions == (previous, Abandon((f"author stopped: {reason.value} after 0 turns",)))


async def test_stage_restart_replays_on_fresh_machine_without_repeating_inference(proposal: TaskProposal) -> None:
    class Recovers(Envs):
        @override
        async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
            env = await super().create(requirements)
            self.fail = False
            return env

    envs = Recovers(fail=True)
    resource = resources(envs, script())
    result = await run_workpiece(proposal, config(), resource, EventLog(), install=install)
    assert result.terminal == Terminal.ACCEPTED
    assert len(envs.created) == 2
    assert [env.executed for env in envs.created] == [["inspect", "write-draft"], ["inspect", "write-draft"]]
    assert isinstance(resource.inference, FakeInference)
    assert len(resource.inference.requests) == len(script())


async def test_stage_restart_stops_after_three_attempts_with_cause(proposal: TaskProposal) -> None:
    envs = Envs(fail=True)
    resource = resources(envs, script())
    result = await run_workpiece(proposal, config(), resource, EventLog(), install=install)
    assert result.terminal == Terminal.HELD
    assert len(envs.created) == 3
    assert "fake machine died" in str(result.decisions[-1])


@pytest.mark.parametrize("command", ["one", "run"])
async def test_container_filter_skips_before_runtime_or_subprocess(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    args = driver_args
    Path(args.policy).write_text("{}")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Filtered proposal must not start runtime or subprocess")

    monkeypatch.setattr(driver, "runtime", unexpected)
    monkeypatch.setattr(driver.subprocess, "Popen", unexpected)
    await (driver.one(args) if command == "one" else driver.run(args))
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "out").exists()


async def test_driver_only_resumes_held_on_request_and_reexecutes_ungraded_trials(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unavailable(SolverEngine):
        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            return GradedRollout(Rollout(run_id or "trial", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))

    first = resources(Envs(), script())
    first = replace(first, rollouts=replace(first.rollouts, engine=Unavailable(2)))
    second = resources(Envs(), ())
    attempts = iter((first, second))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD
    head = (destination / HEAD).read_bytes()
    driver_args.work_dir += "-skip"
    await driver.one(driver_args)
    assert (destination / HEAD).read_bytes() == head
    driver_args.resume_held = True
    driver_args.work_dir += "-resume"
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert isinstance(second.inference, FakeInference) and not second.inference.requests


@pytest.mark.parametrize("failures", [1, 3])
async def test_environment_startup_transient_replays_stage_or_holds_without_bug(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    class Starting(Envs):
        calls = 0

        @override
        async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
            self.calls += 1
            if self.calls <= failures:
                raise Transient("Environment startup did not resolve")
            return await super().create(requirements)

    envs = Starting()

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), resources(envs, script())

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = published_workpiece(destination)
    assert envs.calls == min(failures + 1, 3)
    assert result.terminal is (Terminal.ACCEPTED if failures == 1 else Terminal.HELD)
    if failures == 3:
        assert "Transient: Environment startup did not resolve" in str(result.decisions[-1])
        assert "bug:" not in str(result.decisions[-1])


@pytest.mark.parametrize("found", ["format 1", "format 3"])
async def test_driver_fails_closed_on_another_journal_format_without_touching_it(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, found: str
) -> None:
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    destination.mkdir(parents=True)
    if found == "format 1":
        (destination / "events.jsonl").write_text('{"sequence": 1}\n')
    else:
        (destination / HEAD).write_text(json.dumps({"format": 3, "writer": "newer"}))
    before = {path: path.read_bytes() for path in destination.rglob("*")}

    @asynccontextmanager
    async def unexpected(*_: object) -> AsyncIterator[None]:
        pytest.fail("A journal in another format must not start the runtime")
        yield

    monkeypatch.setattr(driver, "runtime", unexpected)
    with pytest.raises(LogFailure, match=f"journal {found}"):
        await driver.one(driver_args)
    assert {path: path.read_bytes() for path in destination.rglob("*")} == before


async def test_second_driver_on_a_live_root_supersedes_the_first_which_publishes_nothing(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relaunch racing a not-yet-dead driver takes the head; the old driver stops and cannot publish."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    takeover = asyncio.Event()

    class TakenOver(FakeInference):
        @override
        async def complete(self, request: InferenceRequest) -> Completion:
            if len(self.requests) == 2:
                journal = published(destination)
                await CheckpointSink(
                    JsonlSink(str(Path(driver_args.work_dir) / "relaunch.jsonl")), str(destination)
                ).claim(journal)
                takeover.set()
            return await super().complete(request)

    first = replace(resources(Envs(), ()), inference=TakenOver(script()))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), first

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    with pytest.raises(ExceptionGroup) as raised:
        await driver.one(driver_args)
    assert raised.group_contains(Superseded)
    assert takeover.is_set()
    journal = published(destination)
    assert journal.head.status.terminal is None and journal.head.workpiece is None
    assert isinstance(first.inference, FakeInference) and len(first.inference.requests) < len(script())


class ScriptedAuthor(Harness):
    def __init__(self, *results: Exit | None) -> None:
        self.results = iter(results)
        self.calls = 0

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        self.calls += 1
        return cast(Exit, next(self.results))


@pytest.mark.parametrize(
    "result, named",
    [(Decided(Accept(BandOutcome.IN_BAND)), "author returned Decided"), (None, "author returned NoneType")],
)
async def test_stage_result_without_a_transition_from_that_stage_fails(
    proposal: TaskProposal, result: Exit | None, named: str
) -> None:
    author = ScriptedAuthor(result, Stopped(StopReason.FINISHED, Conversation((), ())))
    settings = replace(config(), author=author)
    with pytest.raises(ValueError, match=named):
        await run_workpiece(proposal, settings, resources(Envs(), ()), EventLog(), install=install)
    assert author.calls == 1


async def test_hold_from_a_terminal_stage_observer_records_that_stage(proposal: TaskProposal) -> None:
    async def hold_when_terminal(event: StageCompleted) -> None:
        if event.terminal is not None:
            raise Hold("late observer")

    def observed(ctx: Context, bundle: Resources) -> None:
        install(ctx, bundle)
        ctx.observe(StageCompleted, hold_when_terminal)

    log = EventLog()
    wp = await run_workpiece(proposal, config(), resources(Envs(), script()), log, install=observed)
    assert wp.terminal is Terminal.HELD and "late observer" in str(wp.decisions[-1])
    completed = [(r.event.stage, r.event.exit) for r in log.records if isinstance(r.event, StageCompleted)]
    assert completed == [("author", "Submitted"), ("validate", "Validated"), ("review", "Decided"), ("review", "Held")]


PREVIOUS_SUBMIT_INSTRUCTION = (
    "Use `submit` with `notes` when ready; it submits the current `task.json` and `controls.json`. If `validate` is "
    "available, it runs the current draft and returns evidence so you can revise before submitting. Use `finish` only "
    "when you cannot complete the task."
)
"""The author system prompt's submit paragraph before the T arm was removed."""


class ResolvedTurnGuard(FakeInference):
    """Fail on any author turn the journal already resolved, matched without the system prompt."""

    def __init__(self, completions: Sequence[Completion], resolved: Sequence[InferenceRequest]) -> None:
        super().__init__(completions)
        self.resolved = {self.turn(request) for request in resolved}

    @staticmethod
    def turn(request: InferenceRequest) -> tuple[Message, ...]:
        return tuple(message for message in request.conversation if message.role != Role.SYSTEM)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if self.turn(request) in self.resolved:
            raise AssertionError("an already-resolved author turn reached inference")
        return await super().complete(request)


@pytest.mark.parametrize("pin_prompt", [True, False])
async def test_root_recorded_with_previous_author_prompt_resumes_through_a_prompt_overlay(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin_prompt: bool
) -> None:
    previous = tmp_path / "previous-defaults"
    shutil.copytree(DEFAULTS_DIR, previous)
    system = previous / "prompts/author/system.md"
    current = system.read_text()
    submit = next(line for line in current.splitlines() if line.startswith("Use `submit`"))
    assert submit != PREVIOUS_SUBMIT_INSTRUCTION
    system.write_text(current.replace(submit, PREVIOUS_SUBMIT_INSTRUCTION))
    first = replace(resources(Envs(), ()), inference=FakeInference((*script()[:2], Hold("relay moved"))))
    built: list[Resources] = [first]

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), built[-1]

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    monkeypatch.setattr(driver, "DEFAULTS_DIR", previous)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD
    assert isinstance(first.inference, FakeInference) and len(first.inference.requests) == 3

    monkeypatch.setattr(driver, "DEFAULTS_DIR", DEFAULTS_DIR)
    if pin_prompt:
        overlay = json.loads(Path(driver_args.policy).read_text())
        Path(driver_args.policy).write_text(json.dumps({**overlay, "author.system": system.read_text()[:-1]}))
    resumed = ResolvedTurnGuard(script()[2:], first.inference.requests[:2])
    built.append(replace(resources(Envs(), ()), inference=resumed))
    driver_args.resume_held = True
    driver_args.work_dir += "-resume"
    if not pin_prompt:
        with pytest.raises(AssertionError, match="already-resolved"):
            await driver.one(driver_args)
        return
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [ResolvedTurnGuard.turn(request)[-1].tool_call_id for request in resumed.requests] == ["write", "grade"]
