# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import override

import pytest
from lib.taskforge.tests.resources.iris_fakes import SITES, Fault, Injection, LocalController, local_iris_machine
from lib.taskforge.tests.stage_fakes import CandidateEngine, SolverEngine, solver_model
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, Rollout, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from shellbox.machine import Command, ExitReason, GuestHelperFailed, Machine, Result
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from taskforge import checkpoint, driver
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import STAGE_REPLAY_ATTEMPTS, run_workpiece
from taskforge.effects.events import StageCompleted
from taskforge.effects.issues import Restart, Transient
from taskforge.effects.log import EventLog, JsonlSink, fold
from taskforge.harness.protocol import Stopped, StopReason
from taskforge.llm.glm_client import EndpointSource
from taskforge.model.controls import CONTROL, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, ToolCall
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal
from taskforge.model.review import Abandon, Accept, BandRules, Budget, Cause, Decision, Incomplete, Modify, Retry
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Terminal, Workpiece
from taskforge.policy import Policy, load_policy
from taskforge.resources.bundle import Resources
from taskforge.resources.clock import Clock
from taskforge.resources.env import CommandResult, EnvFactory, FakeCommand, FakeEnv, ShellboxEnv
from taskforge.resources.grading import GradingResource
from taskforge.resources.ids import Ids
from taskforge.resources.inference import FakeInference, InferenceRequest
from taskforge.resources.rollouts import Engine, RolloutResource
from taskforge.resources.verifier import RoundVerifier
from taskforge.resources.web import FakeWeb
from taskforge.stages.classify import replayable_grade
from taskforge.stages.rules import NO_CONTROLS, default_review
from taskforge.wiring import build_config, install


class RecordingEnv(FakeEnv):
    def __init__(self, commands: Mapping[str, FakeCommand], fail: bool) -> None:
        super().__init__("workspace", commands, lambda _: None)
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


def script(arm: str) -> tuple[Completion, ...]:
    validate = (call("validate", {}, "validate"),) if arm == "T" else ()
    return (
        call("shell", {"command": "inspect", "timeout": 5}, "inspect"),
        call("shell", {"command": "write-draft", "timeout": 5}, "write"),
        call("grade", {"reply": "12", "files": []}, "grade"),
        *validate,
        call(
            "submit",
            {"notes": "done"},
            "submit",
        ),
    )


def resources(envs: EnvFactory, responses: Sequence[Completion]) -> Resources:
    return Resources(
        envs,
        FakeInference(responses),
        RolloutResource(SolverEngine(2), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Clock(lambda: 1),
        Ids(lambda: "workpiece"),
        {},
        FakeWeb({}, {}),
    )


def config(arm: str) -> TaskforgeConfig:
    root = Path(__file__).parents[1] / "src/taskforge/policy/defaults"
    policy = load_policy(root).overlaid({"solver.k": 2, "adversary.roster": [], "budget.max_modify_rounds": 1})
    return build_config(policy, arm, {})


@pytest.mark.parametrize("arm", ("S", "T"))
async def test_both_arms_accept_and_record_stage_sequence(proposal: TaskProposal, arm: str) -> None:
    log = EventLog()
    wp = await run_workpiece(
        Workpiece(proposal.header.id, proposal), config(arm), resources(Envs(), script(arm)), log, install=install
    )
    assert wp.terminal == Terminal.ACCEPTED
    stages = [record.event.stage for record in log.records if isinstance(record.event, StageCompleted)]
    assert stages == (
        ["author", "validate", "review"] if arm == "S" else ["author", "validate", "author", "validate", "review"]
    )
    assert fold(tuple(record.event for record in log.records)).terminal == "accepted"


async def test_crash_resume_replays_commands_without_repeating_inference(proposal: TaskProposal, tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    first = EventLog(sink=sink)
    broken = resources(Envs(fail=True), script("S"))
    assert (
        await run_workpiece(Workpiece(proposal.header.id, proposal), config("S"), broken, first, install=install)
    ).terminal == Terminal.HELD
    assert isinstance(broken.inference, FakeInference)
    assert len(broken.inference.requests) == 2
    resumed = resources(Envs(), script("S")[2:])
    log = EventLog(replay_from=await sink.read(), sink=sink)
    wp = await run_workpiece(Workpiece(proposal.header.id, proposal), config("S"), resumed, log, install=install)
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
    settings = replace(config("S"), review=needs_revision)
    wp = await run_workpiece(
        Workpiece(proposal.header.id, proposal),
        settings,
        replace(resources(Envs(), (*script("S"), *script("S"))), verifiers=pool),
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
    settings = config("S")

    class UngradedEngine(Engine):
        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            assert run_id is not None
            return GradedRollout(Rollout(run_id, (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))

    bundle = resources(Envs(), script("S"))
    bundle = replace(bundle, rollouts=replace(bundle.rollouts, engine=UngradedEngine()))
    log = EventLog()
    wp = await run_workpiece(Workpiece(proposal.header.id, proposal), settings, bundle, log, install=install)
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

    without_grade_tool = tuple(c for c in script("S") if c.tool_calls[0].name != "grade")
    bundle = replace(resources(Envs(), without_grade_tool), grading=GradingResource(OnceUnavailable(), "grader"))
    wp = await run_workpiece(
        Workpiece(proposal.header.id, proposal),
        config("S"),
        bundle,
        EventLog(replayable=replayable_grade),
        install=install,
    )
    assert [type(decision) for decision in wp.decisions] == [Retry, Accept]
    assert wp.terminal == Terminal.ACCEPTED


class FlakySolver(Engine):
    """Trial 0 raises ``Restart`` the given number of times for each validation attempt index."""

    def __init__(self, failures: dict[int, int]) -> None:
        self.failures = failures

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        trial, attempt = (int(part) for part in run_id.split(":")[-2:])
        if trial == 0 and self.failures.get(attempt, 0):
            self.failures[attempt] -= 1
            raise Restart("solver sandbox preempted")
        return GradedRollout(Rollout(run_id, (), "stop"), GradeResult(Outcome.GRADED, float(trial % 2 == 0)))


# Review 2026-11-20: a same-round validation retry inherited the earlier visit's replay attempts and held.
async def test_stage_replay_budget_restarts_after_each_completed_stage(proposal: TaskProposal) -> None:
    def retry_once(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
        if any(isinstance(decision, Retry) for decision in workpiece.decisions):
            return default_review(workpiece, band, budget)
        return Retry(Cause.AGENT_TIMEOUT, 1)

    solver = FlakySolver({0: STAGE_REPLAY_ATTEMPTS - 1, 1: 1})
    bundle = resources(Envs(), script("S"))
    bundle = replace(bundle, rollouts=replace(bundle.rollouts, engine=solver))
    wp = await run_workpiece(
        Workpiece(proposal.header.id, proposal),
        replace(config("S"), review=retry_once),
        bundle,
        EventLog(),
        install=install,
    )
    assert [type(decision) for decision in wp.decisions] == [Retry, Accept]
    assert wp.terminal == Terminal.ACCEPTED and not any(solver.failures.values())


@pytest.fixture
def driver_args(proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    monkeypatch.setenv("GLM_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("GLM_API_KEY", "fake")
    monkeypatch.delenv("GLM_RELAY_JOB", raising=False)
    source, overlay = tmp_path / "test.proposal.json", tmp_path / "policy.json"
    source.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    overlay.write_text('{"proposals.environments": ["container"]}')
    return Namespace(
        proposals=str(source),
        policy=str(overlay),
        arm="S",
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
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, arm, {}), resources(Envs(), ())

    monkeypatch.setattr(driver, "runtime", broken_runtime)
    args = driver_args

    with pytest.raises(RuntimeError, match="exhaust"):
        await driver.one(args)
    destination = tmp_path / "out" / driver.proposal_key(args.proposals)
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert result.terminal == Terminal.HELD
    assert "bug:" in str(result.decisions[-1]) and "Traceback" in str(result.decisions[-1])
    assert json.loads((destination / "status.json").read_text())["terminal"] == "held"
    assert await JsonlSink(str(destination / "events.jsonl")).read()

    # An Iris retry has a fresh local directory, but must not rerun this held workpiece.
    args.work_dir = str(tmp_path / "retry-work")
    await driver.one(args)
    assert (Path(args.work_dir) / driver.proposal_key(args.proposals) / "events.jsonl").read_bytes() == (
        destination / "events.jsonl"
    ).read_bytes()


async def test_driver_resumes_published_partial_log_without_repeated_inference(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedInference(FakeInference):
        @override
        async def complete(self, request: InferenceRequest) -> Completion:
            if len(self.requests) == 2:
                raise asyncio.CancelledError()
            return await super().complete(request)

    first = replace(resources(Envs(), ()), inference=InterruptedInference(script("S")))
    second = resources(Envs(), script("S")[2:])
    attempts = iter((first, second))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    args = driver_args

    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    final_publish = driver.publish
    # Simulate preemption: only periodic snapshots reach storage, not the finally upload.
    monkeypatch.setattr(driver, "publish", lambda *_args: None)
    with pytest.raises(asyncio.CancelledError):
        await driver.one(args)
    monkeypatch.setattr(driver, "publish", final_publish)
    destination = tmp_path / "out" / driver.proposal_key(args.proposals)
    assert json.loads((destination / "status.json").read_text())["terminal"] is None
    args.work_dir = str(tmp_path / "second")
    await driver.one(args)
    assert WORKPIECE.validate_json((destination / "workpiece.json").read_bytes()).terminal == Terminal.ACCEPTED
    assert isinstance(first.inference, FakeInference) and len(first.inference.requests) == 2
    assert isinstance(second.inference, FakeInference) and len(second.inference.requests) == len(script("S")) - 2
    published = (destination / "events.jsonl").read_bytes()
    # Even with no resources left to construct, another fresh attempt exits successfully.
    args.work_dir = str(tmp_path / "third")
    await driver.one(args)
    assert (destination / "events.jsonl").read_bytes() == published


@pytest.mark.parametrize("reason", tuple(StopReason))
def test_author_abandonment_publishes_stop_cause(proposal: TaskProposal, reason: StopReason) -> None:
    settings = config("S")
    edge = settings.transitions[(Stage.AUTHOR, Stopped)]
    assert edge[0] is None
    previous = Modify(("prior review",))
    wp = Workpiece(proposal.header.id, proposal).with_(decisions=(previous,))
    session = Conversation((), ())
    stopped = edge[1](wp, Stopped(reason, session))
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
    resource = resources(envs, script("S"))
    result = await run_workpiece(
        Workpiece(proposal.header.id, proposal), config("S"), resource, EventLog(), install=install
    )
    assert result.terminal == Terminal.ACCEPTED
    assert len(envs.created) == 2
    assert [env.executed for env in envs.created] == [["inspect", "write-draft"], ["inspect", "write-draft"]]
    assert isinstance(resource.inference, FakeInference)
    assert len(resource.inference.requests) == len(script("S"))


async def test_stage_restart_stops_after_three_attempts_with_cause(proposal: TaskProposal) -> None:
    envs = Envs(fail=True)
    resource = resources(envs, script("S"))
    result = await run_workpiece(
        Workpiece(proposal.header.id, proposal), config("S"), resource, EventLog(), install=install
    )
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

    first = resources(Envs(), script("S"))
    first = replace(first, rollouts=replace(first.rollouts, engine=Unavailable(2)))
    second = resources(Envs(), ())
    attempts = iter((first, second))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert WORKPIECE.validate_json((destination / "workpiece.json").read_bytes()).terminal is Terminal.HELD
    published = (destination / "events.jsonl").read_bytes()
    driver_args.work_dir += "-skip"
    await driver.one(driver_args)
    assert (destination / "events.jsonl").read_bytes() == published
    driver_args.resume_held = True
    driver_args.work_dir += "-resume"
    await driver.one(driver_args)
    assert WORKPIECE.validate_json((destination / "workpiece.json").read_bytes()).terminal is Terminal.ACCEPTED
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
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), resources(envs, script("S"))

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert envs.calls == min(failures + 1, 3)
    assert result.terminal is (Terminal.ACCEPTED if failures == 1 else Terminal.HELD)
    if failures == 3:
        assert "Transient: Environment startup did not resolve" in str(result.decisions[-1])
        assert "bug:" not in str(result.decisions[-1])


async def test_recover_resumes_a_stream_published_without_status(tmp_path: Path) -> None:
    """An interrupted publish leaves the stream without status.json; that is a resume, never a fresh start."""
    destination, workdir = tmp_path / "published", tmp_path / "work"
    workdir.mkdir()
    assert await driver.recover(workdir, str(destination)) is None
    destination.mkdir()
    sink = JsonlSink(str(destination / "events.jsonl"))
    await EventLog(sink=sink).record(StageCompleted(Stage.AUTHOR, "", "Held", 0, Terminal.HELD))
    restored = await driver.recover(workdir, str(destination))
    assert restored is not None and restored.terminal is Terminal.HELD and not restored.finished
    assert (workdir / "events.jsonl").read_bytes() == Path(sink.path).read_bytes()
    assert not (workdir / "status.json").exists()


async def test_driver_holds_an_undecodable_stream_without_touching_it(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / "events.jsonl.parts").mkdir(parents=True)
    first = JsonlSink(str(destination / "events.jsonl"))
    await EventLog(sink=first).record(StageCompleted(Stage.AUTHOR, "", "Held", 0, Terminal.HELD))
    second = JsonlSink(str(Path(driver_args.work_dir + "-other") / "events.jsonl"))
    log = EventLog(sink=second)
    await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
    await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
    (destination / "events.jsonl.parts" / "000000000002.jsonl").write_bytes(Path(second.path).read_bytes())
    (destination / "status.json").write_text(json.dumps({"terminal": None}))
    published = (destination / "events.jsonl").read_bytes()

    @asynccontextmanager
    async def unexpected(*_: object) -> AsyncIterator[None]:
        pytest.fail("An undecodable stream must not start the runtime")
        yield

    monkeypatch.setattr(driver, "runtime", unexpected)
    await driver.one(driver_args)
    workpiece = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert workpiece.terminal is Terminal.HELD
    assert isinstance(decision := workpiece.decisions[-1], Incomplete) and "diverged" in decision.reasons[0]
    assert json.loads((destination / "status.json").read_text())["terminal"] == "held"
    assert (destination / "events.jsonl").read_bytes() == published


class HelperlessMachine(Machine):
    """Runs every command; file downloads fail because the image lacks a transfer helper."""

    @override
    async def run(self, command: Command) -> Result:
        return Result(0, b"", b"", False, False, ExitReason.EXITED)

    @override
    async def upload(self, source: Path, target: str) -> None:
        pass

    @override
    async def download(self, source: str, target: Path) -> None:
        raise GuestHelperFailed("Iris command failed (127): wc: not found")

    @override
    async def close(self) -> None:
        pass


class HelperlessEnvs(EnvFactory):
    @override
    async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
        return ShellboxEnv(HelperlessMachine(), "image", lambda _: None, AsyncExitStack())


# Review 2026-11-20 (follow-up 1): a missing guest `wc` was stored as a bug hold and failed the child process.
async def test_broken_guest_helper_holds_with_its_cause_without_a_bug(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), resources(HelperlessEnvs(), script("S"))

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert result.terminal is Terminal.HELD
    assert "wc: not found" in str(result.decisions[-1]) and "bug:" not in str(result.decisions[-1])


class SandboxController(LocalController):
    """A local controller whose guest ``/workspace`` is ``root``."""

    def __init__(self, root: Path, faults: Sequence[Injection], prelude: str) -> None:
        super().__init__(faults, prelude)
        self.root = root

    @override
    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        command = [part.replace("/workspace", str(self.root)) for part in request.command]  # type: ignore[attr-defined]
        return super().exec_in_container(SimpleNamespace(command=command), timeout_ms)


class LocalIrisEnvs(EnvFactory):
    """Each environment is a fresh local Iris sandbox in its own directory with the same scripted faults."""

    def __init__(self, root: Path, faults: Sequence[Injection], prelude: str = "") -> None:
        self.root, self.faults, self.prelude = root, faults, prelude
        self.created: list[Path] = []

    @override
    async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
        workdir = self.root / f"sandbox-{len(self.created)}"
        workdir.mkdir(parents=True)
        self.created.append(workdir)
        machine = local_iris_machine(workdir, SandboxController(workdir, self.faults, self.prelude))
        return ShellboxEnv(machine, "image", lambda _: None, AsyncExitStack())


# Review 2026-11-20 (follow-up 3): a lost WebSocket upgrade reply reran the command in the same sandbox, and a
# missing `wc` or an OOM-killed sandbox during capture held only after three sandboxes with one append each.
@pytest.mark.parametrize(
    "faults,prelude,sandboxes,cause",
    [
        ((("command", Fault.SETUP_FAILED, True),), "", STAGE_REPLAY_ATTEMPTS, "command outcome unknown"),
        ((), "wc() { echo 'wc: not found' >&2; return 127; }; ", 1, "wc: not found"),
        ((("stdout-size", Fault.OOM_KILLED, True),), "", 1, "OOMKilled"),
    ],
    ids=["lost-upgrade-reply", "capture-without-wc", "capture-oom-killed"],
)
async def test_iris_command_runs_once_per_sandbox_and_holds_with_its_cause(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faults: tuple[Injection, ...],
    prelude: str,
    sandboxes: int,
    cause: str,
) -> None:
    envs = LocalIrisEnvs(tmp_path / "sandboxes", faults, prelude)
    counted = (call("shell", {"command": "echo ran >> counter", "timeout": 5}, "inspect"), *script("S")[1:])

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), resources(envs, counted)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert result.terminal is Terminal.HELD
    assert cause in str(result.decisions[-1]) and "bug:" not in str(result.decisions[-1])
    assert [(workdir / "counter").read_text() for workdir in envs.created] == ["ran\n"] * sandboxes


# Review 2026-11-21 (follow-up 4): a lost upgrade reply for an upload chunk retried the write in the same sandbox; the
# original append landed after the retry truncated the file, and the next read returned `proofproof` in an ACCEPTED run.
async def test_lost_upload_reply_never_reads_back_a_doubled_write(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FirstSandboxFaulted(LocalIrisEnvs):
        @override
        async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
            env = await super().create(requirements)
            self.faults = ()
            return env

    # The stage hydrates proposal.json first; fault the append of proof.txt itself.
    monkeypatch.setitem(SITES, "proof-append", lambda script: "| base64 -d >>" in script and "proof.txt" in script)
    envs = FirstSandboxFaulted(tmp_path / "sandboxes", (("proof-append", Fault.SETUP_FAILED_LATE, True),))
    responses = (
        call("write_file", {"path": "proof.txt", "content": "proof"}, "write-proof"),
        call("read_file", {"path": "proof.txt"}, "read-proof"),
        *(
            call("write_file", {"path": str(path), "content": data.decode()}, path.name)
            for path, data in Envs().commands["write-draft"].writes.items()
        ),
        *script("S")[2:],
    )

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), resources(envs, responses)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert result.terminal is Terminal.ACCEPTED and result.session is not None
    assert [message.content for message in result.session.messages if message.tool_call_id == "read-proof"] == ["proof"]
    assert len(envs.created) == 2 and (envs.created[1] / "proof.txt").read_bytes() == b"proof"


async def test_driver_repairs_source_task_without_proposal(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "arithmetic.task.json"
    source.write_text(arithmetic_task().model_dump_json())
    driver_args.proposals = str(source)
    bundle = resources(Envs(), script("S"))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, arm: str, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield config(arm), bundle

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)

    destination = tmp_path / "out" / driver.proposal_key(str(source))
    wp = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert wp.terminal == Terminal.ACCEPTED
    assert wp.proposal is None
    first, second = wp.decisions
    assert isinstance(first, Modify) and first.findings == (NO_CONTROLS,)
    assert isinstance(second, Accept)
    assert wp.submission is not None and wp.submission.controls
    records = await JsonlSink(str(destination / "events.jsonl")).read()
    stages = [record.event.stage for record in records if isinstance(record.event, StageCompleted)]
    assert stages == ["validate", "review", "author", "validate", "review"]
    assert isinstance(bundle.inference, FakeInference) and isinstance(bundle.envs, Envs)
    opening = bundle.inference.requests[0].messages[1].content
    assert opening == config("S").policy.prompt("author.repair").render(kind="review", detail=NO_CONTROLS)
    assert not any(
        path.name == "proposal.json" for env in bundle.envs.created for path in env.files
    ), "a source task has no proposal to hydrate"
