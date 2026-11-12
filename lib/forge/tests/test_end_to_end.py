# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import override

import pytest
from lib.forge.tests.stage_fakes import CandidateEngine, SolverEngine, solver_model
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, Rollout, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements
from taskforge.llm.glm_client import EndpointSource

from forge import checkpoint, driver
from forge.config import ForgeConfig
from forge.coordinator import run_workpiece
from forge.effects.events import StageCompleted
from forge.effects.issues import Restart, Transient
from forge.effects.log import EventLog, JsonlSink, fold
from forge.harness.protocol import Stopped, StopReason
from forge.model.controls import CONTROL, Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, FinishReason, ToolCall
from forge.model.proposal import TASK_PROPOSAL, TaskProposal
from forge.model.review import Abandon, BandRules, Budget, Decision, Incomplete, Modify
from forge.model.stage import Stage
from forge.model.workpiece import WORKPIECE, Conversation, Terminal, Workpiece
from forge.policy import Policy, load_policy
from forge.resources.bundle import Resources
from forge.resources.clock import Clock
from forge.resources.env import CommandResult, EnvFactory, FakeCommand, FakeEnv
from forge.resources.grading import GradingResource
from forge.resources.ids import Ids
from forge.resources.inference import FakeInference, InferenceRequest
from forge.resources.rollouts import Engine, RolloutResource
from forge.resources.verifier import RoundVerifier
from forge.resources.web import FakeWeb
from forge.wiring import build_config, install

pytest_plugins = ("lib.forge.tests.test_proposal",)


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


def resources(envs: Envs, responses: Sequence[Completion]) -> Resources:
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


def config(arm: str, reviewer_rounds: int = 0) -> ForgeConfig:
    root = Path(__file__).parents[1] / "src/forge/policy/defaults"
    policy = load_policy(root).overlaid(
        {"solver.k": 2, "adversary.roster": [], "budget.max_modify_rounds": 1, "reviewer.max_rounds": reviewer_rounds}
    )
    return build_config(policy, arm, {})


@pytest.mark.parametrize("arm", ("S", "T"))
async def test_both_arms_accept_and_record_stage_sequence(proposal: TaskProposal, arm: str) -> None:
    log = EventLog()
    wp = await run_workpiece(proposal, config(arm), resources(Envs(), script(arm)), log, install=install)
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
    assert (await run_workpiece(proposal, config("S"), broken, first, install=install)).terminal == Terminal.HELD
    assert isinstance(broken.inference, FakeInference)
    assert len(broken.inference.requests) == 2
    resumed = resources(Envs(), script("S")[2:])
    log = EventLog(replay_from=await sink.read(), sink=sink)
    wp = await run_workpiece(proposal, config("S"), resumed, log, install=install)
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
        proposal,
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
    wp = await run_workpiece(proposal, settings, bundle, log, install=install)
    assert wp.terminal == Terminal.HELD
    assert len(wp.decisions) == settings.budget.max_validation_retries + 1
    stages = [r.event.stage for r in log.records if isinstance(r.event, StageCompleted)]
    assert stages == ["author", *[stage for _ in wp.decisions for stage in ("validate", "review")]]


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
    ) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
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
    ) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
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
    result = await run_workpiece(proposal, config("S"), resource, EventLog(), install=install)
    assert result.terminal == Terminal.ACCEPTED
    assert len(envs.created) == 2
    assert [env.executed for env in envs.created] == [["inspect", "write-draft"], ["inspect", "write-draft"]]
    assert isinstance(resource.inference, FakeInference)
    assert len(resource.inference.requests) == len(script("S"))


async def test_stage_restart_stops_after_three_attempts_with_cause(
    proposal: TaskProposal, caplog: pytest.LogCaptureFixture
) -> None:
    envs = Envs(fail=True)
    resource = resources(envs, script("S"))
    result = await run_workpiece(proposal, config("S"), resource, EventLog(), install=install)
    assert result.terminal == Terminal.HELD
    assert len(envs.created) == 3
    assert "fake machine died" in str(result.decisions[-1])
    for attempt in (1, 2, 3):
        assert f"attempt {attempt} of 3" in caplog.text


@pytest.mark.parametrize("command", ["one", "run"])
async def test_container_filter_skips_before_runtime_or_subprocess(
    proposal: TaskProposal,
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    args = driver_args
    Path(args.policy).write_text("{}")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Filtered proposal must not start runtime or subprocess")

    monkeypatch.setattr(driver, "runtime", unexpected)
    monkeypatch.setattr(driver.subprocess, "Popen", unexpected)
    with caplog.at_level(logging.INFO):
        await (driver.one(args) if command == "one" else driver.run(args))
    output = caplog.text + capsys.readouterr().out
    assert f"SKIP {proposal.header.id}: environment=container" in output
    if command == "run":
        assert "SUMMARY submitted=0 completed=0 failed=0 skipped=1" in output
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
    ) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
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
    ) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
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


# Mark 2026-11-11: a rollout reviewer's suggestions send an accepted task back to the author for one round.
async def test_reviewer_suggestions_get_one_author_round_then_accept(proposal: TaskProposal) -> None:
    excerpt = "unsure whether to round"
    review = (
        call("list_trials", {}, "list"),
        call("grep_trials", {"pattern": "round"}, "grep"),
        call(
            "suggest",
            {
                "revisions": [
                    {
                        "kind": "ambiguous_prompt",
                        "evidence": [{"trial": 1, "excerpt": excerpt}],
                        "suggested_edit": "State that the answer is an exact integer.",
                    }
                ]
            },
            "suggest",
        ),
    )
    responses = (*script("S"), *review, *script("S"))
    log = EventLog()
    bundle = resources(Envs(), responses)
    wp = await run_workpiece(proposal, config("S", reviewer_rounds=1), bundle, log, install=install)
    assert wp.terminal == Terminal.ACCEPTED and wp.round == 1 and wp.accepted_with is None
    assert [type(decision).__name__ for decision in wp.decisions] == ["Revise", "Accept"]
    first, second = wp.evidence
    assert first.reviewer is not None and first.reviewer.suggestions[0].evidence[0].excerpt == excerpt
    assert second.reviewer is None  # the reviewer budget is spent; the revision is accepted on its own evidence
    assert isinstance(bundle.inference, FakeInference)
    tool_replies = [m.content for m in bundle.inference.requests[len(script("S")) + 2].conversation if m.role == "tool"]
    assert "trial 1:" in tool_replies[0] and "Trial 1: unsure whether to round" in tool_replies[1]
    revision_prompt = bundle.inference.requests[len(script("S")) + len(review)].messages[-1].content
    assert "State that the answer is an exact integer." in revision_prompt and "may decline" in revision_prompt
    assert wp.fallback is not None and wp.fallback.submission.digest == first.task_digest
