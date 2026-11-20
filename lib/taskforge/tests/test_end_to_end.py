# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import hashlib
import json
import shutil
import sys
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast, override

import pytest
from botocore.exceptions import ClientError, IncompleteReadError
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference, FakeWeb
from lib.taskforge.tests.stage_fakes import CandidateEngine, SolverEngine, solver_model
from rigging.filesystem.conditional_object import LocalConditionalObject, VersionedBytes
from rigging.timing import ExponentialBackoff
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, Rollout, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from taskforge import checkpoint, driver
from taskforge.checkpoint import HEAD, HEAD_JSON, CheckpointSink, Head, Journal, read_journal
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import advance, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, StageCompleted
from taskforge.effects.issues import Hold, LogFailure, PublicationFailed, Restart, Superseded, Transient
from taskforge.effects.log import EventLog, JsonlSink, Record, Status, decode, fold
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


def published(destination: Path) -> Journal:
    journal = read_journal(str(destination))
    assert journal is not None
    return journal


def history(destination: Path) -> tuple[Record, ...]:
    return decode(published(destination).events())


def published_workpiece(destination: Path) -> Workpiece:
    workpiece = published(destination).workpiece()
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
    log = EventLog(replay_from=local_records(sink), sink=sink)
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
    assert published(destination).events()

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


def tree(destination: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}


@asynccontextmanager
async def unexpected_runtime(*_: object) -> AsyncIterator[None]:
    pytest.fail("A journal that cannot be resumed must not start the runtime")
    yield


@pytest.mark.parametrize(
    "artefact, found",
    [
        ("events.jsonl", "format 1"),
        ("events.jsonl.parts/000000000001.jsonl", "format 1"),
        ("status.json", "format 1"),
        ("workpiece.json", "format 1"),
        (HEAD, "format 3"),
    ],
)
async def test_driver_fails_closed_on_another_journal_format_without_touching_it(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, artefact: str, found: str
) -> None:
    """Format 1 wrote parts and sidecars before its consolidated stream; any one of them marks the root."""
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / artefact).parent.mkdir(parents=True)
    (destination / artefact).write_text(json.dumps({"format": 3, "sequence": 1}) + "\n")
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure, match=f"journal {found}"):
        await driver.one(driver_args)
    assert tree(destination) == before


async def test_drivers_sharing_a_work_directory_publish_only_their_own_history(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh root starts from its own empty history, whatever another run left in the shared work directory."""
    built = [resources(Envs(), script()), resources(Envs(), script())]
    attempts = iter(built)

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    identity = driver.proposal_key(driver_args.proposals)
    await driver.one(driver_args)
    first = history(Path(driver_args.output) / identity)
    driver_args.output += "-second"
    await driver.one(driver_args)
    second = history(Path(driver_args.output) / identity)
    assert published_workpiece(Path(driver_args.output) / identity).terminal is Terminal.ACCEPTED
    assert [record.event for record in second] == [record.event for record in first]
    assert [record.sequence for record in second] == list(range(1, len(first) + 1))
    assert isinstance(built[1].inference, FakeInference) and len(built[1].inference.requests) == len(script())


def corrupt(line: bytes, corruption: str, sequence: int) -> bytes:
    row = json.loads(line)
    if corruption == "torn tail":
        return line.rstrip(b"\n")
    if corruption == "unknown kind":
        row["event"]["type"] = "test.unknown"
    else:
        row["sequence"] = sequence + 1
    return json.dumps(row, sort_keys=True).encode() + b"\n"


@pytest.mark.parametrize("corruption", ["unknown kind", "bad sequence", "torn tail"])
async def test_resume_decodes_published_history_before_claiming_and_leaves_it_untouched(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    class Unavailable(SolverEngine):
        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            return GradedRollout(Rollout(run_id or "trial", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))

    held = resources(Envs(), script())
    held = replace(held, rollouts=replace(held.rollouts, engine=Unavailable(2)))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), held

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    head = json.loads((destination / HEAD).read_bytes())
    assert head["status"]["terminal"] == Terminal.HELD and head["workpiece"] is not None
    lines = b"".join((destination / "chunks" / name).read_bytes() for name in head["chunks"]).splitlines(True)
    events = b"".join((*lines[:-1], corrupt(lines[-1], corruption, len(lines))))
    # Stored under its own digest, so the reader gets past content identity and must decode it.
    name = hashlib.sha256(events).hexdigest() + ".jsonl"
    (destination / "chunks" / name).write_bytes(events)
    (destination / HEAD).write_text(json.dumps({**head, "chunks": [name], "size": len(events)}))
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    driver_args.resume_held = True
    driver_args.work_dir += "-resume"
    with pytest.raises(LogFailure):
        await driver.one(driver_args)
    assert tree(destination) == before


MALFORMED_REQUEST = {
    "type": Resolved.kind,
    "payload": {"request": {}, "fingerprints": [], "value": None, "replayed": False},
}
FINISHED = {"stage": "review", "round": 1, "terminal": "accepted"}


@pytest.mark.parametrize(
    "change, event, code",
    [
        ({"format": "3"}, None, driver.CORRUPT),
        ({"format": 3.0}, None, driver.CORRUPT),
        ({"format": 2.0}, None, driver.CORRUPT),
        ({"format": True}, None, driver.CORRUPT),
        ({"format": 3}, None, driver.NEEDS_MIGRATION),
        ({"status": FINISHED, "workpiece": None}, None, driver.CORRUPT),
        ({}, MALFORMED_REQUEST, driver.CORRUPT),
    ],
    ids=["string", "float", "float 2", "bool", "format 3", "finished without workpiece", "resolved without request"],
)
async def test_driver_refuses_a_damaged_head_or_history_before_skipping_or_claiming(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, change: dict[str, object], event: object, code: int
) -> None:
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / "chunks").mkdir(parents=True)
    events = json.dumps({"sequence": 1, "wall_time": 0.0, "event": event}).encode() + b"\n" if event else b""
    chunks = (hashlib.sha256(events).hexdigest() + ".jsonl",) if events else ()
    for name in chunks:
        (destination / "chunks" / name).write_bytes(events)
    head = HEAD_JSON.dump_python(Head(2, "earlier", chunks, len(events), Status(), None, 0.0), mode="json")
    (destination / HEAD).write_text(json.dumps({**head, **change}))
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == code
    assert tree(destination) == before


async def test_a_collision_at_a_checkpoint_digest_exits_corrupt_and_publishes_no_hold(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other bytes already sit under the name of the first checkpoint's chunk; the real ``_put`` refuses them."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    put = checkpoint._put

    def collide_once(directory: str, suffix: str, data: bytes) -> str:
        target = Path(directory) / (hashlib.sha256(data).hexdigest() + suffix)
        if not target.parent.exists():
            target.parent.mkdir(parents=True)
            target.write_bytes(b"other bytes\n")
        return put(directory, suffix, data)

    monkeypatch.setattr(checkpoint, "_put", collide_once)
    built = resources(Envs(), script())

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), built

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    with pytest.raises(Exception) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.CORRUPT
    head = published(Path(driver_args.output) / driver.proposal_key(driver_args.proposals)).head
    assert (head.chunks, head.status, head.workpiece) == ((), Status(), None)


def serve(monkeypatch: pytest.MonkeyPatch, *attempts: Resources) -> None:
    """Each ``driver.one`` launch gets the next resources."""
    remaining = iter(attempts)

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(remaining)

    monkeypatch.setattr(driver, "runtime", fake_runtime)


def executions(*attempts: Resources) -> tuple[int, int]:
    """Inference requests and solver trials the attempts ran; replayed effects run nothing."""
    inference = [attempt.inference for attempt in attempts]
    solvers = [attempt.rollouts.engine for attempt in attempts]
    assert all(isinstance(client, FakeInference) for client in inference)
    assert all(isinstance(engine, SolverEngine) for engine in solvers)
    return (
        sum(len(cast(FakeInference, client).requests) for client in inference),
        sum(len(cast(SolverEngine, engine).started) for engine in solvers),
    )


def fail_put_once(monkeypatch: pytest.MonkeyPatch, error: Exception, directory: str = "chunks") -> list[bytes]:
    """Raise ``error`` on the first creation of an object under ``directory``; return every creation's bytes."""
    put, calls, failed = checkpoint._put, [], []

    def failing(target: str, suffix: str, data: bytes) -> str:
        calls.append(data)
        if Path(target).name == directory and not failed:
            failed.append(error)
            raise error
        return put(target, suffix, data)

    monkeypatch.setattr(checkpoint, "_put", failing)
    return calls


def s3_error(code: str, status: int) -> ClientError:
    response: Any = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}
    return ClientError(response, "PutObject")


@pytest.mark.parametrize(
    "error",
    [
        s3_error("SlowDown", 503),
        ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
        s3_error("BadGateway", 502),
    ],
    ids=["slowdown", "reset", "bad-gateway"],
)
async def test_a_recoverable_error_on_a_count_triggered_chunk_loses_no_paid_work(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The first checkpoint's chunk fails once; its lines stay pending, later events wait for the timer, and the final
    publication carries them, so a relaunch executes nothing again."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    calls = fail_put_once(monkeypatch, error)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [record.sequence for record in history(destination)] == list(range(1, len(history(destination)) + 1))
    # The failed chunk, then only the final chunk and workpiece: no count-triggered retry per later event.
    assert len(calls) == 3
    driver_args.work_dir = str(Path(driver_args.work_dir).parent / "relaunch")
    await driver.one(driver_args)
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


def short_body() -> IncompleteReadError:
    """What botocore's ``StreamingBody`` raises when the response ends before its declared length."""
    return IncompleteReadError(actual_bytes=17, expected_bytes=412)


async def test_a_short_head_read_at_a_count_triggered_checkpoint_loses_no_paid_work(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first checkpoint's head read ends short; its lines stay pending and the final publication carries them."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    read, reads = LocalConditionalObject.read, []

    def short_second_read(target: LocalConditionalObject) -> VersionedBytes | None:
        reads.append(target.path)
        if len(reads) == 2:  # After the launch's own read of the head, the first checkpoint's.
            raise short_body()
        return read(target)

    monkeypatch.setattr(LocalConditionalObject, "read", short_second_read)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [record.sequence for record in history(destination)] == list(range(1, len(history(destination)) + 1))
    driver_args.work_dir = str(Path(driver_args.work_dir).parent / "relaunch")
    await driver.one(driver_args)
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


async def test_a_short_chunk_read_on_resume_is_retried_and_repeats_no_paid_work(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preempted launch checkpointed every event but never published its end; the relaunch's first chunk read ends
    short, is read again, and the relaunch replays everything."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    instant_final_retries(monkeypatch)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    final_publish = CheckpointSink.finish

    async def preempted(*_args: object) -> None:
        """Only checkpoints reach storage, not the final publication."""

    monkeypatch.setattr(CheckpointSink, "finish", preempted)
    await driver.one(driver_args)
    monkeypatch.setattr(CheckpointSink, "finish", final_publish)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published(destination).head.status.terminal is None and published(destination).head.chunks
    open_url, opened = checkpoint.open_url, []

    def short_first_chunk(url: str, mode: str = "rb") -> object:
        opened.append(url)
        if len(opened) == 1:
            raise short_body()
        return open_url(url, mode)

    monkeypatch.setattr(checkpoint, "open_url", short_first_chunk)
    driver_args.work_dir = str(Path(driver_args.work_dir).parent / "relaunch")
    await driver.one(driver_args)
    assert opened[0] == opened[1]
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


@pytest.mark.parametrize(
    "refusal", [s3_error("AccessDenied", 403), PermissionError(errno.EACCES, "Permission denied")], ids=["s3", "local"]
)
async def test_a_storage_refusal_on_a_count_triggered_chunk_stops_publication_without_a_bug_hold(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, refusal: Exception
) -> None:
    calls = fail_put_once(monkeypatch, refusal)
    serve(monkeypatch, resources(Envs(), script()))
    with pytest.raises(Exception) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) is None and len(calls) == 1
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published(destination).head.status == Status()


def instant_final_retries(monkeypatch: pytest.MonkeyPatch, attempts: int = checkpoint.FINAL_ATTEMPTS) -> None:
    monkeypatch.setattr(checkpoint, "FINAL_ATTEMPTS", attempts)
    monkeypatch.setattr(checkpoint, "FINAL_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0))


@pytest.mark.parametrize("fault", ["workpiece", "head-lost", "head-applied"])
async def test_a_transient_error_in_final_publication_retries_without_repeating_solver_work(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Periodic checkpoints succeed; one SlowDown hits the terminal workpiece's creation or the final head swap, which
    may have been applied before its acknowledgment was lost. The final publication retries and reconciles."""
    instant_final_retries(monkeypatch)
    finals: list[Head] = []
    if fault == "workpiece":
        fail_put_once(monkeypatch, s3_error("SlowDown", 503), "workpieces")
    else:
        swap = CheckpointSink._swap

        def flaky(sink: CheckpointSink, head: Head, expected: str | None) -> None:
            if head.status.terminal is not None:
                finals.append(head)
                if len(finals) == 1:
                    if fault == "head-applied":
                        swap(sink, head, expected)
                    raise s3_error("SlowDown", 503)
            swap(sink, head, expected)

        monkeypatch.setattr(CheckpointSink, "_swap", flaky)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published(destination).head.status.terminal is Terminal.ACCEPTED
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [record.sequence for record in history(destination)] == list(range(1, len(history(destination)) + 1))
    # A lost swap is redone; an applied one is recognised from the observed head and not swapped again.
    assert len(finals) == {"workpiece": 0, "head-lost": 2, "head-applied": 1}[fault]
    driver_args.work_dir = str(Path(driver_args.work_dir).parent / "relaunch")
    await driver.one(driver_args)
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


async def test_a_final_publication_that_keeps_failing_exits_and_a_relaunch_resumes_from_the_last_checkpoint(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausted retries fail the process as an ordinary failure; the head keeps the last checkpoint (here the claim),
    and a relaunch replays the history it covers and re-executes every effect after it."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1000)
    instant_final_retries(monkeypatch, attempts=2)
    put = checkpoint._put

    def refuse_workpieces(target: str, suffix: str, data: bytes) -> str:
        if Path(target).name == "workpieces":
            raise s3_error("SlowDown", 503)
        return put(target, suffix, data)

    monkeypatch.setattr(checkpoint, "_put", refuse_workpieces)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    with pytest.raises(PublicationFailed) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) is None
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    head = published(destination).head
    assert (head.status, head.workpiece, head.chunks) == (Status(), None, ())
    monkeypatch.setattr(checkpoint, "_put", put)
    driver_args.work_dir = str(Path(driver_args.work_dir).parent / "relaunch")
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert executions(first) == executions(relaunch) == (len(script()), 2)


class TakenOverWhileBlocked(FakeInference):
    """On the third call another writer takes the root; the call then blocks like a long model call."""

    def __init__(self, completions: Sequence[Completion], destination: Path, workdir: Path) -> None:
        super().__init__(completions)
        self.destination, self.workdir = destination, workdir
        self.successor: CheckpointSink | None = None
        self.history: tuple[Record, ...] = ()
        self.cancelled = False

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if len(self.requests) == 2:
            (stale,) = self.workdir.glob("*.jsonl")
            while published(self.destination).head.size < stale.stat().st_size:
                await asyncio.sleep(0.01)  # Take over only once the old writer has nothing pending.
            self.successor = CheckpointSink(self.workdir, str(self.destination))
            self.history = self.successor.claim(published(self.destination))
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().complete(request)


async def test_a_writer_blocked_with_nothing_pending_stops_within_a_tick_of_a_takeover(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relaunch sharing the work directory takes the head; the idle old driver is cancelled and publishes nothing."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    identity = driver.proposal_key(driver_args.proposals)
    destination = Path(driver_args.output) / identity
    inference = TakenOverWhileBlocked(script(), destination, Path(driver_args.work_dir) / identity)
    first = replace(resources(Envs(), ()), inference=inference)

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), first

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    with pytest.raises(ExceptionGroup) as raised:
        await driver.one(driver_args)
    assert raised.group_contains(Superseded) and inference.cancelled
    journal = published(destination)
    assert inference.successor is not None and journal.head.writer == inference.successor.writer
    assert (journal.head.status, journal.head.workpiece) == (Status(), None)
    # The successor's private stream is exactly the committed history, and it resumes from all of it.
    assert Path(inference.successor.local.path).read_bytes() == journal.events()
    assert inference.history == decode(journal.events())


def leaves(error: BaseException) -> list[BaseException]:
    if not isinstance(error, BaseExceptionGroup):
        return [error]
    return [leaf for inner in error.exceptions for leaf in leaves(inner)]


async def test_a_cleanup_error_after_a_takeover_stays_a_failure_and_appends_nothing(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timer finds a successor and cancels the run; the runtime's own cleanup then fails independently."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    identity = driver.proposal_key(driver_args.proposals)
    destination = Path(driver_args.output) / identity
    inference = TakenOverWhileBlocked(script(), destination, Path(driver_args.work_dir) / identity)
    first = replace(resources(Envs(), ()), inference=inference)

    @asynccontextmanager
    async def failing_cleanup(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        try:
            yield build_config(policy, {}), first
        finally:
            raise RuntimeError("sandbox teardown failed")

    monkeypatch.setattr(driver, "runtime", failing_cleanup)
    with pytest.raises(ExceptionGroup) as raised:
        await driver.one(driver_args)
    # One leaf each: the stopped writer attempted no bug or hold record, which would raise its stop again.
    assert sorted(map(type, leaves(raised.value)), key=str) == [RuntimeError, Superseded]
    assert driver.exit_code(raised.value) is None
    assert inference.successor is not None
    assert published(destination).head.writer == inference.successor.writer


def in_process(monkeypatch: pytest.MonkeyPatch, command: list[str]) -> SimpleNamespace:
    """Run a child's real entry point in this process, exiting as the interpreter would."""

    def wait() -> int:
        monkeypatch.setattr(sys, "argv", command[3:])
        try:
            driver.main()
        except SystemExit as error:
            return cast(int, error.code)
        except Exception:
            return 1
        return 0

    return SimpleNamespace(wait=wait, poll=lambda: 0)


async def test_a_superseded_child_with_another_failure_fails_the_parent(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a child whose every failure is supersession is benign; a cleanup error beside it is a failure."""

    async def superseded_then_cleanup_failed(_: Namespace) -> None:
        raise ExceptionGroup("one", [Superseded("another writer took the head"), RuntimeError("cleanup failed")])

    monkeypatch.setattr(driver, "one", superseded_then_cleanup_failed)
    monkeypatch.setattr(driver.subprocess, "Popen", partial(in_process, monkeypatch))
    with pytest.raises(RuntimeError, match="1 workpieces failed"):
        await driver.run(driver_args)
    assert "failed=1 skipped=0 superseded=0" in capsys.readouterr().out


async def test_parent_summary_separates_superseded_and_corrupt_children_from_good_work(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    paths = {name: proposals / f"{name}.proposal.json" for name in ("corrupt", "good", "superseded")}
    for path in paths.values():
        path.write_bytes(Path(driver_args.proposals).read_bytes())
    driver_args.proposals, driver_args.concurrency = str(proposals), 1
    roots = {name: Path(driver_args.output) / driver.proposal_key(str(path)) for name, path in paths.items()}
    torn = b'{"sequence": 1'
    name = hashlib.sha256(torn).hexdigest() + ".jsonl"
    (roots["corrupt"] / "chunks").mkdir(parents=True)
    (roots["corrupt"] / "chunks" / name).write_bytes(torn)
    head = Head(2, "earlier", (name,), len(torn), Status(), None, 0.0)
    (roots["corrupt"] / HEAD).write_bytes(HEAD_JSON.dump_json(head))
    corrupt = tree(roots["corrupt"])
    workdir = Path(driver_args.work_dir) / driver.proposal_key(str(paths["superseded"]))
    taken = replace(resources(Envs(), ()), inference=TakenOverWhileBlocked(script(), roots["superseded"], workdir))
    built = iter((resources(Envs(), script()), taken))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(built)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    monkeypatch.setattr(driver.subprocess, "Popen", partial(in_process, monkeypatch))
    with pytest.raises(RuntimeError, match="1 workpieces failed") as raised:
        await driver.run(driver_args)
    assert str(paths["corrupt"]) in str(raised.value) and str(paths["superseded"]) not in str(raised.value)
    out = capsys.readouterr().out
    assert "failed=1 skipped=0 superseded=1 needs_migration=0 corrupt=1" in out
    lines = {line.split(" proposal=")[0]: line for line in out.splitlines() if line.startswith("OUTCOME")}
    assert f"{paths['corrupt']}: move this root aside" in lines[f"OUTCOME corrupt exit={driver.CORRUPT}"]
    assert f"{paths['superseded']}: another writer owns" in lines[f"OUTCOME superseded exit={driver.SUPERSEDED}"]
    assert published_workpiece(roots["good"]).terminal is Terminal.ACCEPTED
    assert tree(roots["corrupt"]) == corrupt


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
