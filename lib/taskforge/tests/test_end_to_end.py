# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import errno
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast, override

import httpx
import pytest
from botocore.exceptions import ClientError, IncompleteReadError
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.llm.fake_glm import serving
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeCommand, FakeEnv, FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import Act, LoopTime, ScriptedProvider, Time, chunk, report, segment
from lib.taskforge.tests.resources.iris_fakes import SITES, Fault, Injection, LocalController, local_iris_machine
from lib.taskforge.tests.stage_fakes import CandidateEngine, SolverEngine, solver_model, transcript
from pydantic import TypeAdapter, ValidationError
from rigging.filesystem.conditional_object import ConditionalWriteError, LocalConditionalObject, VersionedBytes
from rigging.timing import ExponentialBackoff
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import (
    Graded,
    GradedRollout,
    Model,
    Rollout,
    RolloutEvent,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
)
from rolloutengine.test_support import arithmetic_task
from shellbox.machine import Command, ExitReason, GuestHelperFailed, Machine, Result
from taskcompendium.decision_model import DecisionModelRequest
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

from taskforge import checkpoint, coordinator, driver, wiring
from taskforge.checkpoint import HEAD, HEAD_JSON, CheckpointSink, Head, Journal, read_journal
from taskforge.config import TaskforgeConfig, WorkspaceHydration
from taskforge.coordinator import STAGE_REPLAY_ATTEMPTS, advance, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import (
    Failed,
    Fingerprint,
    Note,
    ReplayPolicy,
    Resolved,
    StageCompleted,
    event_digest,
)
from taskforge.effects.issues import (
    CorruptJournal,
    DecisionModelUnavailable,
    Hold,
    LogFailure,
    OutputBudgetExhausted,
    PublicationFailed,
    Restart,
    Superseded,
    Transient,
    UnsupportedFormat,
)
from taskforge.effects.log import Billing, BufferSink, EventLog, Record, Status, decode, fold
from taskforge.harness.protocol import Stopped, StopReason
from taskforge.llm.client import AttemptFailed, AttemptOutcome, ModelClient, Progress, Provider, Segment
from taskforge.llm.glm_provider import EndpointSource, FixedEndpoint, GlmEndpoint, GlmProvider, Pool
from taskforge.llm.policy import GLM_5_3, ModelProfile
from taskforge.lowering import rollout_engine
from taskforge.model.controls import CONTROL, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal, Verification
from taskforge.model.review import (
    VALIDATION_RESULT,
    Abandon,
    Accept,
    AdversaryStatus,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    Decision,
    Incomplete,
    Modify,
    Retry,
    ReviewEnd,
    TrialOutcome,
    TrialResult,
)
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Fallback, Submission, Terminal, Workpiece
from taskforge.policy import Policy, load_policy, sampling
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report as usage_report
from taskforge.resources.bundle import Resources
from taskforge.resources.decision_model import (
    ClefReader,
    DecisionModel,
    DecisionModelDown,
    DecisionModelHealth,
    DecisionModelPolicy,
    DecisionModelRead,
    DecisionModelReader,
    read_level,
)
from taskforge.resources.env import (
    CommandResult,
    EnvFactory,
    ReadFile,
    ShellboxEnv,
    WriteFile,
    WriteFiles,
)
from taskforge.resources.grading import GradingResource
from taskforge.resources.ids import Ids
from taskforge.resources.inference import GlmInference, InferenceClient, InferenceRequest
from taskforge.resources.rollouts import Engine, RolloutResource, SolverRollout
from taskforge.resources.verifier import RoundVerifier
from taskforge.stages import STAGES
from taskforge.stages.author import AuthorReference
from taskforge.stages.classify import replayable_grade
from taskforge.stages.rules import default_review
from taskforge.wiring import build_config, build_resources, install, retry_policy, solver_turn


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
    def __init__(self, commands: Mapping[str, FakeCommand], fail: bool, image: str = "workspace") -> None:
        super().__init__(image, commands)
        self.executed: list[str] = []
        self.fail = fail

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.executed.append(cmd)
        if self.fail and len(self.executed) == 2:
            raise Restart("fake machine died")
        return await super().run(cmd, timeout=timeout)


class Envs(EnvFactory):
    def __init__(self, *, fail: bool = False, image: str = "workspace") -> None:
        self.fail, self.image = fail, image
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
        env = RecordingEnv(self.commands, self.fail, self.image)
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


def resources(envs: EnvFactory, responses: Sequence[Completion]) -> Resources:
    return Resources(
        envs,
        FakeInference(responses),
        RolloutResource(SolverEngine(2), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Ids(lambda: "workpiece"),
        {},
        FakeWeb({}, {}),
        {"solve_review": CleanReviewer()},
    )


SETTINGS: Mapping[str, object] = {
    "solver.k": 2,
    "adversary.roster": [],
    "budget.max_modify_rounds": 1,
    "solver.confirmation_k": 0,
}
"""Without the confirmation step, for the tests that count pre-confirmation events; ``CONFIRMING`` runs the
ordinary accept, partial-validation resume, confirmation-outage recovery and restored-fallback cases at the shipped
default, and ``test_confirmation.py`` the step itself."""

CONFIRMING: Mapping[str, object] = {key: value for key, value in SETTINGS.items() if key != "solver.confirmation_k"}
SHIPPED_K = load_policy(DEFAULTS_DIR).setting("solver.confirmation_k", int)
"""The confirmation count ``CONFIRMING`` inherits from the shipped policy."""


def config(settings: Mapping[str, object] = SETTINGS) -> TaskforgeConfig:
    return build_config(load_policy(DEFAULTS_DIR).overlaid(settings), {})


def confirming(driver_args: Namespace) -> None:
    """Run the driver at the shipped confirmation default."""
    overlay = Path(driver_args.policy)
    settings = {key: value for key, value in json.loads(overlay.read_text()).items() if key != "solver.confirmation_k"}
    overlay.write_text(json.dumps(settings))


def confirmed_by(wp: Workpiece, reviewer: InferenceClient, started: Sequence[str], round_: int = 0) -> None:
    """``wp`` accepted on the confirmation record of evidence index ``round_``: its two scored slots, then the shipped
    ``SHIPPED_K`` slots under the confirmation's attempt (``started`` the engine's run ids), read by a final review
    that reported."""
    record, final = wp.evidence[-1], wp.decisions[-1]
    assert wp.terminal is Terminal.ACCEPTED and isinstance(final, Accept) and record.confirms == round_
    assert SHIPPED_K > 0 and [trial.trial for trial in record.solver] == list(range(2 + SHIPPED_K))
    attempt = len(wp.evidence) - 1
    assert sorted(run for run in started if run.endswith(f":{attempt}")) == [
        f"arithmetic:{t}:{attempt}" for t in range(2, 2 + SHIPPED_K)
    ]
    assert record.review is not None and record.review.end is ReviewEnd.REPORTED
    assert isinstance(reviewer, CleanReviewer) and "Earlier solve-review reports" in str(
        reviewer.requests[-1].conversation
    )
    assert any(reason.startswith("Confirmation: ") for reason in final.reasons)


async def test_workflow_accepts_and_records_stage_sequence(proposal: TaskProposal) -> None:
    log, bundle = EventLog(), resources(Envs(), script())
    wp = await run_workpiece(proposal, config(CONFIRMING), bundle, log, install=install)
    stages = [record.event.stage for record in log.records if isinstance(record.event, StageCompleted)]
    assert stages == ["author", "validate", "review", "confirm", "review"]
    assert fold(tuple(record.event for record in log.records)).terminal == "accepted"
    engine = bundle.rollouts.engine
    assert isinstance(engine, SolverEngine)
    confirmed_by(wp, bundle.role_inference["solve_review"], engine.started)
    assert sorted(engine.started)[:2] == ["arithmetic:0:0", "arithmetic:1:0"] and len(engine.started) == 7


async def test_crash_resume_replays_commands_without_repeating_inference(proposal: TaskProposal) -> None:
    sink = BufferSink()
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


class FirstRoundSolves(Engine):
    """Every trial of the first validation solves (too easy); every later one grades the task invalid (a fault)."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        first = run_id.endswith(":0")
        grade = GradeResult(Outcome.GRADED, 1.0) if first else GradeResult(Outcome.INVALID_TASK, None, "bad task")
        return GradedRollout(Rollout(run_id, (transcript("12"),), "stop"), grade)


class RevisionFaults(SolverEngine):
    """Every trial solves (too easy), except that the revision's validation (attempt 1) grades the task invalid."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        solved = await super().run_task(spec, model, run_id=run_id)
        fault = run_id is not None and run_id.endswith(":1")
        grade = GradeResult(Outcome.INVALID_TASK, None, "bad task") if fault else GradeResult(Outcome.GRADED, 1.0)
        return replace(solved, grade=grade)


# Regression: a faulted revision resubmitted the same task and controls with new notes; review
# chose the stored fallback but inferred from the equal digest that it had chosen the revision, so restore=None and
# the faulted round's notes shipped as a clean acceptance. At the shipped default the restore is confirmed on its own
# round (index 0), not the faulted one of the same digest.
async def test_a_faulted_resubmission_with_the_same_digest_restores_the_stored_submission(
    proposal: TaskProposal,
) -> None:
    resubmit = (*script()[:3], call("submit", {"notes": "revised notes"}, "submit"))
    bundle = resources(Envs(), (*script(), *resubmit))
    bundle = replace(bundle, rollouts=replace(bundle.rollouts, engine=RevisionFaults(2)))
    wp = await run_workpiece(proposal, config(CONFIRMING), bundle, EventLog(), install=install)
    first, restore, final = wp.decisions
    assert isinstance(first, Modify) and first.placement is BandOutcome.TOO_EASY
    assert isinstance(restore, Accept) and wp.round == 1
    assert wp.evidence[0].task_digest == wp.evidence[1].task_digest  # the digest cannot tell the rounds apart
    assert wp.submission is not None and wp.submission.notes == "done" and restore.restore == wp.submission.digest
    assert any(reason.startswith("The revision has a fault; restored submission") for reason in restore.reasons)
    engine = bundle.rollouts.engine
    assert isinstance(engine, SolverEngine)
    confirmed_by(wp, bundle.role_inference["solve_review"], engine.started)
    assert wp.evidence[-1].solver[:2] == wp.evidence[0].solver  # the restored round's own slots
    assert isinstance(final, Accept) and final.band is BandOutcome.TOO_EASY and wp.accepted_with is BandOutcome.TOO_EASY


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
            return GradedRollout(Rollout(run_id, (transcript("12"),), "stop"), GradeResult(Outcome.UNAVAILABLE, None))

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
        return GradedRollout(
            Rollout(run_id, (transcript("12"),), "stop"), GradeResult(Outcome.GRADED, float(trial % 2 == 0))
        )


# Regression: a same-round validation retry inherited the earlier visit's replay attempts and held.
async def test_stage_replay_budget_restarts_after_each_completed_stage(proposal: TaskProposal) -> None:
    def retry_once(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
        if any(isinstance(decision, Retry) for decision in workpiece.decisions):
            return default_review(workpiece, band, budget)
        return Retry(Cause.AGENT_TIMEOUT, 1)

    solver = FlakySolver({0: STAGE_REPLAY_ATTEMPTS - 1, 1: 1})
    bundle = resources(Envs(), script())
    bundle = replace(bundle, rollouts=replace(bundle.rollouts, engine=solver))
    wp = await run_workpiece(proposal, replace(config(), review=retry_once), bundle, EventLog(), install=install)
    assert [type(decision) for decision in wp.decisions] == [Retry, Accept]
    assert wp.terminal == Terminal.ACCEPTED and not any(solver.failures.values())


def caller_providers() -> dict[str, ScriptedProvider]:
    author = ScriptedProvider(tuple(segment(c.content, c.tool_calls, FinishReason.TOOL_CALLS) for c in script()))
    return {
        "author": author,
        "solver": ScriptedProvider((segment("12"), segment("9"))),
        "adversary": ScriptedProvider(()),
    }


async def run_on(
    proposal: TaskProposal, providers: Mapping[str, Provider], profiles: Mapping[str, ModelProfile], log: EventLog
) -> Workpiece:
    """A workpiece through the public wiring on ``providers`` and ``profiles``, with fake machines and graders."""
    models = {f"{role}.model": provider.model for role, provider in providers.items()}
    policy = load_policy(DEFAULTS_DIR).overlaid({**SETTINGS, **models})
    built = build_resources(policy, providers, {}, FakeWeb({}, {}), profiles)
    bundle = replace(
        built,
        envs=Envs(),
        rollouts=replace(built.rollouts, engine=CandidateEngine()),
        grading=GradingResource(CandidateEngine(), "grader"),
        role_inference={**built.role_inference, "solve_review": CleanReviewer()},
    )
    return await run_workpiece(proposal, build_config(policy, {}, profiles), bundle, log, install=install)


async def test_a_workpiece_runs_end_to_end_on_a_caller_provider_and_profile(proposal: TaskProposal) -> None:
    """The library's GLM provider and GLM-5.3 profile are defaults: a caller passes its own per role through wiring."""
    providers = caller_providers()
    profiles = {"author": replace(GLM_5_3, name="glm-5.3/output-4096", max_output_tokens=4096)}
    wp = await run_on(proposal, providers, profiles, EventLog())
    assert wp.terminal == Terminal.ACCEPTED
    assert [body["max_tokens"] for body in providers["author"].bodies] == [4096] * len(script())
    solver = providers["solver"].bodies
    assert len(solver) == 2 and not any("max_tokens" in body for body in solver)


# Regression: a result recorded without an output limit replayed under a profile with one.
@pytest.mark.parametrize(
    ("change", "calls"), [("none", 0), ("output-limit", len(script()) + 2), ("formatter", len(script()) + 2)]
)
async def test_a_journal_replays_model_calls_only_under_the_profile_that_recorded_them(
    proposal: TaskProposal, change: str, calls: int
) -> None:
    profile = {
        "none": GLM_5_3,
        "output-limit": replace(GLM_5_3, name="glm-5.3/output-4", max_output_tokens=4),
        "formatter": replace(GLM_5_3, name="glm-5.3/spaced", replay_arguments=lambda arguments: f" {arguments}"),
    }[change]
    sink = BufferSink()
    assert (await run_on(proposal, caller_providers(), {}, EventLog(sink=sink))).terminal == Terminal.ACCEPTED
    providers = caller_providers()
    log = EventLog(replay_from=local_records(sink), sink=sink)
    await run_on(proposal, providers, {"author": profile, "solver": profile}, log)
    assert sum(len(provider.bodies) for provider in providers.values()) == calls


# Regression: the harness and the inference effect both applied the formatter, so the wire got it twice.
async def test_a_profile_formatter_shapes_a_replayed_tool_call_once(proposal: TaskProposal) -> None:
    spaced = replace(GLM_5_3, name="glm-5.3/spaced", replay_arguments=lambda arguments: f" {arguments}")
    providers = caller_providers()
    await run_on(proposal, providers, {"author": spaced}, EventLog())
    [replayed] = [message for message in providers["author"].bodies[1]["messages"] if message.get("tool_calls")]
    assert replayed["tool_calls"][0]["function"]["arguments"] == f" {script()[0].tool_calls[0].arguments}"


@pytest.fixture
def driver_args(proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    monkeypatch.setenv("GLM_API_KEY", "fake")
    source, overlay = tmp_path / "test.proposal.json", tmp_path / "policy.json"
    source.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    overlay.write_text(json.dumps({"proposals.environments": ["container"], **SETTINGS}))
    return Namespace(
        proposals=str(source),
        policy=str(overlay),
        output=str(tmp_path / "out"),
        resume_held=False,  # and no probe: an in-process caller built before the flag existed
        concurrency=2,
        buckets=4096,
        bucket_range=None,
        ranks=(),
        glm_base_url="http://fake/v1",
        glm_relay_job=None,
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

    def launch(command: list[str], **_: object) -> SimpleNamespace:
        launched.append(command[command.index("--proposals") + 1])
        assert command[command.index("--glm-base-url") + 1] == "http://fake/v1" and "--glm-relay-job" not in command
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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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

    # An Iris retry must not rerun this held workpiece.
    head = (destination / HEAD).read_bytes()
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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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
    await driver.one(args)
    assert published_workpiece(destination).terminal == Terminal.ACCEPTED
    assert isinstance(first.inference, FakeInference) and len(first.inference.requests) == 2
    assert isinstance(second.inference, FakeInference) and len(second.inference.requests) == len(script()) - 2
    head = (destination / HEAD).read_bytes()
    # Even with no resources left to construct, another fresh attempt exits successfully.
    await driver.one(args)
    assert (destination / HEAD).read_bytes() == head


@pytest.mark.parametrize("reason", tuple(StopReason))
def test_author_abandonment_publishes_stop_cause(proposal: TaskProposal, reason: StopReason) -> None:
    previous = Modify(("prior review",))
    wp = replace(Workpiece(proposal.header.id, proposal), decisions=(previous,))
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


# Pilot 2026-11-21 (d09): the causes of the replayed attempts before a hold reached only the driver log.
async def test_stage_restart_stops_after_three_attempts_with_cause(proposal: TaskProposal) -> None:
    envs = Envs(fail=True)
    resource = resources(envs, script())
    log = EventLog()
    result = await run_workpiece(proposal, config(), resource, log, install=install)
    assert result.terminal == Terminal.HELD
    assert len(envs.created) == 3
    assert "fake machine died" in str(result.decisions[-1])
    notes = [record.event.msg for record in log.records if isinstance(record.event, Note)]
    assert [note.split(":")[0] for note in notes] == [f"Stage author round 0 attempt {n} of 3" for n in (1, 2, 3)]
    assert all("fake machine died" in note for note in notes)


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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(attempts)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD
    head = (destination / HEAD).read_bytes()
    await driver.one(driver_args)
    assert (destination / HEAD).read_bytes() == head
    driver_args.resume_held = True
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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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
    "artefact, written, found",
    [
        ("events.jsonl", 1, "format 1"),
        ("events.jsonl.parts/000000000001.jsonl", 1, "format 1"),
        ("status.json", 1, "format 1"),
        ("workpiece.json", 1, "format 1"),
        # The version gate refuses every format but its own and format 8, whose rows it reads unchanged; the format-2
        # reader refuses this one's heads.
        (HEAD, 7, "format 7"),
        (HEAD, checkpoint.JOURNAL_FORMAT + 1, f"format {checkpoint.JOURNAL_FORMAT + 1}"),
    ],
)
async def test_driver_fails_closed_on_another_journal_format_without_touching_it(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, artefact: str, written: int, found: str
) -> None:
    """Format 1 wrote parts and sidecars before its consolidated stream; any one of them marks the root."""
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / artefact).parent.mkdir(parents=True)
    (destination / artefact).write_text(json.dumps({"format": written, "sequence": 1}) + "\n")
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure, match=f"journal {found}") as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.NEEDS_MIGRATION
    assert tree(destination) == before


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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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
        ({"format": checkpoint.JOURNAL_FORMAT + 1}, None, driver.NEEDS_MIGRATION),
        ({"status": FINISHED, "workpiece": None}, None, driver.CORRUPT),
        ({}, MALFORMED_REQUEST, driver.CORRUPT),
    ],
    ids=["string", "float", "float 2", "bool", "next format", "finished without workpiece", "resolved without request"],
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
    head = HEAD_JSON.dump_python(
        Head(checkpoint.JOURNAL_FORMAT, "earlier", chunks, len(events), Status(), None, 0.0), mode="json"
    )
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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), built

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    with pytest.raises(Exception) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.CORRUPT
    head = published(Path(driver_args.output) / driver.proposal_key(driver_args.proposals)).head
    assert (head.chunks, head.status, head.workpiece) == ((), Status(), None)


STALL_TIMEOUT = 600.0
TOKEN_INTERVAL = 10.0


class Fleet(Provider):
    """A solver fleet on a virtual clock. While it is up, requests get ``acts`` in order; ``lose`` takes every worker
    away, after which each request is refused with a 503 and ``/health`` reports the pool empty.

    As the rollout engine's observer, it sets ``served`` once a trial is graded, so a sibling act that advances the
    clock cannot move the healthy trial's grading deadline while its grader still runs."""

    def __init__(self, time: LoopTime, acts: Sequence[Act]) -> None:
        self.time, self.acts, self.alive = time, list(acts), True
        self.requests = 0
        self.served = asyncio.Event()

    @property
    @override
    def model(self) -> str:
        return "glm-5.3"

    @override
    async def stream(self, body: Mapping[str, Any], progress: Progress) -> Segment:
        self.requests += 1
        if not self.alive:
            raise AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, "no workers", 503)
        return await self.acts.pop(0)(progress)

    @override
    async def no_workers(self) -> bool:
        return not self.alive

    async def answer(self, _progress: Progress) -> Segment:
        return segment("12")

    async def on_event(self, event: RolloutEvent) -> None:
        if isinstance(event.payload, Graded) and event.payload.status is Outcome.GRADED:
            self.served.set()

    async def silent(self, seconds: float) -> None:
        for _ in range(int(seconds / TOKEN_INTERVAL)):
            await self.time.sleep(TOKEN_INTERVAL)

    async def generate(self, progress: Progress, seconds: float) -> None:
        for _ in range(int(seconds / TOKEN_INTERVAL)):
            await self.time.sleep(TOKEN_INTERVAL)
            report(progress, chunk(content="x"))

    async def stall(self) -> Segment:
        await self.silent(STALL_TIMEOUT)
        raise AttemptFailed(AttemptOutcome.STALLED, f"no token for {STALL_TIMEOUT}s")

    async def loses_workers_late(self, progress: Progress) -> Segment:
        """A healthy generation whose workers all go away at second 4,060 of the 4,200-second bound."""
        await self.served.wait()
        await self.generate(progress, 4060)
        self.alive = False
        return await self.stall()

    async def stalls_in_a_tool_call(self, progress: Progress) -> Segment:
        await self.served.wait()
        report(progress, chunk(role="assistant"))
        report(progress, chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "shell"}}]))
        return await self.stall()

    async def streams_a_tool_delta_then_loses_workers(self, progress: Progress) -> Segment:
        report(progress, chunk(tool_calls=[{"index": 0, "function": {"arguments": "{"}}]))
        self.alive = False
        return await self.stall()

    async def streams_empty_tool_deltas(self, progress: Progress) -> Segment:
        """Eleven tool-call frames 415 seconds apart, carrying only an index or empty arguments: no generated text."""
        await self.served.wait()
        for frame in range(11):
            if frame:
                await self.time.sleep(415)
            report(
                progress, chunk(tool_calls=[{"index": 0, "function": {"arguments": ""}} if frame % 2 else {"index": 0}])
            )
        self.alive = False
        return await self.stall()

    async def aborts_past_the_bound(self, progress: Progress) -> Segment:
        """Content every 10 seconds past the bound; at second 4,220 a chunk carries content and the server's abort."""
        await self.served.wait()
        await self.generate(progress, 4210)
        await self.time.sleep(TOKEN_INTERVAL)
        report(progress, json.dumps({"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "abort"}]}))
        self.alive = False
        raise AttemptFailed(AttemptOutcome.ABORTED, "server aborted the request")

    async def delivers_a_buffered_token_past_the_bound(self, progress: Progress) -> Segment:
        """A token buffered at second 4,199.5 of the bound arrives at 4,200.5 from a worker that died meanwhile, while
        another replica keeps ``/health`` positive; then the stream ends."""
        await self.served.wait()
        await self.generate(progress, 4200)
        await self.time.sleep(TOKEN_INTERVAL + 0.5)
        report(progress, chunk(content="x"))
        self.alive = False
        raise AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream lacked finish_reason or usage")

    async def tails_past_the_bound(self, progress: Progress) -> Segment:
        """The worker exits at second 4,200 of the bound; its queued output keeps arriving past the bound, two frames
        1.2 s apart."""
        await self.served.wait()
        await self.generate(progress, 4200)
        for delay in (TOKEN_INTERVAL + 0.5, 1.2):
            await self.time.sleep(delay)
            report(progress, chunk(content="x"))
        raise AssertionError("the second frame past the bound confirms it")

    async def generates_past_the_bound(self, progress: Progress) -> Segment:
        await self.served.wait()
        await self.generate(progress, 8400)
        raise AssertionError("the generation bound ends this stream")


FLEET_FAULTS = {
    "late-fleet-loss": lambda fleet: (fleet.loses_workers_late, fleet.answer),
    "second-tool-stall": lambda fleet: (
        fleet.stalls_in_a_tool_call,
        fleet.answer,
        fleet.streams_a_tool_delta_then_loses_workers,
    ),
    "empty-tool-deltas": lambda fleet: (fleet.streams_empty_tool_deltas, fleet.answer),
    "abort-past-the-bound": lambda fleet: (fleet.aborts_past_the_bound, fleet.answer),
    "buffered-token-past-the-bound": lambda fleet: (fleet.delivers_a_buffered_token_past_the_bound, fleet.answer),
}


def fleet_resources(policy: Policy, fleet: Fleet) -> Resources:
    """The production solver path (RolloutEngine, ``solver_turn``, ``ModelClient``) on ``fleet`` and its clock."""
    author = ScriptedProvider(
        tuple(segment(c.content, c.tool_calls, FinishReason.TOOL_CALLS) for c in script()),
        policy.setting("author.model", str),
    )
    adversary = ScriptedProvider((), policy.setting("adversary.model", str))
    built = build_resources(policy, {"author": author, "solver": fleet, "adversary": adversary}, {}, FakeWeb({}, {}))
    model = ModelClient(
        fleet, GLM_5_3, retry_policy(policy, "solver"), role="solver", clock=fleet.time.clock, sleep=fleet.time.sleep
    )
    engine = rollout_engine({}, command_timeout=None)
    engine.observer = fleet
    rollouts = replace(
        built.rollouts,
        engine=engine,
        model=partial(solver_turn, model, sampling(policy, GLM_5_3.sampling)),
    )
    grading = GradingResource(CandidateEngine(), "grader")
    return replace(
        built,
        envs=Envs(),
        rollouts=rollouts,
        grading=grading,
        role_inference={**built.role_inference, "solve_review": CleanReviewer()},
    )


def solver_slots(destination: Path) -> list[Resolved[object]]:
    """The solver trials the journal resolved by running them; a stage replay serves them again as replayed."""
    return [
        record.event
        for record in history(destination)
        if isinstance(record.event, Resolved)
        and isinstance(record.event.request, SolverRollout)
        and not record.event.replayed
    ]


# Regression: each fault journaled a failed solver slot and published ACCEPTED at 1/2. The faults also include
# empty tool deltas, an abort past the bound, and a buffered token delivered past it from a dead worker.
@pytest.mark.parametrize("fault", FLEET_FAULTS)
async def test_a_fleet_fault_during_a_solver_turn_records_no_failed_trial_and_holds(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    fleet = Fleet(LoopTime(), ())
    fleet.acts = list(FLEET_FAULTS[fault](fleet))
    policy = driver.run_policy(driver_args.policy)
    serve(monkeypatch, fleet_resources(policy, fleet))
    await driver.one(driver_args)
    destination = tmp_path / "out" / driver.proposal_key(driver_args.proposals)
    [solved] = solver_slots(destination)
    assert cast(GradedRollout, solved.value).grade.reward == 1
    workpiece = published_workpiece(destination)
    assert (workpiece.terminal, workpiece.evidence) == (Terminal.HELD, ())


# Regression: a confirmed tail from an exited worker is still charged (a known limit of the
# client-side bound, which consults no pool state).
@pytest.mark.parametrize("tail", [False, True], ids=["live", "exited-worker-tail"])
async def test_a_solver_generation_sustained_past_its_bound_is_a_failed_trial(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail: bool
) -> None:
    fleet = Fleet(LoopTime(), ())
    past = fleet.tails_past_the_bound if tail else fleet.generates_past_the_bound
    fleet.acts = [past, fleet.answer]
    serve(monkeypatch, fleet_resources(driver.run_policy(driver_args.policy), fleet))
    await driver.one(driver_args)
    destination = tmp_path / "out" / driver.proposal_key(driver_args.proposals)
    assert len(solver_slots(destination)) == 2 and fleet.requests == 2
    workpiece = published_workpiece(destination)
    [evidence] = workpiece.evidence
    assert sorted(trial.outcome for trial in evidence.solver) == [TrialOutcome.GRADED, TrialOutcome.TIME_LIMIT]
    assert workpiece.terminal == Terminal.ACCEPTED


CUT_SEEN = 30.0
"""Seconds within which the fake router's next write sees that the client closed a connection."""


# Regression: a model 503 whose body trickled whitespace every 5 ms kept the author call pending
# (1 POST, 0 GETs, nothing resolved) under a 50 ms stall timeout and a 100 ms availability window.
async def test_a_trickling_error_body_holds_the_workpiece_within_the_availability_window(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window's clock moves only when the client backs off: each window holds two attempts and one health poll.
    window = {
        "provider.unavailable_wait": 3600.0,
        "provider.backoff_initial": 1800.0,
        "provider.backoff_maximum": 1800.0,
    }
    policy = driver.run_policy(driver_args.policy).overlaid(window)
    clock = Time()
    with serving() as fake:
        for _ in range(100):
            fake.status(503, "busy", trickle=400)  # each body takes 2 s to arrive; the stall timeout is 50 ms
        source = FixedEndpoint(GlmEndpoint(fake.base_url, "token", Pool.HIGH))
        async with GlmProvider(source, policy.setting("author.model", str), 0.05, 8) as author:
            roles = ("solver", "adversary")
            providers = {role: ScriptedProvider((), policy.setting(f"{role}.model", str)) for role in roles}
            built = build_resources(policy, {**providers, "author": author}, {}, FakeWeb({}, {}))
            retry = retry_policy(policy, "author")
            client = GlmInference(
                ModelClient(author, GLM_5_3, retry, role="author", clock=clock.clock, sleep=clock.sleep), "author"
            )
            serve(
                monkeypatch,
                replace(
                    built,
                    envs=Envs(),
                    inference=client,
                    role_inference={**built.role_inference, "author": client, "solve_review": CleanReviewer()},
                ),
            )
            await driver.one(driver_args)
        # Every attempt was cut before its body completed (test_provider pins the configured deadline).
        assert all(fake.cut.acquire(timeout=CUT_SEEN) for _ in fake.requests), "an error body was read to its end"
    destination = tmp_path / "out" / driver.proposal_key(driver_args.proposals)
    workpiece = published_workpiece(destination)
    assert (workpiece.terminal, solver_slots(destination)) == (Terminal.HELD, [])
    resolved = [record.event for record in history(destination) if isinstance(record.event, Resolved)]
    assert not [event for event in resolved if isinstance(event.request, InferenceRequest)]
    # Each stage attempt ended when its window closed, not when a body completed.
    assert (len(fake.requests), fake.health_polls) == (2 * STAGE_REPLAY_ATTEMPTS, STAGE_REPLAY_ATTEMPTS)
    assert clock.waits == [1800.0] * 2 * STAGE_REPLAY_ATTEMPTS


FORMAT_12_RETRY = Path(__file__).parent / "fixtures" / "format12-retry-accepted-root.json.gz"
"""The whole root the format-12 writer published for an accepted run whose first control grade was
unavailable: its same-round Retry stores the repeated ``stage.completed`` marker (row 61) as a digest reference."""
FORMAT_12_REPORT = (
    "workpieces=1 since_hours=all",
    "terminal=running count=0",
    "terminal=accepted count=1",
    "terminal=rejected count=0",
    "terminal=abandoned count=0",
    "terminal=held count=0",
    *(
        f"solver_trial_outcomes scope=terminal_records outcome={outcome} count={int(outcome == 'graded') * 4}"
        for outcome in ("graded", "generation_limit", "time_limit", "budget_exhausted", "faulted", "unscored")
    ),
    *(
        f"tokens role={role} prompt=0 cached=0 completion=0"
        for role in ("author", "solver", "adversary", "solve_review", "judge", "confirmation", "final_review")
    ),
    "stage=author n=1 p50_seconds=0.014 p90_seconds=0.014",
    "stage=validate n=2 p50_seconds=0.002 p90_seconds=0.011",
    "stage=review n=0 p50_seconds=n/a p90_seconds=n/a",
    "stage=confirm n=0 p50_seconds=n/a p90_seconds=n/a",
    "event_gap=RunCommand n=2 p50_seconds=0.000 p90_seconds=0.000",
    "event_gap=WriteFile n=0 p50_seconds=n/a p90_seconds=n/a",
    "event_gap=InferenceRequest n=4 p50_seconds=0.000 p90_seconds=0.000",
)
"""The format-12 report of ``FORMAT_12_RETRY``, its clock-relative acceptance rates aside."""


# Regression: the format-13 reader hashed a marker without a commitment otherwise, so this root's digest reference
# named no decoded event and the report failed on it.
async def test_the_report_reads_a_format_12_root_whose_markers_repeat_as_its_own_report_does(tmp_path: Path) -> None:
    root = plant_root(tmp_path / "out" / "retry", FORMAT_12_RETRY)
    plant_root(tmp_path / "out" / "held", FORMAT_11_HELD)
    alone = [line for line in await usage_report(str(root)) if not line.startswith("accepted_per_hour")]
    together = [line for line in await usage_report(str(tmp_path / "out")) if not line.startswith("accepted_per_hour")]
    assert tuple(alone) == FORMAT_12_REPORT
    # The format-12 report of both roots: the format-11 author hold adds its hold and its author's samples.
    assert together == [
        "workpieces=2 since_hours=all",
        *FORMAT_12_REPORT[1:5],
        "terminal=held count=1",
        "terminal=held reason=Hold count=1",
        *FORMAT_12_REPORT[6:19],
        "stage=author n=2 p50_seconds=0.011 p90_seconds=0.014",
        *FORMAT_12_REPORT[20:23],
        "event_gap=RunCommand n=4 p50_seconds=0.000 p90_seconds=0.000",
        FORMAT_12_REPORT[24],
        "event_gap=InferenceRequest n=7 p50_seconds=0.000 p90_seconds=0.000",
    ]


def serve(monkeypatch: pytest.MonkeyPatch, *attempts: Resources, workspace: WorkspaceHydration | None = None) -> None:
    """Each ``driver.one`` launch gets the next resources, and ``workspace`` when given."""
    remaining = iter(attempts)

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        config = build_config(policy, {})
        yield (config if workspace is None else replace(config, workspace=workspace)), next(remaining)

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
    await driver.one(driver_args)
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


@pytest.mark.parametrize(
    ("fault", "applied"),
    [("claim", True), ("checkpoint", True), ("first chunk", False), ("final head", False)],
    ids=["412-after-applied-claim", "412-after-applied-checkpoint", "409-on-first-chunk", "409-on-final-head"],
)
async def test_an_inconclusive_conditional_write_is_decided_by_reading_the_target(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, fault: str, applied: bool
) -> None:
    """One conditional write fails as a 412 that follows the SDK resending an applied write whose acknowledgement was
    lost, or as a retryable 409 that left the target as it was. The sole writer finishes and repeats no paid work."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    instant_final_retries(monkeypatch)
    write, heads, chunks, fired = LocalConditionalObject.write, [], [], []

    def inconclusive(target: LocalConditionalObject, data: bytes, *, expected_version: str | None) -> str:
        is_head = target.path.endswith(HEAD)
        (heads if is_head else chunks).append(data)
        chosen = {
            "claim": is_head and len(heads) == 1,
            "checkpoint": is_head and len(heads) == 2,
            "first chunk": PurePosixPath(target.path).parent.name == "chunks" and len(chunks) == 1,
            "final head": is_head and HEAD_JSON.validate_json(data).status.terminal is not None,
        }[fault]
        if not chosen or fired:
            return write(target, data, expected_version=expected_version)
        fired.append(fault)
        if applied:
            write(target, data, expected_version=expected_version)
        raise ConditionalWriteError(f"conditional write failed for {target.path}")

    monkeypatch.setattr(LocalConditionalObject, "write", inconclusive)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert fired == [fault]
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [record.sequence for record in history(destination)] == list(range(1, len(history(destination)) + 1))
    await driver.one(driver_args)
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


@pytest.mark.parametrize("change", [{"workpiece": ""}, {"size": -1}], ids=["empty workpiece", "negative size"])
async def test_a_malformed_terminal_head_is_corrupt_from_one_head_read(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, change: dict[str, object]
) -> None:
    serve(monkeypatch, resources(Envs(), script()))
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / HEAD).write_text(json.dumps({**json.loads((destination / HEAD).read_text()), **change}))
    before, read, reads = tree(destination), LocalConditionalObject.read, []

    def counted(target: LocalConditionalObject) -> VersionedBytes | None:
        reads.append(target.path)
        return read(target)

    monkeypatch.setattr(LocalConditionalObject, "read", counted)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.CORRUPT
    assert reads == [str(destination / HEAD)] and tree(destination) == before


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
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert executions(first) == executions(relaunch) == (len(script()), 2)


# Regression: the terminal event was durable but the driver finalised its initial workpiece, publishing an
# accepted head with no submission that every later driver skipped.
async def test_cancellation_while_the_terminal_event_is_written_leaves_a_resumable_head(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached, release, draining = asyncio.Event(), asyncio.Event(), asyncio.Event()
    write, drain = BufferSink.write, Context.drain

    async def paused(sink: BufferSink, records: Sequence[Record]) -> None:
        await write(sink, records)
        if any(isinstance(r.event, StageCompleted) and r.event.terminal is Terminal.ACCEPTED for r in records):
            reached.set()
            await release.wait()

    async def root_drain(ctx: Context) -> None:
        draining.set()
        await drain(ctx)

    monkeypatch.setattr(BufferSink, "write", paused)
    monkeypatch.setattr(Context, "drain", root_drain)
    first, relaunch = resources(Envs(), script()), resources(Envs(), script())
    serve(monkeypatch, first, relaunch)
    launch = asyncio.create_task(driver.one(driver_args))
    await reached.wait()
    await draining.wait()
    launch.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await launch
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published(destination).head.status == Status()
    assert fold([record.event for record in history(destination)]).terminal is Terminal.ACCEPTED
    monkeypatch.setattr(BufferSink, "write", write)
    await driver.one(driver_args)
    accepted = published_workpiece(destination)
    assert published(destination).head.status.terminal is accepted.terminal is Terminal.ACCEPTED
    assert accepted.submission is not None
    assert executions(first) == (len(script()), 2) and executions(relaunch) == (0, 0)


class TakenOverWhileBlocked(FakeInference):
    """On the third call another writer takes the root; the call then blocks like a long model call."""

    def __init__(self, completions: Sequence[Completion], destination: Path, writers: list[CheckpointSink]) -> None:
        super().__init__(completions)
        self.destination, self.writers = destination, writers
        self.successor: CheckpointSink | None = None
        self.history: tuple[Record, ...] = ()
        self.cancelled = False

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if len(self.requests) == 2:
            (stale,) = [writer.local for writer in self.writers if writer.destination == str(self.destination)]
            while published(self.destination).head.size < stale.offset + len(stale.data):
                await asyncio.sleep(0.01)  # Take over only once the old writer has nothing pending.
            self.successor = CheckpointSink(str(self.destination))
            self.history = self.successor.claim(published(self.destination))
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().complete(request)


def writers(monkeypatch: pytest.MonkeyPatch) -> list[CheckpointSink]:
    """Every checkpoint writer the driver builds from now on."""
    built: list[CheckpointSink] = []

    def build(destination: str) -> CheckpointSink:
        built.append(CheckpointSink(destination))
        return built[-1]

    monkeypatch.setattr(driver, "CheckpointSink", build)
    return built


async def test_a_writer_blocked_with_nothing_pending_stops_within_a_tick_of_a_takeover(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relaunch takes the head; the idle old driver is cancelled and publishes nothing."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    identity = driver.proposal_key(driver_args.proposals)
    destination = Path(driver_args.output) / identity
    inference = TakenOverWhileBlocked(script(), destination, writers(monkeypatch))
    first = replace(resources(Envs(), ()), inference=inference)

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), first

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    with pytest.raises(ExceptionGroup) as raised:
        await driver.one(driver_args)
    assert raised.group_contains(Superseded) and inference.cancelled
    journal = published(destination)
    assert inference.successor is not None and journal.head.writer == inference.successor.writer
    assert (journal.head.status, journal.head.workpiece) == (Status(), None)
    # The successor holds nothing unpublished and resumes from all of the committed history.
    assert (inference.successor.local.offset, inference.successor.local.data) == (journal.head.size, b"")
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
    inference = TakenOverWhileBlocked(script(), destination, writers(monkeypatch))
    first = replace(resources(Envs(), ()), inference=inference)

    @asynccontextmanager
    async def failing_cleanup(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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


def in_process(monkeypatch: pytest.MonkeyPatch, command: list[str], **_: object) -> SimpleNamespace:
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


class HeldChild:
    """A child process that runs until it is terminated."""

    def __init__(self, command: list[str]) -> None:
        self.proposal = command[command.index("--proposals") + 1]
        self.stopped = threading.Event()
        self.joined = False

    def wait(self) -> int:
        self.stopped.wait()
        self.joined = True
        return -15

    def poll(self) -> int | None:
        return -15 if self.stopped.is_set() else None

    def terminate(self) -> None:
        self.stopped.set()


def held_children(driver_args: Namespace, tmp_path: Path, names: Sequence[str]) -> list[Path]:
    paths = [tmp_path / f"{name}.proposal.json" for name in names]
    for name, path in zip(names, paths, strict=True):
        path.write_bytes(b"{" if name == "bad" else Path(driver_args.proposals).read_bytes())
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(map(str, paths)))
    driver_args.proposals, driver_args.concurrency = str(manifest), len(paths)
    return paths


async def test_a_failing_dispatch_terminates_and_joins_every_held_child(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = held_children(driver_args, tmp_path, ("first", "second", "bad"))
    children: list[HeldChild] = []
    monkeypatch.setattr(
        driver.subprocess, "Popen", lambda command, **_: children.append(HeldChild(command)) or children[-1]
    )
    with pytest.raises(ValidationError):
        await driver.run(driver_args)
    assert [child.proposal for child in children] == [str(path) for path in paths[:2]]
    assert all(child.stopped.is_set() and child.joined for child in children)
    assert "SUMMARY" not in capsys.readouterr().out


async def test_a_malformed_first_input_starts_no_child(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch reads each input before it starts that input's child, so a malformed first one starts none."""
    held_children(driver_args, tmp_path, ("bad", "first", "second"))
    children: list[HeldChild] = []
    monkeypatch.setattr(
        driver.subprocess, "Popen", lambda command, **_: children.append(HeldChild(command)) or children[-1]
    )
    with pytest.raises(ValidationError):
        await driver.run(driver_args)
    assert children == []


async def test_a_cancelled_run_terminates_and_joins_every_held_child(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = held_children(driver_args, tmp_path, ("first", "second"))
    children: list[HeldChild] = []
    launched = asyncio.Event()

    def launch(command: list[str], **_: object) -> HeldChild:
        children.append(HeldChild(command))
        if len(children) == len(paths):
            launched.set()
        return children[-1]

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    running = asyncio.create_task(driver.run(driver_args))
    await launched.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert [child.proposal for child in children] == list(map(str, paths))
    assert all(child.stopped.is_set() and child.joined for child in children)


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
    head = Head(checkpoint.JOURNAL_FORMAT, "earlier", (name,), len(torn), Status(), None, 0.0)
    (roots["corrupt"] / HEAD).write_bytes(HEAD_JSON.dump_json(head))
    corrupt = tree(roots["corrupt"])
    inference = TakenOverWhileBlocked(script(), roots["superseded"], writers(monkeypatch))
    taken = replace(resources(Envs(), ()), inference=inference)
    built = iter((resources(Envs(), script()), taken))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
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
    if not pin_prompt:
        with pytest.raises(AssertionError, match="already-resolved"):
            await driver.one(driver_args)
        return
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert [ResolvedTurnGuard.turn(request)[-1].tool_call_id for request in resumed.requests] == ["write", "grade"]


STAMP = PurePosixPath("/workspace/stamp.txt")


class DriftingEnvs(Envs):
    """Each machine's ``stamp`` writes the same file but prints its own launch and machine, like a PID.

    ``stamp`` is the given command result instead when one is scripted for the launch.
    """

    def __init__(self, launch: int, stamp: FakeCommand | None = None, *, image: str = "workspace") -> None:
        super().__init__(image=image)
        self.launch, self.stamp = launch, stamp

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = await super().create(requirements)
        drifting = FakeCommand(CommandResult(0, f"pid {self.launch}{len(self.created)}\n", "", 0, False), {STAMP: b"ok"})
        env.commands = {**env.commands, "stamp": self.stamp or drifting}
        return env


def drifting_script() -> tuple[Completion, ...]:
    """Run a command with drifting output, read the file it wrote, then continue the ordinary script."""
    return (
        call("shell", {"command": "stamp", "timeout": 5}, "stamp"),
        call("read_file", {"path": str(STAMP)}, "read"),
        *script()[1:],
    )


class ServedOnce(FakeInference):
    """Fails any request an earlier launch already served: a replay that misses its recording pays twice."""

    def __init__(self, completions: Sequence[Completion | Hold], served: Sequence[InferenceRequest]) -> None:
        super().__init__(completions)
        self.served = {event_digest(request) for request in served}
        self.repeated: list[InferenceRequest] = []

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if event_digest(request) in self.served:
            self.repeated.append(request)
            raise RuntimeError("an already served inference request was sent again")
        return await super().complete(request)


async def held_after_stamp(
    monkeypatch: pytest.MonkeyPatch, driver_args: Namespace, stamp: FakeCommand | None = None
) -> list[InferenceRequest]:
    """Hold the drifting author on its fourth model call and return the requests it was served."""
    held = FakeInference((*drifting_script()[:3], Hold("relay unavailable")))
    first = replace(resources(DriftingEnvs(1, stamp), ()), inference=held)
    serve(monkeypatch, first)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD
    assert isinstance(first.inference, FakeInference)
    driver_args.resume_held = True
    return first.inference.requests[:-1]  # the last request raised the hold


def tool_output(workpiece: Workpiece, call_id: str) -> str:
    assert workpiece.session is not None
    (message,) = (message for message in workpiece.session.messages if message.tool_call_id == call_id)
    return message.content


PID = FakeCommand(CommandResult(0, "pid 11\n", "", 0, False), {STAMP: b"ok"})
"""The first launch's first machine's ``stamp``, for a later machine that prints exactly what it printed."""


@pytest.mark.parametrize(
    ("stamp", "stdout", "reused"),
    [
        (PID, "pid 11\n", 3),
        (None, "pid 21\n", 1),
        (FakeCommand(CommandResult(0, "pid 11\n", "", 0, True), {STAMP: b"ok"}), "pid 11\n", 1),
    ],
    ids=["identical feedback", "stdout drifts", "truncation only"],
)
async def test_an_interrupted_author_reruns_its_commands_and_buys_turns_from_the_first_changed_one(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, stamp: FakeCommand | None, stdout: str, reused: int
) -> None:
    """The resumed author opens one fresh machine and runs ``stamp`` there. With the output its first launch saw, all
    three recorded turns hit the cache; with other output the turn that sees it misses, and every turn after it is
    bought."""
    served = await held_after_stamp(monkeypatch, driver_args)
    resumed = replace(resources(DriftingEnvs(2, stamp), ()), inference=ServedOnce(drifting_script()[reused:], served))
    serve(monkeypatch, resumed)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    workpiece = published_workpiece(destination)
    assert workpiece.terminal is Terminal.ACCEPTED
    assert isinstance(resumed.inference, ServedOnce) and not resumed.inference.repeated
    assert len(resumed.inference.requests) == len(drifting_script()) - reused
    assert json.loads(tool_output(workpiece, "stamp"))["stdout"] == stdout
    assert tool_output(workpiece, "read") == "ok"
    assert isinstance(resumed.envs, Envs)
    author = resumed.envs.created[0]
    assert author.executed == ["stamp", "write-draft"] and author.files[STAMP] == b"ok"


# Regression: resuming a partly validated workpiece must keep its finished solver trials.
async def test_resume_with_completed_solver_siblings_repeats_no_trial_or_model_call(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    confirming(driver_args)
    first = resources(Envs(), script())
    first = replace(first, rollouts=replace(first.rollouts, engine=SolverEngine(2, restart=1)))
    resumed = resources(Envs(), ())
    resumed = replace(
        resumed,
        inference=ServedOnce((), ()),
        rollouts=replace(resumed.rollouts, engine=SolverEngine(1)),
    )
    serve(monkeypatch, first, resumed)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD
    assert isinstance(first.inference, FakeInference) and isinstance(resumed.inference, ServedOnce)
    resumed.inference.served = {event_digest(request) for request in first.inference.requests}
    assert isinstance(first.rollouts.engine, SolverEngine)
    assert sorted(first.rollouts.engine.started)[0].endswith(":0:0")
    driver_args.resume_held = True
    await driver.one(driver_args)
    wp = published_workpiece(destination)
    assert not resumed.inference.repeated
    assert executions(resumed) == (0, 6)  # the unfinished validation slot, then the five confirmation slots
    assert isinstance(resumed.rollouts.engine, SolverEngine) and resumed.rollouts.engine.started[0].endswith(":1:0")
    confirmed_by(wp, resumed.role_inference["solve_review"], resumed.rollouts.engine.started)
    assert isinstance(reviewer := resumed.role_inference["solve_review"], CleanReviewer) and len(reviewer.requests) == 2


class ConfirmationOutage(SolverEngine):
    """``SolverEngine``, except that confirmation slot 4 meets a decision-model outage while it is graded."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if run_id == "arithmetic:4:1":
            self.started.append(run_id)
            error = RolloutInterrupted(Rollout(run_id, (), "interrupted"), RolloutOperation.GRADE)
            error.__cause__ = OUTAGE
            raise error
        return await super().run_task(spec, model, run_id=run_id)


async def test_a_confirmation_outage_parks_and_the_next_launch_buys_only_what_it_left_unbought(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decision-model outage in the confirmation batch parks the root (exit 75); the next launch, unasked, replays
    the validation, its review and the finished confirmation slots, and buys slot 4, the slots the outage kept from
    starting, and the final review."""
    confirming(driver_args)
    first = replace(
        resources(Envs(), script()), rollouts=replace(resources(Envs(), ()).rollouts, engine=ConfirmationOutage(2))
    )
    resumed = replace(resources(Envs(), ()), inference=ServedOnce((), ()))
    resumed = replace(resumed, rollouts=replace(resumed.rollouts, engine=SolverEngine(1)))
    serve(monkeypatch, first, resumed)
    assert await driver.one(driver_args) == driver.DEFERRED
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    held = published_workpiece(destination)
    assert held.terminal is Terminal.HELD and isinstance(last := held.decisions[-1], Incomplete)
    assert last.awaits == "decision_model" and held.accepted_with is None
    assert isinstance(first.inference, FakeInference) and isinstance(resumed.inference, ServedOnce)
    resumed.inference.served = {event_digest(request) for request in first.inference.requests}
    assert await driver.one(driver_args) == 0  # driver_args.resume_held stays False
    before, engine = first.rollouts.engine, resumed.rollouts.engine
    assert isinstance(before, SolverEngine) and isinstance(engine, SolverEngine)
    finished = {run for run in before.started if run != "arithmetic:4:1"}
    assert engine.started[0] == "arithmetic:4:1"
    assert not finished & set(engine.started)  # slots the outage stopped before they started are bought now, once
    confirmed_by(published_workpiece(destination), resumed.role_inference["solve_review"], [*finished, *engine.started])
    assert isinstance(reviewer := resumed.role_inference["solve_review"], CleanReviewer) and len(reviewer.requests) == 1


N4_JOURNAL = Path(__file__).parent / "fixtures" / "n4-held-drift.jsonl.gz"


def bind_reference(monkeypatch: pytest.MonkeyPatch, records: Sequence[Record]) -> None:
    """Serve the author reference ``records`` recorded, whatever the installed models say: a journal golden then pins
    its sandbox and model rows, not TaskCompendium's wording."""
    recorded = {
        record.event.request.decision_model: record.event.value
        for record in records
        if isinstance(record.event, Resolved) and isinstance(record.event.request, AuthorReference)
    }

    async def served(event: AuthorReference) -> str:
        return recorded[event.decision_model]

    monkeypatch.setattr(wiring, "author_reference", served)


def rows(records: Sequence[Record]) -> list[Record]:
    """Records without their wall times: events, payers and the envelopes that link sandbox rows."""
    return [replace(record, wall_time=0.0) for record in records]


FIXTURE_CONFIG_DIGEST = "1797f796269d33067b1b26a1bbf934c262605ebaa992e42691d91cbe014f7384"
"""The policy digest the ``N4_JOURNAL`` and ``N4_POLLS`` fixtures recorded. A later policy entry changes only that
digest in an author-only journal."""


def rebased(recorded: Sequence[Record], fresh: Sequence[Record]) -> list[Record]:
    """``recorded`` with each stage completion's fixture policy digest replaced by the one ``fresh`` recorded."""
    current = next(r.event.config_digest for r in fresh if isinstance(r.event, StageCompleted))
    completed = [r.event for r in recorded if isinstance(r.event, StageCompleted)]
    assert completed and all(event.config_digest == FIXTURE_CONFIG_DIGEST for event in completed)
    return [
        replace(r, event=replace(r.event, config_digest=current)) if isinstance(r.event, StageCompleted) else r
        for r in recorded
    ]


async def test_a_journal_written_by_positional_dispatch_resumes_without_repeated_model_calls(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    """The fixture is the journal this dispatch published for the held first launch of ``held_after_stamp``. It pins
    the recorded events, the stage boundary, request references and replay keys; it serves the author reference it
    recorded, so TaskCompendium's wording does not enter it. A deliberate change to the author prompts, tool schemas,
    event identities or envelopes invalidates it; regenerate it then from that first launch."""
    recorded = decode(gzip.decompress(N4_JOURNAL.read_bytes()))
    bind_reference(monkeypatch, recorded)
    await held_after_stamp(monkeypatch, driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    fresh = history(destination)
    assert rows(fresh) == rows(rebased(recorded, fresh))
    served = [
        record.event.request
        for record in recorded
        if isinstance(record.event, Resolved) and isinstance(record.event.request, InferenceRequest)
    ]
    resumed = replace(resources(DriftingEnvs(2, PID), ()), inference=ServedOnce(drifting_script()[3:], served))
    log = EventLog(replay_from=recorded, replayable=replayable_grade)
    workpiece = await run_workpiece(proposal, config(), resumed, log, install=install)
    assert workpiece.terminal is Terminal.ACCEPTED
    assert isinstance(resumed.inference, ServedOnce) and len(served) == 3 and not resumed.inference.repeated
    assert len(resumed.inference.requests) == len(drifting_script()) - 3
    assert json.loads(tool_output(workpiece, "stamp"))["stdout"] == "pid 11\n"


N3_JOURNAL = Path(__file__).parent / "fixtures" / "n3-held-drift.jsonl.gz"


def plant(driver_args: Namespace, events: bytes, journal_format: int) -> Path:
    """Publish ``events`` as a held root under a head of ``journal_format``, as another writer would."""
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    (destination / "chunks").mkdir(parents=True)
    chunk = hashlib.sha256(events).hexdigest() + ".jsonl"
    (destination / "chunks" / chunk).write_bytes(events)
    workpiece = hashlib.sha256(b"{}").hexdigest() + ".json"
    head = Head(journal_format, "other", (chunk,), len(events), Status(terminal=Terminal.HELD), workpiece, 0.0)
    (destination / HEAD).write_bytes(HEAD_JSON.dump_json(head))
    return destination


# Regression: the format-2 reader claimed a positional journal under the same format number, ignoring its owners.
async def test_a_format_2_root_needs_migration_before_claim_and_stays_untouched(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``N3_JOURNAL`` is the journal the format-2 dispatch (e5ec5eba21) published for the held first launch of
    ``held_after_stamp``."""
    destination = plant(driver_args, gzip.decompress(N3_JOURNAL.read_bytes()), 2)
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    driver_args.resume_held = True
    with pytest.raises(LogFailure, match=f"has journal format 2, not {checkpoint.JOURNAL_FORMAT}") as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.NEEDS_MIGRATION
    assert "fresh output prefix" in driver.OUTCOMES[driver.NEEDS_MIGRATION][1]  # no migration ships
    assert tree(destination) == before


N6_MODIFY_HELD = Path(__file__).parent / "fixtures" / "n6-modify-held-root.json.gz"


def plant_root(directory: Path, fixture: Path) -> Path:
    """Write the published root ``fixture`` holds (relative path to text) at ``directory``."""
    for name, text in json.loads(gzip.decompress(fixture.read_bytes())).items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    return directory


N8_AUTHOR_HELD = Path(__file__).parent / "fixtures" / "n8-author-revision-held-root.json.gz"
N8_ACCEPTED_RESTORED = Path(__file__).parent / "fixtures" / "n8-accepted-restored-root.json.gz"


# Regression: the role runner resumed this format-3 head, accepted a fresh empty adversary report and published
# a round-0 Accept without the author repair, dropping the finding both verifier grades had confirmed and the Modify
# it caused.
# Regression: a format-4 root held in author revision re-held forever at its first positional write, and
# an accepted one lost its restoration provenance on read.
@pytest.mark.parametrize("resume_held", [True, False])
@pytest.mark.parametrize(
    "fixture,found",
    [(N6_MODIFY_HELD, 3), (N8_AUTHOR_HELD, 4), (N8_ACCEPTED_RESTORED, 4)],
    ids=["n6 modify held", "n8 author revision held", "n8 accepted restored"],
)
async def test_an_earlier_format_root_needs_migration_before_claim_and_stays_untouched(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, resume_held: bool, fixture: Path, found: int
) -> None:
    """Each fixture is a whole published root. The format-3 one (d510505b69, ``{"adversary.roster": ["adversary"]}``)
    holds a finding both verifier grades confirmed, Modify, and the author held in round 1 (73 rows); format 4
    changed what an adversary finding means. The format-4 ones (0fc30838db, this file's ``SETTINGS``, every first-attempt
    trial solved and every later one failed) are 58 rows held at the round-1 author's first model call and 96 rows
    accepted on the restored too-easy baseline; format 5 changed what a decision, a fallback and a solver slot
    mean."""
    destination = plant_root(Path(driver_args.output) / driver.proposal_key(driver_args.proposals), fixture)
    assert json.loads((destination / HEAD).read_text())["format"] == found
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)  # no model, sandbox or verifier is ever built
    driver_args.resume_held = resume_held
    with pytest.raises(LogFailure, match=f"has journal format {found}, not {checkpoint.JOURNAL_FORMAT}") as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.NEEDS_MIGRATION
    assert tree(destination) == before


FORMAT_10_HELD = Path(__file__).parent / "fixtures" / "format10-author-held-root.json.gz"
"""The whole root the format-10 writer (f854288c62) published for the held first launch of ``held_after_stamp``: three
author turns and their commands, without an author reference row."""


FORMAT_11_HELD = Path(__file__).parent / "fixtures" / "format11-author-held-root.json.gz"
"""The whole root the format-11 writer (2b214c6c20) published for the held first launch of ``held_after_stamp``: three
author turns and their commands, after its author reference row."""


FORMAT_12_HELD = Path(__file__).parent / "fixtures" / "format12-author-held-root.json.gz"
"""The whole root the format-12 writer published for the held first launch of ``held_after_stamp``."""


FORMAT_13_HELD = Path(__file__).parent / "fixtures" / "format13-author-held-root.json.gz"
"""The whole root the format-13 writer published for the held first launch of ``held_after_stamp``."""


FORMAT_14_HELD = Path(__file__).parent / "fixtures" / "format14-judge-grade-held-root.json.gz"
"""The format-14 writer (701237fcb9) graded an exact reference match at judge temperature 0.7, submitted the
draft, then deferred during validation. Format 15 would return an error at that same author grade."""


FORMAT_15_HELD = Path(__file__).parent / "fixtures" / "format15-author-held-root.json.gz"
"""The whole root the format-15 writer (187294b339) published for the held first launch of ``held_after_stamp``.
Format 16 resumes at stage boundaries and would run that author's commands live instead of rebuilding them."""


@pytest.mark.parametrize(
    "fixture,found",
    [
        (FORMAT_10_HELD, 10),
        (FORMAT_11_HELD, 11),
        (FORMAT_12_HELD, 12),
        (FORMAT_13_HELD, 13),
        (FORMAT_14_HELD, 14),
        (FORMAT_15_HELD, 15),
    ],
)
async def test_a_held_earlier_format_root_is_refused_before_claim_and_the_report_still_reads_it(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, fixture: Path, found: int
) -> None:
    """A format-10 root's author calls embedded a reference this writer does not reproduce, a format-11 root's author
    requests were worded otherwise, a format-12 root committed no reviewed visit, and a format-13 root's trials were
    recorded under another solver clock. A format-14 author could grade or submit a judge draft this writer refuses,
    and a format-15 root has no stage boundary to resume at. The driver refuses each root untouched and names the ways
    on."""
    destination = plant_root(Path(driver_args.output) / driver.proposal_key(driver_args.proposals), fixture)
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    driver_args.resume_held = True
    expected = f"format {found}, not {checkpoint.JOURNAL_FORMAT}: resume it with the writer of its format"
    with pytest.raises(UnsupportedFormat, match=expected) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.NEEDS_MIGRATION
    assert tree(destination) == before
    lines = await usage_report(str(destination))
    assert lines[0] == "workpieces=1 since_hours=all" and not [line for line in lines if "skipped" in line]


async def test_a_finished_format_15_root_is_skipped_untouched(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = plant_root(Path(driver_args.output) / driver.proposal_key(driver_args.proposals), FORMAT_15_HELD)
    head = json.loads((destination / HEAD).read_text())
    (destination / HEAD).write_text(json.dumps({**head, "status": {**head["status"], "terminal": "accepted"}}))
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    driver_args.resume_held = True
    assert await driver.one(driver_args) == 0
    assert tree(destination) == before


def test_an_optional_field_at_its_default_is_absent_from_every_persisted_form(proposal: TaskProposal) -> None:
    """Adding an optional persisted field changes no existing bytes: the author that recorded ``N8_AUTHOR_HELD``
    hydrated these exact bytes."""
    root = json.loads(gzip.decompress(N8_AUTHOR_HELD.read_bytes()))
    records = decode("".join(root[f"chunks/{name}"] for name in json.loads(root[HEAD])["chunks"]).encode())
    (written,) = {
        data
        for record in records
        if isinstance(record.event, WriteFiles)
        for path, data in record.event.files
        if path.name == "0.json" and path.parent.name == ".validation"
    }
    assert VALIDATION_RESULT.dump_json(VALIDATION_RESULT.validate_json(written)) == written
    decisions = (Accept(BandOutcome.IN_BAND), Modify(("harder",), BandOutcome.TOO_EASY), Incomplete(("held",)))
    snapshot = json.loads(WORKPIECE.dump_json(Workpiece("w", proposal, decisions=decisions)))
    keys = {key for decision in snapshot["decisions"] for key in decision}
    assert "fallback" not in snapshot and not {"restore", "fallback", "awaits"} & keys


# Regression: an empty restore and None serialised alike, so an invalid direct restore became an
# ordinary acceptance after a round trip; Modify.fallback collapsed the same way.
def test_every_omitted_field_round_trips_and_distinct_values_encode_distinctly(proposal: TaskProposal) -> None:
    stored = Fallback(Submission(arithmetic_task(), (), "notes"), BandOutcome.TOO_EASY, 0)
    values: dict[str, list[object]] = {
        "detail": [TrialResult(0, None, Cause.HELD, detail) for detail in ("", "refused", " ")],
        "restore": [Accept(BandOutcome.IN_BAND, (), restore) for restore in (None, "a" * 64, "b" * 64)],
        "fallback": [Modify(("x",), None, fallback) for fallback in (None, "a" * 64, "b" * 64)],
        "workpiece": [Workpiece("w", proposal, fallback=fallback) for fallback in (None, stored)],
    }
    for field, items in values.items():
        adapter = TypeAdapter(type(items[0]))
        encoded = [adapter.dump_json(item) for item in items]
        assert [adapter.validate_json(data) for data in encoded] == items, field
        assert len(set(encoded)) == len(items), field
    for empty in (partial(Accept, BandOutcome.IN_BAND, (), ""), partial(Modify, ("x",), None, "")):
        with pytest.raises(ValueError, match="An empty string is never a digest"):
            empty()
    with pytest.raises(ValidationError, match="An empty string is never a digest"):
        TypeAdapter(Accept).validate_json(b'{"band": "in_band", "restore": ""}')


def restoring_bundle(engine: Engine) -> Resources:
    """A too-easy first round, then a faulted resubmission whose review restores the first submission."""
    resubmit = (*script()[:3], call("submit", {"notes": "revised notes"}, "submit"))
    bundle = resources(Envs(), (*script(), *resubmit))
    return replace(bundle, rollouts=replace(bundle.rollouts, engine=engine))


# Regression: after the coordinator returned the accepted fallback, an OSError on runtime exit
# published HELD with the faulted revision's notes and [Modify, Incomplete], though the journal recorded the Accept.
async def test_a_cleanup_failure_after_acceptance_appends_to_the_returned_workpiece(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def failing_exit(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), restoring_bundle(FirstRoundSolves())
        raise OSError(errno.EIO, "runtime cleanup failed")

    monkeypatch.setattr(driver, "runtime", failing_exit)
    with pytest.raises(OSError, match="runtime cleanup failed"):
        await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    workpiece = published_workpiece(destination)
    assert [type(decision) for decision in workpiece.decisions] == [Modify, Accept, Incomplete]
    assert workpiece.submission is not None and workpiece.submission.notes == "done"
    assert any(isinstance(r.event, StageCompleted) and r.event.exit == "Decided" for r in history(destination))


class RevisionWaits(FirstRoundSolves):
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if run_id is not None and not run_id.endswith(":0"):
            self.entered.set()
            await asyncio.Event().wait()  # until cancelled
        return await super().run_task(spec, model, run_id=run_id)


async def test_a_cancelled_run_keeps_its_last_stage_state_under_a_running_head(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = RevisionWaits()
    serve(monkeypatch, restoring_bundle(engine))
    run = asyncio.create_task(driver.one(driver_args))
    await engine.entered.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    journal = published(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    workpiece = journal.workpiece()
    assert journal.head.status.terminal is None and workpiece is not None  # a running head, resumed by replay
    assert workpiece.round == 1 and workpiece.submission is not None and workpiece.submission.notes == "revised notes"


# Regression: a failure persisting the final review held the run from the initial workpiece; the journal had
# Submitted, Validated and Decided, but the snapshot had no task or evidence.
async def test_a_recovered_storage_failure_publishes_the_last_workpiece_it_reached(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    write, failed = BufferSink.write, []

    async def fail_once(sink: BufferSink, records: Sequence[Record]) -> None:
        reviewed = any(isinstance(r.event, StageCompleted) and r.event.stage is Stage.REVIEW for r in records)
        if reviewed and not failed:
            failed.append(LogFailure("Cannot encode the event stream"))
            raise failed[-1]
        await write(sink, records)

    monkeypatch.setattr(BufferSink, "write", fail_once)
    serve(monkeypatch, resources(Envs(), script()))
    await driver.one(driver_args)
    workpiece = published_workpiece(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    assert failed and workpiece.terminal is Terminal.HELD
    assert workpiece.submission is not None and len(workpiece.evidence) == 1


# Regression: a refused foreign restore digest published a bug-class hold whose snapshot had neither the
# submission nor the evidence the refusal was about, because the driver still held its initial workpiece.
async def test_a_refused_restore_publishes_the_last_workpiece_it_reached(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def restore_elsewhere(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
        return Accept(BandOutcome.TOO_EASY, (), "elsewhere")

    @asynccontextmanager
    async def foreign(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield replace(build_config(policy, {}), review=restore_elsewhere), resources(Envs(), script())

    monkeypatch.setattr(driver, "runtime", foreign)
    with pytest.raises(ValueError, match="Cannot restore elsewhere"):
        await driver.one(driver_args)
    workpiece = published_workpiece(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    assert workpiece.terminal is Terminal.HELD and workpiece.submission is not None and len(workpiece.evidence) == 1
    assert isinstance(held := workpiece.decisions[-1], Incomplete) and held.reasons[0].startswith("bug: ")


STATUS = PurePosixPath("/workspace/status.txt")
TASK = PurePosixPath("/workspace/task.json")


class ProducerEnv(RecordingEnv):
    """A guest process rewrites ``path`` before each read of it: read ``n`` finds ``contents[n]`` there, or no file
    for None, and the last content stays. ``physical`` holds what each read of ``path`` found (None: missing)."""

    def __init__(
        self, commands: Mapping[str, FakeCommand], path: PurePosixPath, contents: Sequence[bytes | None]
    ) -> None:
        super().__init__(commands, fail=False)
        self.path, self.contents = path, contents
        self.physical: list[bytes | None] = []

    @override
    async def read(self, path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes:
        if path == self.path and len(self.physical) < len(self.contents):
            if (content := self.contents[len(self.physical)]) is None:
                self.files.pop(path, None)
            else:
                self.files[path] = content
        try:
            data = await super().read(path, offset=offset, limit=limit)
        except FileNotFoundError:
            if path == self.path:
                self.physical.append(None)
            raise
        if path == self.path:
            self.physical.append(data)
        return data


class ProducerEnvs(Envs):
    def __init__(self, path: PurePosixPath, contents: Sequence[bytes | None], draft: bytes | None = None) -> None:
        super().__init__()
        self.path, self.contents = path, contents
        self.producers: list[ProducerEnv] = []
        if draft is not None:  # this machine's draft command writes other task bytes
            writes = {**self.commands["write-draft"].writes, TASK: draft}
            self.commands = {**self.commands, "write-draft": replace(self.commands["write-draft"], writes=writes)}

    @override
    async def create(self, requirements: EnvironmentRequirements) -> ProducerEnv:
        env = ProducerEnv(self.commands, self.path, self.contents)
        self.created.append(env)
        self.producers.append(env)
        return env


def polling_script(polls: int = 2) -> tuple[Completion, ...]:
    """Read a status file ``polls`` times, then continue the ordinary script."""
    poll = {"path": str(STATUS)}
    return (*(call("read_file", poll, f"poll-{n}") for n in range(1, polls + 1)), *script()[1:])


def last_message(request: InferenceRequest) -> str:
    return request.messages[-1].content


# Regression: on an empty journal the second of two identical reads was served the first read's bytes.
async def test_a_first_launch_returns_each_physical_read_of_a_file_that_changes(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs = ProducerEnvs(STATUS, (b"pending", b"done"))
    first = resources(envs, polling_script())
    serve(monkeypatch, first)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    workpiece = published_workpiece(destination)
    assert envs.producers[0].physical == [b"pending", b"done"]
    assert [tool_output(workpiece, poll) for poll in ("poll-1", "poll-2")] == ["pending", "done"]
    assert isinstance(first.inference, FakeInference)
    assert [last_message(request) for request in first.inference.requests[1:3]] == ["pending", "done"]
    assert not [r for r in history(destination) if isinstance(r.event, Note)]
    assert workpiece.terminal is Terminal.ACCEPTED and workpiece.submission is not None
    assert executions(first) == (len(polling_script()), 2)


N4_POLLS = Path(__file__).parent / "fixtures" / "n4-held-polls.jsonl.gz"


async def held_after_polls(
    monkeypatch: pytest.MonkeyPatch, driver_args: Namespace, contents: Sequence[bytes | None]
) -> list[InferenceRequest]:
    """Hold the polling author on the call after its last poll; return the requests it was served."""
    held = FakeInference((*polling_script(len(contents))[: len(contents)], Hold("relay unavailable")))
    serve(monkeypatch, replace(resources(ProducerEnvs(STATUS, contents), ()), inference=held))
    await driver.one(driver_args)
    driver_args.resume_held = True
    return held.requests[:-1]


# Regression: a journal recording pending then done served done to both polls and paid for two turns again.
async def test_a_journal_with_two_polls_serves_each_poll_its_own_recording(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture is the journal this dispatch published for this test's held first launch: two identical reads of a
    file a guest process changed, then a hold on the third model call, serving the author reference it recorded.
    Regenerate it from that launch when a deliberate change to the author prompts, tool schemas, event identities or
    envelopes invalidates it."""
    recorded = decode(gzip.decompress(N4_POLLS.read_bytes()))
    bind_reference(monkeypatch, recorded)
    served = await held_after_polls(monkeypatch, driver_args, (b"pending", b"done"))
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    fresh = history(destination)
    assert rows(fresh) == rows(rebased(recorded, fresh))
    envs = ProducerEnvs(STATUS, (b"pending", b"done"))
    resumed = replace(resources(envs, ()), inference=ServedOnce(polling_script()[2:], served))
    serve(monkeypatch, resumed)
    await driver.one(driver_args)
    workpiece = published_workpiece(destination)
    assert envs.producers[0].physical == [b"pending", b"done"]
    assert [tool_output(workpiece, poll) for poll in ("poll-1", "poll-2")] == ["pending", "done"]
    assert isinstance(resumed.inference, ServedOnce) and not resumed.inference.repeated
    assert len(resumed.inference.requests) == len(polling_script()) - 2
    assert last_message(resumed.inference.requests[0]) == "done"  # the first new turn read the second poll
    assert workpiece.terminal is Terminal.ACCEPTED and workpiece.submission is not None
    assert not [r for r in history(destination) if isinstance(r.event, Note) and "diverged" in r.event.msg]


MISSING = f"File operation failed: [Errno {errno.ENOENT}] No such file: '{STATUS}'"


# Regression: a failed read had no recorded outcome, so a later success on the same path took its place. A resume
# meets what the fresh machine holds: the same failure replays the recorded turns, a present file buys the turns after.
@pytest.mark.parametrize(
    ("recorded", "resumed", "served"),
    [
        ((None, b"B"), (None, b"B"), [MISSING, "B"]),
        ((None, b"B"), (b"B", b"B"), ["B", "B"]),
        ((None, b"B", b"C"), (None, b"B", b"C"), [MISSING, "B", "C"]),
    ],
    ids=["missing again", "present where it was missing", "missing, then two contents"],
)
async def test_a_handled_read_failure_is_journalled_and_a_resume_meets_the_live_outcome(
    driver_args: Namespace,
    monkeypatch: pytest.MonkeyPatch,
    recorded: tuple[bytes | None, ...],
    resumed: tuple[bytes | None, ...],
    served: list[str],
) -> None:
    paid = await held_after_polls(monkeypatch, driver_args, recorded)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert sum(isinstance(r.event, Failed) for r in history(destination)) == 1
    envs = ProducerEnvs(STATUS, resumed)
    reused = len(recorded) if resumed == recorded else 1  # a changed first poll misses from the turn that reads it
    resources_ = replace(resources(envs, ()), inference=ServedOnce(polling_script(len(recorded))[reused:], paid))
    serve(monkeypatch, resources_)
    await driver.one(driver_args)
    workpiece = published_workpiece(destination)
    assert isinstance(resources_.inference, ServedOnce) and not resources_.inference.repeated
    polls = [f"poll-{n}" for n in range(1, len(recorded) + 1)]
    assert envs.producers[0].physical == list(resumed)
    assert [tool_output(workpiece, poll) for poll in polls] == served
    assert last_message(resources_.inference.requests[0]) == served[reused - 1]
    assert len(resources_.inference.requests) == len(polling_script(len(recorded))) - reused
    assert workpiece.terminal is Terminal.ACCEPTED
    assert not [r for r in history(destination) if isinstance(r.event, Note) and "diverged" in r.event.msg]


class LosesWriteDraft(DriftingEnvs):
    """The first machine loses the reply to ``write-draft``, as Shellbox's ``CommandOutcomeUnknown`` does."""

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = await super().create(requirements)
        if len(self.created) == 1:
            env.commands = {**env.commands, "write-draft": Restart("command outcome unknown: reply lost")}
        return env


# Regression: a stage restarted in-process rebuilt from the launch's history only, so the two turns completed before
# the restart were paid for again (7 calls). The restart runs its commands on a fresh machine that prints the same.
async def test_an_in_process_stage_restart_replays_the_turns_its_first_attempt_completed(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs = LosesWriteDraft(1, PID)
    first = resources(envs, drifting_script())
    serve(monkeypatch, first)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    workpiece = published_workpiece(destination)
    assert workpiece.terminal is Terminal.ACCEPTED
    assert executions(first) == (len(drifting_script()), 2)
    assert [env.executed for env in envs.created] == [["stamp", "write-draft"], ["stamp", "write-draft"]]
    assert json.loads(tool_output(workpiece, "stamp"))["stdout"] == "pid 11\n"
    assert tool_output(workpiece, "read") == "ok"


ROLE = "cat /fixture/role"


class RoleEnvs(Envs):
    """Adversary machine ``k``, created in roster order after ``before`` others, prints ``outputs[k]`` for ``ROLE``.

    Its command waits for the machine before it in ``order`` to finish, or with ``overlap``, for its sibling to start.
    """

    def __init__(self, outputs: Sequence[str], order: Sequence[int], overlap: bool, before: int = 1) -> None:
        super().__init__()
        self.outputs, self.order, self.before = outputs, order, before
        self.finished = [asyncio.Event() for _ in outputs]
        self.barrier = asyncio.Barrier(len(outputs)) if overlap else None

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = await super().create(requirements)
        if (index := len(self.created) - 1 - self.before) < 0:
            return env  # the author's machine
        result = FakeCommand(CommandResult(0, self.outputs[index], "", 0, False), {})
        env.commands = {**env.commands, ROLE: result}
        envs, run = self, env.run

        async def ordered(cmd: str, *, timeout: float) -> CommandResult:
            if cmd == ROLE and envs.barrier is not None:
                await envs.barrier.wait()
            elif cmd == ROLE and (turn := envs.order.index(index)):
                await envs.finished[envs.order[turn - 1]].wait()
            try:
                return await run(cmd, timeout=timeout)
            finally:
                envs.finished[index].set()

        env.run = ordered  # type: ignore[method-assign]
        return env


BRIEFED = {"false_accept": "hunt false accepts", "false_reject": "hunt false rejects"}
"""Each default brief by a phrase its first user message carries."""


class Adversaries(InferenceClient):
    """Each adversary runs ``ROLE``, then reports no findings, or with ``hold`` fails on that turn instead."""

    def __init__(self, hold: bool, served: Sequence[InferenceRequest] = ()) -> None:
        self.hold, self.served = hold, {event_digest(request) for request in served}
        self.requests: list[InferenceRequest] = []

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", "adversary", ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if event_digest(request) in self.served:
            raise AssertionError("an already-resolved adversary turn reached inference")
        self.requests.append(request)
        if not any(message.role == Role.ASSISTANT for message in request.conversation):
            return call("shell", {"command": ROLE, "timeout": 5}, "role")
        if self.hold:
            raise Hold("relay unavailable")
        return call("report", {"findings": []}, "report")

    def outputs(self) -> list[tuple[str, str]]:
        """(adversary name, the role output its report turn saw), for each report turn."""
        return sorted(
            (
                next(name for name in BRIEFED if BRIEFED[name] in request.conversation[1].content),
                json.loads(request.conversation[-1].content)["stdout"],
            )
            for request in self.requests
            if any(message.role == Role.ASSISTANT for message in request.conversation)
        )


# Regression: siblings whose readiness reversed on resume swapped outputs, and overlapping identical results held.
# Each resumed search now runs on its own fresh machine and reports on what that machine printed.
@pytest.mark.parametrize(
    ("roster", "recorded", "fresh", "overlap"),
    [
        (("false_accept", "false_reject"), ("accept 1\n", "reject 1\n"), ("accept 2\n", "reject 2\n"), False),
        (("false_accept", "false_accept"), ("first 1\n", "second 1\n"), ("first 2\n", "second 2\n"), False),
        (("false_accept", "false_reject"), ("/workspace\n", "/workspace\n"), ("/workspace\n", "/workspace\n"), True),
    ],
    ids=["reversed readiness", "duplicate roster names", "identical results overlapping"],
)
async def test_each_adversary_reruns_on_its_own_fresh_machine_whatever_its_siblings_do(
    driver_args: Namespace,
    monkeypatch: pytest.MonkeyPatch,
    roster: tuple[str, str],
    recorded: tuple[str, str],
    fresh: tuple[str, str],
    overlap: bool,
) -> None:
    overlay = json.loads(Path(driver_args.policy).read_text())
    Path(driver_args.policy).write_text(json.dumps({**overlay, "adversary.roster": list(roster)}))
    held = Adversaries(hold=True)
    first = replace(
        resources(RoleEnvs(recorded, (0, 1), overlap), script()),
        role_inference={"solve_review": CleanReviewer(), "adversary": held},
    )
    serve(monkeypatch, first)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    held_wp = published_workpiece(destination)  # both searches failed on their report turn, so review held
    assert held_wp.terminal is Terminal.HELD and isinstance(held_wp.decisions[-1], Incomplete)
    assert [result.status for result in held_wp.evidence[-1].adversary] == [AdversaryStatus.FAILED] * 2
    # Same-named searches send one identical first turn, which the cache resolves once for both.
    assert len(held.requests) == (3 if roster[0] == roster[1] else 4)
    resolved = [r.event.request for r in history(destination) if isinstance(r.event, Resolved)]
    adversary = Adversaries(
        hold=False, served=[r for r in resolved if isinstance(r, InferenceRequest) and r in held.requests]
    )
    envs = RoleEnvs(fresh, (1, 0), overlap, before=0)
    resumed = replace(
        resources(envs, ()),
        inference=ServedOnce((), ()),
        role_inference={"solve_review": CleanReviewer(), "adversary": adversary},
    )
    serve(monkeypatch, resumed)
    driver_args.resume_held = True
    await driver.one(driver_args)
    wp = published_workpiece(destination)  # the explicit resume reopened validation, not the review that held
    assert wp.terminal is Terminal.ACCEPTED and not any(isinstance(d, Incomplete) for d in wp.decisions)
    assert executions(resumed) == (0, 0) and len(adversary.requests) == 2  # only the two report turns are new
    assert [env.executed for env in envs.created] == [[ROLE], [ROLE]]  # no author machine: its boundary holds
    names = [roster[0], roster[1]]
    assert held.outputs() == sorted(zip(names, recorded, strict=True))
    assert adversary.outputs() == sorted(zip(names, fresh, strict=True))


MALFORMED = b'{"malformed"'


# Regression: the first submit after a resume was served the bytes an earlier grade had read, and accepted.
@pytest.mark.parametrize("rebuilt", ["replaced after the grade", "malformed when rebuilt"])
async def test_a_resumed_submit_never_accepts_bytes_the_sandbox_does_not_hold(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, rebuilt: str
) -> None:
    held = FakeInference((*script()[:3], Hold("relay unavailable")))
    serve(monkeypatch, replace(resources(Envs(), ()), inference=held))
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    valid = arithmetic_task().model_dump_json().encode()
    if rebuilt == "replaced after the grade":  # a guest process rewrites the draft between grade and submit
        envs = ProducerEnvs(TASK, (valid, MALFORMED))
    else:  # the fresh machine's draft command produced other bytes
        envs = ProducerEnvs(TASK, (), draft=MALFORMED)
    resumed = replace(resources(envs, ()), inference=ServedOnce((script()[3], Hold("stop")), held.requests[:-1]))
    serve(monkeypatch, resumed)
    driver_args.resume_held = True
    await driver.one(driver_args)
    workpiece = published_workpiece(destination)
    assert workpiece.terminal is Terminal.HELD and workpiece.submission is None
    assert isinstance(resumed.inference, ServedOnce) and not resumed.inference.repeated
    assert executions(resumed)[1] == 0
    # Grade and submit each read the fresh machine; submit refuses what it read and the model hears why.
    graded = valid if rebuilt == "replaced after the grade" else MALFORMED
    assert envs.producers[0].physical == [graded, MALFORMED]
    assert len(resumed.inference.requests) == 2
    assert last_message(resumed.inference.requests[1]).startswith("Invalid draft")


class ExhaustedRole(Envs):
    """Every machine prints ``alpha`` for ``ROLE``; its first ``exhausted`` runs exceed the output budget instead."""

    def __init__(self, exhausted: int) -> None:
        super().__init__()
        self.exhausted = exhausted

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = await super().create(requirements)
        env.commands = {**env.commands, ROLE: FakeCommand(CommandResult(0, "alpha\n", "", 0, False), {})}
        envs, run = self, env.run

        async def exhausting(cmd: str, *, timeout: float) -> CommandResult:
            if cmd == ROLE and envs.exhausted:
                envs.exhausted -= 1
                env.executed.append(cmd)
                raise OutputBudgetExhausted("output budget exhausted")
            return await run(cmd, timeout=timeout)

        env.run = exhausting  # type: ignore[method-assign]
        return env


# Regression: a resumed adversary whose recorded command exhausted the output budget consumed its position
# unverified, so the search counted as EXHAUSTED evidence and the workpiece was accepted with no new model call.
async def test_a_resumed_command_that_exhausts_its_budget_once_reruns_the_search_on_a_fresh_machine(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = json.loads(Path(driver_args.policy).read_text())
    Path(driver_args.policy).write_text(json.dumps({**overlay, "adversary.roster": ["false_accept"]}))
    held = Adversaries(hold=True)
    serve(
        monkeypatch,
        replace(
            resources(ExhaustedRole(0), script()), role_inference={"solve_review": CleanReviewer(), "adversary": held}
        ),
    )
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published_workpiece(destination).terminal is Terminal.HELD and len(held.requests) == 2
    adversary = Adversaries(hold=False, served=held.requests[:1])
    envs = ExhaustedRole(1)
    resumed = replace(
        resources(envs, ()),
        inference=ServedOnce((), ()),
        role_inference={"solve_review": CleanReviewer(), "adversary": adversary},
    )
    serve(monkeypatch, resumed)
    driver_args.resume_held = True
    await driver.one(driver_args)
    workpiece = published_workpiece(destination)
    assert executions(resumed) == (0, 0) and envs.exhausted == 0
    statuses = [result.status for evidence in workpiece.evidence for result in evidence.adversary]
    assert workpiece.terminal is Terminal.ACCEPTED and statuses[-1] is AdversaryStatus.REPORTED
    assert len(adversary.requests) == 1  # only the report turn is new: the restored task renders the same opening


class CorruptArchiveOnce(DriftingEnvs):
    """The first machine's backend hands back a corrupt archive for the read of ``STAMP``."""

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = await super().create(requirements)
        first, read = len(self.created) == 1, env.read

        async def corrupt(path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes:
            if first and path == STAMP:
                raise tarfile.ReadError("not a gzip file")
            return await read(path, offset=offset, limit=limit)

        env.read = corrupt  # type: ignore[method-assign]
        return env


# Regression: a corrupt transfer archive became "File operation failed" feedback without a recorded outcome,
# so the author's next tool call on that machine held on "no attempt ... ready for a request".
@pytest.mark.parametrize("held", [False, True], ids=["same launch", "resumed through the failure"])
async def test_a_corrupt_transfer_archive_restarts_the_stage_instead_of_becoming_feedback(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, held: bool
) -> None:
    envs = CorruptArchiveOnce(1, PID)
    inference = FakeInference((*drifting_script()[:3], Hold("relay unavailable")) if held else drifting_script())
    first = replace(resources(envs, ()), inference=inference)
    serve(monkeypatch, first)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert [env.executed for env in envs.created[:2]] == [["stamp"], ["stamp", "write-draft"]]
    assert published_workpiece(destination).terminal is (Terminal.HELD if held else Terminal.ACCEPTED)
    assert len(inference.requests) == (4 if held else len(drifting_script()))  # no turn was paid for twice
    if held:
        served = ServedOnce(drifting_script()[3:], inference.requests[:-1])
        resumed = replace(resources(DriftingEnvs(2, PID), ()), inference=served)
        serve(monkeypatch, resumed)
        driver_args.resume_held = True
        await driver.one(driver_args)
        assert isinstance(resumed.inference, ServedOnce) and not resumed.inference.repeated
        assert executions(resumed) == (len(drifting_script()) - 3, 2)
    workpiece = published_workpiece(destination)
    assert workpiece.terminal is Terminal.ACCEPTED and tool_output(workpiece, "read") == "ok"
    assert workpiece.session is not None
    assert not any("File operation failed" in message.content for message in workpiece.session.messages)


SANDBOX = (Fingerprint("env", "definition", ReplayPolicy.REEXECUTE),)
READ_STATUS = ReadFile(STATUS)


def lines(records: Sequence[Record], *changes: tuple[int, str, object]) -> bytes:
    """``records`` as a writer encodes them, with each (row, field, value) change applied to the encoded row."""
    sink, rows = BufferSink(), []
    for record in records:
        rows.append(json.loads(sink._line(record)))
    for row, name, value in changes:
        rows[row - 1][name] = value
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


@pytest.mark.parametrize(
    "events",
    [
        lines((Record(1, 0.0, READ_STATUS),), (1, "sequence", True)),
        lines(
            (
                Record(1, 0.0, READ_STATUS),
                Record(2, 0.0, Resolved(READ_STATUS, SANDBOX, b"A"), answers=1, billed=Billing.AUTHOR),
            )
        ),
    ],
    ids=["true as row 1", "billed sandbox outcome"],
)
async def test_claim_refuses_a_malformed_row_and_leaves_the_head(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, events: bytes
) -> None:
    destination = plant(driver_args, events, checkpoint.JOURNAL_FORMAT)
    before = tree(destination)
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    driver_args.resume_held = True
    with pytest.raises(CorruptJournal) as raised:
        await driver.one(driver_args)
    assert driver.exit_code(raised.value) == driver.CORRUPT
    assert tree(destination) == before


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
        return ShellboxEnv(HelperlessMachine(), "image", AsyncExitStack())


# Regression: a missing guest `wc` was stored as a bug hold and failed the child process.
async def test_broken_guest_helper_holds_with_its_cause_without_a_bug(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), resources(HelperlessEnvs(), script())

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = published_workpiece(destination)
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

    def __init__(
        self, root: Path, faults: Sequence[Injection], prelude: str = "", cancel_error: Exception | None = None
    ) -> None:
        self.root, self.faults, self.prelude, self.cancel_error = root, faults, prelude, cancel_error
        self.created: list[Path] = []

    def cancel(self) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error

    @override
    async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
        workdir = self.root / f"sandbox-{len(self.created)}"
        workdir.mkdir(parents=True)
        self.created.append(workdir)
        machine = local_iris_machine(workdir, SandboxController(workdir, self.faults, self.prelude), self.cancel)
        return ShellboxEnv(machine, "image", AsyncExitStack())


# Regression: a lost WebSocket upgrade reply reran the command in the same sandbox, and a
# missing `wc` or an OOM-killed sandbox during capture held only after three sandboxes with one append each.
# Regression: after a lone `timer 137` was classified as OOM or unknown, a failed
# TerminateJob (503 or 403) replaced that outcome with the bare transport error, and the run left through the bug path.
TIMER_KILL_CANCEL_FAILS = [
    (
        (("command", Fault.TIMER_KILLED, True), ("status", Fault.OOM_KILLED, False)),
        1,
        "OOMKilled",
        ConnectError(code, "TerminateJob failed"),
    )
    for code in (Code.UNAVAILABLE, Code.PERMISSION_DENIED)
] + [
    (
        (("command", Fault.TIMER_KILLED, True),),
        STAGE_REPLAY_ATTEMPTS,
        "timer 137",
        ConnectError(code, "TerminateJob failed"),
    )
    for code in (Code.UNAVAILABLE, Code.PERMISSION_DENIED)
]


@pytest.mark.parametrize(
    "faults,prelude,sandboxes,cause,cancel_error",
    [
        ((("command", Fault.SETUP_FAILED, True),), "", STAGE_REPLAY_ATTEMPTS, "command outcome unknown", None),
        ((), "wc() { echo 'wc: not found' >&2; return 127; }; ", 1, "wc: not found", None),
        ((("stdout-size", Fault.OOM_KILLED, True),), "", 1, "OOMKilled", None),
        *((faults, "", sandboxes, cause, error) for faults, sandboxes, cause, error in TIMER_KILL_CANCEL_FAILS),
    ],
    ids=[
        "lost-upgrade-reply",
        "capture-without-wc",
        "capture-oom-killed",
        "timer-kill-oom-cancel-503",
        "timer-kill-oom-cancel-403",
        "timer-kill-unknown-cancel-503",
        "timer-kill-unknown-cancel-403",
    ],
)
async def test_iris_command_runs_once_per_sandbox_and_holds_with_its_cause(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faults: tuple[Injection, ...],
    prelude: str,
    sandboxes: int,
    cause: str,
    cancel_error: Exception | None,
) -> None:
    envs = LocalIrisEnvs(tmp_path / "sandboxes", faults, prelude, cancel_error)
    counted = (call("shell", {"command": "echo ran >> counter", "timeout": 5}, "inspect"), *script()[1:])

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), resources(envs, counted)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = published_workpiece(destination)
    assert result.terminal is Terminal.HELD
    assert cause in str(result.decisions[-1]) and "bug:" not in str(result.decisions[-1])
    assert [(workdir / "counter").read_text() for workdir in envs.created] == ["ran\n"] * sandboxes


# Regression: a lost upgrade reply for an upload chunk retried the write in the same sandbox; the
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
        *script()[2:],
    )

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), resources(envs, responses)

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    result = published_workpiece(destination)
    assert result.terminal is Terminal.ACCEPTED and result.session is not None
    assert [message.content for message in result.session.messages if message.tool_call_id == "read-proof"] == ["proof"]
    assert len(envs.created) == 2 and (envs.created[1] / "proof.txt").read_bytes() == b"proof"
    # The faulted request stays unanswered, so no Failed outcome replays; only its replacement resolves.
    records = history(destination)
    writes = [r for r in records if isinstance(r.event, WriteFile) and r.event.path.name == "proof.txt"]
    assert [[type(r.event) for r in records if r.answers == write.sequence] for write in writes] == [[], [Resolved]]


class ScratchController(SandboxController):
    """A sandbox whose guest ``/tmp`` hydration archive lives under ``scratch`` instead."""

    def __init__(self, root: Path, scratch: Path) -> None:
        super().__init__(root, (), "")
        self.scratch = scratch

    @override
    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        relocated = str(self.scratch / "forge-hydration-")
        command = [part.replace("/tmp/forge-hydration-", relocated) for part in request.command]  # type: ignore[attr-defined]
        return super().exec_in_container(SimpleNamespace(command=command), timeout_ms)


class HydrationEnvs(LocalIrisEnvs):
    """Local Iris sandboxes whose hydration scratch is unwritable when ``locked``; ``blocked`` caller paths are made
    directories, or, in a read-only parent, files the guest may not replace."""

    def __init__(self, root: Path, *, locked: bool = False, blocked: Mapping[str, int] | None = None) -> None:
        super().__init__(root, ())
        self.locked, self.blocked = locked, dict(blocked or {})

    @override
    async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
        workdir = self.root / f"sandbox-{len(self.created)}"
        scratch = self.root / f"scratch-{len(self.created)}"
        workdir.mkdir(parents=True)
        scratch.mkdir(0o500 if self.locked else 0o700)
        for name, code in self.blocked.items():
            if code == errno.EISDIR:
                (workdir / name).mkdir(parents=True)
            else:
                (workdir / name).parent.mkdir(parents=True, mode=0o500)
        self.created.append(workdir)
        return ShellboxEnv(local_iris_machine(workdir, ScratchController(workdir, scratch)), "image", AsyncExitStack())


REFERENCE = WorkspaceHydration(
    tuple(inline_resource(f"workspace/reference/{n}.txt", f"ref {n}".encode()) for n in range(3))
)


def drafting_script() -> tuple[Completion, ...]:
    drafts = Envs().commands["write-draft"].writes.items()
    return (*(call("write_file", {"path": str(p), "content": d.decode()}, p.name) for p, d in drafts), *script()[2:])


# Regression: with guest /tmp unwritable, four-file hydration journalled EACCES for
# Taskforge's own archive as the caller's failure, and every resume on a healthy sandbox held as unreproducible.
async def test_unstageable_hydration_archive_holds_as_infrastructure_and_resumes(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked = HydrationEnvs(tmp_path / "locked", locked=True)
    first = resources(locked, drafting_script())
    serve(monkeypatch, first, workspace=REFERENCE)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    held = published_workpiece(destination)
    assert held.terminal is Terminal.HELD and "Permission denied" in str(held.decisions[-1])
    assert "bug:" not in str(held.decisions[-1]) and executions(first) == (0, 0)
    assert not [r for r in history(destination) if isinstance(r.event, Failed)]
    driver_args.resume_held = True
    healthy = HydrationEnvs(tmp_path / "healthy")
    resumed = resources(healthy, drafting_script())
    serve(monkeypatch, resumed, workspace=REFERENCE)
    await driver.one(driver_args)
    workpiece = published_workpiece(destination)
    assert workpiece.terminal is Terminal.ACCEPTED, workpiece.decisions[-1]
    assert executions(resumed)[0] == len(drafting_script())
    assert (healthy.created[0] / "reference" / "2.txt").read_bytes() == b"ref 2"


# Control: a four-file hydration that fails on the caller's own destination stays the caller's defect, a bug hold with
# the guest's error text and no journalled outcome, and a resume on the same sandbox shape fails the same way.
@pytest.mark.parametrize("code", [errno.EISDIR, errno.EACCES], ids=errno.errorcode.get)
async def test_hydration_failure_on_a_caller_path_stays_the_callers_and_repeats(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    blocked = {"reference/1.txt" if code == errno.EISDIR else "locked/1.txt": code}
    hydration = WorkspaceHydration((*REFERENCE.static[:2], inline_resource("workspace/locked/1.txt", b"ref 1")))
    for attempt in ("first", "resume"):
        envs = HydrationEnvs(tmp_path / attempt, blocked=blocked)
        serve(monkeypatch, attempt_resources := resources(envs, drafting_script()), workspace=hydration)
        with pytest.raises(RuntimeError, match=f"Could not hydrate files: .*{os.strerror(code)}"):
            await driver.one(driver_args)
        destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
        workpiece = published_workpiece(destination)
        assert workpiece.terminal is Terminal.HELD and "bug:" in str(workpiece.decisions[-1])
        assert "Unreproducible" not in str(workpiece.decisions[-1]) and executions(attempt_resources) == (0, 0)
        assert not [r for r in history(destination) if isinstance(r.event, Failed)]
        driver_args.resume_held = True


OUTAGE = DecisionModelUnavailable("clef@fixture-1 answered no read for 900s; last HTTP 503: clef_unavailable")


# A decision-model outage is no case for a person: the published hold names what it awaits, and the next launch
# resumes it without --resume-held, paying only what the held launch had not bought. Any other hold stays put.
@pytest.mark.parametrize("hold, resumes", [(OUTAGE, True), (Hold("relay unavailable"), False)], ids=["outage", "hold"])
async def test_a_decision_model_outage_defers_and_the_next_launch_resumes_it_unasked(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, hold: Hold, resumes: bool
) -> None:
    first = replace(resources(Envs(), ()), inference=FakeInference((*script()[:2], hold)))
    resumed = replace(resources(Envs(), ()), inference=ServedOnce(script()[2:], ()))
    serve(monkeypatch, first, resumed)
    assert await driver.one(driver_args) == (driver.DEFERRED if resumes else 0)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    held = published_workpiece(destination)
    assert held.terminal is Terminal.HELD and isinstance(last := held.decisions[-1], Incomplete)
    assert last.awaits == ("decision_model" if resumes else None)
    assert isinstance(first.inference, FakeInference) and isinstance(resumed.inference, ServedOnce)
    resumed.inference.served = {event_digest(request) for request in first.inference.requests[:-1]}
    assert await driver.one(driver_args) == 0  # driver_args.resume_held stays False
    if not resumes:
        assert published_workpiece(destination) == held and not resumed.inference.requests
        return
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert not resumed.inference.repeated and len(resumed.inference.requests) == len(script()) - 2


# Regression: an ordinary launch entered a human hold and replayed its refusals as if an
# operator had passed --resume-held. Only the exact typed marker that an outage-only hold records is entered again,
# and the refusal replay rule always sees the operator's own flag.
@pytest.mark.parametrize(
    "error, enters",
    [
        (OUTAGE, True),
        (Hold("waits for the decision model: an operator rejected clef@fixture-1's levels"), False),
        (BaseExceptionGroup("validation", [OUTAGE, Hold("request rejected")]), False),
    ],
    ids=["outage", "human hold naming the decision model", "outage beside another hold"],
)
async def test_only_the_typed_outage_marker_reopens_a_hold_without_the_operators_flag(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, error: BaseException, enters: bool
) -> None:
    async def held(*_: object) -> object:
        raise error

    entered: list[bool] = []

    def rule(previous: Sequence[Record], resume: bool) -> Any:
        entered.append(resume)
        return lambda _: False

    monkeypatch.setattr(coordinator, "STAGES", {**STAGES, Stage.AUTHOR: held})
    monkeypatch.setattr(driver, "replaced_refusals", rule)
    serve(monkeypatch, resources(Envs(), ()), resources(Envs(), ()))
    deferred = driver.DEFERRED if enters else 0
    assert await driver.one(driver_args) == deferred
    assert await driver.one(driver_args) == deferred  # driver_args.resume_held stays False
    assert entered == [False] * (1 + enters)
    held_wp = published_workpiece(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    assert held_wp.terminal is Terminal.HELD and driver.deferred(held_wp) is enters
    positional = Incomplete(("held",), "incomplete")  # the earlier constructor: the second argument is still the kind
    assert (positional.kind, positional.awaits) == ("incomplete", None)
    assert not driver.deferred(replace(held_wp, decisions=(*held_wp.decisions, positional)))


def answering(*answers: DecisionModelHealth) -> tuple[list[DecisionModelHealth], Any]:
    """A health read giving ``answers`` in turn and then the last one; the list collects each answer given."""
    given: list[DecisionModelHealth] = []

    async def health(_policy: Policy) -> DecisionModelHealth:
        given.append(answers[min(len(given), len(answers) - 1)])
        return given[-1]

    return given, health


def batch(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: TaskProposal,
    codes: Mapping[str, Sequence[int]],
    concurrency: int = 1,
) -> list[str]:
    """A run over one proposal per ``codes`` entry, decision-model ones unless named ``code``, whose children exit with
    their codes in turn and then 0; the list collects each child launch, starred for a probe, and none carries
    --resume-held."""
    decision = replace(proposal, header=replace(proposal.header, verification=Verification.DECISION_MODEL))
    paths = {name: tmp_path / f"{name}.proposal.json" for name in codes}
    for name, path in paths.items():
        path.write_bytes(TASK_PROPOSAL.dump_json(proposal if name == "code" else decision))
    manifest, overlay = tmp_path / "manifest.txt", Path(driver_args.policy)
    manifest.write_text("\n".join(map(str, paths.values())))
    overlay.write_text(json.dumps({**json.loads(overlay.read_text()), "decision_model.retry_interval": 0.0}))
    driver_args.proposals, driver_args.concurrency = str(manifest), concurrency
    remaining = {str(path): list(codes[name]) for name, path in paths.items()}
    launches: list[str] = []

    async def child(args: Namespace, _controller_url: str | None) -> int:
        assert not args.resume_held, "only the operator reopens a hold"
        launches.append(Path(args.proposals).stem.removesuffix(".proposal") + "*" * getattr(args, "probe", False))
        return (remaining[args.proposals] or [0]).pop(0)

    monkeypatch.setattr(driver, "one", child)
    monkeypatch.setattr(driver.subprocess, "Popen", partial(in_process, monkeypatch))
    return launches


async def test_a_deferring_run_starts_unrelated_work_and_holds_decision_model_proposals_until_healthy(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: TaskProposal,
    capsys: pytest.CaptureFixture[str],
) -> None:
    codes = {"first": [driver.DEFERRED], "second": [], "code": []}
    launches = batch(driver_args, tmp_path, monkeypatch, proposal, codes)
    given, health = answering(DecisionModelHealth.DOWN, DecisionModelHealth.HEALTHY)
    monkeypatch.setattr(driver, "clef", lambda policy, *_: SimpleNamespace(health=partial(health, policy)))
    await driver.run(driver_args)
    assert launches == ["first", "code", "first", "second"]
    assert given == [DecisionModelHealth.DOWN, DecisionModelHealth.HEALTHY]
    out = capsys.readouterr().out
    assert f"OUTCOME deferred exit={driver.DEFERRED} proposal={tmp_path / 'first.proposal.json'}" in out
    assert "submitted=4 completed=4 failed=0 skipped=0" in out


# Regression: with one child slot, a code proposal never started while a health read was
# pending. The run reads health in the background and keeps reaping and launching children while it waits.
async def test_a_pending_health_read_holds_back_no_unrelated_proposal(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    launches = batch(driver_args, tmp_path, monkeypatch, proposal, {"first": [driver.DEFERRED], "code": []})

    async def health(_policy: Policy) -> DecisionModelHealth:
        while "code" not in launches:  # answers only once the unrelated proposal has run
            await asyncio.sleep(0)
        return DecisionModelHealth.HEALTHY

    monkeypatch.setattr(driver, "clef", lambda policy, *_: SimpleNamespace(health=partial(health, policy)))
    async with asyncio.timeout(30.0):  # a health read that blocks dispatch never answers
        await driver.run(driver_args)
    assert launches == ["first", "code", "first"]


# Regression: without a health route, a probe child that crashed, was superseded, needed a
# migration or found a corrupt journal reopened the queue. Only a probe that is not deferred again releases the
# rest; any other exit is reported as usual and another parked proposal probes after the interval.
@pytest.mark.parametrize("code", [1, driver.SUPERSEDED, driver.NEEDS_MIGRATION, driver.CORRUPT])
async def test_without_a_health_route_only_an_answered_probe_reopens_decision_model_work(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: TaskProposal,
    capsys: pytest.CaptureFixture[str],
    code: int,
) -> None:
    codes = {"a": [driver.DEFERRED, code], "b": [driver.DEFERRED], "c": [driver.ANSWERED]}
    launches = batch(driver_args, tmp_path, monkeypatch, proposal, codes)
    given, health = answering(DecisionModelHealth.UNSUPPORTED)
    monkeypatch.setattr(driver, "clef", lambda policy, *_: SimpleNamespace(health=partial(health, policy)))
    with contextlib.suppress(RuntimeError):  # a failed probe fails the run, once every proposal has run
        await driver.run(driver_args)
    # a defers, and b and c wait behind it; a probes and fails, so b probes and defers again, then c probes and is
    # answered, which releases b. Health is read before each probe: after a's hold, a's failed probe and b's hold.
    assert (launches, len(given)) == (["a", "a*", "b*", "c*", "b"], 3)
    label = "failed" if code == 1 else driver.OUTCOMES[code][0]
    assert f"OUTCOME {label} exit={code} proposal={tmp_path / 'a.proposal.json'}" in capsys.readouterr().out


# Regression: an ordinary child exiting 10 was counted a success. Only the
# designated probe's 10 is an answer; from any other child it is the unexpected exit it is.
async def test_an_answer_from_a_child_that_is_no_probe_is_a_failure(
    driver_args: Namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: TaskProposal,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launches = batch(driver_args, tmp_path, monkeypatch, proposal, {"code": [driver.ANSWERED]})
    with pytest.raises(RuntimeError, match="1 workpieces failed"):
        await driver.run(driver_args)
    assert launches == ["code"]
    assert f"OUTCOME failed exit={driver.ANSWERED} proposal={tmp_path / 'code.proposal.json'}" in capsys.readouterr().out


# Regression: a probe whose grade replayed a cached read exited 0 having made no
# decision-model call, and released every parked proposal. A probe that proves no fresh answer stays a probe: each
# parked proposal probes in turn after the interval, and none is released on its evidence.
async def test_a_probe_that_shows_no_fresh_answer_releases_nothing(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    launches = batch(driver_args, tmp_path, monkeypatch, proposal, {"a": [driver.DEFERRED, 0], "b": [], "c": []})
    given, health = answering(DecisionModelHealth.UNSUPPORTED)
    monkeypatch.setattr(driver, "clef", lambda policy, *_: SimpleNamespace(health=partial(health, policy)))
    await driver.run(driver_args)
    assert (launches, len(given)) == (["a", "a*", "b*", "c*"], 3)


# Regression: after one healthy reading the first released child deferred again,
# and the rest of the released queue kept starting while the next health read was pending. A fresh outage parks
# everything not yet started: no decision-model child launches while health is being read.
async def test_a_fresh_outage_parks_released_work_that_has_not_started(
    driver_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    codes = {"a": [driver.DEFERRED, driver.DEFERRED], "b": [], "c": [], "d": []}
    launches, started, seen = batch(driver_args, tmp_path, monkeypatch, proposal, codes), list[str](), list[int]()
    popen = driver.subprocess.Popen
    monkeypatch.setattr(driver.subprocess, "Popen", lambda command, **_: started.append(command[-1]) or popen(command))

    async def health(_policy: Policy) -> DecisionModelHealth:
        seen.append(len(started))  # admission is synchronous: a launch beside this read happened before it
        return DecisionModelHealth.HEALTHY

    monkeypatch.setattr(driver, "clef", lambda policy, *_: SimpleNamespace(health=partial(health, policy)))
    await driver.run(driver_args)
    assert (seen, sorted(launches)) == ([1, 2], ["a", "a", "a", "b", "c", "d"])


class ProbeReader(DecisionModelReader):
    """Answers 1.0, except that "13" is down and "hang" is dispatched and never answered; ``reads`` collects them."""

    def __init__(self, reads: list[str]) -> None:
        self.reads = reads

    async def read(self, request: DecisionModelRequest) -> float:
        self.reads.append(request.candidate)
        if request.candidate == "13":
            raise DecisionModelDown("HTTP 503: clef_unavailable")
        if request.candidate == "hang":
            await asyncio.Event().wait()
        return 1.0

    async def health(self) -> DecisionModelHealth:
        return DecisionModelHealth.HEALTHY


def asked(candidate: str) -> DecisionModelRead:
    return DecisionModelRead(
        DecisionModelRequest("clef@fixture-1", "whole", ("no", "yes"), "Sum.", "The sum.", "12", candidate)
    )


def probe_bundles(
    reads: list[str], count: int, reader: DecisionModelReader | None = None, request_timeout: float = 30.0
) -> list[Resources]:
    """Launch resources whose decision model reads through ``ProbeReader``: two slots, a 10 s wait, a 30 s request
    deadline, and a backoff that lasts until it is cancelled."""

    async def backoff(_seconds: float) -> None:
        await asyncio.Event().wait()

    policy = DecisionModelPolicy("clef@fixture-1", 2, request_timeout, 10.0, 1.0, 1.0, 0.0)
    return [
        replace(resources(Envs(), ()), decision_model=DecisionModel(reader or ProbeReader(reads), policy, sleep=backoff))
        for _ in range(count)
    ]


# Regression: what a probe child reports. Only a probe that resolved a decision-
# model read in its own process, not one replayed from the journal, answers; a skipped root proves nothing either.
async def test_a_probe_child_answers_only_for_a_read_resolved_in_its_own_process(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []

    async def reading(ctx: Context, _wp: object, _config: object, bundle: Resources) -> object:
        ctx.handle(DecisionModelRead, partial(read_level, ctx=ctx), bundle.decision_model)
        await ctx.emit(asked("12"))
        raise Hold("a human reads this")

    monkeypatch.setattr(coordinator, "STAGES", {**STAGES, Stage.AUTHOR: reading})
    serve(monkeypatch, *probe_bundles(reads, 3))
    codes = []
    for resume in (False, True, False):  # fresh; replayed on an explicit resume; skipped as held
        driver_args.probe, driver_args.resume_held = True, resume
        codes.append(await driver.one(driver_args))
    assert (codes, reads) == ([driver.ANSWERED, 0, 0], ["12"])


CLEF_BODIES: Mapping[str, httpx.Response] = {
    "off-scale": httpx.Response(200, content=b'{"answers": {"whole": {"probabilities": {"0": Infinity, "1": 1}}}}'),
    "forbidden": httpx.Response(403, text="forbidden"),
    "malformed": httpx.Response(200, json={"unexpected": True}),
}


# Regression: through the default reader, one 200 followed only by an off-scale level, an
# HTTP 403 or a 200 without answers left the earlier answer standing, and the probe released its peers with the root
# manual. Every dispatched read that fails records the failure, whatever its class; only the caller's own cancellation
# of a read in flight leaves the earlier answer standing. A read whose own request deadline expired, its
# waiter then cancelled during the HTTP cleanup, ended in CancelledError and was taken for the caller's leaving; a
# cancellation is the caller's only while neither the request's deadline nor the overall wait has expired.
@pytest.mark.parametrize(
    "second, code",
    [("off-scale", 0), ("forbidden", 0), ("malformed", 0), ("cancelled", driver.ANSWERED), ("lapsed", 0)],
)
async def test_a_probe_through_the_default_reader_counts_every_failed_dispatch(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, second: str, code: int
) -> None:
    sent, clock, cleanup = list[str](), LoopTime(), asyncio.Event()

    async def gateway(call: httpx.Request) -> httpx.Response:
        candidate = json.loads(call.content)["state"]["z_candidate_answer"]
        sent.append(candidate)
        if candidate == "lapsed":  # its request deadline cancels it; the waiter's cancellation arrives during cleanup
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await asyncio.Event().wait()
        if candidate == "cancelled":
            await asyncio.Event().wait()
        probabilities = {"0": 0.0, "1": 1.0}
        return CLEF_BODIES.get(candidate) or httpx.Response(
            200, json={"answers": {"whole": {"probabilities": probabilities}}}
        )

    source = FixedEndpoint(GlmEndpoint("http://gateway/v1", "secret", Pool.HIGH))
    reader = ClefReader(source, httpx.AsyncClient(transport=httpx.MockTransport(gateway)))

    async def reading(ctx: Context, _wp: object, _config: object, bundle: Resources) -> object:
        ctx.handle(DecisionModelRead, partial(read_level, ctx=ctx), bundle.decision_model)
        assert await ctx.emit(asked("12")) == 1.0
        read = asyncio.create_task(ctx.emit(asked(second)))
        while len(sent) < 2:
            await asyncio.sleep(0)
        if second == "lapsed":  # past the 1 s request deadline, well inside the 10 s wait
            await clock.sleep(2.0)
            while not cleanup.is_set():
                await asyncio.sleep(0)
        if second in ("cancelled", "lapsed"):  # the caller leaves while its read is in flight
            read.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await read
        raise Hold("a sibling's refusal: a human reads this")

    monkeypatch.setattr(coordinator, "STAGES", {**STAGES, Stage.AUTHOR: reading})
    serve(monkeypatch, *probe_bundles([], 1, reader, request_timeout=1.0))
    driver_args.probe = True
    assert await driver.one(driver_args) == code
    held = published_workpiece(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    assert held.terminal is Terminal.HELD and not driver.deferred(held) and sent == ["12", second]


# Regression: a probe that read 200 and then met a 503 ended on a human hold yet
# answered, first in general and then when the 503's retry row landed before the 200's resolution in one loop turn;
# and three answers followed by a fourth dispatched read that expired in flight left no row at all and answered. The
# reader's last physical outcome in the probe's process decides: a later failure, or a dispatched read unanswered at
# its deadline, proves nothing; a read that expired only waiting for a slot is no evidence; a pure outage defers.
@pytest.mark.parametrize(
    "scenario, code",
    [
        ("answered", 10),
        ("same-tick-200-then-503", 0),
        ("in-flight-expiry", 0),
        ("queue-only-expiry", 10),
        ("pure-outage", 75),
    ],
)
async def test_a_probe_child_reports_its_readers_last_physical_outcome(
    driver_args: Namespace, monkeypatch: pytest.MonkeyPatch, scenario: str, code: int
) -> None:
    reads: list[str] = []
    clock = LoopTime()

    async def lapsed(read: asyncio.Task[float]) -> asyncio.Task[float]:
        for _ in range(10):  # the read arms its deadline, then the clock passes the 10 s wait
            await asyncio.sleep(0)
        await clock.sleep(11.0)
        return read

    async def expired(ctx: Context, candidate: str) -> None:
        with pytest.raises(DecisionModelUnavailable):
            await (await lapsed(asyncio.create_task(ctx.emit(asked(candidate)))))

    async def reading(ctx: Context, _wp: object, _config: object, bundle: Resources) -> object:
        model = bundle.decision_model
        assert model is not None
        ctx.handle(DecisionModelRead, partial(read_level, ctx=ctx), model)
        if scenario == "pure-outage":
            await (await lapsed(asyncio.create_task(ctx.emit(asked("13")))))  # the typed outage: the workpiece defers
        if scenario == "same-tick-200-then-503":
            answered, down = asyncio.create_task(ctx.emit(asked("12"))), asyncio.create_task(ctx.emit(asked("13")))
            await answered
            while len(reads) < 2:
                await asyncio.sleep(0)
            down.cancel()
            await asyncio.gather(down, return_exceptions=True)
        else:
            for candidate in ("1", "2", "3") if scenario == "in-flight-expiry" else ("12",):
                await ctx.emit(asked(candidate))
        if scenario == "in-flight-expiry":
            await expired(ctx, "hang")
        if scenario == "queue-only-expiry":  # both slots held, so the read is never dispatched
            for _ in range(2):
                await model.gate.acquire()
            await expired(ctx, "queued")
        raise Hold("a sibling's refusal: a human reads this")

    monkeypatch.setattr(coordinator, "STAGES", {**STAGES, Stage.AUTHOR: reading})
    serve(monkeypatch, *probe_bundles(reads, 1))
    driver_args.probe = True
    assert await driver.one(driver_args) == code
    held = published_workpiece(Path(driver_args.output) / driver.proposal_key(driver_args.proposals))
    assert held.terminal is Terminal.HELD and driver.deferred(held) is (code == driver.DEFERRED)
    assert "queued" not in reads


def test_a_second_process_replays_a_provider_context_refusal_without_asking_again(tmp_path: Path) -> None:
    """The provider refused the author's request for a full context window and the author stopped; a cold launch in
    another process, whose provider would now answer, replays that stop from the journal and makes no provider call,
    and so does an explicit resume: the settled stop decides what follows it."""
    probe, root = [sys.executable, "-m", "lib.taskforge.tests.context_replay_probe"], Path(__file__).parents[3]
    stop = "('author stopped: context_limit after 1 turns: maximum context length exceeded',)"
    for mode, calls in (("record", 1), ("launch", 0), ("resume", 0)):
        done = subprocess.run([*probe, mode, str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        assert done.stdout.strip() == f"{mode}: abandoned, {stop}, {calls} provider call(s)"
