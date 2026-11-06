# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from rolloutengine.models import GradedRollout, Model, ModelRequest, ModelTurn, Rollout, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from forge import driver
from forge.config import ForgeConfig
from forge.coordinator import run_workpiece
from forge.effects.events import StageCompleted
from forge.effects.issues import Restart
from forge.effects.log import EventLog, JsonlSink, fold
from forge.harness.web import FakeWeb
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, FinishReason, ToolCall
from forge.model.proposal import TaskProposal
from forge.model.review import BandRules, Cause, Decision, Modify, Retry, ValidationResult
from forge.model.serialization import CONTROL, TASK_PROPOSAL, WORKPIECE
from forge.model.workpiece import Terminal
from forge.policy import Policy, load_policy
from forge.resources.bundle import Resources
from forge.resources.clock import Clock
from forge.resources.env import CommandResult, EnvFactory, FakeCommand, FakeEnv
from forge.resources.grading import GradingResource
from forge.resources.ids import Ids
from forge.resources.inference import FakeInference
from forge.resources.rollouts import Engine, RolloutResource
from forge.wiring import build_config

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
    def __init__(self, *, fail: bool = False, host_failure: bool = False) -> None:
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
        result = CommandResult(1 if host_failure else 0, "", "No space left on device" if host_failure else "", 0, False)
        self.commands = {"inspect": FakeCommand(result, {}), "write-draft": FakeCommand(result, writes)}

    @override
    async def create(self, requirements: EnvironmentRequirements) -> RecordingEnv:
        env = RecordingEnv(self.commands, self.fail)
        self.created.append(env)
        return env


class GradeEngine(Engine):
    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        if ":grade:" in run_id:
            response = await model(ModelRequest((), {}))
            reward = float(response.message["content"] == "12")
        else:
            reward = float(run_id.split(":")[-2] == "0")
        return GradedRollout(Rollout(run_id, (), "stop"), GradeResult(Outcome.GRADED, reward, "fake grade"))


async def solver(_request: ModelRequest) -> ModelTurn:
    return ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")


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
            {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json", "notes": "done"},
            "submit",
        ),
    )


def resources(envs: Envs, responses: Sequence[Completion]) -> Resources:
    engine = GradeEngine()
    return Resources(
        envs,
        FakeInference(responses),
        RolloutResource(engine, solver, "solver"),
        GradingResource(engine, "grader"),
        Clock(lambda: 1),
        Ids(lambda: "workpiece"),
        {},
        FakeWeb({}, {}),
    )


def config(arm: str) -> ForgeConfig:
    root = Path(__file__).parents[1] / "src/forge/policy/defaults"
    policy = load_policy(root).overlaid({"solver.k": 2, "adversary.roster": [], "budget.max_modify_rounds": 1})
    return build_config(policy, arm, {})


@pytest.mark.parametrize("arm", ("S", "T"))
async def test_both_arms_accept_and_record_stage_sequence(proposal: TaskProposal, arm: str) -> None:
    log = EventLog()
    wp = await run_workpiece(proposal, config(arm), resources(Envs(), script(arm)), log)
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
    assert (await run_workpiece(proposal, config("S"), broken, first)).terminal == Terminal.HELD
    assert isinstance(broken.inference, FakeInference)
    assert len(broken.inference.requests) == 2
    resumed = resources(Envs(), script("S")[2:])
    log = EventLog(replay_from=await sink.read(), sink=sink)
    wp = await run_workpiece(proposal, config("S"), resumed, log)
    assert wp.terminal == Terminal.ACCEPTED
    assert isinstance(resumed.inference, FakeInference)
    assert isinstance(resumed.envs, Envs)
    assert len(resumed.inference.requests) == 2
    assert resumed.envs.created[0].executed == ["inspect", "write-draft"]


def needs_revision(result: ValidationResult, band: BandRules) -> Decision:
    return Modify((f"Revise {result.task_digest} for band {band.low}-{band.high}",))


async def test_modify_budget_ends_abandoned(proposal: TaskProposal) -> None:
    settings = replace(config("S"), review=needs_revision)
    wp = await run_workpiece(proposal, settings, resources(Envs(), (*script("S"), *script("S"))), EventLog())
    assert wp.terminal == Terminal.ABANDONED
    assert wp.round == 1
    assert len(wp.decisions) == 2


async def test_host_classifier_holds_workpiece(proposal: TaskProposal) -> None:
    wp = await run_workpiece(proposal, config("S"), resources(Envs(host_failure=True), script("S")), EventLog())
    assert wp.terminal == Terminal.HELD
    assert "space" in str(wp.decisions[-1]).lower()


def retry_review(result: ValidationResult, band: BandRules) -> Decision:
    assert result.solver and band.low > 0
    return Retry(Cause.MODEL_UNAVAILABLE, 1)


async def test_validation_retry_budget_holds_without_restarting_author(proposal: TaskProposal) -> None:
    settings = replace(config("S"), review=retry_review)
    log = EventLog()
    wp = await run_workpiece(proposal, settings, resources(Envs(), script("S")), log)
    assert wp.terminal == Terminal.HELD
    assert len(wp.decisions) == settings.budget.max_validation_retries + 1
    stages = [r.event.stage for r in log.records if isinstance(r.event, StageCompleted)]
    assert stages == ["author", *[stage for _ in wp.decisions for stage in ("validate", "review")]]


async def test_driver_persists_programming_failure_then_reraises(
    proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def broken_runtime(policy: Policy, arm: str) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
        yield build_config(policy, arm, {}), resources(Envs(), ())

    monkeypatch.setattr(driver, "runtime", broken_runtime)
    source, overlay = tmp_path / "test.proposal.json", tmp_path / "policy.json"
    source.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    overlay.write_text("{}")
    args = Namespace(
        proposals=str(source),
        policy=str(overlay),
        arm="S",
        output=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
    )
    with pytest.raises(RuntimeError, match="exhaust"):
        await driver.one(args)
    destination = tmp_path / "out" / driver.proposal_key(str(source))
    result = WORKPIECE.validate_json((destination / "workpiece.json").read_bytes())
    assert result.terminal == Terminal.HELD
    assert "bug:" in str(result.decisions[-1]) and "Traceback" in str(result.decisions[-1])
    assert json.loads((destination / "status.json").read_text())["terminal"] == "held"
    assert await JsonlSink(str(destination / "events.jsonl")).read()
