# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from functools import partial
from pathlib import PurePosixPath

import pytest
from lib.taskforge.tests.resources.rollout_fakes import FakeEngine, twelve
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.machines import MachineSetupError
from rolloutengine.models import (
    GradedRollout,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    TraceMode,
)
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits, rollout_spec
from shellbox.machine import (
    ExecSetupFailed,
    ExitReason,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineTerminated,
    MachineUnavailable,
    Result,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    GraderMachine,
    ResourceGroups,
    ScriptGrader,
    StdoutReward,
    TextMessage,
    ToolResult,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import NumericSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Event, Retrying
from taskforge.effects.issues import EnvironmentSetupFailed, OutputBudgetExhausted, Restart, Transient
from taskforge.effects.log import EventLog
from taskforge.harness.protocol import ToolResult as ToolReply
from taskforge.harness.tools.grade import GradeTool
from taskforge.lowering import GRADING_MACHINE, LoweringSettings, requirements, rollout_engine
from taskforge.model.controls import Candidate
from taskforge.resources.env import ShellboxEnvFactory, WriteFile
from taskforge.resources.grading import GradeCandidate, GradingResource, candidate_turns, grade
from taskforge.resources.rollouts import RolloutResource, SolverRollout, interruption, run_rollout
from taskforge.resources.sandbox.factories import shellsim_factory
from taskforge.stages.author import install_env


def test_candidate_file_calls_do_not_collide_with_conversation_history() -> None:
    history = (TextMessage(role="user", content="Solve"),)
    for call_id in ("file-0", "file-0-", "file-1"):
        history += (
            AssistantToolCalls(calls=(ConversationToolCall(call_id=call_id, name="shell", arguments={}),)),
            ToolResult(call_id=call_id, content="done"),
        )
    task = arithmetic_task(context=ConversationInput(events=history))
    candidate = Candidate("12", (inline_resource("a", b"a"), inline_resource("b", b"b")))
    turns = candidate_turns(candidate, rollout_spec(task))
    assert len(turns) == 3
    used_ids = {"file-0", "file-0-", "file-1"}
    for turn in turns[:-1]:
        calls = turn["tool_calls"]
        assert isinstance(calls, list)
        call = calls[0]
        assert call["id"] not in used_ids
        used_ids.add(call["id"])
        event = AssistantToolCalls(calls=(ConversationToolCall(call_id=call["id"], name="shell", arguments={}),))
        ConversationTrace(events=(*history, event))
        history += (event, ToolResult(call_id=call["id"], content="done"))
    ConversationTrace(events=(*history, TextMessage(role="assistant", content="12")))


@pytest.mark.parametrize("operation", tuple(RolloutOperation))
async def test_rollout_interruptions_select_retry_or_restart(operation: RolloutOperation) -> None:
    interrupted = RolloutInterrupted(Rollout("trial", (), "interrupted"), operation)
    cause = RuntimeError("sandbox startup failed")
    cause.__cause__ = RuntimeError("ImagePullBackOff: registry.example.com/" + "a" * 400)
    interrupted.__cause__ = cause
    engine = FakeEngine((interrupted,))
    resource = RolloutResource(engine, twelve, "test-config")
    decision = Transient if operation == RolloutOperation.START else Restart
    with pytest.raises(decision) as failure:
        await resource.run(SolverRollout(rollout_spec(arithmetic_task()), 4, 2))
    assert failure.value.__cause__ is interrupted
    detail = (f"RuntimeError: {cause} <- RuntimeError: {cause.__cause__}")[:300]
    assert str(failure.value) == f"Rollout interrupted during {operation}: {detail}"
    assert "ImagePullBackOff" in str(failure.value)
    assert engine.rollout_ids == ["arithmetic:4:2"]


async def test_grading_runs_real_engine_without_inference() -> None:
    spec = rollout_spec(arithmetic_task(), trace=TraceMode.MESSAGES)
    engine = RolloutEngine({}, {"shell": ShellHarness(command_timeout=None)})
    grader = GradingResource(engine, "test-config")
    correct = await grader.grade(GradeCandidate(spec, Candidate("12")))
    incorrect = await grader.grade(GradeCandidate(spec, Candidate("9")))
    assert correct.status == Outcome.GRADED and correct.reward == 1
    assert incorrect.status == Outcome.GRADED and incorrect.reward == 0
    authored = arithmetic_task(
        id="new-task", grader=verifyit_package(NumericSpec("9", tolerance_abs=0, tolerance_rel=0)).grader
    )
    new_grade = await grader.grade(GradeCandidate(rollout_spec(authored), Candidate("9")))
    assert new_grade.status == Outcome.GRADED and new_grade.reward == 1


async def test_grading_timeout_replays_candidate_on_new_attempt_once() -> None:
    timeout = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "timeout",
        failure=GradingFailure.TIMEOUT,
    )
    success = GradeResult(Outcome.GRADED, 1, "correct")
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, success)))
    grader = GradingResource(engine, "config")
    assert await grader.grade(GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))) == success
    assert engine.replies == ["12", "12"]
    assert len(set(engine.rollout_ids)) == 2


async def test_grading_repeated_timeout_returns_the_second_grade() -> None:
    timeout = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "timeout",
        failure=GradingFailure.TIMEOUT,
    )
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, timeout)))
    grader = GradingResource(engine, "config")
    assert await grader.grade(GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))) == timeout
    assert len(engine.rollout_ids) == 2


@pytest.mark.parametrize("failures", [1, 3])
async def test_grading_retries_pre_command_websocket_setup_failures(failures: int) -> None:
    interrupted = RolloutInterrupted(Rollout("trial", (), "interrupted"), RolloutOperation.GRADE)
    interrupted.__cause__ = ExecSetupFailed("handshake refused")
    success = GradedRollout(Rollout("trial", (), "stop"), GradeResult(Outcome.GRADED, 1, "ok"))
    engine = FakeEngine((*([interrupted] * failures), success))
    events: list[Event[None]] = []
    event = GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))
    resource = GradingResource(engine, "test")
    if failures == 3:
        with pytest.raises(Restart, match="exhausted its retry budget"):
            await grade(event, resource, notify=events.append)
    else:
        assert (await grade(event, resource, notify=events.append)).reward == 1
    assert len(engine.rollout_ids) == min(failures + 1, 3)
    assert len(events) == min(failures, 2)
    assert all(isinstance(record, Retrying) for record in events)


@pytest.mark.parametrize("solver", [False, True])
async def test_early_grader_cut_is_not_cached_for_stage_replay(solver: bool) -> None:
    cut = RolloutInterrupted(Rollout("cut", (), "interrupted"), RolloutOperation.GRADE)
    cut.__cause__ = MachineTerminated("Iris command cut after 63s before 600s budget")
    success = GradeResult(Outcome.GRADED, 1)
    engine = FakeEngine((cut, GradedRollout(Rollout("ok", (), "stop"), success)))
    ctx = Context(log=EventLog())
    spec = rollout_spec(arithmetic_task())
    if solver:
        resource = RolloutResource(engine, twelve, "cut-test")
        ctx.resource(resource, resource.fingerprint)
        ctx.handle(SolverRollout, partial(run_rollout, notify=ctx.emit), uses=(resource,))
        event = SolverRollout(spec, 0, 0)
    else:
        resource = GradingResource(engine, "cut-test")
        ctx.resource(resource, resource.fingerprint)
        ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(resource,))
        event = GradeCandidate(spec, Candidate("12"))
    with pytest.raises(Restart):
        await ctx.emit(event)
    await ctx.emit(event)
    await ctx.emit(event)
    await ctx.drain()
    assert len(engine.rollout_ids) == 2


@pytest.mark.parametrize("operation", list(RolloutOperation))
@pytest.mark.parametrize(
    "failure",
    [
        MachineCancelled("parent job terminated"),
        MachineUnavailable("controller unavailable"),
        MachineOutputLimitExceeded("output budget exhausted"),
    ],
)
def test_machine_failure_policy(operation: RolloutOperation, failure: RuntimeError | ConnectionError) -> None:
    error = RolloutInterrupted(Rollout("failure", (), "interrupted"), operation)
    error.__cause__ = failure
    expected = (
        (OutputBudgetExhausted if operation is RolloutOperation.ADVANCE else Restart)
        if isinstance(failure, MachineOutputLimitExceeded)
        else Transient
    )
    decision = interruption(error)
    assert type(decision) is expected and str(failure) in str(decision)


def test_a_failing_setup_command_is_the_tasks_evidence_not_a_retry() -> None:
    error = RolloutInterrupted(Rollout("setup", (), "interrupted"), RolloutOperation.START)
    error.__cause__ = MachineSetupError(
        "git clone", Result(128, b"", b"fatal: could not read from remote", False, False, ExitReason.EXITED)
    )
    decision = interruption(error)
    assert type(decision) is EnvironmentSetupFailed and "git clone" in str(decision)


@pytest.mark.parametrize("path", ["answer.py", "/workspace/answer.py", "/elsewhere/answer.py"])
async def test_grade_tool_stages_candidate_file_at_its_workspace_path(path: str) -> None:
    script = b"""
from pathlib import Path
staged = Path('/workspace/answer.py')
print(int(staged.exists() and staged.read_text() == 'print(12)' and not Path('/workspace/workspace').exists()))
"""
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        reward=StdoutReward(),
        placement=GraderMachine(environment=requirements(image=None)),
        answer_path="/app/answer.txt",
        timeout=120,
    )
    task = arithmetic_task(
        grader=grader,
        resources=ResourceGroups(verifier=(inline_resource("grade.py", script),)),
        environment_requirements=requirements(image=None),
        output_paths=("/workspace/answer.py",),
    )
    resource = GradingResource(rollout_engine({"shellsim": shellsim_factory()}, command_timeout=None), "test")
    ctx = Context(log=EventLog())
    env = await ShellboxEnvFactory(shellsim_factory(), machine_runtime(backend="shellsim"), 5, ctx.emit).create(
        requirements(image=None)
    )
    try:
        install_env(ctx, env)
        ctx.resource(resource, resource.fingerprint)
        ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(resource,))
        await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), task.model_dump_json().encode()))
        await ctx.emit(WriteFile(PurePosixPath("/workspace/answer.py"), b"print(12)"))
        await ctx.emit(WriteFile(PurePosixPath("/elsewhere/answer.py"), b"print(12)"))
        settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        tool = GradeTool("/workspace/task.json", {"shellsim": shellsim_factory()}, settings)
        result = await tool(ctx, {"reply": "12", "files": [path]})
        await ctx.drain()
    finally:
        await env.close()
    assert isinstance(result, ToolReply)
    if path.startswith("/elsewhere"):
        assert "pass a path relative to /workspace" in result.content
    else:
        assert json.loads(result.content)["reward"] == 1
