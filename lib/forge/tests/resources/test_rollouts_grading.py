# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from functools import partial
from typing import override

import pytest
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import (
    GradedRollout,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
    TraceMode,
)
from rolloutengine.test_support import arithmetic_task, rollout_spec
from shellbox.machine import (
    ExecSetupFailed,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineTerminated,
    MachineUnavailable,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    TextMessage,
    ToolResult,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import NumericSpec

from forge.effects.context import Context
from forge.effects.events import Event, Retrying
from forge.effects.issues import OutputBudgetExhausted, Restart, Transient
from forge.effects.log import EventLog
from forge.model.controls import Candidate
from forge.resources.grading import GradeCandidate, GradingResource, candidate_turns, grade
from forge.resources.rollouts import Engine, RolloutResource, SolverRollout, interruption, run_rollout


class FakeEngine(Engine):
    def __init__(self, results: Sequence[GradedRollout | RolloutInterrupted]) -> None:
        self.results = tuple(results)
        self.run_ids: list[str | None] = []
        self.replies: list[str] = []

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        index = len(self.run_ids)
        self.run_ids.append(run_id)
        reply = await model(ModelRequest((), {}))
        self.replies.append(str(reply.message.get("content")))
        result = self.results[index]
        if isinstance(result, RolloutInterrupted):
            raise result
        return result


async def twelve(request: ModelRequest) -> ModelTurn:
    return ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")


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
    assert turns == candidate_turns(candidate, rollout_spec(task))
    for turn in turns[:-1]:
        calls = turn["tool_calls"]
        assert isinstance(calls, list)
        call = calls[0]
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
    assert engine.run_ids == ["arithmetic:4:2"]


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
        diagnostics={"grader_elapsed_seconds": 300.0, "grader_timeout_seconds": 300.0},
    )
    success = GradeResult(Outcome.GRADED, 1, "correct")
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, success)))
    grader = GradingResource(engine, "config")
    assert await grader.grade(GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))) == success
    assert engine.replies == ["12", "12"]
    assert len(set(engine.run_ids)) == 2


async def test_grading_repeated_timeout_returns_the_second_grade() -> None:
    timeout = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "timeout",
        failure=GradingFailure.TIMEOUT,
        diagnostics={"grader_elapsed_seconds": 300.0, "grader_timeout_seconds": 300.0},
    )
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, timeout)))
    grader = GradingResource(engine, "config")
    assert await grader.grade(GradeCandidate(rollout_spec(arithmetic_task()), Candidate("12"))) == timeout
    assert len(engine.run_ids) == 2


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
    assert len(engine.run_ids) == min(failures + 1, 3)
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
    assert len(engine.run_ids) == 2


def test_empty_interruption_causes_keep_exception_types() -> None:
    cancelled = asyncio.CancelledError()
    timeout = TimeoutError()
    timeout.__cause__ = cancelled
    interrupted = RolloutInterrupted(Rollout("test", (), "stop"), RolloutOperation.GRADE)
    interrupted.__cause__ = timeout
    decision = interruption(interrupted)
    assert isinstance(decision, Restart)
    assert str(decision) == "Rollout interrupted during grade: TimeoutError <- CancelledError"


@pytest.mark.parametrize("operation", list(RolloutOperation))
def test_parent_cancellation_is_transient_in_every_rollout_phase(operation: RolloutOperation) -> None:
    error = RolloutInterrupted(Rollout("cancelled", (), "interrupted"), operation)
    error.__cause__ = MachineCancelled("parent job terminated")
    decision = interruption(error)
    assert isinstance(decision, Transient)
    assert "parent job terminated" in str(decision)


@pytest.mark.parametrize("operation", [RolloutOperation.ADVANCE, RolloutOperation.GRADE])
def test_only_agent_output_exhaustion_counts_as_agent_failure(operation: RolloutOperation) -> None:
    error = RolloutInterrupted(Rollout("output", (), "interrupted"), operation)
    error.__cause__ = MachineOutputLimitExceeded("ShellSim terminated: output_limit_exceeded")
    assert isinstance(interruption(error), OutputBudgetExhausted) is (operation is RolloutOperation.ADVANCE)


@pytest.mark.parametrize("operation", list(RolloutOperation))
def test_controller_unavailability_is_transient_in_every_phase(operation: RolloutOperation) -> None:
    error = RolloutInterrupted(Rollout("unavailable", (), "interrupted"), operation)
    error.__cause__ = MachineUnavailable("Iris controller unavailable: Controller is shutting down")
    assert isinstance(interruption(error), Transient)
