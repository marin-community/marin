# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
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
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome

from forge.effects.issues import Restart, Transient
from forge.model.controls import Candidate
from forge.resources.grading import GradeCandidate, GradingResource
from forge.resources.rollouts import Engine, RolloutResource, SolverRollout


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


@pytest.mark.parametrize("operation", tuple(RolloutOperation))
async def test_rollout_interruptions_select_retry_or_restart(operation: RolloutOperation) -> None:
    interrupted = RolloutInterrupted(Rollout("trial", (), "interrupted"), operation)
    engine = FakeEngine((interrupted,))
    resource = RolloutResource(engine, twelve, "test-config")
    decision = Transient if operation == RolloutOperation.START else Restart
    with pytest.raises(decision) as failure:
        await resource.run(SolverRollout(rollout_spec(arithmetic_task()), 4, 2))
    assert failure.value.__cause__ is interrupted
    assert engine.run_ids == ["arithmetic:4:2"]


async def test_grading_runs_real_engine_without_inference() -> None:
    spec = rollout_spec(arithmetic_task(), trace=TraceMode.MESSAGES)
    engine = RolloutEngine({}, {"shell": ShellHarness(command_timeout=None)})
    grader = GradingResource(engine, {"task": spec}, "test-config")
    correct = await grader.grade(GradeCandidate("task", Candidate("12")))
    incorrect = await grader.grade(GradeCandidate("task", Candidate("9")))
    assert correct.status == Outcome.GRADED and correct.reward == 1
    assert incorrect.status == Outcome.GRADED and incorrect.reward == 0


async def test_grading_timeout_replays_candidate_on_new_attempt_once() -> None:
    timeout = GradeResult(Outcome.INFRA_ERROR, None, "timeout", failure=GradingFailure.TIMEOUT)
    success = GradeResult(Outcome.GRADED, 1, "correct")
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, success)))
    grader = GradingResource(engine, {"task": rollout_spec(arithmetic_task())}, "config")
    assert await grader.grade(GradeCandidate("task", Candidate("12"))) == success
    assert engine.replies == ["12", "12"]
    assert len(set(engine.run_ids)) == 2


async def test_grading_repeated_timeout_returns_the_second_grade() -> None:
    timeout = GradeResult(Outcome.INFRA_ERROR, None, "timeout", failure=GradingFailure.TIMEOUT)
    rollout = Rollout("trial", (), "stop")
    engine = FakeEngine((GradedRollout(rollout, timeout), GradedRollout(rollout, timeout)))
    grader = GradingResource(engine, {"task": rollout_spec(arithmetic_task())}, "config")
    assert await grader.grade(GradeCandidate("task", Candidate("12"))) == timeout
    assert len(engine.run_ids) == 2
