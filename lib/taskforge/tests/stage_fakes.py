# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from rolloutengine.models import (
    GradedRollout,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
)
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from taskforge.effects.context import Context
from taskforge.effects.issues import Hold, Restart
from taskforge.harness.protocol import Harness, Tool
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Conversation
from taskforge.resources.env import EnvFactory, FakeEnv
from taskforge.resources.inference import InferenceClient
from taskforge.resources.rollouts import Engine


class Environments(EnvFactory):
    def __init__(self) -> None:
        self.created: list[FakeEnv] = []
        self.requested: list[EnvironmentRequirements] = []

    @override
    async def create(self, requirements: EnvironmentRequirements) -> FakeEnv:
        self.requested.append(requirements)
        env = FakeEnv(requirements.model_dump_json(), {}, lambda event: None)
        self.created.append(env)
        return env


class RecordingHarness(Harness):
    def __init__(self, result: Exit | Restart | Hold | ExceptionGroup) -> None:
        self.result = result
        self.session: Conversation | None = None
        self.client: InferenceClient | None = None

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        self.session = session
        self.client = client
        if isinstance(self.result, (Restart, Hold, ExceptionGroup)):
            raise self.result
        return self.result


class SolverEngine(Engine):
    """Release trials only when the entire wave has started; a serial stage deadlocks."""

    def __init__(self, count: int, restart: int | None = None) -> None:
        self.count, self.restart = count, restart
        self.started: list[str] = []
        self.ready = asyncio.Event()

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        self.started.append(run_id)
        if len(self.started) == self.count:
            self.ready.set()
        await self.ready.wait()
        trial = int(run_id.split(":")[-2])
        rollout = Rollout(run_id, (), "stop")
        if trial == self.restart:
            raise RolloutInterrupted(rollout, RolloutOperation.MODEL)
        return GradedRollout(rollout, GradeResult(Outcome.GRADED, float(trial % 2 == 0)))


@dataclass(frozen=True, slots=True)
class CandidateEngine(Engine):
    result: GradeResult | None = None

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        turn = await model(ModelRequest((), {}))
        return GradedRollout(
            Rollout(run_id or "grade", (), "stop"),
            self.result or GradeResult(Outcome.GRADED, float(turn.message["content"] == "12")),
        )


async def solver_model(request: ModelRequest) -> ModelTurn:
    return ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")
