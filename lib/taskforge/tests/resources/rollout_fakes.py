# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted rollout engine and a fixed-answer model shared by the rollout, grading and verifier tests."""

from collections.abc import Sequence
from typing import override

from rolloutengine.models import GradedRollout, Model, ModelRequest, ModelTurn, RolloutInterrupted, TaskRolloutSpec

from taskforge.resources.rollouts import Engine


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
