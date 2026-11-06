# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay a candidate through the solver's machines and verifier, without inference."""

import base64
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar

from rolloutengine.models import ModelRequest, ModelTurn, RolloutInterrupted, TaskRolloutSpec, TraceMode
from taskcompendium.grading_result import GradeResult, GradingFailure
from taskcompendium.models import AssistantToolCalls, TextMessage
from taskcompendium.runtime.resources import resource_bytes

from forge.effects.events import Event, Fingerprint, ReplayPolicy, event_digest
from forge.model.controls import Candidate
from forge.resources.execution import Notify, retry_effect
from forge.resources.rollouts import Engine, interruption
from forge.resources.sandbox.images import DEFAULT_MODE


@dataclass(frozen=True, slots=True)
class GradeCandidate(Event[GradeResult]):
    result_type: ClassVar[object] = GradeResult
    spec: TaskRolloutSpec
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class ScriptedModel:
    turns: tuple[Mapping[str, object], ...]
    context_assistant_turns: int

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages) - self.context_assistant_turns
        if served >= len(self.turns):
            raise RuntimeError("Candidate script exhausted before rollout ended")
        message = self.turns[served]
        return ModelTurn(dict(message), (), (), None, "tool_calls" if "tool_calls" in message else "stop")


def candidate_turns(candidate: Candidate, spec: TaskRolloutSpec) -> tuple[Mapping[str, object], ...]:
    turns: list[Mapping[str, object]] = []
    used = {
        call.call_id
        for event in spec.task.context.events
        if isinstance(event, AssistantToolCalls)
        for call in event.calls
    }
    for index, file in enumerate(sorted(candidate.files, key=lambda file: file.path)):
        call_id = f"file-{index}"
        while call_id in used:
            call_id += "-"
        target = PurePosixPath("/", file.path)
        path, parent = shlex.quote(str(target)), shlex.quote(str(target.parent))
        data = shlex.quote(base64.b64encode(resource_bytes(file)).decode())
        command = (
            f"mkdir -p {parent} && printf %s {data} | base64 -d > {path} && chmod {file.mode or DEFAULT_MODE} {path}"
        )
        turns.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                    }
                ],
            }
        )
    return (*turns, {"role": "assistant", "content": candidate.reply})


@dataclass(frozen=True, slots=True)
class GradingResource:
    engine: Engine
    config_digest: str

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("grading", self.config_digest, ReplayPolicy.CACHE)

    async def grade(self, event: GradeCandidate) -> GradeResult:
        spec = event.spec
        if event.candidate.files and spec.task_machine is None:
            raise ValueError("Candidate files require a task machine")
        turns = candidate_turns(event.candidate, spec)
        count = sum(
            isinstance(message, AssistantToolCalls) or (isinstance(message, TextMessage) and message.role == "assistant")
            for message in spec.task.context.events
        )
        model = ScriptedModel(turns, count)
        spec = spec.model_copy(
            update={"trace": TraceMode.MESSAGES, "limits": spec.limits.model_copy(update={"max_turns": len(turns)})}
        )
        for attempt in range(2):
            try:
                result = await self.engine.run_task(spec, model, run_id=f"{event_digest(event)}:grade:{attempt}")
            except RolloutInterrupted as error:
                raise interruption(error) from error
            if result.grade.failure != GradingFailure.TIMEOUT or attempt == 1:
                return result.grade
        raise RuntimeError("Candidate grading produced no result")


async def grade(event: GradeCandidate, resource: GradingResource, *, notify: Notify) -> GradeResult:
    return await retry_effect(event, lambda: resource.grade(event), notify)
