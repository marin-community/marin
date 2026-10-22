# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task, model, and runtime helpers for RolloutEngine tests."""

import json
from dataclasses import dataclass, field

from shellbox.machine import NetworkPolicy
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.tools import ToolHarness
from rolloutengine.models import (
    GradedRollout,
    MachineRuntimeSpec,
    Model,
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TaskRolloutSpec,
)

TWELVE = NumericSpec("12", tolerance_abs=0, tolerance_rel=0)
SHELL = "shell"


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.continuation.served_token_ids, 90, 91) if request.continuation is not None else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


def arithmetic_task(**update) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "id": "arithmetic",
            "context": ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
            "environment_requirements": EnvironmentRequirements(),
            "answer_type": AnswerType.NUMBER,
            "answer_format": PlainText(),
            "grader": verifyit_package(TWELVE).grader,
            "source": Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            **update,
        }
    )


def machine_runtime(**overrides) -> MachineRuntimeSpec:
    return MachineRuntimeSpec.model_validate(
        {
            "backend": "local",
            "network": NetworkPolicy.DENY,
            "cpus": None,
            "memory_mb": None,
            "storage_mb": None,
            "gpus": 0,
            "user": None,
            "startup_timeout": None,
            "cleanup_timeout": None,
            **overrides,
        }
    )


def rollout_limits(**overrides) -> RolloutLimits:
    return RolloutLimits.model_validate(
        {
            "max_turns": 3,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": None,
            "attempt_timeout": None,
            "verifier_timeout": 5,
            "cleanup_timeout": 5,
            **overrides,
        }
    )


def rollout_spec(
    task: TaskSpec,
    *,
    harness=SHELL,
    machine=None,
    verifier_machine=None,
    trace=None,
    judge_connection=None,
    **limits,
) -> TaskRolloutSpec:
    return TaskRolloutSpec.model_validate(
        {
            "task": task,
            "harness": harness,
            "task_machine": machine,
            "verifier_machine": verifier_machine,
            "judge_connection": judge_connection,
            **({} if trace is None else {"trace": trace}),
            "limits": rollout_limits(**limits),
        }
    )


@dataclass(frozen=True)
class ModelEngine:
    """A rollout engine and the one model that a test runs through it."""

    engine: RolloutEngine
    model: Model

    def check(self, spec: TaskRolloutSpec) -> None:
        self.engine.check(spec)

    async def run_task(self, spec: TaskRolloutSpec, run_id: str | None = None) -> GradedRollout:
        return await self.engine.run_task(spec, self.model, run_id=run_id)


def engine(model, machine_factories=None, *, harnesses=None, observer=None) -> ModelEngine:
    """Run ``model.complete`` with the tool harness under ``SHELL``, unless ``harnesses`` replaces it."""
    rollout_engine = RolloutEngine(
        {} if machine_factories is None else machine_factories,
        {SHELL: ToolHarness(command_timeout=None), **({} if harnesses is None else harnesses)},
        observer=observer,
    )
    return ModelEngine(rollout_engine, model.complete)


def shell_call(command: str, *, call_id: str = "write") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }
