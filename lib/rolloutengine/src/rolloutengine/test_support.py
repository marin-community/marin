# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task, model, and runtime helpers for RolloutEngine tests."""

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

from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec

TWELVE = NumericSpec("12", tolerance_abs=0, tolerance_rel=0)


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
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


def lowered(task: TaskSpec, *, machine=None, verifier_machine=None, **limits) -> LoweredTaskSpec:
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=machine, verifier_machine=verifier_machine),
        session=TaskSessionSpec.model_validate(
            {
                "task_session": "shellbox",
                "max_turns": 3,
                "model_turn_timeout": None,
                "command_timeout": None,
                "tool_turn_timeout": None,
                "total_turn_timeout": None,
                "attempt_timeout": None,
                "verifier_timeout": 5,
                "cleanup_timeout": 5,
                "max_total_tokens": None,
                **limits,
            }
        ),
    )


def engine(model, factories=None, *, sessions=None, providers=None) -> RolloutEngine:
    return RolloutEngine(model.complete, {} if factories is None else factories, sessions=sessions, providers=providers)
