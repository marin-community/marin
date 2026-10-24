"""Native task execution through the existing SkyRL training boundary."""

import json
from dataclasses import dataclass, field

import numpy as np
import pytest
from omegaconf import OmegaConf
from rolloutengine.models import MachineRuntimeSpec
from skyrl_gym.rollout_specs import rollout_spec
from shellbox.machine import NetworkPolicy
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

from skyrl_gym.verification import VerificationStatus
from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.types import TokenProvenance, TrajectoryID, TaskRequestBatch
from tests.cpu.task_specs import task_input


pytestmark = pytest.mark.asyncio


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, _messages, **_kwargs):
        return [10, 11]


@dataclass
class ReplayClient:
    turns: list
    provenance: TokenProvenance = TokenProvenance.ENGINE
    requests: list = field(default_factory=list)

    async def generate(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        message = self.turns[index]
        if isinstance(message, Exception):
            raise message
        continuation = request["chat_continuations"][0]
        prompt = [10, 11] if continuation is None else [*continuation["served_prefix_token_ids"], 90, 91]
        tokens = [20 + index]
        return {
            "prompt_ids": [prompt],
            "response_ids": [tokens],
            "response_logprobs": [[-0.5]],
            "responses": [message.get("content", "")],
            "assistant_messages": [message],
            "stop_reasons": ["stop"],
            "token_provenance": self.provenance,
            "student_topk_indices": [[[tokens[0], 50]]],
            "behavior_topk_logprobs": [[[-0.5, -2.0]]],
            "routed_experts": [np.asarray([[[index + 1, 2]]], dtype=np.uint8)],
            "engine_dp_ranks": [1 - index % 2],
        }


def answer_task():
    return TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def lowered(task):
    machine = MachineRuntimeSpec(
        backend="fixture",
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )
    return rollout_spec(
        task,
        dict(
            task_session="shellbox",
            max_turns=3,
            model_turn_timeout=None,
            command_timeout=1,
            tool_turn_timeout=5,
            total_turn_timeout=None,
            attempt_timeout=10,
            verifier_timeout=5,
            cleanup_timeout=5,
            max_total_tokens=None,
        ),
        task_machine=machine if task.answer_type == AnswerType.FILE else None,
        verifier_machine=machine if isinstance(task.grader, ScriptGrader) else None,
    )


def runner(client, factories=None, *, error_handling=None):
    config = OmegaConf.merge(
        get_default_config().generator,
        {
            "backend": "vllm",
            "max_input_length": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {"logprobs": True, "max_generate_length": 32},
            "error_handling": error_handling or {},
        },
    )
    return TaskRolloutWorker(
        config,
        WholeTaskProjection(config, Tokenizer()),
        client,
        factories or {},
        shutdown_timeout=30,
    )


def request(record):
    return TaskRequestBatch(
        rows=(task_input(record, teacher_route="math"),),
        sampling_params={"logprobs": True},
        trajectory_ids=(TrajectoryID("task", 0),),
    )


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0), ("not a number", 0.0)])
async def test_native_answer_returns_trainable_grade_and_exact_evidence(answer, reward):
    record = lowered(answer_task())
    client = ReplayClient([{"role": "assistant", "content": answer}])
    output = await runner(client).run(request(record))

    assert output["rewards"] == [reward]
    assert output["verification_results"][0].status == VerificationStatus.VERIFIED
    assert output["prompt_token_ids"] == [[10, 11]]
    assert output["response_ids"] == [[20]]
    assert output["loss_masks"] == [[1]]
    assert output["exclude_from_baseline"] == [False]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.5])
    np.testing.assert_array_equal(output["student_topk_indices"][0], [[20, 50]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]]])
    assert output["rollout_engine_dp_ranks"] == [1]
    assert output["trajectory_ids"] == [TrajectoryID("task", 0)]
    assert output["teacher_route_keys"] == ["math"]
    assert output["data_sources"] == ["fixture"]
    assert "12" not in json.dumps(client.requests[0]["prompts"])
