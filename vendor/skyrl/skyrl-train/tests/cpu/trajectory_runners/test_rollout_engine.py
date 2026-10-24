import json
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, ConversationInput, PlainText, Source, TaskSpec, TextMessage
from verifyit.spec import NumericSpec

from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.base import UNKNOWN_ENGINE_DP_RANK
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.types import (
    ENGINE_DP_RANKS_KEY,
    ROLLOUT_ENGINE_DP_RANKS_KEY,
    TaskRequestBatch,
    TokenProvenance,
    TrajectoryID,
)
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, _messages, **_kwargs):
        return [10, 11]


@dataclass
class ReplayClient:
    """Answers model turn ``i`` with ``turns[i]``, served by the engine of data-parallel rank ``ranks[i]``."""

    turns: list
    ranks: list
    requests: list = field(default_factory=list)

    async def generate(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        continuation = request["chat_continuations"][0]
        prompt = [10, 11] if continuation is None else [*continuation["served_prefix_token_ids"], 90, 91]
        return {
            "prompt_ids": [prompt],
            "response_ids": [[20 + index]],
            "response_logprobs": [[-0.5]],
            "responses": [self.turns[index].get("content", "")],
            "assistant_messages": [self.turns[index]],
            "stop_reasons": ["stop"],
            "token_provenance": TokenProvenance.ENGINE,
            ENGINE_DP_RANKS_KEY: [self.ranks[index]],
        }


def answer_task_rollout():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=SHELL_ENVIRONMENT,
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    return lowered_task(task, max_turns=3, model_turn_timeout=None, attempt_timeout=10, backend="shellsim")


async def rollout_engine_ranks(client):
    config = OmegaConf.merge(
        get_default_config().generator,
        {
            "backend": "vllm",
            "max_input_length": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {"logprobs": True, "max_generate_length": 32},
        },
    )
    factories = {"shellsim": ShellSimMachineFactory()}
    worker = TaskRolloutWorker(config, WholeTaskProjection(config, Tokenizer()), client, factories, shutdown_timeout=30)
    request = TaskRequestBatch(
        rows=(task_input(answer_task_rollout()),),
        sampling_params={"logprobs": True},
        trajectory_ids=(TrajectoryID("task", 0),),
    )
    output = await worker.run(request)
    return output[ROLLOUT_ENGINE_DP_RANKS_KEY]


@pytest.mark.asyncio
async def test_task_rows_carry_the_serving_engine_rank_or_unknown_when_turns_differ():
    answer = {"role": "assistant", "content": "12"}
    assert await rollout_engine_ranks(ReplayClient([answer], ranks=[1])) == [1]

    # A shell call leads to a second model turn, which another engine serves.
    shell_call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": "true"})},
            }
        ],
    }
    assert await rollout_engine_ranks(ReplayClient([shell_call, answer], ranks=[1, 0])) == [UNKNOWN_ENGINE_DP_RANK]
