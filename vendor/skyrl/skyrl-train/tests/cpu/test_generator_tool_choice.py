"""generator.tool_choice reaches every chat rollout request, with and without row tools."""

import json

import pytest
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import source_row_task
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID

from tests.cpu.task_specs import task_input

SESSION = {
    "max_turns": 1,
    "max_total_tokens": None,
    "model_turn_timeout": None,
    "tool_turn_timeout": None,
    "total_turn_timeout": None,
    "attempt_timeout": None,
    "verifier_timeout": None,
    "cleanup_timeout": 60.0,
}
TOOLS = [{"type": "function", "function": {"name": "get_price", "parameters": {"type": "object"}}}]
EOS = 99


def row(tools: bool) -> dict:
    return {
        "prompt": [{"role": "user", "content": "Price of Apple?"}],
        "env_class": "prompt_only",
        "chat_template_kwargs": json.dumps({"enable_thinking": False}),
        "tools": json.dumps(TOOLS) if tools else None,
        "data_source": "fixture",
    }


class Tokenizer:
    eos_token_id = EOS

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(token) for token in ids)


class Engine:
    model_name = "student"
    tokenizer = Tokenizer()

    def __init__(self):
        self.tokenize_bodies = []
        self.chat_bodies = []

    async def tokenize(self, request):
        self.tokenize_bodies.append(request["json"])
        return {"tokens": [1, 2]}

    async def chat_completion(self, request):
        self.chat_bodies.append(request["json"])
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "150"},
                    "token_ids": [11, 12, EOS],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}, {"logprob": -0.3}]},
                }
            ]
        }


def generator_config(tool_choice):
    return OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 1,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "chat_template_kwargs": {},
            "tool_choice": tool_choice,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": None,
            },
        }
    )


def test_base_config_declares_tool_choice_unset():
    assert get_default_config().generator.tool_choice is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [True, False])
@pytest.mark.parametrize("tool_choice", ["none", None])
async def test_generator_tool_choice_reaches_render_and_chat_requests(tools, tool_choice):
    config = generator_config(tool_choice)
    spec = source_row_task(row(tools), 0, source_name="fixture", environment_configs={"session": SESSION, "prompt_only": {}})
    request = TaskRequestBatch(rows=(task_input(spec),), trajectory_ids=(TrajectoryID("row", 0),))
    engine = Engine()
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(config, Tokenizer()), DirectModelClient(engine), {}, shutdown_timeout=30
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()

    assert len(engine.chat_bodies) == len(engine.tokenize_bodies) == 1
    for body in (engine.tokenize_bodies[0], engine.chat_bodies[0]):
        if tool_choice is None:
            assert "tool_choice" not in body
        else:
            assert body["tool_choice"] == tool_choice
        assert body.get("tools") == (TOOLS if tools else None)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.response_token_ids == (11, 12, EOS)
