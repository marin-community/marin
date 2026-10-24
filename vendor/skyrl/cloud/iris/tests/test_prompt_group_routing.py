# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.base import SESSION_ID_HEADER
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from skyrl_gym.source_task import source_task
from taskcompendium.models import Source
from tests.cpu.task_specs import lowered_task, task_input

SAMPLES_PER_PROMPT = 4
ENGINES = 4


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]

    def decode(self, token_ids, **options):
        return "done"


class Engine:
    def __init__(self, index: int, served: list[tuple[str, int]]):
        self.index = index
        self.served = served

    async def generate(self, request):
        self.served.append((request["session_ids"][0], self.index))
        return {
            "responses": ["done"],
            "prompt_ids": request["prompt_token_ids"],
            "response_ids": [[11, 99]],
            "response_logprobs": [[-0.1, -0.2]],
            "stop_reasons": ["stop"],
        }

    async def tokenize(self, request):
        return {"tokens": [1, 2]}

    async def chat_completion(self, request):
        self.served.append((request["headers"][SESSION_ID_HEADER], self.index))
        return {
            "id": "completion",
            "object": "chat.completion",
            "model": request["json"]["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "token_ids": [11, 99],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"token": "done", "logprob": -0.1, "top_logprobs": []}] * 2},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["assistant_prefix", "prompt_only"], ids=["token-request", "chat-request"])
async def test_a_prompt_groups_samples_are_served_by_one_engine(harness):
    runner_config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 1,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": 0,
            },
        }
    )
    client_config = get_default_config()
    client_config.generator.enable_http_endpoint = False
    served: list[tuple[str, int]] = []
    client = InferenceEngineClient([Engine(index, served) for index in range(ENGINES)], Tokenizer(), client_config)
    worker = TaskRolloutWorker(
        runner_config,
        WholeTaskProjection(runner_config, Tokenizer()),
        DirectModelClient(client),
        {},
        shutdown_timeout=30,
    )
    messages = [{"role": "user", "content": "Finish"}]
    if harness == "assistant_prefix":
        messages.append({"role": "assistant", "content": "Almost"})
    task = source_task(messages, {}, {}, Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, harness, max_turns=1)),) * SAMPLES_PER_PROMPT,
        trajectory_ids=tuple(TrajectoryID("group", sample) for sample in range(SAMPLES_PER_PROMPT)),
    )
    try:
        await worker.generate(request)
    finally:
        await worker.shutdown()

    assert sorted(session for session, _ in served) == [item.to_string() for item in request.trajectory_ids]
    assert len({engine for _, engine in served}) == 1
