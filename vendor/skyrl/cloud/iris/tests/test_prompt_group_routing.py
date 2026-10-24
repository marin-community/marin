# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from taskcompendium.models import ConversationInput, TextMessage
from tests.cpu.task_specs import lowered_task, task_input, text_task

SAMPLES_PER_PROMPT = 4
ENGINES = 4


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]


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


@pytest.mark.asyncio
async def test_a_prompt_groups_samples_are_served_by_one_engine():
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
                "logprobs": True,
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
    task = text_task("group").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="Finish"), TextMessage(role="assistant", content="Almost"))
            )
        }
    )
    row = task_input(lowered_task(task, "assistant_prefix", max_turns=1))
    request = TaskRequestBatch(
        rows=(row,) * SAMPLES_PER_PROMPT,
        trajectory_ids=tuple(TrajectoryID("group", sample) for sample in range(SAMPLES_PER_PROMPT)),
    )
    try:
        await worker.generate(request)
    finally:
        await worker.shutdown()

    assert sorted(session for session, _ in served) == [f"group_{sample}" for sample in range(SAMPLES_PER_PROMPT)]
    assert len({engine for _, engine in served}) == 1
