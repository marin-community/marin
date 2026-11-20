# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted receipts attest returned token stamps and fully validated replay."""

import asyncio
import hashlib
import json
import struct
from collections import Counter
from copy import deepcopy

import pytest
from loguru import logger
from omegaconf import OmegaConf

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
from skyrl_train.weight_sync.expert_block.schedule import Schedule


@pytest.fixture
def receipt_logs():
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _receipts(messages, kind):
    return [json.loads(message[len(kind) + 1 :]) for message in messages if message.startswith(f"{kind} ")]


def _client(engine):
    config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": "test-model"}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 30,
            },
        }
    )
    return InferenceEngineClient([engine], object(), config)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["generate", "chat_completion"])
async def test_generation_receipt_preserves_engine_tokens_and_bounds_stamp_histogram(transport, receipt_logs):
    tokens = list(range(8205))
    steps = [-1] + [index % 20 + 1 for index in range(len(tokens) - 1)]

    class Engine:
        async def generate(self, _request):
            return {
                "responses": ["private response"],
                "response_ids": [tokens],
                "response_policy_steps": [steps],
                "stop_reasons": ["stop"],
            }

        async def chat_completion(self, _request):
            return {
                "id": "request-identity",
                "prompt_token_ids": [7, 8],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "private response"},
                        "finish_reason": "stop",
                        "token_ids": tokens,
                        "policy_steps": steps,
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": len(tokens), "total_tokens": len(tokens) + 2},
            }

    client = _client(Engine())
    if transport == "generate":
        result = await client.generate({"prompt_token_ids": [[7, 8]], "sampling_params": {}})
        assert result["response_ids"] == [tokens]
        assert result["response_policy_steps"] == [steps]
    else:
        result = await client.chat_completion({"json": {"messages": [{"role": "user", "content": "private prompt"}]}})
        assert result["choices"][0]["token_ids"] == tokens
        assert result["choices"][0]["policy_steps"] == steps
    (receipt,) = _receipts(receipt_logs, "policy_generation_receipt")
    counts = Counter(steps)
    assert receipt["response_count"] == 1
    assert receipt["token_count"] == receipt["policy_token_count"] == len(tokens)
    assert receipt["stamped_token_count"] == len(tokens) - 1
    assert receipt["aligned"] is True
    assert len(receipt["policy_step_counts"]) == 16
    assert receipt["policy_step_histogram_complete"] is False
    assert receipt["other_policy_step_tokens"] == sum(
        count for step, count in counts.items() if str(step) not in receipt["policy_step_counts"]
    )
    for step, count in receipt["policy_step_counts"].items():
        assert count == counts[int(step)]
    assert (
        receipt["token_ids_sha256"]
        == hashlib.sha256(struct.pack(f"<Q{len(tokens)}q", len(tokens), *tokens)).hexdigest()
    )
    assert (
        receipt["policy_steps_sha256"]
        == hashlib.sha256(struct.pack(f"<Q{len(steps)}q", len(steps), *steps)).hexdigest()
    )
    assert "private" not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_generation_receipt_aggregates_actual_policy_stamps_across_pause_retry(receipt_logs):
    replies = [
        {
            "prompt_token_ids": [7, 8],
            "choices": [
                {
                    "message": {"role": "assistant", "content": "first"},
                    "finish_reason": "abort",
                    "token_ids": [11, 12],
                    "policy_steps": [2, 2],
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
        {
            "prompt_token_ids": [7, 8, 11, 12],
            "choices": [
                {
                    "message": {"role": "assistant", "content": "second"},
                    "finish_reason": "stop",
                    "token_ids": [13],
                    "policy_steps": [3],
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        },
    ]

    class Engine:
        async def chat_completion(self, _request):
            return deepcopy(replies.pop(0))

    result = await _client(Engine()).chat_completion({"json": {"messages": [{"role": "user", "content": "prompt"}]}})
    assert result["choices"][0]["token_ids"] == [11, 12, 13]
    assert result["choices"][0]["policy_steps"] == [2, 2, 3]
    (receipt,) = _receipts(receipt_logs, "policy_generation_receipt")
    assert receipt["token_count"] == receipt["stamped_token_count"] == 3
    assert receipt["policy_step_counts"] == {"2": 2, "3": 1}
    assert receipt["policy_step_histogram_complete"] is True
    assert receipt["aligned"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatched", [False, True])
async def test_replay_receipts_require_every_receiver_to_pass_validation(mismatched, receipt_logs):
    class Policy:
        def async_run_ray_method(self, *_args):
            async def reply():
                return {"replicas": {"participant": 0, "version": 1, "compared_bytes": 0, "mismatched_bytes": 0}}

            return [reply()]

    class Engines:
        generation_paused_event = asyncio.Event()

        async def expert_block_rpc(self, *_args):
            return [
                [
                    {
                        "participant": 1,
                        "version": 1,
                        "compared_bytes": 16,
                        "parameter_bytes": 16,
                        "mismatched_bytes": 0,
                    },
                    {
                        "participant": 2,
                        "version": 1,
                        "compared_bytes": 32,
                        "parameter_bytes": 32,
                        "mismatched_bytes": int(mismatched),
                    },
                ]
            ]

    engines = Engines()
    engines.generation_paused_event.set()
    sync = ExpertBlockSync(policy_model=Policy(), inference_engine_client=engines, timeout_seconds=10)
    sync.schedule = Schedule(1, (), (), (), ((1, 16), (2, 32)), ((1, 0), (2, 0)))
    if mismatched:
        with pytest.raises(RuntimeError, match="replayed bytes differ"):
            await sync.verify(1)
        assert _receipts(receipt_logs, "expert_block_replay_receipt") == []
    else:
        result = await sync.verify(1)
        assert result["verify_seconds"] >= 0
        receipts = _receipts(receipt_logs, "expert_block_replay_receipt")
        assert {row["participant"]: row["compared_bytes"] for row in receipts} == {1: 16, 2: 32}
        assert all(row["version"] == 1 and row["mismatched_bytes"] == 0 for row in receipts)
        assert all(row["compared_bytes"] == row["parameter_bytes"] for row in receipts)
