"""Native score evidence survives routing, chat adaptation and aborted continuations."""

import base64
import io
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf
from skyrl_train.inference_engines.base import CHOICE_POLICY_STEPS_KEY, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.response_topk import CHAT_TOPK_WIDTH_KEY, CHOICE_TOPK_KEY, ResponseTopK
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.selected_topk import align_student_topk


@pytest.mark.parametrize("mask,expected_ids,expected_scores", [
    ([1, 1], [[3, 7], [4, 8]], [[-1.0, -2.0], [-3.0, -4.0]]),
    ([1, 0, 1], [[3, 7], [-1, -1], [4, 8]], [[-1.0, -2.0], [0.0, 0.0], [-3.0, -4.0]]),
])
def test_numeric_heads_remain_aligned_across_masked_response_positions(mask, expected_ids, expected_scores):
    response_ids = [11, 12] if len(mask) == 2 else [11, 99, 12]
    ids = np.array([[3, 7], [4, 8]], dtype=np.int32)
    scores = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    result = align_student_topk(response_ids, mask, [11, 12], ids, scores)
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_array_equal(result.topk_logprobs, expected_scores)
    assert align_student_topk(response_ids, mask, [12, 11], ids, scores) is None


def _make_min_cfg():
    return OmegaConf.create(
        {
            "trainer": {
                "policy": {"model": {"path": "dummy-model"}},
            },
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 30.0,
            },
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("num_engines", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 3])
async def test_generate_preserves_response_topk_across_engine_routing(num_engines, num_prompts):
    class TopKEngine:
        async def generate(self, input_batch):
            bases = [row[0] for row in input_batch["prompt_token_ids"]]
            return InferenceEngineOutput(
                responses=[str(base) for base in bases],
                response_ids=[[base, base + 1] for base in bases],
                stop_reasons=["stop"] * len(bases),
                response_logprobs=[[-0.1, -0.2] for _ in bases],
                student_topk_indices=[[[base, base + 2], [base + 1, base + 3]] for base in bases],
                behavior_topk_logprobs=[[[-0.1, -1.1], [-0.2, -1.2]] for _ in bases],
            )

    client = InferenceEngineClient(
        engines=[TopKEngine() for _ in range(num_engines)],
        tokenizer=object(),
        full_config=_make_min_cfg(),
    )
    output = await client.generate(
        InferenceEngineInput(prompt_token_ids=[[base] for base in range(num_prompts)], sampling_params={"logprobs": 2})
    )

    np.testing.assert_array_equal(
        output["student_topk_indices"], [[[base, base + 2], [base + 1, base + 3]] for base in range(num_prompts)]
    )
    np.testing.assert_array_equal(
        output["behavior_topk_logprobs"],
        np.asarray([[[-0.1, -1.1], [-0.2, -1.2]] for _ in range(num_prompts)], dtype=np.float32),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exact_prompt", [None, [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("compact", [False, True])
async def test_chat_completion_retry_accumulates_and_sends_continuations(exact_prompt, compact):
    """
    First response aborts with tokens; second aborts with 0 tokens (ignored);
    third finishes. Assert:
    - Continuation requests append accumulated assistant content with correct role
    - continue_final_message/add_generation_prompt flags are set
    - remaining max_tokens decreases by accumulated completion tokens
    - Final response accumulates content, logprobs, token_ids and recomputes usage correctly
    - Each retry request is what we expect the engine to receive
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture full request payloads {"json":..., "headers":...}
            # Pre-programmed partial responses
            self.responses = [
                # 1) abort with 1 token "A"
                {
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A"},
                            "finish_reason": "abort",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:11",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:11", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [11],
                            CHOICE_POLICY_STEPS_KEY: [7],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
                # 2) abort with 0 tokens (should be ignored for accumulation)
                {
                    "id": "cmpl-2",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "abort",
                            "logprobs": {"content": []},
                            "token_ids": [],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
                },
                # 3) finish with 1 token "B"
                {
                    "id": "cmpl-3",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "B"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:12",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:12", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [12],
                            CHOICE_POLICY_STEPS_KEY: [8],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            ]

        async def chat_completion(self, request_payload):
            self.calls.append(deepcopy(request_payload))
            idx = len(self.calls) - 1
            assert idx < len(self.responses), f"Unexpected extra call {idx}"
            return deepcopy(self.responses[idx])

    engines = [MockEngine()]
    if compact:
        for response in engines[0].responses:
            choice = response["choices"][0]
            tokens = choice["token_ids"]
            choice["logprobs"] = None
            choice[CHOICE_TOPK_KEY] = ResponseTopK(
                [-0.1] * len(tokens),
                np.asarray([[token, token - 8] for token in tokens], dtype=np.int32).reshape(-1, 2),
                np.asarray([[-0.1, -0.2]] * len(tokens), dtype=np.float32).reshape(-1, 2),
            )
    cfg = _make_min_cfg()
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)

    original = {
        "json": {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            # ask for structures that the client can accumulate
            "logprobs": True,
            "top_logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
        "headers": {"Content-Type": "application/json"},
    }
    if exact_prompt is not None:
        original["json"]["_skyrl_exact_prompt_token_ids"] = exact_prompt
    if compact:
        original[CHAT_TOPK_WIDTH_KEY] = 2

    out = await client.chat_completion(original)

    assert len(engines[0].calls) == 3
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]
    third_call = engines[0].calls[2]

    # First call should be identical to original json (no continuation flags)
    assert first_call["json"] == {**original["json"], "return_token_ids": True}
    assert first_call["headers"] == original["headers"]
    if compact:
        assert all(call[CHAT_TOPK_WIDTH_KEY] == 2 for call in engines[0].calls)
    assert first_call["json"].get("continue_final_message") is None
    assert first_call["json"].get("add_generation_prompt") is None
    assert first_call["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert first_call["json"]["max_tokens"] == 8

    # Second/third calls should be continuation requests
    for call in (second_call, third_call):
        if exact_prompt is not None:
            assert call["json"]["_skyrl_exact_prompt_token_ids"] == [1, 2, 3, 4, 5, 11]
        assert call["headers"] == original["headers"]
        # Flags
        assert call["json"].get("continue_final_message") is True
        assert call["json"].get("add_generation_prompt") is False
        # Accumulated assistant message appended with content "A"
        assert call["json"]["messages"][-1] == {"role": "assistant", "content": "A"}
        # Original user message preserved
        assert call["json"]["messages"][0] == {"role": "user", "content": "Hi"}
        # Remaining max_tokens reduced by 1 (we already generated one token)
        assert call["json"].get("max_tokens") == 7
        # Other params preserved
        assert call["json"]["model"] == "dummy-model"
        assert call["json"]["logprobs"] is True
        assert call["json"]["top_logprobs"] == 1
        assert call["json"]["return_tokens_as_token_ids"] is True

    # Final response should accumulate content/logprobs/token_ids and usage
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "AB"
    if compact:
        assert choice["logprobs"] is None
        selected = choice[CHOICE_TOPK_KEY]
        assert selected.sampled_logprobs == [-0.1, -0.1]
        np.testing.assert_array_equal(selected.topk_indices, [[11, 3], [12, 4]])
        np.testing.assert_allclose(selected.topk_logprobs, [[-0.1, -0.2]] * 2, rtol=0, atol=1e-7)
    else:
        assert len(choice["logprobs"]["content"]) == 2
        assert choice["logprobs"]["content"][0]["token"] == "token_id:11"
        assert choice["logprobs"]["content"][1]["token"] == "token_id:12"
    assert choice["token_ids"] == [11, 12]
    assert choice[CHOICE_POLICY_STEPS_KEY] == [7, 8]

    # usage: prompt_tokens from base (5), completion_tokens summed (2), total 7
    assert out["usage"]["prompt_tokens"] == 5
    assert out["usage"]["completion_tokens"] == 2
    assert out["usage"]["total_tokens"] == 7


def _encoded_routes(rows):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(rows, dtype=np.uint16), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_direct_chat_client_captures_exact_student_topk_ids():
    engine = AsyncMock()
    engine.model_name = "teacher"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [9, 10],
                "provider_specific_fields": {"routed_experts": _encoded_routes([[[1, 2]], [[3, 4]], [[4, 7]]])},
                "logprobs": None,
                CHOICE_TOPK_KEY: ResponseTopK(
                    [-7.0, -0.1],
                    np.asarray([[2, 3], [10, 11]], dtype=np.int32),
                    np.asarray([[-0.1, -0.2], [-0.1, -0.2]], dtype=np.float32),
                ),
            }
        ]
    }

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "session_ids": ["test"],
            "sampling_params": {"logprobs": 2},
            "chat_completion_params": [{}],
        }
    )

    body = engine.chat_completion.await_args.args[0]["json"]
    assert body["top_logprobs"] == 3
    assert body["return_tokens_as_token_ids"] is True
    assert engine.chat_completion.await_args.args[0][CHAT_TOPK_WIDTH_KEY] == 2
    np.testing.assert_array_equal(output["student_topk_indices"], [[[2, 3], [10, 11]]])
    np.testing.assert_allclose(output["behavior_topk_logprobs"], [[[-0.1, -0.2], [-0.1, -0.2]]], rtol=0, atol=1e-7)
    assert output["response_logprobs"] == [[-7.0, -0.1]]
    np.testing.assert_array_equal(output["routed_experts"][0], [[[3, 4]], [[4, 7]]])
    assert output["routed_experts"][0].dtype == np.uint8
