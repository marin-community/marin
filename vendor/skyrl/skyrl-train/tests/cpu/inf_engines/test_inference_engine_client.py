import asyncio
import base64
import io
import socket
from copy import deepcopy
from http import HTTPStatus

import numpy as np
import httpx
import pytest
import ray.exceptions
from jinja2 import TemplateError
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.base import CHOICE_POLICY_STEPS_KEY, InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    ErrorResponse,
    create_app,
    set_global_state,
)
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.inference_engines.utils import (
    _RENDEZVOUS_PORT_START,
    _RENDEZVOUS_PORT_STOP,
    _reserve_available_rendezvous_ports,
    get_vllm_sampling_params,
    hash_with_sha256,
    postprocess_completion_request,
)
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts

# (num_engines, num_prompts, with_session_ids): a single engine, an uneven even-split, session-id
# routing with repeated ids, and more engines than prompts.
ROUTING_CASES = [(1, 1, False), (3, 50, False), (4, 50, True), (16, 5, False)]


def _routing_session_ids(num_prompts: int) -> list[int]:
    # Repeated ids exercise session stickiness; the fixed modulus keeps the routing deterministic.
    return [i % 7 for i in range(num_prompts)]


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
                "inference_engine_data_parallel_size": 1,
                "run_engines_locally": True,
            },
        }
    )


def test_rendezvous_port_reservations_skip_busy_ports_and_hold_until_released(monkeypatch):
    monkeypatch.setattr("skyrl_train.inference_engines.utils.random.shuffle", lambda _ports: None)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", _RENDEZVOUS_PORT_START))
        reservations = _reserve_available_rendezvous_ports(2)
    ports = [reservation.getsockname()[1] for reservation in reservations]

    # The ports stay outside Linux's ephemeral client range and skip the port that had a listener.
    assert all(_RENDEZVOUS_PORT_START <= port < _RENDEZVOUS_PORT_STOP for port in ports)
    assert _RENDEZVOUS_PORT_START not in ports

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as contender, pytest.raises(OSError):
        contender.bind(("", ports[0]))

    for reservation in reservations:
        reservation.close()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as contender:
        contender.bind(("", ports[0]))


class _CommunicatorEngine:
    def __init__(self, relative_rank_offset=None, *, tp_size=1, pp_size=1):
        self.weight_sync_relative_rank_offset = relative_rank_offset
        self._tp_size = tp_size
        self._pp_size = pp_size
        self.received_rank_offset = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    async def init_weight_update_communicator(self, **kwargs):
        self.received_rank_offset = kwargs["rank_offset"]


@pytest.mark.parametrize(
    ("engines", "expected_offsets"),
    [
        ([_CommunicatorEngine(offset) for offset in (0, 0, 2, 2)], [1, 1, 3, 3]),
        ([_CommunicatorEngine(tp_size=2), _CommunicatorEngine(tp_size=2)], [1, 3]),
    ],
    ids=["logical-dp-engines", "legacy-sequential-engines"],
)
def test_weight_sync_communicator_rank_offsets(engines, expected_offsets):
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    asyncio.run(
        client.init_weight_update_communicator(
            master_addr="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=5,
            group_name="test",
            backend="nccl",
        )
    )

    assert [engine.received_rank_offset for engine in engines] == expected_offsets


@pytest.mark.parametrize(
    ("prompt", "session_id", "expected_session_ids", "expected_prompt"),
    [
        ("hello world", 123, [123], ["hello world"]),
        ([1, 2, 3], [8], [8], [[1, 2, 3]]),
        ([1, 2, 3], [8, 9], HTTPStatus.BAD_REQUEST, [[1, 2, 3]]),
        ([[1, 2], [3, 4, 5]], ["a", "b"], ["a", "b"], [[1, 2], [3, 4, 5]]),
        (["p0", "p1", "p2"], 10, HTTPStatus.BAD_REQUEST, ["p0", "p1", "p2"]),
        # A list of one string is a batch of one, not a single prompt to wrap.
        (["p0"], None, None, ["p0"]),
    ],
)
def test_postprocess_completion_request_normalizes_prompt_and_session_ids(
    prompt, session_id, expected_session_ids, expected_prompt
):
    session_ids, processed = postprocess_completion_request(prompt, session_id)

    assert processed == expected_prompt
    if expected_session_ids is HTTPStatus.BAD_REQUEST:
        assert isinstance(session_ids, ErrorResponse)
        assert session_ids.error.code == HTTPStatus.BAD_REQUEST.value
    else:
        assert session_ids == expected_session_ids


# -------------------------------------------
# tests for InferenceEngineClient.completion
# --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["stop", "abort"])
async def test_generate_single_selected_scores_fail_closed_on_abort(stop_reason):
    class ScoringEngine:
        async def generate(self, input_batch):
            candidate_ids = input_batch["sampling_params_per_prompt"][0]["prompt_logprob_token_ids"][1]
            return InferenceEngineOutput(
                responses=[""],
                response_ids=[[]],
                stop_reasons=[stop_reason],
                prompt_logprobs=[[None, {token_id: float(-token_id) for token_id in candidate_ids}]],
            )

    client = InferenceEngineClient(engines=[ScoringEngine()], tokenizer=object(), full_config=_make_min_cfg())
    request = InferenceEngineInput(
        prompt_token_ids=[[1, 2]],
        sampling_params={"prompt_logprobs": 2, "max_tokens": 1},
        sampling_params_per_prompt=[{"prompt_logprob_token_ids": [[1, 1], [7, 17]]}],
    )
    if stop_reason == "abort":
        with pytest.raises(RuntimeError, match="cannot continue"):
            await client.generate(request)
    else:
        output = await client.generate(request)
        assert output["prompt_logprobs"] == [[None, {7: -7.0, 17: -17.0}]]


@pytest.mark.parametrize(("num_engines", "num_prompts", "with_session_id"), ROUTING_CASES)
def test_completion_batched_routing_and_order_preservation(num_prompts, with_session_id, num_engines):
    class MockEngine:
        async def completion(self, request_payload):
            """
            Given input [i, j, k, ...], return output [f"{i}{i}", f"{j}{j}", f"{k}{k}", ...] with
            indices 0, 1, 2, 3, ...
            """
            body = request_payload["json"]
            my_prompts = body["prompt"]
            # Return per-sub-batch indices 0..len-1; client is expected to remap to global order
            choices = []
            for i, p in enumerate(my_prompts):
                choices.append(
                    {
                        "index": i,
                        "text": f"{p}{p}",
                        "finish_reason": "stop",
                    }
                )
            num_prompt_tokens = sum(len(p) for p in my_prompts)
            num_completion_tokens = num_prompt_tokens * 2  # since we doubled the prompts
            return {
                "id": "cmpl-mock",
                "object": "text_completion",
                "model": body.get("model", "dummy-model"),
                "choices": choices,
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "total_tokens": num_prompt_tokens + num_completion_tokens,
                    "completion_tokens": num_completion_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": num_prompt_tokens,
                    },
                },
            }

    cfg = _make_min_cfg()

    engines = [MockEngine() for _ in range(num_engines)]
    tokenizer = object()  # not used by completion()
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    prompts = [str(i) for i in range(num_prompts)]
    session_ids = _routing_session_ids(num_prompts) if with_session_id else None
    request_payload = {
        "json": {
            "model": "dummy-model",
            "prompt": prompts,
            "session_id": session_ids,
            "max_tokens": 32,
        },
        "headers": {"Content-Type": "application/json"},
    }

    resp = asyncio.run(client.completion(request_payload))

    assert resp.get("object") != "error"
    assert "choices" in resp and len(resp["choices"]) == len(prompts)
    # Ensure outputs align with inputs and indices are global order 0..n-1
    expected_texts = [f"{i}{i}" for i in range(num_prompts)]
    for i, choice in enumerate(resp["choices"]):
        assert choice["index"] == i
        assert choice["text"] == expected_texts[i]

    # also check usage aggregation here
    global_num_prompt_tokens = sum(len(p) for p in prompts)
    global_num_completion_tokens = global_num_prompt_tokens * 2  # since we doubled the prompts
    assert resp["usage"] == {
        "prompt_tokens": global_num_prompt_tokens,
        "total_tokens": global_num_prompt_tokens + global_num_completion_tokens,
        "completion_tokens": global_num_completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": global_num_prompt_tokens,
        },
    }


# -------------------------------------------
# tests for InferenceEngineClient.generate
# --------------------------------------------


@pytest.mark.parametrize(("num_engines", "num_prompts", "with_session_id"), ROUTING_CASES)
def test_generate_batched_routing_and_order_preservation(num_prompts, with_session_id, num_engines):
    class MockEngine:
        def __init__(self):
            self.inputs = []

        async def generate(self, input_batch):
            self.inputs.append(deepcopy(input_batch))
            # input_batch["prompt_token_ids"] is a local sub-batch list of token id lists
            prompt_token_ids = input_batch["prompt_token_ids"]
            responses = []
            response_ids = []
            stop_reasons = []
            routed_experts = []
            topk_indices = []
            prompt_logprobs = []
            for ids, scored in zip(prompt_token_ids, input_batch["sampling_params_per_prompt"], strict=True):
                # construct a deterministic text and token output based on first id
                base = ids[0]
                responses.append(f"{base}{base}")
                response_ids.append([base, base])
                stop_reasons.append("stop")
                routed_experts.append([[[base, base + 1]], [[base + 2, base + 3]]])
                topk_indices.append([[base, base + 2], [base + 1, base + 3]])
                # Score the candidate IDs this prompt's own per-prompt override asked for.
                prompt_logprobs.append(
                    [{token_id: float(-token_id) for token_id in scored["prompt_logprob_token_ids"][0]}]
                )
            return {
                "responses": responses,
                "response_ids": response_ids,
                "stop_reasons": stop_reasons,
                "routed_experts": routed_experts,
                "student_topk_indices": topk_indices,
                "behavior_topk_logprobs": [[[float(-i) for i in row] for row in rows] for rows in topk_indices],
                "prompt_logprobs": prompt_logprobs,
            }

    cfg = _make_min_cfg()

    engines = [MockEngine() for _ in range(num_engines)]
    tokenizer = object()  # not used when prompt_token_ids are provided
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    prompt_token_ids = [[i] for i in range(num_prompts)]
    session_ids = _routing_session_ids(num_prompts) if with_session_id else None

    input_batch = {
        "prompts": None,
        "prompt_token_ids": prompt_token_ids,
        "sampling_params": None,
        "sampling_params_per_prompt": [{"prompt_logprob_token_ids": [[7 + i, 17 + i]]} for i in range(num_prompts)],
        "session_ids": session_ids,
    }

    out = asyncio.run(client.generate(input_batch))

    # Validate reconstruction and ordering
    assert len(out["responses"]) == num_prompts
    assert len(out["response_ids"]) == num_prompts
    assert len(out["stop_reasons"]) == num_prompts
    expected_texts = [f"{i}{i}" for i in range(num_prompts)]
    for i in range(num_prompts):
        assert out["responses"][i] == expected_texts[i]
        assert out["response_ids"][i] == [i, i]
        assert out["stop_reasons"][i] == "stop"
        np.testing.assert_array_equal(out["routed_experts"][i], [[[i, i + 1]], [[i + 2, i + 3]]])
        assert out["student_topk_indices"][i] == [[i, i + 2], [i + 1, i + 3]]
        assert out["behavior_topk_logprobs"][i] == [[-i, -i - 2], [-i - 1, -i - 3]]
        assert out["prompt_logprobs"][i] == [{7 + i: float(-7 - i), 17 + i: float(-17 - i)}]
    if session_ids is not None:
        dispatched = [
            (engine_idx, session_id)
            for engine_idx, engine in enumerate(engines)
            for batch in engine.inputs
            for session_id in batch["session_ids"]
        ]
        assert sorted(session_id for _, session_id in dispatched) == sorted(session_ids)
        # Every prompt of a session goes to one engine, so later turns reuse that engine's prefix cache.
        engines_per_session = {
            session_id: len({engine_idx for engine_idx, sid in dispatched if sid == session_id})
            for session_id in set(session_ids)
        }
        assert engines_per_session == dict.fromkeys(set(session_ids), 1)


# -------------------------------------------
# tests for InferenceEngineClient.chat_completion retry logic
# --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "direct"])
@pytest.mark.parametrize("interrupted", [False, True])
async def test_chat_transports_preserve_route_tokens(transport, interrupted):
    """Ordinary Harbor requests must work when the server captures expert routes."""

    class Tokenizer:
        eos_token_id = 99

        def decode(self, ids, **kwargs):
            return "".join({11: "A", 12: "B"}[token] for token in ids)

    class Engine:
        def __init__(self):
            self.requests = []

        async def tokenize(self, request):
            return {"tokens": [1, 2]}

        async def chat_completion(self, request):
            self.requests.append(deepcopy(request))
            body = request["json"]
            continuation = body.get("continue_final_message", False)
            prompt_ids = body.get("_skyrl_exact_prompt_token_ids", [1, 2])
            response_ids = [12] if continuation else [11] if interrupted else [11, 12]
            values = [1, 2, 3] if continuation or not interrupted else [1, 2]
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(values, dtype=np.uint8).reshape(-1, 1, 1))
            choice = {
                "message": {"role": "assistant", "content": Tokenizer().decode(response_ids)},
                "finish_reason": "abort" if interrupted and not continuation else "stop",
                "routed_experts": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
            response = {
                "choices": [choice],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(response_ids),
                    "total_tokens": len(prompt_ids) + len(response_ids),
                },
            }
            # vLLM returns IDs only when requested, even when route capture is enabled.
            if body.get("return_token_ids"):
                response["prompt_token_ids"] = prompt_ids
                choice["token_ids"] = response_ids
            return response

    engine = Engine()
    backend = InferenceEngineClient([engine], Tokenizer(), _make_min_cfg())
    messages = [{"role": "user", "content": "Hi"}]
    if transport == "http":
        set_global_state(backend, None)
        payload = {"model": "dummy-model", "messages": messages, "max_tokens": 2}
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                response = await client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            choice = result["choices"][0]
            assert choice["message"]["content"] == "AB"
            assert result["prompt_token_ids"] == [1, 2]
            assert choice["token_ids"] == [11, 12]
            routes = normalize_routed_experts(choice["routed_experts"], [1, 2], [11, 12])
        finally:
            set_global_state(None, None)
    else:
        result = await DirectModelClient(backend).generate(
            {"prompts": [messages], "chat_completion_params": [{}], "sampling_params": {"max_tokens": 2}}
        )
        assert result["responses"] == ["AB"]
        assert result["prompt_ids"] == [[1, 2]]
        assert result["response_ids"] == [[11, 12]]
        routes = result["routed_experts"][0]
    np.testing.assert_array_equal(routes, [[[2]], [[3]]])
    if interrupted:
        assert engine.requests[-1]["json"]["_skyrl_exact_prompt_token_ids"] == [1, 2, 11]


@pytest.mark.asyncio
@pytest.mark.parametrize("exact_prompt", [None, [1, 2, 3, 4, 5]])
async def test_chat_completion_retry_accumulates_and_sends_continuations(exact_prompt):
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

    out = await client.chat_completion(original)

    assert len(engines[0].calls) == 3
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]
    third_call = engines[0].calls[2]

    assert first_call["json"] == {**original["json"], "return_token_ids": True}
    assert first_call["headers"] == original["headers"]
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
        assert call["json"]["messages"][-1] == {"role": "assistant", "content": "A"}
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
    assert len(choice["logprobs"]["content"]) == 2
    assert choice["logprobs"]["content"][0]["token"] == "token_id:11"
    assert choice["logprobs"]["content"][1]["token"] == "token_id:12"
    assert choice["token_ids"] == [11, 12]
    assert choice[CHOICE_POLICY_STEPS_KEY] == [7, 8]

    # usage: prompt_tokens from base (5), completion_tokens summed (2), total 7
    assert out["usage"]["prompt_tokens"] == 5
    assert out["usage"]["completion_tokens"] == 2
    assert out["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_chat_completion_retry_rejects_a_continuation_without_routes():
    def encoded_routes(values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values, dtype=np.uint8).reshape(-1, 1, 1))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    class Engine:
        def __init__(self):
            self.responses = [
                {
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "AB"},
                            "finish_reason": "abort",
                            "token_ids": [11, 12],
                            "routed_experts": encoded_routes([1, 2, 3]),
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                },
                {
                    "prompt_token_ids": [1, 2, 11, 12],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "C"},
                            "finish_reason": "stop",
                            "token_ids": [13],
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                },
            ]

        async def chat_completion(self, request):
            return deepcopy(self.responses.pop(0))

    client = InferenceEngineClient([Engine()], object(), _make_min_cfg())
    request = {"json": {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}}
    with pytest.raises(ValueError, match="routed_experts capture is incomplete"):
        await client.chat_completion(request)


@pytest.mark.asyncio
async def test_chat_completion_accepts_tool_call_response_without_text_content():
    """Tool parsers may return a tool-call-only message with nullable content."""

    response = {
        "id": "cmpl-tool-call",
        "object": "chat.completion",
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }

    class MockEngine:
        async def chat_completion(self, request_payload):
            return deepcopy(response)

    client = InferenceEngineClient(engines=[MockEngine()], tokenizer=object(), full_config=_make_min_cfg())
    request = {
        "json": {"model": "dummy-model", "messages": [{"role": "user", "content": "Run pwd"}]},
        "headers": {},
    }

    assert await client.chat_completion(request) == response


# -------------------------------------------
# tests for terminal-bench tokenization
# --------------------------------------------


@pytest.mark.asyncio
async def test_tokenize_surfaces_remote_chat_render_failure_as_template_error():
    template_error = TemplateError("assistant and tool roles are incompatible")
    validation_error = RuntimeError("chat template validation failed")
    validation_error.__cause__ = template_error

    class WrappedValidationError(RuntimeError):
        def as_instanceof_cause(self):
            return validation_error

    class RejectingEngine:
        async def tokenize(self, _request_payload):
            raise WrappedValidationError()

    client = InferenceEngineClient(engines=[RejectingEngine()], tokenizer=object(), full_config=_make_min_cfg())

    with pytest.raises(TemplateError, match="assistant and tool roles"):
        await client.tokenize({"json": {"messages": []}, "headers": {}})


# -------------------------------------------
# tests for InferenceEngineClient.generate retry logic
# --------------------------------------------


class _LetterTokenizer:
    """Decodes token ID 21 as "A", 22 as "B", and so on."""

    def decode(self, ids, **kwargs):
        return "".join(chr(ord("A") + token - 21) for token in ids)


@pytest.fixture(scope="module")
def vllm_sampling_params() -> dict:
    """The trainer's default vLLM sampling params, with a 10-token generation budget."""
    sampling = get_default_config().generator.sampling_params
    sampling.max_generate_length = 10
    return get_vllm_sampling_params(sampling)


@pytest.mark.parametrize("expert_offset", [0, 256], ids=["uint8-routes", "retry-crosses-uint8-limit"])
@pytest.mark.asyncio
async def test_generate_retry_some_gen_no_gen_finish(expert_offset, vllm_sampling_params):
    """
    Test that generate() with retry logic properly accumulates tokens and adjusts subsequent requests.

    First response aborts with tokens [21, 22]; second aborts with 0 tokens (ignored);
    third finishes with tokens [23, 24]. Assert:
    - Continuation requests append accumulated tokens to prompt_token_ids
    - remaining max_tokens decreases by accumulated tokens
    - Final response accumulates all tokens and uses last stop_reason
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture InferenceEngineInput calls
            # Pre-programmed responses
            self.responses = [
                # 1) abort with 2 tokens
                InferenceEngineOutput(
                    responses=["something"],
                    response_ids=[[21, 22]],
                    stop_reasons=["abort"],
                    response_logprobs=[[-0.1, -0.2]],
                    routed_experts=[[[[3, 4]], [[5, 6]]]],
                    response_policy_steps=[[1, 1]],
                ),
                # 2) abort with 0 tokens (should be ignored)
                InferenceEngineOutput(
                    responses=[""],
                    response_ids=[[]],
                    stop_reasons=["abort"],
                    response_logprobs=None,
                ),
                # 3) finish with 2 tokens
                InferenceEngineOutput(
                    responses=[" something"],  # will be ignored since we decode the final output
                    response_ids=[[23, 24]],
                    stop_reasons=["stop"],
                    response_logprobs=[[-0.3, -0.4]],
                    routed_experts=[np.array([[[7, 8]], [[9, 10]]], dtype=np.uint16) + expert_offset],
                    response_policy_steps=[[2, 2]],
                ),
            ]

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            idx = len(self.calls) - 1
            assert idx < len(self.responses), f"Unexpected extra call {idx}"
            return deepcopy(self.responses[idx])

    engines = [MockEngine()]
    client = InferenceEngineClient(engines=engines, tokenizer=_LetterTokenizer(), full_config=_make_min_cfg())

    # Original request
    prompt_token_ids = [[1, 2, 3, 4, 5]]
    sampling_params = dict(vllm_sampling_params)

    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        session_ids=["stable-session"],
    )

    out = await client.generate(input_batch)

    assert len(engines[0].calls) == 3
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]
    third_call = engines[0].calls[2]

    assert first_call["prompt_token_ids"] == [[1, 2, 3, 4, 5]]
    assert first_call["sampling_params"] == sampling_params

    # Continuations append the generated tokens and shrink the budget by them; the empty
    # second abort is ignored. Every other sampling field is carried over unchanged.
    for call in (second_call, third_call):
        assert call["prompt_token_ids"] == [[1, 2, 3, 4, 5, 21, 22]]
        assert call["sampling_params"] == {**sampling_params, "max_tokens": 8}
    assert [call["session_ids"] for call in engines[0].calls] == [["stable-session"]] * 3

    # Final response should accumulate all tokens, and its text is decoded from them.
    expected_final_response_ids = [21, 22, 23, 24]
    assert out["responses"] == ["ABCD"]
    assert out["response_ids"] == [expected_final_response_ids]
    assert out["stop_reasons"] == ["stop"]
    assert out["response_logprobs"] == [[-0.1, -0.2, -0.3, -0.4]]
    assert out["response_policy_steps"] == [[1, 1, 2, 2]]
    np.testing.assert_array_equal(
        out["routed_experts"][0],
        [[[3, 4]], [[5, 6]], [[7 + expert_offset, 8 + expert_offset]], [[9 + expert_offset, 10 + expert_offset]]],
    )
    assert out["routed_experts"][0].dtype == (np.uint16 if expert_offset else np.uint8)


# -------------------------------------------
# completion behavior at the weight-sync pause boundary
# --------------------------------------------


class _MockCompletionEngine:
    """Engine whose ``completion`` replays a scripted list of responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.entered = asyncio.Event()

    async def completion(self, request_payload):
        self.entered.set()
        self.calls.append(deepcopy(request_payload["json"]))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, BaseException):
            raise response
        return response

    async def pause_generation(self):
        pass

    async def resume_generation(self, policy_step):
        pass


class _MidflightPauseCompletionEngine(_MockCompletionEngine):
    """Holds the first request in flight until the scheduler pauses, which aborts it."""

    def __init__(self):
        super().__init__(
            [
                _completion_response("partial", "abort", completion_tokens=2),
                _completion_response("complete answer", "stop"),
            ]
        )
        self.scheduler_paused = asyncio.Event()

    async def completion(self, request_payload):
        if not self.calls:
            self.entered.set()
            self.calls.append(deepcopy(request_payload["json"]))
            await self.scheduler_paused.wait()
            return self.responses[0]
        return await super().completion(request_payload)

    async def pause_generation(self):
        self.scheduler_paused.set()


def _completion_response(text, finish_reason, *, completion_tokens=None):
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "model": "dummy-model",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": completion_tokens if completion_tokens is not None else len(text),
            "total_tokens": 4 + (completion_tokens if completion_tokens is not None else len(text)),
        },
    }


def _completion_payload(prompt, **extra):
    return {
        "json": {"model": "dummy-model", "prompt": prompt, "max_tokens": 32, **extra},
        "headers": {"Content-Type": "application/json"},
    }


# The two single-prompt shapes: a raw string, and the flat token-id list harbor's TITO
# transport sends. A list OF strings is batched even at length 1 (see
# `postprocess_completion_request`), which is why the shapes are pinned here.
SINGLE_PROMPTS = [pytest.param("hello", id="string"), pytest.param([1, 2, 3, 4], id="token_ids")]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", SINGLE_PROMPTS)
async def test_completion_single_prompt_blocks_while_paused_then_resumes(prompt):
    engines = [_MockCompletionEngine([_completion_response("done", "stop")])]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())
    await client.pause_generation()

    task = asyncio.create_task(client.completion(_completion_payload(prompt, session_id="trial-7")))
    # A parked request never runs on a bare yield, so one that was going to reach the engine
    # would do so within these turns.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not engines[0].entered.is_set(), "request reached the engine while generation was paused"
    assert not task.done()

    await client.resume_generation(policy_step=1)
    result = await asyncio.wait_for(task, timeout=5)

    assert result["choices"][0]["text"] == "done"
    assert len(engines[0].calls) == 1
    assert "session_id" not in engines[0].calls[0], "session_id must be stripped before it reaches the engine"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected_finish_reason", "expected_text"),
    [
        pytest.param(
            [
                _completion_response("partial", "abort", completion_tokens=2),
                _completion_response("complete answer", "stop"),
            ],
            "stop",
            "complete answer",
            id="retry-completes",
        ),
        pytest.param(
            [_completion_response("partial", "abort", completion_tokens=2)],
            "abort",
            "partial",
            id="second-abort-returned",
        ),
    ],
)
async def test_completion_single_prompt_reissues_once_after_a_midflight_abort(
    responses, expected_finish_reason, expected_text
):
    """A pause that begins mid-flight comes back as finish_reason "abort" (the abort-mode
    scheduler pause returns what was generated so far). /completions cannot continue a
    partial generation the way chat can, so the request is re-issued from the start, once.
    One retry, not a loop: a re-issue restarts generation, so retrying until success could
    livelock a long generation under frequent weight syncs. A second abort goes back to the caller."""
    engines = [_MockCompletionEngine(responses)]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())
    prompt = [1, 2, 3, 4]

    result = await client.completion(_completion_payload(prompt))

    assert result["choices"][0]["finish_reason"] == expected_finish_reason
    assert result["choices"][0]["text"] == expected_text
    assert len(engines[0].calls) == 2
    # The re-issue restarts from the ORIGINAL prompt (no continuation protocol here), and
    # every other sampling field is carried over unchanged.
    for call in engines[0].calls:
        assert call["prompt"] == [prompt]
        assert call["max_tokens"] == 32


@pytest.mark.asyncio
async def test_completion_single_prompt_reissue_waits_for_a_pause_that_is_still_on():
    engine = _MidflightPauseCompletionEngine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())

    task = asyncio.create_task(client.completion(_completion_payload([1, 2, 3, 4])))
    await asyncio.wait_for(engine.entered.wait(), timeout=5)
    await client.pause_generation()
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(engine.calls) == 1, "the retry entered the engine while generation was paused"
    assert not task.done()

    await client.resume_generation(policy_step=1)
    result = await asyncio.wait_for(task, timeout=5)

    assert result["choices"][0]["finish_reason"] == "stop"
    assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_completion_pause_retry_stays_on_failover_engine():
    engines = [
        _MockCompletionEngine([ray.exceptions.RayActorError()]),
        _MockCompletionEngine(
            [
                _completion_response("partial", "abort", completion_tokens=2),
                _completion_response("complete", "stop"),
            ]
        ),
    ]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())
    session_id = next(sid for sid in ("trial-0", "trial-1") if hash_with_sha256(sid) % 2 == 0)

    result = await client.completion(_completion_payload([1, 2, 3, 4], session_id=session_id))

    assert result["choices"][0]["text"] == "complete"
    assert len(engines[0].calls) == 1
    assert len(engines[1].calls) == 2


@pytest.mark.asyncio
async def test_completion_single_prompt_engine_error_is_wrapped_not_retried():
    """An engine error payload carries no usable `choices`; it is not an abort, so it is
    returned as the client-level error response without a second attempt."""
    error_payload = {
        "object": "error",
        "message": "This model's maximum context length is 32768 tokens",
        "type": HTTPStatus.BAD_REQUEST.phrase,
        "code": HTTPStatus.BAD_REQUEST.value,
    }
    engines = [_MockCompletionEngine([error_payload])]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    result = await client.completion(_completion_payload([1, 2, 3, 4]))

    assert result["error"]["code"] == HTTPStatus.BAD_REQUEST.value
    assert "maximum context length" in result["error"]["message"]
    assert len(engines[0].calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_engine", "send"),
    [
        pytest.param(
            lambda: _MockCompletionEngine([_completion_response("done", "stop")]),
            lambda client: client.completion(_completion_payload(["hello"])),
            id="completion-one-string-is-still-batched",
        ),
        pytest.param(
            lambda: _MockCompletionEngine([_completion_response("done", "stop")]),
            lambda client: client.completion(_completion_payload([["a"], ["b"]])),
            id="completion-batched",
        ),
        pytest.param(
            lambda: _MockGenerateEngine(),
            lambda client: client.generate(
                InferenceEngineInput(prompt_token_ids=[[1, 2], [3, 4]], sampling_params={"max_tokens": 5})
            ),
            id="generate-batched",
        ),
    ],
)
async def test_batched_completion_and_generate_raise_while_paused(make_engine, send):
    """Batched /completions and batched generate() fan out across engines with no per-sub-request
    pause handling, so they reject a request that arrives during a pause before it reaches an engine.
    A list of ONE string is batched by `postprocess_completion_request`, and stays on that path here too."""
    engine = make_engine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())
    await client.pause_generation()

    with pytest.raises(RuntimeError, match="unsupported for batched"):
        await send(client)

    assert not engine.entered.is_set()


# -------------------------------------------
# generate() at the weight-sync pause boundary
# --------------------------------------------


class _MockGenerateEngine:
    """Engine whose ``generate`` records every request and answers with a fixed completion."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.requests = []
        self.scheduler_paused = False

    async def generate(self, request):
        assert not self.scheduler_paused, "request reached the engine while its scheduler was paused"
        self.entered.set()
        self.requests.append(deepcopy(request))
        return InferenceEngineOutput(
            responses=["answer"],
            response_ids=[[21, 22, 23]],
            stop_reasons=["stop"],
            response_logprobs=[[-0.1, -0.2, -0.3]],
        )

    async def pause_generation(self):
        self.scheduler_paused = True

    async def resume_generation(self, policy_step):
        self.scheduler_paused = False


@pytest.mark.asyncio
async def test_generate_single_prompt_waits_for_resume_then_reaches_engine():
    """A single-prompt generate() that arrives during a weight-sync pause is held, not rejected,
    and reaches the engine unchanged once generation resumes."""
    engine = _MockGenerateEngine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())
    # Simulate the pause directly: the engine fan-out is not what this test exercises.
    client.generation_paused_event.set()
    engine.scheduler_paused = True

    request = InferenceEngineInput(
        prompt_token_ids=[[1, 2, 3]], sampling_params={"max_tokens": 5, "temperature": 0}, session_ids=["prompt-0"]
    )
    task = asyncio.create_task(client.generate(request))
    # A paused waiter never runs on a bare yield; if the request were sent (or rejected) it
    # would happen within these turns of the loop.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done(), "generate() returned or raised while generation was paused"
    assert engine.requests == []

    await client.resume_generation(policy_step=1)
    output = await asyncio.wait_for(task, timeout=5)

    assert engine.requests == [
        {
            "prompt_token_ids": [[1, 2, 3]],
            "sampling_params": {"max_tokens": 5, "temperature": 0},
            "session_ids": ["prompt-0"],
        }
    ]
    assert output["response_ids"] == [[21, 22, 23]]
    assert output["responses"] == ["answer"]
    assert output["stop_reasons"] == ["stop"]
    assert output["response_logprobs"] == [[-0.1, -0.2, -0.3]]


class _DeadEngine(_MockGenerateEngine):
    """Engine whose actor has died: every call fails the way a dead Ray actor does."""

    async def generate(self, request):
        raise ray.exceptions.RayActorError()

    async def pause_generation(self):
        raise ray.exceptions.RayActorError()


async def _fail_over_engine_zero(client: InferenceEngineClient) -> None:
    """Send a session routed to engine 0 so its actor error marks engine 0 dead."""
    session_id = next(
        sid for sid in (f"trial-{i}" for i in range(10)) if hash_with_sha256(sid) % len(client.engines) == 0
    )
    await client.generate(
        InferenceEngineInput(prompt_token_ids=[[1, 2, 3]], sampling_params={"max_tokens": 5}, session_ids=[session_id])
    )


@pytest.mark.asyncio
async def test_weight_sync_pause_and_exactness_check_skip_an_engine_that_already_died():
    """A pause and the exactness check's generation must not wait on, or fail because of, an engine the client
    already knows is dead."""
    dead, live = _DeadEngine(), _MockGenerateEngine()
    client = InferenceEngineClient(engines=[dead, live], tokenizer=object(), full_config=_make_min_cfg())
    await _fail_over_engine_zero(client)
    assert len(live.requests) == 1

    await client.pause_generation()
    assert live.scheduler_paused
    await client.resume_generation(policy_step=1)
    assert not live.scheduler_paused

    outputs = await client.generate_on_live_engines([1, 2, 3], {"max_tokens": 5})
    assert [rank for rank, _ in outputs] == [0]


class _DraftGenerateEngine(_MockGenerateEngine):
    def __init__(self):
        super().__init__()
        self.draft_updates = []

    async def update_draft_weights(self, weights_path):
        self.draft_updates.append(weights_path)
        return {"active": True}


class _DeadDraftEngine(_DraftGenerateEngine):
    async def generate(self, request):
        raise ray.exceptions.RayActorError()


class _FailingDraftEngine(_DraftGenerateEngine):
    async def update_draft_weights(self, weights_path):
        raise RuntimeError("load failed")


@pytest.mark.asyncio
async def test_draft_refresh_skips_dead_engines_and_reports_failed_loads() -> None:
    dead, live, failing = _DeadDraftEngine(), _DraftGenerateEngine(), _FailingDraftEngine()
    client = InferenceEngineClient(engines=[dead, live, failing], tokenizer=object(), full_config=_make_min_cfg())
    await _fail_over_engine_zero(client)

    weights_path = "s3://bucket/drafts/draft-step-4/model.safetensors"
    coverage = await client.update_draft_weights(weights_path)

    # One live engine's failed load is reported in its own entry instead of failing the refresh.
    assert coverage == [{"active": True}, {"active": False, "error": "RuntimeError: load failed"}]
    assert dead.draft_updates == []
    assert live.draft_updates == [weights_path]


class _NeverAnsweringEngine(_MockGenerateEngine):
    """Engine whose pause RPC hangs, as a wedged scheduler's would."""

    async def pause_generation(self):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_weight_sync_pause_fails_when_an_engine_never_acknowledges():
    cfg = _make_min_cfg()
    cfg.generator.weight_sync_pause_timeout_seconds = 0.01
    client = InferenceEngineClient(engines=[_NeverAnsweringEngine()], tokenizer=object(), full_config=cfg)

    with pytest.raises(TimeoutError, match="weight_sync_pause_timeout_seconds"):
        await client.pause_generation()


class _MockStreamEngine:
    """Minimal engine whose ``chat_completion_stream`` records when it is entered."""

    def __init__(self):
        self.entered = asyncio.Event()

    async def pause_generation(self):
        pass

    async def resume_generation(self, policy_step):
        pass

    async def chat_completion_stream(self, request_payload):
        self.entered.set()
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield "data: [DONE]\n\n"


def test_resume_wakes_a_request_parked_on_another_event_loop():
    """The HTTP endpoint serves requests on its own event loop; a request parked there during a
    pause must be released by a resume issued from the trainer's loop. Two loops are driven by
    hand in one thread so the request is provably parked before the resume."""
    engines = [_MockStreamEngine()]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())
    payload = {"json": {"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]}, "headers": {}}
    trainer_loop, server_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
    try:
        trainer_loop.run_until_complete(client.pause_generation())

        async def consume():
            return [chunk async for chunk in client.chat_completion_stream(payload)]

        request = server_loop.create_task(consume())
        # One iteration of the server loop runs the request up to the pause barrier and no further.
        server_loop.run_until_complete(asyncio.sleep(0))
        assert not request.done()
        assert not engines[0].entered.is_set(), "request reached the engine while generation was paused"

        trainer_loop.run_until_complete(client.resume_generation(policy_step=1))
        chunks = server_loop.run_until_complete(asyncio.wait_for(request, timeout=5))
    finally:
        server_loop.close()
        trainer_loop.close()

    assert engines[0].entered.is_set()
    assert any("[DONE]" in chunk for chunk in chunks)
