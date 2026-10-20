"""
Test for `skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` functinoalities
that can be mocked. Also tests for `skyrl-train/skyrl_train/inference_engines/utils.py`.

Run with:
uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_inference_engine_client.py
"""

import asyncio
import base64
import io
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http import HTTPStatus
from unittest.mock import patch

import numpy as np
import httpx
import pytest
import ray.exceptions
import requests
from jinja2 import TemplateError
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base
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
    _find_available_rendezvous_port,
    _reserve_available_rendezvous_ports,
    get_vllm_sampling_params,
    hash_with_sha256,
    postprocess_completion_request,
    route_prompts_to_engines,
)
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from transformers import AutoTokenizer

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
            },
        }
    )


def test_rendezvous_port_avoids_ephemeral_range_and_existing_listener(monkeypatch):
    first = _RENDEZVOUS_PORT_START
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", first))
        monkeypatch.setattr("skyrl_train.inference_engines.utils.random.shuffle", lambda _ports: None)
        second = _find_available_rendezvous_port()

    assert _RENDEZVOUS_PORT_START <= second < _RENDEZVOUS_PORT_STOP
    assert second == first + 1


def test_rendezvous_port_fails_when_range_is_excluded():
    with pytest.raises(RuntimeError, match="No free rendezvous port"):
        _find_available_rendezvous_port(range(_RENDEZVOUS_PORT_START, _RENDEZVOUS_PORT_STOP))


def test_rendezvous_port_reservations_hold_ports_until_released(monkeypatch):
    monkeypatch.setattr("skyrl_train.inference_engines.utils.random.shuffle", lambda _ports: None)
    reservations = _reserve_available_rendezvous_ports(2)
    ports = [reservation.getsockname()[1] for reservation in reservations]

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
        ("hello world", None, None, ["hello world"]),
        ("hello world", 123, [123], ["hello world"]),
        ("hello world", ["abc"], ["abc"], ["hello world"]),
        ("hello world", [1, 2], HTTPStatus.BAD_REQUEST, ["hello world"]),
        ([1, 2, 3], None, None, [[1, 2, 3]]),
        ([1, 2, 3], 7, [7], [[1, 2, 3]]),
        ([1, 2, 3], [8], [8], [[1, 2, 3]]),
        ([1, 2, 3], [8, 9], HTTPStatus.BAD_REQUEST, [[1, 2, 3]]),
        ([[1, 2], [3, 4, 5]], None, None, [[1, 2], [3, 4, 5]]),
        ([[1, 2], [3, 4, 5]], ["a", "b"], ["a", "b"], [[1, 2], [3, 4, 5]]),
        ([[1, 2], [3, 4, 5]], [1], HTTPStatus.BAD_REQUEST, [[1, 2], [3, 4, 5]]),
        (["p0", "p1"], None, None, ["p0", "p1"]),
        (["p0", "p1", "p2"], [10, 11, 12], [10, 11, 12], ["p0", "p1", "p2"]),
        (["p0", "p1", "p2"], [10, 11], HTTPStatus.BAD_REQUEST, ["p0", "p1", "p2"]),
        (["p0", "p1", "p2"], 10, HTTPStatus.BAD_REQUEST, ["p0", "p1", "p2"]),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("num_engines", [1, 2])
async def test_generate_preserves_per_prompt_selected_scores_across_engine_routing(num_engines):
    class ScoringEngine:
        async def generate(self, input_batch):
            rows = input_batch["sampling_params_per_prompt"]
            return InferenceEngineOutput(
                responses=[""] * len(rows),
                response_ids=[[] for _ in rows],
                stop_reasons=["stop"] * len(rows),
                prompt_logprobs=[
                    [None, {token_id: float(-token_id) for token_id in row["prompt_logprob_token_ids"][1]}]
                    for row in rows
                ],
            )

    client = InferenceEngineClient(
        engines=[ScoringEngine() for _ in range(num_engines)],
        tokenizer=object(),
        full_config=_make_min_cfg(),
    )
    output = await client.generate(
        InferenceEngineInput(
            prompt_token_ids=[[1, 2], [3, 4], [5, 6]],
            sampling_params={"prompt_logprobs": 2, "max_tokens": 1},
            sampling_params_per_prompt=[
                {"prompt_logprob_token_ids": [[1, 1], [7 + row, 17 + row]]} for row in range(3)
            ],
        )
    )

    assert output["prompt_logprobs"] == [
        [None, {7 + row: float(-7 - row), 17 + row: float(-17 - row)}] for row in range(3)
    ]


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

    assert output["student_topk_indices"] == [[[base, base + 2], [base + 1, base + 3]] for base in range(num_prompts)]
    assert output["behavior_topk_logprobs"] == [[[-0.1, -1.1], [-0.2, -1.2]] for _ in range(num_prompts)]


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
            for ids in prompt_token_ids:
                # construct a deterministic text and token output based on first id
                base = ids[0]
                responses.append(f"{base}{base}")
                response_ids.append([base, base])
                stop_reasons.append("stop")
                routed_experts.append([[[base, base + 1]], [[base + 2, base + 3]]])
            return {
                "responses": responses,
                "response_ids": response_ids,
                "stop_reasons": stop_reasons,
                "routed_experts": routed_experts,
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
    if session_ids is not None:
        observed = [session_id for engine in engines for batch in engine.inputs for session_id in batch["session_ids"]]
        assert sorted(observed) == sorted(session_ids)


# -----------------------------
# Test for route_prompts_to_engines function that routes prompts to inference engines
# in inference engine client.
# -------------------------------


def test_route_prompts_to_engines_single_prompt_no_trajectory_random_engine():
    with patch("random.randint", return_value=1):
        mapping = route_prompts_to_engines(num_prompts=1, num_inference_engines=4, session_ids=None)
    assert mapping == {1: [0]}


def test_route_prompts_to_engines_batched_even_split_exact_multiple():
    # 4 prompts, 2 engines => [0,1] and [2,3]
    num_prompts = 4
    num_engines = 2
    mapping = route_prompts_to_engines(num_prompts=num_prompts, num_inference_engines=num_engines, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3]}


def test_route_prompts_to_engines_batched_uneven_split():
    # 5 prompts, 2 engines => ceil(5/2)=3 => [0,1,2] and [3,4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=2, session_ids=None)
    assert mapping == {0: [0, 1, 2], 1: [3, 4]}

    # 5 prompts, 3 engines => ceil(5/3)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=3, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    # 5 prompts, 4 engines => ceil(5/4)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    mapping = route_prompts_to_engines(num_prompts=129, num_inference_engines=4, session_ids=None)
    assert mapping == {0: list(range(33)), 1: list(range(33, 66)), 2: list(range(66, 99)), 3: list(range(99, 129))}


def test_route_prompts_to_engines_batched_more_engines_than_prompts():
    # 2 prompts, 4 engines => size=1 => {0:[0], 1:[1]}
    mapping = route_prompts_to_engines(num_prompts=2, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0], 1: [1]}


def test_route_prompts_to_engines_with_session_ids_grouping_and_partition():
    num_engines = 4
    # Same session IDs route to the same engine; the mapping pins the sha256-based assignment.
    sids = ["A", "A", "B", "C", "B"]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=num_engines, session_ids=sids)

    assert mapping == {1: [0, 1, 3], 0: [2, 4]}


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


class _DraftUpdateEngine:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def update_draft_weights(self, weights_path):
        self.calls.append(weights_path)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_draft_refresh_retains_per_engine_exceptions() -> None:
    engines = [
        _DraftUpdateEngine({"active": True}),
        _DraftUpdateEngine(RuntimeError("load failed")),
    ]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    weights_path = "s3://bucket/drafts/draft-step-4/model.safetensors"
    coverage = await client.update_draft_weights(weights_path)

    assert coverage == [
        {"active": True},
        {"active": False, "error": "RuntimeError: load failed"},
    ]
    assert [engine.calls for engine in engines] == [[weights_path], [weights_path]]


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
@pytest.mark.parametrize("drop_second_routes", [False, True])
async def test_chat_completion_retry_keeps_routes_across_interrupted_chunks(drop_second_routes):
    def encoded_routes(values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values, dtype=np.uint8).reshape(-1, 1, 1))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    class Engine:
        def __init__(self):
            self.requests = []
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
                            "routed_experts": encoded_routes([1, 2, 3, 4]),
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                },
            ]

        async def chat_completion(self, request):
            self.requests.append(deepcopy(request))
            response = deepcopy(self.responses.pop(0))
            if drop_second_routes and len(self.requests) == 2:
                response["choices"][0].pop("routed_experts")
            return response

    engine = Engine()
    client = InferenceEngineClient([engine], object(), _make_min_cfg())
    request = {"json": {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}}
    if drop_second_routes:
        with pytest.raises(ValueError, match="routed_experts capture is incomplete"):
            await client.chat_completion(request)
        return
    response = await client.chat_completion(request)
    choice = response["choices"][0]

    assert choice["token_ids"] == [11, 12, 13]
    assert engine.requests[1]["json"]["_skyrl_exact_prompt_token_ids"] == [1, 2, 11, 12]
    np.testing.assert_array_equal(
        normalize_routed_experts(choice["routed_experts"], response["prompt_token_ids"], choice["token_ids"]),
        [[[2]], [[3]], [[4]]],
    )


@pytest.mark.asyncio
async def test_chat_completion_retry_resends_original_when_no_tokens_generated_yet():
    """
    First response aborts with 0 tokens, so the next request should resend the original
    payload unchanged. Second response finishes; client returns it directly.
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture full payloads
            self.responses = [
                # 1) abort with 0 tokens
                {
                    "id": "cmpl-a1",
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
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
                },
                # 2) finish with tokens "XYZ" (3 tokens)
                {
                    "id": "cmpl-a2",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "XYZ"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:21",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:21", "logprob": -0.1, "bytes": [116]}],
                                    },
                                    {
                                        "token": "token_id:22",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:22", "logprob": -0.1, "bytes": [116]}],
                                    },
                                    {
                                        "token": "token_id:23",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:23", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [21, 22, 23],
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            ]

        async def chat_completion(self, request_payload):
            self.calls.append(deepcopy(request_payload))
            return deepcopy(self.responses[len(self.calls) - 1])

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)

    original = {
        "json": {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
        "headers": {"Content-Type": "application/json"},
    }

    out = await client.chat_completion(original)

    assert len(engines[0].calls) == 2
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]

    # After 0-token abort, the next call should resend the original unchanged
    assert first_call["json"] == {**original["json"], "return_token_ids": True}
    assert second_call["json"] == first_call["json"]
    assert first_call["headers"] == original["headers"]
    assert second_call["headers"] == original["headers"]
    assert first_call["json"].get("continue_final_message") is None
    assert second_call["json"].get("continue_final_message") is None

    # Since finish_reason != abort on the second call and base_response was None,
    # client should return the second response directly (no accumulation)
    assert out == engines[0].responses[1]


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


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def vllm_sampling_params() -> dict:
    """The trainer's default vLLM sampling params, with a 10-token generation budget."""
    sampling = get_default_config().generator.sampling_params
    sampling.max_generate_length = 10
    return get_vllm_sampling_params(sampling)


@pytest.mark.parametrize("max_tokens_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("expert_offset", [0, 256], ids=["uint8-routes", "retry-crosses-uint8-limit"])
@pytest.mark.asyncio
async def test_generate_retry_some_gen_no_gen_finish(max_tokens_key, expert_offset, tokenizer, vllm_sampling_params):
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
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=_make_min_cfg())

    # Original request
    prompt_token_ids = [[1, 2, 3, 4, 5]]
    sampling_params = dict(vllm_sampling_params)
    sampling_params[max_tokens_key] = sampling_params.pop("max_tokens")

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
        assert call["sampling_params"] == {**sampling_params, max_tokens_key: 8}
    assert [call["session_ids"] for call in engines[0].calls] == [["stable-session"]] * 3

    # Final response should accumulate all tokens
    expected_final_response_ids = [21, 22, 23, 24]
    expected_final_text_response = tokenizer.decode(expected_final_response_ids, skip_special_tokens=True)
    assert out["responses"] == [expected_final_text_response]
    assert out["response_ids"] == [expected_final_response_ids]
    assert out["stop_reasons"] == ["stop"]
    assert out["response_logprobs"] == [[-0.1, -0.2, -0.3, -0.4]]
    assert out["response_policy_steps"] == [[1, 1, 2, 2]]
    np.testing.assert_array_equal(
        out["routed_experts"][0],
        [[[3, 4]], [[5, 6]], [[7 + expert_offset, 8 + expert_offset]], [[9 + expert_offset, 10 + expert_offset]]],
    )
    assert out["routed_experts"][0].dtype == (np.uint16 if expert_offset else np.uint8)


@pytest.mark.asyncio
async def test_generate_retry_direct_return():
    """
    Test that if the first generate() request doesn't abort, it returns directly without retries.
    """

    class MockEngine:
        def __init__(self):
            self.calls = []
            # Single response that completes immediately
            self.response = InferenceEngineOutput(
                responses=["something"],
                response_ids=[[21, 22, 23, 24]],
                stop_reasons=["stop"],
                response_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            )

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            return deepcopy(self.response)

    engines = [MockEngine()]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    prompt_token_ids = [[1, 2, 3, 4, 5]]
    sampling_params = {"max_tokens": 10}

    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    out = await client.generate(input_batch)

    assert len(engines[0].calls) == 1

    # Verify response is returned as-is
    expected_final_response_ids = [21, 22, 23, 24]
    # note since we completed in one turn, we return the text response of the first turn returned by
    # the underlying engine instead re-tokenizing like others
    assert out["responses"] == ["something"]
    assert out["response_ids"] == [expected_final_response_ids]
    assert out["stop_reasons"] == ["stop"]
    assert out["response_logprobs"] == [[-0.1, -0.2, -0.3, -0.4]]


@pytest.mark.asyncio
async def test_generate_retry_no_gen_finish():
    """
    First response aborts with 0 tokens; next finishes.
    The second request should resend the original unchanged and the final output equals the second response.
    """
    final_response_ids = [21, 22, 23]

    class MockEngine:
        def __init__(self):
            self.calls = []
            self.responses = [
                # 1) abort with 0 tokens
                InferenceEngineOutput(
                    responses=[""],
                    response_ids=[[]],
                    stop_reasons=["abort"],
                    response_logprobs=[[]],
                ),
                # 2) finish directly
                InferenceEngineOutput(
                    responses=["something"],
                    response_ids=[final_response_ids],
                    stop_reasons=["stop"],
                    response_logprobs=[[-0.1, -0.1, -0.1]],
                ),
            ]

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            return deepcopy(self.responses[len(self.calls) - 1])

    engines = [MockEngine()]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    original_prompt_ids = [7, 8, 9]
    input_batch = InferenceEngineInput(
        prompt_token_ids=[original_prompt_ids],
        sampling_params={"max_tokens": 16},
    )

    out = await client.generate(input_batch)

    # Two calls should have been made, identical inputs since first had 0 tokens
    assert len(engines[0].calls) == 2
    first_call, second_call = engines[0].calls
    assert first_call["prompt_token_ids"] == [original_prompt_ids]
    assert second_call["prompt_token_ids"] == [original_prompt_ids]
    assert first_call["sampling_params"]["max_tokens"] == 16
    assert second_call["sampling_params"]["max_tokens"] == 16

    assert out == {**engines[0].responses[1], "prompt_logprobs": None}


# -------------------------------------------
# tests for InferenceEngineClient.chat_completion_stream weight-sync boundary guard
# --------------------------------------------


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


class _MockWeightSyncEngine:
    def __init__(self):
        self.scheduler_paused = False
        self.outstanding_requests = 388
        self.reloads = 0

    async def pause_generation(self):
        self.scheduler_paused = True
        self.outstanding_requests = 0

    async def update_named_weights(self, **_request):
        if not self.scheduler_paused or self.outstanding_requests:
            raise RuntimeError("reshape_and_cache_flash attempted to run with Meta tensors")
        self.reloads += 1

    async def resume_generation(self, policy_step):
        self.scheduler_paused = False


@pytest.mark.asyncio
async def test_weight_sync_pauses_loaded_scheduler_until_reload_finishes():
    engine = _MockWeightSyncEngine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())

    await client.pause_generation()
    await client.update_named_weights(request={"names": ["model.weight"]})

    assert engine.scheduler_paused
    assert engine.outstanding_requests == 0
    assert engine.reloads == 1

    await client.resume_generation(policy_step=1)
    assert not engine.scheduler_paused


@pytest.mark.asyncio
async def test_chat_completion_stream_blocks_while_paused_then_resumes():
    """Regression for the vLLM meta-tensor EngineDeadError at the weight-sync boundary.

    A NEW stream must not reach the engine while generation is paused for a weight
    sync — otherwise it would register a fresh request in the vLLM scheduler during
    the pause -> reload -> resume window (after the scheduler pause drained the engine)
    and the next engine step would run a forward pass against meta-device params.
    The streaming path must honor the same ``generation_paused_event`` barrier the
    non-streaming retry loop uses. Once resumed, the stream proceeds normally.
    """
    engines = [_MockStreamEngine()]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    await client.pause_generation()

    payload = {"json": {"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]}, "headers": {}}

    async def _consume():
        return [chunk async for chunk in client.chat_completion_stream(payload)]

    task = asyncio.create_task(_consume())

    # While paused, the stream must block before reaching the engine. A parked stream never
    # runs on a bare yield, so if it were going to reach the engine it would within these turns.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not engines[0].entered.is_set(), "stream reached the engine while generation was paused"
    assert not task.done()

    # Resume -> the stream should now proceed to the engine and complete.
    await client.resume_generation(policy_step=1)
    chunks = await asyncio.wait_for(task, timeout=5)
    assert engines[0].entered.is_set()
    assert any("[DONE]" in c for c in chunks)


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
@pytest.mark.parametrize("prompt", SINGLE_PROMPTS)
async def test_completion_single_prompt_reissues_once_after_a_midflight_abort(prompt):
    """A pause that begins mid-flight comes back as finish_reason "abort" (the abort-mode
    scheduler pause returns what was generated so far). /completions cannot continue a
    partial generation the way chat can, so the request is re-issued from the start, once,
    and the caller sees only the completed response."""
    engines = [
        _MockCompletionEngine(
            [
                _completion_response("partial", "abort", completion_tokens=2),
                _completion_response("complete answer", "stop"),
            ]
        )
    ]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    result = await client.completion(_completion_payload(prompt))

    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["text"] == "complete answer"
    assert len(engines[0].calls) == 2
    # The re-issue restarts from the ORIGINAL prompt (no continuation protocol here), and
    # every other sampling field is carried over unchanged.
    expected_prompt = [prompt]
    for call in engines[0].calls:
        assert call["prompt"] == expected_prompt
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
async def test_completion_single_prompt_returns_a_second_abort_as_is():
    """One retry, not a loop: a re-issue restarts generation, so retrying forever could
    livelock a long generation under frequent weight syncs. A second abort is handed back
    to the caller — the behaviour it would have seen before any of this handling."""
    engines = [_MockCompletionEngine([_completion_response("partial", "abort", completion_tokens=2)])]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())

    result = await client.completion(_completion_payload([1, 2, 3, 4]))

    assert result["choices"][0]["finish_reason"] == "abort"
    assert len(engines[0].calls) == 2


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
    "prompt",
    [pytest.param(["hello"], id="one_string_is_still_batched"), pytest.param([["a"], ["b"]], id="batched")],
)
async def test_completion_batched_still_raises_while_paused(prompt):
    """Batched /completions fans out across engines with no per-sub-request pause handling,
    so it keeps the pre-existing behaviour. A list of ONE string is batched by
    `postprocess_completion_request`, and stays on that path here too."""
    engines = [_MockCompletionEngine([_completion_response("done", "stop")])]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=_make_min_cfg())
    client.generation_paused_event.set()

    with pytest.raises(RuntimeError, match="unsupported for batched /completions"):
        await client.completion(_completion_payload(prompt))

    assert not engines[0].entered.is_set()


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


def test_client_serves_http_only_after_engines_acknowledge_startup_pause(unused_tcp_port, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class DelayedPauseEngine(_MockGenerateEngine):
        async def pause_generation(self):
            entered.set()
            assert release.wait(10), "Engine pause acknowledgment was not released"
            await super().pause_generation()

    engine = DelayedPauseEngine()
    cfg = get_default_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.enable_http_endpoint = True
    cfg.generator.http_endpoint_port = unused_tcp_port
    OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", "dummy", force_add=True)
    cfg.generator.weight_sync_transport = "expert_block"
    experiment = main_base.BasePPOExp.__new__(main_base.BasePPOExp)
    experiment.cfg, experiment.tokenizer, experiment.colocate_pg = cfg, object(), None
    monkeypatch.setattr(main_base, "create_ray_wrapped_inference_engines_from_config", lambda *args, **kwargs: [engine])
    url = f"http://127.0.0.1:{unused_tcp_port}/v1/models"
    client = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(experiment.create_inference_engine_client)
        try:
            assert entered.wait(10), "The client did not request an engine pause"
            with pytest.raises(requests.ConnectionError):
                requests.get(url, timeout=1)
            release.set()
            client = future.result(timeout=10)
            assert requests.get(url, timeout=1).status_code == HTTPStatus.OK
        finally:
            release.set()
            if client is not None:
                client.shutdown_http_endpoint()


@pytest.mark.asyncio
async def test_generate_single_prompt_waits_for_resume_then_reaches_engine():
    """A single-prompt generate() that arrives during a weight-sync pause is held, not rejected,
    and reaches the engine unchanged once generation resumes."""
    engine = _MockGenerateEngine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())
    # Simulate the pause directly, as the batched-completion test does: the engine fan-out is
    # not what this test exercises.
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


@pytest.mark.asyncio
async def test_generate_batched_still_raises_while_paused():
    """Batched generate() has no per-prompt retry loop, so it keeps rejecting during a pause."""
    engine = _MockGenerateEngine()
    client = InferenceEngineClient(engines=[engine], tokenizer=object(), full_config=_make_min_cfg())
    client.generation_paused_event.set()

    request = InferenceEngineInput(prompt_token_ids=[[1, 2], [3, 4]], sampling_params={"max_tokens": 5})
    with pytest.raises(RuntimeError, match="batched"):
        await client.generate(request)
    assert engine.requests == []


class _DeadEngine(_MockGenerateEngine):
    """Engine whose actor has died: every call fails the way a dead Ray actor does."""

    async def generate(self, request):
        raise ray.exceptions.RayActorError()

    async def pause_generation(self):
        raise ray.exceptions.RayActorError()


async def _fail_over_engine_zero(client: InferenceEngineClient) -> None:
    """Send a session routed to engine 0 so its actor error marks engine 0 dead."""
    session_id = next(sid for sid in ("trial-0", "trial-1") if hash_with_sha256(sid) % 2 == 0)
    await client.generate(
        InferenceEngineInput(prompt_token_ids=[[1, 2, 3]], sampling_params={"max_tokens": 5}, session_ids=[session_id])
    )


@pytest.mark.asyncio
async def test_weight_sync_pause_skips_an_engine_that_already_died():
    """A pause must not wait on, or fail because of, an engine the client already knows is dead."""
    dead, live = _DeadEngine(), _MockGenerateEngine()
    client = InferenceEngineClient(engines=[dead, live], tokenizer=object(), full_config=_make_min_cfg())
    await _fail_over_engine_zero(client)
    assert len(live.requests) == 1

    await client.pause_generation()
    assert live.scheduler_paused
    await client.resume_generation(policy_step=1)
    assert not live.scheduler_paused


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


@pytest.mark.asyncio
async def test_draft_refresh_skips_dead_engines() -> None:
    dead, live = _DeadDraftEngine(), _DraftGenerateEngine()
    client = InferenceEngineClient(engines=[dead, live], tokenizer=object(), full_config=_make_min_cfg())
    await _fail_over_engine_zero(client)

    weights_path = "s3://bucket/drafts/draft-step-4/model.safetensors"
    coverage = await client.update_draft_weights(weights_path)

    assert coverage == [{"active": True}]
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
