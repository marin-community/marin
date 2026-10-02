# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import logging
import math
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer, decoders, models

from levanter.inference.jit_scheduler import FinishReason
from levanter.inference.utils import is_valid
from levanter.layers.kv_cache import KvPageCache
from levanter.models.llama import LlamaLMHeadModel
from levanter.testing.helpers import skip_if_no_torch
from levanter.testing.model_configs import llama_test_config
from levanter.trainer import TrainerConfig
from levanter.tokenizers import HfMarinTokenizer

try:
    import httpx
    import uvicorn
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from openai.types import Completion

    from levanter.inference.engine import (
        InferenceEngine,
        InferenceEngineConfig,
        score_token_sequence_logprobs,
    )
    from levanter.inference.openai import (
        InferenceBatch,
        InferenceContext,
        InferenceResponse,
        InferenceServer,
        InferenceServerConfig,
        _compute_tokens,
    )
    from levanter.inference.openai_protocol import ChatMessage

except ImportError:
    pytest.skip("Serving imports not installed, use --extra=serve", allow_module_level=True)

logger = logging.getLogger(__name__)

TEST_MODEL_NAME = "tiny-random-llama"
TEST_MAX_SEQ_LEN = 64
TEST_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


@pytest.fixture(scope="module")
def trainer_config():
    return TrainerConfig()


@pytest.fixture(scope="module")
def inference_server_config():
    return InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=TEST_MAX_SEQ_LEN,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=32,
            hbm_utilization=0.1,
        ),
        model_name=TEST_MODEL_NAME,
        temperature=0.7,
        seed=42,
    )


@pytest.fixture(scope="module")
def llama_model_config(local_gpt2_tokenizer_path):
    return dataclasses.replace(
        llama_test_config(seq_len=TEST_MAX_SEQ_LEN, num_kv_heads=2),
        tie_word_embeddings=True,
        reference_checkpoint=None,
        tokenizer=local_gpt2_tokenizer_path,
    )


@pytest.fixture(scope="module")
def generated_model_and_tokenizer(trainer_config, llama_model_config, local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        model = LlamaLMHeadModel.init(
            hax.Axis("vocab", len(tokenizer)),
            llama_model_config,
            key=jax.random.PRNGKey(0),
        )

    return model, tokenizer


@pytest.fixture(scope="module")
def inference_server(trainer_config, inference_server_config, generated_model_and_tokenizer):
    """Create an InferenceServer instance."""
    model, tokenizer = generated_model_and_tokenizer
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        return InferenceServer.create(inference_server_config, model, tokenizer)


@pytest.fixture(scope="module")
def test_client(inference_server):
    """Create a test client for the inference server."""
    with TestClient(inference_server.app) as client:
        yield client, inference_server


@pytest.fixture(scope="module")
def local_hf_checkpoint(tmp_path_factory, trainer_config, generated_model_and_tokenizer):
    model, _tokenizer = generated_model_and_tokenizer
    checkpoint_path = tmp_path_factory.mktemp("tiny_llama_hf")
    converter = model.config.hf_checkpoint_converter()
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter.save_pretrained(
            model,
            str(checkpoint_path),
            save_reference_code=False,
            chat_template=TEST_CHAT_TEMPLATE,
        )
    return checkpoint_path


@pytest.fixture(scope="module")
def hf_reference_model_and_tokenizer(local_hf_checkpoint):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(local_hf_checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(local_hf_checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()

    return model, tokenizer


@skip_if_no_torch
def test_greedy_correctness_against_hf(test_client, hf_reference_model_and_tokenizer):
    """Ensure deterministic (greedy) Levanter generations match HF reference outputs."""
    (client, _server) = test_client
    hf_model, hf_tokenizer = hf_reference_model_and_tokenizer
    torch = pytest.importorskip("torch")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In a distant future, humanity",
    ]
    max_tokens = 10
    levanter_generations: list[tuple[list[int], str]] = []

    for prompt in prompts:
        response = client.post(
            "/v1/completions",
            json={
                "model": TEST_MODEL_NAME,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "logprobs": True,
                "seed": 0,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        choice = payload["choices"][0]
        logprobs = choice.get("logprobs") or {}

        tokens = logprobs.get("tokens") or []
        token_ids = hf_tokenizer.convert_tokens_to_ids(tokens)
        levanter_generations.append((token_ids, choice["text"]))

    for prompt, (levanter_ids, levanter_text) in zip(prompts, levanter_generations, strict=True):
        inputs = hf_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=hf_tokenizer.eos_token_id,
                eos_token_id=hf_tokenizer.eos_token_id,
            )[0]

        generated_ids = output_ids[input_length:].tolist()
        hf_text = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)

        assert levanter_ids == generated_ids, f"Token mismatch for prompt '{prompt}'"
        assert levanter_text == hf_text, f"Text mismatch for prompt '{prompt}'"


def test_endpoints_exist(test_client):
    """Test that the endpoints are properly defined"""
    _, server = test_client
    routes = [route.path for route in server.app.routes]
    assert "/health" in routes
    assert "/v1/models" in routes
    assert "/v1/completions" in routes
    assert "/v1/chat/completions" in routes


def test_models_endpoint_reports_the_configured_model(test_client):
    """A client discovering the server (OpenAI SDK, dashboards) reads the id it should send back."""
    client, server = test_client

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == [server.config.model_name]


def test_chat_completion_without_a_chat_template_is_rejected(test_client, monkeypatch, local_gpt2_tokenizer):
    """A model with no chat template cannot represent a conversation, so chat requests are refused.

    Rendering one anyway would feed the model a prompt format it was never trained on and return
    it as a normal completion, leaving callers unable to tell a chat model from a base one.
    """
    client, server = test_client
    monkeypatch.setattr(server.inference_context, "tokenizer", local_gpt2_tokenizer)

    response = client.post(
        "/v1/chat/completions",
        json={"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )

    assert response.status_code == 400
    assert "no chat template" in response.json()["detail"]


def test_chat_completion_renders_template_arguments_and_tool_definitions(local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer.with_chat_template(
        "{% if enable_thinking is sameas false %}thinking=disabled\n{% endif %}"
        "{% if custom_instructions %}instructions={{ custom_instructions }}\n{% endif %}"
        "{% if tools %}tool={{ tools[0]['function']['name'] }}\n{% endif %}"
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup_weather", "parameters": {"type": "object"}},
        }
    ]

    tokens = _compute_tokens(
        [ChatMessage(role="user", content="Will it rain?")],
        tokenizer,
        tools,
        chat_template_kwargs={
            "enable_thinking": False,
            "custom_instructions": "Be concise.",
            "tools": [{"type": "function", "function": {"name": "ignored_tool"}}],
        },
    )
    rendered = tokenizer.decode(tokens)

    assert "thinking=disabled" in rendered
    assert "instructions=Be concise." in rendered
    assert "tool=lookup_weather" in rendered
    assert "ignored_tool" not in rendered


def test_chat_completion_rejects_rendering_argument_overrides(local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)

    with pytest.raises(HTTPException):
        _compute_tokens(
            [ChatMessage(role="user", content="Hello")],
            tokenizer,
            chat_template_kwargs={"tokenize": False},
        )


class _OpenAITestTokenizer:
    _id_to_piece = {0: "A", 1: " B", 2: " C", 3: " X"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("The test tokenizer does not define special tokens.")
        if text == "A":
            return [0]
        if text == "A B":
            return [0, 1]
        if text == " X":
            return [3]
        raise ValueError(f"Unexpected test text: {text}")

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._id_to_piece[int(token_id)] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_piece[int(token_id)]


class _DeterministicCompletionScoringModel(eqx.Module):
    Vocab: hax.Axis = eqx.field(static=True)

    def __init__(self):
        self.Vocab = hax.Axis("vocab", 4)

    def initial_cache(self, spec, *, dtype):
        return KvPageCache.init(spec, hax.Axis("kv_head", 1), hax.Axis("embed", 1), dtype=dtype)

    def decode(self, input_ids, cache, batch_info, pos_ids):
        logits = hax.nn.one_hot(3, self.Vocab, dtype=jnp.float32).broadcast_axis(input_ids.resolve_axis("position"))
        return logits, cache

    def __call__(
        self,
        input_ids: hax.NamedArray,
        attn_mask: object,
        pos_ids: hax.NamedArray,
        key: object,
    ) -> hax.NamedArray:
        Pos = input_ids.resolve_axis("position")
        logits = jnp.full((Pos.size, self.Vocab.size), -8.0, dtype=jnp.float32)
        if Pos.size > 0:
            logits = logits.at[0, 1].set(4.0)
        if Pos.size > 1:
            logits = logits.at[1, 3].set(3.0)
        return hax.named(logits, (Pos, self.Vocab))


def _sse_chunks(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


class _FakeCompletionContext:
    def __init__(self, max_seq_len: int = 4096):
        self.config = InferenceServerConfig(service=InferenceEngineConfig(max_seq_len=max_seq_len))
        self.model = _DeterministicCompletionScoringModel()
        self.tokenizer = _OpenAITestTokenizer()
        self.submitted_requests = 0
        self.admission_lock = threading.Lock()
        self.active_requests: dict[str, threading.Event] = {}

    def submit_request(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        stop_tokens: list[list[int]] | None,
        seed: int | None,
        future,
        n_generations: int = 1,
        echo_logprobs_top_k: int | None = None,
        cancel_event: threading.Event | None = None,
        on_delta=None,
    ) -> str:
        if (
            prompt_tokens != [0, 1]
            or max_tokens != 1
            or temperature != 0
            or stop_tokens is not None
            or seed != 1234
            or n_generations != 1
            or echo_logprobs_top_k != 1
        ):
            raise ValueError("The deterministic test context only supports one fixed completion request.")
        self.submitted_requests += 1
        echo_token_ids = prompt_tokens + [3]
        future.set_result(
            [
                InferenceResponse(
                    request_id="req_0",
                    text=" X",
                    tokens=[3],
                    prompt_tokens=len(prompt_tokens),
                    completion_tokens=1,
                    finish_reason=FinishReason.LENGTH,
                    model_version=0,
                    logprobs=[-123.0],
                    echo_token_ids=echo_token_ids,
                    echo_logprobs=score_token_sequence_logprobs(self.model, echo_token_ids, echo_logprobs_top_k),
                )
            ]
        )
        return "req_0"


@pytest.mark.parametrize("as_token_ids", [False, True])
def test_completion_echo_logprobs_are_lm_eval_aligned(as_token_ids):
    ctx = _FakeCompletionContext()
    app = InferenceServer._create_app(ctx)

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt2",
                "prompt": "A B",
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": 1234,
                "echo": True,
                "return_tokens_as_token_ids": as_token_ids,
            },
        )

    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    expected_prompt_logprob = float(jax.nn.log_softmax(jnp.array([-8.0, 4.0, -8.0, -8.0]))[1])
    expected_completion_logprob = float(jax.nn.log_softmax(jnp.array([-8.0, -8.0, -8.0, 3.0]))[3])

    assert choice["finish_reason"] == "length"
    assert choice["text"] == "A B X"
    token_labels = ["token_id:0", "token_id:1", "token_id:3"] if as_token_ids else ["A", " B", " X"]
    assert logprobs["tokens"] == token_labels
    assert logprobs["token_logprobs"] == pytest.approx([0.0, expected_prompt_logprob, expected_completion_logprob])
    assert logprobs["text_offset"] == [0, 1, 3]
    assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"])
    assert len(logprobs["tokens"]) == len(logprobs["top_logprobs"])
    assert logprobs["top_logprobs"][0] == {token_labels[0]: 0.0}
    assert logprobs["top_logprobs"][1][token_labels[1]] == pytest.approx(expected_prompt_logprob)
    assert logprobs["top_logprobs"][2][token_labels[2]] == pytest.approx(expected_completion_logprob)


def test_completion_echo_logprobs_rejects_scored_sequence_over_context():
    ctx = _FakeCompletionContext(max_seq_len=2)
    app = InferenceServer._create_app(ctx)

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt2",
                "prompt": ["A", "A B"],
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": 1234,
                "echo": True,
            },
        )

    assert response.status_code == 400, response.text
    assert "echo logprobs" in response.json()["detail"]
    assert ctx.submitted_requests == 0


def test_score_token_sequence_logprobs_empty_and_single_token_sequences():
    model = _DeterministicCompletionScoringModel()

    empty_result = score_token_sequence_logprobs(model, [], top_k=1)
    assert empty_result.token_logprobs == []
    assert empty_result.top_token_logprobs == []

    single_token_result = score_token_sequence_logprobs(model, [2], top_k=3)
    assert single_token_result.token_logprobs == [0.0]
    assert single_token_result.top_token_logprobs == [{2: 0.0}]


def test_logprobs_deterministic_behavior(test_client):
    """Test that logprobs are deterministic with same seed."""
    client, server = test_client

    # Make the same request twice with same seed
    request_data = {
        "model": TEST_MODEL_NAME,
        "prompt": "Once upon a time",
        "max_tokens": 4,
        "temperature": 0.0,  # Deterministic
        "logprobs": True,
        "seed": 12345,
    }

    response1 = client.post("/v1/completions", json=request_data)
    response2 = client.post("/v1/completions", json=request_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    completion1 = Completion.model_validate(response1.json())
    completion2 = Completion.model_validate(response2.json())

    logprobs1 = completion1.choices[0].logprobs
    logprobs2 = completion2.choices[0].logprobs

    assert len(logprobs1.tokens) == len(logprobs2.tokens)

    for t1, t2 in zip(logprobs1.tokens, logprobs2.tokens):
        assert t1 == t2

    for lp1, lp2 in zip(logprobs1.token_logprobs, logprobs2.token_logprobs):
        assert abs(lp1 - lp2) < 1e-6

    print("Deterministic logprobs test passed!")


def test_many_requests_threaded(test_client):
    executor = ThreadPoolExecutor(max_workers=8)
    client, server = test_client
    futures = []
    num_requests = 20
    for i in range(num_requests):
        futures.append(
            executor.submit(
                client.post,
                "/v1/completions",
                json={
                    "model": TEST_MODEL_NAME,
                    "prompt": "The quick brown fox",
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "seed": i,
                },
            )
        )

    for i, future in enumerate(futures):
        response = future.result()
        assert response.status_code == 200
        completion = Completion.model_validate(response.json())
        choice = completion.choices[0]
        assert choice.text
        print(f"Request {i} generated text: '{choice.text}'")


def test_reload_with_zeros_clears_outputs(test_client):
    """Test that reloading with a zeroed-out model properly clears outputs."""
    client, server = test_client

    # Make a request before reload to establish baseline
    response1 = client.post(
        "/v1/completions",
        json={
            "model": TEST_MODEL_NAME,
            "prompt": "The quick brown fox",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
        },
    )

    assert response1.status_code == 200
    completion1 = Completion.model_validate(response1.json())
    original_text = completion1.choices[0].text
    assert len(original_text.strip()) > 0

    original_model = server.inference_context.model

    # Force a reload with a zeroed-out model callback
    def _new_model(old_model):
        return jax.tree_util.tree_map(lambda x: x * 0, old_model)

    server.reload(_new_model, expected_version=server.model_version)

    # Make a request after reload - should get all zero tokens in theory
    response2 = client.post(
        "/v1/completions",
        json={
            "model": TEST_MODEL_NAME,
            "prompt": "The quick brown fox",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
        },
    )

    assert response2.status_code == 200
    completion2 = Completion.model_validate(response2.json())
    zeroed_text = completion2.choices[0].text

    # With zeroed weights, the output should be different from the original
    # probably empty but depends on the tokenizer & stop tokens
    assert completion2.usage.completion_tokens > 0
    print(f"Original text: '{original_text}'")
    print(f"Zeroed model text: '{zeroed_text}'")

    # now reload the original weights back
    def _original_model(old_model):
        return original_model

    server.reload(_original_model, expected_version=server.model_version)
    response3 = client.post(
        "/v1/completions",
        json={
            "model": TEST_MODEL_NAME,
            "prompt": "The quick brown fox",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
        },
    )
    assert response3.status_code == 200
    completion3 = Completion.model_validate(response3.json())
    restored_text = completion3.choices[0].text
    assert restored_text == original_text


def test_tokens_endpoint(test_client):
    """Test the tokens endpoint for tokenizing chat messages."""
    client, server = test_client

    response = client.post(
        "/v1/tokens",
        json={
            "model": TEST_MODEL_NAME,
            "message_list": [
                [{"role": "user", "content": "Hello, how are you?"}],
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2

    # Check that each result has tokens
    for token_list in result["results"]:
        assert "tokens" in token_list
        assert isinstance(token_list["tokens"], list)
        assert len(token_list["tokens"]) > 0
        assert all(isinstance(t, int) for t in token_list["tokens"])

    print(f"Tokenization results: {result['results']}")


def test_completion_stop_alternatives_and_length_report_actual_termination():
    config = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=4,
            max_seqs_in_prefill=2,
            compute_dtype=jnp.float32,
        )
    )
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _DeterministicCompletionScoringModel(), _OpenAITestTokenizer())
    request = {"model": "gpt2", "prompt": "A", "temperature": 0, "max_tokens": 3}
    try:
        with TestClient(server.app) as client:
            length_response = client.post("/v1/completions", json=request)
            stop_response = client.post("/v1/completions", json={**request, "stop": ["A B", " X"]})
        assert length_response.status_code == stop_response.status_code == 200
        assert length_response.json()["choices"][0]["finish_reason"] == "length"
        assert length_response.json()["usage"]["completion_tokens"] == 3
        assert stop_response.json()["choices"][0]["finish_reason"] == "stop"
        assert stop_response.json()["usage"]["completion_tokens"] == 1
    finally:
        server.inference_context.shutdown()


class _AliasingChatTokenizer(_OpenAITestTokenizer):
    # IDs 2 and 3 decode identically, but encode chooses 2. Retokenizing sampled
    # text would therefore change the model's next-token distribution.
    _id_to_piece = {0: "A", 1: " B", 2: " X", 3: " X"}
    chat_template = "test"

    def encode(self, text, add_special_tokens=False):
        return [2] if text == " X" else super().encode(text, add_special_tokens)

    def apply_chat_template(self, messages, *, add_generation_prompt, continue_final_message, **kwargs):
        if continue_final_message:
            return [0, 1, 2]
        return [0, 1] if add_generation_prompt else [0]


class _TokenSensitiveCompletionModel(_DeterministicCompletionScoringModel):
    def decode(self, input_ids, cache, batch_info, pos_ids):
        return hax.nn.one_hot((input_ids + 2) % 4, self.Vocab, dtype=jnp.float32), cache


def _exact_token_config():
    return InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=4,
            max_seqs_in_prefill=2,
            compute_dtype=jnp.float32,
        )
    )


@pytest.fixture
def exact_token_server():
    config = _exact_token_config()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _TokenSensitiveCompletionModel(), _AliasingChatTokenizer())
    try:
        yield server
    finally:
        server.inference_context.shutdown()


@pytest.fixture
def exact_token_client(exact_token_server):
    with TestClient(exact_token_server.app) as client:
        yield client


def test_completion_integer_prompts_keep_token_identity_through_stopping(exact_token_client):
    response = exact_token_client.post(
        "/v1/completions",
        json={
            "model": "gpt2",
            "prompt": [[0, 3], [0, 2]],
            "temperature": 0,
            "max_tokens": 3,
            "stop_token_ids": [1],
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
        },
    )
    assert response.status_code == 200, response.text
    first, second = response.json()["choices"]
    assert first["prompt_token_ids"] == [0, 3]
    assert second["prompt_token_ids"] == [0, 2]
    assert first["token_ids"] == [1]
    assert second["token_ids"] == [0, 2, 0]
    assert [first["finish_reason"], second["finish_reason"]] == ["stop", "length"]
    assert first["logprobs"]["tokens"] == ["token_id:1"]
    assert len(second["logprobs"]["token_logprobs"]) == 3
    flat = exact_token_client.post(
        "/v1/completions",
        json={
            "model": "gpt2",
            "prompt": [0, 3],
            "temperature": 0,
            "max_tokens": 1,
            "return_token_ids": True,
        },
    )
    assert flat.status_code == 200, flat.text
    assert flat.json()["choices"][0]["token_ids"] == [1]


def test_chat_exact_token_continuation_matches_uninterrupted_decode(exact_token_client):
    messages = [{"role": "user", "content": "A B"}]
    tokenized = exact_token_client.post(
        "/tokenize",
        json={
            "model": "gpt2",
            "messages": messages,
            "add_generation_prompt": True,
        },
    )
    assert tokenized.status_code == 200, tokenized.text
    assert tokenized.json() == {"tokens": [0, 1], "count": 2, "max_model_len": 8}
    body = {
        "model": "gpt2",
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": 2,
        "return_token_ids": True,
        "logprobs": True,
    }
    full_response = exact_token_client.post("/v1/chat/completions", json=body)
    partial_response = exact_token_client.post("/v1/chat/completions", json={**body, "max_completion_tokens": 1})
    assert full_response.status_code == partial_response.status_code == 200
    full, partial = full_response.json(), partial_response.json()
    assert full["prompt_token_ids"] == partial["prompt_token_ids"] == [0, 1]
    assert partial["choices"][0]["token_ids"] == [3]
    retry = {
        **body,
        "max_completion_tokens": 1,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "_skyrl_exact_prompt_token_ids": [0, 1, 3],
        "messages": [*messages, partial["choices"][0]["message"]],
    }
    resumed_response = exact_token_client.post("/v1/chat/completions", json=retry)
    assert resumed_response.status_code == 200, resumed_response.text
    resumed = resumed_response.json()
    assert resumed["prompt_token_ids"] == [0, 1, 3]
    prefix, suffix, expected = partial["choices"][0], resumed["choices"][0], full["choices"][0]
    assert prefix["token_ids"] + suffix["token_ids"] == expected["token_ids"] == [3, 1]
    assert prefix["logprobs"]["content"] + suffix["logprobs"]["content"] == expected["logprobs"]["content"]
    assert suffix["finish_reason"] == expected["finish_reason"] == "length"
    streamed = exact_token_client.post("/v1/chat/completions", json={**body, "stream": True})
    chunks = _sse_chunks(streamed.text)
    assert chunks[0]["prompt_token_ids"] == full["prompt_token_ids"]
    assert [token for chunk in chunks for token in chunk["choices"][0].get("token_ids", [])] == expected["token_ids"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_paused_server_returns_abort_then_resumes_exact_generation(exact_token_server):
    request = {
        "model": "gpt2",
        "prompt": [0, 1],
        "temperature": 0,
        "max_tokens": 2,
        "return_token_ids": True,
        "logprobs": 0,
    }
    with TestClient(exact_token_server.app) as client:
        before = client.post("/v1/completions", json=request)
        exact_token_server.pause_generation()
        aborted = client.post("/v1/completions", json=request)
        echo_aborted = client.post("/v1/completions", json={**request, "echo": True, "logprobs": 1})
        exact_token_server.resume_generation()
        after = client.post("/v1/completions", json=request)
    assert before.status_code == aborted.status_code == after.status_code == 200
    assert aborted.json()["choices"][0]["finish_reason"] == "abort"
    assert aborted.json()["choices"][0]["token_ids"] == []
    assert aborted.json()["choices"][0]["logprobs"]["token_logprobs"] == []
    assert aborted.json()["usage"]["completion_tokens"] == 0
    assert echo_aborted.status_code == 200
    assert echo_aborted.json()["choices"][0]["finish_reason"] == "abort"
    assert echo_aborted.json()["choices"][0]["logprobs"] is None
    assert before.json()["choices"] == after.json()["choices"]
    assert after.json()["choices"][0]["token_ids"] == [3, 1]


@pytest.mark.asyncio
async def test_pause_invalidates_requests_collected_before_the_barrier():
    config = _exact_token_config()
    model = _TokenSensitiveCompletionModel()
    tokenizer = _AliasingChatTokenizer()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config.service)
        context = InferenceContext(model, tokenizer, engine, config)
        future = asyncio.get_running_loop().create_future()
        context.submit_request([0, 1], 2, 0.0, 1.0, None, 0, future)
        # The batching thread can have removed a request from its queue when pause begins.
        collected = InferenceBatch([context.request_queue.get_nowait()])
        context.pause_generation()
        context.resume_generation()
        context._execute_batch(collected)
        aborted = await future
        assert aborted[0].finish_reason == FinishReason.ABORT
        assert aborted[0].tokens == aborted[0].logprobs == []
        resumed_future = asyncio.get_running_loop().create_future()
        context.submit_request([0, 1], 2, 0.0, 1.0, None, 0, resumed_future)
        context._execute_batch(InferenceBatch([context.request_queue.get_nowait()]))
        resumed = await resumed_future
    assert resumed[0].finish_reason == FinishReason.LENGTH
    assert resumed[0].tokens == [3, 1]


class _BlockingTokenModel(_TokenSensitiveCompletionModel):
    entered: threading.Event = eqx.field(static=True)
    release: threading.Event = eqx.field(static=True)

    def __init__(self, entered, release):
        super().__init__()
        self.entered = entered
        self.release = release

    def _wait_for_release(self):
        self.entered.set()
        assert self.release.wait(30), "test did not release the in-flight model call"

    def decode(self, input_ids, cache, batch_info, pos_ids):
        jax.debug.callback(self._wait_for_release)
        return super().decode(input_ids, cache, batch_info, pos_ids)


@pytest.mark.parametrize("stream", [False, True])
def test_http_pause_preserves_partial_tokens_and_logprobs(stream):
    entered, release = threading.Event(), threading.Event()
    config = _exact_token_config()
    model = _BlockingTokenModel(entered, release)
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, model, _AliasingChatTokenizer())
    request = {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "A"}],
        "max_completion_tokens": 3,
        "temperature": 0,
        "logprobs": True,
        "return_token_ids": True,
        "stream": stream,
    }
    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(client.post, "/v1/chat/completions", json=request)
            assert entered.wait(30), "generation did not reach the model"
            pausing = pool.submit(client.post, "/pause_generation", json={"mode": "abort", "clear_cache": True})
            assert server.inference_context.pause_event.wait(5)
            release.set()
            assert pausing.result(timeout=30).status_code == 200
            response = pending.result(timeout=30)
            assert response.status_code == 200, response.text
            if stream:
                chunks = _sse_chunks(response.text)
                content = chunks[0]
                finish_reason = chunks[-1]["choices"][0]["finish_reason"]
            else:
                content = response.json()
                finish_reason = content["choices"][0]["finish_reason"]
            choice = content["choices"][0]
            assert finish_reason == "abort"
            assert content["prompt_token_ids"] == [0, 1]
            assert choice["token_ids"] == [3]
            partial_logprobs = choice["logprobs"]["content"]
            assert len(partial_logprobs) == 1
            paused = client.post("/v1/chat/completions", json={**request, "stream": False}).json()
            assert paused["choices"][0]["finish_reason"] == "abort"
            assert paused["choices"][0]["token_ids"] == []
            assert client.post("/resume_generation").status_code == 200
            full = client.post("/v1/chat/completions", json={**request, "stream": False}).json()
            continuation = client.post(
                "/v1/chat/completions",
                json={
                    **request,
                    "stream": False,
                    "max_completion_tokens": 2,
                    "_skyrl_exact_prompt_token_ids": [0, 1, 3],
                },
            ).json()
            assert choice["token_ids"] + continuation["choices"][0]["token_ids"] == full["choices"][0]["token_ids"]
            assert (
                partial_logprobs + continuation["choices"][0]["logprobs"]["content"]
                == full["choices"][0]["logprobs"]["content"]
            )
    finally:
        release.set()
        server.inference_context.shutdown()


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "keep", "clear_cache": True},
        {"mode": "wait", "clear_cache": True},
        {"mode": "abort", "clear_cache": False},
    ],
)
def test_http_unsupported_pause_keeps_generation_available(exact_token_client, policy):
    request = {"model": "gpt2", "prompt": [0, 1], "max_tokens": 2, "temperature": 0, "return_token_ids": True}
    before = exact_token_client.post("/v1/completions", json=request).json()["choices"]
    assert exact_token_client.post("/pause_generation", json=policy).status_code == 400
    after = exact_token_client.post("/v1/completions", json=request).json()["choices"]
    assert after == before
    assert after[0]["finish_reason"] == "length"


class _WeightedTokenModel(_TokenSensitiveCompletionModel):
    bias: jax.Array

    def __init__(self):
        super().__init__()
        self.bias = jnp.zeros(4)

    def decode(self, input_ids, cache, batch_info, pos_ids):
        logits, _ = super().decode(input_ids, cache, batch_info, pos_ids)
        cache = dataclasses.replace(cache, kv_pages=cache.kv_pages + 1)
        return logits + hax.named(self.bias, self.Vocab), cache


def test_weight_publication_stages_before_install_and_preserves_failed_version():
    config = _exact_token_config()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _WeightedTokenModel(), _AliasingChatTokenizer())
    request = {
        "model": "gpt2",
        "prompt": [0, 1],
        "max_tokens": 2,
        "temperature": 0,
        "return_token_ids": True,
        "logprobs": 0,
    }

    def staging_failure(model):
        raise ValueError("checkpoint staging failed")

    try:
        with TestClient(server.app) as client:
            before = client.post("/v1/completions", json=request).json()["choices"][0]
            assert before["token_ids"] == [3, 1]
            assert before["model_version"] == 0
            assert bool(jnp.any(server.inference_context.engine.gen_state.cache.kv_pages.array))
            with pytest.raises(ValueError, match="checkpoint staging failed"):
                server.reload(staging_failure, expected_version=0)
            with pytest.raises(ValueError):
                server.reload(lambda model: eqx.tree_at(lambda m: m.bias, model, jnp.ones(8)), expected_version=0)
            assert server.model_version == 0
            assert not server.inference_context.pause_event.is_set()
            assert client.post("/v1/completions", json=request).json()["choices"][0] == before

            def replacement(model):
                return eqx.tree_at(lambda m: m.bias, model, model.bias.at[0].set(8))

            server.pause_generation()
            assert server.reload(replacement, expected_version=0) == 1
            assert server.inference_context.pause_event.is_set()
            assert not bool(jnp.any(server.inference_context.engine.gen_state.cache.kv_pages.array))
            server.resume_generation()
            after = client.post("/v1/completions", json=request).json()["choices"][0]
            assert after["token_ids"] == [0, 0]
            assert after["model_version"] == 1
            with pytest.raises(ValueError, match="Expected model version 0, serving 1"):
                server.reload(staging_failure, expected_version=0)
            assert client.post("/v1/completions", json=request).json()["choices"][0] == after

            staging, release = threading.Event(), threading.Event()

            def delayed_replacement(model):
                staging.set()
                assert release.wait(30)
                return replacement(model)

            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(server.reload, delayed_replacement, expected_version=1)
                try:
                    assert staging.wait(5)
                    assert client.post("/v1/completions", json=request).json()["choices"][0] == after
                    assert (
                        server.reload(
                            lambda model: eqx.tree_at(lambda m: m.bias, model, model.bias * 0), expected_version=1
                        )
                        == 2
                    )
                finally:
                    release.set()
                with pytest.raises(ValueError, match="Expected model version 1, serving 2"):
                    pending.result(timeout=30)
            restored = client.post("/v1/completions", json=request).json()["choices"][0]
            assert restored["token_ids"] == before["token_ids"]
            assert restored["model_version"] == 2
            chat_request = {
                "model": "gpt2",
                "messages": [{"role": "user", "content": "A"}],
                "max_completion_tokens": 1,
                "temperature": 0,
                "return_token_ids": True,
            }
            assert client.post("/v1/chat/completions", json=chat_request).json()["model_version"] == 2
            streamed = client.post("/v1/chat/completions", json={**chat_request, "stream": True})
            chunks = _sse_chunks(streamed.text)
            assert all(chunk["model_version"] == 2 for chunk in chunks)
    finally:
        server.inference_context.shutdown()


@pytest.mark.parametrize("stream", [False, True])
def test_individual_http_abort_preserves_peer_and_exact_continuation(stream):
    entered, release = threading.Event(), threading.Event()
    config = _exact_token_config()
    model = _BlockingTokenModel(entered, release)
    tokenizer = _AliasingChatTokenizer()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config.service)
    context = InferenceContext(model, tokenizer, engine, config)
    server = InferenceServer(config, context, InferenceServer._create_app(context))
    request = {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "A"}],
        "max_completion_tokens": 3,
        "temperature": 0,
        "logprobs": True,
        "return_token_ids": True,
    }
    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
            try:
                cancelled = pool.submit(
                    client.post,
                    "/v1/chat/completions",
                    json={**request, "stream": stream},
                    headers={"x-request-id": "cancel-me"},
                )
                survivor = pool.submit(
                    client.post, "/v1/chat/completions", json=request, headers={"x-request-id": "keep-me"}
                )
                # Start the actual batching threads after both HTTP requests have reached admission.
                collected = []
                try:
                    collected.append(context.request_queue.get(timeout=30))
                    collected.append(context.request_queue.get(timeout=30))
                finally:
                    for queued in collected:
                        context.request_queue.put(queued)
                    context.start()
                assert entered.wait(30)
                server.abort(["cancel-me"])
                release.set()
                cancelled_response = cancelled.result(timeout=30)
                survivor_response = survivor.result(timeout=30)
                assert cancelled_response.status_code == survivor_response.status_code == 200
                if stream:
                    chunks = _sse_chunks(cancelled_response.text)
                    partial = chunks[0]["choices"][0]
                    assert chunks[-1]["choices"][0]["finish_reason"] == "abort"
                else:
                    partial = cancelled_response.json()["choices"][0]
                    assert partial["finish_reason"] == "abort"
                complete = survivor_response.json()["choices"][0]
                assert partial["token_ids"] == [3]
                assert complete["token_ids"] == [3, 1, 3]
                assert complete["finish_reason"] == "length"
                retry = client.post(
                    "/v1/chat/completions",
                    json={**request, "max_completion_tokens": 2, "_skyrl_exact_prompt_token_ids": [0, 1, 3]},
                    headers={"x-request-id": "cancel-me"},
                ).json()["choices"][0]
                assert partial["token_ids"] + retry["token_ids"] == complete["token_ids"]
                assert partial["logprobs"]["content"] + retry["logprobs"]["content"] == complete["logprobs"]["content"]
            finally:
                release.set()
    finally:
        release.set()
        context.shutdown()


@pytest.mark.asyncio
async def test_http_disconnect_cancels_generation_without_poisoning_next_request():
    entered, release = threading.Event(), threading.Event()
    config = _exact_token_config()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _BlockingTokenModel(entered, release), _AliasingChatTokenizer())
    messages = asyncio.Queue()

    async def send(message):
        pass

    body = {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "A"}],
        "max_completion_tokens": 3,
        "temperature": 0,
        "return_token_ids": True,
    }
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"x-request-id", b"disconnected")],
        "server": ("testserver", 80),
        "client": ("client", 1),
    }
    await messages.put({"type": "http.request", "body": json.dumps(body).encode(), "more_body": False})
    task = asyncio.create_task(server.app(scope, messages.get, send))
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        event = server.inference_context.active_requests["disconnected"]
        await messages.put({"type": "http.disconnect"})
        with pytest.raises(asyncio.CancelledError):
            await task
        assert event.is_set()
        assert "disconnected" not in server.inference_context.active_requests
        release.set()
        with TestClient(server.app) as client:
            response = await asyncio.to_thread(
                client.post, "/v1/chat/completions", json=body, headers={"x-request-id": "disconnected"}
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["token_ids"] == [3, 1, 3]
        assert response.json()["choices"][0]["finish_reason"] == "length"
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        server.inference_context.shutdown()


class _DecodeGatedModel(_TokenSensitiveCompletionModel):
    entered: tuple[threading.Event, threading.Event] = eqx.field(static=True)
    release: tuple[threading.Event, threading.Event] = eqx.field(static=True)

    def __init__(self, entered, release):
        super().__init__()
        self.entered, self.release = entered, release

    def _gate(self, positions):
        last_position = int(positions.max())
        if last_position >= 2:
            gate = min(last_position - 2, 1)
            self.entered[gate].set()
            assert self.release[gate].wait(30), "test did not release the decode gate"

    def decode(self, input_ids, cache, batch_info, pos_ids):
        jax.debug.callback(self._gate, jnp.where(is_valid(pos_ids.array), pos_ids.array, -1))
        return super().decode(input_ids, cache, batch_info, pos_ids)


class _ReadyHttpServer(uvicorn.Server):
    def __init__(self, app, ready):
        super().__init__(uvicorn.Config(app, log_level="error", lifespan="off"))
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets)
        self.ready.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_live_sse_emits_tokens_and_abort_before_peer_finishes(endpoint, disconnect):
    entered = (threading.Event(), threading.Event())
    release = (threading.Event(), threading.Event())
    config = _exact_token_config()
    config = dataclasses.replace(config, service=dataclasses.replace(config.service, max_rounds=1))
    model, tokenizer = _DecodeGatedModel(entered, release), _AliasingChatTokenizer()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config.service)
    context = InferenceContext(model, tokenizer, engine, config)
    server = InferenceServer(config, context, InferenceServer._create_app(context))
    body = {"model": "gpt2", "max_tokens": 5, "temperature": 0, "return_token_ids": True}
    if endpoint == "completions":
        body.update(prompt=[0, 1], logprobs=0)
    else:
        body.update(messages=[{"role": "user", "content": "A"}], logprobs=True)
    received = asyncio.Queue()

    async def read_stream(client):
        async with client.stream(
            "POST", f"/v1/{endpoint}", json={**body, "stream": True}, headers={"x-request-id": "cancel-me"}
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    await received.put(line[6:])

    ready = asyncio.Event()
    http_server = _ReadyHttpServer(server.app, ready)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        serving = asyncio.create_task(http_server.serve(sockets=[listener]))
        await asyncio.wait_for(ready.wait(), 10)
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                streaming = asyncio.create_task(read_stream(client))
                peer = asyncio.create_task(client.post(f"/v1/{endpoint}", json=body, headers={"x-request-id": "peer"}))
                collected = []
                try:
                    # Force both real HTTP requests into one engine batch.
                    collected.append(await asyncio.to_thread(context.request_queue.get, True, 30))
                    collected.append(await asyncio.to_thread(context.request_queue.get, True, 30))
                finally:
                    for request in collected:
                        context.request_queue.put(request)
                    context.start()
                try:
                    first = json.loads(await asyncio.wait_for(received.get(), 30))
                    assert first["choices"][0]["token_ids"] == [3]
                    assert first["choices"][0]["finish_reason"] is None
                    assert await asyncio.to_thread(entered[0].wait, 30)
                    assert not peer.done()
                    if disconnect:
                        cancellation = context.active_requests["cancel-me"]
                        streaming.cancel()
                        await asyncio.gather(streaming, return_exceptions=True)
                        assert await asyncio.to_thread(cancellation.wait, 10)
                        for gate in release:
                            gate.set()
                        complete = (await peer).json()["choices"][0]
                        retry = await client.post(f"/v1/{endpoint}", json=body, headers={"x-request-id": "cancel-me"})
                        assert retry.status_code == 200
                        assert retry.json()["choices"][0]["token_ids"] == complete["token_ids"] == [3, 1, 3, 1, 3]
                        return
                    server.abort(["cancel-me"])
                    release[0].set()
                    chunks = [first]
                    while (payload := await asyncio.wait_for(received.get(), 30)) != "[DONE]":
                        chunks.append(json.loads(payload))
                    await streaming
                    assert chunks[-1]["choices"][0]["finish_reason"] == "abort"
                    assert await asyncio.to_thread(entered[1].wait, 30)
                    assert not peer.done()
                    release[1].set()
                    complete_response = await peer
                    assert complete_response.status_code == 200
                    complete = complete_response.json()["choices"][0]
                    partial_ids = [token for chunk in chunks for token in chunk["choices"][0].get("token_ids", [])]
                    assert partial_ids == complete["token_ids"][: len(partial_ids)] == [3, 1]
                    retry_body = {**body, "max_tokens": 5 - len(partial_ids)}
                    retry_body.update(
                        {"prompt": [0, 1, *partial_ids]}
                        if endpoint == "completions"
                        else {"_skyrl_exact_prompt_token_ids": [0, 1, *partial_ids]}
                    )
                    retry_response = await client.post(f"/v1/{endpoint}", json=retry_body)
                    assert retry_response.status_code == 200
                    retry = retry_response.json()["choices"][0]
                    assert partial_ids + retry["token_ids"] == complete["token_ids"]
                    key = "token_logprobs" if endpoint == "completions" else "content"
                    partial_logprobs = [
                        item for chunk in chunks for item in (chunk["choices"][0].get("logprobs") or {}).get(key, [])
                    ]
                    assert partial_logprobs + retry["logprobs"][key] == complete["logprobs"][key]
                finally:
                    for gate in release:
                        gate.set()
                    streaming.cancel()
                    peer.cancel()
                    await asyncio.gather(streaming, peer, return_exceptions=True)
        finally:
            for gate in release:
                gate.set()
            http_server.should_exit = True
            await serving
            context.shutdown()


def test_streamed_prompt_batches_keep_choice_ids_and_logprobs(exact_token_client):
    body = {
        "model": "gpt2",
        "prompt": [[0, 1], [0, 2]],
        "n": 2,
        "max_tokens": 2,
        "temperature": 0,
        "return_token_ids": True,
        "logprobs": 0,
    }
    full = exact_token_client.post("/v1/completions", json=body).json()
    streamed = exact_token_client.post("/v1/completions", json={**body, "stream": True})
    assert streamed.status_code == 200
    chunks = _sse_chunks(streamed.text)
    for expected in full["choices"]:
        choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"][0]["index"] == expected["index"]]
        assert [token for choice in choices for token in choice["token_ids"]] == expected["token_ids"]
        assert [lp for choice in choices for lp in choice["logprobs"]["token_logprobs"]] == expected["logprobs"][
            "token_logprobs"
        ]
        assert "".join(choice["text"] for choice in choices) == expected["text"]
        assert choices[-1]["finish_reason"] == expected["finish_reason"]


class _UnicodeTokenModel(_TokenSensitiveCompletionModel):
    def decode(self, input_ids, cache, batch_info, pos_ids):
        return hax.nn.one_hot((pos_ids + 1) % 4, self.Vocab, dtype=jnp.float32), cache


def test_streamed_unicode_keeps_byte_token_ids_without_replacement_text():
    vocab = {"<0xAC>": 0, "A": 1, "<0xE2>": 2, "<0x82>": 3}
    rust_tokenizer = Tokenizer(models.WordLevel(vocab))
    rust_tokenizer.decoder = decoders.ByteFallback()
    tokenizer = HfMarinTokenizer(
        _tokenizer=rust_tokenizer,
        _name_or_path="unicode-test",
        _bos_id=None,
        _eos_id=None,
        _pad_id=None,
        _bos_token=None,
        _eos_token=None,
        _chat_template=None,
        _vocab_size=4,
        _all_special_ids=[],
    )
    config = _exact_token_config()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _UnicodeTokenModel(), tokenizer)
    try:
        with TestClient(server.app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "gpt2",
                    "prompt": [1, 1],
                    "max_tokens": 3,
                    "temperature": 0,
                    "stream": True,
                    "return_token_ids": True,
                    "logprobs": 0,
                },
            )
        assert response.status_code == 200
        choices = [chunk["choices"][0] for chunk in _sse_chunks(response.text)]
        assert [token for choice in choices for token in choice["token_ids"]] == [2, 3, 0]
        assert "".join(choice["text"] for choice in choices) == "€"
        assert choices[-1]["finish_reason"] == "length"
    finally:
        server.inference_context.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remote_skyrl_pause_retries_exact_tokens_after_resume():
    """Run with the paired SkyRL checkout on PYTHONPATH and its client dependencies installed."""
    remote = pytest.importorskip("skyrl_train.inference_engines.remote_inference_engine")
    skyrl_client = pytest.importorskip("skyrl_train.inference_engines.inference_engine_client")
    policy = pytest.importorskip("skyrl_train.config.weight_sync_pause")
    omegaconf = pytest.importorskip("omegaconf")
    entered, release = threading.Event(), threading.Event()
    config = _exact_token_config()
    tokenizer = _AliasingChatTokenizer()
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, _BlockingTokenModel(entered, release), tokenizer)
    ready = asyncio.Event()
    http_server = _ReadyHttpServer(server.app, ready)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        serving = asyncio.create_task(http_server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            engine = remote.RemoteInferenceEngine(
                f"127.0.0.1:{listener.getsockname()[1]}",
                "gpt2",
                "vllm",
                tokenizer,
                weight_sync_pause_policy=policy.WeightSyncPausePolicy(policy.WeightSyncPauseMode.ABORT, True),
            )
            client = skyrl_client.InferenceEngineClient(
                engines=[engine],
                tokenizer=tokenizer,
                full_config=omegaconf.OmegaConf.create(
                    {
                        "trainer": {"policy": {"model": {"path": "gpt2"}}},
                        "generator": {
                            "backend": "vllm",
                            "enable_http_endpoint": False,
                            "http_endpoint_host": "127.0.0.1",
                            "http_endpoint_port": 0,
                            "weight_sync_pause_timeout_seconds": 30,
                        },
                    }
                ),
            )
            request = {"prompt_token_ids": [[0, 1]], "sampling_params": {"temperature": 0, "max_tokens": 3}}
            pending = asyncio.create_task(client.generate(request))
            pausing = None
            try:
                assert await asyncio.to_thread(entered.wait, 30)
                pausing = asyncio.create_task(client.pause_generation())
                assert await asyncio.to_thread(server.inference_context.pause_event.wait, 10)
                release.set()
                await pausing
                # The real client must park retries until resume; direct callers receive abort.
                paused = await engine.generate(request)
                assert paused["stop_reasons"] == ["abort"]
                assert paused["response_ids"] == paused["response_logprobs"] == [[]]
                assert not pending.done()
                await client.resume_generation()
                resumed = await asyncio.wait_for(pending, 30)
                full = await engine.generate(request)
                assert resumed["response_ids"] == full["response_ids"] == [[3, 1, 3]]
                assert resumed["response_logprobs"] == full["response_logprobs"]
                assert resumed["stop_reasons"] == full["stop_reasons"] == ["length"]
            finally:
                release.set()
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                if pausing is not None:
                    await asyncio.gather(pausing, return_exceptions=True)
        finally:
            release.set()
            http_server.should_exit = True
            await serving
            server.inference_context.shutdown()


@pytest.mark.parametrize("prompt", [[0, 1, 3, 2], [0, 1, 3, 2, 0, 1, 2, 3]])
def test_teacher_echo_scores_without_generating_at_context_limit(exact_token_client, prompt):
    response = exact_token_client.post(
        "/v1/completions",
        json={
            "model": "gpt2",
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 2,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    choice = payload["choices"][0]
    assert payload["usage"]["completion_tokens"] == 0
    assert choice["token_ids"] == []
    assert choice["logprobs"]["tokens"] == [f"token_id:{token}" for token in prompt]
    assert len(choice["logprobs"]["top_logprobs"]) == len(prompt)
    expected = [0.0, -math.log1p(3 * math.exp(-12)), -math.log1p(3 * math.exp(-11))]
    expected.extend([-math.log(4)] * (len(prompt) - 3))
    assert choice["logprobs"]["token_logprobs"] == pytest.approx(expected, abs=1e-6)
    for position, scores in enumerate(choice["logprobs"]["top_logprobs"][1:], 1):
        assert len(scores) == 2
        if position < 3:
            assert scores[f"token_id:{prompt[position]}"] == pytest.approx(expected[position], abs=1e-6)


@pytest.mark.integration
@pytest.mark.torch
@pytest.mark.asyncio
@pytest.mark.parametrize("evidence_kind", ["chosen_token", "topk_distribution"])
async def test_remote_skyrl_teacher_scores_exact_full_context_and_masked_rows(exact_token_server, evidence_kind):
    torch = pytest.importorskip("torch")
    specs = pytest.importorskip("marinskyrl.distillation")
    evidence_types = pytest.importorskip("skyrl_train.distillation")
    teacher_oracle = pytest.importorskip("skyrl_train.inference_engines.openai_teacher_oracle")
    kind = specs.TeacherEvidenceKind(evidence_kind)
    fingerprint = "sha256:" + "a" * 64
    teacher = specs.OpenAICompatibleTeacherSpec(
        id="teacher",
        source=specs.TeacherSource.OPENAI_COMPATIBLE,
        placement=specs.TeacherPlacement.EXTERNAL,
        model=specs.TeacherModelSpec(path="gpt2", revision="synthetic-v0"),
        evidence=kind,
        endpoints=(),
        tokenizer_fingerprint=fingerprint,
        max_sequence_length=8,
        request_timeout_seconds=30,
        top_k=2 if kind is specs.TeacherEvidenceKind.TOPK_DISTRIBUTION else None,
    )
    mask = torch.tensor([[True, True, False, False, False, False], [True] * 6])
    request = evidence_types.TeacherScoreRequest(
        trajectory_ids=("short", "full"),
        route_ids=("test", "test"),
        teacher_id="teacher",
        tokenizer_fingerprint=fingerprint,
        plan_version="test-v0",
        prompt_token_ids=torch.tensor([[0, 1], [0, 1]]),
        prompt_mask=torch.ones((2, 2), dtype=torch.bool),
        response_token_ids=torch.tensor([[3, 2, 0, 0, 0, 0], [3, 2, 0, 1, 2, 3]]),
        response_mask=mask,
        evidence=kind,
        top_k=teacher.top_k,
    )
    ready = asyncio.Event()
    http_server = _ReadyHttpServer(exact_token_server.app, ready)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        serving = asyncio.create_task(http_server.serve(sockets=[listener]))
        oracle = teacher_oracle.OpenAICompatibleTeacherOracle(
            teacher=teacher,
            endpoint=specs.TeacherEndpointSpec(
                url=f"http://127.0.0.1:{listener.getsockname()[1]}/v1", auth=None, max_concurrency=1
            ),
            api_key=None,
        )
        try:
            await asyncio.wait_for(ready.wait(), 10)
            evidence = await oracle.score(request)
        finally:
            await oracle.close()
            http_server.should_exit = True
            await serving
    assert torch.equal(evidence.valid_mask, mask)
    chosen_first = -math.log1p(3 * math.exp(-11))
    if kind is specs.TeacherEvidenceKind.CHOSEN_TOKEN:
        expected = torch.full((2, 6), -math.log(4))
        expected[:, 0] = chosen_first
        expected[~mask] = torch.nan
        torch.testing.assert_close(evidence.chosen_logprobs, expected, rtol=0, atol=1e-6, equal_nan=True)
    else:
        expected_ids = torch.tensor([[[3, 0]] + [[0, 1]] * 5] * 2)
        assert torch.equal(evidence.topk_indices[mask], expected_ids[mask])
        expected = torch.full((2, 6, 2), -math.log(4))
        expected[:, 0] = torch.tensor([chosen_first, chosen_first - 11])
        torch.testing.assert_close(evidence.topk_logprobs[mask], expected[mask], rtol=0, atol=1e-6)
