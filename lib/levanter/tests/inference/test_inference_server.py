# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest

from levanter.inference.jit_scheduler import FinishReason
from levanter.layers.kv_cache import KvPageCache
from levanter.models.llama import LlamaLMHeadModel
from levanter.testing.helpers import skip_if_no_torch
from levanter.testing.model_configs import llama_test_config
from levanter.trainer import TrainerConfig

try:
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


class _FakeCompletionContext:
    def __init__(self, max_seq_len: int = 4096):
        self.config = InferenceServerConfig(service=InferenceEngineConfig(max_seq_len=max_seq_len))
        self.model = _DeterministicCompletionScoringModel()
        self.tokenizer = _OpenAITestTokenizer()
        self.submitted_requests = 0

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

    server.reload(_new_model)

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

    server.reload(_original_model)
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
    chunks = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: {")]
    assert chunks[0]["prompt_token_ids"] == full["prompt_token_ids"]
    assert chunks[0]["choices"][0]["token_ids"] == expected["token_ids"]
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
            pausing = pool.submit(server.pause_generation)
            assert server.inference_context.pause_event.wait(5)
            release.set()
            pausing.result(timeout=30)
            response = pending.result(timeout=30)
            assert response.status_code == 200, response.text
            if stream:
                chunks = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
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
            server.resume_generation()
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
