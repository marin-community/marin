# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import dataclasses
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Event

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.sampler import LogprobMode
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy_model import PolicyBatch, response_logprobs
from levanter.testing.devices import use_cpu_devices
from levanter.testing.helpers import skip_if_no_torch
from levanter.testing.model_configs import llama_test_config
from levanter.testing.tokenizer import llama3_style_chat_tokenizer
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from openai.types import Completion

    from levanter.inference.binary import (
        GENERATE_PATH,
        MEDIA_TYPE,
        PROMPT_TOKEN_IDS,
        choice_expert_routes,
        decode_frame,
        encode_frame,
    )
    from levanter.inference.engine import (
        InferenceEngineConfig,
        score_token_sequence_logprobs,
    )
    from levanter.inference.jit_scheduler import FinishReason
    from levanter.inference.openai import (
        InferenceResponse,
        InferenceServer,
        InferenceServerConfig,
        _compute_tokens,
    )
    from levanter.inference.snowball_publication import publish_snowball_checkpoint
    from levanter.inference.openai_protocol import ChatMessage
    from levanter.inference.policy_replay import BinaryRolloutChoice, prepare_binary_policy_batch

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


def test_shutdown_waits_for_active_generation(
    monkeypatch, trainer_config, inference_server_config, generated_model_and_tokenizer
):
    model, tokenizer = generated_model_and_tokenizer
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        server = InferenceServer.create(inference_server_config, model, tokenizer)

    ctx = server.inference_context
    started = threading.Event()
    unblock = threading.Event()
    shutdown_started = threading.Event()
    wait_timeout = 15
    original_execute = ctx._execute_batch

    def blocked_execute(requests, poll):
        started.set()
        assert unblock.wait(timeout=wait_timeout)
        original_execute(requests, poll)

    def shutdown():
        shutdown_started.set()
        server.shutdown()

    monkeypatch.setattr(ctx, "_execute_batch", blocked_execute)
    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as executor:
            result = executor.submit(
                client.post,
                "/v1/chat/completions",
                json={
                    "model": TEST_MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 2,
                    "seed": 483,
                },
            )
            try:
                assert started.wait(timeout=wait_timeout)
                stopping = executor.submit(shutdown)
                assert shutdown_started.wait(timeout=wait_timeout)
                with pytest.raises(TimeoutError):
                    stopping.result(timeout=2.5)
            finally:
                unblock.set()

            stopping.result(timeout=wait_timeout)
            assert result.result(timeout=wait_timeout).status_code == 200
    finally:
        server.shutdown()


def test_continuous_poll_reads_separate_ready_http_batches(monkeypatch, inference_server_config, local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)
    model_config = SnowballConfig(
        vocab_size=len(tokenizer),
        hidden_dim=16,
        intermediate_dim=32,
        shared_expert_intermediate_dim=24,
        num_layers=1,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=TEST_MAX_SEQ_LEN,
        sliding_window=4,
        attention_implementation="reference",
    )
    service = dataclasses.replace(inference_server_config.service, max_seqs=4, max_rounds=1, max_retained_prompts=8)
    config = dataclasses.replace(inference_server_config, service=service)
    mesh = compact_grug_mesh()
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", len(tokenizer)), model_config, key=jax.random.key(19))
        server = InferenceServer.create(config, model, tokenizer, mesh=mesh)

    ctx = server.inference_context
    entered = threading.Event()
    enqueued = threading.Event()
    resume = threading.Event()
    timeout = 25
    first_tokens_per_round = []
    put = ctx.batch_queue.put
    generate = ctx.engine.generate_continuously

    def put_and_signal(batch):
        put(batch)
        enqueued.set()

    def observe_generation(requests, *, on_tokens, **kwargs):
        entered.set()
        assert resume.wait(timeout=timeout)
        started_requests = set()

        def record_tokens(request_id, choice, tokens):
            started_requests.add(request_id)
            on_tokens(request_id, choice, tokens)

        generate(
            requests,
            on_tokens=record_tokens,
            step_callback=lambda _step: first_tokens_per_round.append(len(started_requests)),
            **kwargs,
        )

    monkeypatch.setattr(ctx.batch_queue, "put", put_and_signal)
    monkeypatch.setattr(ctx.engine, "generate_continuously", observe_generation)
    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=4) as executor:

            def submit():
                return executor.submit(
                    client.post,
                    "/v1/chat/completions",
                    json={
                        "model": TEST_MODEL_NAME,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                        "stream": True,
                        "seed": 483,
                    },
                )

            responses = [submit()]
            try:
                assert entered.wait(timeout=timeout) and enqueued.wait(timeout=timeout)
                # Send each request only after the admission thread has queued the preceding batch.
                for _ in range(3):
                    enqueued.clear()
                    responses.append(submit())
                    assert enqueued.wait(timeout=timeout)
            finally:
                resume.set()

            for response in responses:
                result = response.result(timeout=timeout)
                assert result.status_code == 200
                assert "data: [DONE]" in result.text
            assert first_tokens_per_round[:2] == [1, 4]
    finally:
        server.shutdown()


def test_models_endpoint_reports_the_configured_model(test_client):
    """A client discovering the server (OpenAI SDK, dashboards) reads the id it should send back."""
    client, server = test_client

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == [server.config.model_name]
    assert payload["data"][0]["max_model_len"] == server.inference_context.engine.config.max_seq_len


def test_chat_stream_matches_complete_text_for_each_choice(test_client):
    client, _ = test_client
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "seed": 483,
        "n": 2,
    }
    complete = client.post("/v1/chat/completions", json=payload)
    assert complete.status_code == 200, complete.text
    streamed = client.post(
        "/v1/chat/completions",
        json={
            **payload,
            "max_tokens": 2,
            "max_completion_tokens": payload["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert streamed.status_code == 200, streamed.text
    events = [line.removeprefix("data: ") for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == complete.json()["usage"]
    assert chunks[-1]["usage"]["completion_tokens"] == 10
    for expected in complete.json()["choices"]:
        deltas = [row for chunk in chunks for row in chunk["choices"] if row["index"] == expected["index"]]
        assert "".join(row["delta"].get("content") or "" for row in deltas) == expected["message"]["content"]
        assert deltas[0]["delta"]["role"] == "assistant"
        assert deltas[-1]["finish_reason"] == expected["finish_reason"]


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

    @asynccontextmanager
    async def http_generation(self, _request):
        yield Event()

    def submit_request(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        stop_tokens: list[int] | None,
        seed: int | None,
        future,
        n_generations: int = 1,
        echo_logprobs_top_k: int | None = None,
        *,
        abort_event: Event | None = None,
    ) -> str:
        del abort_event
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


def test_completion_echo_logprobs_are_lm_eval_aligned():
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
            },
        )

    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    expected_prompt_logprob = float(jax.nn.log_softmax(jnp.array([-8.0, 4.0, -8.0, -8.0]))[1])
    expected_completion_logprob = float(jax.nn.log_softmax(jnp.array([-8.0, -8.0, -8.0, 3.0]))[3])

    assert choice["text"] == "A B X"
    assert logprobs["tokens"] == ["A", " B", " X"]
    assert logprobs["token_logprobs"] == pytest.approx([0.0, expected_prompt_logprob, expected_completion_logprob])
    assert logprobs["text_offset"] == [0, 1, 3]
    assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"])
    assert len(logprobs["tokens"]) == len(logprobs["top_logprobs"])
    assert logprobs["top_logprobs"][0] == {"A": 0.0}
    assert logprobs["top_logprobs"][1][" B"] == pytest.approx(expected_prompt_logprob)
    assert logprobs["top_logprobs"][2][" X"] == pytest.approx(expected_completion_logprob)


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


def test_binary_generation_returns_exact_token_ids_and_scores(test_client):
    client, _ = test_client
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)
    body = encode_frame({"max_tokens": 2, "temperature": 0, "seed": 17}, {PROMPT_TOKEN_IDS: prompt})
    response = client.post(GENERATE_PATH, content=body, headers={"content-type": MEDIA_TYPE})
    assert response.status_code == 200, response.text
    metadata, tensors = decode_frame(response.content)

    np.testing.assert_array_equal(tensors[PROMPT_TOKEN_IDS], prompt)
    assert metadata["choices"][0]["completion_tokens"] == 2
    assert metadata["choices"][0]["finish_reason"] == "length"
    assert tensors["choice.0.token_ids"].shape == (2,)
    assert tensors["choice.0.token_ids"].dtype == np.int32
    assert tensors["choice.0.behavior_logprobs"].shape == (2,)
    assert np.isfinite(tensors["choice.0.behavior_logprobs"]).all()

    wrong_dtype = encode_frame({"max_tokens": 2}, {PROMPT_TOKEN_IDS: prompt.astype(np.float32)})
    response = client.post(GENERATE_PATH, content=wrong_dtype, headers={"content-type": MEDIA_TYPE})
    assert response.status_code == 400


def test_binary_generation_returns_unwarped_policy_logprobs(inference_server_config, generated_model_and_tokenizer):
    model, tokenizer = generated_model_and_tokenizer
    config = dataclasses.replace(
        inference_server_config,
        service=dataclasses.replace(inference_server_config.service, logprob_mode=LogprobMode.POLICY),
    )
    model = jax.tree.map(jnp.zeros_like, model)
    server = InferenceServer.create(config, model, tokenizer)
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)
    fields = {"max_tokens": 1, "temperature": 0.7, "top_p": 0.05, "seed": 17}
    try:
        with TestClient(server.app) as client:
            response = client.post(
                GENERATE_PATH,
                content=encode_frame(fields | {"expected_logprob_mode": "policy"}, {"prompt_token_ids": prompt}),
                headers={"content-type": MEDIA_TYPE},
            )
            assert response.status_code == 200, response.text
            metadata, tensors = decode_frame(response.content)
            assert metadata["logprob_mode"] == "policy"
            assert "choice.0.behavior_logprobs" not in tensors
            expected = -np.log(model.Vocab.size)
            np.testing.assert_allclose(tensors["choice.0.policy_logprobs"], [expected], atol=1e-5, rtol=0)

            mismatch = client.post(
                GENERATE_PATH,
                content=encode_frame(fields | {"expected_logprob_mode": "behavior"}, {"prompt_token_ids": prompt}),
                headers={"content-type": MEDIA_TYPE},
            )
            assert mismatch.status_code == 400
    finally:
        server.shutdown()


def test_binary_generation_fences_rollouts_by_installed_policy(inference_server_config, generated_model_and_tokenizer):
    model, tokenizer = generated_model_and_tokenizer
    server = InferenceServer.create(inference_server_config, model, tokenizer)
    try:
        with TestClient(server.app) as client:
            _assert_binary_policy_fence(client, server)
    finally:
        server.shutdown()


def _assert_binary_policy_fence(client, server):
    prompt = np.array([18, 7, 3], dtype=np.int32)

    def generate(policy):
        return client.post(
            GENERATE_PATH,
            content=encode_frame(
                {"max_tokens": 2, "temperature": 0, "expected_policy_id": policy}, {"prompt_token_ids": prompt}
            ),
            headers={"content-type": MEDIA_TYPE},
        )

    engine = server.inference_context.engine
    page_buffers = jax.tree.leaves(engine.gen_state.cache)
    previous_storage = [buffer.addressable_data(0).unsafe_buffer_pointer() for buffer in page_buffers]
    assert server.reload(lambda model: model, expected_version=0, policy_id="policy-a") == 1
    new_buffers = jax.tree.leaves(server.inference_context.engine.gen_state.cache)
    assert [buffer.addressable_data(0).unsafe_buffer_pointer() for buffer in new_buffers] == previous_storage
    first_metadata, first_tensors = decode_frame(generate("policy-a").content)
    assert first_metadata["choices"][0] == {
        "completion_tokens": 2,
        "finish_reason": "length",
        "model_version": 1,
        "policy_id": "policy-a",
    }

    def fail_load(model):
        raise RuntimeError("invalid replacement")

    with pytest.raises(RuntimeError, match="invalid replacement"):
        server.reload(fail_load, expected_version=1, policy_id="policy-b")
    still_first, still_first_tensors = decode_frame(generate("policy-a").content)
    assert still_first == first_metadata
    np.testing.assert_array_equal(still_first_tensors["choice.0.token_ids"], first_tensors["choice.0.token_ids"])

    assert server.reload(lambda model: model, expected_version=1, policy_id="policy-b") == 2
    with pytest.raises(ValueError):
        server.reload(lambda model: model, expected_version=1, policy_id="policy-c")
    with pytest.raises(ValueError):
        server.reload(lambda model: model, expected_version=2, policy_id="policy-a")
    assert generate("policy-a").status_code == 409
    second_metadata, _ = decode_frame(generate("policy-b").content)
    assert second_metadata["choices"][0] == {
        "completion_tokens": 2,
        "finish_reason": "length",
        "model_version": 2,
        "policy_id": "policy-b",
    }


def test_binary_snowball_generation_replays_exact_expert_routes(local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)
    cfg = SnowballConfig(
        vocab_size=len(tokenizer),
        hidden_dim=16,
        intermediate_dim=32,
        shared_expert_intermediate_dim=24,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=32,
        sliding_window=4,
        attention_implementation="reference",
    )
    server_config = InferenceServerConfig(
        initial_policy_id="test-snowball-policy",
        service=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=16,
            max_seqs=4,
            max_seqs_in_prefill=4,
            max_prefill_size=24,
            max_queued_tokens=8,
            max_rounds=2,
            page_size=2,
            compute_dtype=jnp.float32,
            logprob_mode=LogprobMode.POLICY,
        ),
    )
    prompt = np.array([2, 7, 4, 9, 6], dtype=np.int32)
    with use_cpu_devices(1):
        mesh = compact_grug_mesh()
        with jax.set_mesh(mesh):
            model = SnowballLMHeadModel.init(hax.Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(19))
            server = InferenceServer.create(server_config, model, tokenizer, mesh=mesh)
            try:
                with TestClient(server.app) as client:
                    body = encode_frame(
                        {"max_tokens": 4, "temperature": 0, "return_route_captures": True},
                        {"prompt_token_ids": prompt},
                    )
                    response = client.post(GENERATE_PATH, content=body, headers={"content-type": MEDIA_TYPE})
                    assert response.status_code == 200, response.text
                    _, tensors = decode_frame(response.content)
                    rollout = BinaryRolloutChoice(response.content, 0)
                    messages = [{"role": "user", "content": "hi"}]
                    chat = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": server.config.model_name,
                            "messages": messages,
                            "max_tokens": 2,
                            "temperature": 0,
                            "stream": True,
                            "return_route_captures": True,
                        },
                    )
                    assert chat.status_code == 200
                    events = [json.loads(line[6:]) for line in chat.text.splitlines() if line.startswith("data: {")]
                    (final,) = [event for event in events if "levanter_route_frame" in event]
                    frame = base64.b64decode(final["levanter_route_frame"], validate=True)
                    metadata, stream_tensors = decode_frame(frame)
                    expected_prompt = _compute_tokens([ChatMessage(**messages[0])], tokenizer, None, None)
                    np.testing.assert_array_equal(stream_tensors["prompt_token_ids"], expected_prompt)
                    streamed_ids = stream_tensors["choice.0.token_ids"]
                    assert streamed_ids.size == metadata["choices"][0]["completion_tokens"] == 2
                    _, stream_valid = choice_expert_routes(
                        stream_tensors, 0, np.concatenate((expected_prompt, streamed_ids))
                    )
                    assert stream_valid[:-1].all() and not stream_valid[-1]

            finally:
                server.inference_context.shutdown()

            inputs = np.concatenate((prompt, tensors["choice.0.token_ids"]))
            experts, valid = choice_expert_routes(tensors, 0, inputs)
            tokens = hax.named(jnp.asarray(inputs[None]), ("batch", "position"))
            replay = model.prepare_route_replay(jnp.asarray(experts[:, None]), hax.named(valid[None], tokens.axes))
            learner_loss, _ = model.next_token_loss_and_metrics(
                LmExample(tokens, hax.ones(tokens.axes)), route_replay=replay, reduction=None
            )

            padded = hax.named(jnp.asarray(np.pad(inputs, (2, 1))[None]), ("batch", "position"))
            attended = hax.named(jnp.asarray([[0, 0] + [1] * len(inputs) + [0]]), padded.axes)
            response_axes = (padded.resolve_axis("batch"), hax.Axis("response", 5))
            objective = hax.named(jnp.asarray([[1, 1, 1, 1, 0]], dtype=jnp.float32), response_axes)
            batch = PolicyBatch(
                tokens=padded,
                attention_mask=attended,
                position_ids=hax.maximum(hax.cumsum(attended, "position") - 1, 0),
                response_mask=objective,
                advantages=objective,
                policy_weights=objective / 4,
                kl_weights=objective / 4,
                old_logprobs=hax.zeros(response_axes),
                reference_logprobs=None,
            )
            prepared = prepare_binary_policy_batch(model, batch, [rollout], policy_id="test-snowball-policy")
            np.testing.assert_array_equal(prepared.old_logprobs.array[0, :-1], tensors["choice.0.policy_logprobs"])
            np.testing.assert_allclose(
                response_logprobs(model, prepared, block_size=8).array[0, :-1],
                tensors["choice.0.policy_logprobs"],
                atol=1e-5,
                rtol=0,
            )
            with pytest.raises(ValueError):
                prepare_binary_policy_batch(model, batch, [rollout], policy_id="stale-policy")

    np.testing.assert_allclose(
        tensors["choice.0.policy_logprobs"], -learner_loss.array[0, len(prompt) - 1 : -1], atol=1e-5, rtol=0
    )
    wrong_inputs = inputs.copy()
    wrong_inputs[0] += 1
    with pytest.raises(ValueError):
        choice_expert_routes(tensors, 0, wrong_inputs)


def test_paused_snowball_replicas_converge_after_an_incomplete_checkpoint(tmp_path, local_gpt2_tokenizer):
    cfg = SnowballConfig(
        vocab_size=48,
        hidden_dim=16,
        intermediate_dim=32,
        shared_expert_intermediate_dim=24,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
    )
    options = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=16,
            max_pages=16,
            max_seqs=2,
            max_rounds=2,
            page_size=2,
            max_seqs_in_prefill=2,
            max_prefill_size=12,
            max_queued_tokens=4,
            max_retained_prompts=2,
            logprob_mode=LogprobMode.POLICY,
            compute_dtype=jnp.float32,
        ),
        initial_policy_id="starting",
    )
    with use_cpu_devices(1), jax.set_mesh(compact_grug_mesh()):
        mesh = compact_grug_mesh()
        vocab = hax.Axis("vocab", cfg.vocab_size)
        initial = SnowballLMHeadModel.init(vocab, cfg, key=jax.random.key(18)).with_packed_experts()
        learned = SnowballLMHeadModel.init(vocab, cfg, key=jax.random.key(21))
        path = str(tmp_path / "native-policy")
        save_checkpoint({"model": learned}, 3, path, metadata={"policy_version": 2, "policy_id": "learned:2"})
        incomplete = str(tmp_path / "incomplete-policy")
        save_checkpoint(
            {"model": {"other": jnp.array(0)}},
            3,
            incomplete,
            metadata={"policy_version": 2, "policy_id": "learned:2"},
        )
        server = InferenceServer.create(options, initial, local_gpt2_tokenizer, mesh=mesh)
        peer = InferenceServer.create(options, initial, local_gpt2_tokenizer, mesh=mesh)
        try:
            with TestClient(server.app) as client, TestClient(peer.app) as peer_client:
                prompt = np.array([2, 5, 7], dtype=np.int32)

                def generate(policy_id, target=client):
                    request = encode_frame(
                        {
                            "max_tokens": 3,
                            "temperature": 0,
                            "return_route_captures": True,
                            "expected_policy_id": policy_id,
                        },
                        {"prompt_token_ids": prompt},
                    )
                    return target.post(GENERATE_PATH, content=request, headers={"content-type": MEDIA_TYPE})

                before = generate("starting")
                assert before.status_code == 200
                with pytest.raises(ValueError):
                    publish_snowball_checkpoint(server, path, policy_version=1, expected_serving_version=0)
                assert generate("starting", peer_client).status_code == 200

                for target in (client, peer_client):
                    paused = target.post("/pause_generation", json={"mode": "abort", "clear_cache": True})
                    assert paused.status_code == 200, paused.text

                assert publish_snowball_checkpoint(server, path, policy_version=2, expected_serving_version=0) == 1
                with pytest.raises(FileNotFoundError):
                    publish_snowball_checkpoint(peer, incomplete, policy_version=2, expected_serving_version=0)
                assert generate("starting").status_code == 409
                assert generate("learned:2", peer_client).status_code == 409
                for target, policy_id in ((client, "learned:2"), (peer_client, "starting")):
                    reply = generate(policy_id, target)
                    assert reply.status_code == 200, reply.text
                    receipt, arrays = decode_frame(reply.content)
                    assert receipt["choices"][0]["policy_id"] == policy_id
                    assert receipt["choices"][0]["finish_reason"] == "abort"
                    assert not arrays["choice.0.token_ids"].size

                assert publish_snowball_checkpoint(peer, path, policy_version=2, expected_serving_version=0) == 1
                for target in (client, peer_client):
                    assert target.post("/resume_generation").status_code == 200
                result = generate("learned:2")
                assert result.status_code == 200
                metadata, tensors = decode_frame(result.content)
                assert metadata["choices"][0]["policy_id"] == "learned:2"
                reply = generate("learned:2", peer_client)
                assert reply.status_code == 200, reply.text
                peer_metadata, peer_tensors = decode_frame(reply.content)
                assert peer_metadata["choices"] == metadata["choices"]
                np.testing.assert_array_equal(peer_tensors["choice.0.token_ids"], tensors["choice.0.token_ids"])
                np.testing.assert_array_equal(
                    peer_tensors["choice.0.policy_logprobs"], tensors["choice.0.policy_logprobs"]
                )
                _, previous = decode_frame(before.content)
                assert not np.array_equal(tensors["choice.0.policy_logprobs"], previous["choice.0.policy_logprobs"])
                tokens = np.r_[prompt, tensors["choice.0.token_ids"]]
                experts, valid = choice_expert_routes(tensors, 0, tokens)
                inputs = hax.named(jnp.asarray(tokens[None]), ("batch", "position"))
                replay = learned.prepare_route_replay(
                    jnp.asarray(experts[:, None]), hax.named(valid[None], inputs.axes)
                )
                loss, _ = learned.next_token_loss_and_metrics(
                    LmExample(inputs, hax.ones(inputs.axes)), route_replay=replay, reduction=None
                )
                np.testing.assert_allclose(
                    tensors["choice.0.policy_logprobs"], -loss.array[0, prompt.size - 1 : -1], rtol=0, atol=1e-5
                )
        finally:
            peer.shutdown()
            server.shutdown()


@pytest.fixture
def greedy_client(inference_server_config, local_gpt2_tokenizer, fixed_token_model):
    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)
    token_ids = tokenizer.encode("a", add_special_tokens=False)
    assert len(token_ids) == 1
    model = fixed_token_model(len(tokenizer), token_ids[0])
    service = dataclasses.replace(inference_server_config.service, max_pages=32)
    config = dataclasses.replace(inference_server_config, service=service)
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, model, tokenizer)  # type: ignore[arg-type]
    try:
        with TestClient(server.app) as client:
            yield client
    finally:
        server.shutdown()


@pytest.mark.parametrize("chat", [False, True])
def test_openai_generation_reports_stop_at_the_token_limit(greedy_client, chat):
    client = greedy_client
    path = "/v1/chat/completions" if chat else "/v1/completions"
    payload = {
        "model": TEST_MODEL_NAME,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 17,
        "n": 2,
    }
    if chat:
        payload["messages"] = [{"role": "user", "content": "Hello, my name is"}]
    else:
        payload["prompt"] = "Hello, my name is"

    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert [choice["finish_reason"] for choice in response.json()["choices"]] == ["length", "length"]

    stopped = client.post(path, json={**payload, "stop": ["A B", "a"]})
    assert stopped.status_code == 200, stopped.text
    assert [choice["finish_reason"] for choice in stopped.json()["choices"]] == ["stop", "stop"]


def test_binary_multi_choice_generation_reports_stop_at_the_token_limit(greedy_client):
    client = greedy_client
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)
    options = {"max_tokens": 1, "temperature": 0, "seed": 17, "n": 2}

    def generate(metadata):
        response = client.post(
            GENERATE_PATH,
            content=encode_frame(metadata, {PROMPT_TOKEN_IDS: prompt}),
            headers={"content-type": MEDIA_TYPE},
        )
        assert response.status_code == 200, response.text
        return decode_frame(response.content)

    limited, limited_tensors = generate(options)
    token = int(limited_tensors["choice.0.token_ids"][0])
    other = (token + 1) % 19
    stopped, stopped_tensors = generate({**options, "stop_tokens": [[other, other], [token]]})

    assert [choice["finish_reason"] for choice in limited["choices"]] == ["length", "length"]
    assert [choice["finish_reason"] for choice in stopped["choices"]] == ["stop", "stop"]
    for i in range(2):
        for name in ("token_ids", "behavior_logprobs"):
            key = f"choice.{i}.{name}"
            np.testing.assert_array_equal(stopped_tensors[key], limited_tensors[key])


PREFILL_GATE_TIMEOUT = 30
ADMISSION_TIMEOUT = 5


@pytest.fixture
def gated_binary_server(inference_server_config, local_gpt2_tokenizer, fixed_token_model):
    started = Event()
    release = Event()
    registered = {request_id: Event() for request_id in ("running", "queued", "disconnecting")}
    disconnect = Event()
    disconnect_observed = Event()

    def gate(_positions):
        if not started.is_set():
            started.set()
            if not release.wait(PREFILL_GATE_TIMEOUT):
                raise TimeoutError("Test did not release the blocked prefill")

    class GatedModel(fixed_token_model):
        def decode(self, input_ids, kv_cache, batch_info, pos_ids):
            jax.debug.callback(gate, pos_ids.array)
            return super().decode(input_ids, kv_cache, batch_info, pos_ids)

    tokenizer = local_gpt2_tokenizer.with_chat_template(TEST_CHAT_TEMPLATE)
    model = GatedModel(vocab_size=len(tokenizer), token_id=3)
    config = dataclasses.replace(
        inference_server_config,
        service=dataclasses.replace(inference_server_config.service, max_seqs=1, max_rounds=1, max_pages=16),
        initial_policy_id="policy-0",
    )
    server = InferenceServer.create(config, model, tokenizer)  # type: ignore[arg-type]

    async def observe_registration(scope, receive, send):
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode()
        read_body = False

        async def receive_request():
            nonlocal read_body
            if read_body and request_id in registered:
                registered[request_id].set()
            if read_body and request_id == "disconnecting" and await asyncio.to_thread(disconnect.wait, 30):
                asyncio.get_running_loop().call_soon(disconnect_observed.set)
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                read_body = True
            return message

        await server.app(scope, receive_request, send)

    try:
        with TestClient(observe_registration) as client:
            yield client, started, release, registered, disconnect, disconnect_observed
    finally:
        release.set()
        disconnect.set()
        server.shutdown()


def _generate_policy_zero(client, ids, request_id, max_tokens=4):
    response = client.post(
        GENERATE_PATH,
        content=encode_frame(
            {"max_tokens": max_tokens, "temperature": 0, "expected_policy_id": "policy-0"},
            {PROMPT_TOKEN_IDS: ids},
        ),
        headers={"content-type": MEDIA_TYPE, "x-request-id": request_id},
    )
    assert response.status_code == 200, response.text
    return decode_frame(response.content)


def test_binary_abort_preserves_exact_running_output_and_cancels_queued_work(gated_binary_server):
    client, started, release, registered, _, _ = gated_binary_server
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            running = pool.submit(_generate_policy_zero, client, prompt, "running")
            assert started.wait(PREFILL_GATE_TIMEOUT)
            queued = pool.submit(_generate_policy_zero, client, prompt, "queued")
            assert registered["queued"].wait(ADMISSION_TIMEOUT)
            abort = client.post("/abort", json={"request_ids": ["running", "queued"]})
            assert abort.status_code == 200
            empty_metadata, empty_tensors = queued.result(timeout=ADMISSION_TIMEOUT)
            assert empty_metadata["choices"] == [
                {"finish_reason": "abort", "completion_tokens": 0, "policy_id": "policy-0", "model_version": 0}
            ]
            assert not empty_tensors["choice.0.token_ids"].size
            assert not empty_tensors["choice.0.behavior_logprobs"].size
        finally:
            release.set()
        partial_metadata, partial = running.result(timeout=PREFILL_GATE_TIMEOUT)

    assert partial_metadata["choices"] == [
        {"finish_reason": "abort", "completion_tokens": 1, "policy_id": "policy-0", "model_version": 0}
    ]
    ids_key, score_key = "choice.0.token_ids", "choice.0.behavior_logprobs"
    _, resumed = _generate_policy_zero(client, np.concatenate((prompt, partial[ids_key])), "resumed", max_tokens=3)
    _, uninterrupted = _generate_policy_zero(client, prompt, "uninterrupted")
    for key in (ids_key, score_key):
        np.testing.assert_array_equal(np.concatenate((partial[key], resumed[key])), uninterrupted[key])


def test_binary_disconnect_releases_running_generation(gated_binary_server):
    client, started, release, registered, disconnect, disconnect_observed = gated_binary_server
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)
    body = encode_frame({"max_tokens": 4, "temperature": 0}, {PROMPT_TOKEN_IDS: prompt})

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            running = pool.submit(
                client.post,
                GENERATE_PATH,
                content=body,
                headers={"content-type": MEDIA_TYPE, "x-request-id": "disconnecting"},
            )
            assert started.wait(PREFILL_GATE_TIMEOUT)
            assert registered["disconnecting"].wait(ADMISSION_TIMEOUT)
            disconnect.set()
            assert disconnect_observed.wait(ADMISSION_TIMEOUT)
        finally:
            disconnect.set()
            release.set()
        response = running.result(timeout=PREFILL_GATE_TIMEOUT)

    assert response.status_code == 200, response.text
    metadata, tensors = decode_frame(response.content)
    assert metadata["choices"][0]["finish_reason"] == "abort"
    assert tensors["choice.0.token_ids"].tolist() == [3]

    next_response = client.post(GENERATE_PATH, content=body, headers={"content-type": MEDIA_TYPE})
    assert next_response.status_code == 200, next_response.text
    metadata, tensors = decode_frame(next_response.content)
    assert metadata["choices"][0]["finish_reason"] == "length"
    assert tensors["choice.0.token_ids"].tolist() == [3] * 4


@pytest.mark.parametrize("stream", [False, True])
def test_http_abort_reason_is_serialized_for_queued_completions(gated_binary_server, stream):
    client, started, release, registered, _, _ = gated_binary_server
    body = encode_frame(
        {"max_tokens": 4, "temperature": 0},
        {PROMPT_TOKEN_IDS: np.array([18, 7, 3, 18], dtype=np.int32)},
    )
    completion = {"model": TEST_MODEL_NAME, "prompt": "A B", "temperature": 0, "max_tokens": 2, "stream": stream}
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            running = pool.submit(client.post, GENERATE_PATH, content=body, headers={"content-type": MEDIA_TYPE})
            assert started.wait(PREFILL_GATE_TIMEOUT)
            queued = pool.submit(client.post, "/v1/completions", json=completion, headers={"x-request-id": "queued"})
            assert registered["queued"].wait(ADMISSION_TIMEOUT)
            assert client.post("/abort", json={"request_ids": ["queued"]}).status_code == 200
            response = queued.result(timeout=ADMISSION_TIMEOUT)
            assert response.status_code == 200, response.text
            if stream:
                data = next(
                    line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")
                )
                payload = json.loads(data)
            else:
                payload = response.json()
            assert payload["choices"][0]["finish_reason"] == "abort"
            assert payload["choices"][0]["text"] == ""
        finally:
            release.set()
        assert running.result(timeout=30).status_code == 200


def test_pause_discards_queued_work_and_resumes_from_exact_partial_ids(gated_binary_server):
    client, started, release, registered, _, _ = gated_binary_server
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)

    with ThreadPoolExecutor(max_workers=4) as pool:
        try:
            running = pool.submit(_generate_policy_zero, client, prompt, "running")
            assert started.wait(PREFILL_GATE_TIMEOUT)
            queued = pool.submit(_generate_policy_zero, client, prompt, "queued")
            assert registered["queued"].wait(ADMISSION_TIMEOUT)
            pause = pool.submit(client.post, "/pause_generation", json={"mode": "abort", "clear_cache": True})
            queued_metadata, queued_tensors = queued.result(timeout=ADMISSION_TIMEOUT)
            assert queued_metadata["choices"][0]["finish_reason"] == "abort"
            assert not queued_tensors["choice.0.token_ids"].size
            assert not queued_tensors["choice.0.behavior_logprobs"].size
            during_pause, _ = _generate_policy_zero(client, prompt, "during-pause")
            assert during_pause["choices"][0]["finish_reason"] == "abort"
            assert during_pause["choices"][0]["completion_tokens"] == 0
            assert not pause.done()
            resume = pool.submit(client.post, "/resume_generation")
        finally:
            release.set()
        assert pause.result(timeout=PREFILL_GATE_TIMEOUT).status_code == 200
        assert resume.result(timeout=PREFILL_GATE_TIMEOUT).status_code == 200
        partial_metadata, partial = running.result(timeout=PREFILL_GATE_TIMEOUT)

    assert partial_metadata["choices"][0]["finish_reason"] == "abort"
    assert partial_metadata["choices"][0]["policy_id"] == "policy-0"
    ids_key, score_key = "choice.0.token_ids", "choice.0.behavior_logprobs"
    assert partial[ids_key].tolist() == [3]
    _, resumed = _generate_policy_zero(client, np.concatenate((prompt, partial[ids_key])), "resumed", max_tokens=3)
    uninterrupted_metadata, uninterrupted = _generate_policy_zero(client, prompt, "uninterrupted")
    assert uninterrupted_metadata["choices"][0]["finish_reason"] == "length"
    for key in (ids_key, score_key):
        np.testing.assert_array_equal(np.concatenate((partial[key], resumed[key])), uninterrupted[key])


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "keep", "clear_cache": True},
        {"mode": "abort", "clear_cache": False},
        {"mode": "abort", "clear_cache": "yes"},
    ],
)
def test_unsupported_pause_policy_keeps_admission_open(greedy_client, policy):
    assert greedy_client.post("/pause_generation", json=policy).status_code == 422
    response = greedy_client.post(
        "/v1/completions", json={"model": TEST_MODEL_NAME, "prompt": "A B", "max_tokens": 1, "temperature": 0}
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_staged_weights_leave_admission_open_and_recheck_version(
    inference_server_config, llama_model_config, local_gpt2_tokenizer
):
    tokenizer = local_gpt2_tokenizer
    config = dataclasses.replace(inference_server_config, initial_policy_id="initial")
    mesh = config.trainer.device_mesh
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(config.trainer.compute_axis_mapping):
        model = LlamaLMHeadModel.init(hax.Axis("vocab", len(tokenizer)), llama_model_config, key=jax.random.key(0))
        server = InferenceServer.create(config, model, tokenizer, mesh=mesh)
    entered, release = threading.Event(), threading.Event()
    prompt = np.array([18, 7, 3, 18], dtype=np.int32)

    def restore(model):
        entered.set()
        if not release.wait(30):
            raise TimeoutError("Test did not release checkpoint staging")
        return jax.tree.map(jnp.zeros_like, model)

    def generate(client, policy_id):
        response = client.post(
            GENERATE_PATH,
            content=encode_frame(
                {"max_tokens": 2, "temperature": 0, "expected_policy_id": policy_id},
                {"prompt_token_ids": prompt},
            ),
            headers={"content-type": MEDIA_TYPE},
        )
        return response

    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
            try:
                staging = pool.submit(server.publish, restore, expected_version=0, policy_id="staged")
                assert entered.wait(5)
                during_stage = pool.submit(generate, client, "initial").result(timeout=30)
                assert during_stage.status_code == 200, during_stage.text
                during_metadata, _ = decode_frame(during_stage.content)
                assert during_metadata["choices"][0]["model_version"] == 0
                assert server.reload(lambda model: model, expected_version=0, policy_id="next") == 1
            finally:
                release.set()
            with pytest.raises(ValueError, match="Expected model version"):
                staging.result(timeout=30)
            with pytest.raises(ValueError, match="new, nonempty policy ID"):
                server.publish(lambda model: model, expected_version=1, policy_id="next")
            assert generate(client, "staged").status_code == 409
            response = generate(client, "next")
            assert response.status_code == 200
            metadata, _ = decode_frame(response.content)
            assert metadata["choices"][0]["model_version"] == 1
            assert metadata["choices"][0]["policy_id"] == "next"
    finally:
        server.shutdown()


def test_chat_generation_reports_stop_on_an_end_of_turn_token(tmp_path, inference_server_config, fixed_token_model):
    llama3_style_chat_tokenizer().save_pretrained(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    model = fixed_token_model(len(tokenizer), tokenizer.get_vocab()["<|eot_id|>"])
    options = dataclasses.replace(
        inference_server_config,
        service=dataclasses.replace(inference_server_config.service, max_pages=32),
    )
    with options.trainer.use_device_mesh(), hax.axis_mapping(options.trainer.compute_axis_mapping):
        server = InferenceServer.create(options, model, tokenizer)  # type: ignore[arg-type]
    try:
        with TestClient(server.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": TEST_MODEL_NAME,
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0,
                    "max_tokens": 1,
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        assert response.json()["usage"]["completion_tokens"] == 1
    finally:
        server.shutdown()


def test_live_chat_abort_returns_terminal_sse_event(gated_binary_server):
    client, started, release, registered, _, _ = gated_binary_server
    payload = {
        "model": TEST_MODEL_NAME,
        "temperature": 0,
        "max_tokens": 4,
        "stream": True,
        "messages": [{"role": "user", "content": "A B"}],
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            pending = pool.submit(
                client.post, "/v1/chat/completions", json=payload, headers={"x-request-id": "running"}
            )
            assert started.wait(PREFILL_GATE_TIMEOUT)
            assert registered["running"].wait(ADMISSION_TIMEOUT)
            assert client.post("/abort", json={"request_ids": ["running"]}).status_code == 200
        finally:
            release.set()
        response = pending.result(timeout=PREFILL_GATE_TIMEOUT)

    assert response.status_code == 200, response.text
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "abort"
    assert "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)
