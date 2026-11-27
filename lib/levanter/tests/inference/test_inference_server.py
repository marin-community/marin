# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor

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
from levanter.trainer import TrainerConfig

try:
    import grpc
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from openai.types import Completion

    from levanter.inference import inference_pb2, inference_pb2_grpc
    from levanter.inference.engine import (
        InferenceEngineConfig,
        score_token_sequence_logprobs,
    )
    from levanter.inference.openai import (
        InferenceResponse,
        InferenceServer,
        InferenceServerConfig,
        _compute_tokens,
    )
    from levanter.inference.snowball_publication import publish_snowball_checkpoint
    from levanter.inference.openai_protocol import ChatMessage
    from levanter.inference.policy_replay import RolloutChoice, prepare_rollout_policy_batch
    from levanter.inference.router_replay import read_router_replay

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
        host="127.0.0.1",
        grpc_port=0,
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
        "/v1/chat/completions", json={**payload, "stream": True, "stream_options": {"include_usage": True}}
    )
    assert streamed.status_code == 200, streamed.text
    events = [line.removeprefix("data: ") for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == complete.json()["usage"]
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


def test_grpc_generation_returns_exact_token_ids_and_scores(test_client):
    _, server = test_client
    prompt = [18, 7, 3, 18]
    with grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)
        result = stub.Generate(
            inference_pb2.GenerateRequest(
                prompt_token_ids=prompt, max_new_tokens=2, temperature=0, seed=17, num_choices=2
            ),
            timeout=30,
        )
        assert list(result.prompt_token_ids) == prompt
        assert len(result.choices) == 2
        for choice in result.choices:
            assert len(choice.token_ids) == len(choice.behavior_logprobs) == 2
            assert np.isfinite(choice.behavior_logprobs).all()

        stop = result.choices[0].token_ids[0]
        other = (stop + 1) % server.inference_context.model.Vocab.size
        stopped = stub.Generate(
            inference_pb2.GenerateRequest(
                prompt_token_ids=prompt,
                max_new_tokens=2,
                temperature=0,
                seed=17,
                num_choices=2,
                stop_sequences=[
                    inference_pb2.TokenSequence(token_ids=[other, other]),
                    inference_pb2.TokenSequence(token_ids=[stop]),
                ],
            ),
            timeout=30,
        )
        assert [list(choice.token_ids) for choice in stopped.choices] == [[stop], [stop]]

        with pytest.raises(grpc.RpcError) as invalid:
            stub.Generate(
                inference_pb2.GenerateRequest(prompt_token_ids=prompt, max_new_tokens=TEST_MAX_SEQ_LEN),
                timeout=30,
            )
        assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(grpc.RpcError) as unsupported:
            stub.Generate(
                inference_pb2.GenerateRequest(prompt_token_ids=prompt, max_new_tokens=2, capture_router_replay=True),
                timeout=30,
            )
        assert unsupported.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_grpc_generation_returns_unwarped_policy_logprobs(inference_server_config, generated_model_and_tokenizer):
    model, tokenizer = generated_model_and_tokenizer
    config = dataclasses.replace(
        inference_server_config,
        service=dataclasses.replace(inference_server_config.service, logprob_mode=LogprobMode.POLICY),
    )
    model = jax.tree.map(jnp.zeros_like, model)
    server = InferenceServer.create(config, model, tokenizer)
    request = inference_pb2.GenerateRequest(
        prompt_token_ids=[18, 7, 3, 18],
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.05,
        seed=17,
        expected_logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
    )
    try:
        with (
            TestClient(server.app),
            grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel,
        ):
            stub = inference_pb2_grpc.InferenceStub(channel)
            response = stub.Generate(request, timeout=30)
            assert response.logprob_mode == inference_pb2.LOGPROB_MODE_POLICY
            assert not response.choices[0].behavior_logprobs
            np.testing.assert_allclose(
                response.choices[0].policy_logprobs, [-np.log(model.Vocab.size)], atol=1e-5, rtol=0
            )

            request.expected_logprob_mode = inference_pb2.LOGPROB_MODE_BEHAVIOR
            with pytest.raises(grpc.RpcError) as mismatch:
                stub.Generate(request, timeout=30)
            assert mismatch.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.shutdown()


def test_grpc_generation_fences_rollouts_by_installed_policy(inference_server_config, generated_model_and_tokenizer):
    model, tokenizer = generated_model_and_tokenizer
    server = InferenceServer.create(inference_server_config, model, tokenizer)
    try:
        with (
            TestClient(server.app),
            grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel,
        ):
            _assert_grpc_policy_fence(inference_pb2_grpc.InferenceStub(channel), server)
    finally:
        server.shutdown()


def _assert_grpc_policy_fence(stub, server):
    def generate(policy):
        request = inference_pb2.GenerateRequest(
            prompt_token_ids=[18, 7, 3], max_new_tokens=2, temperature=0, expected_policy_id=policy
        )
        return stub.Generate(request, timeout=30).choices[0]

    engine = server.inference_context.engine
    page_buffers = jax.tree.leaves(engine.gen_state.cache)
    previous_storage = [buffer.addressable_data(0).unsafe_buffer_pointer() for buffer in page_buffers]
    assert server.reload(lambda model: model, expected_version=0, policy_id="policy-a") == 1
    new_buffers = jax.tree.leaves(server.inference_context.engine.gen_state.cache)
    assert [buffer.addressable_data(0).unsafe_buffer_pointer() for buffer in new_buffers] == previous_storage
    first = generate("policy-a")
    assert (first.policy_id, first.model_version) == ("policy-a", 1)

    def fail_load(model):
        raise RuntimeError("invalid replacement")

    with pytest.raises(RuntimeError):
        server.reload(fail_load, expected_version=1, policy_id="policy-b")
    assert generate("policy-a") == first

    assert server.reload(lambda model: model, expected_version=1, policy_id="policy-b") == 2
    with pytest.raises(ValueError):
        server.reload(lambda model: model, expected_version=1, policy_id="policy-c")
    with pytest.raises(ValueError):
        server.reload(lambda model: model, expected_version=2, policy_id="policy-a")
    with pytest.raises(grpc.RpcError) as stale:
        generate("policy-a")
    assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    second = generate("policy-b")
    assert (second.policy_id, second.model_version) == ("policy-b", 2)


def test_grpc_snowball_generation_replays_exact_expert_routes(local_gpt2_tokenizer):
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
        host="127.0.0.1",
        grpc_port=0,
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
                with (
                    TestClient(server.app) as client,
                    grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel,
                ):
                    stub = inference_pb2_grpc.InferenceStub(channel)
                    response = stub.Generate(
                        inference_pb2.GenerateRequest(
                            prompt_token_ids=prompt.tolist(),
                            max_new_tokens=4,
                            temperature=0,
                            capture_router_replay=True,
                            expected_logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
                        ),
                        timeout=30,
                    )
                    plain = stub.Generate(
                        inference_pb2.GenerateRequest(
                            prompt_token_ids=prompt.tolist(), max_new_tokens=4, temperature=0
                        ),
                        timeout=30,
                    )
                    choice = response.choices[0]
                    assert list(choice.token_ids) == list(plain.choices[0].token_ids)
                    assert not plain.choices[0].HasField("router_replay")
                    rollout = RolloutChoice(response, 0)
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
                    (final,) = [event for event in events if "levanter_generation" in event]
                    capture = inference_pb2.GenerateResponse.FromString(
                        base64.b64decode(final["levanter_generation"], validate=True)
                    )
                    expected_prompt = _compute_tokens([ChatMessage(**messages[0])], tokenizer, None, None)
                    np.testing.assert_array_equal(capture.prompt_token_ids, expected_prompt)
                    streamed_ids = capture.choices[0].token_ids
                    assert len(streamed_ids) == 2
                    _, stream_valid = read_router_replay(
                        capture, 0, np.concatenate((expected_prompt, streamed_ids)).astype(np.int32)
                    )
                    assert stream_valid[:-1].all() and not stream_valid[-1]

            finally:
                server.inference_context.shutdown()

            inputs = np.concatenate((prompt, choice.token_ids)).astype(np.int32)
            experts, valid = read_router_replay(response, 0, inputs)
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
            prepared = prepare_rollout_policy_batch(model, batch, [rollout], policy_id="test-snowball-policy")
            np.testing.assert_array_equal(prepared.old_logprobs.array[0, :-1], choice.policy_logprobs)
            np.testing.assert_allclose(
                response_logprobs(model, prepared, block_size=8).array[0, :-1],
                choice.policy_logprobs,
                atol=1e-5,
                rtol=0,
            )
            with pytest.raises(ValueError):
                prepare_rollout_policy_batch(model, batch, [rollout], policy_id="stale-policy")

    np.testing.assert_allclose(choice.policy_logprobs, -learner_loss.array[0, len(prompt) - 1 : -1], atol=1e-5, rtol=0)
    wrong_inputs = inputs.copy()
    wrong_inputs[0] += 1
    with pytest.raises(ValueError):
        read_router_replay(response, 0, wrong_inputs)


def test_openai_rejects_stop_sequences_that_exceed_engine_capacity(test_client):
    client, server = test_client
    stops = ["A"] * (server.config.service.max_stop_seqs + 1)
    response = client.post(
        "/v1/completions",
        json={"model": TEST_MODEL_NAME, "prompt": "A", "max_tokens": 1, "stop": stops},
    )
    assert response.status_code == 400


def test_publish_snowball_checkpoint_drives_the_live_grpc_snapshot(tmp_path, local_gpt2_tokenizer):
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
        host="127.0.0.1",
        grpc_port=0,
        service=InferenceEngineConfig(
            max_seq_len=16,
            max_pages=16,
            max_seqs=2,
            max_rounds=2,
            page_size=2,
            max_seqs_in_prefill=2,
            max_prefill_size=12,
            max_queued_tokens=4,
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
        server = InferenceServer.create(options, initial, local_gpt2_tokenizer, mesh=mesh)
        try:
            with (
                TestClient(server.app),
                grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel,
            ):
                stub = inference_pb2_grpc.InferenceStub(channel)
                prompt = np.array([2, 5, 7], dtype=np.int32)

                def generate(policy_id):
                    return stub.Generate(
                        inference_pb2.GenerateRequest(
                            prompt_token_ids=prompt.tolist(),
                            max_new_tokens=3,
                            temperature=0,
                            capture_router_replay=True,
                            expected_policy_id=policy_id,
                        ),
                        timeout=30,
                    )

                before = generate("starting").choices[0]
                with pytest.raises(ValueError):
                    publish_snowball_checkpoint(server, path, policy_version=1, expected_serving_version=0)
                assert publish_snowball_checkpoint(server, path, policy_version=2, expected_serving_version=0) == 1
                with pytest.raises(grpc.RpcError) as stale:
                    generate("starting")
                assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                result = generate("learned:2")
                choice = result.choices[0]
                assert choice.policy_id == "learned:2"
                assert not np.array_equal(choice.policy_logprobs, before.policy_logprobs)
                tokens = np.r_[prompt, choice.token_ids].astype(np.int32)
                experts, valid = read_router_replay(result, 0, tokens)
                inputs = hax.named(jnp.asarray(tokens[None]), ("batch", "position"))
                replay = learned.prepare_route_replay(
                    jnp.asarray(experts[:, None]), hax.named(valid[None], inputs.axes)
                )
                loss, _ = learned.next_token_loss_and_metrics(
                    LmExample(inputs, hax.ones(inputs.axes)), route_replay=replay, reduction=None
                )
                np.testing.assert_allclose(
                    choice.policy_logprobs, -loss.array[0, prompt.size - 1 : -1], rtol=0, atol=1e-5
                )
        finally:
            server.shutdown()
