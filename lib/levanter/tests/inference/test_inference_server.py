# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig

try:
    from fastapi.testclient import TestClient
    from openai.types import Completion
    from openai.types.chat import ChatCompletion

    from levanter.inference.engine import (
        GenerationResult,
        InferenceEngineConfig,
        score_token_sequence_logprobs,
    )
    from levanter.inference.openai import (
        InferenceBatch,
        InferenceContext,
        InferenceResponse,
        InferenceRequest,
        InferenceServer,
        InferenceServerConfig,
    )

except ImportError:
    pytest.skip("Serving imports not installed, use --extra=serve", allow_module_level=True)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def trainer_config():
    return TrainerConfig()


@pytest.fixture(scope="module")
def baby_llama_config():
    return InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=32,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=32,
            hbm_utilization=0.1,
        ),
        temperature=0.7,
        seed=42,
    )


@pytest.fixture(scope="module")
def loaded_model(trainer_config):
    """Load the baby llama model and tokenizer."""
    hf_checkpoint = "timinar/baby-llama-58m"
    model_config = LlamaConfig()
    tokenizer = load_tokenizer(hf_checkpoint)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = HFCheckpointConverter(
            LlamaConfig,
            reference_checkpoint=hf_checkpoint,
            tokenizer=tokenizer,
        )

        model = converter.load_pretrained(
            model_config.model_type,
            ref=hf_checkpoint,
            dtype=trainer_config.mp.compute_dtype,
            axis_mapping=trainer_config.parameter_axis_mapping,
        )

    return model, tokenizer


@pytest.fixture(scope="module")
def inference_server(trainer_config, baby_llama_config, loaded_model):
    """Create an InferenceServer instance."""
    model, tokenizer = loaded_model
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        return InferenceServer.create(baby_llama_config, model, tokenizer)


@pytest.fixture(scope="module")
def test_client(baby_llama_config, loaded_model, inference_server):
    """Create a test client for the inference server."""
    with TestClient(inference_server.app) as client:
        yield client, inference_server


@pytest.fixture(scope="module")
def hf_reference_model_and_tokenizer():
    """Load the HF reference model used for correctness comparisons."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model_name = "timinar/baby-llama-58m"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    return model, tokenizer


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
                "model": "timinar/baby-llama-58m",
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
    assert "/v1/completions" in routes
    assert "/v1/chat/completions" in routes


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
        stop_tokens: list[int] | None,
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


def test_execute_batch_forwards_top_p_into_engine_requests():
    class _FakeTokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return "decoded"

    class _FakeLoop:
        def call_soon_threadsafe(self, callback, *args):
            callback(*args)

    class _FakeFuture:
        def __init__(self):
            self._loop = _FakeLoop()
            self.result = None
            self.exception = None

        def get_loop(self):
            return self._loop

        def set_result(self, value):
            self.result = value

        def set_exception(self, exc):
            self.exception = exc

    class _FakeEngine:
        def __init__(self):
            self.requests = None

        def generate(self, requests):
            self.requests = list(requests)
            return GenerationResult(tokens=[[1, 2]], logprobs=[[-0.1, -0.2]], total_generated=2)

    engine = _FakeEngine()
    ctx = InferenceContext(
        model=None,
        tokenizer=_FakeTokenizer(),
        engine=engine,
        config=InferenceServerConfig(),
    )
    future = _FakeFuture()
    batch = InferenceBatch(
        [
            InferenceRequest(
                request_id="req_0",
                prompt_tokens=[1, 2, 3],
                max_tokens=2,
                temperature=0.7,
                top_p=0.73,
                stop_tokens=None,
                seed=123,
                future=future,
                n_generations=1,
            )
        ]
    )

    ctx._execute_batch(batch)

    assert engine.requests is not None
    assert np.asarray(engine.requests[0].decode_params.top_p, dtype=np.float32).item() == pytest.approx(0.73)
    assert future.exception is None
    assert future.result is not None


@pytest.mark.slow
def test_short_request(test_client):
    client, server = test_client

    response = client.post(
        "/v1/completions",
        json={
            "model": "timinar/baby-llama-58m",
            "prompt": "The quick brown fox",
            "max_tokens": 10,
            "temperature": 0.7,
            "stop": ".",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    completion = Completion.model_validate(response.json())

    choice = completion.choices[0]
    assert choice.text
    assert choice.finish_reason == "stop"
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0
    assert completion.usage.total_tokens == completion.usage.prompt_tokens + completion.usage.completion_tokens
    assert completion.usage.completion_tokens <= 10

    print(f"Generated text: '{choice.text}'")
    print(f"Usage: {completion.usage}")


@pytest.mark.slow
def test_weight_reloading_during_requests(test_client):
    """
    Test that weight reloading works correctly while requests are being processed.

    This test queues multiple requests on a background thread and triggers a reload
    while they are being processed, ensuring all requests complete successfully.
    """
    client, server = test_client

    # Wait for the inference service to fully initialize
    time.sleep(1.0)

    if not server.inference_context:
        pytest.skip("Inference context not initialized")

    # Dummy weight callback that just returns the same model
    def dummy_weight_callback(model):
        time.sleep(0.1)  # Simulate some work
        return model

    # Submit several requests concurrently using ThreadPoolExecutor
    def make_request(request_id):
        response = client.post(
            "/v1/completions",
            json={
                "model": "timinar/baby-llama-58m",
                "prompt": f"Request {request_id}: The quick brown fox",
                "max_tokens": 8,
                "temperature": 0.7,
                "seed": request_id,
            },
        )
        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
        }

    # Start multiple concurrent requests
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit requests that will be processed before, during, and after reload
        futures = [executor.submit(make_request, i) for i in range(6)]

        # Wait a moment for some requests to start processing
        time.sleep(0.2)

        # Trigger model reload while requests are in flight
        print("Triggering model reload...")
        reload_start = time.time()
        server.reload(dummy_weight_callback)
        reload_duration = time.time() - reload_start
        print(f"Model reload completed in {reload_duration:.2f}s")

        # Collect all results
        results = [future.result() for future in futures]

    # Analyze results
    successful_requests = [r for r in results if r["status_code"] == 200]
    failed_requests = [r for r in results if r["status_code"] != 200]

    print(f"Total successful requests: {len(successful_requests)}")
    print(f"Total failed requests: {len(failed_requests)}")

    # Verify all requests completed successfully
    assert len(successful_requests) > 0, "Expected at least some successful requests"
    assert len(failed_requests) == 0, f"No requests should fail, but got: {failed_requests}"

    # Verify response structure for successful requests
    for result in successful_requests:
        response_data = result["response"]
        assert "choices" in response_data
        assert "usage" in response_data
        assert len(response_data["choices"]) > 0
        assert "text" in response_data["choices"][0]

    print("Weight reloading test passed successfully!")


@pytest.mark.slow
def test_completion_with_logprobs(test_client):
    """Test text completion endpoint with logprobs enabled."""
    client, server = test_client

    response = client.post(
        "/v1/completions",
        json={
            "model": "timinar/baby-llama-58m",
            "prompt": "The quick brown",
            "max_tokens": 5,
            "temperature": 0.0,  # Use deterministic sampling
            "logprobs": True,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    completion = Completion.model_validate(response.json())

    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert len(choice.logprobs.tokens) > 0
    assert len(choice.logprobs.tokens) == len(choice.logprobs.token_logprobs)

    for token, logprob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
        assert logprob <= 0.0

    print(f"Generated {len(choice.logprobs.tokens)} tokens with logprobs")
    print(f"First few tokens: {choice.logprobs.tokens[:3]}")
    print(f"First few logprobs: {choice.logprobs.token_logprobs[:3]}")


@pytest.mark.slow
def test_chat_completion_with_logprobs(test_client):
    """Test chat completion endpoint with logprobs enabled."""
    client, server = test_client

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "timinar/baby-llama-58m",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    chat_completion = ChatCompletion.model_validate(response.json())

    logger.info("Chat response: %s", chat_completion)

    choice = chat_completion.choices[0]
    assert choice.logprobs is not None
    assert len(choice.logprobs.content) > 0

    for token_logprob in choice.logprobs.content:
        assert token_logprob.logprob <= 0.0

    print(f"Chat generated {len(choice.logprobs.content)} tokens with logprobs")


@pytest.mark.slow
def test_logprobs_with_multiple_generations(test_client):
    """Test logprobs with n > 1 (multiple generations)."""
    client, server = test_client

    response = client.post(
        "/v1/completions",
        json={
            "model": "timinar/baby-llama-58m",
            "prompt": "One plus one is",
            "max_tokens": 10,
            "temperature": 0.7,
            "logprobs": True,
            "n": 2,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    completion = Completion.model_validate(response.json())

    assert len(completion.choices) == 2

    logprob_arrays = []

    for i, choice in enumerate(completion.choices):
        assert choice.index == i
        assert choice.logprobs is not None
        assert len(choice.logprobs.tokens) > 0, choice
        assert len(choice.logprobs.token_logprobs) == len(choice.logprobs.tokens), choice
        logprob_arrays.append(choice.logprobs.token_logprobs)
        print(f"Choice {i} - {choice.text} {choice.logprobs.tokens} {choice.logprobs.token_logprobs}")

    # Ensure the two generations are different
    assert np.all(
        np.array(logprob_arrays[0]) != np.array(logprob_arrays[1])
    ), f"Expected different generations, got {logprob_arrays}"


def test_logprobs_deterministic_behavior(test_client):
    """Test that logprobs are deterministic with same seed."""
    client, server = test_client

    # Make the same request twice with same seed
    request_data = {
        "model": "timinar/baby-llama-58m",
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
                    "model": "timinar/baby-llama-58m",
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
            "model": "timinar/baby-llama-58m",
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
            "model": "timinar/baby-llama-58m",
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
            "model": "timinar/baby-llama-58m",
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
            "model": "timinar/baby-llama-58m",
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


@pytest.mark.slow
def test_logprobs_match_full_forward_pass(test_client, loaded_model, trainer_config):
    """Test that logprobs from inference server match those computed from a single full forward pass."""
    client, server = test_client
    model, tokenizer = loaded_model

    # Step 1: Get prompt tokens using the /v1/tokens endpoint
    messages = [{"role": "user", "content": "The capital of France is"}]

    tokens_response = client.post(
        "/v1/tokens",
        json={
            "model": "timinar/baby-llama-58m",
            "message_list": [messages],
        },
    )

    assert tokens_response.status_code == 200
    prompt_tokens = tokens_response.json()["results"][0]["tokens"]
    print(f"Prompt tokens from /v1/tokens: {prompt_tokens}")

    # Step 2: Get logprobs from inference server using chat completions
    chat_response = client.post(
        "/v1/chat/completions",
        json={
            "model": "timinar/baby-llama-58m",
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0.0,  # deterministic
            "logprobs": True,
            "seed": 42,
        },
    )

    assert chat_response.status_code == 200
    chat_completion = ChatCompletion.model_validate(chat_response.json())
    choice = chat_completion.choices[0]

    assert choice.logprobs is not None
    assert len(choice.logprobs.content) > 0

    # Extract generated token IDs and logprobs from server response
    server_logprobs = []
    generated_token_ids = []

    print(f"Server returned {len(choice.logprobs.content)} tokens:")
    for i, token_logprob in enumerate(choice.logprobs.content):
        token_str = token_logprob.token
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"  Token {i}: '{token_str}' -> {token_id}, logprob={token_logprob.logprob}")
        generated_token_ids.append(token_id)
        server_logprobs.append(token_logprob.logprob)

    print(f"Generated {len(generated_token_ids)} tokens: {generated_token_ids}")
    print(f"Server logprobs: {server_logprobs}")

    # Step 3: Run full forward pass with [prompt + generated tokens]
    print("Computing model logprobs using full forward pass")

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        from levanter.layers.attention import AttentionMask

        # Concatenate prompt + generated tokens
        full_sequence = prompt_tokens + generated_token_ids

        # Run full forward pass on entire sequence (not auto-regressive)
        Pos = hax.Axis("position", len(full_sequence))
        input_ids = hax.named(jnp.array(full_sequence, dtype=jnp.int32), Pos)
        pos_ids = hax.named(jnp.arange(len(full_sequence), dtype=jnp.int32), Pos)

        # Use causal attention mask
        attn_mask = AttentionMask.causal()

        # Call model directly for full forward pass
        logits = model(input_ids=input_ids, attn_mask=attn_mask, pos_ids=pos_ids, key=None)

        # Extract logits at positions corresponding to generated tokens
        # The first generated token is predicted by the last prompt token
        # So we want logits at positions [len(prompt_tokens)-1 : len(full_sequence)-1]
        model_logprobs = []
        for i, token_id in enumerate(generated_token_ids):
            # Position in the full sequence that predicts this token
            pred_pos = len(prompt_tokens) + i - 1
            logits_at_pos = logits.array[pred_pos].astype(jnp.float32)
            token_logit = logits_at_pos[token_id]
            log_z = jax.nn.logsumexp(logits_at_pos)
            token_logprob = token_logit - log_z
            model_logprobs.append(float(token_logprob))
            print(
                f"Token {i} (id={token_id}): logit={token_logit:.6f}, log_z={log_z:.6f}, logprob={token_logprob:.6f}"
            )

    print(f"Model logprobs: {model_logprobs}")

    # Step 4: Compare logprobs
    assert len(server_logprobs) == len(
        model_logprobs
    ), f"Length mismatch: server has {len(server_logprobs)}, model has {len(model_logprobs)}"

    for i, (server_lp, model_lp) in enumerate(zip(server_logprobs, model_logprobs)):
        diff = abs(float(server_lp) - float(model_lp))
        print(f"Token {i}: server={server_lp:.6f}, model={model_lp:.6f}, diff={diff:.6f}")
        # Allow larger tolerance due to accumulated bfloat16 precision errors in KV cache
        # The errors accumulate as we process more tokens auto-regressively
        assert diff < 3e-3, f"Logprob mismatch at token {i}: server={server_lp}, model={model_lp}, diff={diff}"

    print("All logprobs match successfully!")
