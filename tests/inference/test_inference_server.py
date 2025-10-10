# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from concurrent.futures import ThreadPoolExecutor

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

    from levanter.inference.engine import InferenceEngineConfig
    from levanter.inference.openai import InferenceServer, InferenceServerConfig

except ImportError:
    pytest.skip("Serving imports not installed, use --extra=serve", allow_module_level=True)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def trainer_config():
    return TrainerConfig(model_axis_size=1)


@pytest.fixture(scope="module")
def baby_llama_config():
    return InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=16,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=8,
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
def inference_server(baby_llama_config, loaded_model):
    """Create an InferenceServer instance."""
    model, tokenizer = loaded_model
    return InferenceServer.create(baby_llama_config, model, tokenizer)


@pytest.fixture(scope="module")
def test_client(baby_llama_config, loaded_model):
    """Create a test client for the inference server."""
    model, tokenizer = loaded_model
    server = InferenceServer.create(baby_llama_config, model, tokenizer)
    with TestClient(server.app) as client:
        yield client, server


def test_endpoints_exist(test_client):
    """Test that the endpoints are properly defined"""
    _, server = test_client
    routes = [route.path for route in server.app.routes]
    assert "/health" in routes
    assert "/v1/completions" in routes
    assert "/v1/chat/completions" in routes


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
        # Encode the token string to get the ID
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        print(f"  Token {i}: '{token_str}' -> {token_ids}, logprob={token_logprob.logprob}")
        if len(token_ids) == 1:
            generated_token_ids.append(token_ids[0])
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
