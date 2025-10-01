# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PrimeIntellectEnv integration with verifiers library."""

from unittest.mock import Mock, patch

import jax.random
import numpy as np
import pytest
from transformers import AutoTokenizer

try:
    from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
except ImportError:
    pytest.skip("verifiers library not installed", allow_module_level=True)


@pytest.fixture
def tokenizer():
    """Create a simple tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def mock_inference_ctx(tokenizer):
    """Create a mock inference context."""
    ctx = Mock()
    ctx.tokenizer = tokenizer
    ctx.openai_client = Mock(return_value=Mock())
    return ctx


@pytest.fixture
def mock_vf_env():
    """Create a mock verifiers environment."""
    env = Mock()

    # Mock the generate() method to return GenerateOutputs
    mock_result = Mock()
    mock_result.prompt = ["What is 2+2?", "What is 3+3?"]
    mock_result.completion = ["4", "4", "6", "6"]  # 2 generations per prompt
    mock_result.reward = [1.0, 0.9, 1.0, 0.8]
    mock_result.metrics = {"accuracy": 0.95}

    env.generate = Mock(return_value=mock_result)
    env.dataset = Mock()
    env.eval_dataset = Mock()
    env.get_dataset = Mock(return_value=Mock(repeat=Mock(return_value=Mock())))
    env.get_eval_dataset = Mock(return_value=Mock(repeat=Mock(return_value=Mock())))

    return env


def test_prime_intellect_env_sample(tokenizer, mock_inference_ctx, mock_vf_env):
    """Test sampling from PrimeIntellectEnv with mocked verifiers."""
    env = PrimeIntellectEnv(env_id="primeintellect/word-count", env_args={}, max_tokens=1024, max_concurrent=32)

    # Mock the load_prime_intellect_env method to return our mock environment
    with patch.object(env, "load_prime_intellect_env", return_value=mock_vf_env), patch("subprocess.run"):
        # Sample from the environment
        prng_key = jax.random.PRNGKey(0)
        rollout_groups, metrics = env.sample(
            inference_ctx=mock_inference_ctx,
            n_examples=2,
            n_generations=2,
            temperature=0.7,
            prng_key=prng_key,
            mode="train",
        )

    # Verify the mock was called correctly
    mock_vf_env.generate.assert_called_once()
    call_kwargs = mock_vf_env.generate.call_args.kwargs
    assert call_kwargs["sampling_args"]["temperature"] == 0.7
    assert call_kwargs["sampling_args"]["max_tokens"] == 1024
    assert call_kwargs["max_concurrent"] == 32

    # Verify rollout groups structure
    assert len(rollout_groups) == 2, "Should have 2 rollout groups (one per prompt)"

    for group in rollout_groups:
        assert len(group.rollouts) == 2, "Each group should have 2 rollouts"

        for rollout in group.rollouts:
            assert rollout.env_name == "prime_intellect:primeintellect/word-count"
            assert rollout.env_example_id.startswith("primeintellect/word-count_example_")
            assert len(rollout.prompt_tokens) > 0
            assert len(rollout.response_tokens) > 0
            assert len(rollout.response_logprobs) == len(rollout.response_tokens)
            assert len(rollout.token_rewards) == len(rollout.response_tokens)
            assert 0.0 <= rollout.episode_reward <= 1.0

    # Verify metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.95
    assert "primeintellect/word-count.mean_reward" in metrics
    assert "primeintellect/word-count.total_rollouts" in metrics


def test_prime_intellect_env_openai_client_called(tokenizer, mock_inference_ctx, mock_vf_env):
    """Test that the OpenAI client is properly requested from inference context."""
    env = PrimeIntellectEnv(env_id="primeintellect/math", env_args={})

    with patch.object(env, "load_prime_intellect_env", return_value=mock_vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(42)
        env.sample(
            inference_ctx=mock_inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=prng_key,
        )

    # Verify that openai_client() was called on the inference context
    mock_inference_ctx.openai_client.assert_called_once()

    # Verify the client was passed to vf_env.generate()
    call_kwargs = mock_vf_env.generate.call_args.kwargs
    assert "client" in call_kwargs
    assert call_kwargs["client"] == mock_inference_ctx.openai_client.return_value


def test_prime_intellect_env_handles_empty_results(tokenizer, mock_inference_ctx):
    """Test that environment handles empty results gracefully."""
    env = PrimeIntellectEnv(env_id="primeintellect/empty", env_args={})

    # Create a mock environment that returns empty results
    mock_vf_env = Mock()
    mock_result = Mock()
    mock_result.prompt = []
    mock_result.completion = []
    mock_result.reward = []
    mock_result.metrics = {}
    mock_vf_env.generate = Mock(return_value=mock_result)
    mock_vf_env.dataset = Mock()
    mock_vf_env.get_dataset = Mock(return_value=Mock(repeat=Mock(return_value=Mock())))

    with patch.object(env, "load_prime_intellect_env", return_value=mock_vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(0)
        rollout_groups, metrics = env.sample(
            inference_ctx=mock_inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.5,
            prng_key=prng_key,
        )

    assert len(rollout_groups) == 0
    assert isinstance(metrics, dict)


def test_prime_intellect_env_tokenization(tokenizer, mock_inference_ctx, mock_vf_env):
    """Test that prompts and completions are properly tokenized."""
    env = PrimeIntellectEnv(env_id="primeintellect/tokenize", env_args={})

    with patch.object(env, "load_prime_intellect_env", return_value=mock_vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(123)
        rollout_groups, _ = env.sample(
            inference_ctx=mock_inference_ctx,
            n_examples=2,
            n_generations=2,
            temperature=0.8,
            prng_key=prng_key,
        )

    # Verify tokenization was done
    for group in rollout_groups:
        for rollout in group.rollouts:
            # Tokens should be valid int32 arrays
            assert rollout.prompt_tokens.dtype == np.int32
            assert rollout.response_tokens.dtype == np.int32

            # Decode and verify we can reconstruct something
            decoded_prompt = tokenizer.decode(rollout.prompt_tokens.tolist())
            decoded_response = tokenizer.decode(rollout.response_tokens.tolist())

            assert isinstance(decoded_prompt, str)
            assert isinstance(decoded_response, str)
            assert len(decoded_prompt) > 0
            assert len(decoded_response) > 0
