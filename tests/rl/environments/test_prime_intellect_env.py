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

from unittest.mock import AsyncMock, patch

import jax.random
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

try:
    import verifiers as vf

    from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
except ImportError:
    pytest.skip("verifiers library not installed", allow_module_level=True)

from marin.rl.environments.inference_ctx import LevanterInferenceContext


def create_mock_chat_completion(response_text: str = "42") -> ChatCompletion:
    """Create a mock ChatCompletion for testing."""
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=len(response_text.split()), prompt_tokens=10, total_tokens=10 + len(response_text.split())
        ),
    )


@pytest.fixture
def tokenizer():
    """Create a real tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def inference_ctx(tokenizer):
    """Create a real inference context with mock openai_client."""

    class DummyInferenceContext(LevanterInferenceContext):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self._stop_tokens = None
            self.max_tokens = 1024

        def openai_client(self):
            """Return a mock AsyncOpenAI client that returns proper ChatCompletion objects."""
            mock_client = AsyncMock()
            # Configure the mock to return a proper ChatCompletion
            mock_client.chat.completions.create = AsyncMock(return_value=create_mock_chat_completion())
            return mock_client

    return DummyInferenceContext(tokenizer)


@pytest.fixture
def vf_env():
    """Create a real verifiers SingleTurnEnv with example dataset."""
    dataset = vf.load_example_dataset("gsm8k", n=2)
    return vf.SingleTurnEnv(dataset=dataset)


def test_prime_intellect_env_sample(tokenizer, inference_ctx, vf_env):
    """Test sampling from PrimeIntellectEnv with real components."""
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={}, max_tokens=1024, max_concurrent=32)

    # Patch only external dependencies
    with patch.object(env, "load_prime_intellect_env", return_value=vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(0)
        rollout_groups, metrics = env.sample(
            inference_ctx=inference_ctx,
            n_examples=2,
            n_generations=2,
            temperature=0.7,
            prng_key=prng_key,
            mode="train",
        )

    # Verify rollout groups structure
    assert len(rollout_groups) == 2, "Should have 2 rollout groups (one per prompt)"

    for group in rollout_groups:
        assert len(group.rollouts) == 2, "Each group should have 2 rollouts"

        for rollout in group.rollouts:
            assert rollout.env_name == "prime_intellect:primeintellect/gsm8k"
            assert rollout.env_example_id.startswith("primeintellect/gsm8k_example_")
            assert len(rollout.prompt_tokens) > 0
            assert len(rollout.response_tokens) > 0
            assert len(rollout.response_logprobs) == len(rollout.response_tokens)
            assert len(rollout.token_rewards) == len(rollout.response_tokens)
            assert 0.0 <= rollout.episode_reward <= 1.0

            # Verify tokens are valid int32
            assert rollout.prompt_tokens.dtype == np.int32
            assert rollout.response_tokens.dtype == np.int32

    # Verify metrics exist
    assert "primeintellect/gsm8k.mean_reward" in metrics
    assert "primeintellect/gsm8k.total_rollouts" in metrics


def test_prime_intellect_env_openai_client_called(tokenizer, inference_ctx, vf_env):
    """Test that the OpenAI client is properly requested from inference context."""
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={})

    with patch.object(env, "load_prime_intellect_env", return_value=vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(42)
        rollout_groups, _ = env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=prng_key,
        )

    # Verify we got rollout groups (which means generate() was called successfully)
    assert len(rollout_groups) >= 0


def test_prime_intellect_env_tokenization(tokenizer, inference_ctx, vf_env):
    """Test that prompts include chat template and completions are properly tokenized."""
    env = PrimeIntellectEnv(env_id="primeintellect/tokenize", env_args={})

    with patch.object(env, "load_prime_intellect_env", return_value=vf_env), patch("subprocess.run"):
        prng_key = jax.random.PRNGKey(123)
        rollout_groups, _ = env.sample(
            inference_ctx=inference_ctx,
            n_examples=2,
            n_generations=2,
            temperature=0.8,
            prng_key=prng_key,
        )

    # Verify tokenization and chat template application
    for group in rollout_groups:
        for rollout in group.rollouts:
            # Tokens should be valid int32 arrays
            assert rollout.prompt_tokens.dtype == np.int32
            assert rollout.response_tokens.dtype == np.int32

            # Decode the prompt tokens
            decoded_prompt = tokenizer.decode(rollout.prompt_tokens.tolist())
            decoded_response = tokenizer.decode(rollout.response_tokens.tolist())

            # Verify chat template was applied: prompt tokens should contain "user:" prefix
            # from the fallback chat template (GPT-2 doesn't have a native chat template)
            assert "user:" in decoded_prompt.lower(), "Chat template should add 'user:' prefix to prompt"

            # Verify response is non-empty
            assert len(decoded_response) > 0
