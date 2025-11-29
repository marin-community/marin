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

"""Basic tests for MathEnv with new InferenceContext paradigm."""

import jax.random
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

from marin.rl.environments.math_env import MathEnv
from marin.rl.environments.inference_ctx import LevanterInferenceContext


def create_mock_chat_completion(tokenizer) -> ChatCompletion:
    """Create a mock ChatCompletion with logprobs for testing."""
    response_text: str = "<answer>4</answer>"
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    logprobs_content = [
        ChatCompletionTokenLogprob(
            token=tokenizer.convert_ids_to_tokens([tok])[0], logprob=-0.5, bytes=[], top_logprobs=[]
        )
        for tok in tokens
    ]

    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs={"content": logprobs_content},
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=len(tokens), prompt_tokens=10, total_tokens=10 + len(tokens)),
    )


class DummyInferenceContext(LevanterInferenceContext):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._stop_tokens = None
        self.max_tokens = 512

    def batch_completions(self, prompts, temperature, n, max_tokens=None, stop=None, system_prompt=None):
        """Return mock completions for each prompt."""
        return [create_mock_chat_completion(self.tokenizer) for prompt in prompts]


def test_math_env_reward_calculation():
    """Test that math env correctly calculates rewards and creates rollouts."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inference_ctx = DummyInferenceContext(tokenizer)
    train_data = [
        {"problem": "What is 2+2?", "solution": "\\boxed{4}"},
    ]

    env = MathEnv(train_dataset=train_data, eval_dataset=[], max_train_examples=1)

    prng_key = jax.random.PRNGKey(42)
    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.7,
        prng_key=prng_key,
        mode="train",
    )

    # Verify structure
    assert len(rollout_groups) == 1
    rollout = rollout_groups[0].rollouts[0]

    response_txt = tokenizer.decode(rollout.response_tokens)
    prompt_txt = tokenizer.decode(rollout.prompt_tokens)
    assert "What is 2+2?" in prompt_txt, (prompt_txt, rollout)
    assert "<answer>4</answer>" in response_txt, (response_txt, rollout)

    # Verify basic rollout properties
    assert rollout.env_name == "math"
    assert rollout.prompt_tokens.dtype == np.int32
    assert rollout.response_tokens.dtype == np.int32
    assert len(rollout.response_logprobs) == len(rollout.response_tokens)
    assert len(rollout.token_rewards) == len(rollout.response_tokens)

    # Verify chat template was applied to prompt
    decoded_prompt = tokenizer.decode(rollout.prompt_tokens.tolist())
    assert "user:" in decoded_prompt.lower()

    np.testing.assert_allclose(rollout.token_rewards, 1.2), (rollout, metrics)
    assert rollout.episode_reward == pytest.approx(1.2), (rollout, metrics)
