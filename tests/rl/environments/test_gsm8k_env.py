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

"""Tests for the GSM8KEnv environment."""

import jax.random
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

from marin.rl.environments.gsm8k_env import GSM8KEnv
from marin.rl.environments.inference_ctx import LevanterInferenceContext


def create_mock_chat_completion(tokenizer) -> ChatCompletion:
    """Create a mock ChatCompletion mirroring the MathEnv tests."""

    response_text = "<answer>72</answer>"
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

    def batch_completions(self, prompts, temperature, n, max_tokens=None, stop=None, system_prompt=None, top_k=None):
        """Return mock completions for each prompt."""

        return [create_mock_chat_completion(self.tokenizer) for prompt in prompts]


def test_gsm8k_env_samples_and_scores_rollouts():
    """Ensure GSM8KEnv builds rollouts and metrics mirroring MathEnv behavior."""

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inference_ctx = DummyInferenceContext(tokenizer)
    train_data = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and half as many in May.",
            "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.\n#### 72",
        }
    ]

    env = GSM8KEnv(train_dataset=train_data, eval_dataset=[], max_train_examples=1, max_eval_examples=0)
    assert env.train_examples[0].processed_answer == "72"

    prng_key = jax.random.PRNGKey(0)
    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.5,
        prng_key=prng_key,
        mode="train",
    )

    assert len(rollout_groups) == 1
    rollout = rollout_groups[0].rollouts[0]
    prompt_text = tokenizer.decode(rollout.prompt_tokens)
    response_text = tokenizer.decode(rollout.response_tokens)

    assert "Natalia sold clips" in prompt_text
    assert "<answer>72</answer>" in response_text
    assert rollout.env_name == "gsm8k"
    assert rollout.response_tokens.dtype == np.int32

    np.testing.assert_allclose(rollout.token_rewards, 1.2)
    assert rollout.episode_reward == pytest.approx(1.2)

    assert metrics["gsm8k.train_mean_reward"] == pytest.approx(1.2)
    assert metrics["gsm8k.train_correct_accuracy"] == pytest.approx(1.0)


def test_extract_final_answer_handles_missing_separator():
    """Fallback to the raw answer when #### is absent."""

    env = GSM8KEnv(train_dataset=[], eval_dataset=[], max_train_examples=0, max_eval_examples=0)
    assert env._extract_final_answer("No hash answer") == "No hash answer"
    assert env._extract_final_answer("Reasoning #### 19") == "19"
