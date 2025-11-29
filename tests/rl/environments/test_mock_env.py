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

import jax.numpy as jnp
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage

from marin.rl.environments.mock_env import (
    AdditionTask,
    MoarCatsTask,
    NumberComparisonTask,
    OppositesTask,
    compute_soft_reward,
)
from marin.rl.types import Rollout, RolloutGroup


def create_test_tokenizer():
    """Create a simple test tokenizer that encodes chars as ord values."""

    class SimpleTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]

        def decode(self, token_ids):
            return "".join(chr(tid) for tid in token_ids)

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            # Simple: just return tokens for the user message content
            return [ord(c) for c in messages[0]["content"]]

        def convert_tokens_to_ids(self, token):
            # In our simple test tokenizer, tokens are single chars
            return ord(token[0]) if token else 0

    return SimpleTokenizer()


def create_test_logprobs(text: str):
    """Create logprobs content for a response text."""
    from openai.types.chat.chat_completion_chunk import ChoiceLogprobsLogprob

    logprobs_content = []
    for c in text:
        logprobs_content.append(
            ChoiceLogprobsLogprob(
                token=c,
                logprob=-1.0,
                bytes=[ord(c)],
                top_logprobs=[],
            )
        )
    return ChoiceLogprobs(content=logprobs_content)


def create_test_chat_completion(prompt: str, responses: list[str]) -> ChatCompletion:
    """Create a test ChatCompletion with multiple choices."""
    choices = []
    for i, response_text in enumerate(responses):
        choice = Choice(
            finish_reason="stop",
            index=i,
            message=ChatCompletionMessage(role="assistant", content=response_text),
            logprobs=create_test_logprobs(response_text),
        )
        choices.append(choice)

    return ChatCompletion(
        id=f"chatcmpl-test-{hash(prompt)}",
        choices=choices,
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=sum(len(r) for r in responses), prompt_tokens=len(prompt), total_tokens=0
        ),
    )


def create_test_inference_context():
    """Create a test inference context that returns test completions."""

    class TestInferenceContext:
        def __init__(self):
            self.tokenizer = create_test_tokenizer()

        def batch_completions(self, prompts, temperature, n, max_tokens=None, stop=None, system_prompt=None):
            completions = []
            for prompt in prompts:
                responses = [f"mock_response_{i}" for i in range(n)]
                completion = create_test_chat_completion(prompt, responses)
                completions.append(completion)
            return completions

        def tokenize_prompt(self, prompt):
            return np.array([ord(c) for c in prompt], dtype=np.int32)

        def get_choice_tokens(self, choice):
            return np.array([ord(c) for c in choice.message.content], dtype=np.int32)

        def get_choice_logprobs(self, choice):
            return np.full(len(choice.message.content), -1.0, dtype=np.float32)

        def create_rollout_from_choice(
            self, prompt, choice, env_name, env_example_id, reward, temperature,
            system_prompt=None, correctness_reward=None,
        ):
            prompt_tokens = self.tokenize_prompt(prompt)
            response_tokens = self.get_choice_tokens(choice)
            response_logprobs = self.get_choice_logprobs(choice)
            token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)

            return Rollout(
                env_name=env_name,
                env_example_id=env_example_id,
                prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
                response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
                response_logprobs=jnp.array(response_logprobs, dtype=jnp.float32),
                token_rewards=token_rewards,
                episode_reward=float(reward),
                temperature=temperature,
                is_truncated=False,
                correctness_reward=correctness_reward,
            )

        def create_rollout_group_from_completion(self, prompt, completion, env_name, env_example_id, reward_fn):
            rollouts = []
            for choice in completion.choices:
                response_text = choice.message.content
                reward = reward_fn(response_text)
                rollout = self.create_rollout_from_choice(
                    prompt, choice, env_name, env_example_id, reward, temperature=1.0,
                )
                rollouts.append(rollout)

            return RolloutGroup(rollouts=rollouts)

    return TestInferenceContext()


@pytest.fixture
def test_tokenizer():
    return create_test_tokenizer()


@pytest.fixture
def test_inference_ctx():
    return create_test_inference_context()


def test_compute_soft_reward_format_loss():
    assert compute_soft_reward("42", "42") > compute_soft_reward("42", "42 extra words")
    assert compute_soft_reward("42", "42") > compute_soft_reward("42", "wrong")

    assert compute_soft_reward("42", "42") == pytest.approx(1.0)
    assert compute_soft_reward("42", "wrong") == pytest.approx(0.0)

    short_format_score = compute_soft_reward("42", "43")
    long_format_score = compute_soft_reward("42", "43 with lots of extra words")
    assert short_format_score == long_format_score == 0.0


def test_addition_task_reward():
    task = AdditionTask()
    examples = task.generate_examples(10, np.random.default_rng(42))
    assert len(examples) == 10
    assert all("+" in ex["prompt"] for ex in examples)

    assert task.compute_reward("42", "42") == pytest.approx(1.0)
    assert task.compute_reward("42", "43") == pytest.approx(0.0)
    assert task.compute_reward("42", "-") == pytest.approx(0.0)
    assert task.compute_reward("42", "-2") == pytest.approx(0.0)


def test_opposites_task_reward():
    task = OppositesTask()
    examples = task.generate_examples(10, np.random.default_rng(42))
    assert len(examples) == 10

    assert task.compute_reward("cold", "cold") == pytest.approx(1.0)
    assert task.compute_reward("cold", "warm") == pytest.approx(0.0)


def test_number_comparison_task_format_bonus():
    task = NumberComparisonTask()

    digit_reward = task.compute_reward("42", "42")
    non_digit_reward = task.compute_reward("42", "forty-two")

    assert digit_reward > non_digit_reward
    assert digit_reward == pytest.approx(1.0)
    assert non_digit_reward == pytest.approx(0.0)


def test_cats_task_reward():
    task = MoarCatsTask()

    assert task.compute_reward("cats", "cats cats cats") > task.compute_reward("cats", "cats")
    assert task.compute_reward("cats", "i love cats") > task.compute_reward("cats", "i like cats")

    assert task.compute_reward("cats", "cat") > 0
    assert task.compute_reward("cats", "dog") == 0
