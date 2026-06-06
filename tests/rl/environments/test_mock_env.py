# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
from marin.inference.types import (
    TokenizedRollout,
    TokenizedRolloutBatchResult,
    TokenizerIdentity,
    TokenRolloutAdmissionMetadata,
    TokenRolloutFinishReason,
    TokenRolloutTiming,
)
from marin.rl.decoding import DecodingConfig
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.environments.mock_env import (
    AdditionTask,
    MoarCatsTask,
    MockEnv,
    NumberComparisonTask,
    OppositesTask,
    compute_soft_reward,
)
from marin.rl.types import Rollout, RolloutGroup
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage


def create_test_tokenizer():
    """Create a simple test tokenizer that encodes chars as ord values."""

    class SimpleTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(chr(tid) for tid in token_ids)

        def __len__(self):
            return 256

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            # Simple: just return tokens for the user message content
            return [ord(c) for c in messages[0]["content"]]

        def convert_tokens_to_ids(self, token):
            # In our simple test tokenizer, tokens are single chars
            return ord(token[0]) if token else 0

    return SimpleTokenizer()


def create_test_logprobs(text: str):
    """Create logprobs content for a response text."""
    # Local import: ChoiceLogprobsLogprob is not exported from this submodule in
    # older openai SDK versions installed in some CI environments.
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

        def supports_token_rollouts(self):
            return False

        def batch_completions(
            self,
            prompts,
            n,
            decoding,
            system_prompt=None,
        ):
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
            self,
            prompt,
            choice,
            env_name,
            env_example_id,
            reward,
            decoding,
            system_prompt=None,
            correctness_reward=None,
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
                decoding=decoding.as_trace(),
                is_truncated=False,
                correctness_reward=correctness_reward,
            )

        def create_rollout_group_from_completion(self, prompt, completion, env_name, env_example_id, reward_fn):
            rollouts = []
            for choice in completion.choices:
                response_text = choice.message.content
                reward = reward_fn(response_text)
                rollout = self.create_rollout_from_choice(
                    prompt,
                    choice,
                    env_name,
                    env_example_id,
                    reward,
                    decoding=DecodingConfig(temperature=1.0),
                )
                rollouts.append(rollout)

            return RolloutGroup(rollouts=rollouts)

    return TestInferenceContext()


class DummyTokenInferenceContext(BaseInferenceContext):
    def __init__(self):
        self.tokenizer = create_test_tokenizer()
        self.batch = None

    def reload_model(self, model, state_dict):
        return model

    def shutdown(self) -> None:
        pass

    def supports_token_rollouts(self) -> bool:
        return True

    def tokenizer_identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(name_or_path="simple-tokenizer", vocab_size=len(self.tokenizer))

    def batch_completions(self, prompts, n, decoding, system_prompt=None):
        raise AssertionError("token-native path should not call batch_completions")

    def generate_token_rollouts(self, batch):
        self.batch = batch
        response_tokens = tuple(self.tokenizer.encode("mock_response", add_special_tokens=False))
        rollouts = []
        for request in batch.requests:
            for generation_index in range(request.n_generations):
                rollouts.append(
                    TokenizedRollout(
                        request_id=request.request_id,
                        generation_index=generation_index,
                        prompt_token_ids=request.prompt_token_ids,
                        completion_token_ids=response_tokens,
                        completion_logprobs=tuple(-1.0 for _ in response_tokens),
                        finish_reason=TokenRolloutFinishReason.STOP,
                        prompt_mask=tuple(False for _ in request.prompt_token_ids),
                        completion_mask=tuple(True for _ in response_tokens),
                        metadata={"backend": "dummy-token"},
                    )
                )
        return TokenizedRolloutBatchResult(
            batch_id=batch.batch_id,
            tokenizer=batch.tokenizer,
            policy=batch.policy,
            rollouts=tuple(rollouts),
            timing=TokenRolloutTiming(total=0.1),
            admission=TokenRolloutAdmissionMetadata(prefill_admissions=1, prefill_prompt_tokens_per_admission=(10,)),
        )


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


def test_mock_env_uses_token_rollout_path_when_supported():
    env = MockEnv(task_type="addition", seed=0)
    inference_ctx = DummyTokenInferenceContext()

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=2,
        decoding=DecodingConfig(temperature=0.7, top_k=8, max_output_tokens=16),
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert metrics == {}
    assert inference_ctx.batch is not None
    assert inference_ctx.batch.batch_id == "mock_env.addition.train"
    assert len(inference_ctx.batch.requests) == 1
    assert inference_ctx.batch.requests[0].n_generations == 2
    assert inference_ctx.batch.requests[0].sampling.top_k == 8

    assert len(rollout_groups) == 1
    assert len(rollout_groups[0].rollouts) == 2
    rollout = rollout_groups[0].rollouts[0]
    assert rollout.env_name == "mock_env:addition"
    assert rollout.metadata.token_rollout_backend == "dummy-token"
    assert rollout.metadata.token_rollout_request_id == "mock_env.addition.train:0"
    assert rollout.metadata.token_rollout_generation_index == 0
    assert rollout.metadata.token_rollout_finish_reason == TokenRolloutFinishReason.STOP.value
