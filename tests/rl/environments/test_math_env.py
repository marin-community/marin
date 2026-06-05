# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for MathEnv with new InferenceContext paradigm."""

import jax.random
import numpy as np
import pytest
from marin.inference.types import (
    PolicyIdentity,
    TokenizedRollout,
    TokenizedRolloutBatchResult,
    TokenizerIdentity,
    TokenRolloutAdmissionMetadata,
    TokenRolloutFinishReason,
    TokenRolloutTiming,
)
from marin.rl.decoding import DecodingConfig
from marin.rl.environments.inference_ctx import LevanterInferenceContext
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.environments.math_env import MathEnv
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage


def create_mock_chat_completion(tokenizer) -> ChatCompletion:
    """Create a mock ChatCompletion with logprobs for testing."""
    response_text: str = "\\boxed{4}"
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

    def batch_completions(
        self,
        prompts,
        n,
        decoding,
        system_prompt=None,
    ):
        """Return mock completions for each prompt."""
        return [create_mock_chat_completion(self.tokenizer) for prompt in prompts]

    def supports_token_rollouts(self) -> bool:
        return False


class DummyTokenInferenceContext(BaseInferenceContext):
    def __init__(self, tokenizer, generation_limit: int | None = None):
        self.tokenizer = tokenizer
        self.batch = None
        self.generation_limit = generation_limit

    def reload_model(self, model, state_dict):
        return model

    def shutdown(self) -> None:
        pass

    def supports_token_rollouts(self) -> bool:
        return True

    def tokenizer_identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(name_or_path="gpt2", vocab_size=len(self.tokenizer))

    def policy_identity(self) -> PolicyIdentity:
        return PolicyIdentity(policy_name="test-policy", checkpoint_ref="test-checkpoint")

    def generate_token_rollouts(self, batch):
        self.batch = batch
        response_tokens = tuple(self.tokenizer.encode("\\boxed{4}", add_special_tokens=False))
        rollouts = []
        for request in batch.requests:
            generation_count = request.n_generations
            if self.generation_limit is not None:
                generation_count = min(generation_count, self.generation_limit)
            for generation_index in range(generation_count):
                rollouts.append(
                    TokenizedRollout(
                        request_id=request.request_id,
                        generation_index=generation_index,
                        prompt_token_ids=request.prompt_token_ids,
                        completion_token_ids=response_tokens,
                        completion_logprobs=tuple(-0.5 for _ in response_tokens),
                        finish_reason=TokenRolloutFinishReason.STOP,
                        prompt_mask=tuple(False for _ in request.prompt_token_ids),
                        completion_mask=tuple(True for _ in response_tokens),
                    )
                )
        return TokenizedRolloutBatchResult(
            batch_id=batch.batch_id,
            tokenizer=batch.tokenizer,
            policy=batch.policy,
            rollouts=tuple(rollouts),
            timing=TokenRolloutTiming(total=0.25),
            admission=TokenRolloutAdmissionMetadata(prefill_admissions=1, prefill_prompt_tokens_per_admission=(10,)),
        )

    def batch_completions(self, prompts, n, decoding, system_prompt=None):
        raise AssertionError("token-native path should not call batch_completions")


def test_math_env_reward_calculation(gpt2_tokenizer):
    """Test that math env correctly calculates rewards and creates rollouts."""
    tokenizer = gpt2_tokenizer
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
        decoding=DecodingConfig(temperature=0.7),
        prng_key=prng_key,
        mode="train",
    )

    # Verify structure
    assert len(rollout_groups) == 1
    rollout = rollout_groups[0].rollouts[0]

    response_txt = tokenizer.decode(rollout.response_tokens)
    prompt_txt = tokenizer.decode(rollout.prompt_tokens)
    assert "What is 2+2?" in prompt_txt, (prompt_txt, rollout)
    assert "boxed{4}" in response_txt, (response_txt, rollout)

    # Verify basic rollout properties
    assert rollout.env_name == "math"
    assert rollout.prompt_tokens.dtype == np.int32
    assert rollout.response_tokens.dtype == np.int32
    assert len(rollout.response_logprobs) == len(rollout.response_tokens)
    assert len(rollout.token_rewards) == len(rollout.response_tokens)

    # Original MathEnv reward formula: format_coef * (format_valid - 1) + correct_answer
    # With format_coef=0.1, format_valid=1.0, correct_answer=1.0: reward = 0.1 * 0 + 1.0 = 1.0
    np.testing.assert_allclose(rollout.token_rewards, 1.0), (rollout, metrics)
    assert rollout.episode_reward == pytest.approx(1.0), (rollout, metrics)


def test_math_env_uses_token_rollout_path_when_supported(gpt2_tokenizer):
    inference_ctx = DummyTokenInferenceContext(gpt2_tokenizer)
    train_data = [
        {"problem": "What is 2+2?", "solution": "\\boxed{4}"},
    ]
    env = MathEnv(train_dataset=train_data, eval_dataset=[], max_train_examples=1)

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=2,
        decoding=DecodingConfig(temperature=0.7, top_k=8, max_output_tokens=16),
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert inference_ctx.batch is not None
    assert inference_ctx.batch.batch_id == "math.train"
    assert len(inference_ctx.batch.requests) == 1
    assert inference_ctx.batch.requests[0].n_generations == 2
    assert inference_ctx.batch.requests[0].sampling.top_k == 8
    assert "What is 2+2?" in gpt2_tokenizer.decode(inference_ctx.batch.requests[0].prompt_token_ids)

    assert len(rollout_groups) == 1
    assert len(rollout_groups[0].rollouts) == 2
    rollout = rollout_groups[0].rollouts[0]
    np.testing.assert_array_equal(
        rollout.response_tokens,
        np.array(gpt2_tokenizer.encode("\\boxed{4}", add_special_tokens=False), dtype=np.int32),
    )
    np.testing.assert_allclose(rollout.response_logprobs, -0.5)
    assert rollout.episode_reward == pytest.approx(1.0)
    assert metrics["math.train_total_responses"] == 2.0
    assert metrics["math.train_token_rollout_prefill_admissions"] == 1.0


def test_math_env_token_rollout_path_rejects_missing_generations(gpt2_tokenizer):
    inference_ctx = DummyTokenInferenceContext(gpt2_tokenizer, generation_limit=1)
    env = MathEnv(
        train_dataset=[{"problem": "What is 2+2?", "solution": "\\boxed{4}"}],
        eval_dataset=[],
        max_train_examples=1,
    )

    with pytest.raises(RuntimeError, match="returned 1 generations; expected 2"):
        env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=2,
            decoding=DecodingConfig(temperature=0.7, max_output_tokens=16),
            prng_key=jax.random.PRNGKey(0),
            mode="train",
        )


def test_token_rollout_batch_rejects_unsupported_decoding_fields(gpt2_tokenizer):
    inference_ctx = DummyTokenInferenceContext(gpt2_tokenizer)

    with pytest.raises(ValueError, match="min_p"):
        inference_ctx.create_token_rollout_batch(
            batch_id="batch",
            prompts=["prompt"],
            n=1,
            decoding=DecodingConfig(temperature=0.7, min_p=0.1),
        )
