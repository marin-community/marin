# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from marin.rl.environments.base import FiniteDatasetEnv
from marin.rl.types import Rollout
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


@dataclass(frozen=True)
class _ToyExample:
    prompt: str
    example_id: str


class _ToyFiniteEnv(FiniteDatasetEnv):
    def __init__(self):
        self.train_examples = [_ToyExample(f"train-{idx}", f"train_{idx}") for idx in range(3)]
        self.eval_examples = [_ToyExample(f"eval-{idx}", f"eval_{idx}") for idx in range(3)]

    @property
    def env_name(self) -> str:
        return "toy"

    def train_len(self) -> int:
        return len(self.train_examples)

    def eval_len(self) -> int:
        return len(self.eval_examples)

    def train_examples_by_indices(self, indices):
        return [self.train_examples[idx] for idx in indices]

    def eval_examples_by_indices(self, indices):
        return [self.eval_examples[idx] for idx in indices]

    def inference_prompt_for_example(self, example):
        return example.prompt

    def rollout_prompt_for_example(self, example) -> str:
        return example.prompt

    def example_id(self, example) -> str:
        return example.example_id

    def score_choice(self, example, response_text: str, finish_reason: str, tokenizer):
        return 1.0, 1.0, 1.0


class _ToyInferenceContext:
    def __init__(self):
        self.calls = []
        self.tokenizer = object()

    def batch_completions(self, prompts, temperature, n, max_tokens=None, top_k=None, stop=None, system_prompt=None):
        self.calls.append(
            {
                "prompts": prompts,
                "temperature": temperature,
                "n": n,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "stop": stop,
            }
        )
        return [_completion(n) for _ in prompts]

    def create_rollout_from_choice(
        self,
        prompt,
        choice,
        env_name,
        env_example_id,
        reward,
        temperature,
        top_k=None,
        system_prompt=None,
        correctness_reward=None,
    ):
        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=np.array([ord(prompt[-1])], dtype=np.int32),
            response_tokens=np.array([1], dtype=np.int32),
            response_logprobs=np.array([-0.1], dtype=np.float32),
            token_rewards=np.array([reward], dtype=np.float32),
            episode_reward=reward,
            temperature=temperature,
            top_k=top_k,
            is_truncated=False,
            correctness_reward=correctness_reward,
        )


def _completion(n: int) -> ChatCompletion:
    return ChatCompletion(
        id="completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=idx,
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
            for idx in range(n)
        ],
        created=0,
        model="test",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=n, prompt_tokens=1, total_tokens=n + 1),
    )


def test_finite_dataset_env_sample_by_indices_preserves_requested_order():
    env = _ToyFiniteEnv()
    inference_ctx = _ToyInferenceContext()

    groups, metrics = env.sample_by_indices(
        inference_ctx=inference_ctx,
        indices=[2, 0, 1],
        n_generations=2,
        temperature=0.7,
        mode="eval",
        max_tokens=16,
        top_k=5,
    )

    assert inference_ctx.calls[0] == {
        "prompts": ["eval-2", "eval-0", "eval-1"],
        "temperature": 0.7,
        "n": 2,
        "max_tokens": 16,
        "top_k": 5,
        "stop": None,
    }
    assert [group.rollouts[0].env_example_id for group in groups] == ["eval_2", "eval_0", "eval_1"]
    assert metrics["toy.eval_sampled_examples"] == 3.0
    assert metrics["toy.eval_total_responses"] == 6.0
