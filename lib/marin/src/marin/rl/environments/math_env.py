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

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from marin.rl.environments.tinker_environments.math_env import (
    MathEnv as TinkerMathEnvBase,
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
)
from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer, normalize_answer
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.math_utils import last_boxed_only_string
from marin.rl.types import Rollout, RolloutGroup

from .base import MarinEnv

logger = logging.getLogger(__name__)


@dataclass
class DataExample:
    """Single data example with transformations for debugging."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MathEnv(MarinEnv):
    """Math environment using Tinker's grading and prompt format."""

    def __init__(
        self,
        tokenizer=None,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        format_coef: float = 0.1,
        train_dataset: list[dict[str, Any]] | None = None,
        eval_dataset: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self.format_coef = format_coef
        self._rng = np.random.default_rng(seed)

        # Get few-shot prefix from TinkerMathEnvBase
        self.fewshot_prefix = TinkerMathEnvBase.standard_fewshot_prefix()

        # Use provided datasets or load defaults
        if train_dataset is not None:
            self.train_examples = self._prepare_split(train_dataset, "train", max_train_examples)
        else:
            self.train_examples = list(self.training_data())
            if max_train_examples is not None:
                self.train_examples = self.train_examples[:max_train_examples]

        if eval_dataset is not None:
            self.eval_examples = self._prepare_split(eval_dataset, "test", max_eval_examples)
        else:
            self.eval_examples = list(self.eval_data())
            if max_eval_examples is not None:
                self.eval_examples = self.eval_examples[:max_eval_examples]

        logger.info(
            "Initialized MathEnv with %d train examples and %d eval examples.",
            len(self.train_examples),
            len(self.eval_examples),
        )

    def _prepare_split(
        self,
        dataset: list[dict[str, Any]],
        split_name: str,
        limit: int | None,
    ) -> list[DataExample]:
        """Process a dataset split into DataExample objects."""
        examples = []
        for idx, item in enumerate(dataset):
            example_id = f"{split_name}_{idx}"
            example = self.clean_example(item["problem"], item["solution"], example_id)
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
        return examples

    @classmethod
    def question_suffix(cls) -> str:
        """Use Tinker's question suffix format."""
        return TinkerMathEnvBase.question_suffix()

    def add_instruction(self, math_problem: str) -> str:
        """Add the standard instruction to a math problem."""
        return f"{math_problem}{self.question_suffix()}"

    def check_format(self, sample_str: str) -> bool:
        """Check if the response follows the boxed format."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str, ground_truth: str) -> bool:
        """Grade the answer using Tinker's grading."""
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return grade_answer(answer, ground_truth)

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts, evaluate responses, and create rollouts."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode}")

        available_examples = self.train_examples if mode == "train" else self.eval_examples
        if not available_examples:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, len(available_examples))
        if isinstance(prng_key, int):
            seed = prng_key
        else:
            seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

        # Build message lists with few-shot examples for each prompt
        prompts = [
            [*self.fewshot_prefix, {"role": "user", "content": example.processed_prompt}] for example in sampled_examples
        ]
        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=None,  # No system prompt - use few-shot examples instead
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        format_sum = 0.0
        correct_sum = 0.0
        response_token_count = 0
        truncated_count = 0

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []

            for choice in completion.choices:
                reward, fmt_score, correct_score = self._score_choice(
                    example=example,
                    response_text=choice.message.content,
                    finish_reason=choice.finish_reason,
                    tokenizer=inference_ctx.tokenizer,
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="math",
                    env_example_id=example.example_id,
                    reward=reward,
                    correctness_reward=correct_score,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=system_prompt,
                )

                group_rollouts.append(rollout)
                total_choices += 1
                reward_sum += reward
                format_sum += fmt_score
                correct_sum += correct_score
                response_token_count += rollout.response_tokens.size

                if choice.finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = f"math.{mode}"
        metrics = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_format_accuracy": format_sum / total_choices,
            f"{prefix}_correct_accuracy": correct_sum / total_choices,
            f"{prefix}_mean_response_tokens": response_token_count / total_choices,
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(sampled_examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }

        return rollout_groups, metrics

    def _score_choice(
        self, example: DataExample, response_text: str, finish_reason: str, tokenizer
    ) -> tuple[float, float, float]:
        """Score a single generated response text using MathEnv logic."""

        decoded_response = response_text.strip()

        # Penalize truncated responses
        parse_success = finish_reason != "length"

        # Check format
        format_valid = float(parse_success and self.check_format(decoded_response))

        true_answer = example.processed_answer.strip()

        # Grade using Tinker's grading
        correct_answer = float(self.check_answer(decoded_response, true_answer))

        reward = self.format_coef * (format_valid - 1) + correct_answer

        return reward, format_valid, correct_answer

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Sample evaluation examples deterministically."""

        if not self.eval_examples:
            return []

        eval_key = jax.random.PRNGKey(42)
        n_to_sample = min(n_examples, len(self.eval_examples))
        indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
        return [
            {
                "prompt": self.eval_examples[int(idx)].processed_prompt,
                "answer": self.eval_examples[int(idx)].processed_answer,
                "example_id": self.eval_examples[int(idx)].example_id,
            }
            for idx in indices
        ]

    def clean_example(self, raw_prompt: str, raw_answer: str, example_id: str) -> DataExample:
        """Clean and process a single example."""
        # Show the transformation pipeline
        boxed_answer = last_boxed_only_string(raw_answer)
        # Use Tinker's normalize_answer instead of our own
        cleaned_answer = normalize_answer(boxed_answer) if boxed_answer else normalize_answer(raw_answer)

        # Just add instruction suffix - chat template will be applied by inference context
        processed_prompt = self.add_instruction(raw_prompt)

        return DataExample(
            raw_prompt=raw_prompt,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=cleaned_answer,
            example_id=example_id,
        )

    def training_data(self) -> Iterator[DataExample]:
        train_dataset = _get_hendrycks_math_train()

        for idx, item in enumerate(train_dataset):
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            example_id = f"train_{idx}"

            yield self.clean_example(raw_prompt, raw_answer, example_id)

    def eval_data(self) -> Iterator[DataExample]:
        test_dataset = _get_hendrycks_math_test()

        for idx, item in enumerate(test_dataset):
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            example_id = f"test_{idx}"
            yield self.clean_example(raw_prompt, raw_answer, example_id)
