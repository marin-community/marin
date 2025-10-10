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

"""Math-focused RL environment mirroring post-training reward logic."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterable

import datasets
import jax
import jax.numpy as jnp
import numpy as np

from marin.rl.math_utils import (
    grade_answer,
    last_boxed_only_string,
    latex_to_text,
    normalize_answer,
    validate_format,
)
from marin.rl.types import InferenceContext, Rollout, RolloutGroup

from .base import MarinEnv

logger = logging.getLogger(__name__)


TRAIN_DATA_SOURCE = "di-zhang-fdu/MATH12000"
EVAL_DATA_SOURCE = "HuggingFaceH4/MATH-500"


@dataclass(slots=True)
class MathEnvExample:
    """Single math example with cleaned prompt/answer."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


LoadDatasetFn = Callable[..., Any]


class MathEnv(MarinEnv):
    """Math environment for RL training and evaluation."""

    INSTRUCTION: str = (
        "Return the final answer in <answer> </answer> tags using standard math notation. "
        "e.g. <answer>42</answer>, or <answer>1/23</answer>."
    )

    def __init__(
        self,
        train_source: str = TRAIN_DATA_SOURCE,
        eval_source: str = EVAL_DATA_SOURCE,
        *,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        trust_remote_code: bool = True,
        datasets_loader: LoadDatasetFn | None = None,
        train_dataset: Iterable[dict[str, Any]] | None = None,
        eval_dataset: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the math environment.

        Args:
            train_source: Hugging Face dataset path for training split.
            eval_source: Hugging Face dataset path for evaluation split.
            max_train_examples: Optional limit on cached train examples.
            max_eval_examples: Optional limit on cached eval examples.
            seed: Seed for deterministic sampling.
            trust_remote_code: Forwarded to HF dataset loader when used.
            datasets_loader: Injection hook for tests; defaults to datasets.load_dataset.
            train_dataset: Optional iterable of pre-loaded train examples.
            eval_dataset: Optional iterable of pre-loaded eval examples.
        """

        self.train_source = train_source
        self.eval_source = eval_source
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self._trust_remote_code = trust_remote_code
        self._datasets_loader = datasets_loader or datasets.load_dataset
        self._rng = np.random.default_rng(seed)

        self.train_examples = self._prepare_split(
            split_name="train",
            examples_iter=train_dataset,
            source=train_source,
            limit=max_train_examples,
        )
        self.eval_examples = self._prepare_split(
            split_name="test",
            examples_iter=eval_dataset,
            source=eval_source,
            limit=max_eval_examples,
        )

        logger.info(
            "Initialized MathEnv with %d train examples and %d eval examples.",
            len(self.train_examples),
            len(self.eval_examples),
        )

    # ------------------------------------------------------------------
    # Dataset preparation helpers
    # ------------------------------------------------------------------
    def add_instruction(self, math_problem: str) -> str:
        """Append the standard instruction to a math problem."""

        return f"{math_problem}\n\n{self.INSTRUCTION}"

    def clean_example(self, raw_prompt: str, raw_answer: str, example_id: str) -> MathEnvExample | None:
        """Normalize prompt/answer pair.

        Returns None if processed answer could not be computed.
        """

        boxed_answer = last_boxed_only_string(raw_answer)
        cleaned_answer = normalize_answer(boxed_answer) if boxed_answer else normalize_answer(raw_answer)
        if cleaned_answer is None:
            return None

        processed_prompt = self.add_instruction(latex_to_text(raw_prompt))

        return MathEnvExample(
            raw_prompt=raw_prompt,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=cleaned_answer,
            example_id=example_id,
        )

    def _prepare_split(
        self,
        *,
        split_name: str,
        examples_iter: Iterable[dict[str, Any]] | None,
        source: str,
        limit: int | None,
    ) -> list[MathEnvExample]:
        """Load and clean dataset split."""

        if examples_iter is None:
            dataset_dict = self._datasets_loader(source, trust_remote_code=self._trust_remote_code)
            if isinstance(dataset_dict, dict):
                dataset = dataset_dict.get(split_name) or dataset_dict.get("train")
            else:
                dataset = dataset_dict  # type: ignore[assignment]
        else:
            dataset = examples_iter

        cleaned_examples: list[MathEnvExample] = []
        total = 0
        for idx, item in enumerate(dataset):
            example_id = f"{split_name}_{idx}"
            example = self.clean_example(item["problem"], item["solution"], example_id)
            if example is None:
                continue

            example.metadata.update({"split": split_name, "source_index": idx, "source_dataset": source})
            cleaned_examples.append(example)
            total += 1
            if limit is not None and total >= limit:
                break

        return cleaned_examples

    # ------------------------------------------------------------------
    # RL Environment interface
    # ------------------------------------------------------------------
    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts, evaluate responses, and create rollouts."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode}")

        available_examples = self.train_examples if mode == "train" else self.eval_examples
        if not available_examples:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, len(available_examples))
        seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

        prompts = [example.processed_prompt for example in sampled_examples]
        responses = inference_ctx.generate(prompts=prompts, temperature=temperature, n_generations=n_generations)

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        format_sum = 0.0
        correct_sum = 0.0
        response_token_count = 0

        for example, response in zip(sampled_examples, responses, strict=True):
            group_rollouts: list[Rollout] = []

            for choice in response.choices:
                (
                    reward,
                    fmt_score,
                    correct_score,
                    token_reward,
                ) = self._score_choice(example=example, choice=choice, tokenizer=inference_ctx.tokenizer)

                prompt_tokens = jnp.array(response.prompt_tokens, dtype=jnp.int32)
                response_tokens = jnp.array(choice.response_tokens, dtype=jnp.int32)
                response_logprobs = jnp.array(choice.logprobs, dtype=jnp.float32)
                token_rewards = jnp.full(response_tokens.shape, token_reward, dtype=jnp.float32)

                rollout = Rollout(
                    env_name="math",
                    env_example_id=example.example_id,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    response_logprobs=response_logprobs,
                    token_rewards=token_rewards,
                    episode_reward=float(reward),
                )
                group_rollouts.append(rollout)

                total_choices += 1
                reward_sum += reward
                format_sum += fmt_score
                correct_sum += correct_score
                response_token_count += response_tokens.size

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
        }

        return rollout_groups, metrics

    def _score_choice(self, example: MathEnvExample, choice, tokenizer) -> tuple[float, float, float, float]:
        """Score a single generated choice.

        Returns (reward, format_score, correct_score, token_reward_value).
        """

        decoded_response = (choice.response_text or "").strip()
        if not decoded_response:
            decoded_response = tokenizer.decode(choice.response_tokens, skip_special_tokens=True)
        validation = validate_format(decoded_response + ">")

        true_answer = example.processed_answer.strip()
        weak_correct = 1.0 if true_answer and true_answer in decoded_response else 0.0

        if validation["is_valid"]:
            grade = grade_answer(validation["answer"], true_answer)
        else:
            tokens = decoded_response.split()
            grade = grade_answer(tokens[-1], true_answer) if tokens else 0.0

        reward = 0.3 * weak_correct + 0.1 * float(validation["is_valid"]) + 0.8 * float(grade)

        return reward, float(validation["is_valid"]), float(grade), reward

    # ------------------------------------------------------------------
    # Dataset inspection helpers
    # ------------------------------------------------------------------
    def training_data(self) -> Iterator[MathEnvExample]:
        """Stream cleaned training examples for debugging."""

        yield from self.train_examples

    def eval_data(self) -> Iterator[MathEnvExample]:
        """Stream cleaned evaluation examples for debugging."""

        yield from self.eval_examples

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
