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

"""GSM8K RL environment with MathEnv-compatible scoring."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import datasets
import jax
import numpy as np

from marin.rl.math_utils import grade_answer, normalize_answer, validate_format
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup
from .base import MarinEnv

logger = logging.getLogger(__name__)

TRAIN_DATA_SOURCE = "openai/gsm8k"
EVAL_DATA_SOURCE = "openai/gsm8k"
DATASET_CONFIG = "main"


@dataclass(slots=True)
class GSM8KExample:
    """Container for a single GSM8K prompt/answer pair."""

    raw_question: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


LoadDatasetFn = Callable[..., Any]


class GSM8KEnv(MarinEnv):
    """GSM8K environment mirroring MathEnv formatting and scoring."""

    INSTRUCTION: str = (
        "Return the final answer in <answer> </answer> tags using standard math notation. "
        "e.g. <answer>42</answer>, or <answer>1/23</answer>."
    )

    def __init__(
        self,
        train_source: str = TRAIN_DATA_SOURCE,
        eval_source: str = EVAL_DATA_SOURCE,
        *,
        dataset_config: str = DATASET_CONFIG,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        trust_remote_code: bool = True,
        datasets_loader: LoadDatasetFn | None = None,
        train_dataset: Iterable[dict[str, Any]] | None = None,
        eval_dataset: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.train_source = train_source
        self.eval_source = eval_source
        self.dataset_config = dataset_config
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self._trust_remote_code = trust_remote_code
        self._datasets_loader = datasets_loader or datasets.load_dataset
        self._rng = np.random.default_rng(seed)

        self.train_examples = self._prepare_split(
            split_name="train",
            hf_split="train",
            examples_iter=train_dataset,
            source=train_source,
            limit=max_train_examples,
        )
        self.eval_examples = self._prepare_split(
            split_name="test",
            hf_split="test",
            examples_iter=eval_dataset,
            source=eval_source,
            limit=max_eval_examples,
        )

        logger.info(
            "Initialized GSM8KEnv with %d train examples and %d eval examples.",
            len(self.train_examples),
            len(self.eval_examples),
        )

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def add_instruction(self, question: str) -> str:
        """Append the standard instruction to the raw GSM8K question."""

        return f"{question.strip()}\n\n{self.INSTRUCTION}"

    def clean_example(self, raw_question: str, raw_answer: str, example_id: str) -> GSM8KExample | None:
        """Normalize the GSM8K example into a usable prompt/answer pair."""

        processed_prompt = self.add_instruction(raw_question)
        final_answer = self._extract_final_answer(raw_answer)
        normalized_answer = normalize_answer(final_answer)

        if normalized_answer is None:
            return None

        return GSM8KExample(
            raw_question=raw_question,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=normalized_answer,
            example_id=example_id,
        )

    def _prepare_split(
        self,
        *,
        split_name: str,
        hf_split: str,
        examples_iter: Iterable[dict[str, Any]] | None,
        source: str,
        limit: int | None,
    ) -> list[GSM8KExample]:
        """Load and clean a dataset split."""

        if examples_iter is None:
            dataset = self._datasets_loader(
                source,
                name=self.dataset_config,
                split=hf_split,
                trust_remote_code=self._trust_remote_code,
            )
        else:
            dataset = examples_iter

        cleaned_examples: list[GSM8KExample] = []
        total = 0
        for idx, item in enumerate(dataset):
            example_id = f"{split_name}_{idx}"
            example = self.clean_example(item["question"], item["answer"], example_id)
            if example is None:
                continue

            example.metadata.update({"split": split_name, "source_index": idx, "source_dataset": source})
            cleaned_examples.append(example)
            total += 1
            if limit is not None and total >= limit:
                break

        return cleaned_examples

    def _extract_final_answer(self, raw_answer: str | None) -> str | None:
        """Extract the canonical answer from the GSM8K solution text."""

        if not raw_answer:
            return None

        parts = raw_answer.split("####")
        final_segment = parts[-1].strip() if parts else raw_answer.strip()
        return final_segment or None

    # ------------------------------------------------------------------
    # RL Environment interface
    # ------------------------------------------------------------------
    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
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
        seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

        prompts = [example.processed_prompt for example in sampled_examples]
        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            stop=stop,
            system_prompt=system_prompt,
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
                reward, fmt_score, correct_score, token_reward = self._score_choice(
                    example=example, response_text=choice.message.content, tokenizer=inference_ctx.tokenizer
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="gsm8k",
                    env_example_id=example.example_id,
                    reward=token_reward,
                    correctness_reward=correct_score,
                    temperature=temperature,
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

        prefix = f"gsm8k.{mode}"
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

    def _score_choice(self, example: GSM8KExample, response_text: str, tokenizer) -> tuple[float, float, float, float]:
        """Score a single generated response text using MathEnv logic."""

        decoded_response = response_text.strip()
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
    def training_data(self) -> Iterator[GSM8KExample]:
        """Stream cleaned training examples for debugging."""

        yield from self.train_examples

    def eval_data(self) -> Iterator[GSM8KExample]:
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
