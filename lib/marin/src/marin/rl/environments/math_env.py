# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from marin.rl.environments.tinker_environments.math_env import (
    MathEnv as TinkerMathEnvBase,
)
from marin.rl.environments.tinker_environments.math_env import (
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
)
from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer, normalize_answer
from marin.rl.math_utils import last_boxed_only_string

from .base import FiniteDatasetEnv

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


class MathEnv(FiniteDatasetEnv):
    """Math environment using Tinker's grading and prompt format."""

    @property
    def env_name(self) -> str:
        return "math"

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

    def train_len(self) -> int:
        return len(self.train_examples)

    def eval_len(self) -> int:
        return len(self.eval_examples)

    def train_examples_by_indices(self, indices: Sequence[int]) -> list[DataExample]:
        return [self.train_examples[idx] for idx in indices]

    def eval_examples_by_indices(self, indices: Sequence[int]) -> list[DataExample]:
        return [self.eval_examples[idx] for idx in indices]

    def inference_prompt_for_example(self, example: DataExample) -> list[dict[str, str]]:
        return [*self.fewshot_prefix, {"role": "user", "content": example.processed_prompt}]

    def rollout_prompt_for_example(self, example: DataExample) -> str:
        return example.processed_prompt

    def example_id(self, example: DataExample) -> str:
        return example.example_id

    def score_choice(
        self, example: DataExample, response_text: str, finish_reason: str, tokenizer
    ) -> tuple[float, float, float]:
        return self._score_choice(example, response_text, finish_reason, tokenizer)

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
        """Return evaluation examples in stable dataset order."""

        if not self.eval_examples:
            return []

        n_to_sample = min(n_examples, len(self.eval_examples))
        return [
            {
                "prompt": example.processed_prompt,
                "answer": example.processed_answer,
                "example_id": example.example_id,
            }
            for example in self.eval_examples[:n_to_sample]
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
