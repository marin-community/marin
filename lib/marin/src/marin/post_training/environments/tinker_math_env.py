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

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
from tqdm.auto import tqdm

from marin.post_training.tinker_environments.math_env import (
    MathEnv as TinkerMathEnvBase,
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
)
from marin.post_training.tinker_environments.math_grading import extract_boxed, grade_answer, normalize_answer

from .marin_env import EnvExample, EnvStep, InferenceContext, MarinEnv
from .math_utils import last_boxed_only_string

logger = logging.getLogger(__name__)


@dataclass
class DataExample:
    """Single data example with transformations for debugging."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)




class TinkerMathEnv(MarinEnv):
    """Math environment using Tinker's grading and prompt format."""

    def __init__(
        self,
        tokenizer,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        format_coef: float = 0.1,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.grader = grader
        self.timeout = timeout
        self.format_coef = format_coef
        
        # Get few-shot prefix from TinkerMathEnvBase
        self.fewshot_prefix = TinkerMathEnvBase.standard_fewshot_prefix()

        self.train_examples = [
            {"prompt": ex.processed_prompt, "answer": ex.processed_answer} for ex in self.training_data()
        ]

        self.eval_examples = [{"prompt": ex.processed_prompt, "answer": ex.processed_answer} for ex in self.eval_data()]

        print(
            f"Initialized TinkerMathEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

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

    def step(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            inference_ctx: Context for generating responses (hides model params)
            n_examples: Number of examples to sample
            prng_key: Random key for sampling
            mode: "train" or "eval"
            n_generations: Number of generations per example
            temperature: Generation temperature
        """
        # Sample examples from dataset using JAX random (synchronized across workers)
        if mode == "train":
            available_examples = self.train_examples
        else:
            available_examples = self.eval_examples

        # Use JAX random for consistent sampling across all TPU workers
        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(available_examples))
            indices = jax.random.choice(prng_key, len(available_examples), shape=(n_to_sample,), replace=False)
            examples = [available_examples[int(idx)] for idx in indices]

        # Generate responses using the model
        prompts = [example["prompt"] for example in examples]
        responses = inference_ctx.generate(
            prompts,
            temperature=temperature,
            n_generations=n_generations,
        )

        # Compute rewards
        rewards, metrics = self._compute_rewards(examples, responses, inference_ctx.tokenizer)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

    def _compute_rewards(
        self,
        examples: list[EnvExample],
        responses: list[list[dict[str, np.ndarray]]],
        tokenizer,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compute rewards for generated responses using Tinker's grading."""
        all_rewards = []
        all_format_rewards = []
        all_correct_rewards = []
        all_lens = []

        for i, response in tqdm(enumerate(responses)):
            group_rewards = []
            group_format_rewards = []
            group_correct_rewards = []
            for j, inner_response in enumerate(response):
                all_lens.append(len(inner_response["tokens"]))
                decoded_response = tokenizer.decode(inner_response["tokens"], skip_special_tokens=True)

                true_answer = examples[i]["answer"].strip()

                # Follows: https://github.com/thinking-machines-lab/tinker-cookbook/blob/5469e4a2453cf8ecd950a0e66f5bb0a1ee898b9a/tinker_cookbook/rl/problem_env.py#L62
                # Check format using Tinker's extract_boxed
                format_valid = float(self.check_format(decoded_response))

                # Grade using Tinker's grading
                correct_answer = float(self.check_answer(decoded_response, true_answer))

                reward = self.format_coef * (format_valid - 1) + correct_answer

                if i == 0 and j == 0:
                    print("=" * 25)
                    print(f"Prompt: {examples[i]['prompt']}")
                    print(f"Response: {decoded_response}")
                    print("=" * 25)
                    print(f"True answer: {true_answer}")
                    extracted_answer = None
                    if format_valid:
                        try:
                            extracted_answer = extract_boxed(decoded_response)
                        except ValueError:
                            pass
                    print(f"Extracted answer: {extracted_answer}")
                    print(f"Valid format: {format_valid}")
                    print(f"Correct: {correct_answer}")
                    print(f"Reward: {reward}")
                    print("=" * 25)

                group_rewards.append(reward)
                group_format_rewards.append(float(format_valid))
                group_correct_rewards.append(correct_answer)

            all_rewards.append(group_rewards)
            all_format_rewards.append(group_format_rewards)
            all_correct_rewards.append(group_correct_rewards)

        all_rewards = np.asarray(all_rewards)
        all_format_rewards = np.asarray(all_format_rewards)
        all_correct_rewards = np.asarray(all_correct_rewards)
        all_lens = np.asarray(all_lens)

        metrics = {
            "train/rewards": np.mean(all_rewards),
            "train/format_rewards": np.mean(all_format_rewards),
            "train/correct_rewards": np.mean(all_correct_rewards),
            "train/output_len": np.mean(all_lens),
        }

        return all_rewards, metrics

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples for evaluation."""
        # Use a fixed seed for reproducible evaluation
        eval_key = jax.random.PRNGKey(42)
        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(self.eval_examples))
            indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
            return [self.eval_examples[int(idx)] for idx in indices]

    def clean_example(self, raw_prompt, raw_answer) -> DataExample:
        """Clean and process a single example."""
        # Show the transformation pipeline
        boxed_answer = last_boxed_only_string(raw_answer)
        # Use Tinker's normalize_answer instead of our own
        cleaned_answer = normalize_answer(boxed_answer) if boxed_answer else normalize_answer(raw_answer)
        
        # Build chat messages with few-shot examples
        question_text = self.add_instruction(raw_prompt)
        messages = self.fewshot_prefix + [
            {"role": "user", "content": question_text}
        ]
        
        # Apply chat template to get the formatted prompt
        processed_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return DataExample(
            raw_prompt=raw_prompt,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=cleaned_answer,
        )

    def training_data(self) -> Iterator[DataExample]:
        train_dataset = _get_hendrycks_math_train()

        for item in train_dataset:
            raw_prompt = item["problem"]
            raw_answer = item["solution"]

            yield self.clean_example(raw_prompt, raw_answer)

    def eval_data(self) -> Iterator[DataExample]:
        test_dataset = _get_hendrycks_math_test()

        for item in test_dataset:
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            yield self.clean_example(raw_prompt, raw_answer)
