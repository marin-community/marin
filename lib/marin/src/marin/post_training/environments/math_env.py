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

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import datasets
import jax
import numpy as np
from tqdm.auto import tqdm

from marin.post_training.flax.utils import validate_format

from .marin_env import EnvExample, EnvStep, InferenceContext, MarinEnv
from .math_utils import grade_answer, last_boxed_only_string, latex_to_text, normalize_answer


@dataclass
class DataExample:
    """Single data example with transformations for debugging."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


TRAIN_DATA_SOURCE = "di-zhang-fdu/MATH12000"
TEST_DATA_SOURCE = "HuggingFaceH4/MATH-500"


class MathEnv(MarinEnv):
    INSTRUCTION: str = (
        "Return the final answer in <answer> </answer> tags using standard math notation. "
        + "e.g. <answer>42</answer>, or <answer>1/23</answer>."
    )

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        self.train_examples = [
            {"prompt": ex.processed_prompt, "answer": ex.processed_answer} for ex in self.training_data()
        ]

        self.eval_examples = [{"prompt": ex.processed_prompt, "answer": ex.processed_answer} for ex in self.eval_data()]

        print(
            f"Initialized MathEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

    def add_instruction(self, math_problem: str) -> str:
        """Add the standard instruction to a math problem."""
        return f"{math_problem} {self.INSTRUCTION}"

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
        """Compute rewards for generated responses."""
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
                validation = validate_format(decoded_response + ">")

                true_answer = examples[i]["answer"].strip()

                # give a weak correct response if the answer is contained anywhere in the response
                if re.search(rf"\b{re.escape(true_answer)}\b", decoded_response):
                    weak_correct = 1.0
                else:
                    weak_correct = 0.0

                if validation["is_valid"]:
                    grade = grade_answer(validation["answer"], examples[i]["answer"])
                else:
                    # try the last token to see if we get a hit
                    splits = decoded_response.split()
                    if len(splits) > 0:
                        grade = grade_answer(decoded_response.split()[-1], examples[i]["answer"])
                    else:
                        grade = 0.0

                reward = 0.3 * weak_correct + 0.1 * validation["is_valid"] + 0.8 * grade

                if i == 0 and j == 0:
                    print("=" * 25)
                    print(f"Prompt: {examples[i]['prompt']}")
                    print(f"Response: {decoded_response}")
                    print("=" * 25)
                    print(f"True answer: {examples[i]['answer']}")
                    print(f"Extracted answer: {validation['answer']}")
                    print(f"Weak correct: {weak_correct}")
                    print(f"Valid format: {validation['is_valid']}")
                    print(f"Grade: {grade}")
                    print(f"Reward: {reward}")
                    print("=" * 25)

                group_rewards.append(reward)
                group_format_rewards.append(float(validation["is_valid"]))
                group_correct_rewards.append(float(grade))

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
        cleaned_answer = normalize_answer(boxed_answer) if boxed_answer else normalize_answer(raw_answer)
        processed_prompt = self.add_instruction(latex_to_text(raw_prompt))

        return DataExample(
            raw_prompt=raw_prompt,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=cleaned_answer,
        )

    def training_data(self) -> Iterator[DataExample]:
        train_dataset = datasets.load_dataset(TRAIN_DATA_SOURCE, trust_remote_code=True)["train"]

        for item in train_dataset:
            raw_prompt = item["problem"]
            raw_answer = item["solution"]

            yield self.clean_example(raw_prompt, raw_answer)

    def eval_data(self) -> Iterator[DataExample]:
        test_dataset = datasets.load_dataset(TEST_DATA_SOURCE, trust_remote_code=True)["test"]

        for item in test_dataset:
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            yield self.clean_example(raw_prompt, raw_answer)
