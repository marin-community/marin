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

"""Mock environment for testing RL training without external dependencies."""

from typing import Any

import jax
import numpy as np

from marin.post_training.inference import batch_inference

from .marin_env import EnvStep, MarinEnv


class MockEnv(MarinEnv):
    """Mock environment that generates synthetic data for testing."""

    INSTRUCTION: str = "Answer the following question: "

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Create small synthetic datasets
        self.train_examples = [
            {"prompt": self.add_instruction("What is 2 + 2?"), "answer": "4"},
            {"prompt": self.add_instruction("What is 3 + 3?"), "answer": "6"},
            {"prompt": self.add_instruction("What is 5 + 1?"), "answer": "6"},
            {"prompt": self.add_instruction("What is 4 + 4?"), "answer": "8"},
        ]

        self.eval_examples = [
            {"prompt": self.add_instruction("What is 1 + 1?"), "answer": "2"},
            {"prompt": self.add_instruction("What is 7 + 3?"), "answer": "10"},
        ]

        print(
            f"Initialized MockEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

    def step(
        self,
        sampler,
        params,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
        **kwargs,
    ) -> EnvStep:
        """Generate synthetic rollouts for testing."""
        # Sample examples from dataset using JAX random
        if mode == "train":
            available_examples = self.train_examples
        else:
            available_examples = self.eval_examples

        # Use JAX random for consistent sampling across all TPU workers
        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(available_examples))
            indices = jax.random.choice(prng_key, len(available_examples), shape=(n_to_sample,), replace=True)
            indices = jax.device_get(indices)  # Materialize to local CPU for indexing
            examples = [available_examples[int(idx)] for idx in indices]

        # Generate responses
        prompts = [example["prompt"] for example in examples]
        responses = batch_inference(sampler, params, prompts, prng_key, n_generations, verbose=False)

        # Compute rewards and metrics
        rewards, metrics = self._compute_rewards(examples, responses)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples."""
        return self.eval_examples[:n_examples]

    def add_instruction(self, problem: str) -> str:
        """Add instruction to problem."""
        return self.INSTRUCTION + problem

    def _compute_rewards(self, examples: list[dict], responses: list[list[dict[str, Any]]]) -> tuple[np.ndarray, dict]:
        """Compute rewards for generated responses."""
        n_examples = len(examples)
        n_generations = len(responses[0]) if responses else 1

        # Initialize rewards array
        rewards = np.zeros((n_examples, n_generations), dtype=np.float32)

        correct_count = 0
        total_count = 0

        for example_idx, (example, response_list) in enumerate(zip(examples, responses, strict=False)):
            correct_answer = example["answer"]

            for gen_idx, response in enumerate(response_list):
                # Decode the generated tokens to text
                decoded_response = self.tokenizer.decode(response["tokens"], skip_special_tokens=True)

                # Simple reward: 1 if generation contains the correct answer, 0 otherwise
                reward = 1.0 if correct_answer in decoded_response else 0.0
                rewards[example_idx, gen_idx] = reward

                if reward > 0:
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        mean_reward = float(np.mean(rewards))

        metrics = {
            "train_accuracy": accuracy,
            "train_mean_reward": mean_reward,
            "train_total_examples": total_count,
        }

        return rewards, metrics
