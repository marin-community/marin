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

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from marin.rl.types import InferenceContext, Rollout, RolloutGroup

from .base import MarinEnv

NUM_TRAIN_EXAMPLES = 1000
NUM_EVAL_EXAMPLES = 100

logger = logging.getLogger(__name__)


@dataclass
class MockEnvExample:
    """Single data example with transformations for debugging."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    """Protocol for task implementations."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        """Generate training examples."""
        ...

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        """Generate evaluation examples."""
        ...

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        """Compute reward for a response."""
        ...


class AdditionTask:
    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(0, 1000)
            b = rng.integers(0, 1000)
            result = a + b
            prompt = f"What is {a}+{b}? Output just the number:"
            answer = str(result)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class OppositesTask:
    """Simple opposite words - 75% success rate target."""

    OPPOSITES: ClassVar[list[tuple[str, str]]] = [
        ("hot", "cold"),
        ("big", "small"),
        ("up", "down"),
        ("yes", "no"),
        ("in", "out"),
        ("day", "night"),
        ("fast", "slow"),
        ("old", "new"),
        ("happy", "sad"),
        ("good", "bad"),
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            word, opposite = self.OPPOSITES[rng.integers(len(self.OPPOSITES))]
            prompt = f"Opposite of {word}? One word:"
            answer = opposite
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class NumberComparisonTask:
    """Compare two numbers - 50% success rate target."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(1, 100)
            b = rng.integers(1, 100)
            if a != b:
                if rng.random() < 0.5:
                    prompt = f"Bigger: {a} or {b}? Just the number:"
                    answer = str(max(a, b))
                else:
                    prompt = f"Smaller: {a} or {b}? Just the number:"
                    answer = str(min(a, b))
                examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        format_score = 1.0 if actual_response.strip().isdigit() else 0.0
        return 0.1 * format_score + 0.9 * compute_soft_reward(correct_answer, actual_response)


def compute_soft_reward(correct_answer: str, actual_response: str) -> float:
    """Compute soft reward with partial credit for correctness and format."""
    if not actual_response:
        return 0.0

    # remove commas from numbers
    correct_answer = correct_answer.replace(",", "").lower()

    tokens = actual_response.split()
    correct_score = 0
    for token in tokens:
        token = token.replace(",", "").lower()

        if token == correct_answer:
            correct_score = 1
            break

    format_score = min(1.0, 1 / max(len(tokens), 1))
    return 0.7 * correct_score + 0.3 * format_score


class MoarCatsTask:
    """Make moar cats."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            prompt = "i like cats, i love cats, give me moar cats."
            answer = "cats"
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        # how many cats
        num_cats = actual_response.lower().count("cat")
        love_cats = actual_response.lower().count("love cats")

        return (num_cats + (10 * love_cats)) / np.sqrt(1 + len(actual_response))


# Task mappings
TASKS = {
    "cats": MoarCatsTask(),
    "addition": AdditionTask(),
    "opposites": OppositesTask(),
    "number_comparison": NumberComparisonTask(),
}


class MockEnv(MarinEnv):
    """Mock environment that generates synthetic data for testing."""

    def __init__(self, task_type: str, seed=None, **kwargs):
        self.task_type = task_type

        # Get the task instance for this task type
        self.task = TASKS.get(task_type)
        if not self.task:
            raise ValueError(f"Unknown task type: {task_type}")

        # Generate examples using the task
        rng = np.random.default_rng(seed)
        self.train_examples = self.task.generate_training_examples(NUM_TRAIN_EXAMPLES, rng)
        self.eval_examples = self.task.generate_eval_examples(NUM_EVAL_EXAMPLES, rng)

        print(
            f"Initialized MockEnv with task '{task_type}': {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample examples, generate responses, and create rollouts."""
        # Select dataset
        if mode == "train":
            available_examples = self.train_examples
        else:
            available_examples = self.eval_examples

        # Sample examples
        n_to_sample = min(n_examples, len(available_examples))
        seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        logger.info("Selecting %d examples with seed %d", n_to_sample, seed)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=True)

        sampled_examples = []
        for idx in indices:
            example = available_examples[int(idx)]
            sampled_examples.append(
                {
                    "prompt": example["prompt"],
                    "answer": example["answer"],
                    "example_id": f"{self.task_type}_example_{idx}",
                }
            )

        # Generate responses
        prompts = [ex["prompt"] for ex in sampled_examples]
        responses = inference_ctx.generate(
            prompts=prompts,
            temperature=temperature,
            n_generations=n_generations,
        )

        # Evaluate and create rollouts
        max_input_length = 2048  # Default, will use what we can get from responses
        rollout_groups = []
        correct_count = 0
        total_count = 0
        format_correct_count = 0

        for example, response in zip(sampled_examples, responses, strict=True):
            rollouts = []

            for choice in response.choices:
                # Extract response text (already decoded by inference context)
                # Note: response_text may include the prompt, so extract the actual response
                if choice.response_text.startswith(example["prompt"]):
                    actual_response = choice.response_text[len(example["prompt"]) :]
                else:
                    actual_response = choice.response_text

                # Compute reward
                reward = self.task.compute_reward(example["answer"], actual_response)

                # Create rollout
                prompt_tokens = response.prompt_tokens[-max_input_length:]
                token_rewards = jnp.full(len(choice.response_tokens), reward, dtype=jnp.float32)

                rollout = Rollout(
                    env_name=f"mock:{self.task_type}",
                    env_example_id=example["example_id"],
                    prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
                    response_tokens=jnp.array(choice.response_tokens, dtype=jnp.int32),
                    response_logprobs=jnp.array(choice.logprobs, dtype=jnp.float32),
                    token_rewards=token_rewards,
                    episode_reward=float(reward),
                )
                rollouts.append(rollout)

                # Track metrics
                if reward > 0:
                    correct_count += 1
                if actual_response:
                    format_correct_count += 1
                total_count += 1

            if rollouts:
                rollout_groups.append(RolloutGroup(rollouts=rollouts))

        # Compute metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        format_accuracy = format_correct_count / total_count if total_count > 0 else 0.0
        mean_reward = (
            sum(r.episode_reward for g in rollout_groups for r in g.rollouts) / total_count if total_count > 0 else 0.0
        )

        metrics = {
            f"{self.task_type}.accuracy": accuracy,
            f"{self.task_type}.format_accuracy": format_accuracy,
            f"{self.task_type}.mean_reward": mean_reward,
            f"{self.task_type}.total_examples": total_count,
        }

        return rollout_groups, metrics

    def training_data(self) -> Iterator[MockEnvExample]:
        """Stream training data."""
        for example in self.train_examples:
            yield MockEnvExample(
                raw_prompt=example["prompt"],
                raw_answer=example["answer"],
                processed_prompt=example["prompt"],  # MockEnv doesn't transform
                processed_answer=example["answer"],
                metadata={"task_type": self.task_type},
            )

    def eval_data(self) -> Iterator[MockEnvExample]:
        """Stream evaluation data."""
        for example in self.eval_examples:
            yield MockEnvExample(
                raw_prompt=example["prompt"],
                raw_answer=example["answer"],
                processed_prompt=example["prompt"],  # MockEnv doesn't transform
                processed_answer=example["answer"],
                metadata={"task_type": self.task_type},
            )

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples."""
        return self.eval_examples[:n_examples]
