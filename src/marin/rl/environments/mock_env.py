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
import numpy as np
from transformers import PreTrainedTokenizer

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import RolloutGroup

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

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
        """Generate examples."""
        ...

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer=None) -> float:
        """Compute reward for a response."""
        ...


class AdditionTask:
    """Addition task with configurable difficulty."""

    DIFFICULTY_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "easy": (0, 10),  # 1 digit
        "medium": (0, 100),  # 2 digits
        "hard": (0, 10000),  # 4-5 digits
    }

    def __init__(self, difficulty: str = "medium"):
        if difficulty not in self.DIFFICULTY_RANGES:
            raise ValueError(f"Unknown difficulty: {difficulty}. Must be one of {list(self.DIFFICULTY_RANGES.keys())}")
        self.difficulty = difficulty
        self.min_val, self.max_val = self.DIFFICULTY_RANGES[difficulty]

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(self.min_val, self.max_val)
            b = rng.integers(self.min_val, self.max_val)
            result = a + b
            prompt = f"What is {a}+{b}? Output just the number:"
            answer = str(result)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer=None) -> float:
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

    def __init__(self, difficulty: str = "medium"):
        # Difficulty not used for this task
        pass

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            word, opposite = self.OPPOSITES[rng.integers(len(self.OPPOSITES))]
            prompt = f"Opposite of {word}? One word:"
            answer = opposite
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer=None) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class NumberComparisonTask:
    """Compare two numbers - 50% success rate target."""

    def __init__(self, difficulty: str = "medium"):
        # Difficulty not used for this task
        pass

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
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

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer=None) -> float:
        return compute_soft_reward(correct_answer, actual_response)


def compute_soft_reward(correct_answer: str, actual_response: str) -> float:
    """Compute soft reward with partial credit for correctness and format."""
    if not actual_response:
        return 0

    # remove commas from numbers
    correct_answer = correct_answer.replace(",", "").lower()

    tokens = actual_response.split()
    correct_score = 0
    for token in tokens:
        token = token.replace(",", "").lower()

        if token == correct_answer:
            correct_score = 1
            break

    return correct_score / len(tokens)


class MoarCatsTask:
    """Make moar cats."""

    def __init__(self, difficulty: str = "medium"):
        # Difficulty not used for this task
        pass

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            prompt = "i like cats, i love cats, give me moar cats."
            num_cats = int(rng.integers(1, 5))
            answer = "cats" * num_cats + " love cats" * int(num_cats > 1)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer=None) -> float:
        # how many cats
        num_cats = actual_response.lower().count("cat")
        love_cats = actual_response.lower().count("love cats")

        return (num_cats + (10 * love_cats)) / np.sqrt(1 + len(actual_response))


class SequentialDigitsTask:
    """Train model to produce digits in sequential order.

    Rewards responses that contain increasing digit sequences.
    Examples:
      - "12345" = high reward
      - "0123456789" = very high reward
      - "5231" = low/negative reward
      - "catcat" = very negative reward
    """

    def __init__(self, difficulty: str = "medium"):
        pass

    def generate_examples(self, n_examples: int, rng: np.random.Generator, tokenizer=None) -> list[dict[str, str]]:
        """Generate examples that ask for sequential digits."""
        examples = []

        for _ in range(n_examples):
            start_idx = rng.integers(0, 5)
            end_idx = start_idx + rng.integers(start_idx, 9)
            prompt = f"{start_idx} to {end_idx}:"
            answer = "".join(str(i) for i in range(start_idx, end_idx))
            examples.append({"prompt": prompt, "answer": answer})

        return examples

    def compute_reward(self, correct_answer: str, actual_response: str, tokenizer: PreTrainedTokenizer) -> float:
        """Compute reward based on sequential digit quality.

        Reward structure:
          - Heavily reward increasing sequential digits
          - Penalize non-digits
          - Penalize decreasing or out-of-order digits
          - Bonus for longer sequences
        """
        if not actual_response:
            return 0

        actual_tokens = tokenizer.encode(actual_response, add_special_tokens=False)

        # score is a function of how many digits & how sequential they are
        digit_count = 0
        order_count = 0
        last_digit = -1
        for token in actual_tokens:
            token_str = tokenizer.decode([token])
            if token_str.isdigit():
                digit = int(token_str)
                digit_count += 1
                if digit > last_digit:
                    order_count += 1
                last_digit = digit

        return digit_count / len(actual_tokens) + order_count / len(actual_tokens)


# Task mappings
TASKS = {
    "cats": MoarCatsTask,
    "addition": AdditionTask,
    "opposites": OppositesTask,
    "number_comparison": NumberComparisonTask,
    "sequential_digits": SequentialDigitsTask,
}


class MockEnv(MarinEnv):
    """Mock environment that generates synthetic data for testing."""

    def __init__(self, task_type: str, seed=None, difficulty: str = "medium", **kwargs):
        self.task_type = task_type

        # Get the task class for this task type
        task_class = TASKS.get(task_type)
        if not task_class:
            raise ValueError(f"Unknown task type: {task_type}")

        # Instantiate the task with difficulty
        self.task = task_class(difficulty=difficulty)

        # Generate examples using the task
        rng = np.random.default_rng(seed)
        self.train_examples = self.task.generate_examples(NUM_TRAIN_EXAMPLES, rng)
        self.eval_examples = self.task.generate_examples(NUM_EVAL_EXAMPLES, rng)

        difficulty_str = f" (difficulty={difficulty})" if task_type == "addition" else ""
        print(
            f"Initialized MockEnv with task '{task_type}'{difficulty_str}: "
            f"{len(self.train_examples)} train examples and {len(self.eval_examples)} eval examples."
        )

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
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

        sampled_examples = {}
        for idx in indices:
            example = available_examples[int(idx)]
            sampled_examples[example["prompt"]] = example["answer"]

        prompts = list(sampled_examples.keys())
        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
        )

        # Evaluate and create rollouts
        rollout_groups = []

        for prompt, completion in zip(prompts, completions, strict=True):
            group = []
            for choice in completion.choices:
                true_answer = sampled_examples[prompt]
                reward = self.task.compute_reward(true_answer, choice.message.content, tokenizer=inference_ctx.tokenizer)
                rollout = inference_ctx.create_rollout_from_choice(
                    prompt, choice, env_name=f"mock_env:{self.task_type}", env_example_id=hash(prompt), reward=reward
                )
                group.append(rollout)
            rollout_groups.append(RolloutGroup(rollouts=group))

        return rollout_groups, {}

    def training_data(self) -> Iterator[MockEnvExample]:
        """Stream training data."""
        for example in self.train_examples:
            yield MockEnvExample(
                raw_prompt=example["prompt"],
                raw_answer=example["answer"],
                processed_prompt=example["prompt"],
                processed_answer=example["answer"],
                metadata={"task_type": self.task_type},
            )

    def eval_data(self) -> Iterator[MockEnvExample]:
        """Stream evaluation data."""
        for example in self.eval_examples:
            yield MockEnvExample(
                raw_prompt=example["prompt"],
                raw_answer=example["answer"],
                processed_prompt=example["prompt"],
                processed_answer=example["answer"],
                metadata={"task_type": self.task_type},
            )

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples."""
        return self.eval_examples[:n_examples]
