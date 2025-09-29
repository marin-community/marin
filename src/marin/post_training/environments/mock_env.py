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

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import jax
import numpy as np

from .marin_env import EnvResponse, EnvStep, InferenceContext, MarinEnv

NUM_TRAIN_EXAMPLES = 1000
NUM_EVAL_EXAMPLES = 100


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


class SimpleAdditionTask:
    """Simple single-digit addition - 90% success rate target."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(0, 25)
            b = rng.integers(0, 25)
            result = a + b
            prompt = f"What is {a}+{b}? Just the number:"
            answer = str(result)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class SimpleOppositesTask:
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


def extract_first_n_tokens(response: str, n: int = 10) -> list[str]:
    """Extract first N tokens from response."""
    tokens = response.split()
    return tokens[: min(n, len(tokens))]


def compute_soft_reward(correct_answer: str, actual_response: str, strict_format: bool = False) -> float:
    """Compute soft reward with partial credit for correctness and format."""
    if not actual_response:
        return 0.0

    tokens = actual_response.split()
    first_10_tokens = extract_first_n_tokens(actual_response, 10)

    correctness = 0.0
    if tokens and tokens[0].lower() == correct_answer.lower():
        correctness = 1.0  # Perfect match in first position
    elif tokens and tokens[0].lower() == correct_answer.lower().rstrip(".,!?"):
        correctness = 0.9  # Match with punctuation
    elif correct_answer.lower() in [t.lower() for t in first_10_tokens]:
        correctness = 0.6  # Answer appears early
    elif correct_answer.lower() in actual_response.lower():
        correctness = 0.3  # Answer appears somewhere

    format_score = 0.0
    if len(tokens) == 1:
        format_score = 1.0  # Perfect - single word
    elif len(tokens) <= 2 and strict_format:
        format_score = 0.5  # Stricter scoring for patterns
    elif len(tokens) <= 3:
        format_score = 0.7  # Good - very brief
    elif len(tokens) <= 10:
        format_score = 0.4 if strict_format else 0.3  # Okay - short sentence
    else:
        format_score = 0.1  # Poor - verbose

    # Weighted combination (70% correctness, 30% format)
    return 0.7 * correctness + 0.3 * format_score


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
    "simple_addition": SimpleAdditionTask(),
    "simple_opposites": SimpleOppositesTask(),
    "number_comparison": NumberComparisonTask(),
}


class MockEnv(MarinEnv):
    """Mock environment that generates synthetic data for testing."""

    def __init__(self, tokenizer, task_type: str, seed=None, **kwargs):
        self.tokenizer = tokenizer
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
        """Generate synthetic rollouts for testing."""
        if mode == "train":
            available_examples = self.train_examples
        else:
            available_examples = self.eval_examples

        # Use numpy random for sampling
        n_to_sample = min(n_examples, len(available_examples))
        seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=True)
        examples = [available_examples[int(idx)] for idx in indices]

        # Generate responses
        prompts = [example["prompt"] for example in examples]
        responses = inference_ctx.generate(
            prompts,
            temperature=temperature,
            n_generations=n_generations,
        )

        # Compute rewards and metrics
        rewards, metrics = self._compute_rewards(examples, responses, inference_ctx.tokenizer)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

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

    def _compute_rewards(
        self, examples: list[MockEnvExample], responses: list[list[EnvResponse]], tokenizer
    ) -> tuple[np.ndarray, dict]:
        """Compute rewards for generated responses."""
        n_examples = len(examples)
        n_generations = len(responses[0]) if responses else 1

        # Initialize rewards array
        rewards = np.zeros((n_examples, n_generations), dtype=np.float32)

        correct_count = 0
        total_count = 0
        format_correct_count = 0

        for example_idx, (example, response_list) in enumerate(zip(examples, responses, strict=False)):
            correct_answer = example["answer"]

            for gen_idx, response in enumerate(response_list):
                # Decode the generated tokens to text
                decoded_response = tokenizer.decode(response["tokens"], skip_special_tokens=True)

                # Extract response after the prompt
                prompt = example["prompt"]
                if decoded_response.startswith(prompt):
                    actual_response = decoded_response[len(prompt) :]
                else:
                    actual_response = decoded_response
                reward = self.task.compute_reward(correct_answer, actual_response)
                rewards[example_idx, gen_idx] = reward

                # Print sample outputs for debugging
                if example_idx == 0 and gen_idx == 0:
                    print("=" * 50)
                    print(f"Task: {self.task_type}")
                    print(f"Prompt: {prompt}")
                    print(f"Full decoded response: {decoded_response}")
                    print(f"Extracted response: '{actual_response}'")
                    print(f"Expected answer: '{correct_answer}'")
                    print(f"Reward: {reward}")
                    print("=" * 50)

                if reward > 0:
                    correct_count += 1
                if actual_response:  # Non-empty response counts as format correct
                    format_correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        format_accuracy = format_correct_count / total_count if total_count > 0 else 0.0
        mean_reward = float(np.mean(rewards))

        metrics = {
            f"{self.task_type}.accuracy": accuracy,
            f"{self.task_type}.format_accuracy": format_accuracy,
            f"{self.task_type}.mean_reward": mean_reward,
            f"{self.task_type}.total_examples": total_count,
        }

        return rewards, metrics
