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

from pathlib import Path
from typing import Any, ClassVar, Protocol

import jax
import numpy as np

from marin.post_training.inference import batch_inference

from .marin_env import EnvStep, MarinEnv

# Constants
NUM_TRAIN_EXAMPLES = 1000
NUM_EVAL_EXAMPLES = 100


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


class CountTask:
    """Count the number of words in the input."""

    WORDS: ClassVar[list[str]] = [
        "apple",
        "banana",
        "cat",
        "dog",
        "bird",
        "fish",
        "tree",
        "car",
        "book",
        "pen",
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            count = rng.integers(1, 6)
            selected_words = rng.choice(self.WORDS, count, replace=False).tolist()

            word_list = " ".join(selected_words)
            prompt = f"How many words are in '{word_list}'? One word answer only."
            answer = str(count)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            count = rng.integers(1, 4)  # 1-3 words for eval
            selected_words = rng.choice(self.WORDS, count, replace=False).tolist()

            word_list = " ".join(selected_words)
            prompt = f"Count: {word_list}"
            answer = str(count)
            examples.append({"prompt": prompt, "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class PatternTask:
    """Complete simple patterns."""

    PATTERNS: ClassVar[list[tuple[str, str]]] = [
        ("A B A B A", "B"),
        ("1 2 1 2 1", "2"),
        ("X Y X Y X", "Y"),
        ("red blue red blue red", "blue"),
        ("up down up down up", "down"),
        ("cat dog cat dog cat", "dog"),
        ("sun moon sun moon sun", "moon"),
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            idx = rng.choice(len(self.PATTERNS))
            pattern, answer = self.PATTERNS[idx]
            examples.append({"prompt": f"Pattern: {pattern}", "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        eval_patterns = [("A B A B A", "B"), ("1 2 1 2 1", "2"), ("X Y X Y X", "Y")]
        for _ in range(n_examples):
            pattern, answer = eval_patterns[rng.integers(len(eval_patterns))]
            examples.append({"prompt": f"Pattern: {pattern}", "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response, strict_format=True)


class ReverseTask:
    """Reverse the letters in words."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        word_list = Path("/usr/share/dict/words").read_text().splitlines()
        for _ in range(n_examples):
            word = rng.choice(word_list)
            reversed_word = word[::-1]
            examples.append(
                {
                    "prompt": f"Reverse the letters in '{word}'. One word answer only: ",
                    "answer": reversed_word,
                }
            )
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class FirstWordTask:
    """Output the first word from a list."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        word_list = Path("/usr/share/dict/words").read_text().splitlines()
        examples = []
        for _ in range(n_examples):
            words = [rng.choice(word_list) for _ in range(rng.integers(2, 6))]
            sentence = " ".join(words)
            examples.append(
                {
                    "prompt": f"Output the first word of this sentence. '{sentence}'. One word answer only: ",
                    "answer": words[0],
                }
            )
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        return self.generate_training_examples(n_examples, rng)

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


class ArithmeticTask:
    """Simple single-digit arithmetic."""

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(1, 10)
            b = rng.integers(1, 10)
            result = a + b
            examples.append({"prompt": f"Add: {a} + {b}", "answer": str(result)})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            a = rng.integers(1, 10)
            b = rng.integers(1, 10)
            result = a + b
            examples.append({"prompt": f"Add: {a} + {b}", "answer": str(result)})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        return compute_soft_reward(correct_answer, actual_response)


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


# Task mappings
TASKS = {
    "count": CountTask(),
    "pattern": PatternTask(),
    "reverse": ReverseTask(),
    "first": FirstWordTask(),
    "arithmetic": ArithmeticTask(),
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
        sampler,
        params,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
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
        responses = batch_inference(sampler, params, prompts, prng_key, n_generations, verbose=False)

        # Compute rewards and metrics
        rewards, metrics = self._compute_rewards(examples, responses)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples."""
        return self.eval_examples[:n_examples]

    def _compute_rewards(self, examples: list[dict], responses: list[list[dict[str, Any]]]) -> tuple[np.ndarray, dict]:
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
                decoded_response = self.tokenizer.decode(response["tokens"], skip_special_tokens=True)

                # Extract response after the prompt
                prompt = example["prompt"]
                if prompt in decoded_response:
                    actual_response = decoded_response.split(prompt, 1)[1].strip()
                else:
                    actual_response = decoded_response.strip()

                # Clean up the response - take only the first word/token
                actual_response = actual_response.split()[0] if actual_response.split() else ""

                # Task-specific reward computation
                reward = self._compute_task_reward(correct_answer, actual_response)
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
            f"{self.task_type}_accuracy": accuracy,
            f"{self.task_type}_format_accuracy": format_accuracy,
            f"{self.task_type}_mean_reward": mean_reward,
            f"{self.task_type}_total_examples": total_count,
        }

        return rewards, metrics

    def _compute_task_reward(self, correct_answer: str, actual_response: str) -> float:
        """Compute task-specific reward."""
        return self.task.compute_reward(correct_answer, actual_response)
