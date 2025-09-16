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
            prompt = f"Count: {word_list}"
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
        if not actual_response:
            return 0.0
        # Exact match or if response contains the number
        if correct_answer.lower() == actual_response.lower() or correct_answer in actual_response:
            return 1.0
        return 0.0


class LengthTask:
    """Classify sentence as short (≤3 words) or long (>3 words)."""

    SHORT_SENTENCES: ClassVar[list[str]] = [
        "Hi there",
        "Good morning",
        "Thank you",
        "I agree",
        "Very nice",
        "See you",
        "Take care",
        "No problem",
        "Of course",
        "Sounds good",
    ]

    LONG_SENTENCES: ClassVar[list[str]] = [
        "This is a longer sentence here",
        "I really enjoy reading books every day",
        "The weather today is quite nice outside",
        "Can you please help me with this",
        "I think we should go to the store",
        "Let me know when you are ready",
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            if rng.random() < 0.5:
                sentence = rng.choice(self.SHORT_SENTENCES)
                answer = "short"
            else:
                sentence = rng.choice(self.LONG_SENTENCES)
                answer = "long"
            examples.append({"prompt": f"Length: {sentence}", "answer": answer})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator | None = None) -> list[dict[str, str]]:
        examples = []
        if rng is None:
            rng = np.random.default_rng()
        eval_short = ["Hello", "Good luck", "Thank you very much"]
        eval_long = ["This is a test sentence for evaluation", "Please let me know what you think"]

        for _ in range(n_examples):
            if rng.random() < 0.5:
                sentence = rng.choice(eval_short)
                answer = "short"
            else:
                sentence = rng.choice(eval_long)
                answer = "long"
            examples.append({"prompt": f"Length: {sentence}", "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


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

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator | None = None) -> list[dict[str, str]]:
        examples = []
        if rng is None:
            rng = np.random.default_rng()
        eval_patterns = [("A B A B A", "B"), ("1 2 1 2 1", "2"), ("X Y X Y X", "Y")]
        for _ in range(n_examples):
            pattern, answer = eval_patterns[rng.integers(len(eval_patterns))]
            examples.append({"prompt": f"Pattern: {pattern}", "answer": answer})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


class ReverseTask:
    """Reverse 3-letter words."""

    WORDS: ClassVar[list[str]] = [
        "cat",
        "dog",
        "bat",
        "rat",
        "hat",
        "mat",
        "pen",
        "bed",
        "red",
        "run",
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            word = rng.choice(self.WORDS)
            reversed_word = word[::-1]
            examples.append({"prompt": f"Reverse: {word}", "answer": reversed_word})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator | None = None) -> list[dict[str, str]]:
        examples = []
        if rng is None:
            rng = np.random.default_rng()
        eval_words = ["car", "bag", "cup"]
        for _ in range(n_examples):
            word = rng.choice(eval_words)
            reversed_word = word[::-1]
            examples.append({"prompt": f"Reverse: {word}", "answer": reversed_word})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


class FirstTask:
    """Output the first word from a list."""

    WORD_LISTS: ClassVar[list[list[str]]] = [
        ["apple", "banana"],
        ["cat", "dog", "bird"],
        ["red", "blue"],
        ["big", "small", "tiny"],
        ["run", "jump"],
        ["book", "pen", "paper"],
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            idx = rng.choice(len(self.WORD_LISTS))
            words = self.WORD_LISTS[idx]
            word_string = " ".join(words)
            first_word = words[0]
            examples.append({"prompt": f"First: {word_string}", "answer": first_word})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator | None = None) -> list[dict[str, str]]:
        examples = []
        if rng is None:
            rng = np.random.default_rng()
        eval_lists = [["house", "tree"], ["happy", "sad", "angry"]]
        for _ in range(n_examples):
            words = eval_lists[rng.integers(len(eval_lists))]
            word_string = " ".join(words)
            first_word = words[0]
            examples.append({"prompt": f"First: {word_string}", "answer": first_word})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


class LastTask:
    """Output the last word from a list."""

    WORD_LISTS: ClassVar[list[list[str]]] = [
        ["apple", "banana"],
        ["cat", "dog", "bird"],
        ["red", "blue"],
        ["big", "small", "tiny"],
        ["run", "jump"],
        ["book", "pen", "paper"],
    ]

    def generate_training_examples(self, n_examples: int, rng: np.random.Generator) -> list[dict[str, str]]:
        examples = []
        for _ in range(n_examples):
            idx = rng.choice(len(self.WORD_LISTS))
            words = self.WORD_LISTS[idx]
            word_string = " ".join(words)
            last_word = words[-1]
            examples.append({"prompt": f"Last: {word_string}", "answer": last_word})
        return examples

    def generate_eval_examples(self, n_examples: int, rng: np.random.Generator | None = None) -> list[dict[str, str]]:
        examples = []
        if rng is None:
            rng = np.random.default_rng()
        eval_lists = [["house", "tree"], ["happy", "sad", "angry"]]
        for _ in range(n_examples):
            words = eval_lists[rng.integers(len(eval_lists))]
            word_string = " ".join(words)
            last_word = words[-1]
            examples.append({"prompt": f"Last: {word_string}", "answer": last_word})
        return examples

    def compute_reward(self, correct_answer: str, actual_response: str) -> float:
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


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
        if not actual_response:
            return 0.0
        return 1.0 if correct_answer.lower() == actual_response.lower() else 0.0


# Task mappings
TASKS = {
    "count": CountTask(),
    "length": LengthTask(),
    "pattern": PatternTask(),
    "reverse": ReverseTask(),
    "first": FirstTask(),
    "last": LastTask(),
    "arithmetic": ArithmeticTask(),
}


class MockEnv(MarinEnv):
    """Mock environment that generates synthetic data for testing."""

    def __init__(self, tokenizer, task_type="count", seed=None, **kwargs):
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
        # Convert JAX prng_key to numpy seed for consistency
        seed = int(jax.random.randint(prng_key, (), 0, 2**31))
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
