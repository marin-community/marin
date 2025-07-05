import random
from typing import Any

import datasets
import jax
import numpy as np
from tqdm.auto import tqdm

from marin.post_training.inference import batch_inference
from marin.post_training.utils import validate_format
from .marin_env import EnvStep, MarinEnv
from .math_utils import grade_answer, last_boxed_only_string, remove_boxed


class MathEnv(MarinEnv):
    INSTRUCTION: str = (
        "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> "
        "tags. Assistant: Let me solve this step by step. <think>"
    )

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Initialize datasets
        data_source = "DigitalLearningGmbH/MATH-lighteval"
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Convert to the format expected by the training code and pre-tokenize
        self.train_examples = []
        for item in tqdm(train_dataset, desc="Processing train set"):
            prompt = f"{item['problem']} {self.INSTRUCTION}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))

            self.train_examples.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                }
            )

        self.eval_examples = []
        for item in tqdm(test_dataset, desc="Processing test set"):
            prompt = f"{item['problem']} {self.INSTRUCTION}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))

            self.eval_examples.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                }
            )

        print(
            f"Initialized MathEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

    def step(
        self, sampler, params, n_examples: int, prng_key, mode: str = "train", n_generations: int = 1, **kwargs
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            sampler: The inference sampler
            params: Model parameters
            n_examples: Number of examples to sample
            prng_key: Random key for sampling
            mode: "train" or "eval"
            n_generations: Number of generations per example
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
        responses = batch_inference(
            sampler,
            params,
            [example["prompt"] for example in examples],
            prng_key,
            n_generations=n_generations,
            verbose=True,
        )

        # Compute rewards
        rewards, metrics = self._compute_rewards(examples, responses)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

    def _compute_rewards(
        self, examples: list[dict[str, Any]], responses: list[list[dict[str, np.ndarray]]]
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
            for inner_response in response:
                all_lens.append(len(inner_response["tokens"]))
                decoded_response = self.tokenizer.decode(inner_response["tokens"], skip_special_tokens=True)
                validation = validate_format(decoded_response + ">")
                if validation["is_valid"]:
                    grade = grade_answer(validation["answer"], examples[i]["answer"])
                else:
                    grade = False

                if random.random() < 1 / 64:
                    print("=" * 25)
                    print(examples[i]["prompt"])
                    print(decoded_response + ">")
                    print("=" * 25)
                    print("gt answer: ", examples[i]["answer"])
                    print("extracted answer: ", validation["answer"])
                    print("grade: ", grade)
                    print("=" * 25)

                score = float(grade)
                group_rewards.append(float(score))
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
            "train_rewards": np.mean(all_rewards),
            "train_format_rewards": np.mean(all_format_rewards),
            "train_correct_rewards": np.mean(all_correct_rewards),
            "train_output_len": np.mean(all_lens),
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
