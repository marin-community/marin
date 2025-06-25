import random
from typing import Any

import datasets
import numpy as np
from inference import batch_inference
from tqdm.auto import tqdm
from utils import validate_format

from .marin_env import EnvStep, MarinEnv
from .math_utils import grade_answer, last_boxed_only_string, remove_boxed


class MathEnv(MarinEnv):
    def __init__(self, tokenizer, max_input_length: int, pad_token_id: int, **kwargs):
        """Initialize math environment with pre-tokenized prompts."""
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.pad_token_id = pad_token_id

        # Initialize datasets
        data_source = "DigitalLearningGmbH/MATH-lighteval"
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Convert to the format expected by the training code and pre-tokenize
        instruction = (
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> "
            "tags. Assistant: Let me solve this step by step. <think>"
        )

        print("Pre-tokenizing training examples...")
        self.train_examples = []
        for item in tqdm(train_dataset, desc="Processing train set"):
            prompt = f"{item['problem']} {instruction}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))

            # Pre-tokenize the prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)[-self.max_input_length :]
            prompt_attention_mask = [0] * (self.max_input_length - len(prompt_tokens)) + [1] * len(prompt_tokens)
            prompt_tokens = [self.pad_token_id] * (self.max_input_length - len(prompt_tokens)) + prompt_tokens

            self.train_examples.append(
                {
                    "prompt": prompt,  # Keep original for inference
                    "answer": answer,
                    "prompt_tokens": np.asarray(prompt_tokens),
                    "prompt_attention_mask": np.asarray(prompt_attention_mask),
                }
            )

        print("Pre-tokenizing evaluation examples...")
        self.eval_examples = []
        for item in tqdm(test_dataset, desc="Processing test set"):
            prompt = f"{item['problem']} {instruction}"
            answer = remove_boxed(last_boxed_only_string(item["solution"]))

            # Pre-tokenize the prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)[-self.max_input_length :]
            prompt_attention_mask = [0] * (self.max_input_length - len(prompt_tokens)) + [1] * len(prompt_tokens)
            prompt_tokens = [self.pad_token_id] * (self.max_input_length - len(prompt_tokens)) + prompt_tokens

            self.eval_examples.append(
                {
                    "prompt": prompt,  # Keep original for inference
                    "answer": answer,
                    "prompt_tokens": np.asarray(prompt_tokens),
                    "prompt_attention_mask": np.asarray(prompt_attention_mask),
                }
            )

        print(
            f"Initialized MathEnv with {len(self.train_examples)} train examples "
            "and {len(self.eval_examples)} eval examples"
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
        # Sample examples from dataset
        if mode == "train":
            examples = random.sample(self.train_examples, min(n_examples, len(self.train_examples)))
        else:
            examples = random.sample(self.eval_examples, min(n_examples, len(self.eval_examples)))

        # Generate responses using the model (still need original prompts for
        # inference)
        samples = batch_inference(
            sampler,
            params,
            [example["prompt"] for example in examples],
            prng_key,
            n_generations=n_generations,
            verbose=True,
        )

        # Compute rewards
        rewards, metrics = self._compute_rewards(examples, samples)

        return EnvStep(examples=examples, samples=samples, rewards=rewards, metrics=metrics)

    def _compute_rewards(
        self, examples: list[dict[str, Any]], samples: list[list[dict[str, np.ndarray]]]
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compute rewards for generated samples."""
        all_rewards = []
        all_format_rewards = []
        all_correct_rewards = []
        all_lens = []

        for i, sample in tqdm(enumerate(samples)):
            group_rewards = []
            group_format_rewards = []
            group_correct_rewards = []
            for inner_sample in sample:
                all_lens.append(len(inner_sample["tokens"]))
                decoded_sample = self.tokenizer.decode(inner_sample["tokens"], skip_special_tokens=True)
                validation = validate_format(decoded_sample + ">")
                if validation["is_valid"]:
                    grade = grade_answer(validation["answer"], examples[i]["answer"])
                else:
                    grade = False

                if random.random() < 1 / 64:
                    print("=" * 25)
                    print(examples[i]["prompt"])
                    print(decoded_sample + ">")
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
            "train_rs": np.mean(all_rewards),
            "train_format_rs": np.mean(all_format_rewards),
            "train_correct_rs": np.mean(all_correct_rewards),
            "train_output_len": np.mean(all_lens),
        }

        return all_rewards, metrics

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples for evaluation."""
        return random.sample(self.eval_examples, min(n_examples, len(self.eval_examples)))
