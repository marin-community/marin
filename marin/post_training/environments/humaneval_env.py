import datasets
import jax
import numpy as np

from post_training.inference import batch_inference
from tqdm.auto import tqdm
from typing import Any
from .marin_env import EnvStep, MarinEnv


class HumanEvalEnv(MarinEnv):
    """
    Environment for HumanEval benchmark, which evaluates models on code generation tasks.
    https://github.com/openai/human-eval
    """

    def __init__(self, tokenizer, output_dir_path: str, **kwargs):
        self.tokenizer = tokenizer

        # Initialize datasets
        data_source = "openai/openai_humaneval"
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
        test_dataset = dataset["test"]

        # Convert to format expected by the training code and pre-tokenize
        self.eval_examples = []
        for item in tqdm(test_dataset, desc="Processing test set"):
            self.eval_examples.append(
                {
                    "prompt": item["task_id"],
                }
            )
        
        print(
            f"Initialized HumanEvalEnv with {len(self.eval_examples)} eval examples."
        )
    
    def step(
        self, sampler, params, n_examples: int, prng_key, mode: str = "eval", n_generations: int = 1, **kwargs
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            sampler: The inference sampler
            params: Model parameters
            n_examples: Number of examples to sample
            prng_key: Random key for sampling
            mode: "eval"
            n_generations: Number of generations per example
        """
        available_examples = self.eval_examples

        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(available_examples))
            indices = jax.random.choice(prng_key, len(available_examples), shape=(n_to_sample,), replace=False)
            examples = [available_examples[int(idx)] for idx in indices]
        
        responses = batch_inference(
            sampler,
            params,
            [example["prompt"] for example in examples],
            prng_key,
            n_generations=n_generations,
            verbose=True,
        )

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
        
        metrics = {
            "train_rewards": np.mean(all_rewards),
            "train_format_rewards": np.mean(all_format_rewards),
            "train_correct_rewards": np.mean(all_correct_rewards),
            "train_output_len": np.mean(all_lens),
        }

        all_rewards = np.asarray(all_rewards)
        all_format_rewards = np.asarray(all_format_rewards)
        all_correct_rewards = np.asarray(all_correct_rewards)
        all_lens = np.asarray(all_lens)
        
        return all_rewards, metrics
        

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples for evaluation."""
        # Use a fixed seed for reproducible evaluation
        eval_key = jax.random.PRNGKey(42)
        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(self.eval_examples))
            indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
            return [self.eval_examples[int(idx)] for idx in indices]

