from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .environments.marin_env import EnvStep


class RLDataset:
    """Dataset class for RL training that stores processed training data and provides iteration."""

    def __init__(
        self,
        data_items: dict[str, np.ndarray],
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_output_length: int,
        pad_token_id: int,
    ):
        """Initialize RLDataset with processed data items.

        Args:
            data_items: Dictionary containing processed training data with keys:
                - returns: Advantage values for each token position
                - policy_logprobs: Log probabilities from the policy model
                - reference_logprobs: Log probabilities from the reference model
                - prompt_tokens: Tokenized prompts
                - prompt_masks: Attention masks for prompts
                - output_tokens: Tokenized outputs
                - output_masks: Attention masks for outputs
            tokenizer: Tokenizer used for decoding
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            pad_token_id: ID of the padding token
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pad_token_id = pad_token_id

        # Validate data consistency
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that all data items have consistent shapes."""
        expected_keys = {
            "returns",
            "policy_logprobs",
            "reference_logprobs",
            "prompt_tokens",
            "prompt_masks",
            "output_tokens",
            "output_masks",
        }

        if not expected_keys.issubset(set(self.data_items.keys())):
            missing_keys = expected_keys - set(self.data_items.keys())
            raise ValueError(f"Missing required keys in data_items: {missing_keys}")

        # Check that all arrays have the same batch dimension
        batch_sizes = {key: arr.shape[0] for key, arr in self.data_items.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(f"Inconsistent batch sizes across data items: {batch_sizes}")

    @property
    def size(self) -> int:
        """Return the number of examples in the dataset."""
        return self.data_items["returns"].shape[0]

    @classmethod
    def from_env_step(
        cls,
        env_step: EnvStep,
        reference_params: Any,
        get_logprobs_fn: Callable,
        reference_logprobs_bsize: int,
        max_input_length: int,
        max_output_length: int,
        pad_token_id: int,
        tokenizer: AutoTokenizer,
        kl_coef: float = 0.0,
    ) -> "RLDataset":
        """Create RLDataset from environment step.

        Args:
            env_step: Environment step containing examples, samples, and rewards
            reference_params: Parameters for the reference model
            get_logprobs_fn: Function to compute log probabilities
            reference_logprobs_bsize: Batch size for reference logprob computation
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            pad_token_id: ID of the padding token
            tokenizer: Tokenizer for processing text
            kl_coef: KL divergence coefficient (for future use)

        Returns:
            RLDataset instance with processed training data
        """
        examples = env_step.examples
        samples = env_step.samples
        rewards = env_step.rewards

        # Prepare data to compute reference logprobs
        batch_items = []
        for i, example in enumerate(examples):
            prompt_tokens = tokenizer.encode(example["prompt"], add_special_tokens=True)[-max_input_length:]
            prompt_attention_mask = [0] * (max_input_length - len(prompt_tokens)) + [1] * len(prompt_tokens)
            prompt_tokens = [pad_token_id] * (max_input_length - len(prompt_tokens)) + prompt_tokens

            for sample in samples[i]:
                answer_tokens = sample["tokens"][:max_output_length]
                answer_attention_mask = [1] * len(answer_tokens) + [0] * (max_output_length - len(answer_tokens))
                answer_tokens = answer_tokens + [pad_token_id] * (max_output_length - len(answer_tokens))
                answer_logprobs = sample["logprobs"][:max_output_length]
                answer_logprobs = answer_logprobs + [0] * (max_output_length - len(answer_logprobs))

                batch_items.append(
                    {
                        "prompt_tokens": np.asarray(prompt_tokens)[None],
                        "prompt_attention_mask": np.asarray(prompt_attention_mask)[None],
                        "answer_tokens": np.asarray(answer_tokens)[None],
                        "answer_attention_mask": np.asarray(answer_attention_mask)[None],
                        "answer_logprobs": np.asarray(answer_logprobs)[None],
                    }
                )

        true_batch_items_len = len(batch_items)

        # Pad batch_items to be divisible by reference_logprobs_bsize
        if true_batch_items_len % reference_logprobs_bsize != 0:
            padding_needed = reference_logprobs_bsize - (true_batch_items_len % reference_logprobs_bsize)
            for _ in range(padding_needed):
                batch_items.append(
                    {
                        "prompt_tokens": np.full((1, max_input_length), pad_token_id, dtype=np.int32),
                        "prompt_attention_mask": np.zeros((1, max_input_length), dtype=np.int32),
                        "answer_tokens": np.full((1, max_output_length), pad_token_id, dtype=np.int32),
                        "answer_attention_mask": np.zeros((1, max_output_length), dtype=np.int32),
                        "answer_logprobs": np.zeros((1, max_output_length), dtype=np.float32),
                    }
                )

        # Compute reference logprobs in batches
        all_reference_logprobs, all_logprobs = [], []
        prompt_tokens_list, prompt_masks_list = [], []
        output_tokens_list, output_masks_list = [], []

        for i in tqdm(range(0, len(batch_items), reference_logprobs_bsize)):
            curr_batch = batch_items[i : (i + reference_logprobs_bsize)]
            curr_batch = {k: np.concatenate([item[k] for item in curr_batch], axis=0) for k in curr_batch[0].keys()}

            reference_logprobs = np.asarray(
                get_logprobs_fn(
                    reference_params,
                    curr_batch["prompt_tokens"],
                    curr_batch["prompt_attention_mask"],
                    curr_batch["answer_tokens"],
                    curr_batch["answer_attention_mask"],
                )
            )

            # Determine the actual batch size for this iteration
            if (i // reference_logprobs_bsize) == (len(batch_items) // reference_logprobs_bsize) - 1:
                true_batch_size = true_batch_items_len % reference_logprobs_bsize
                if true_batch_size == 0:
                    true_batch_size = reference_logprobs.shape[0]
            else:
                true_batch_size = reference_logprobs.shape[0]

            # Only keep the non-padded examples
            for x in range(true_batch_size):
                all_reference_logprobs.append(reference_logprobs[x])
                all_logprobs.append(curr_batch["answer_logprobs"][x])
                output_masks_list.append(curr_batch["answer_attention_mask"][x])
                output_tokens_list.append(curr_batch["answer_tokens"][x])
                prompt_tokens_list.append(curr_batch["prompt_tokens"][x])
                prompt_masks_list.append(curr_batch["prompt_attention_mask"][x])

        # Stack all arrays
        all_reference_logprobs = np.stack(all_reference_logprobs, axis=0)
        all_logprobs = np.stack(all_logprobs, axis=0)
        output_masks = np.stack(output_masks_list, axis=0)
        output_tokens = np.stack(output_tokens_list, axis=0)
        prompt_tokens = np.stack(prompt_tokens_list, axis=0)
        prompt_masks = np.stack(prompt_masks_list, axis=0)

        # Compute RLOO advantages
        all_rloo_advantages = []
        for rewards_group in rewards:
            advantages = compute_rloo_advantages_for_group(rewards_group)
            all_rloo_advantages.append(advantages)
        all_rloo_advantages = np.concatenate(all_rloo_advantages, axis=0)

        # Compute returns (repeat advantages for each token position)
        all_returns = jnp.repeat(all_rloo_advantages[..., None], output_masks.shape[1], axis=1)

        data_items = {
            "returns": all_returns,
            "policy_logprobs": all_logprobs,
            "reference_logprobs": all_reference_logprobs,
            "prompt_tokens": prompt_tokens,
            "prompt_masks": prompt_masks,
            "output_tokens": output_tokens,
            "output_masks": output_masks,
        }

        return cls(
            data_items=data_items,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            pad_token_id=pad_token_id,
        )

    def iterate_batches(
        self, batch_size: int, shuffle: bool = True, loop: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        """Create an iterator over training batches.

        Args:
            batch_size: Size of each training batch
            shuffle: Whether to shuffle the data
            loop: Whether to loop infinitely over the data

        Yields:
            Dictionary containing a batch of training data ready for RLOO training
        """
        N = self.size
        rng = jax.random.PRNGKey(0)

        while True:
            with jax.default_device(jax.devices("cpu")[0]):
                # Generate indices
                idxs = []
                for _ in range((batch_size + (N - 1)) // N):
                    if shuffle:
                        rng, subrng = jax.random.split(rng)
                        curr_idxs = jax.random.permutation(subrng, np.arange(N))
                        idxs.extend(curr_idxs.tolist())
                    else:
                        curr_idxs = np.arange(N)
                        idxs.extend(curr_idxs.tolist())
                idxs = np.asarray(idxs)

                # Yield batches
                for batch_idx in range(len(idxs) // batch_size):
                    batch_idxs = idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                    batch_examples = {
                        k: np.asarray([self.data_items[k][idx] for idx in batch_idxs]) for k in self.data_items.keys()
                    }
                    batch = self._prepare_rloo_examples(batch_examples)
                    yield batch

                if not loop:
                    break

    def _prepare_rloo_examples(self, examples: dict[str, Any]) -> dict[str, np.ndarray]:
        """Prepare examples for RLOO training.

        Args:
            examples: Dictionary containing batched examples with keys matching self.data_items

        Returns:
            Dictionary containing processed batch ready for training
        """
        # Concatenate prompt and output tokens
        full_tokens = np.concatenate((examples["prompt_tokens"], examples["output_tokens"]), axis=1)
        full_attention_mask = np.concatenate((examples["prompt_masks"], examples["output_masks"]), axis=1)

        # Create position IDs
        full_position_ids = np.maximum(np.cumsum(full_attention_mask, axis=1) - 1, 0)

        # Prepare input and target sequences for language modeling
        input_tokens = full_tokens[:, :-1]
        input_attention_mask = full_attention_mask[:, :-1]
        target_tokens = full_tokens[:, 1:]
        position_ids = full_position_ids[:, :-1]

        # Create loss masks (only compute loss on output tokens, not prompt tokens)
        loss_masks = np.concatenate(
            [
                np.zeros(
                    (examples["prompt_masks"].shape[0], examples["prompt_masks"].shape[1] - 1),
                    dtype=np.float32,
                ),
                examples["output_masks"].astype(np.float32),
            ],
            axis=1,
        )

        # Create loss weights (advantages/returns for policy gradient)
        loss_weights = np.concatenate(
            [
                np.zeros(
                    (examples["prompt_masks"].shape[0], examples["prompt_masks"].shape[1] - 1),
                    dtype=np.float32,
                ),
                examples["returns"].astype(np.float32),
            ],
            axis=1,
        )

        # Create reference logprobs for KL penalty
        reference_logprobs = np.concatenate(
            [
                np.zeros(
                    (examples["prompt_masks"].shape[0], examples["prompt_masks"].shape[1] - 1),
                    dtype=np.float32,
                ),
                examples["reference_logprobs"].astype(np.float32),
            ],
            axis=1,
        )

        return {
            "input_ids": input_tokens,
            "attention_mask": input_attention_mask,
            "position_ids": position_ids,
            "target_ids": target_tokens,
            "loss_masks": loss_masks,
            "loss_weights": loss_weights,
            "reference_logprobs": reference_logprobs,
        }


def compute_rloo_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """Compute RLOO advantages for a group of rewards.

    Args:
        rewards: Array of rewards for a group

    Returns:
        Normalized advantages
    """
    advantages = (rewards - rewards.mean()) / np.clip(rewards.std(), 1e-8, None)
    return advantages


def create_dataset_from_environment(
    environment,
    sampler,
    params,
    reference_params,
    get_logprobs_fn,
    n_examples: int,
    prng_key,
    reference_logprobs_bsize: int,
    max_input_length: int,
    max_output_length: int,
    pad_token_id: int,
    tokenizer: AutoTokenizer,
    generation_config: dict[str, Any],
    mode: str = "train",
) -> tuple["RLDataset", dict[str, float]]:
    """Create RLDataset by stepping through the environment.

    Args:
        environment: Environment to step through
        sampler: Inference sampler
        params: Current model parameters
        reference_params: Reference model parameters
        get_logprobs_fn: Function to compute log probabilities
        n_examples: Number of examples to sample
        prng_key: Random key for sampling
        reference_logprobs_bsize: Batch size for reference logprob computation
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        pad_token_id: ID of the padding token
        tokenizer: Tokenizer for processing text
        generation_config: Configuration for generation
        mode: Mode for environment stepping ("train" or "eval")

    Returns:
        Tuple of (RLDataset, metrics from environment)
    """
    # Get environment step
    env_step = environment.step(
        sampler=sampler,
        params=params,
        n_examples=n_examples,
        prng_key=prng_key,
        mode=mode,
        n_generations=generation_config["n_generations"],
    )

    # Create dataset from environment step
    dataset = RLDataset.from_env_step(
        env_step=env_step,
        reference_params=reference_params,
        get_logprobs_fn=get_logprobs_fn,
        reference_logprobs_bsize=reference_logprobs_bsize,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
    )

    return dataset, env_step.metrics
