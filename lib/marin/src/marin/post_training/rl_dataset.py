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

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
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
        reference_ctx,
        max_input_length: int,
        max_output_length: int,
        pad_token_id: int,
        kl_coef: float = 0.0,
    ) -> "RLDataset":
        """Create RLDataset from environment step.

        Args:
            env_step: Environment step containing examples, samples, and rewards
            reference_ctx: Reference model context with tokenizer and logprobs function
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            pad_token_id: ID of the padding token
            kl_coef: KL divergence coefficient (for future use)

        Returns:
            RLDataset instance with processed training data
        """
        examples = env_step.examples
        responses = env_step.responses
        rewards = env_step.rewards

        # Get tokenizer from reference context
        tokenizer = reference_ctx.tokenizer

        # Prepare data to compute reference logprobs
        batch_items = []
        for i, example in enumerate(examples):
            prompt_tokens = tokenizer.encode(example["prompt"], add_special_tokens=True)[-max_input_length:]
            prompt_attention_mask = [0] * (max_input_length - len(prompt_tokens)) + [1] * len(prompt_tokens)
            prompt_tokens = [pad_token_id] * (max_input_length - len(prompt_tokens)) + prompt_tokens

            for response in responses[i]:
                answer_tokens = response["tokens"][:max_output_length]
                answer_attention_mask = [1] * len(answer_tokens) + [0] * (max_output_length - len(answer_tokens))
                answer_tokens = answer_tokens + [pad_token_id] * (max_output_length - len(answer_tokens))
                answer_logprobs = response["logprobs"][:max_output_length]
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

        # Prepare all data as numpy arrays
        all_prompt_tokens = np.concatenate([item["prompt_tokens"] for item in batch_items])
        all_prompt_masks = np.concatenate([item["prompt_attention_mask"] for item in batch_items])
        all_answer_tokens = np.concatenate([item["answer_tokens"] for item in batch_items])
        all_answer_masks = np.concatenate([item["answer_attention_mask"] for item in batch_items])
        all_policy_logprobs = np.concatenate([item["answer_logprobs"] for item in batch_items])

        # Compute reference logprobs using reference context (context handles batching internally)
        reference_logprobs = reference_ctx.compute_logprobs(
            all_prompt_tokens,
            all_prompt_masks,
            all_answer_tokens,
            all_answer_masks,
        )

        # Split back into lists for consistency with rest of function
        all_reference_logprobs = [reference_logprobs[i] for i in range(len(batch_items))]
        all_logprobs = [all_policy_logprobs[i] for i in range(len(batch_items))]
        output_masks_list = [all_answer_masks[i] for i in range(len(batch_items))]
        output_tokens_list = [all_answer_tokens[i] for i in range(len(batch_items))]
        prompt_tokens_list = [all_prompt_tokens[i] for i in range(len(batch_items))]
        prompt_masks_list = [all_prompt_masks[i] for i in range(len(batch_items))]

        # Stack all arrays
        all_reference_logprobs = np.stack(all_reference_logprobs, axis=0)
        all_logprobs = np.stack(all_logprobs, axis=0)
        output_masks = np.stack(output_masks_list, axis=0)
        output_tokens = np.stack(output_tokens_list, axis=0)
        prompt_tokens = np.stack(prompt_tokens_list, axis=0)
        prompt_masks = np.stack(prompt_masks_list, axis=0)

        all_advantages = []
        for rewards_group in rewards:
            advantages = compute_drgrpo_advantages_for_group(rewards_group)
            all_advantages.append(advantages)
        all_advantages = np.concatenate(all_advantages, axis=0)

        # Compute returns (repeat advantages for each token position)
        all_returns = jnp.repeat(all_advantages[..., None], output_masks.shape[1], axis=1)

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
        """Prepare data for RLOO training.

        Takes batched RL training data and transforms it into the format needed for
        RLLOO with policy gradient losses and KL penalties.

        Args:
            examples: Dictionary containing batched training examples with keys:
                - prompt_tokens: (batch_size, max_input_length) tokenized prompts
                - prompt_masks: (batch_size, max_input_length) attention masks for prompts
                - output_tokens: (batch_size, max_output_length) tokenized model outputs
                - output_masks: (batch_size, max_output_length) attention masks for outputs
                - returns: (batch_size, max_output_length) advantage values for each output token
                - reference_logprobs: (batch_size, max_output_length) log probs from reference model

        Returns:
            Dictionary containing processed batch ready for language model training:
                - input_ids: (batch_size, seq_len-1) input token sequences
                - attention_mask: (batch_size, seq_len-1) attention masks for inputs
                - position_ids: (batch_size, seq_len-1) position indices for each token
                - target_ids: (batch_size, seq_len-1) target tokens (shifted by 1 for next-token prediction)
                - loss_masks: (batch_size, seq_len-1) binary masks indicating which positions to compute loss on
                  (1 for output tokens, 0 for prompt tokens)
                - loss_weights: (batch_size, seq_len-1) advantage values used as weights in policy gradient loss
                - reference_logprobs: (batch_size, seq_len-1) reference model logprobs for KL penalty computation

            Where seq_len = max_input_length + max_output_length
        """
        return prepare_training_batch(
            prompt_tokens=examples["prompt_tokens"],
            prompt_masks=examples["prompt_masks"],
            output_tokens=examples["output_tokens"],
            output_masks=examples["output_masks"],
            loss_weights=examples["returns"],
            reference_logprobs=examples["reference_logprobs"],
            policy_logprobs=examples["policy_logprobs"],
        )


def compute_rloo_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """Compute RLOO advantages for a group of rewards.

    Args:
        rewards: Array of rewards for a group

    Returns:
        Normalized advantages
    """
    n = len(rewards)
    if n <= 1:
        return np.zeros_like(rewards)

    total = rewards.sum()
    leave_one_out_baselines = (total - rewards) / (n - 1)
    advantages = rewards - leave_one_out_baselines

    # Add random noise to avoid failure cases when all rewards are identical/zero
    generator = np.random.default_rng()
    advantages += generator.normal(loc=0.0, scale=1e-6, size=advantages.shape)
    return advantages


def compute_drgrpo_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """Compute Dr.GRPO advantages for a group of rewards by centering them.

    Args:
        rewards: Array of rewards for a group

    Returns:
        Centered advantages (rewards - mean)
    """
    advantages = rewards - rewards.mean()
    return advantages


def prepare_training_batch(
    prompt_tokens: np.ndarray,
    prompt_masks: np.ndarray,
    output_tokens: np.ndarray,
    output_masks: np.ndarray,
    loss_weights: np.ndarray,
    reference_logprobs: np.ndarray,
    policy_logprobs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Prepare training batch from prompt/output components for RLOO training.

    Takes prompt and output tokens/masks along with loss weights and logprobs,
    and transforms them into the format needed for language model training with
    RLOO policy gradient losses.

    Args:
        prompt_tokens: (batch_size, max_input_length) tokenized prompts
        prompt_masks: (batch_size, max_input_length) attention masks for prompts
        output_tokens: (batch_size, max_output_length) tokenized model outputs
        output_masks: (batch_size, max_output_length) attention masks for outputs
        loss_weights: (batch_size, max_output_length) advantage values for each output token
        reference_logprobs: (batch_size, max_output_length) log probs from reference model
        policy_logprobs: (batch_size, max_output_length) log probs from policy model

    Returns:
        Dictionary containing processed batch ready for language model training:
            - input_ids: (batch_size, seq_len-1) input token sequences
            - attention_mask: (batch_size, seq_len-1) attention masks for inputs
            - position_ids: (batch_size, seq_len-1) position indices for each token
            - target_ids: (batch_size, seq_len-1) target tokens (shifted by 1 for next-token prediction)
            - loss_masks: (batch_size, seq_len-1) binary masks indicating which positions to compute loss on
              (1 for output tokens, 0 for prompt tokens)
            - loss_weights: (batch_size, seq_len-1) advantage values used as weights in policy gradient loss
            - reference_logprobs: (batch_size, seq_len-1) reference model logprobs for KL penalty computation
            - policy_logprobs: (batch_size, seq_len-1) policy model logprobs

        Where seq_len = max_input_length + max_output_length
    """
    # Concatenate prompt and output tokens
    full_tokens = np.concatenate((prompt_tokens, output_tokens), axis=1)
    full_attention_mask = np.concatenate((prompt_masks, output_masks), axis=1)

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
                (prompt_masks.shape[0], prompt_masks.shape[1] - 1),
                dtype=np.float32,
            ),
            output_masks.astype(np.float32),
        ],
        axis=1,
    )

    # Create loss weights (advantages/returns for policy gradient)
    formatted_loss_weights = np.concatenate(
        [
            np.zeros(
                (prompt_masks.shape[0], prompt_masks.shape[1] - 1),
                dtype=np.float32,
            ),
            loss_weights.astype(np.float32),
        ],
        axis=1,
    )

    # Create reference logprobs for KL penalty
    formatted_reference_logprobs = np.concatenate(
        [
            np.zeros(
                (prompt_masks.shape[0], prompt_masks.shape[1] - 1),
                dtype=np.float32,
            ),
            reference_logprobs.astype(np.float32),
        ],
        axis=1,
    )

    formatted_policy_logprobs = np.concatenate(
        [
            np.zeros(
                (prompt_masks.shape[0], prompt_masks.shape[1] - 1),
                dtype=np.float32,
            ),
            policy_logprobs.astype(np.float32),
        ],
        axis=1,
    )

    return {
        "input_ids": input_tokens,
        "attention_mask": input_attention_mask,
        "position_ids": position_ids,
        "target_ids": target_tokens,
        "loss_masks": loss_masks,
        "loss_weights": formatted_loss_weights,
        "reference_logprobs": formatted_reference_logprobs,
        "policy_logprobs": formatted_policy_logprobs,
    }


def create_dataset_from_environment(
    environment,
    policy_ctx,
    reference_ctx,
    n_examples: int,
    prng_key,
    n_generations: int,
    max_input_length: int,
    max_output_length: int,
    pad_token_id: int,
    mode: str = "train",
    temperature: float = 1.0,
    step: int = 0,
) -> tuple["RLDataset", dict[str, float], EnvStep]:
    """Create RLDataset by stepping through the environment.

    Args:
        environment: Environment to step through
        policy_ctx: Context wrapping policy model (includes tokenizer, params, etc.)
        reference_ctx: Context wrapping reference model
        n_examples: Number of examples to process
        prng_key: Random key for sampling
        n_generations: Number of generations per example
        max_input_length: Maximum input length for padding
        max_output_length: Maximum output length for padding
        pad_token_id: Padding token ID
        mode: "train" or "eval"
        temperature: Generation temperature
        step: Current training step index

    Returns:
        RLDataset, metrics dictionary, and the raw EnvStep
    """
    # Step environment with policy context
    env_step = environment.step(
        inference_ctx=policy_ctx,
        n_examples=n_examples,
        prng_key=prng_key,
        mode=mode,
        n_generations=n_generations,
        temperature=temperature,
        step=step,
    )

    # Create dataset from environment step
    dataset = RLDataset.from_env_step(
        env_step=env_step,
        reference_ctx=reference_ctx,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        pad_token_id=pad_token_id,
    )

    return dataset, env_step.metrics, env_step
