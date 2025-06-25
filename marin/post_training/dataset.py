from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
from transformers import AutoTokenizer


class Dataset:
    """Simplified dataset class that only handles training data preparation from generated samples."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_output_length: int,
        pad_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pad_token_id = pad_token_id

    def prepare_training_data_iterable(
        self, data_items: dict[str, np.ndarray], bsize: int, shuffle: bool = True, loop: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        """Create an iterable over processed training data."""
        N = data_items["returns"].shape[0]
        rng = jax.random.PRNGKey(0)

        while True:
            with jax.default_device(jax.devices("cpu")[0]):
                idxs = []
                for _ in range((bsize + (N - 1)) // N):
                    if shuffle:
                        rng, subrng = jax.random.split(rng)
                        curr_idxs = jax.random.permutation(subrng, np.arange(N))
                        idxs.extend(curr_idxs.tolist())
                    else:
                        curr_idxs = np.arange(N)
                        idxs.extend(curr_idxs.tolist())
                idxs = np.asarray(idxs)

                for batch_idx in range(len(idxs) // bsize):
                    batch_idxs = idxs[batch_idx * bsize : (batch_idx + 1) * bsize]
                    batch_examples = {
                        k: np.asarray([data_items[k][idx] for idx in batch_idxs]) for k in data_items.keys()
                    }
                    batch = self._prepare_rloo_examples(batch_examples)
                    yield batch

                if not loop:
                    break

    def _prepare_rloo_examples(self, examples: dict[str, Any]) -> dict[str, np.ndarray]:
        """Prepare examples for RLOO training."""
        full_tokens = np.concatenate(
            (
                examples["prompt_tokens"],
                examples["output_tokens"],
            ),
            axis=1,
        )
        full_attention_mask = np.concatenate(
            (
                examples["prompt_masks"],
                examples["output_masks"],
            ),
            axis=1,
        )
        full_position_ids = np.maximum(
            np.cumsum(full_attention_mask, axis=1) - 1,
            0,
        )
        input_tokens = full_tokens[:, :-1]
        input_attention_mask = full_attention_mask[:, :-1]
        target_tokens = full_tokens[:, 1:]
        position_ids = full_position_ids[:, :-1]
        loss_masks = np.concatenate(
            [
                np.zeros(
                    (
                        examples["prompt_masks"].shape[0],
                        examples["prompt_masks"].shape[1] - 1,
                    ),
                    dtype=np.float32,
                ),
                examples["output_masks"].astype(np.float32),
            ],
            axis=1,
        )
        loss_weights = np.concatenate(
            [
                np.zeros(
                    (
                        examples["prompt_masks"].shape[0],
                        examples["prompt_masks"].shape[1] - 1,
                    ),
                    dtype=np.float32,
                ),
                examples["returns"].astype(np.float32),
            ],
            axis=1,
        )
        reference_logprobs = np.concatenate(
            [
                np.zeros(
                    (
                        examples["prompt_masks"].shape[0],
                        examples["prompt_masks"].shape[1] - 1,
                    ),
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
