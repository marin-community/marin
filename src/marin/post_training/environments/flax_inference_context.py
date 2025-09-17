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

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from ..inference import batch_inference, FlaxSampler


class FlaxInferenceContext:
    """Wraps Flax inference components and hides model parameters."""

    def __init__(
        self,
        params: PyTree,
        sampler: FlaxSampler,
        prng_key: jnp.ndarray,
        tokenizer,
        get_logprobs_fn,
        reference_logprobs_bsize: int = 32,
    ):
        """Initialize with all inference components.

        Args:
            params: Model parameters (hidden from external use)
            sampler: Configured FlaxSampler (already has max_input_length, bsize, etc.)
            prng_key: JAX PRNG key for sampling
            tokenizer: Tokenizer
            get_logprobs_fn: Function to compute logprobs with params
            reference_logprobs_bsize: Batch size for logprobs computation (internal)
        """
        self._params = params
        self._sampler = sampler
        self._prng_key = prng_key
        self._tokenizer = tokenizer
        self._get_logprobs_fn = get_logprobs_fn
        self._reference_logprobs_bsize = reference_logprobs_bsize

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        n_generations: int = 1,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """Generate using hidden params and sampler."""

        # Split key for this generation
        self._prng_key, subkey = jax.random.split(self._prng_key)

        # The sampler already knows its max_input_length, bsize, etc.
        return batch_inference(
            self._sampler,
            self._params,
            prompts,
            subkey,
            n_generations=n_generations,
            verbose=False,
        )

    def compute_logprobs(
        self,
        input_tokens: np.ndarray,
        input_attention_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute log probabilities."""

        # Handle batching internally if needed
        batch_size = input_tokens.shape[0]
        if batch_size > self._reference_logprobs_bsize:
            # Process in chunks internally
            all_logprobs = []
            for i in range(0, batch_size, self._reference_logprobs_bsize):
                end = min(i + self._reference_logprobs_bsize, batch_size)
                batch_input_tokens = input_tokens[i:end]
                batch_input_attention_mask = input_attention_mask[i:end]
                batch_target_tokens = target_tokens[i:end]
                batch_target_attention_mask = target_attention_mask[i:end]

                batch_logprobs = self._get_logprobs_fn(
                    self._params,
                    batch_input_tokens,
                    batch_input_attention_mask,
                    batch_target_tokens,
                    batch_target_attention_mask,
                )
                all_logprobs.append(batch_logprobs)
            return np.concatenate(all_logprobs, axis=0)
        else:
            return self._get_logprobs_fn(
                self._params,
                input_tokens,
                input_attention_mask,
                target_tokens,
                target_attention_mask,
            )

    def update_params(self, new_params: PyTree):
        """Update the hidden model parameters (used by training loop)."""
        self._params = new_params
