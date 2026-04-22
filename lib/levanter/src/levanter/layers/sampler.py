# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import AxisSelector, NamedArray

__all__ = ["Sampler"]


class Sampler(eqx.Module):
    """Simple temperature-based sampler for autoregressive models.

    Given logits and per-example temperatures, returns token indices. For
    ``temperature == 0`` we return greedy (argmax) tokens; otherwise we sample
    from the softmax distribution after scaling the logits by the inverse
    temperature.

    The inputs are expected to be ``NamedArray`` instances from *haliax*.
    ``logits`` must include a vocabulary axis (passed via *vocab_axis* when the
    sampler is created, default name "vocab"). The *temperatures* tensor should
    have the same axes as *logits* except for the vocabulary axis – e.g. a
    scalar, a per-batch, or a per-batch-and-time temperature.
    """

    Vocab: AxisSelector = eqx.field(static=True)

    def __init__(self, Vocab: hax.AxisSelector = "vocab"):
        self.Vocab = Vocab

    def __call__(
        self,
        logits: NamedArray,
        temperatures: NamedArray | float | jnp.ndarray,
        *,
        top_ps: NamedArray | float | jnp.ndarray | None = None,
        key: PRNGKeyArray,
    ) -> tuple[NamedArray, NamedArray]:
        """Sample token IDs and their log-probs.

        Args:
            logits : NamedArray
                Logits for each token in the vocabulary, with axes including *vocab_axis*.
            temperatures : NamedArray | float | jnp.ndarray
                Temperature values for sampling. Scalar or named array with the same axes as *logits* except for the vocabulary axis.
            top_ps : NamedArray | float | jnp.ndarray | None
                Optional nucleus-sampling cutoff. When set, only the smallest prefix of
                probability mass whose cumulative mass exceeds ``top_p`` remains
                eligible for sampling.
            key : PRNGKeyArray
                JAX random key for sampling.

        Returns:
            tokens : NamedArray
                Sampled token indices with the same axes as *temperatures*.
            log_probs : NamedArray
                Log-probabilities for each sampled token (same shape as *tokens*).
        """

        # Ensure float32 for numerical stability
        logits_f32 = logits.astype(jnp.float32)

        # Greedy tokens for temperature == 0
        greedy = hax.argmax(logits_f32, axis=self.Vocab)

        # Scale logits by temperature (broadcast across vocab axis)
        # Avoid division by zero by putting a dummy value (we'll mask later)
        safe_t = hax.where(temperatures == 0, 1.0, temperatures).astype(jnp.float32)
        scaled_logits = logits_f32 / safe_t

        sampling_logits = self._apply_top_p(scaled_logits, top_ps)
        samples = hax.random.categorical(key, sampling_logits, axis=self.Vocab)

        # Where temperature == 0, fall back to greedy choice
        tokens = hax.where(temperatures == 0, greedy, samples)

        vocab_axis = sampling_logits.resolve_axis(self.Vocab)
        vocab_axis_index = sampling_logits.axes.index(vocab_axis)
        sampling_logits_array = jnp.moveaxis(sampling_logits.array, vocab_axis_index, -1)
        selected_logits = jnp.take_along_axis(
            sampling_logits_array,
            jnp.expand_dims(tokens.array.astype(jnp.int32), axis=-1),
            axis=-1,
        ).squeeze(-1)
        log_z = jax.nn.logsumexp(sampling_logits_array, axis=-1)
        log_prob_tokens = hax.named(selected_logits - log_z, tokens.axes)

        return tokens, log_prob_tokens

    def _apply_top_p(
        self,
        scaled_logits: NamedArray,
        top_ps: NamedArray | float | jnp.ndarray | None,
    ) -> NamedArray:
        """Apply nucleus sampling in vocabulary space and return masked logits."""
        if top_ps is None:
            return scaled_logits

        vocab_axis = scaled_logits.resolve_axis(self.Vocab)
        vocab_axis_index = scaled_logits.axes.index(vocab_axis)
        logits_array = jnp.moveaxis(scaled_logits.array, vocab_axis_index, -1)
        sorted_indices = jnp.argsort(logits_array, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(logits_array, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        if isinstance(top_ps, NamedArray):
            top_p_array = top_ps.array
        else:
            top_p_array = top_ps
        threshold = jnp.clip(jnp.asarray(top_p_array, dtype=jnp.float32), min=0.0, max=1.0)[..., None]

        keep_sorted = cumulative_probs <= threshold
        keep_sorted = jnp.concatenate(
            [jnp.ones_like(keep_sorted[..., :1], dtype=bool), keep_sorted[..., 1:]],
            axis=-1,
        )
        filtered_sorted_logits = jnp.where(keep_sorted, sorted_logits, -jnp.inf)
        inverse_permutation = jnp.argsort(sorted_indices, axis=-1)
        filtered_logits = jnp.take_along_axis(filtered_sorted_logits, inverse_permutation, axis=-1)
        return hax.named(jnp.moveaxis(filtered_logits, -1, vocab_axis_index), scaled_logits.axes)
