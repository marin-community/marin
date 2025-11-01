# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
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
        key: PRNGKeyArray,
    ) -> tuple[NamedArray, NamedArray]:
        """Sample token IDs and their log-probs.

        Args:
            logits : NamedArray
                Logits for each token in the vocabulary, with axes including *vocab_axis*.
            temperatures : NamedArray | float | jnp.ndarray
                Temperature values for sampling. Scalar or named array with the same axes as *logits* except for the vocabulary axis.
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

        samples = hax.random.categorical(key, scaled_logits, axis=self.Vocab)

        # Where temperature == 0, fall back to greedy choice
        tokens = hax.where(temperatures == 0, greedy, samples)

        # Compute log-prob of each sampled token: logit - log_sum_exp(logits)
        oh = hnn.one_hot(tokens, self.Vocab)
        selected_logits = hax.sum(scaled_logits * oh, axis=self.Vocab)
        log_z = hnn.logsumexp(scaled_logits, axis=self.Vocab)
        log_prob_tokens = selected_logits - log_z

        return tokens, log_prob_tokens
