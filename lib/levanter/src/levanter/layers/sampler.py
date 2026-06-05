# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import AxisSelector, NamedArray

__all__ = ["Sampler", "SamplerTopKMode"]


class SamplerTopKMode(StrEnum):
    CANDIDATE = "candidate"
    THRESHOLD_MASK = "threshold_mask"


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
    top_k_mode: SamplerTopKMode = eqx.field(static=True)

    def __init__(self, Vocab: hax.AxisSelector = "vocab", *, top_k_mode: SamplerTopKMode = SamplerTopKMode.CANDIDATE):
        self.Vocab = Vocab
        self.top_k_mode = top_k_mode

    def __call__(
        self,
        logits: NamedArray,
        temperatures: NamedArray | float | jnp.ndarray,
        *,
        top_ps: NamedArray | float | jnp.ndarray | None = None,
        top_ks: NamedArray | int | jnp.ndarray | None = None,
        top_k_limit: int | None = None,
        key: PRNGKeyArray,
        return_log_probs: bool = True,
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

        if top_ks is not None:
            # The top-k serving path is already a sampled path; tracing the
            # all-greedy branch keeps a full-vocabulary logsumexp in the HLO.
            return self._sample(
                logits_f32,
                temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                top_k_limit=top_k_limit,
                key=key,
                return_log_probs=return_log_probs,
            )

        temperature_array = temperatures.array if isinstance(temperatures, NamedArray) else jnp.asarray(temperatures)
        all_greedy = jnp.all(temperature_array == 0)

        return jax.lax.cond(
            all_greedy,
            lambda: self._greedy(logits_f32, return_log_probs=return_log_probs),
            lambda: self._sample(
                logits_f32,
                temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                top_k_limit=top_k_limit,
                key=key,
                return_log_probs=return_log_probs,
            ),
        )

    def _greedy(self, logits_f32: NamedArray, *, return_log_probs: bool) -> tuple[NamedArray, NamedArray]:
        greedy = hax.argmax(logits_f32, axis=self.Vocab)
        if not return_log_probs:
            return greedy, hax.zeros(greedy.axes, dtype=jnp.float32)

        vocab_axis = logits_f32.resolve_axis(self.Vocab)
        vocab_axis_index = logits_f32.axes.index(vocab_axis)
        logits_array = jnp.moveaxis(logits_f32.array, vocab_axis_index, -1)
        selected_logits = jnp.take_along_axis(
            logits_array,
            jnp.expand_dims(greedy.array.astype(jnp.int32), axis=-1),
            axis=-1,
        ).squeeze(-1)
        log_z = jax.nn.logsumexp(logits_array, axis=-1)
        log_prob_tokens = hax.named(selected_logits - log_z, greedy.axes)

        return greedy, log_prob_tokens

    def _sample(
        self,
        logits_f32: NamedArray,
        temperatures: NamedArray | float | jnp.ndarray,
        *,
        top_ps: NamedArray | float | jnp.ndarray | None,
        top_ks: NamedArray | int | jnp.ndarray | None,
        top_k_limit: int | None,
        key: PRNGKeyArray,
        return_log_probs: bool,
    ) -> tuple[NamedArray, NamedArray]:
        # Scale logits by temperature (broadcast across vocab axis)
        # Avoid division by zero by putting a dummy value (we'll mask later)
        safe_t = hax.where(temperatures == 0, 1.0, temperatures).astype(jnp.float32)
        scaled_logits = logits_f32 / safe_t

        candidate_logits, candidate_token_ids = self._apply_optional_top_k(scaled_logits, top_ks, top_k_limit)
        candidate_logits = self._with_unsharded_vocab(candidate_logits)
        if candidate_token_ids is not None:
            candidate_token_ids = self._with_unsharded_vocab(candidate_token_ids)
        sampling_logits = self._apply_optional_top_p(candidate_logits, top_ps)
        sample_indices = self._sample_from_logits_by_categorical(sampling_logits, key)
        if candidate_token_ids is None:
            samples = sample_indices
            selected_indices = sample_indices
        else:
            samples = candidate_token_ids[self._vocab_axis_name(), sample_indices]
            selected_indices = sample_indices

        temperature_array = temperatures.array if isinstance(temperatures, NamedArray) else jnp.asarray(temperatures)
        any_greedy = jnp.any(temperature_array == 0)

        def choose_mixed_temperature_tokens() -> tuple[NamedArray, NamedArray]:
            if candidate_token_ids is None:
                greedy_tokens = hax.argmax(logits_f32, axis=self.Vocab)
                greedy_indices = greedy_tokens
            else:
                greedy_indices = hax.zeros_like(sample_indices)
                greedy_tokens = candidate_token_ids[self._vocab_axis_name(), greedy_indices]
            return (
                hax.where(temperatures == 0, greedy_tokens, samples),
                hax.where(temperatures == 0, greedy_indices, selected_indices),
            )

        tokens, selected_indices = jax.lax.cond(
            any_greedy,
            choose_mixed_temperature_tokens,
            lambda: (samples, selected_indices),
        )
        if not return_log_probs:
            return tokens, hax.zeros(tokens.axes, dtype=jnp.float32)

        vocab_axis = sampling_logits.resolve_axis(self._vocab_axis_name())
        vocab_axis_index = sampling_logits.axes.index(vocab_axis)
        sampling_logits_array = jnp.moveaxis(sampling_logits.array, vocab_axis_index, -1)
        selected_logits = jnp.take_along_axis(
            sampling_logits_array,
            jnp.expand_dims(selected_indices.array.astype(jnp.int32), axis=-1),
            axis=-1,
        ).squeeze(-1)
        log_z = jax.nn.logsumexp(sampling_logits_array, axis=-1)
        log_prob_tokens = hax.named(selected_logits - log_z, tokens.axes)

        return tokens, log_prob_tokens

    def _apply_optional_top_k(
        self,
        scaled_logits: NamedArray,
        top_ks: NamedArray | int | jnp.ndarray | None,
        top_k_limit: int | None,
    ) -> tuple[NamedArray, NamedArray | None]:
        if top_ks is None:
            return scaled_logits, None
        if top_k_limit is None:
            raise ValueError("top_k_limit must be configured when top_ks is provided")

        vocab_axis = scaled_logits.resolve_axis(self.Vocab)
        candidate_count = min(int(top_k_limit), vocab_axis.size)
        if candidate_count < 1:
            raise ValueError("top_k_limit must be positive")
        if self.top_k_mode == SamplerTopKMode.THRESHOLD_MASK:
            return self._apply_optional_top_k_threshold_mask(scaled_logits, top_ks, candidate_count), None
        if self.top_k_mode != SamplerTopKMode.CANDIDATE:
            raise ValueError(f"Unknown top_k_mode {self.top_k_mode!r}")

        vocab_axis_index = scaled_logits.axes.index(vocab_axis)
        logits_array = jnp.moveaxis(scaled_logits.array, vocab_axis_index, -1)
        top_values_array, top_indices_array = jax.lax.top_k(logits_array, candidate_count)

        candidate_axis = hax.Axis(vocab_axis.name, candidate_count)
        candidate_axes = tuple(candidate_axis if axis == vocab_axis else axis for axis in scaled_logits.axes)
        top_values = hax.named(jnp.moveaxis(top_values_array, -1, vocab_axis_index), candidate_axes)
        top_indices = hax.named(
            jnp.moveaxis(top_indices_array.astype(jnp.int32), -1, vocab_axis_index), candidate_axes
        )

        ranks = hax.arange(candidate_axis, dtype=jnp.int32)
        if isinstance(top_ks, NamedArray):
            requested = hax.clip(top_ks.astype(jnp.int32), 1, candidate_count)
            mask = ranks.broadcast_axis(requested.axes) < requested.broadcast_axis(candidate_axis)
            mask = mask.broadcast_to(top_values.axes)
        else:
            requested = jnp.clip(jnp.asarray(top_ks).astype(jnp.int32), 1, candidate_count)
            mask = ranks < requested
        return hax.where(mask, top_values, -jnp.inf), top_indices

    def _apply_optional_top_k_threshold_mask(
        self,
        scaled_logits: NamedArray,
        top_ks: NamedArray | int | jnp.ndarray,
        top_k_limit: int,
    ) -> NamedArray:
        logits = self._with_unsharded_vocab(scaled_logits)
        vocab_axis = logits.resolve_axis(self._vocab_axis_name())
        vocab_axis_index = logits.axes.index(vocab_axis)
        logits_array = jnp.moveaxis(logits.array, vocab_axis_index, -1)
        if isinstance(top_ks, NamedArray):
            requested = hax.clip(top_ks.astype(jnp.int32), 1, top_k_limit).array
        else:
            requested = jnp.clip(jnp.asarray(top_ks).astype(jnp.int32), 1, top_k_limit)
        masked_array = _top_k_threshold_mask(logits_array, requested, replace_val=-jnp.inf)
        return hax.named(jnp.moveaxis(masked_array, -1, vocab_axis_index), logits.axes)

    def _with_unsharded_vocab(self, logits: NamedArray) -> NamedArray:
        mapping = hax.partitioning.current_thread_local_mapping()
        if mapping is None:
            return logits

        vocab_axis = logits.resolve_axis(self._vocab_axis_name())
        if vocab_axis.name not in mapping:
            return logits

        unsharded_mapping = dict(mapping)
        del unsharded_mapping[vocab_axis.name]
        return hax.shard(logits, unsharded_mapping)

    def _vocab_axis_name(self) -> str:
        if isinstance(self.Vocab, hax.Axis):
            return self.Vocab.name
        return str(self.Vocab)

    def _sample_from_logits_by_categorical(self, sampling_logits: NamedArray, key: PRNGKeyArray) -> NamedArray:
        vocab_axis = sampling_logits.resolve_axis(self._vocab_axis_name())
        vocab_axis_index = sampling_logits.axes.index(vocab_axis)
        logits_array = jnp.moveaxis(sampling_logits.array, vocab_axis_index, -1)
        # Gumbel-max samples exactly from softmax(logits) while avoiding the
        # candidate-space sort that jax.random.categorical lowers to on TPU.
        gumbel = jax.random.gumbel(key, logits_array.shape, dtype=logits_array.dtype)
        sample = jnp.argmax(logits_array + gumbel, axis=-1).astype(jnp.int32)
        sample_axes = tuple(axis for axis in sampling_logits.axes if axis != vocab_axis)
        return hax.named(sample, sample_axes)

    def _apply_optional_top_p(
        self,
        scaled_logits: NamedArray,
        top_ps: NamedArray | float | jnp.ndarray | None,
    ) -> NamedArray:
        if top_ps is None:
            return scaled_logits
        if not isinstance(top_ps, NamedArray):
            try:
                top_p_array = np.asarray(top_ps)
            except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
                pass
            else:
                if top_p_array.shape == () and float(top_p_array) >= 1.0:
                    return scaled_logits

        top_p_array = top_ps.array if isinstance(top_ps, NamedArray) else jnp.asarray(top_ps)
        top_p_full_distribution = jnp.all(top_p_array >= 1.0)
        return jax.lax.cond(
            top_p_full_distribution,
            lambda: scaled_logits,
            lambda: self._apply_top_p(scaled_logits, top_ps),
        )

    def _apply_top_p(
        self,
        scaled_logits: NamedArray,
        top_ps: NamedArray | float | jnp.ndarray | None,
    ) -> NamedArray:
        """Apply nucleus sampling in vocabulary space and return masked logits."""
        if top_ps is None:
            return scaled_logits

        vocab_axis = scaled_logits.resolve_axis(self._vocab_axis_name())
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

        # Keep the smallest prefix whose cumulative mass reaches the threshold.
        # The cutoff-crossing token should remain eligible, but we should not
        # include the next token when the threshold is met exactly. A small
        # tolerance absorbs floating-point imprecision in softmax/cumsum (TPU
        # softmax differs from the true probabilities by ~5e-5) so the cutoff
        # doesn't shift by one token across devices.
        threshold_tol = jnp.asarray(1e-4, dtype=cumulative_probs.dtype)
        keep_sorted = cumulative_probs < threshold - threshold_tol
        keep_sorted = jnp.concatenate(
            [jnp.ones_like(keep_sorted[..., :1], dtype=bool), keep_sorted[..., :-1]],
            axis=-1,
        )
        filtered_sorted_logits = jnp.where(keep_sorted, sorted_logits, -jnp.inf)
        inverse_permutation = jnp.argsort(sorted_indices, axis=-1)
        filtered_logits = jnp.take_along_axis(filtered_sorted_logits, inverse_permutation, axis=-1)
        return hax.named(jnp.moveaxis(filtered_logits, -1, vocab_axis_index), scaled_logits.axes)


def _top_k_threshold_mask(logits: jax.Array, top_ks: jax.Array, *, replace_val: float) -> jax.Array:
    """Mask logits below a per-row top-k threshold using reduction-only search."""

    batch_shape = logits.shape[:-1]
    requested = jnp.broadcast_to(top_ks, batch_shape)
    finite_logits = jnp.where(jnp.isfinite(logits), logits, -1.0e20)
    lo = jnp.min(finite_logits, axis=-1) - 1.0
    hi = jnp.max(finite_logits, axis=-1)

    def body(_, bounds):
        lo, hi = bounds
        midpoint = (lo + hi) * 0.5
        count_gt = jnp.sum(logits > jnp.expand_dims(midpoint, axis=-1), axis=-1)
        lo = jnp.where(count_gt >= requested, midpoint, lo)
        hi = jnp.where(count_gt >= requested, hi, midpoint)
        return lo, hi

    cutoff, _ = jax.lax.fori_loop(0, 32, body, (lo, hi))
    return jnp.where(logits >= jnp.expand_dims(cutoff, axis=-1), logits, jnp.full_like(logits, replace_val))
