# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side comparison helpers for Snowball training and paged serving.

Serving emits prefill tokens sequence-major and subsequent decode steps
position-major.  This module makes that layout conversion explicit before any
metric is computed; comparing the two flat buffers directly is invalid for a
multi-sequence request.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from levanter.layers.attention import AttentionMask
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.models.lm_model import LmExample

_DEFAULT_KL_NEGATIVE_TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True)
class AlignedLogits:
    """Logits aligned as ``[sequence, position, vocab]`` with sample labels."""

    expected: np.ndarray
    actual: np.ndarray
    sample_ids: tuple[tuple[int, int], ...]


@dataclasses.dataclass(frozen=True)
class SnowballParameterBudget:
    """Pre-construction parameter and tensor-plus-gradient memory estimate."""

    parameter_count: int
    tensor_bytes: int
    gradient_bytes: int

    @property
    def tensor_and_gradient_bytes(self) -> int:
        return self.tensor_bytes + self.gradient_bytes


def snowball_parameter_budget(config: Any) -> SnowballParameterBudget:
    """Estimate Snowball parameter memory from config shapes without building a model.

    The estimate includes the embedding/output matrices, all layer weights, and FP32 RMSNorm
    vectors. It intentionally excludes optimizer state, activations, KV pages, and compiler
    temporaries; callers should use it as a construction guard, not a peak-memory prediction.
    """
    hidden = config.hidden_dim
    head_dim = config.inferred_head_dim
    attention = (
        2 * hidden * config.num_heads * head_dim
        + 2 * hidden * config.num_kv_heads * head_dim
        + hidden * config.num_heads
    )
    gated_norm = 2 * hidden * 128
    routed_moe = (
        hidden * config.num_experts + config.num_experts + 3 * config.num_experts * hidden * config.intermediate_dim
    )
    shared_mlp = 3 * hidden * config.shared_expert_intermediate_dim
    per_block = attention + 2 * hidden + 2 * gated_norm + routed_moe + shared_mlp
    parameter_count = 2 * config.vocab_size * hidden + config.num_layers * per_block + 2 * hidden + 2 * gated_norm
    tensor_bytes = parameter_count * 4
    return SnowballParameterBudget(parameter_count, tensor_bytes, tensor_bytes)


def snowball_actual_parameter_count(model: Any) -> int:
    """Count array-valued model leaves for comparison with the config estimate."""
    return sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def _scalar_value(value: NamedArray | jnp.ndarray) -> float:
    if isinstance(value, NamedArray):
        value = value.array
    return float(np.asarray(value))


def _weighted_target_cross_entropy(logits: np.ndarray, target_ids: np.ndarray, loss_weight: np.ndarray) -> float:
    """Compute the host-side served-logit CE with LmExample's final-position mask."""
    target_ids = np.asarray(target_ids)
    loss_weight = np.asarray(loss_weight, dtype=np.float32)
    valid_targets = target_ids >= 0
    if loss_weight.shape != target_ids.shape:
        raise ValueError("loss_weight and target_ids must have the same shape")
    effective_weight = loss_weight * valid_targets
    logprobs = _host_fp32_logprobs(logits)
    rows = np.arange(target_ids.size)[valid_targets]
    weighted_loss = -logprobs[rows, target_ids[valid_targets]] * effective_weight[valid_targets]
    denominator = np.sum(effective_weight, dtype=np.float32)
    return float(np.sum(weighted_loss, dtype=np.float32) / denominator) if denominator != 0 else 0.0


def diagnose_snowball_training_vs_paged_decode(
    model,
    example: LmExample,
    *,
    prefill_length: int,
    page_size: int = 2,
    source: str,
    seed: int,
) -> dict[str, Any]:
    """Compare standard Snowball teacher forcing with reassembled paged outputs.

    The model is supplied by the caller, so the full forward and paged decode use the same
    parameter tree. Gradients are computed only through the standard full-forward loss; paged
    serving remains a forward diagnostic because serving-only kernels need not be differentiable.
    ``example`` must contain a batch of contiguous valid prefixes, represented by negative segment
    IDs for padding when padding is present.
    """
    if not isinstance(example.attn_mask, AttentionMask):
        raise ValueError("paged Snowball comparison requires a structured causal AttentionMask")
    attn_mask = example.attn_mask
    if not attn_mask.is_causal or attn_mask.explicit_mask is not None or attn_mask.causal_offset is not None:
        raise ValueError("paged Snowball comparison supports only ordinary causal masks")
    if attn_mask.bidirectional_window is not None:
        raise ValueError("paged Snowball comparison does not support bidirectional masks")

    tokens = np.asarray(example.tokens.array)
    if tokens.ndim != 2:
        raise ValueError(f"example tokens must have shape [batch, position], got {tokens.shape}")
    batch_size, sequence_length = tokens.shape
    if batch_size > 2 or sequence_length > 16:
        raise ValueError("bounded Snowball diagnostic supports at most batch=2 and sequence_length=16")
    if not 0 < prefill_length < sequence_length:
        raise ValueError("prefill_length must leave at least one continuation position")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    segment_ids = attn_mask.segment_ids
    if segment_ids is None:
        valid = np.ones_like(tokens, dtype=bool)
    else:
        query_segments, key_segments = segment_ids
        query_segments_array = np.asarray(query_segments.array)
        key_segments_array = np.asarray(key_segments.array)
        if query_segments_array.shape != tokens.shape or key_segments_array.shape != tokens.shape:
            raise ValueError("paged Snowball comparison requires batch-aligned q/k segment IDs")
        if not np.array_equal(query_segments_array, key_segments_array):
            raise ValueError("paged Snowball comparison requires matching q/k segment IDs")
        positive_segment_count = [len(np.unique(row[row >= 0])) for row in query_segments_array]
        if any(count > 1 for count in positive_segment_count):
            raise ValueError("paged Snowball comparison supports one segment per sequence")
        valid = np.asarray(segment_ids[0].array) >= 0
    lengths = valid.sum(axis=1).astype(np.int32)
    if np.any(lengths < prefill_length):
        raise ValueError("every sequence must have at least prefill_length valid tokens")
    expected_valid = np.arange(sequence_length)[None, :] < lengths[:, None]
    if not np.array_equal(valid, expected_valid):
        raise ValueError("valid positions must be contiguous prefixes")
    loss_weights = np.asarray(example.loss_weight.array, dtype=np.float32)
    if loss_weights.shape != tokens.shape:
        raise ValueError("LmExample loss_weight must match token batch and position axes")
    if np.any((~valid) & (loss_weights != 0)):
        raise ValueError("paged Snowball comparison cannot score padded positions")
    global_next_token_mask = np.arange(sequence_length)[None, :] < sequence_length - 1
    effective_loss_weights = loss_weights * global_next_token_mask

    compute_axis_mapping = {
        "batch": ("replica_dcn", "data", "expert"),
        # The bounded fixture uses context=1 and keeps its input position axis
        # replicated; context-sharded input placement is a separate acceptance case.
        "position": None,
        "vocab": "model",
    }
    with hax.axis_mapping(compute_axis_mapping):
        example = hax.auto_sharded(example)
        full_logits = np.asarray(model(example.tokens, attn_mask=example.attn_mask).array, dtype=np.float32)
    with hax.axis_mapping(compute_axis_mapping):
        standard_loss = model.compute_next_token_loss(example)
    standard_loss_value = _scalar_value(standard_loss)

    def reference_loss(current_model):
        logits = current_model(example.tokens, attn_mask=example.attn_mask).astype(jnp.float32)
        targets = hax.roll(example.tokens, -1, "position")
        target_distribution = hax.nn.one_hot(targets, current_model.Vocab, dtype=jnp.float32)
        position = example.tokens.axes[-1]
        final_position_mask = hax.logical_not(hax.nn.one_hot(-1, position, dtype=jnp.bool_))
        loss = hax.nn.cross_entropy_loss(
            logits,
            current_model.Vocab,
            target_distribution,
            weight=example.loss_weight * final_position_mask,
        )
        return loss.array if isinstance(loss, NamedArray) else loss

    with hax.axis_mapping(compute_axis_mapping):
        reference_loss_value, reference_gradients = eqx.filter_value_and_grad(reference_loss)(model)

    def standard_loss_for_grad(current_model):
        loss = current_model.compute_next_token_loss(example)
        return loss.array if isinstance(loss, NamedArray) else loss

    with hax.axis_mapping(compute_axis_mapping):
        _, standard_gradients = eqx.filter_value_and_grad(standard_loss_for_grad)(model)
    gradient_differences = []
    gradient_finite = True
    for reference_leaf, standard_leaf in zip(
        jax.tree_util.tree_leaves(reference_gradients), jax.tree_util.tree_leaves(standard_gradients), strict=True
    ):
        if isinstance(reference_leaf, jax.Array) and isinstance(standard_leaf, jax.Array):
            reference_array = np.asarray(reference_leaf)
            standard_array = np.asarray(standard_leaf)
            gradient_finite &= bool(np.isfinite(reference_array).all() and np.isfinite(standard_array).all())
            gradient_differences.append(np.max(np.abs(reference_array - standard_array)))

    pages_per_sequence = int(max((lengths + page_size - 1) // page_size))
    page_indices = np.arange(batch_size * pages_per_sequence, dtype=np.int32).reshape(batch_size, pages_per_sequence)
    cache = model.initial_cache(
        PageTableSpec(batch_size * pages_per_sequence, page_size, batch_size), dtype=jnp.float32
    )

    def page_batch(active_positions: list[tuple[int, int]]) -> PageBatchInfo:
        active_sequences = list(dict.fromkeys(sequence for sequence, _ in active_positions))
        positions_by_sequence = {
            sequence: [position for candidate, position in active_positions if candidate == sequence]
            for sequence in active_sequences
        }
        counts = [len(positions_by_sequence[sequence]) for sequence in active_sequences]
        cumulative = np.cumsum([0, *counts], dtype=np.int32)
        seq_lens = np.asarray(
            [max(positions_by_sequence[sequence]) + 1 for sequence in active_sequences], dtype=np.int32
        )
        return PageBatchInfo(
            slot_ids=hax.named(jnp.asarray(active_sequences, dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray(page_indices[active_sequences]), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray(seq_lens), "seq"),
            cu_q_lens=hax.named(jnp.asarray(cumulative), "seq"),
            num_seqs=jnp.asarray(len(active_sequences), dtype=jnp.int32),
            new_token_dests=hax.named(
                jnp.asarray(
                    [
                        page_indices[sequence, position // page_size] * page_size + position % page_size
                        for sequence, position in active_positions
                    ],
                    dtype=jnp.int32,
                ),
                "position",
            ),
            page_size=page_size,
        )

    prefill_positions = [(sequence, position) for sequence in range(batch_size) for position in range(prefill_length)]
    prefill_ids = np.asarray([tokens[sequence, position] for sequence, position in prefill_positions], dtype=np.int32)
    prefill_pos_ids = np.asarray([position for _, position in prefill_positions], dtype=np.int32)
    prefill_logits, cache = model.decode(
        hax.named(jnp.asarray(prefill_ids), "position"),
        cache,
        page_batch(prefill_positions),
        hax.named(jnp.asarray(prefill_pos_ids), "position"),
    )
    actual_chunks = [np.asarray(prefill_logits.array, dtype=np.float32)]
    actual_rows = [(sequence, position, row) for row, (sequence, position) in enumerate(prefill_positions)]
    for position in range(prefill_length, int(lengths.max())):
        active = [(sequence, position) for sequence in range(batch_size) if position < lengths[sequence]]
        if not active:
            continue
        ids = np.asarray([tokens[sequence, position] for sequence, _ in active], dtype=np.int32)
        positions = np.asarray([position for _, position in active], dtype=np.int32)
        logits, cache = model.decode(
            hax.named(jnp.asarray(ids), "position"),
            cache,
            page_batch(active),
            hax.named(jnp.asarray(positions), "position"),
        )
        actual_offset = len(actual_rows)
        actual_rows.extend(
            (sequence, position, actual_offset + row) for row, (sequence, position) in enumerate(active)
        )
        actual_chunks.append(np.asarray(logits.array, dtype=np.float32))

    actual = np.concatenate(actual_chunks, axis=0)
    expected = np.asarray([full_logits[sequence, position] for sequence, position, _ in actual_rows], dtype=np.float32)
    actual_ordered = np.asarray([actual[row] for _, _, row in actual_rows], dtype=np.float32)
    sample_ids = tuple((sequence, position) for sequence, position, _ in actual_rows)
    targets = np.asarray(
        [
            int(tokens[sequence, position + 1]) if position + 1 < sequence_length else -1
            for sequence, position, _ in actual_rows
        ],
        dtype=np.int32,
    )
    aligned_loss_weights = np.asarray(
        [effective_loss_weights[sequence, position] for sequence, position, _ in actual_rows], dtype=np.float32
    )
    served_loss = _weighted_target_cross_entropy(actual_ordered, targets, aligned_loss_weights)
    metrics = compare_logits(expected, actual_ordered, sample_ids=sample_ids, target_ids=targets)
    return {
        "source": source,
        "seed": seed,
        "config": dataclasses.asdict(model.config),
        "parameter_count": snowball_parameter_budget(model.config).parameter_count,
        "valid_positions": sample_ids,
        "targets": tuple(int(target) for target in targets),
        "training_loss": standard_loss_value,
        "served_logit_cross_entropy": served_loss,
        "served_loss_abs_difference": abs(standard_loss_value - served_loss),
        "independent_full_forward_loss": float(np.asarray(reference_loss_value)),
        "loss_abs_difference": abs(standard_loss_value - float(np.asarray(reference_loss_value))),
        "gradient_finite": gradient_finite,
        "gradient_max_abs_difference": float(max(gradient_differences, default=0.0)),
        "parameter_count_actual": snowball_actual_parameter_count(model),
        "effective_weight_sum": float(np.sum(effective_loss_weights, dtype=np.float32)),
        "effective_target_count": int(np.count_nonzero(effective_loss_weights)),
        "metrics": metrics,
        "alignment": "sequence-major full forward and prefill; position-major continuation reassembled by (sequence, position)",
        "paged_decode_gradient": "not computed; serving forward only",
    }


def _as_fp32_logits(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be float32, got {values.dtype}")
    if values.ndim != 2:
        raise ValueError(f"{name} must be a flat [token, vocab] array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite logits")
    return values


def align_phase_major_logits(
    expected_training: np.ndarray,
    actual_prefill: np.ndarray,
    actual_continuation: np.ndarray,
    *,
    num_sequences: int,
    sequence_length: int,
    prefill_length: int,
) -> AlignedLogits:
    """Align teacher-forced and paged outputs by explicit sequence/position identity.

    ``expected_training`` and ``actual_prefill`` are sequence-major.  The
    continuation buffer is position-major: each decode step contributes one
    row per sequence.  The returned arrays are both shaped
    ``[num_sequences, sequence_length, vocab]``.
    """

    if not 0 < prefill_length <= sequence_length:
        raise ValueError("prefill_length must be within the sequence")
    expected_training = _as_fp32_logits(expected_training, name="expected_training")
    actual_prefill = _as_fp32_logits(actual_prefill, name="actual_prefill")
    actual_continuation = _as_fp32_logits(actual_continuation, name="actual_continuation")
    vocab_size = expected_training.shape[1]
    expected_count = num_sequences * sequence_length
    prefill_count = num_sequences * prefill_length
    continuation_count = num_sequences * (sequence_length - prefill_length)
    if expected_training.shape != (expected_count, vocab_size):
        raise ValueError(f"expected_training shape {expected_training.shape} does not match the layout")
    if actual_prefill.shape != (prefill_count, vocab_size):
        raise ValueError(f"actual_prefill shape {actual_prefill.shape} does not match the layout")
    if actual_continuation.shape != (continuation_count, vocab_size):
        raise ValueError(f"actual_continuation shape {actual_continuation.shape} does not match the layout")

    expected = expected_training.reshape(num_sequences, sequence_length, vocab_size)
    prefill = actual_prefill.reshape(num_sequences, prefill_length, vocab_size)
    continuation = actual_continuation.reshape(sequence_length - prefill_length, num_sequences, vocab_size).transpose(
        1, 0, 2
    )
    actual = np.concatenate((prefill, continuation), axis=1)
    sample_ids = tuple(
        (sequence, position) for sequence in range(num_sequences) for position in range(sequence_length)
    )
    return AlignedLogits(expected=expected, actual=actual, sample_ids=sample_ids)


def _ordered_fp32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    sign = np.uint32(1 << 31)
    return np.where((bits & sign) != 0, ~bits, bits ^ sign).astype(np.int64)


def _host_fp64_logprobs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    return values - maximum - np.log(np.sum(np.exp(values - maximum), axis=-1, keepdims=True))


def _host_fp32_logprobs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    maximum = np.max(values, axis=-1, keepdims=True).astype(np.float32)
    shifted = values - maximum
    normalizer = np.log(np.sum(np.exp(shifted, dtype=np.float32), axis=-1, keepdims=True)).astype(np.float32)
    return shifted - normalizer


def _row_sha256(values: np.ndarray) -> tuple[str, ...]:
    """Hash each complete FP32 row using its contiguous native byte encoding."""
    return tuple(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in values)


def compare_logits(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    sample_ids: Sequence[tuple[int, int]] | None = None,
    target_ids: np.ndarray | None = None,
    negative_kl_tolerance: float = _DEFAULT_KL_NEGATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Compare raw FP32 logits and explicitly host-computed log-probability oracles.

    The ``host_fp32_*`` fields use host NumPy float32 log-softmax. The
    ``rounded_fp64_oracle_*`` fields use host float64 log-softmax followed by
    an FP32 cast. Neither is a captured trainer or serving log-probability
    kernel measurement. The full-vocabulary row fingerprints use the
    contiguous native FP32 bytes of each complete vocabulary row and preserve
    the comparison row order.
    """

    reference = _as_fp32_logits(reference, name="reference")
    actual = _as_fp32_logits(actual, name="actual")
    if reference.shape != actual.shape:
        raise ValueError(f"logit shapes differ: {reference.shape} != {actual.shape}")
    if negative_kl_tolerance < 0:
        raise ValueError("negative_kl_tolerance must be non-negative")
    if sample_ids is not None and len(sample_ids) != reference.shape[0]:
        raise ValueError("sample_ids must have one identity per compared row")
    if target_ids is not None:
        target_ids = np.asarray(target_ids)
        if target_ids.shape != (reference.shape[0],):
            raise ValueError("target_ids must have one target per compared row")
        if np.any((target_ids >= reference.shape[1]) | (target_ids < -1)):
            raise ValueError("target_ids must be -1 or valid vocabulary indices")

    reference_logprobs = _host_fp64_logprobs(reference)
    actual_logprobs = _host_fp64_logprobs(actual)
    kl = np.sum(np.exp(reference_logprobs) * (reference_logprobs - actual_logprobs), axis=-1, dtype=np.float64)
    min_kl = float(np.min(kl))
    if min_kl < -negative_kl_tolerance:
        raise ValueError(f"FP64 diagnostic KL is materially negative: {min_kl}")
    reference_logprobs32 = reference_logprobs.astype(np.float32)
    actual_logprobs32 = actual_logprobs.astype(np.float32)
    reference_fp32_logprobs = _host_fp32_logprobs(reference)
    actual_fp32_logprobs = _host_fp32_logprobs(actual)
    logit_ulp = np.abs(_ordered_fp32(reference) - _ordered_fp32(actual))
    logprob_ulp = np.abs(_ordered_fp32(reference_logprobs32) - _ordered_fp32(actual_logprobs32))
    relative_logit_denominator = np.where(np.abs(reference) > 0, np.abs(reference), 1.0)
    metrics = {
        "entry_count": int(reference.size),
        "sample_count": int(reference.shape[0]),
        "valid": True,
        "logit_bitwise_mismatch_count": int(np.count_nonzero(reference.view(np.uint32) != actual.view(np.uint32))),
        "rounded_fp64_oracle_logprob_bitwise_mismatch_count": int(
            np.count_nonzero(reference_logprobs32.view(np.uint32) != actual_logprobs32.view(np.uint32))
        ),
        "max_abs_logit_error": float(np.max(np.abs(reference - actual))),
        "max_relative_logit_error": float(np.max(np.abs(reference - actual) / relative_logit_denominator)),
        "max_logit_ulp_error": int(np.max(logit_ulp)),
        "rounded_fp64_oracle_max_abs_logprob_error": float(np.max(np.abs(reference_logprobs32 - actual_logprobs32))),
        "rounded_fp64_oracle_max_logprob_ulp_error": int(np.max(logprob_ulp)),
        "host_fp32_logprob_bitwise_mismatch_count": int(
            np.count_nonzero(reference_fp32_logprobs.view(np.uint32) != actual_fp32_logprobs.view(np.uint32))
        ),
        "host_fp32_max_abs_logprob_error": float(np.max(np.abs(reference_fp32_logprobs - actual_fp32_logprobs))),
        "fp64_diagnostic_kl": float(np.mean(kl, dtype=np.float64)),
        "fp64_diagnostic_kl_min": min_kl,
        "fp64_diagnostic_kl_negative_count": int(np.count_nonzero(kl < 0)),
        "fp64_diagnostic_kl_negative_tolerance": negative_kl_tolerance,
        "full_vocabulary": True,
        "reference_row_sha256": _row_sha256(reference),
        "actual_row_sha256": _row_sha256(actual),
        "reference_host_fp32_logprob_row_sha256": _row_sha256(reference_fp32_logprobs),
        "actual_host_fp32_logprob_row_sha256": _row_sha256(actual_fp32_logprobs),
    }
    if target_ids is not None:
        valid_targets = target_ids >= 0
        rows = np.arange(reference.shape[0])[valid_targets]
        ref_token_logprobs = reference_fp32_logprobs[rows, target_ids[valid_targets]]
        actual_token_logprobs = actual_fp32_logprobs[rows, target_ids[valid_targets]]
        metrics.update(
            {
                "target_logprob_count": int(rows.size),
                "host_fp32_target_logprob_bitwise_mismatch_count": int(
                    np.count_nonzero(ref_token_logprobs.view(np.uint32) != actual_token_logprobs.view(np.uint32))
                ),
                "host_fp32_target_logprob_max_abs_error": float(
                    np.max(np.abs(ref_token_logprobs - actual_token_logprobs), initial=0.0)
                ),
            }
        )
    return metrics


def analyze_raw_cases(
    raw: Mapping[str, np.ndarray],
    *,
    num_sequences: int,
    sequence_length: int,
    prefill_length: int,
) -> dict[str, Any]:
    """Analyze one or more bounded raw probe cases without rerunning the model."""

    prefix = "single_composition_capacity32" if num_sequences == 1 else "pair_composition_capacity64"
    aligned = align_phase_major_logits(
        raw[f"{prefix}_training"],
        raw[f"{prefix}_prefill"],
        raw[f"{prefix}_continuation"],
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        prefill_length=prefill_length,
    )
    stage_rows = {
        "prefill": prefill_length,
        "continuation": sequence_length - prefill_length,
        "full": sequence_length,
    }
    metrics = {
        "ad_vs_training": compare_logits(
            aligned.expected[0],
            _as_fp32_logits(raw[f"{prefix}_ad"], name=f"{prefix}_ad"),
            sample_ids=tuple((0, position) for position in range(sequence_length)),
        ),
        "prefill": compare_logits(
            aligned.expected[:, :prefill_length].reshape(-1, aligned.expected.shape[-1]),
            aligned.actual[:, :prefill_length].reshape(-1, aligned.actual.shape[-1]),
            sample_ids=tuple(
                (sequence, position) for sequence in range(num_sequences) for position in range(prefill_length)
            ),
        ),
        "continuation": compare_logits(
            aligned.expected[:, prefill_length:].reshape(-1, aligned.expected.shape[-1]),
            aligned.actual[:, prefill_length:].reshape(-1, aligned.actual.shape[-1]),
            sample_ids=tuple(
                (sequence, position)
                for sequence in range(num_sequences)
                for position in range(prefill_length, sequence_length)
            ),
        ),
        "full": compare_logits(
            aligned.expected.reshape(-1, aligned.expected.shape[-1]),
            aligned.actual.reshape(-1, aligned.actual.shape[-1]),
            sample_ids=aligned.sample_ids,
        ),
    }
    return {
        "case": prefix,
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "prefill_length": prefill_length,
        "sample_counts": {name: num_sequences * length for name, length in stage_rows.items()},
        "stage_metrics": metrics,
        "alignment": "[sequence, position, vocab]; continuation reshaped from position-major",
    }


def compare_target_batch_invariance(
    single_raw: Mapping[str, np.ndarray],
    batched_raw: Mapping[str, np.ndarray],
    *,
    sequence_length: int,
    prefill_length: int,
) -> dict[str, Any]:
    """Compare sequence zero across separately captured batch compositions."""

    single = align_phase_major_logits(
        single_raw["single_composition_capacity32_training"],
        single_raw["single_composition_capacity32_prefill"],
        single_raw["single_composition_capacity32_continuation"],
        num_sequences=1,
        sequence_length=sequence_length,
        prefill_length=prefill_length,
    )
    batched = align_phase_major_logits(
        batched_raw["pair_composition_capacity64_training"],
        batched_raw["pair_composition_capacity64_prefill"],
        batched_raw["pair_composition_capacity64_continuation"],
        num_sequences=2,
        sequence_length=sequence_length,
        prefill_length=prefill_length,
    )
    target_ids = tuple((0, position) for position in range(sequence_length))
    return {
        "target_sequence": 0,
        "comparison": "single composition vs first sequence of pair composition",
        "training": compare_logits(single.expected[0], batched.expected[0], sample_ids=target_ids),
        "prefill": compare_logits(
            single.actual[0, :prefill_length],
            batched.actual[0, :prefill_length],
            sample_ids=target_ids[:prefill_length],
        ),
        "continuation": compare_logits(
            single.actual[0, prefill_length:],
            batched.actual[0, prefill_length:],
            sample_ids=target_ids[prefill_length:],
        ),
        "full": compare_logits(single.actual[0], batched.actual[0], sample_ids=target_ids),
    }
