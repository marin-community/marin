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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_DEFAULT_KL_NEGATIVE_TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True)
class AlignedLogits:
    """Logits aligned as ``[sequence, position, vocab]`` with sample labels."""

    expected: np.ndarray
    actual: np.ndarray
    sample_ids: tuple[tuple[int, int], ...]


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


def compare_logits(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    sample_ids: Sequence[tuple[int, int]] | None = None,
    negative_kl_tolerance: float = _DEFAULT_KL_NEGATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Compare raw FP32 logits and a rounded host-FP64 log-probability oracle.

    The ``rounded_fp64_oracle_*`` fields are computed by host ``float64``
    log-softmax followed by an FP32 cast. They are diagnostics for these
    helper inputs, not observed runtime or device token log-probabilities.
    """

    reference = _as_fp32_logits(reference, name="reference")
    actual = _as_fp32_logits(actual, name="actual")
    if reference.shape != actual.shape:
        raise ValueError(f"logit shapes differ: {reference.shape} != {actual.shape}")
    if negative_kl_tolerance < 0:
        raise ValueError("negative_kl_tolerance must be non-negative")
    if sample_ids is not None and len(sample_ids) != reference.shape[0]:
        raise ValueError("sample_ids must have one identity per compared row")

    reference_logprobs = _host_fp64_logprobs(reference)
    actual_logprobs = _host_fp64_logprobs(actual)
    kl = np.sum(np.exp(reference_logprobs) * (reference_logprobs - actual_logprobs), axis=-1, dtype=np.float64)
    min_kl = float(np.min(kl))
    if min_kl < -negative_kl_tolerance:
        raise ValueError(f"FP64 diagnostic KL is materially negative: {min_kl}")
    reference_logprobs32 = reference_logprobs.astype(np.float32)
    actual_logprobs32 = actual_logprobs.astype(np.float32)
    logit_ulp = np.abs(_ordered_fp32(reference) - _ordered_fp32(actual))
    logprob_ulp = np.abs(_ordered_fp32(reference_logprobs32) - _ordered_fp32(actual_logprobs32))
    return {
        "entry_count": int(reference.size),
        "sample_count": int(reference.shape[0]),
        "valid": True,
        "logit_bitwise_mismatch_count": int(np.count_nonzero(reference.view(np.uint32) != actual.view(np.uint32))),
        "rounded_fp64_oracle_logprob_bitwise_mismatch_count": int(
            np.count_nonzero(reference_logprobs32.view(np.uint32) != actual_logprobs32.view(np.uint32))
        ),
        "max_abs_logit_error": float(np.max(np.abs(reference - actual))),
        "max_logit_ulp_error": int(np.max(logit_ulp)),
        "rounded_fp64_oracle_max_abs_logprob_error": float(np.max(np.abs(reference_logprobs32 - actual_logprobs32))),
        "rounded_fp64_oracle_max_logprob_ulp_error": int(np.max(logprob_ulp)),
        "fp64_diagnostic_kl": float(np.mean(kl, dtype=np.float64)),
        "fp64_diagnostic_kl_min": min_kl,
        "fp64_diagnostic_kl_negative_count": int(np.count_nonzero(kl < 0)),
        "fp64_diagnostic_kl_negative_tolerance": negative_kl_tolerance,
    }


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
