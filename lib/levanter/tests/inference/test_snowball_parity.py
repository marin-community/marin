# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from levanter.inference.snowball_parity import (
    align_phase_major_logits,
    analyze_raw_cases,
    compare_logits,
    compare_target_batch_invariance,
)


@pytest.fixture
def paired_raw_logits() -> dict[str, np.ndarray]:
    training = np.array(
        [
            [0.0, 1.0, -2.0],
            [2.0, -1.0, -2.0],
            [1.0, 0.0, -3.0],
            [-2.0, 3.0, 0.0],
            [3.0, 0.0, -1.0],
            [0.0, 2.0, -4.0],
            [4.0, -2.0, 1.0],
            [2.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    return {
        "pair_composition_capacity64_training": training,
        "pair_composition_capacity64_ad": training[:4].copy(),
        "pair_composition_capacity64_prefill": np.array(
            [[0.0, 1.0, -2.0], [2.0, -1.0, -2.0], [3.0, 0.0, -1.0], [0.0, 2.0, -4.0]],
            dtype=np.float32,
        ),
        "pair_composition_capacity64_continuation": np.array(
            [[1.0, 0.0, -3.0], [4.0, -2.0, 1.0], [-2.0, 3.0, 0.0], [2.0, -1.0, 3.0]],
            dtype=np.float32,
        ),
    }


def test_align_phase_major_logits_preserves_sequence_position_identity(paired_raw_logits):
    aligned = align_phase_major_logits(
        paired_raw_logits["pair_composition_capacity64_training"],
        paired_raw_logits["pair_composition_capacity64_prefill"],
        paired_raw_logits["pair_composition_capacity64_continuation"],
        num_sequences=2,
        sequence_length=4,
        prefill_length=2,
    )
    np.testing.assert_array_equal(aligned.expected, aligned.actual)
    assert aligned.sample_ids == ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))


def test_compare_logits_distinguishes_identical_and_nonidentical_distributions():
    same = np.array([[0.0, 1.0, -2.0]], dtype=np.float32)
    shifted = np.array([[2.0, -1.0, -2.0]], dtype=np.float32)
    identical = compare_logits(same, same)
    different = compare_logits(same, shifted)
    assert identical["fp64_diagnostic_kl"] == 0.0
    assert different["fp64_diagnostic_kl"] > 1e-3
    assert identical["rounded_fp64_oracle_logprob_bitwise_mismatch_count"] == 0
    assert different["rounded_fp64_oracle_max_logprob_ulp_error"] > 0


def test_compare_target_batch_invariance_detects_a_changed_target_decode_step(paired_raw_logits):
    single = {
        "single_composition_capacity32_training": paired_raw_logits["pair_composition_capacity64_training"][:4],
        "single_composition_capacity32_prefill": np.array([[0.0, 1.0, -2.0], [2.0, -1.0, -2.0]], dtype=np.float32),
        "single_composition_capacity32_continuation": np.array([[1.0, 0.0, -3.0], [-2.0, 3.0, 0.0]], dtype=np.float32),
    }
    unchanged = compare_target_batch_invariance(single, paired_raw_logits, sequence_length=4, prefill_length=2)
    for stage in ("training", "prefill", "continuation", "full"):
        assert unchanged[stage]["logit_bitwise_mismatch_count"] == 0

    changed_pair = {key: values.copy() for key, values in paired_raw_logits.items()}
    continuation = changed_pair["pair_composition_capacity64_continuation"]
    continuation[2, 0] = np.nextafter(continuation[2, 0], np.float32(np.inf))
    changed = compare_target_batch_invariance(single, changed_pair, sequence_length=4, prefill_length=2)
    assert changed["prefill"]["logit_bitwise_mismatch_count"] == 0
    assert changed["continuation"]["logit_bitwise_mismatch_count"] == 1
    assert changed["full"]["logit_bitwise_mismatch_count"] == 1


def test_analyze_raw_cases_detects_sequence_major_continuation_misalignment(paired_raw_logits):
    aligned = analyze_raw_cases(paired_raw_logits, num_sequences=2, sequence_length=4, prefill_length=2)
    assert aligned["sample_counts"] == {"prefill": 4, "continuation": 4, "full": 8}
    for metric in aligned["stage_metrics"].values():
        assert metric["logit_bitwise_mismatch_count"] == 0
        assert metric["fp64_diagnostic_kl"] == 0.0

    misordered = {key: values.copy() for key, values in paired_raw_logits.items()}
    continuation = misordered["pair_composition_capacity64_continuation"]
    misordered["pair_composition_capacity64_continuation"] = continuation[[0, 2, 1, 3]]
    wrong = analyze_raw_cases(misordered, num_sequences=2, sequence_length=4, prefill_length=2)
    assert wrong["stage_metrics"]["prefill"]["logit_bitwise_mismatch_count"] == 0
    assert wrong["stage_metrics"]["continuation"]["logit_bitwise_mismatch_count"] > 0
    assert wrong["stage_metrics"]["continuation"]["fp64_diagnostic_kl"] > 0
