# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import haliax as hax
from haliax import Axis

from levanter.inference.snowball_parity import (
    compare_logits,
    diagnose_snowball_training_vs_paged_decode,
    _weighted_target_cross_entropy,
    snowball_parameter_budget,
)
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.grug.sharding import compact_grug_mesh


def _diagnostic_example() -> LmExample:
    batch = Axis("batch", 2)
    position = Axis("position", 8)
    tokens = jnp.asarray(
        [[3, 5, 7, 11, 13, 17, 0, 0], [19, 23, 29, 31, 37, 0, 0, 0]],
        dtype=jnp.int32,
    )
    valid = jnp.asarray(
        [[True, True, True, True, True, True, False, False], [True, True, True, True, True, False, False, False]]
    )
    segments = jnp.where(valid, jnp.arange(2, dtype=jnp.int32)[:, None], -1)
    loss_weight = jnp.asarray(
        [[1.0, 0.5, 0.25, 0.75, 1.0, 0.9, 0.0, 0.0], [1.0, 0.2, 0.8, 0.4, 0.6, 0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    token_array = hax.named(tokens, (batch, position))
    segment_array = hax.named(segments, (batch, position))
    return LmExample(
        tokens=token_array,
        loss_weight=hax.named(loss_weight.astype(jnp.float32), (batch, position)),
        attn_mask=AttentionMask.causal().with_segment_ids(segment_array),
    )


def test_snowball_training_serving_parity_reports_aligned_full_vocab_metrics():
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=12,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="scatter",
        rms_norm_implementation="native",
    )
    budget = snowball_parameter_budget(config)
    assert budget.parameter_count < 100_000_000
    assert budget.tensor_and_gradient_bytes < 32 * 1024 * 1024

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(0))
        report = diagnose_snowball_training_vs_paged_decode(
            model,
            _diagnostic_example(),
            prefill_length=2,
            page_size=2,
            source="deterministic random Snowball parameters",
            seed=0,
        )

    assert report["valid_positions"] == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (0, 4),
        (1, 4),
        (0, 5),
    )
    assert report["targets"] == (5, 7, 23, 29, 11, 31, 13, 37, 17, 0, 0)
    assert report["metrics"]["entry_count"] == len(report["valid_positions"]) * config.vocab_size
    assert np.isfinite(report["metrics"]["max_abs_logit_error"])
    assert np.isfinite(report["metrics"]["max_relative_logit_error"])
    assert report["gradient_finite"]
    assert report["gradient_max_abs_difference"] < 1e-5
    assert report["parameter_count_actual"] == budget.parameter_count
    assert report["effective_weight_sum"] == pytest.approx(7.4)
    assert report["effective_target_count"] == 11
    assert np.isfinite(report["served_logit_cross_entropy"])
    assert report["loss_abs_difference"] < 1e-4
    assert report["served_loss_abs_difference"] < 1e-5


def test_weighted_served_loss_oracle_detects_target_and_row_misalignment():
    logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    targets = np.asarray([0, 1, 0, -1], dtype=np.int32)
    weights = np.asarray([0.2, 0.25, 0.0, 10.0], dtype=np.float32)
    expected = _weighted_target_cross_entropy(logits, targets, weights)
    wrong_target = _weighted_target_cross_entropy(logits, np.asarray([1, 1, 0, -1]), weights)
    wrong_order = _weighted_target_cross_entropy(logits[[1, 0, 2, 3]], targets, weights)
    zero_weight = _weighted_target_cross_entropy(logits, targets, np.zeros_like(weights))
    assert expected != wrong_target
    assert expected != wrong_order
    assert zero_weight == 0.0


def test_compare_logits_fingerprints_preserve_values_across_layouts():
    contiguous = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    backing = np.empty((2, 6), dtype=np.float32)
    backing[:, ::2] = contiguous
    backing[:, 1::2] = -1.0
    noncontiguous = backing[:, ::2]

    contiguous_metrics = compare_logits(contiguous, contiguous)
    noncontiguous_metrics = compare_logits(noncontiguous, noncontiguous)

    assert contiguous_metrics["full_vocabulary"] is True
    assert noncontiguous_metrics["reference_row_sha256"] == contiguous_metrics["reference_row_sha256"]
    assert noncontiguous_metrics["actual_row_sha256"] == contiguous_metrics["actual_row_sha256"]
    assert (
        noncontiguous_metrics["reference_host_fp32_logprob_row_sha256"]
        == contiguous_metrics["reference_host_fp32_logprob_row_sha256"]
    )


def test_compare_logits_fingerprints_change_only_changed_raw_row():
    reference = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    actual = reference.copy()
    actual[1, 2] = np.float32(9.0)

    metrics = compare_logits(reference, actual)

    assert metrics["actual_row_sha256"][1] != metrics["reference_row_sha256"][1]
    assert metrics["actual_row_sha256"][0] == metrics["reference_row_sha256"][0]
    assert metrics["actual_row_sha256"][2] == metrics["reference_row_sha256"][2]


def test_compare_logits_fingerprints_follow_public_row_order():
    valid_positions = ((1, 4), (0, 2), (1, 5))
    rows = np.asarray([[sequence, position] for sequence, position in valid_positions], dtype=np.float32)
    metrics = compare_logits(rows, rows, sample_ids=valid_positions)
    expected = tuple(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows)

    assert metrics["reference_row_sha256"] == expected
    assert metrics["actual_row_sha256"] == expected


def test_snowball_training_serving_parity_rejects_noncontiguous_validity():
    example = _diagnostic_example()
    invalid_segments = example.attn_mask.segment_ids[0].array.at[0, 2].set(-1).at[0, 3].set(0)
    invalid_mask = AttentionMask.causal().with_segment_ids(hax.named(invalid_segments, example.tokens.axes))
    invalid_example = LmExample(
        tokens=example.tokens,
        loss_weight=example.loss_weight,
        attn_mask=invalid_mask,
    )
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=12,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="scatter",
        rms_norm_implementation="native",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(1))
        with pytest.raises(ValueError, match="contiguous prefixes"):
            diagnose_snowball_training_vs_paged_decode(
                model,
                invalid_example,
                prefill_length=2,
                source="deterministic random Snowball parameters",
                seed=1,
            )


def test_snowball_training_serving_parity_rejects_packed_segments():
    example = _diagnostic_example()
    packed = example.attn_mask.segment_ids[0].array.at[0, 2].set(1)
    packed_example = LmExample(
        tokens=example.tokens,
        loss_weight=example.loss_weight,
        attn_mask=AttentionMask.causal().with_segment_ids(hax.named(packed, example.tokens.axes)),
    )
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=12,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="scatter",
        rms_norm_implementation="native",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(2))
        with pytest.raises(ValueError, match="one segment per sequence"):
            diagnose_snowball_training_vs_paged_decode(
                model,
                packed_example,
                prefill_length=2,
                source="deterministic random Snowball parameters",
                seed=2,
            )
