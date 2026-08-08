# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tile_lifetime.attention import (
    AttentionPartial,
    finalize_attention_partial,
    finalize_normalized_attention_partial,
    merge_attention_partials,
    merge_normalized_attention_partials,
    normalize_attention_partial,
)


def _partial(seed: int) -> AttentionPartial:
    rng = np.random.default_rng(seed)
    row_max = rng.normal(size=(4, 3)).astype(np.float32)
    row_sum = rng.uniform(0.25, 5.0, size=(4, 3)).astype(np.float32)
    weighted = rng.normal(size=(4, 3, 8)).astype(np.float32) * row_sum[..., None]
    return AttentionPartial(row_max=row_max, row_sum_exp=row_sum, weighted_value_accumulator=weighted)


def test_compact_partial_merge_is_the_same_normalized_exponential_fold() -> None:
    left = _partial(11)
    right = _partial(17)

    full = finalize_attention_partial(merge_attention_partials(left, right))
    compact = finalize_normalized_attention_partial(
        merge_normalized_attention_partials(
            normalize_attention_partial(left),
            normalize_attention_partial(right),
        )
    )

    np.testing.assert_allclose(compact, full, rtol=2e-6, atol=2e-6)


def test_compact_partial_merge_is_deterministic_and_order_symmetric() -> None:
    left = normalize_attention_partial(_partial(23))
    right = normalize_attention_partial(_partial(29))

    first = merge_normalized_attention_partials(left, right)
    repeated = merge_normalized_attention_partials(left, right)
    reversed_order = merge_normalized_attention_partials(right, left)

    np.testing.assert_array_equal(first.row_log_normalizer, repeated.row_log_normalizer)
    np.testing.assert_array_equal(first.normalized_weighted_value, repeated.normalized_weighted_value)
    np.testing.assert_allclose(first.row_log_normalizer, reversed_order.row_log_normalizer, rtol=0, atol=0)
    np.testing.assert_allclose(
        first.normalized_weighted_value,
        reversed_order.normalized_weighted_value,
        rtol=1e-6,
        atol=1e-6,
    )
