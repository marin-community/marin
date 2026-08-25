# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.data.text import ConcatDatasetComponent

from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS, _phase_weights
from experiments.grug.moe.long_context_datakit_moe_mix import (
    _BUCKET_LENGTH_TOKENS,
    long_context_datakit_components,
    long_context_phase_weights,
)

LONG_SUFFIX = "-gt_64k"
SHORT_SUFFIX = "-lte_64k"


def test_long_context_components_preserve_phase_weights():
    original = _phase_weights(1)
    weights = long_context_phase_weights()
    components = long_context_datakit_components(2)

    assert weights == original
    assert components.keys() == original.keys()
    assert all(isinstance(component, ConcatDatasetComponent) for component in components.values())


def test_long_context_components_repeat_long_data_twice():
    components = long_context_datakit_components(2)
    token_counts = {bucket: (short_tokens, long_tokens) for bucket, short_tokens, long_tokens in _BUCKET_LENGTH_TOKENS}

    for bucket in components.keys() - {"tail"}:
        children = components[bucket].children
        assert sum(SHORT_SUFFIX in name for name in children) == 1
        assert sum(LONG_SUFFIX in name for name in children) == (2 if token_counts[bucket][1] else 0)

    tail_children = components["tail"].children
    assert sum(SHORT_SUFFIX in name for name in tail_children) == len(_TAIL_BUCKETS)
    assert sum(LONG_SUFFIX in name for name in tail_children) == 2 * sum(
        token_counts[bucket][1] > 0 for bucket in _TAIL_BUCKETS
    )
