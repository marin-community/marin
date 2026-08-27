# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS, _phase_weights
from experiments.grug.moe.long_context_datakit_moe_mix import (
    _BUCKET_LENGTH_TOKENS,
    long_context_datakit_components,
    long_context_phase_weights,
)

LONG_PATH_MARKER = "/length=gt_64k"
SHORT_PATH_MARKER = "/length=lte_64k"


def test_long_context_components_preserve_phase_weights():
    original = _phase_weights(1)
    weights = long_context_phase_weights()
    components = long_context_datakit_components(2)

    assert weights == original
    assert components.keys() == original.keys()


def test_long_context_components_repeat_long_data_twice():
    components = long_context_datakit_components(2)
    token_counts = {bucket: (short_tokens, long_tokens) for bucket, short_tokens, long_tokens in _BUCKET_LENGTH_TOKENS}

    for bucket in components.keys() - {"tail"}:
        children = components[bucket].children
        paths = [component.cache_dir.name for component in children.values()]
        assert sum(SHORT_PATH_MARKER in path for path in paths) == 1
        assert sum(LONG_PATH_MARKER in path for path in paths) == (2 if token_counts[bucket][1] else 0)

    tail_children = components["tail"].children
    tail_paths = [component.cache_dir.name for component in tail_children.values()]
    assert sum(SHORT_PATH_MARKER in path for path in tail_paths) == len(_TAIL_BUCKETS)
    assert sum(LONG_PATH_MARKER in path for path in tail_paths) == 2 * sum(
        token_counts[bucket][1] > 0 for bucket in _TAIL_BUCKETS
    )
