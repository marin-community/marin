# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.moe.benchmark_ep_ring import _routing_statistics, _selected_experts


def test_ep_ring_benchmark_balanced_routing_has_exact_expert_counts():
    selected = _selected_experts(
        routing="balanced",
        tokens=128,
        top_k=4,
        num_experts=64,
        seed=0,
        skew_alpha=1.2,
    )

    counts = np.bincount(np.asarray(selected).reshape(-1), minlength=64)
    np.testing.assert_array_equal(counts, np.full(64, 8))


def test_ep_ring_benchmark_skew_routing_is_seeded_and_reports_padding():
    arguments = {
        "routing": "skew",
        "tokens": 128,
        "top_k": 4,
        "num_experts": 64,
        "skew_alpha": 1.2,
    }
    selected = _selected_experts(seed=17, **arguments)
    repeated = _selected_experts(seed=17, **arguments)
    different_seed = _selected_experts(seed=18, **arguments)

    np.testing.assert_array_equal(selected, repeated)
    assert not np.array_equal(selected, different_seed)

    statistics = _routing_statistics(selected, num_experts=64, capacity_factor=1.25)
    assert statistics["expert_count_max"] > statistics["expert_count_min"]
    assert statistics["padding_total"] == sum(statistics["padding_by_rank"])
    for group_sizes in statistics["quack_group_sizes_by_rank"]:
        assert sum(group_sizes) == statistics["local_capacity"]
