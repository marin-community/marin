# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.post_training.stage0.routing_health import packed_windows, relative_routing_gate, routing_gate


def test_gate_fails_when_a_layer_starves_more_than_five_percent_of_experts():
    healthy = np.full((2, 40), 100)
    healthy[1, 0] = 5  # one starved expert of 40 = 2.5%
    assert routing_gate(healthy)["passed"]
    starved = healthy.copy()
    starved[1, :3] = 5  # 3 of 40 = 7.5% below 10% of the mean
    verdict = routing_gate(starved)
    assert not verdict["passed"] and verdict["worst_layer"] == 1
    assert verdict["layers"][1]["low_load_experts"] == 3


def test_windows_pack_documents_with_segment_ids_and_drop_the_partial_tail():
    rows = list(packed_windows([[1, 2, 3], [4, 5], [6, 7, 8, 9]], window=4))
    assert [ids.tolist() for ids, _ in rows] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert [segments.tolist() for _, segments in rows] == [[0, 0, 0, 1], [1, 2, 2, 2]]


def test_relative_gate_allows_two_points_more_starvation_than_the_base_in_each_layer():
    base = np.full((2, 100), 100)
    base[1, :6] = 1  # the base already starves 6% of layer 1
    candidate = base.copy()
    candidate[1, 6:8] = 1  # 8%: +2 points in layer 1
    assert relative_routing_gate(candidate, base)["passed"]
    candidate[0, :3] = 1  # +3 points in layer 0, which the base keeps healthy
    verdict = relative_routing_gate(candidate, base)
    assert not verdict["passed"] and verdict["worst_layer"] == 0
