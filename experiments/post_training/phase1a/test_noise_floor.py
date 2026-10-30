# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.phase1a.noise_floor import SeedMetrics, gate_inputs


def test_gate_keeps_forward_reverse_and_range_conclusions_distinct():
    records = [
        SeedMetrics(
            method, step, seed, {"kl_P_X": forward[seed], "reverse_P_X": reverse[seed], "kl_P_chat": drift[seed]}
        )
        for method, forward, reverse, drift in (
            ("m1", (0.2, 0.3, 0.4), (0.1, 0.2, 0.3), (0.01, 0.02, 0.03)),
            ("m4", (0.89, 0.90, 0.91), (0.1, 0.2, 0.3), (0.01, 0.02, 0.03)),
        )
        for step in (10, 30, 50, 100)
        for seed in (2, 0, 1)
    ]
    results = gate_inputs(records, forward_baseline=0.9, reverse_baseline=0.7)
    m1, m4 = results[0], results[4]
    assert m1["metrics"]["kl_P_X"]["sample_std"] == pytest.approx(0.1)
    assert m1["metrics"]["kl_P_X"]["range"] == pytest.approx(0.2)
    assert m1["progress"]["forward"] == pytest.approx([7 / 9, 2 / 3, 5 / 9])
    assert m4["gains"]["forward"] == pytest.approx(0)
    drift_gate = m4["metrics"]["kl_P_chat"]["gate"]
    assert not drift_gate["forward"]["sample_std_pass"]
    assert not drift_gate["forward"]["range_pass"]
    assert drift_gate["reverse"]["sample_std_pass"]
    assert drift_gate["reverse"]["range_pass"]
    reverse_gate = m4["metrics"]["reverse_P_X"]["gate"]["reverse"]
    assert reverse_gate["threshold"] == pytest.approx(1 / 6)
    assert reverse_gate["sample_std_pass"]
    assert not reverse_gate["range_pass"]

    # Independently sampled continuations need not emit the same token buckets.
    records[0].metrics["reverse_P_X/bucket/tool_arguments"] = 0.3
    result = gate_inputs(records, forward_baseline=0.9, reverse_baseline=0.7)[0]
    assert result["unavailable_metrics"] == {"reverse_P_X/bucket/tool_arguments": {"missing_seeds": [0, 1]}}
    assert result["metrics"]["reverse_P_X"]["sample_std"] == pytest.approx(0.1)


def test_duplicate_seed_cannot_masquerade_as_three_independent_runs():
    records = [
        SeedMetrics(method, step, seed, {"kl_P_X": 0.3, "reverse_P_X": 0.2})
        for method in ("m1", "m4")
        for step in (10, 30, 50, 100)
        for seed in (0, 1, 1)
    ]
    with pytest.raises(ValueError, match="expected exactly seeds 0, 1, 2"):
        gate_inputs(records, forward_baseline=0.9, reverse_baseline=0.7)
