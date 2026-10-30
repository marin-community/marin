# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.post_training.phase1a.noise_floor import SeedMetrics, gate_inputs, seed_metrics


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


def test_vocabulary_flips_do_not_satisfy_routing_measurement():
    summaries = {}
    for family, count in (("P_X", 1000), ("P_R", 1000), ("P_chat", 1500), ("P_think", 300)):
        for kind in (("kl", "reverse", "flip") if family == "P_X" else ("kl", "flip")):
            field = "flip_rate" if kind == "flip" else "token_mean_kl"
            summaries[f"{kind}_{family}"] = {
                "sequences": count,
                "positions": 1702261,
                field: 0.2,
                "buckets": {"answer": {"positions": 1702261, field: 0.2}},
            }
    records = [
        seed_metrics(dict(method=method, step=step, seed=seed, learning_rate=5e-7, summaries=summaries))
        for method in ("m1", "m4")
        for step in (10, 30, 50, 100)
        for seed in (0, 1, 2)
    ]
    result = gate_inputs(records, forward_baseline=0.9, reverse_baseline=0.7)[0]
    assert result["metrics"]["token_flip_P_X"]["mean"] == pytest.approx(0.2)
    assert "flip_P_X" not in result["metrics"]
    assert result["routing_flip_status"] == "not_measured"

    routed_records = []
    for record in records:
        routing = {}
        for family, count in (("P_X", 1000), ("P_R", 1000), ("P_chat", 1500), ("P_think", 300)):
            total = {"pairs": 4 * count, "flips": 2 * record.seed}
            routing[family] = {
                "complete": True,
                "sequences": count,
                "tokens": 2 * count,
                "layers": [2, 7],
                "base_model": "frozen-B",
                "probe_identity": family,
                **total,
                "by_layer": {str(layer): {"pairs": 2 * count, "flips": record.seed} for layer in (2, 7)},
                "base_margin_quartiles": {"0": total},
                "buckets": {"answer": total},
                "position_buckets": {"0": total},
            }
        routed_records.append(
            seed_metrics(
                dict(
                    method=record.method,
                    step=record.step,
                    seed=record.seed,
                    learning_rate=5e-7,
                    summaries=summaries,
                    routing=routing,
                )
            )
        )
    result = gate_inputs(routed_records, forward_baseline=0.9, reverse_baseline=0.7)[0]
    assert result["routing_flip_status"] == "measured"
    assert result["metrics"]["routing_flip_P_X"]["sample_std"] == pytest.approx(0.0005)
    assert result["metrics"]["routing_flip_P_X"]["range"] == pytest.approx(0.001)
    assert result["metrics"]["token_flip_P_X"]["mean"] == pytest.approx(0.2)
    changed = dict(routed_records[-1].routing)
    changed["P_X"] = replace(changed["P_X"], probe_identity="other-frozen-sequences")
    routed_records[-1] = replace(routed_records[-1], routing=changed)
    with pytest.raises(ValueError, match="same B, frozen probes"):
        gate_inputs(routed_records, forward_baseline=0.9, reverse_baseline=0.7)
