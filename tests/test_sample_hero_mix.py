# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.datakit.sample_hero_mix import cell_name, expected_sampled_tokens, keep_probabilities


def _write_mix(tmp_path):
    mix = {
        "available_tokens": {"c00q0": 1000, "c00q1": 10, "c01q0": 5000},
        "phases": [
            {"name": "initial", "weights": {"c00q0": 0.5, "c00q1": 0.0, "c01q0": 0.5}},
            {"name": "main", "weights": {"c00q0": 0.2, "c00q1": 0.3, "c01q0": 0.5}},
            {"name": "cooldown", "weights": {"c00q0": 0.1, "c00q1": 0.1, "c01q0": 0.8}},
        ],
    }
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(mix))
    return str(p)


def test_cell_name():
    assert cell_name(27, 0) == "c27q0"
    assert cell_name(7, 4) == "c07q4"


def test_keep_probabilities_and_capping(tmp_path):
    path = _write_mix(tmp_path)
    target = 10_000
    p = keep_probabilities(path, target, phase="main")
    # c00q0: 10000*0.2/1000 = 2.0 -> capped at 1.0
    assert p["c00q0"] == 1.0
    # c00q1: 10000*0.3/10 = 300 -> capped at 1.0 (tiny cell)
    assert p["c00q1"] == 1.0
    # c01q0: 10000*0.5/5000 = 1.0 -> exactly 1.0
    assert p["c01q0"] == 1.0


def test_keep_probabilities_subunity(tmp_path):
    path = _write_mix(tmp_path)
    target = 1_000  # small target -> all p < 1
    p = keep_probabilities(path, target, phase="main")
    assert abs(p["c00q0"] - 0.2) < 1e-9  # 1000*0.2/1000
    assert abs(p["c01q0"] - 0.1) < 1e-9  # 1000*0.5/5000
    assert p["c00q1"] == 1.0  # 1000*0.3/10 = 30 -> capped
    # expected sampled tokens: 0.2*1000 + 1.0*10 + 0.1*5000 = 200 + 10 + 500 = 710
    assert abs(expected_sampled_tokens(p, path) - 710.0) < 1e-6


def test_zero_weight_cell_kept_never(tmp_path):
    path = _write_mix(tmp_path)
    p = keep_probabilities(path, 1_000, phase="initial")  # c00q1 has weight 0 in initial
    assert p["c00q1"] == 0.0
