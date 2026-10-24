# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest
import torch
from safetensors.torch import save_file

from experiments.weight_merging.playground.pruning_response import main, retained_energy


def test_energy_scan_counts_zeros_and_tied_values():
    values = torch.tensor([0.0, 0.0, -3.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    assert retained_energy(values, (0.125, 0.25, 0.5, 1.0)) == [9.0, 18.0, 19.0, 19.0]
    assert retained_energy(torch.zeros(8), (0.1, 1.0)) == [0.0, 0.0]


def test_pruning_followup_matches_actual_update_size(tmp_path, monkeypatch):
    anchor, donor = tmp_path / "anchor", tmp_path / "donor"
    anchor.mkdir()
    donor.mkdir()
    base = torch.zeros(4096, dtype=torch.bfloat16)
    tuned = torch.linspace(-0.1, 0.1, 4096).bfloat16()
    save_file({"weight": base}, anchor / "model.safetensors")
    save_file({"weight": tuned}, donor / "model.safetensors")
    recipes, metrics = tmp_path / "recipes.json", tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pruning_response",
            "--anchor",
            str(anchor),
            "--donor",
            str(donor),
            "--output",
            str(recipes),
            "--metrics",
            str(metrics),
        ],
    )
    main()
    actual = json.loads(metrics.read_text())
    assert len(json.loads(recipes.read_text())) == 8
    for values in actual["candidates"].values():
        assert values["fp32_norm_relative_to_full_donor"] == pytest.approx(0.5, rel=1e-5)
        assert abs(values["target_retained_energy"] - values["topk_retained_energy"]) < 0.03
        assert values["cast_error_sq"] >= 0
