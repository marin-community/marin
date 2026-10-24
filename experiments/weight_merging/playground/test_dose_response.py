# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest
import torch
from safetensors.torch import save_file

from experiments.weight_merging.playground.dose_response import main


def test_whole_model_dose_matching_normalizes_all_methods(tmp_path, monkeypatch):
    anchor, donor = tmp_path / "anchor", tmp_path / "donor"
    anchor.mkdir()
    donor.mkdir()
    base = torch.linspace(-1, 1, 2048).bfloat16()
    tuned = (base.float() + torch.linspace(-0.02, 0.03, 2048)).bfloat16()
    save_file({"weight": base}, anchor / "model.safetensors")
    save_file({"weight": tuned}, donor / "model.safetensors")
    recipes, metrics = tmp_path / "recipes.json", tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dose_response",
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
    candidates = json.loads(recipes.read_text())
    actual = json.loads(metrics.read_text())
    assert len(candidates) == 11
    assert sum("magnitude_topk" in candidate["name"] for candidate in candidates) == 3
    for values in actual["candidates"].values():
        assert values["fp32_norm_relative_to_full_donor"] == pytest.approx(values["coefficient_equivalent"], rel=1e-5)
        assert values["cast_error_sq"] >= 0
        assert values["changed_bf16"] <= values["changed_fp32"]
