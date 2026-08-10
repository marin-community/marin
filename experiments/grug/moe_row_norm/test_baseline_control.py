# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_row_norm.baseline_control import baseline_control_recipe
from experiments.grug.moe_row_norm.launch import GATE_1_POINTS, _row_norm_gate_1_recipe
from experiments.grug.moe_row_norm.optimizer import GrugMoeRowNormConfig


def test_baseline_control_matches_d512_variant_recipe_except_optimizer() -> None:
    baseline_model, baseline_optimizer = baseline_control_recipe()
    variant_model, variant_optimizer = _row_norm_gate_1_recipe(GATE_1_POINTS[0])

    assert dataclasses.asdict(baseline_model) == dataclasses.asdict(variant_model)
    assert isinstance(baseline_optimizer, GrugMoeMuonHConfig)
    assert isinstance(variant_optimizer, GrugMoeRowNormConfig)

    baseline_fields = dataclasses.asdict(baseline_optimizer)
    variant_fields = dataclasses.asdict(variant_optimizer)
    assert baseline_fields.keys() == variant_fields.keys()
    for field, baseline_value in baseline_fields.items():
        variant_value = variant_fields[field]
        if isinstance(baseline_value, float):
            assert variant_value == pytest.approx(baseline_value)
        else:
            assert variant_value == baseline_value
