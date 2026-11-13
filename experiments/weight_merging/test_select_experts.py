# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.weight_merging.select_experts import build_recipes


def test_expert_selection_applies_ranked_ids_to_all_projections_with_matched_controls():
    scores = [[-1.0] * 256 for _ in range(26)]
    for layer in range(26):
        scores[layer][layer + 5] = 3.0
        scores[layer][layer + 7] = 2.0
    template = {
        "output": "memory://campaign/template",
        "parameters": {"coefficients": [0.5, 0.5]},
        "tensor_coefficients": {"router": [1, 0]},
        "preserve_rows": {"embedding": [8, 9]},
    }
    recipes, selections = build_recipes(
        {"phases": {"generated": {"selection_score": scores}}}, template, [1, 2], 42, "nupa", "step20"
    )
    for layer in range(26):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.experts.{projection}.weight"
            assert recipes["routing-nupa001-step20"]["row_overrides"][key] == [
                {"row": layer + 5, "coefficients": [1.0, 0.0]}
            ]
            assert recipes["routing-nupa002-step20"]["row_overrides"][key] == [
                {"row": layer + 5, "coefficients": [1.0, 0.0]},
                {"row": layer + 7, "coefficients": [1.0, 0.0]},
            ]
    for name, recipe in recipes.items():
        assert recipe["parameters"]["coefficients"] == [0.0, 1.0]
        assert recipe["preserve_rows"] == {"embedding": [8, 9]}
        assert recipe["tensor_coefficients"] == {}
        assert "router" not in recipe["row_overrides"]
        if "random" in name:
            assert {len(rows) for rows in recipe["row_overrides"].values()} == {
                selections["variants"][name]["experts_per_layer"]
            }
    assert recipes["routing-base-step20"]["row_overrides"] == {}
    assert template["tensor_coefficients"] == {"router": [1, 0]}
