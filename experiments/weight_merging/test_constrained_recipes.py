# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import safetensors.torch
import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, merge_tensor
from marin.merging.isospectral import isospectral_merge
from marin.merging.learned import differentiable_weight_blend

from experiments.weight_merging.behavioral_residual import residuals
from experiments.weight_merging.calibration_model import CalibrationOutput, FrozenGrugBank, allocate_chunk_counts
from experiments.weight_merging.constrained_calibrated import blend_slice
from experiments.weight_merging.constrained_merge import main as constrained_main
from experiments.weight_merging.serial_iso import run as serial_iso_run
from experiments.weight_merging.serial_merge import main as serial_main
from experiments.weight_merging.supports import (
    activation_allocation,
    allocation_prefix,
    matched_random_allocation,
    structured_support,
)


@pytest.mark.parametrize(
    "scores,limits,budget,expected",
    [
        ({"b": 0.0, "a": 0.0}, {"a": 3, "b": 3}, 3, {"b": 2, "a": 1}),
        ({"a": 6.0, "b": 2.0}, {"a": 3, "b": 2}, 7, {"a": 3, "b": 2}),
    ],
)
def test_chunk_allocation_preserves_tie_order_and_stops_at_capacity(scores, limits, budget, expected):
    assert allocate_chunk_counts(scores, limits, budget) == expected


def test_export_slices_match_training_chunks_that_cross_expert_boundaries():
    generator = torch.Generator().manual_seed(17)
    base = torch.randn(5, 3, 4, generator=generator, dtype=torch.bfloat16)
    donors = [torch.randn(5, 3, 4, generator=generator, dtype=torch.bfloat16) for _ in range(2)]
    coefficients = torch.randn(7, 3, generator=generator).softmax(-1)
    trained = differentiable_weight_blend(base, tuple(donors), coefficients[:, 1:], block_elements=8)
    exported = torch.stack(
        [blend_slice(base[row], [d[row] for d in donors], coefficients, row * 12, 60) for row in range(5)]
    )
    torch.testing.assert_close(exported, trained, rtol=0, atol=0)


def test_behavioral_residual_reads_cached_calibration_output():
    class FixedBank:
        def forward(self, weights, tokens, positions):
            return CalibrationOutput(
                logits=torch.tensor([[[2.0, -1.0]]]),
                hidden_states=(torch.tensor([[[1.0, 2.0]]]),),
            )

    example = SimpleNamespace(
        identity="example",
        domain="bfcl",
        tokens=torch.tensor([[1]]),
        positions=torch.tensor([0]),
    )
    cached = CalibrationOutput(
        logits=torch.tensor([[[2.0, -1.0]]]),
        hidden_states=(torch.tensor([[[1.0, 2.0]]]),),
    )

    measured = residuals(cast(FrozenGrugBank, FixedBank()), {}, [example], {example.identity: cached}, hidden_weight=1.0)

    assert measured["bfcl"]["logit_kl"] == pytest.approx(0.0, abs=1e-7)
    assert measured["bfcl"]["hidden_normalized_mse"] == 0.0


def test_component_and_activation_supports_obey_scalar_budget_and_nested_allocation():
    tensors = {}
    for layer in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.{layer}.mlp.experts.{projection}.weight"] = {"shape": [16, 2, 2]}
            tensors[f"model.layers.{layer}.shared_expert.{projection}.weight"] = {"shape": [2, 2]}
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = {"shape": [2, 2]}
    headers = {"weights": tensors}
    total = 4 * (16 * 12 + 16)
    early = structured_support(headers, "early", 0.25, 0)
    late = structured_support(headers, "late", 0.25, 0)
    assert {s.name.split(".")[2] for s in early} == {"0"}
    assert {s.name.split(".")[2] for s in late} == {"3"}
    counts = {"numeric": [[3] * 8 + [1] * 8 for _ in range(4)], "tools": [[1] * 8 + [3] * 8 for _ in range(4)]}
    allocation = activation_allocation(headers, counts, 0.25)
    prefix = allocation_prefix(allocation, total, 0.125)
    assert prefix == allocation[: len(prefix)]
    assert len({(x["layer"], x["expert"]) for x in allocation}) == len(allocation)
    assert all(x["expert"] < 8 if x["donor"] == 0 else x["expert"] >= 8 for x in allocation)
    assert all(sum(x["elements"] for x in allocation if x["donor"] == d) <= total * 0.25 / 2 for d in (0, 1))
    randomized = matched_random_allocation(allocation, 16, 0)
    assert len({(x["layer"], x["expert"]) for x in randomized}) == len(randomized)
    for layer in range(4):
        for donor in (0, 1):
            assert sum(x["layer"] == layer and x["donor"] == donor for x in randomized) == sum(
                x["layer"] == layer and x["donor"] == donor for x in allocation
            )


def test_serial_stages_keep_base_files_and_frozen_weights(tmp_path, monkeypatch):
    base = torch.arange(32, dtype=torch.bfloat16).reshape(8, 2, 2) / 7
    sources = []
    for i in range(3):
        folder = tmp_path / str(i)
        folder.mkdir()
        safetensors.torch.save_file({"experts": base + i / 3, "norm": torch.ones(8)}, folder / "weights.safetensors")
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"experts": "weights.safetensors", "norm": "weights.safetensors"}})
        )
        (folder / "config.json").write_text('{"frozen": true}\n')
        sources.append({"path": str(folder), "revision": str(i)})
    support = [{"name": "experts", "rows": [1, 3]}]
    stages = [
        {
            "donor": i,
            "fraction": 0.5,
            "support": support,
            "output": str(tmp_path / f"stage{i}/model"),
            "audit": str(tmp_path / f"stage{i}/audit.json"),
        }
        for i in range(2)
    ]
    recipe = {
        "id": "serial-test",
        "base": sources[0],
        "donors": sources[1:],
        "threads": 1,
        "support": support,
        "support_cap": 0.25,
        "change_cap": 0.25,
        "stages": stages,
        "output": stages[-1]["output"],
        "audit": stages[-1]["audit"],
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr("sys.argv", ["serial_merge", "--recipe", str(path)])
    invalid_stage = {**stages[0], "audit": stages[0]["output"] + "/sub/../audit.json"}
    path.write_text(json.dumps({**recipe, "stages": [invalid_stage, stages[1]]}))
    with pytest.raises(ValueError, match="outside the model directory"):
        serial_main()
    assert not Path(stages[0]["output"]).exists()
    path.write_text(json.dumps(recipe))
    serial_main()
    expected = base.clone()
    for i in range(1, 3):
        donor = (base + i / 3).float()
        expected[[1, 3]] = (expected[[1, 3]].float() + 0.5 * (donor[[1, 3]] - expected[[1, 3]].float())).to(base.dtype)
    for stage in stages:
        audit = json.loads(Path(stage["audit"]).read_text())
        assert audit["changed_elements"] == 8 and audit["total_elements"] == 40
        assert Path(stage["output"], "config.json").read_bytes() == (tmp_path / "0/config.json").read_bytes()
    actual = safetensors.torch.load_file(Path(stages[-1]["output"]) / "weights.safetensors")
    torch.testing.assert_close(actual["experts"], expected, rtol=0, atol=0)


def test_row_scaled_merge_applies_each_expert_fraction(tmp_path, monkeypatch):
    base = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    donor = base + 4
    sources = []
    for name, weights in [("base", base), ("donor", donor)]:
        folder = tmp_path / name
        folder.mkdir()
        safetensors.torch.save_file({"experts": weights}, folder / "weights.safetensors")
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"experts": "weights.safetensors"}})
        )
        (folder / "config.json").write_text('{"preserve": true}\n')
        sources.append({"path": str(folder), "revision": name})
    output = tmp_path / "output"
    audit = tmp_path / "audit.json"
    recipe = {
        "id": "row-scaled-test",
        "base": sources[0],
        "donors": [sources[1]],
        "threads": 1,
        "support": [{"name": "experts", "rows": [0, 2]}],
        "support_cap": 0.50,
        "change_cap": 0.50,
        "parameters": {"method": "row_scaled"},
        "row_fractions": {"experts": {"0": 0.25, "2": 1.50}},
        "output": str(output),
        "audit": str(audit),
    }
    recipe_path = tmp_path / "row-scaled.json"
    recipe_path.write_text(json.dumps(recipe))
    monkeypatch.setattr("sys.argv", ["constrained_merge", "--recipe", str(recipe_path)])

    constrained_main()

    expected = base.clone()
    expected[0] += 1
    expected[2] += 6
    actual = safetensors.torch.load_file(output / "weights.safetensors")["experts"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stored_audit = json.loads(audit.read_text())
    assert stored_audit["changed_elements"] == 8
    assert (output / "config.json").read_bytes() == (tmp_path / "base/config.json").read_bytes()


def test_serial_iso_matches_two_stage_composition_and_preserves_original_base(tmp_path):
    generator = torch.Generator().manual_seed(12)
    weights = torch.randn(8, 4, 3, generator=generator)
    specialist = weights + 0.15 * torch.randn(weights.shape, generator=generator)
    for name, tensor in [("base", weights), ("donor", specialist)]:
        root = tmp_path / name
        root.mkdir()
        safetensors.torch.save_file({"experts": tensor}, root / "weights.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"experts": "weights.safetensors"}})
        )
        (root / "config.json").write_text('{"preserve": true}\n')
    parameters = {
        "method": "dare_ties",
        "coefficients": [1.0],
        "density": 0.5,
        "ties_density": 0.2,
        "scale": 0.25,
        "seed": 0,
    }
    stages = [
        {"output": str(tmp_path / "stage1"), "audit": str(tmp_path / "audit1.json"), "parameters": parameters},
        {
            "output": str(tmp_path / "stage2"),
            "audit": str(tmp_path / "audit2.json"),
            "parameters": {"method": "iso", "keep": 0.75},
        },
    ]
    recipe = {
        "id": "test",
        "base": {"path": str(tmp_path / "base"), "revision": "base"},
        "donors": [{"path": str(tmp_path / "donor"), "revision": "donor"}],
        "support": [{"name": "experts", "rows": [1, 3]}],
        "support_cap": 0.25,
        "change_cap": 0.25,
        "stages": stages,
        "output": stages[-1]["output"],
        "audit": stages[-1]["audit"],
    }
    serial_iso_run(recipe, torch.device("cpu"))
    expected = weights.clone()
    arithmetic = MergeParameters(**{**parameters, "method": MergeMethod.DARE_TIES, "coefficients": (1.0,)})
    for row in [1, 3]:
        first = merge_tensor(weights[row], [specialist[row]], arithmetic, tensor_name=f"experts:{row}")
        expected[row], _ = isospectral_merge(first, [specialist[row]], 0.75, torch.device("cpu"))
    actual = safetensors.torch.load_file(tmp_path / "stage2/weights.safetensors")["experts"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index in [1, 2]:
        stored_audit = json.loads((tmp_path / f"audit{index}.json").read_text())
        assert stored_audit["base"] == str(tmp_path / "base")
        assert 0 < stored_audit["changed_elements"] <= weights.numel() // 4
        assert (tmp_path / f"stage{index}/config.json").read_bytes() == (tmp_path / "base/config.json").read_bytes()
    torch.testing.assert_close(
        safetensors.torch.load_file(tmp_path / "base/weights.safetensors")["experts"], weights, rtol=0, atol=0
    )
