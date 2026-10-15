# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import safetensors.torch
import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, merge_tensor
from marin.merging.isospectral import isospectral_merge

from experiments.weight_merging.serial_iso import run


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
    params = {
        "method": "dare_ties",
        "coefficients": [1.0],
        "density": 0.5,
        "ties_density": 0.2,
        "scale": 0.25,
        "seed": 0,
    }
    stages = [
        {"output": str(tmp_path / "stage1"), "audit": str(tmp_path / "audit1.json"), "parameters": params},
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
    run(recipe, torch.device("cpu"))
    expected = weights.clone()
    arithmetic = MergeParameters(**{**params, "method": MergeMethod.DARE_TIES, "coefficients": (1.0,)})
    for row in [1, 3]:
        first = merge_tensor(weights[row], [specialist[row]], arithmetic, tensor_name=f"experts:{row}")
        expected[row], _ = isospectral_merge(first, [specialist[row]], 0.75, torch.device("cpu"))
    actual = safetensors.torch.load_file(tmp_path / "stage2/weights.safetensors")["experts"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for i in [1, 2]:
        audit = json.loads((tmp_path / f"audit{i}.json").read_text())
        assert audit["base"] == str(tmp_path / "base")
        assert 0 < audit["changed_elements"] <= weights.numel() // 4
        assert (tmp_path / f"stage{i}/config.json").read_bytes() == (tmp_path / "base/config.json").read_bytes()
    torch.testing.assert_close(
        safetensors.torch.load_file(tmp_path / "base/weights.safetensors")["experts"], weights, rtol=0, atol=0
    )
