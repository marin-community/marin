# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import safetensors.torch
import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import TensorSupport, export_constrained
from marin.merging.isospectral import isospectral_merge


@pytest.mark.parametrize("edit", [False, True])
def test_export_preserves_base_layout_and_audits_only_selected_rows(tmp_path, edit):
    base = tmp_path / "base"
    base.mkdir()
    weights = {"experts": torch.arange(32, dtype=torch.bfloat16).reshape(8, 2, 2), "norm": torch.ones(8)}
    safetensors.torch.save_file(weights, base / "weights-01.safetensors", metadata={"format": "pt"})
    index = {"metadata": {"total_size": 96}, "weight_map": dict.fromkeys(weights, "weights-01.safetensors")}
    (base / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))
    (base / "config.json").write_text('{"unchanged": true}\n')
    (base / "tokenizer_config.json").write_text('{"chat_template": "exact bytes"}\n')
    output = tmp_path / "out"
    reader = CheckpointReader(CheckpointSource(str(base), "pinned"))
    support = (TensorSupport("experts", (1, 3)),) if edit else ()
    audit = export_constrained(reader, str(output), support, 0.25, lambda name, row, value: value + 2)
    actual = safetensors.torch.load_file(output / "weights-01.safetensors")
    expected = weights["experts"].clone()
    if edit:
        expected[[1, 3]] += 2
    torch.testing.assert_close(actual["experts"], expected, rtol=0, atol=0)
    assert audit["changed_elements"] == (8 if edit else 0)
    assert audit["total_elements"] == 40
    for file in base.iterdir():
        if file.suffix != ".safetensors" or not edit:
            assert (output / file.name).read_bytes() == file.read_bytes()
    before = (base / "weights-01.safetensors").read_bytes()
    header_end = 8 + int.from_bytes(before[:8], "little")
    assert (output / "weights-01.safetensors").read_bytes()[:header_end] == before[:header_end]
    assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in base.iterdir())


def test_over_budget_support_produces_no_partial_export(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_file({"weight": torch.ones(4, 4)}, base / "weights.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
    reader = CheckpointReader(CheckpointSource(str(base), "pinned"))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Support exceeds"):
        export_constrained(
            reader, str(output), (TensorSupport("weight", (0, 1)),), 0.25, lambda name, row, value: value * 2
        )
    assert not output.exists()


@pytest.mark.parametrize("keep", [0.75, 0.9, 1.0])
def test_iso_composition_changes_frames_without_changing_spectrum(keep):
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(7, 4, dtype=torch.float64, generator=generator)
    donors = [base + 0.15 * torch.randn(7, 4, dtype=torch.float64, generator=generator) for _ in range(2)]
    merged, coefficients = isospectral_merge(base, donors, keep, torch.device("cpu"))
    torch.testing.assert_close(torch.linalg.svdvals(merged), torch.linalg.svdvals(base), rtol=1e-12, atol=1e-12)
    assert torch.linalg.vector_norm(merged - base) > 0.01
    assert all(0 <= coefficient <= 1.5 for coefficient in coefficients)
