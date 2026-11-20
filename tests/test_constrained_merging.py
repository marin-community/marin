# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import safetensors.torch
import torch
from fsspec.implementations.local import LocalFileSystem
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import TensorSupport, export_constrained, finalize_constrained_export
from marin.merging.isospectral import isospectral_merge


@pytest.fixture
def base_checkpoint(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    weights = {"experts": torch.arange(32, dtype=torch.bfloat16).reshape(8, 2, 2), "norm": torch.ones(8)}
    safetensors.torch.save_file(weights, base / "weights-01.safetensors", metadata={"format": "pt"})
    index = {"metadata": {"total_size": 96}, "weight_map": dict.fromkeys(weights, "weights-01.safetensors")}
    (base / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))
    (base / "config.json").write_text('{"unchanged": true}\n')
    (base / "tokenizer_config.json").write_text('{"chat_template": "exact bytes"}\n')
    return CheckpointReader(CheckpointSource(str(base), "pinned")), weights


@pytest.mark.parametrize("edit", [False, True])
def test_export_preserves_base_layout_and_audits_only_selected_rows(tmp_path, base_checkpoint, edit):
    reader, weights = base_checkpoint
    base = tmp_path / "base"
    output = tmp_path / "out"
    support = (TensorSupport("experts", (1, 3)),) if edit else ()
    audit = export_constrained(reader, str(output), support, 0.25, 0.25, lambda name, row, value: value + 2)
    source = CheckpointSource(str(output), "merged")
    with pytest.raises(ValueError, match="Incomplete merged checkpoint"):
        CheckpointReader(source)
    audit_path = tmp_path / "audit.json"
    finalize_constrained_export(audit, str(audit_path))
    actual = CheckpointReader(source).tensor("experts")
    expected = weights["experts"].clone()
    if edit:
        expected[[1, 3]] += 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stored_audit = json.loads(audit_path.read_text())
    assert stored_audit["changed_elements"] == (8 if edit else 0)
    assert stored_audit["total_elements"] == 40
    for file in base.iterdir():
        if file.suffix != ".safetensors" or not edit:
            assert (output / file.name).read_bytes() == file.read_bytes()
    before = (base / "weights-01.safetensors").read_bytes()
    header_end = 8 + int.from_bytes(before[:8], "little")
    assert (output / "weights-01.safetensors").read_bytes()[:header_end] == before[:header_end]
    assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in base.iterdir())


def test_over_budget_support_produces_no_partial_export(tmp_path, base_checkpoint):
    reader, _ = base_checkpoint
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Support exceeds"):
        export_constrained(
            reader,
            str(output),
            (TensorSupport("experts", (0, 1, 2)),),
            0.25,
            0.25,
            lambda name, row, value: value * 2,
        )
    assert not output.exists()


def test_failed_transform_leaves_checkpoint_unreadable(tmp_path, base_checkpoint):
    reader, _ = base_checkpoint
    output = tmp_path / "out"

    def transform(name, row, value):
        raise RuntimeError("Transform failed")

    with pytest.raises(RuntimeError):
        export_constrained(reader, str(output), (TensorSupport("experts", (1, 3)),), 0.25, 0.25, transform)
    with pytest.raises(ValueError, match="Incomplete merged checkpoint"):
        CheckpointReader(CheckpointSource(str(output), "failed"))


def test_export_rejects_stored_changes_above_change_cap(tmp_path, base_checkpoint):
    reader, _ = base_checkpoint
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Stored checkpoint exceeds scalar budget"):
        export_constrained(
            reader,
            str(output),
            (TensorSupport("experts", (0, 1, 2)),),
            0.75,
            0.25,
            lambda name, row, value: value + 2,
        )
    with pytest.raises(ValueError, match="Incomplete merged checkpoint"):
        CheckpointReader(CheckpointSource(str(output), "rejected"))


@pytest.mark.parametrize("failure", ["raise", "corrupt"])
def test_failed_audit_write_leaves_checkpoint_unreadable(tmp_path, base_checkpoint, monkeypatch, failure):
    reader, _ = base_checkpoint
    output = tmp_path / "out"
    audit = export_constrained(
        reader,
        str(output),
        (TensorSupport("experts", (1, 3)),),
        0.25,
        0.25,
        lambda name, row, value: value + 2,
    )
    audit_path = tmp_path / "audit.json"
    original_pipe = LocalFileSystem.pipe_file

    def pipe_file(fs, path, payload, **kwargs):
        if str(path) == str(audit_path):
            if failure == "raise":
                raise OSError("Audit write failed")
            payload = b"{}"
        return original_pipe(fs, path, payload, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "pipe_file", pipe_file)
    with pytest.raises(OSError if failure == "raise" else ValueError):
        finalize_constrained_export(audit, str(audit_path))
    with pytest.raises(ValueError, match="Incomplete merged checkpoint"):
        CheckpointReader(CheckpointSource(str(output), "unaudited"))


def test_export_allows_support_larger_than_changed_weight_cap(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    weights = torch.ones(4, 4)
    safetensors.torch.save_file({"weight": weights}, base / "weights.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
    reader = CheckpointReader(CheckpointSource(str(base), "pinned"))

    def edit_one_row(name, row, value):
        return value * 2 if row == 0 else value

    output = tmp_path / "out"
    audit = export_constrained(
        reader,
        str(output),
        (TensorSupport("weight", (0, 1, 2)),),
        0.75,
        0.25,
        edit_one_row,
    )
    audit_path = tmp_path / "audit.json"
    finalize_constrained_export(audit, str(audit_path))
    actual = CheckpointReader(CheckpointSource(str(output), "merged")).tensor("weight")
    expected = weights.clone()
    expected[0] *= 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert audit["supported_elements"] == 12
    assert audit["changed_elements"] == 4


@pytest.mark.parametrize("keep", [0.75, 0.9, 1.0])
def test_iso_composition_changes_frames_without_changing_spectrum(keep):
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(7, 4, dtype=torch.float64, generator=generator)
    donors = [base + 0.15 * torch.randn(7, 4, dtype=torch.float64, generator=generator) for _ in range(2)]
    merged, coefficients = isospectral_merge(base, donors, keep, torch.device("cpu"))
    torch.testing.assert_close(torch.linalg.svdvals(merged), torch.linalg.svdvals(base), rtol=1e-12, atol=1e-12)
    assert torch.linalg.vector_norm(merged - base) > 0.01
    assert all(0 <= coefficient <= 1.5 for coefficient in coefficients)
