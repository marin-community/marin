# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import safetensors.torch
import torch
from marin.merging.checkpoint import CheckpointSource

from experiments.weight_merging.identity_restricted import restricted_merge as rm
from experiments.weight_merging.identity_restricted import weight_delta_audit as wda

SLICE = [f"model.layers.18.self_attn.{p}_proj.weight" for p in "qkvo"]
FROZEN = [
    "model.layers.18.mlp.router.bias",
    "model.layers.18.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.embed_tokens.weight",
]
ANCHOR_SHARDS = {
    "model-00001-of-00003.safetensors": [SLICE[0], SLICE[1], FROZEN[0]],
    "model-00002-of-00003.safetensors": [SLICE[2], SLICE[3]],
    "model-00003-of-00003.safetensors": FROZEN[1:],
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checkpoint(root, shards: dict[str, dict[str, torch.Tensor]], extra: dict[str, bytes]) -> None:
    root.mkdir()
    weight_map = {}
    for shard, tensors in shards.items():
        (root / shard).write_bytes(safetensors.torch.save(tensors, metadata={"format": "pt"}))
        weight_map.update(dict.fromkeys(tensors, shard))
    (root / rm.INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    for name, data in extra.items():
        (root / name).write_bytes(data)


@pytest.fixture
def parents(tmp_path, monkeypatch):
    gen = torch.Generator().manual_seed(0)
    donor = {name: torch.randn(4, 8, generator=gen) for name in SLICE + FROZEN}
    donor["model.layers.18.mlp.router.bias"] = torch.randn(8, generator=gen)
    anchor = {
        name: value.clone() if name.endswith("router.bias") else (value + 0.05 * torch.randn(value.shape, generator=gen))
        for name, value in donor.items()
    }
    anchor = {n: v if n.endswith("router.bias") else v.to(torch.bfloat16) for n, v in anchor.items()}
    write_checkpoint(
        tmp_path / "anchor",
        {shard: {n: anchor[n] for n in names} for shard, names in ANCHOR_SHARDS.items()},
        {"config.json": b'{"model_type": "snowball"}', "chat_template.jinja": b"{{ messages }}"},
    )
    write_checkpoint(
        tmp_path / "donor",
        {"model-00001-of-00001.safetensors": donor},
        {".cache_metadata.json": json.dumps({"source_revision": "donor-rev"}).encode()},
    )
    monkeypatch.setitem(rm.VARIANTS, "test-l18", range(18, 19))
    index_sha = sha((tmp_path / "anchor" / rm.INDEX_NAME).read_bytes())
    return tmp_path, anchor, donor, index_sha


def build(tmp_path, index_sha, out="merged"):
    return rm.build(
        "test-l18",
        str(tmp_path / out),
        "code-rev",
        CheckpointSource(str(tmp_path / "anchor"), "anchor-rev"),
        CheckpointSource(str(tmp_path / "donor"), "donor-rev"),
        index_sha,
    )


def test_merge_changes_only_the_slice(parents):
    tmp_path, anchor, donor, index_sha = parents
    receipt = build(tmp_path, index_sha)
    merged = {}
    for shard in ANCHOR_SHARDS:
        merged.update(safetensors.torch.load((tmp_path / "merged" / shard).read_bytes()))
    for name in SLICE:
        assert torch.equal(merged[name], donor[name].to(torch.bfloat16))
    for name in FROZEN:
        assert merged[name].dtype == anchor[name].dtype and torch.equal(merged[name], anchor[name])
    shard3 = "model-00003-of-00003.safetensors"
    assert (tmp_path / "merged" / shard3).read_bytes() == (tmp_path / "anchor" / shard3).read_bytes()
    assert receipt["objects"][shard3]["kind"] == "copied"
    for name in ("config.json", "chat_template.jinja", rm.INDEX_NAME):
        assert (tmp_path / "merged" / name).read_bytes() == (tmp_path / "anchor" / name).read_bytes()
    assert receipt["header_counts"] == {"tensors": 8, "dtypes": {"BF16": 7, "F32": 1}}
    assert receipt["slice"] == sorted(SLICE) and receipt["slice_unchanged_after_cast"] == []
    for name, row in receipt["tensors"].items():
        expected = merged[name].view(torch.uint8) if merged[name].dtype == torch.bfloat16 else merged[name]
        assert row["export_sha256"] == sha(expected.numpy().tobytes())
    assert json.loads((tmp_path / "merged" / rm.RECEIPT_NAME).read_bytes())["tensors"] == receipt["tensors"]


def test_merge_refuses_wrong_pins_and_nonempty_output(parents):
    tmp_path, _, _, index_sha = parents
    with pytest.raises(ValueError, match="index differs"):
        build(tmp_path, "0" * 64)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(ValueError, match="must be empty"):
        build(tmp_path, index_sha, out="busy")
    (tmp_path / "donor" / ".cache_metadata.json").write_text(json.dumps({"source_revision": "other"}))
    with pytest.raises(ValueError, match="pinned revision"):
        build(tmp_path, index_sha, out="merged2")


def source_receipt(tmp_path, monkeypatch) -> object:
    records = wda.shard_records(str(tmp_path / "anchor"))
    fs, _ = wda.filesystem_for(str(tmp_path / "anchor"))
    receipt = {
        "tensors": {n: {"master_sha256": "m", "native_sha256": wda.payload_sha256(fs, r)} for n, r in records.items()}
    }
    path = tmp_path / "source-receipt.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(wda, "SOURCE_RECEIPT_SHA256", sha(path.read_bytes()))
    monkeypatch.setattr(wda, "TENSOR_COUNT", len(records))
    return path


def export_only(tmp_path, monkeypatch, slice_set, out="merged", edit_receipt=None):
    receipt_path = source_receipt(tmp_path, monkeypatch)
    rehash_path = tmp_path / f"rehash-{out}.json"
    rehash_path.write_text(json.dumps(wda.rehash(str(tmp_path / out), str(tmp_path / "anchor"), slice_set)))
    merge_path = tmp_path / out / rm.RECEIPT_NAME
    if edit_receipt:
        merge = json.loads(merge_path.read_bytes())
        edit_receipt(merge)
        merge_path.write_text(json.dumps(merge))
    return wda.audit(receipt_path, rehash_path, slice_set, merge_path=merge_path)


def test_export_only_audit_passes_on_the_merge(parents, monkeypatch):
    tmp_path, _, _, index_sha = parents
    build(tmp_path, index_sha)
    result = export_only(tmp_path, monkeypatch, set(SLICE))
    assert result["verdict"] == "PASS", result["failures"]
    assert result["mode"] == "export-only"
    assert result["counts"]["frozen_equal"] == len(FROZEN) and result["counts"]["slice_export_unchanged"] == 0
    assert "master_equal" not in result["tensors"][FROZEN[0]]
    assert result["tensors"][SLICE[0]]["fraction_changed"] > 0


def test_export_only_audit_fails_on_a_changed_frozen_tensor(parents, monkeypatch):
    tmp_path, _, _, index_sha = parents
    build(tmp_path, index_sha)
    # Audit a narrower slice than was merged: the other three merged tensors are frozen and changed.
    narrow = {SLICE[0]}
    result = export_only(tmp_path, monkeypatch, narrow, edit_receipt=lambda m: m.update(slice=sorted(narrow)))
    assert result["verdict"] == "FAIL"
    assert sorted(f["tensor"] for f in result["failures"]) == sorted(SLICE[1:])
    assert result["frozen_groups"]["attn_qkvo"]["frozen_changed"] == 3


def test_export_only_audit_fails_when_the_receipt_disagrees_with_the_rehash(parents, monkeypatch):
    tmp_path, _, _, index_sha = parents
    build(tmp_path, index_sha)
    lie = FROZEN[3]

    def forge(merge):
        merge["tensors"][lie]["export_sha256"] = sha(b"forged")

    result = export_only(tmp_path, monkeypatch, set(SLICE), edit_receipt=forge)
    assert result["failures"] == [{"tensor": lie, "reason": "recomputed export hash differs from the producer receipt"}]


def test_export_only_audit_lists_a_slice_tensor_equal_to_the_anchor(parents, monkeypatch):
    tmp_path, anchor, _, _ = parents
    # The donor equals the anchor on one slice tensor: listed by name, not fatal.
    donor_path = tmp_path / "donor" / "model-00001-of-00001.safetensors"
    donor = safetensors.torch.load(donor_path.read_bytes())
    donor[SLICE[2]] = anchor[SLICE[2]].float()
    donor_path.write_bytes(safetensors.torch.save(donor, metadata={"format": "pt"}))
    receipt = build(tmp_path, sha((tmp_path / "anchor" / rm.INDEX_NAME).read_bytes()))
    assert receipt["slice_unchanged_after_cast"] == [SLICE[2]]
    result = export_only(tmp_path, monkeypatch, set(SLICE))
    assert result["verdict"] == "PASS"
    assert result["slice_export_unchanged"] == [SLICE[2]]


def test_merge_receipt_slice_must_match_the_audit(parents, monkeypatch):
    tmp_path, _, _, index_sha = parents
    build(tmp_path, index_sha)
    with pytest.raises(SystemExit, match="slice differs"):
        export_only(tmp_path, monkeypatch, {SLICE[0]})
