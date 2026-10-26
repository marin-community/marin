# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import struct

import numpy as np
import pytest

from experiments.weight_merging.identity_restricted import weight_delta_audit as wda

SLICE = {"model.layers.18.self_attn.q_proj.weight", "model.layers.18.self_attn.k_proj.weight"}
FROZEN = (
    "model.layers.0.mlp.router.bias",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)


def bf16(values: np.ndarray) -> bytes:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def write_export(root, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    root.mkdir()
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[:3], "model-00002-of-00002.safetensors": names[3:]}
    weight_map = {}
    for shard, members in shards.items():
        header, offset, payload = {"__metadata__": {"format": "pt"}}, 0, b""
        for name in members:
            dtype, shape, data = tensors[name]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
            offset += len(data)
            payload += data
            weight_map[name] = shard
        encoded = json.dumps(header).encode()
        (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def source_tensors() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for name in sorted(SLICE | set(FROZEN)):
        if name.endswith("router.bias"):
            out[name] = ("F32", [4], rng.standard_normal(4).astype(np.float32).tobytes())
        else:
            out[name] = ("BF16", [4, 8], bf16(rng.standard_normal((4, 8))))
    return out


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Build a source export, its receipt, and a callable that audits an arm built from edits."""
    source = source_tensors()
    write_export(tmp_path / "source", source)
    receipt = {
        "tensors": {n: {"master_sha256": sha(b"m" + d), "native_sha256": sha(d)} for n, (_, _, d) in source.items()}
    }
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(wda, "SOURCE_RECEIPT_SHA256", sha(receipt_path.read_bytes()))
    monkeypatch.setattr(wda, "TENSOR_COUNT", len(source))

    def run(export_edits=(), master_edits=(), updates=16, expected_updates=16, tamper_audit=None):
        arm = dict(source)
        for name in export_edits:
            dtype, shape, data = arm[name]
            values = np.frombuffer(data, np.uint16 if dtype == "BF16" else np.float32).copy()
            if dtype == "BF16":
                values[0] ^= 1
            else:
                values[0] += 1.0
            arm[name] = (dtype, shape, values.tobytes())
        root = tmp_path / f"arm-{len(list(tmp_path.iterdir()))}"
        write_export(root, arm)
        trained = {
            "completed_updates": updates,
            "source_receipt_sha256": wda.SOURCE_RECEIPT_SHA256,
            "tensors": {
                n: {
                    "master_sha256": (
                        sha(b"trained" + d) if n in set(master_edits) else receipt["tensors"][n]["master_sha256"]
                    ),
                    "export_sha256": sha(d),
                }
                for n, (_, _, d) in arm.items()
            },
        }
        if tamper_audit:
            trained["tensors"][tamper_audit]["export_sha256"] = sha(b"forged")
        trained_path = root / "trained-precision-audit.json"
        trained_path.write_text(json.dumps(trained))
        rehash_path = root / "rehash.json"
        rehash_path.write_text(json.dumps(wda.rehash(str(root), str(tmp_path / "source"), SLICE)))
        return wda.audit(receipt_path, rehash_path, SLICE, trained_path=trained_path, completed_updates=expected_updates)

    return run


def test_pass_only_slice_changed(case):
    result = case(export_edits=SLICE, master_edits=SLICE)
    assert result["verdict"] == "PASS"
    assert result["counts"]["frozen_equal"] == len(FROZEN)
    assert result["counts"]["slice_master_changed"] == 2
    stats = result["tensors"]["model.layers.18.self_attn.q_proj.weight"]
    assert stats["fraction_changed"] == pytest.approx(1 / 32)
    assert stats["max_abs_delta"] > 0


def test_pass_reports_slice_export_unchanged_as_count(case):
    result = case(export_edits=["model.layers.18.self_attn.q_proj.weight"], master_edits=SLICE)
    assert result["verdict"] == "PASS"
    assert result["counts"]["slice_export_unchanged"] == 1


def test_fail_frozen_router_bias_changed(case):
    bias = "model.layers.0.mlp.router.bias"
    result = case(export_edits=[*SLICE, bias], master_edits=[*SLICE, bias])
    assert result["verdict"] == "FAIL"
    assert result["frozen_groups"]["router_bias"]["frozen_changed"] == 1
    assert [f["tensor"] for f in result["failures"]] == [bias]


def test_fail_frozen_master_changed_export_equal(case):
    other = "model.layers.0.self_attn.q_proj.weight"
    result = case(export_edits=SLICE, master_edits=[*SLICE, other])
    assert result["verdict"] == "FAIL"
    assert result["failures"] == [{"tensor": other, "reason": "frozen tensor changed: master"}]


def test_fail_slice_master_unchanged(case):
    idle = "model.layers.18.self_attn.k_proj.weight"
    result = case(export_edits=SLICE, master_edits=SLICE - {idle})
    assert result["verdict"] == "FAIL"
    assert result["failures"] == [{"tensor": idle, "reason": "slice master unchanged"}]


def test_fail_wrong_update_count(case):
    with pytest.raises(SystemExit, match="reports 15 updates"):
        case(export_edits=SLICE, master_edits=SLICE, updates=15)


def test_fail_in_job_receipt_disagrees_with_rehash(case):
    lie = "model.embed_tokens.weight"
    result = case(export_edits=SLICE, master_edits=SLICE, tamper_audit=lie)
    assert result["verdict"] == "FAIL"
    assert {"tensor": lie, "reason": "recomputed export hash differs from the producer receipt"} in result["failures"]
