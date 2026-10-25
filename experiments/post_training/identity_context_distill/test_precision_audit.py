import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.models.snowball import SnowballConfig
from safetensors.numpy import load_file, save_file

from experiments.post_training.identity_context_distill import precision_source_audit
from experiments.post_training.identity_context_distill.precision_audit import (
    audit_initialized_masters,
    audit_precision_export,
)
from experiments.post_training.identity_context_distill.precision_source_audit import (
    INDEX,
    ROUTER_KEYS,
    SOURCE_FILES,
    audit_precision_source,
    checkpoint_records,
)


def test_source_audit_imports_without_sft_factory(tmp_path):
    standalone = tmp_path / "precision_source_audit.py"
    shutil.copyfile(Path(__file__).with_name("precision_source_audit.py"), standalone)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "sys.modules['experiments.post_training.identity_context_distill.sft'] = None; "
            "runpy.run_path(sys.argv[1])",
            str(standalone),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def tiny_checkpoint(path):
    path.mkdir()
    tensors = {name: np.full(256, 0.1234567, dtype=np.float32) for name in ROUTER_KEYS}
    tensors.update({f"weight.{index}": np.ones(2, dtype=jnp.bfloat16) for index in range(476)})
    save_file(tensors, str(path / "weights.safetensors"))
    (path / INDEX).write_text(
        json.dumps(
            {
                "weight_map": {name: "weights.safetensors" for name in tensors},
                "metadata": {"total_size": sum(array.nbytes for array in tensors.values())},
            }
        )
    )
    (path / "config.json").write_text(
        SnowballConfig(max_seq_len=262144, qk_mult=1.75).to_hf_config(128256).to_json_string()
    )
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "chat_template.jinja").write_text("synthetic fixture")
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def test_full_tiny_checkpoint_byte_audit_and_metadata_failure(tmp_path):
    source, exported = tmp_path / "source", tmp_path / "export"
    hashes = tiny_checkpoint(source)
    shutil.copytree(source, exported)
    audit_precision_source(str(source), hashes, str(tmp_path / "source-receipt.json"))
    audit_precision_export(str(source), str(exported), hashes)
    receipt = json.loads((exported / "precision-audit.json").read_text())
    assert receipt["tensor_count"] == len(receipt["tensors"]) == 502
    assert receipt["audit_io"]["payload_bytes_read"] == 2 * sum(row["bytes"] for row in receipt["tensors"].values())
    (exported / "chat_template.jinja").write_text("changed")
    with pytest.raises(ValueError, match="chat template"):
        audit_precision_export(str(source), str(exported), hashes)


def test_tensor_byte_change_fails_without_success_receipt(tmp_path):
    source, exported = tmp_path / "source", tmp_path / "export"
    hashes = tiny_checkpoint(source)
    shutil.copytree(source, exported)
    record = checkpoint_records(str(exported))["weight.0"]
    with open(record.file_path, "r+b") as stream:
        stream.seek(record.byte_start)
        stream.write(b"\x00\x00")
    with pytest.raises(ValueError, match="tensor bytes changed"):
        audit_precision_export(str(source), str(exported), hashes)
    assert not (exported / "precision-audit.json").exists()


def test_source_hashes_are_independent_of_read_chunk_size(tmp_path, monkeypatch):
    source = tmp_path / "source"
    hashes = tiny_checkpoint(source)
    large_receipt, small_receipt = tmp_path / "large.json", tmp_path / "small.json"
    audit_precision_source(str(source), hashes, str(large_receipt))
    monkeypatch.setattr(precision_source_audit, "CHUNK_BYTES", 256)
    audit_precision_source(str(source), hashes, str(small_receipt))
    large, small = json.loads(large_receipt.read_text()), json.loads(small_receipt.read_text())
    assert large["tensors"] == small["tensors"]
    assert large["audit_io"]["chunk_bytes"] == 64 * 1024**2
    assert small["audit_io"]["chunk_bytes"] == 256
    arrays = load_file(str(source / "weights.safetensors"))
    for name, array in arrays.items():
        assert large["tensors"][name]["native_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
        assert large["tensors"][name]["master_sha256"] == hashlib.sha256(array.astype(np.float32).tobytes()).hexdigest()
    assert (
        large["audit_io"]["payload_bytes_read"]
        == small["audit_io"]["payload_bytes_read"]
        == sum(array.nbytes for array in arrays.values())
    )


def test_master_audit_detects_low_bits_hidden_by_bf16_export(tmp_path):
    source = tmp_path / "source"
    hashes = tiny_checkpoint(source)
    source_receipt = str(tmp_path / "source-receipt.json")
    audit_precision_source(str(source), hashes, source_receipt)
    tensors = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in load_file(str(source / "weights.safetensors")).items()
    }
    with jax.set_mesh(jax.make_mesh((1,), ("data",))):
        audit_initialized_masters(tensors, source_receipt, str(tmp_path / "master-ok.json"))
        changed = jnp.nextafter(tensors["weight.0"], jnp.float32(2))
        np.testing.assert_array_equal(changed.astype(jnp.bfloat16), tensors["weight.0"].astype(jnp.bfloat16))
        tensors["weight.0"] = changed
        with pytest.raises(ValueError, match="master bits differ"):
            audit_initialized_masters(tensors, source_receipt, str(tmp_path / "master-bad.json"))
    assert not (tmp_path / "master-bad.json").exists()


@pytest.mark.parametrize("damage", ["bounds", "wrong_dtype", "missing_index_key", "huge_header"])
def test_header_or_index_damage_fails(tmp_path, damage):
    path = tmp_path / "source"
    tiny_checkpoint(path)
    shard = path / "weights.safetensors"
    raw = shard.read_bytes()
    length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + length])
    if damage == "huge_header":
        shard.write_bytes((2**40).to_bytes(8, "little") + raw[8:])
    elif damage == "missing_index_key":
        index = json.loads((path / INDEX).read_text())
        del index["weight_map"]["weight.0"]
        (path / INDEX).write_text(json.dumps(index))
    else:
        if damage == "bounds":
            header["weight.0"]["data_offsets"][1] += 2
        else:
            header["weight.0"]["dtype"] = "I16"
        encoded = json.dumps(header).encode()
        shard.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8 + length :])
    with pytest.raises(ValueError):
        checkpoint_records(str(path))
