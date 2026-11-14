# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import shutil

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from safetensors.numpy import save_file

from levanter.compat import hf_export_audit as audit
from levanter.trainer_state import TrainerState


class MasterModel(eqx.Module):
    tensors: dict[str, jax.Array]

    def to_state_dict(self):
        return self.tensors


def write_checkpoint(root, arrays):
    root.mkdir(exist_ok=True)
    save_file(arrays, str(root / "weights.safetensors"))
    (root / audit.INDEX).write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(x.nbytes for x in arrays.values())},
                "weight_map": {name: "weights.safetensors" for name in arrays},
            }
        )
    )


@pytest.mark.parametrize(
    "corruption", [None, "weight", "bias", "metadata", "embedded", "other_key", "receipt", "layout"]
)
def test_trained_export_uses_cast_masters_and_pinned_frozen_values(tmp_path, monkeypatch, corruption):
    # Tiny chunks exercise repeated seek/read/cast/hash boundaries, including BF16.
    monkeypatch.setattr(audit, "CHUNK_BYTES", 16)
    source, exported = tmp_path / "source", tmp_path / "export"
    arrays = {
        "weight": np.arange(20, dtype=np.float32).reshape(4, 5).astype(jnp.bfloat16),
        "bias": np.array([0.1234567, -0.2345678, 0.3456789], dtype=np.float32),
    }
    write_checkpoint(source, arrays)
    for name, contents in {
        "config.json": '{"hidden_size": 5}',
        "generation_config.json": '{"temperature": 1}',
        "tokenizer.json": '{"vocab": {"a": 0}}',
        "tokenizer_config.json": '{"bos_token": "a"}',
        "chat_template.jinja": "{{ messages }}",
    }.items():
        (source / name).write_text(contents)
    metadata_files = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (*metadata_files, audit.INDEX)
    }
    layout = {name: audit.TensorSpec(values.shape, values.dtype) for name, values in arrays.items()}
    source_receipt = tmp_path / "source-audit.json"
    audit.audit_source_checkpoint(
        str(source), expected_metadata_hashes=hashes, expected_layout=layout, receipt_path=str(source_receipt)
    )
    shutil.copytree(source, exported)
    masters = {name: values.astype(np.float32) for name, values in arrays.items()}
    masters["weight"] += np.float32(0.1234567)
    if corruption == "bias":
        masters["bias"] += np.float32(0.001)
    saved = {name: values.astype(arrays[name].dtype) for name, values in masters.items()}
    if corruption == "weight":
        saved["weight"] = arrays["weight"]
    write_checkpoint(exported, saved)
    config = {"bos_token": "a", "chat_template": "{{ messages }}"}
    if corruption == "embedded":
        config["chat_template"] += " "
    elif corruption == "other_key":
        config["bos_token"] = "changed"
    (exported / "tokenizer_config.json").write_text(json.dumps(config))
    if corruption == "metadata":
        (exported / "chat_template.jinja").write_text("changed")
    receipt_sha = hashlib.sha256(source_receipt.read_bytes()).hexdigest()
    if corruption == "receipt":
        source_receipt.write_text(source_receipt.read_text() + " ")
    if corruption == "layout":
        layout["bias"] = audit.TensorSpec((3,), np.dtype(jnp.bfloat16))
    state = TrainerState.init(
        optimizer=optax.sgd(0.1),
        model=MasterModel({name: jnp.asarray(x) for name, x in masters.items()}),
        key=jax.random.PRNGKey(0),
        mp=jmp.get_policy("p=f32,c=bf16,o=f32"),
        is_trainable=True,
    )
    state = dataclasses.replace(state, step=jnp.asarray(16))
    with jax.sharding.set_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))):
        kwargs = dict(
            source_root=str(source),
            source_receipt_path=str(source_receipt),
            source_receipt_sha256=receipt_sha,
            expected_metadata_hashes=hashes,
            expected_layout=layout,
            frozen_keys=frozenset({"bias"}),
            metadata_files=metadata_files,
            allow_embedded_chat_template=True,
        )
        if corruption is None:
            audit.audit_trained_export(state, str(exported), **kwargs)
            receipt = json.loads((exported / "trained-precision-audit.json").read_text())
            assert receipt["tensor_count"] == 2 and receipt["completed_updates"] == 16
            assert receipt["accepted_metadata_additions"] == ["tokenizer_config.json.chat_template"]
            for name, values in saved.items():
                assert receipt["tensors"][name]["export_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
                assert receipt["tensors"][name]["frozen"] == (name == "bias")
        else:
            with pytest.raises(ValueError):
                audit.audit_trained_export(state, str(exported), **kwargs)
            assert not (exported / "trained-precision-audit.json").exists()


@pytest.mark.parametrize("corruption", ["overlap", "truncation", "index"])
def test_checkpoint_records_rejects_corrupt_tensor_coverage(tmp_path, corruption):
    arrays = {"a": np.arange(4, dtype=np.float32), "b": np.arange(4, dtype=np.float32)}
    write_checkpoint(tmp_path, arrays)
    path = tmp_path / "weights.safetensors"
    contents = path.read_bytes()
    length = int.from_bytes(contents[:8], "little")
    header = json.loads(contents[8 : 8 + length])
    if corruption == "overlap":
        header["b"]["data_offsets"] = header["a"]["data_offsets"]
        encoded = json.dumps(header).encode()
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + contents[8 + length :])
    elif corruption == "truncation":
        path.write_bytes(contents[:-1])
    else:
        index = json.loads((tmp_path / audit.INDEX).read_text())
        index["weight_map"].pop("b")
        (tmp_path / audit.INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError):
        audit.checkpoint_records(str(tmp_path))
