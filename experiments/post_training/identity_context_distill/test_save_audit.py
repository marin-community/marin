# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.trainer_state import TrainerState
from safetensors.numpy import load_file, save_file

from experiments.post_training.identity_context_distill.precision_source_audit import audit_precision_source
from experiments.post_training.identity_context_distill.save_audit import audit_trained_export
from experiments.post_training.identity_context_distill.test_precision_audit import tiny_checkpoint


class MasterModel(eqx.Module):
    tensors: dict[str, jax.Array]

    def to_state_dict(self):
        return self.tensors


@pytest.mark.parametrize("corruption", [None, "weight", "bias", "metadata", "embedded_template", "other_key"])
def test_trained_export_compares_to_cast_masters_not_initial_weights(tmp_path, corruption):
    source, exported = tmp_path / "source", tmp_path / "export"
    hashes = tiny_checkpoint(source)
    fixture = json.loads((Path(__file__).parent / "testdata" / "tokenizer-export-r7.json").read_bytes())
    (source / "tokenizer_config.json").write_bytes(base64.b64decode(fixture["source-tokenizer_config.json"]))
    (source / "chat_template.jinja").write_bytes(base64.b64decode(fixture["source-chat_template.jinja"]))
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in hashes}
    source_receipt = tmp_path / "source-audit.json"
    audit_precision_source(str(source), hashes, str(source_receipt))
    shutil.copytree(source, exported)
    saved_tokenizer = json.loads(base64.b64decode(fixture["export-tokenizer_config.json"]))
    if corruption == "embedded_template":
        saved_tokenizer["chat_template"] += " "
    elif corruption == "other_key":
        saved_tokenizer["bos_token"] = "changed"
    (exported / "tokenizer_config.json").write_text(json.dumps(saved_tokenizer))
    arrays = load_file(str(source / "weights.safetensors"))
    masters = {name: values.astype(np.float32) for name, values in arrays.items()}
    masters["weight.0"] += np.float32(0.1234567)
    if corruption == "bias":
        masters["model.layers.0.mlp.router.bias"] += np.float32(0.001)
    saved = {name: values.astype(arrays[name].dtype) for name, values in masters.items()}
    if corruption == "weight":
        saved["weight.0"] = arrays["weight.0"]
    save_file(saved, str(exported / "weights.safetensors"))
    if corruption == "metadata":
        (exported / "chat_template.jinja").write_text("changed template")
    state = TrainerState.init(
        optimizer=optax.sgd(0.1),
        model=MasterModel({name: jnp.asarray(values) for name, values in masters.items()}),
        key=jax.random.PRNGKey(0),
        mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"),
        is_trainable=True,
    )
    state = dataclasses.replace(state, step=jnp.asarray(16))
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    with jax.sharding.set_mesh(mesh):
        if corruption is None:
            audit_trained_export(
                state,
                str(exported),
                source_root=str(source),
                source_receipt_path=str(source_receipt),
                source_receipt_sha256=hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
                expected_hashes=hashes,
            )
            receipt = json.loads((exported / "trained-precision-audit.json").read_text())
            assert receipt["tensor_count"] == 502 and receipt["completed_updates"] == 16
            assert receipt["accepted_metadata_additions"] == ["tokenizer_config.json.chat_template"]
            assert sum(value["frozen_bias"] for value in receipt["tensors"].values()) == 26
        else:
            with pytest.raises(ValueError):
                audit_trained_export(
                    state,
                    str(exported),
                    source_root=str(source),
                    source_receipt_path=str(source_receipt),
                    source_receipt_sha256=hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
                    expected_hashes=hashes,
                )
            assert not (exported / "trained-precision-audit.json").exists()
