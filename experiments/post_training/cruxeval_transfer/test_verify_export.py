# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import save_file

from experiments.post_training.cruxeval_transfer.verify_export import verify_export


def test_export_audit_rejects_changed_biases_and_incomplete_shards(tmp_path):
    name = "model.layers.0.mlp.router.bias"
    biases = np.array([0.25, -0.5], dtype=np.float32)
    parent, export = tmp_path / "parent", tmp_path / "export"
    index = {"weight_map": {name: "model.safetensors"}, "metadata": {"total_size": biases.nbytes}}
    for directory in (parent, export):
        directory.mkdir()
        save_file({name: biases}, directory / "model.safetensors")
        for filename, value in {
            "model.safetensors.index.json": index,
            "config.json": {"model_type": "grug_moe"},
            "tokenizer.json": {"model": {"vocab": {"a": 1}}},
            "tokenizer_config.json": {},
            "generation_config.json": {},
        }.items():
            (directory / filename).write_text(json.dumps(value))
        (directory / "chat_template.jinja").write_text("{{ messages }}")
    paths = StoragePath(str(export)), StoragePath(str(parent))
    report = verify_export(*paths)
    assert report["router_bias_tensors"] == 1
    assert report["tensor_bytes"] == biases.nbytes

    save_file({name: biases + 1}, export / "model.safetensors")
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        verify_export(*paths)

    save_file({name: biases}, export / "model.safetensors")
    shard = export / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(AssertionError):
        verify_export(*paths)
