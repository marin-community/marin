# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.tensorstore_serialization import TensorStoreWriteConfig
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import load_file

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.snowball_sft import model as sft
from experiments.grug.snowball_sft.export import export, restore_selected_weights
from experiments.grug.snowball_sft.validation_checkpoint import BestValidationCheckpoint


def test_selected_sft_export_preserves_frozen_router_and_weights(tmp_path):
    config = sft.GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=32,
        sliding_window=16,
        global_every=2,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = sft.Transformer.init(config, key=jax.random.PRNGKey(17))
        bias = jnp.array([[1, -2, 4, -1], [-4, 1, 2, 7]], dtype=jnp.float32)
        model = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, model, bias)
        selection = BestValidationCheckpoint(str(tmp_path / "best"), "eval/loss", TensorStoreWriteConfig())
        best = selection.consider(model, 20, 2.0)
    metadata = json.loads((StoragePath(best.path) / "metadata.json").read_text())
    request = ExportConfig(
        best.path,
        digest(metadata),
        draccus.decode(GrugModelConfig, draccus.encode(config)),
        str(tmp_path / "export"),
        "2cae063c877172844fe6ff90e94354cc08a273fa",
    )
    export(request)
    root = tmp_path / "export"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in set(index["weight_map"].values()):
        tensors.update(load_file(root / filename))
    with jax.set_mesh(mesh):
        expected = model.to_state_dict()
        bank = model.stacked_blocks.stacked.mlp.expert_mlp
        for projection, values in [("gate_proj", bank.w_gate), ("up_proj", bank.w_up), ("down_proj", bank.w_down)]:
            for layer in range(2):
                expected[f"model.layers.{layer}.mlp.experts.{projection}.weight"] = values[layer].swapaxes(-1, -2)
        assert tensors.keys() == expected.keys()
        for name, value in expected.items():
            wanted = np.asarray(value.astype(jnp.bfloat16))
            assert tensors[name].shape == wanted.shape, name
            assert tensors[name].tobytes() == wanted.tobytes(), name
        with pytest.raises(ValueError, match="pinned permanent"):
            restore_selected_weights(best.path, "wrong-digest", request.model, mesh)
