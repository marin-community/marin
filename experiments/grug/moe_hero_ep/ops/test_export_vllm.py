# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import load_file

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig, export
from experiments.grug.moe_hero_ep.weights import metadata_hash


def native_fixture(root: str, *, master: bool = False) -> tuple[ExportConfig, Transformer]:
    config = GrugModelConfig(
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
        sconv=True,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(7))
        # Distinguish arrays and positions, including layer, expert and projection.
        leaves, structure = jax.tree.flatten(model)
        model = jax.tree.unflatten(
            structure,
            [
                jnp.arange(x.size, dtype=jnp.float32).reshape(x.shape) / 64 + i if eqx.is_inexact_array(x) else x
                for i, x in enumerate(leaves)
            ],
        )
        model = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, model, jnp.full((2, 4), 9.0))
        pending = jnp.array([[1, -2, 4, -1], [-4, 1, 2, 7]], dtype=jnp.float32)
        state = {"params": model, "pending_qb_betas": pending}
        if master:
            state["master_params"] = model
            state["params"] = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), model)
            state = {"train_state": state}
        save_checkpoint(state, step=17, checkpoint_path=root, is_temporary=False)
    metadata = json.loads((StoragePath(root) / "metadata.json").read_text())
    return (
        ExportConfig(
            root, metadata_hash(metadata), config, root + "-export", "44a4188c197a4b5a314e40cc653f150fa9687dcf"
        ),
        model,
    )


def reload_export(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in set(index["weight_map"].values()):
        shard = load_file(root / filename)
        assert all(index["weight_map"][name] == filename for name in shard)
        assert not tensors.keys() & shard.keys()
        tensors.update(shard)
    assert tensors.keys() == index["weight_map"].keys()
    assert index["metadata"]["total_size"] == sum(value.nbytes for value in tensors.values())
    return index, tensors


def test_native_export_preserves_all_weights_and_config(tmp_path):
    request, model = native_fixture(str(tmp_path / "checkpoint"), master=True)
    export(request)
    root = Path(request.destination)
    _, tensors = reload_export(root)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        # The established HF mapping is the oracle for unchanged ordinary weights.
        expected = model.to_state_dict()
        # Known centered negative betas replace the saved bias (9).
        for layer, bias in enumerate([[-0.5, 2.5, -3.5, 1.5], [5.5, 0.5, -0.5, -5.5]]):
            expected[f"model.layers.{layer}.mlp.router.bias"] = jnp.asarray(bias)
        # Check routed experts independently from the exporter and its bank mapping.
        bank = model.stacked_blocks.stacked.mlp.expert_mlp
        for projection, values in [("gate_proj", bank.w_gate), ("up_proj", bank.w_up), ("down_proj", bank.w_down)]:
            for layer in range(2):
                expected.pop(f"model.layers.{layer}.mlp.experts.{projection}.weight")
                for expert in range(4):
                    expected[f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"] = values[layer, expert].T
        assert tensors.keys() == expected.keys()
        for name, value in expected.items():
            wanted = np.asarray(jax.sharding.reshard(value.astype(jnp.bfloat16), P()))
            actual = tensors[name]
            assert actual.shape == wanted.shape and actual.dtype == wanted.dtype, name
            assert actual.tobytes() == wanted.tobytes(), name
    expected_config = request.model.to_hf_config(request.model.vocab_size).to_dict()
    assert json.loads((root / "config.json").read_text()) == json.loads(json.dumps(expected_config))


def test_export_recovery_verifies_shards_and_preserves_completed_output(tmp_path, monkeypatch):
    request, _ = native_fixture(str(tmp_path / "checkpoint"))
    root = Path(request.destination)
    original_upload = StoragePath.upload_from

    def interrupt_upload(path, local_path, **kwargs):
        if path.name == "model-layer-000.safetensors":
            path.write_bytes(b"interrupted")
            raise OSError("upload interrupted")
        original_upload(path, local_path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(StoragePath, "upload_from", interrupt_upload)
        with pytest.raises(OSError, match="upload interrupted"):
            export(request)
    assert not (root / "export-manifest.json").exists()
    global_shard = root / "model-global.safetensors"
    # A resume with a different config must not reuse existing shards.
    with pytest.raises(FileExistsError):
        export(dataclasses.replace(request, model=dataclasses.replace(request.model, qk_mult=1.5)))

    # Corruption with unchanged size must stop resume and preserve the damaged object.
    original = global_shard.read_bytes()
    corrupted = original[:-1] + bytes([original[-1] ^ 1])
    global_shard.write_bytes(corrupted)
    with pytest.raises(ValueError, match="integrity"):
        export(request)
    assert global_shard.read_bytes() == corrupted
    assert not (root / "export-manifest.json").exists()
    global_shard.write_bytes(original)
    committed_mtime = global_shard.stat().st_mtime_ns

    export(request)
    reload_export(root)
    assert global_shard.stat().st_mtime_ns == committed_mtime
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        export(request)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
