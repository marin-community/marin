# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from contextlib import contextmanager
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

from experiments.grug.moe_hero_ep.model import GrugModelConfig, GrugMoeHfConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig, export
from experiments.grug.moe_hero_ep.weights import metadata_hash


def native_fixture(root: str, *, master: bool = False, hidden_dim: int = 16) -> tuple[ExportConfig, Transformer]:
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim * 3 // 2,
        shared_expert_intermediate_dim=hidden_dim * 2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden_dim // 2,
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
        bank = model.stacked_blocks.stacked.mlp.expert_mlp
        # Values distinguish layer, global expert, projection, row and column.
        model = eqx.tree_at(
            lambda m: (
                m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
                m.stacked_blocks.stacked.mlp.expert_mlp.w_up,
                m.stacked_blocks.stacked.mlp.expert_mlp.w_down,
            ),
            model,
            tuple(
                jnp.reshape(jnp.arange(x.size, dtype=jnp.float32), x.shape) / 64 + offset
                for x, offset in [(bank.w_gate, 1), (bank.w_up, -2), (bank.w_down, 3)]
            ),
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


@pytest.mark.parametrize("layout", ["params", "wrapped-master", "manifestless-ocdbt"])
def test_native_export_preserves_expert_identity_and_effective_bias(tmp_path, layout):
    request, model = native_fixture(str(tmp_path / "checkpoint"), master=layout == "wrapped-master")
    if layout == "manifestless-ocdbt":
        (tmp_path / "checkpoint" / "manifest.json").unlink()
    export(request)
    root = Path(request.destination)
    index, tensors = reload_export(root)
    assert all(str(value.dtype) == "bfloat16" for value in tensors.values())
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        bank = model.stacked_blocks.stacked.mlp.expert_mlp
        for layer in range(2):
            for expert in range(4):
                for projection, values in [
                    ("gate_proj", bank.w_gate),
                    ("up_proj", bank.w_up),
                    ("down_proj", bank.w_down),
                ]:
                    expected = np.asarray(jax.sharding.reshard(values[layer, expert].T.astype(jnp.bfloat16), P()))
                    actual = tensors[f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"]
                    assert actual.shape == expected.shape
                    assert actual.tobytes() == expected.tobytes()
    # Known centered negative betas. The saved router bias (9) must not survive or be added.
    for layer, expected in enumerate([[-0.5, 2.5, -3.5, 1.5], [5.5, 0.5, -0.5, -5.5]]):
        np.testing.assert_array_equal(tensors[f"model.layers.{layer}.mlp.router.bias"], expected)
    hf_config = json.loads((root / "config.json").read_text())
    decoded = GrugModelConfig.from_hf_config(GrugMoeHfConfig(**hf_config))
    assert (decoded.hidden_dim, decoded.intermediate_dim, decoded.latent_dim, decoded.sconv) == (16, 24, 8, True)
    manifest = json.loads((root / "export-manifest.json").read_text())
    assert manifest["authoritative_weight_tree"] == ("master_params" if layout == "wrapped-master" else "params")
    assert manifest["tensor_count"] == len(tensors)
    assert manifest["total_safetensors_bytes"] == sum(
        (root / name).stat().st_size for name in set(index["weight_map"].values())
    )


def test_interrupted_export_resumes_and_completed_export_is_preserved(tmp_path, monkeypatch):
    request, _ = native_fixture(str(tmp_path / "checkpoint"))
    root = Path(request.destination)
    original_open = StoragePath.open

    @contextmanager
    def interrupt_upload(path, mode="rb", **kwargs):
        if str(path) == str(root / "model-layer-000.safetensors") and mode == "wb":
            with original_open(path, mode, **kwargs) as handle:
                handle.write(b"interrupted")
            raise OSError("upload interrupted")
        with original_open(path, mode, **kwargs) as handle:
            yield handle

    with monkeypatch.context() as scoped:
        scoped.setattr(StoragePath, "open", interrupt_upload)
        with pytest.raises(OSError, match="upload interrupted"):
            export(request)
    assert not (root / "export-manifest.json").exists()
    global_shard = root / "model-global.safetensors"
    committed_mtime = global_shard.stat().st_mtime_ns
    # A resume with a different config must not reuse existing shards.
    with pytest.raises(FileExistsError):
        export(dataclasses.replace(request, model=dataclasses.replace(request.model, qk_mult=1.5)))
    export(request)
    reload_export(root)
    assert global_shard.stat().st_mtime_ns == committed_mtime
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        export(request)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_resume_detects_same_size_shard_corruption(tmp_path):
    request, _ = native_fixture(str(tmp_path / "checkpoint"))
    export(request)
    root = Path(request.destination)
    (root / "export-manifest.json").unlink()  # Simulate interruption just before completion publication.
    shard = root / "model-global.safetensors"
    contents = bytearray(shard.read_bytes())
    contents[-1] ^= 1
    shard.write_bytes(contents)
    corrupted_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="integrity"):
        export(request)
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == corrupted_hash
    assert not (root / "export-manifest.json").exists()
