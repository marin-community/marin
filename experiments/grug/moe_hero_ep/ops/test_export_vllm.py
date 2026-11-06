# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from pathlib import Path
from threading import Lock

import equinox as eqx
import jax
import jax.numpy as jnp
import levanter.compat.hf_export as hf_export
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import load_file

import experiments.grug.moe_hero_ep.ops.export_cpu as cpu_export
import experiments.grug.moe_hero_ep.ops.export_vllm as hero_export
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig, export
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def test_export_refuses_log_scaling_before_writing(tmp_path):
    destination = tmp_path / "export"
    request = ExportConfig(
        checkpoint="unused",
        metadata_digest="unused",
        model=GrugModelConfig(vocab_size=32, log_scaling_n_floor=16),
        destination=str(destination),
        source_revision="unused",
    )

    with pytest.raises(ValueError, match="log_scaling_n_floor"):
        export(request)
    assert not destination.exists()


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
        ExportConfig(root, digest(metadata), config, root + "-export", "44a4188c197a4b5a314e40cc653f150fa9687dcf"),
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


def test_native_export_preserves_all_weights_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(hero_export, "DEFAULT_MAX_SHARD_SIZE", 4096)
    request, model = native_fixture(str(tmp_path / "checkpoint"), master=True)
    export(request)
    root = Path(request.destination)
    index, tensors = reload_export(root)
    assert len(set(index["weight_map"].values())) > 1
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
    expected_config["dtype"] = "bfloat16"
    assert json.loads((root / "config.json").read_text()) == json.loads(json.dumps(expected_config))


def test_export_recovery_verifies_shards_and_preserves_completed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(hero_export, "DEFAULT_MAX_SHARD_SIZE", 4096)
    request, _ = native_fixture(str(tmp_path / "checkpoint"))
    root = Path(request.destination)
    original_save = hf_export.save_file
    written = []

    def interrupt_write(tensors, path, **kwargs):
        written.append(path)
        if len(written) == 2:
            path.write_bytes(b"interrupted")
            raise OSError("write interrupted")
        original_save(tensors, path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(hf_export, "save_file", interrupt_write)
        with pytest.raises(OSError, match="write interrupted"):
            export(request)
    assert not (root / "export-manifest.json").exists()
    completed_shard = written[0]
    # A resume with a different config must not reuse existing shards.
    with pytest.raises(FileExistsError):
        export(dataclasses.replace(request, model=dataclasses.replace(request.model, qk_mult=1.5)))

    # Corruption with unchanged size must stop resume and preserve the damaged object.
    original = completed_shard.read_bytes()
    corrupted = original[:-1] + bytes([original[-1] ^ 1])
    completed_shard.write_bytes(corrupted)
    with pytest.raises(ValueError, match="integrity"):
        export(request)
    assert completed_shard.read_bytes() == corrupted
    assert not (root / "export-manifest.json").exists()
    completed_shard.write_bytes(original)
    committed_mtime = completed_shard.stat().st_mtime_ns

    resumed = dataclasses.replace(request, export_host_budget_bytes=1024**2, max_concurrent_shards=1)
    export(resumed)
    reload_export(root)
    assert completed_shard.stat().st_mtime_ns == committed_mtime
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        export(request)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_cpu_export_resumes_completed_shards_after_lowering_execution_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_export, "DEFAULT_MAX_SHARD_SIZE", 4096)
    monkeypatch.setattr(hero_export, "DEFAULT_MAX_SHARD_SIZE", 4096)
    native, _ = native_fixture(str(tmp_path / "checkpoint"), master=True)
    config = cpu_export.CpuExportConfig(
        checkpoint=native.checkpoint,
        metadata_digest=native.metadata_digest,
        model=native.model,
        destination=str(tmp_path / "cpu-export"),
        source_revision=native.source_revision,
        read_host_budget_bytes=2 * 1024**2,
        export_host_budget_bytes=2 * 1024**2,
        max_concurrent_shards=2,
        request_concurrency=4,
    )
    original_save = hf_export.save_file
    lock = Lock()
    write_count = 0

    def interrupt_second_write(tensors, path, **kwargs):
        nonlocal write_count
        with lock:
            write_count += 1
            number = write_count
        if number == 2:
            path.write_bytes(b"interrupted")
            raise OSError("write interrupted")
        original_save(tensors, path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(hf_export, "save_file", interrupt_second_write)
        with pytest.raises(OSError, match="write interrupted"):
            cpu_export.export(config)
    root = Path(config.destination)
    assert not (root / "export-manifest.json").exists()
    request = (root / "export-request.json").read_bytes()
    completed = {}
    for receipt in root.glob(".export-progress-*.json"):
        filename = json.loads(receipt.read_text())["filename"]
        shard = root / filename
        completed[filename] = (shard.read_bytes(), shard.stat().st_mtime_ns)
    assert completed

    with pytest.raises(FileExistsError):
        cpu_export.export(dataclasses.replace(config, model=dataclasses.replace(config.model, qk_mult=1.5)))
    retry = dataclasses.replace(
        config,
        read_host_budget_bytes=1024**2,
        export_host_budget_bytes=1024**2,
        max_concurrent_shards=1,
        request_concurrency=2,
    )
    cpu_export.export(retry)
    assert (root / "export-request.json").read_bytes() == request
    for filename, (contents, mtime) in completed.items():
        assert (root / filename).read_bytes() == contents
        assert (root / filename).stat().st_mtime_ns == mtime
    manifest = json.loads((root / "export-manifest.json").read_text())
    assert manifest["request"] == json.loads(request)
    assert manifest["execution"]["max_concurrent_shards"] == 1
    export(native)
    assert {path.name: path.read_bytes() for path in root.glob("*.safetensors")} == {
        path.name: path.read_bytes() for path in Path(native.destination).glob("*.safetensors")
    }


@pytest.mark.parametrize("master,manifest", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("shard_size", [256, 1024**2])
def test_cpu_export_matches_device_bytes_and_reads_expert_chunks_once(
    tmp_path, monkeypatch, master, manifest, shard_size
):
    # Cover both standalone banks (as in Hero) and multiple banks in one file.
    monkeypatch.setattr(hero_export, "DEFAULT_MAX_SHARD_SIZE", shard_size)
    monkeypatch.setattr(cpu_export, "DEFAULT_MAX_SHARD_SIZE", shard_size)
    request, model = native_fixture(str(tmp_path / "checkpoint"), master=master)
    if not manifest:
        (Path(request.checkpoint) / "manifest.json").unlink()
    export(request)
    cpu_request = cpu_export.CpuExportConfig(
        checkpoint=request.checkpoint,
        metadata_digest=request.metadata_digest,
        model=request.model,
        destination=str(tmp_path / "cpu-export"),
        source_revision=request.source_revision,
        read_host_budget_bytes=1024**2,
        export_host_budget_bytes=1024**2,
    )
    cpu_export.export(cpu_request)
    device_root = Path(request.destination)
    cpu_root = Path(cpu_request.destination)
    device_files = {path.name: path.read_bytes() for path in device_root.glob("*.safetensors")}
    cpu_files = {path.name: path.read_bytes() for path in cpu_root.glob("*.safetensors")}
    assert cpu_files == device_files
    for filename in ("config.json", "model.safetensors.index.json"):
        assert (cpu_root / filename).read_bytes() == (device_root / filename).read_bytes()
    completed = json.loads((cpu_root / "export-manifest.json").read_text())
    assert completed["expert_blocks_read"] == 3
    expected_native_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(model))
    pending_bytes = request.model.num_layers * request.model.num_experts * np.dtype(np.float32).itemsize
    assert completed["read_bytes"] == expected_native_bytes + pending_bytes
