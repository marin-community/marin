import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from safetensors.torch import save_file
from s3fs import S3FileSystem
import torch

from cloud.iris.hf_model_cache import stage_model_metadata
from marinskyrl.model_manifest import snapshot_model_manifest
from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore, lazy_first_dim_patterns_for_bridge
from skyrl_train.models.grug_moe import GRUG_STACKED_EXPERT_SLICE_PATTERNS


class CountingFileSystem(AbstractFileSystem):
    protocol = "s3"
    cachable = False

    def __init__(self, payload: bytes, shards: dict[str, bytes] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        self.shards = shards or {}
        self.opens = 0
        self.fetched_bytes = 0

    def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(self.shards.get(path, self.payload))}

    def _open(self, path, mode="rb", block_size=None, cache_type="readahead", **kwargs):
        self.opens += 1
        return AbstractBufferedFile(
            self,
            path,
            mode,
            block_size=S3FileSystem.default_block_size,
            cache_type=cache_type,
            size=len(self.shards.get(path, self.payload)),
        )

    def cat_file(self, path, start=None, end=None, **kwargs):
        payload = self.shards.get(path, self.payload)[start:end]
        self.fetched_bytes += len(payload)
        return payload


def _write_index(metadata_dir: Path, weight_map: dict[str, str]) -> None:
    metadata_dir.mkdir()
    (metadata_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize(("indexed", "expert_layout"), [(True, "stacked"), (False, "stacked"), (True, "split")])
def test_single_key_and_expert_slice_reads_fetch_only_requested_bytes(
    tmp_path: Path, monkeypatch, indexed: bool, expert_layout: str
) -> None:
    shard = tmp_path / "model.safetensors"
    tensors = {f"layer.{index}.weight": torch.arange(8 + index, dtype=torch.float32) for index in range(3)}
    expert_key = "model.layers.0.mlp.experts.down_proj.weight"
    if expert_layout == "stacked":
        tensors[expert_key] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    else:
        for expert in range(3):
            tensors[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] = (
                torch.arange(4, dtype=torch.float32) + expert
            )
    tensors["unrequested.weight"] = torch.ones(1024)
    expert_shard = tmp_path / "experts.safetensors"
    split_shards = indexed and expert_layout == "split"
    if split_shards:
        save_file({key: value for key, value in tensors.items() if ".experts." not in key}, shard)
        save_file({key: value for key, value in tensors.items() if ".experts." in key}, expert_shard)
    else:
        save_file(tensors, shard)
    shards = {f"bucket/policy/{expert_shard.name}": expert_shard.read_bytes()} if split_shards else {}
    filesystem = CountingFileSystem(shard.read_bytes(), shards)
    local_filesystem_for = io.filesystem_and_path
    monkeypatch.setattr(
        io,
        "filesystem_and_path",
        lambda path: (filesystem, path.removeprefix("s3://"))
        if path.startswith("s3://")
        else local_filesystem_for(path),
    )
    metadata = tmp_path / "metadata"
    if indexed:
        _write_index(
            metadata,
            {key: expert_shard.name if split_shards and ".experts." in key else shard.name for key in tensors},
        )
    else:
        metadata.mkdir()
    store = RemoteSafetensorsTensorStore(
        "s3://bucket/policy",
        metadata,
        lazy_first_dim_patterns=GRUG_STACKED_EXPERT_SLICE_PATTERNS,
    )
    for index in (2, 0, 1):
        key = f"layer.{index}.weight"
        torch.testing.assert_close(store.load_tensors([key])[key], tensors[key])
    if expert_layout == "stacked":
        experts = store.load_tensors([expert_key])[expert_key]
        for index in (2, 0, 1):
            torch.testing.assert_close(experts[index], tensors[expert_key][index])
    else:
        expert_keys = [f"model.layers.0.mlp.experts.{expert}.down_proj.weight" for expert in (2, 0, 1)]
        experts = store.load_tensors(expert_keys)
        for key in expert_keys:
            torch.testing.assert_close(experts[key], tensors[key], rtol=0, atol=0)
    unrequested = tensors["unrequested.weight"]
    expected_bytes = (
        len(filesystem.payload) + sum(map(len, shards.values())) - unrequested.numel() * unrequested.element_size()
    )
    assert filesystem.opens == (2 if split_shards else 1)
    assert filesystem.fetched_bytes == expected_bytes
    assert not tuple(metadata.glob("*.safetensors"))


def test_invalid_index_is_not_replaced_by_single_file_fallback(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    save_file({"weight": torch.ones(2)}, remote / "model.safetensors")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "model.safetensors.index.json").write_text("not JSON")

    with pytest.raises(ValueError, match="Invalid safetensors weight index"):
        RemoteSafetensorsTensorStore(str(remote), metadata)


def test_auto_bridge_uses_registered_bridge_remote_slice_patterns() -> None:
    registered_bridge = SimpleNamespace(REMOTE_FIRST_DIM_SLICE_PATTERNS=("model.layers.*.mlp.experts.*",))
    auto_bridge = SimpleNamespace(_model_bridge=registered_bridge)

    assert lazy_first_dim_patterns_for_bridge(auto_bridge) == registered_bridge.REMOTE_FIRST_DIM_SLICE_PATTERNS


def test_twelve_rank_simulation_records_bounded_remote_reads_and_local_disk(tmp_path: Path, monkeypatch) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    rank_count = 12
    stacked_key = "model.layers.0.mlp.experts.down_proj.weight"
    save_file(
        {stacked_key: torch.arange(rank_count * 100_000, dtype=torch.float32).reshape(rank_count, 100_000)},
        remote / shard_name,
    )
    (remote / "config.json").write_text("{}")
    (remote / "tokenizer.json").write_text("{}")
    manifest = snapshot_model_manifest(remote, model_id="snowball/policy", revision="pinned")

    metadata_dirs = [tmp_path / f"rank-{rank}" for rank in range(rank_count)]
    for metadata in metadata_dirs:
        stage_model_metadata(str(remote), manifest, str(metadata))
    filesystem = CountingFileSystem((remote / shard_name).read_bytes())
    local_filesystem_for = io.filesystem_and_path
    monkeypatch.setattr(
        io,
        "filesystem_and_path",
        lambda path: (filesystem, str(path)) if str(path).startswith(str(remote)) else local_filesystem_for(path),
    )
    stores = [
        RemoteSafetensorsTensorStore(
            str(remote),
            metadata,
            lazy_first_dim_patterns=GRUG_STACKED_EXPERT_SLICE_PATTERNS,
        )
        for metadata in metadata_dirs
    ]
    for rank, store in enumerate(stores):
        stacked = store.load_tensors([stacked_key])[stacked_key]
        torch.testing.assert_close(
            stacked[rank],
            torch.arange(rank * 100_000, (rank + 1) * 100_000, dtype=torch.float32),
        )

    artifact_bytes = (remote / shard_name).stat().st_size
    s3_bytes_read = filesystem.fetched_bytes
    local_disk_high_water_bytes = sum(
        path.stat().st_size for metadata in metadata_dirs for path in metadata.rglob("*") if path.is_file()
    )

    assert s3_bytes_read < artifact_bytes * 1.1
    assert local_disk_high_water_bytes < artifact_bytes * 0.01
    assert not tuple(tmp_path.glob("rank-*/*.safetensors"))
