import json
from pathlib import Path
import pytest
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from safetensors.torch import save_file
import torch

from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.models.grug_moe import GRUG_STACKED_EXPERT_SLICE_PATTERNS


class CountingFileSystem(AbstractFileSystem):
    protocol = "s3"
    cachable = False

    def __init__(self, payload: bytes, shards: dict[str, bytes] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        self.shards = shards or {}
        self.opens = 0
        self.handles = []
        self.fetched_bytes = 0

    def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(self.shards.get(path, self.payload))}

    def _open(self, path, mode="rb", block_size=None, cache_type="readahead", **kwargs):
        self.opens += 1
        handle = AbstractBufferedFile(self, path, mode, cache_type=cache_type)
        self.handles.append(handle)
        return handle

    def cat_file(self, path, start=None, end=None, **kwargs):
        payload = self.shards.get(path, self.payload)[start:end]
        self.fetched_bytes += len(payload)
        return payload


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
        lambda path: (
            (filesystem, path.removeprefix("s3://")) if path.startswith("s3://") else local_filesystem_for(path)
        ),
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    if indexed:
        weight_map = {key: expert_shard.name if split_shards and ".experts." in key else shard.name for key in tensors}
        (metadata / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
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
    store.close()
    assert all(handle.closed for handle in filesystem.handles)
