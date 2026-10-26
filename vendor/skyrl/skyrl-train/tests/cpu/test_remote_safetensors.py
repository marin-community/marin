import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import skyrl_train.io.remote_safetensors as remote_safetensors
import torch
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from safetensors.torch import save_file
from skyrl_train.config.remote_read import RemoteReadMode
from skyrl_train.distributed.megatron.remote_weight_tasks import prefetch_items_from_tasks
from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore

from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME


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
        self.gets = 0

    def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(self.shards.get(path, self.payload))}

    def _open(self, path, mode="rb", block_size=None, cache_type="readahead", **kwargs):
        self.opens += 1
        handle = AbstractBufferedFile(self, path, mode, cache_type=cache_type)
        self.handles.append(handle)
        return handle

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.gets += 1
        payload = self.shards.get(path, self.payload)[start:end]
        self.fetched_bytes += len(payload)
        return payload


def _task(hf_param, megatron_param="dense", owned=True):
    return SimpleNamespace(
        megatron_module=object() if owned else None,
        mapping=SimpleNamespace(hf_param=hf_param, megatron_param=megatron_param),
    )


def _write_index(metadata_dir: Path, weight_map: dict[str, str]) -> None:
    metadata_dir.mkdir()
    (metadata_dir / HF_WEIGHT_INDEX_FILENAME).write_text(json.dumps({"weight_map": weight_map}))


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
    metadata = tmp_path / "metadata"
    if indexed:
        _write_index(
            metadata,
            {key: expert_shard.name if split_shards and ".experts." in key else shard.name for key in tensors},
        )
    else:
        metadata.mkdir()
    results = {}
    local_filesystem_for = io.filesystem_and_path
    for mode in RemoteReadMode:
        filesystem = CountingFileSystem(shard.read_bytes(), shards)
        monkeypatch.setattr(
            io,
            "filesystem_and_path",
            lambda path: (
                (filesystem, path.removeprefix("s3://")) if path.startswith("s3://") else local_filesystem_for(path)
            ),
        )
        store = RemoteSafetensorsTensorStore(
            "s3://bucket/policy",
            metadata,
            lazy_first_dim_patterns=(expert_key,),
        )
        tasks = [
            _task({"left": "layer.2.weight", "right": "layer.0.weight"}),
            _task("layer.1.weight"),
            *[
                _task(
                    expert_key
                    if expert_layout == "stacked"
                    else f"model.layers.0.mlp.experts.{index}.down_proj.weight",
                    str(index),
                )
                for index in (2, 0, 1)
            ],
            _task("unrequested.weight", owned=False),
            _task("missing.weight"),
        ]
        if mode == RemoteReadMode.PREFETCH:
            store.plan_prefetch(prefetch_items_from_tasks(tasks, store.lazy_first_dim_keys, int))
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
        if mode == RemoteReadMode.PREFETCH:
            assert store.read_stats.prefetch_misses == store.read_stats.prefetch_unused == 0
        results[mode] = filesystem.gets
    assert results[RemoteReadMode.PREFETCH] < results[RemoteReadMode.PER_KEY]


@pytest.mark.parametrize("window_bytes", [16, 9], ids=["split-ranges", "oversized-window"])
def test_prefetch_bounded_ranges_and_unplanned_reads_preserve_tensors(tmp_path, monkeypatch, window_bytes):
    tensors = {str(index): torch.arange(12, dtype=torch.uint8) + index for index in range(3)}
    tensors.update({str(index): torch.tensor([index, index + 1], dtype=torch.uint8) for index in range(3, 7)})
    tensors.update(
        {
            "3_gap": torch.ones(3, dtype=torch.uint8),
            "4_gap": torch.ones(3, dtype=torch.uint8),
            "5_gap": torch.ones(1, dtype=torch.uint8),
            "outside": torch.arange(4, dtype=torch.uint8),
        }
    )
    shard = tmp_path / "model.safetensors"
    save_file(tensors, shard)
    metadata = tmp_path / "metadata"
    _write_index(metadata, {key: shard.name for key in tensors})
    filesystem = CountingFileSystem(shard.read_bytes())
    monkeypatch.setattr(io, "filesystem_and_path", lambda path: (filesystem, path.removeprefix("s3://")))
    monkeypatch.setattr(remote_safetensors, "_PREFETCH_MAX_BLOCK", 8)
    monkeypatch.setattr(remote_safetensors, "_PREFETCH_MAX_GAP", 2)
    monkeypatch.setattr(remote_safetensors, "_PREFETCH_WINDOW_BYTES", window_bytes)
    payload_start = 8 + int.from_bytes(filesystem.payload[:8], "little")
    header = json.loads(filesystem.payload[8:payload_start])
    cat_ranges = filesystem.cat_ranges
    requested = []

    def bounded_ranges(paths, starts, ends, **kwargs):
        sizes = [end - start for start, end in zip(starts, ends, strict=True)]
        assert sum(sizes) <= max(window_bytes, 12)
        assert max(sizes) <= 8
        requested.extend(zip(starts, ends, strict=True))
        return cat_ranges(paths, starts, ends, **kwargs)

    monkeypatch.setattr(filesystem, "cat_ranges", bounded_ranges)
    store = RemoteSafetensorsTensorStore("s3://bucket/policy", metadata)
    store.plan_prefetch((key, None) for key in map(str, range(7)))
    try:
        for key in ("outside", *map(str, range(7))):
            torch.testing.assert_close(store.load_tensors([key])[key], tensors[key], rtol=0, atol=0)
        assert store.read_stats.prefetch_unused == 0
        assert store.read_stats.prefetch_misses == 1  # The unplanned "outside" tensor.
        assert store.read_stats.individual_reads == 1
    finally:
        store.close()
        assert all(handle.closed for handle in filesystem.handles)
    expected = []
    for key in ("0", "1", "2"):
        start = payload_start + header[key]["data_offsets"][0]
        expected.extend([(start, start + 8), (start + 8, start + 12)])
    expected.extend(
        [
            tuple(payload_start + value for value in header["3"]["data_offsets"]),
            tuple(payload_start + value for value in header["4"]["data_offsets"]),
            (payload_start + header["5"]["data_offsets"][0], payload_start + header["6"]["data_offsets"][1]),
        ]
    )
    assert requested == expected
    assert filesystem.fetched_bytes == payload_start + 36 + 8 + 4 + 1
