# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming sharded safetensors I/O over fsspec.

A 67B fp32 checkpoint is about 268 GB, so nothing here holds a checkpoint in
memory. :class:`SafetensorsCheckpoint` reads the ``model.safetensors.index.json``
weight map and each shard's header, then fetches one tensor, or one expert of a
stacked expert tensor, as a byte-range read. The same path serves local disk and
``s3://marin-us-east-02a``; nothing is downloaded whole.

``safetensors.safe_open`` is not used for reads: it needs a local file, so on S3
every 5 GB shard would be downloaded before one tensor could be read, and it
cannot read a single expert slice. The byte-range reader parses the same
header format; the tests check its output against ``safe_open``.

:func:`write_sharded_checkpoint` writes each shard header from a precomputed
plan, then streams tensor bytes straight to the destination, and writes the
index last.
"""

import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass

import fsspec
import ml_dtypes
import numpy as np

INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"
SHARD_NAME = "model-{index:05d}-of-{total:05d}.safetensors"
# HF's default. The largest Grug tensor (the stacked expert tensor of one layer,
# about 3.4 GB in fp32 at 67B) fits.
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3

# Files copied from B so vLLM can serve a hybrid.
REQUIRED_SERVING_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_SERVING_FILES = ("special_tokens_map.json", "chat_template.jinja")

_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
}
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}
_HEADER_ALIGNMENT = 8


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class _Location:
    shard: str
    spec: TensorSpec
    start: int
    """Absolute byte offset of the tensor data in its shard."""


def _read_header(fs: fsspec.AbstractFileSystem, shard: str) -> tuple[int, dict]:
    """(data start offset, header json) of one shard; reads only the header bytes."""
    size = int.from_bytes(fs.cat_file(shard, start=0, end=8), "little")
    header = json.loads(fs.cat_file(shard, start=8, end=8 + size))
    header.pop("__metadata__", None)
    return 8 + size, header


class SafetensorsCheckpoint:
    """A sharded (or single-file) safetensors checkpoint directory, read lazily."""

    def __init__(self, path: str) -> None:
        self.path = path.rstrip("/")
        self._fs, _ = fsspec.core.url_to_fs(self.path)
        index_path = f"{self.path}/{INDEX_NAME}"
        single_path = f"{self.path}/{SINGLE_FILE_NAME}"
        if self._fs.exists(index_path):
            with self._fs.open(index_path, "r") as handle:
                weight_map: dict[str, str] = json.load(handle)["weight_map"]
            shards = sorted(set(weight_map.values()))
        elif self._fs.exists(single_path):
            weight_map = {}
            shards = [SINGLE_FILE_NAME]
        else:
            raise FileNotFoundError(f"{self.path} has neither {INDEX_NAME} nor {SINGLE_FILE_NAME}")

        self._locations: dict[str, _Location] = {}
        for shard in shards:
            shard_path = f"{self.path}/{shard}"
            data_start, header = _read_header(self._fs, shard_path)
            for name, entry in header.items():
                if name in self._locations:
                    raise ValueError(f"{name!r} appears in more than one shard of {self.path}")
                if weight_map and weight_map.get(name) != shard:
                    raise ValueError(f"{name!r} is in {shard} but the index places it in {weight_map.get(name)}")
                if entry["dtype"] not in _DTYPES:
                    raise ValueError(f"{name!r} in {shard_path} has unsupported dtype {entry['dtype']}")
                spec = TensorSpec(tuple(entry["shape"]), _DTYPES[entry["dtype"]])
                begin, end = entry["data_offsets"]
                if end - begin != spec.nbytes:
                    raise ValueError(
                        f"{name!r} in {shard_path}: data_offsets cover {end - begin} bytes, not {spec.nbytes}"
                    )
                self._locations[name] = _Location(shard_path, spec, data_start + begin)
        if weight_map and set(weight_map) != set(self._locations):
            missing = sorted(set(weight_map) - set(self._locations))[:5]
            raise ValueError(f"{self.path}: the index names tensors no shard holds, e.g. {missing}")

    def names(self) -> list[str]:
        return sorted(self._locations)

    def spec(self, name: str) -> TensorSpec:
        return self._location(name).spec

    def read(self, name: str) -> np.ndarray:
        """One whole tensor in its stored dtype."""
        location = self._location(name)
        return self._read_range(location, 0, location.spec.nbytes).reshape(location.spec.shape)

    def read_expert(self, name: str, expert: int) -> np.ndarray:
        """Slice ``expert`` of a tensor stacked on dim 0, reading only that slice."""
        location = self._location(name)
        shape = location.spec.shape
        if len(shape) != 3:
            raise ValueError(f"{name!r} has shape {shape}; routed experts must be stacked [E, out, in]")
        if not 0 <= expert < shape[0]:
            raise IndexError(f"{name!r} stacks {shape[0]} experts; no expert {expert}")
        slice_bytes = location.spec.nbytes // shape[0]
        return self._read_range(location, expert * slice_bytes, slice_bytes).reshape(shape[1:])

    def _location(self, name: str) -> _Location:
        if name not in self._locations:
            raise KeyError(f"{name!r} is not a tensor of {self.path}")
        return self._locations[name]

    def _read_range(self, location: _Location, offset: int, size: int) -> np.ndarray:
        start = location.start + offset
        data = self._fs.cat_file(location.shard, start=start, end=start + size)
        if len(data) != size:
            raise OSError(f"short read from {location.shard}: {len(data)} of {size} bytes")
        return np.frombuffer(data, dtype=location.spec.dtype)


def _assign_shards(plan: dict[str, TensorSpec], shard_size_bytes: int) -> list[list[str]]:
    """Greedy shard assignment in plan order."""
    shards: list[list[str]] = [[]]
    used = 0
    for name, spec in plan.items():
        if spec.nbytes > shard_size_bytes:
            raise ValueError(f"{name!r} needs {spec.nbytes} bytes, above the shard budget of {shard_size_bytes}")
        if shards[-1] and used + spec.nbytes > shard_size_bytes:
            shards.append([])
            used = 0
        shards[-1].append(name)
        used += spec.nbytes
    return shards


def _shard_header(names: list[str], plan: dict[str, TensorSpec]) -> bytes:
    entries: dict[str, dict] = {}
    offset = 0
    for name in names:
        spec = plan[name]
        entries[name] = {
            "dtype": _DTYPE_NAMES[spec.dtype],
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    encoded = json.dumps(entries, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % _HEADER_ALIGNMENT)
    return len(encoded).to_bytes(8, "little") + encoded


def write_sharded_checkpoint(
    tensors: Iterable[tuple[str, np.ndarray]],
    output_dir: str,
    *,
    plan: dict[str, TensorSpec],
    shard_size_bytes: int,
) -> dict:
    """Stream tensors, arriving in ``plan`` order, to sharded safetensors plus the index.

    Each shard's header comes from the plan, so tensor bytes go straight to the
    destination and at most one tensor is in memory. Raises if a tensor arrives
    out of order, with a shape or dtype other than planned, or if the stream
    stops short. Returns the index json.
    """
    if not plan:
        raise ValueError("refusing to write a checkpoint with no tensors")
    for name, spec in plan.items():
        if spec.dtype not in _DTYPE_NAMES:
            raise ValueError(f"{name!r}: dtype {spec.dtype} cannot be written to safetensors here")
    output = output_dir.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(output)
    fs.makedirs(output, exist_ok=True)
    shards = _assign_shards(plan, shard_size_bytes)
    stream = iter(tensors)
    weight_map: dict[str, str] = {}
    for index, names in enumerate(shards, start=1):
        shard_name = SHARD_NAME.format(index=index, total=len(shards))
        with fs.open(f"{output}/{shard_name}", "wb") as handle:
            handle.write(_shard_header(names, plan))
            for expected in names:
                arrived = next(stream, None)
                if arrived is None:
                    raise ValueError(f"tensor stream ended before {expected!r}")
                name, array = arrived
                if name != expected:
                    raise ValueError(f"tensors must arrive in plan order: got {name!r}, expected {expected!r}")
                spec = plan[name]
                if array.shape != spec.shape or array.dtype != spec.dtype:
                    raise ValueError(f"{name!r}: got {array.shape}/{array.dtype}, plan says {spec.shape}/{spec.dtype}")
                handle.write(np.ascontiguousarray(array).reshape(-1).view(np.uint8).data)
                weight_map[name] = shard_name
    if next(stream, None) is not None:
        raise ValueError("tensor stream has more tensors than the plan")
    index = {"metadata": {"total_size": sum(spec.nbytes for spec in plan.values())}, "weight_map": weight_map}
    with fs.open(f"{output}/{INDEX_NAME}", "w") as handle:
        json.dump(index, handle, indent=2)
    return index


def copy_serving_files(source_dir: str, output_dir: str) -> list[str]:
    """Copy B's config, generation config and tokenizer next to a hybrid's weights.

    Every file in :data:`REQUIRED_SERVING_FILES` must exist in ``source_dir``;
    :data:`OPTIONAL_SERVING_FILES` are copied when present. Returns the names copied.
    """
    source = source_dir.rstrip("/")
    output = output_dir.rstrip("/")
    source_fs, _ = fsspec.core.url_to_fs(source)
    output_fs, _ = fsspec.core.url_to_fs(output)
    missing = [name for name in REQUIRED_SERVING_FILES if not source_fs.exists(f"{source}/{name}")]
    if missing:
        raise FileNotFoundError(f"{source} lacks {missing}; the hybrid would not be servable")
    names = list(REQUIRED_SERVING_FILES)
    names += [name for name in OPTIONAL_SERVING_FILES if source_fs.exists(f"{source}/{name}")]
    output_fs.makedirs(output, exist_ok=True)
    for name in names:
        with source_fs.open(f"{source}/{name}", "rb") as reader, output_fs.open(f"{output}/{name}", "wb") as writer:
            shutil.copyfileobj(reader, writer)
    return names
