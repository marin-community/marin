"""Range-backed safetensors reads for immutable object-store model exports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import fnmatch
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import torch

from marinskyrl.resource_locator import join_resource_path
from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    SAFETENSORS_LENGTH_PREFIX_BYTES,
    read_safetensors_header,
)
from skyrl_train.hf_model_io import HF_WEIGHT_FILENAME
from skyrl_train.io import io


_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def lazy_first_dim_patterns_for_bridge(bridge: object) -> tuple[str, ...]:
    """Return remote slice patterns from an AutoBridge's registered model bridge."""
    registered_bridge = getattr(bridge, "_model_bridge", bridge)
    return tuple(getattr(registered_bridge, "REMOTE_FIRST_DIM_SLICE_PATTERNS", ()))


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid safetensors shard path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class ReadStats:
    """Count shard opens and logical header or tensor reads with their bytes and elapsed time."""

    opens: int = 0
    reads: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0


class RemoteSafetensorsTensorStore:
    """Load requested tensors through one single-threaded handle per shard without local weight files."""

    def __init__(
        self,
        source_uri: str,
        metadata_dir: str | Path,
        *,
        lazy_first_dim_patterns: Iterable[str] = (),
    ) -> None:
        self.source_uri = source_uri.rstrip("/")
        index_path = Path(metadata_dir) / HF_WEIGHT_INDEX_FILENAME
        self._headers: dict[str, tuple[int, dict[str, object]]] = {}
        self._opens = 0
        self._reads = 0
        self._bytes_read = 0
        self._read_seconds = 0.0
        self._handles: dict[str, BinaryIO] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text())
                weight_map = index["weight_map"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid safetensors weight index: {index_path}") from error
        else:
            # A single-file export maps every header key to model.safetensors.
            shard = HF_WEIGHT_FILENAME
            shard_uri = join_resource_path(self.source_uri, shard)
            if not io.exists(shard_uri):
                raise ValueError(f"Missing safetensors weight index and single-file shard: {index_path}")
            _offset, header = self._header(shard)
            weight_map = {key: shard for key in header if key != "__metadata__"}
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors weight index has an empty weight_map: {index_path}")
        self._weight_map = {str(key): _safe_relative_path(str(value)) for key, value in weight_map.items()}
        self._lazy_first_dim_keys = {
            key for key in self._weight_map if any(fnmatch.fnmatch(key, pattern) for pattern in lazy_first_dim_patterns)
        }

    def get_all_keys(self) -> list[str]:
        return sorted(self._weight_map)

    @property
    def read_stats(self) -> ReadStats:
        return ReadStats(self._opens, self._reads, self._bytes_read, self._read_seconds)

    def _handle(self, shard: str) -> BinaryIO:
        source = self._handles.get(shard)
        if source is None:
            started = time.monotonic()
            uri = join_resource_path(self.source_uri, shard)
            filesystem, path = io.filesystem_and_path(uri)
            source = filesystem.open(path, "rb", cache_type="none")
            self._handles[shard] = source
            self._opens += 1
            self._read_seconds += time.monotonic() - started
        return source

    def _header(self, shard: str) -> tuple[int, dict[str, object]]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        source = self._handle(shard)
        source.seek(0)
        started = time.monotonic()
        header_bytes, _keys = read_safetensors_header(source, join_resource_path(self.source_uri, shard))
        self._record_read(len(header_bytes), started)
        cached = len(header_bytes), json.loads(header_bytes[SAFETENSORS_LENGTH_PREFIX_BYTES:])
        self._headers[shard] = cached
        return cached

    def _record_read(self, size: int, started: float) -> None:
        self._reads += 1
        self._bytes_read += size
        self._read_seconds += time.monotonic() - started

    def _tensor_descriptor(
        self, key: str, shard: str, header: dict[str, object]
    ) -> tuple[torch.dtype, tuple[int, ...], int, int]:
        descriptor = header.get(key)
        if not isinstance(descriptor, dict):
            raise ValueError(f"Tensor {key!r} is absent from {join_resource_path(self.source_uri, shard)}")
        try:
            dtype = _TORCH_DTYPES[str(descriptor["dtype"])]
            shape = tuple(int(size) for size in descriptor["shape"])
            start, end = (int(offset) for offset in descriptor["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid safetensors descriptor for {key!r} in {join_resource_path(self.source_uri, shard)}"
            ) from error
        expected_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if start < 0 or end < start or end - start != expected_size:
            raise ValueError(
                f"Invalid safetensors byte range for {key!r} in {join_resource_path(self.source_uri, shard)}"
            )
        return dtype, shape, start, end

    def load_first_dim_slice(self, key: str, index: int) -> torch.Tensor:
        """Read one contiguous first-dimension slice of a stacked tensor."""
        if key not in self._weight_map:
            raise KeyError(f"Tensor is absent from the safetensors index: {key!r}")
        shard = self._weight_map[key]
        shard_uri = join_resource_path(self.source_uri, shard)
        data_offset, header = self._header(shard)
        source = self._handle(shard)
        dtype, shape, start, _end = self._tensor_descriptor(key, shard, header)
        if not shape:
            raise IndexError(f"Cannot slice scalar safetensors tensor {key!r}")
        normalized_index = index + shape[0] if index < 0 else index
        if normalized_index < 0 or normalized_index >= shape[0]:
            raise IndexError(f"First-dimension index {index} is out of bounds for {key!r} with shape {shape}")
        slice_shape = shape[1:]
        slice_size = math.prod(slice_shape) * torch.empty((), dtype=dtype).element_size()
        slice_start = start + normalized_index * slice_size
        source.seek(data_offset + slice_start)
        started = time.monotonic()
        payload = bytearray(source.read(slice_size))
        self._record_read(len(payload), started)
        if len(payload) != slice_size:
            raise ValueError(f"Truncated tensor slice {key!r}[{index}] in {shard_uri}")
        return torch.frombuffer(payload, dtype=dtype).reshape(slice_shape)

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        missing = [key for key in keys if key not in self._weight_map]
        if missing:
            raise KeyError(f"Tensors are absent from the safetensors index: {missing}")
        tensors: dict[str, torch.Tensor] = {
            key: _LazyFirstDimensionTensor(self, key) for key in keys if key in self._lazy_first_dim_keys
        }
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            if key not in self._lazy_first_dim_keys:
                by_shard[self._weight_map[key]].append(key)

        for shard, shard_keys in by_shard.items():
            shard_uri = join_resource_path(self.source_uri, shard)
            data_offset, header = self._header(shard)
            source = self._handle(shard)
            for key in shard_keys:
                dtype, shape, start, end = self._tensor_descriptor(key, shard, header)
                source.seek(data_offset + start)
                started = time.monotonic()
                payload = bytearray(source.read(end - start))
                self._record_read(len(payload), started)
                if len(payload) != end - start:
                    raise ValueError(f"Truncated tensor {key!r} in {shard_uri}")
                tensors[key] = torch.frombuffer(payload, dtype=dtype).reshape(shape)
        return tensors


class _LazyFirstDimensionTensor:
    """Defer a stacked tensor read until a mapping selects its rank-owned slice."""

    def __init__(self, store: RemoteSafetensorsTensorStore, key: str) -> None:
        self._store = store
        self._key = key

    def __getitem__(self, index: int) -> torch.Tensor:
        if not isinstance(index, int):
            raise TypeError(f"Remote stacked tensor slices require an integer index, got {type(index).__name__}")
        return self._store.load_first_dim_slice(self._key, index)
