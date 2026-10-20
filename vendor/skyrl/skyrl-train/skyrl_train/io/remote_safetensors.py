"""Range-backed safetensors reads for immutable object-store model exports."""

from __future__ import annotations

from collections import defaultdict, deque
from bisect import bisect_right
from dataclasses import dataclass
import fnmatch
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, NamedTuple

import torch
from fsspec.spec import AbstractFileSystem
from fsspec.utils import merge_offset_ranges

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

_PREFETCH_WINDOW_BYTES = 1024**3
_PREFETCH_MAX_BLOCK = 64 * 1024**2
_PREFETCH_MAX_GAP = 1024**2
_PREFETCH_BATCH_SIZE = 16


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
    """Count shard opens, logical reads and bytes, range requests, read time and prefetch misses or unused items."""

    opens: int = 0
    reads: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    gets: int = 0
    prefetch_misses: int = 0
    prefetch_unused: int = 0


class _PrefetchItem(NamedTuple):
    key: str
    index: int | None
    shard: str
    start: int
    end: int


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
        self._gets = 0
        self._prefetch_misses = 0
        self._prefetch_unused = 0
        self._handles: dict[str, BinaryIO] = {}
        self._locations: dict[str, tuple[AbstractFileSystem, str]] = {}
        self._prefetch_plan: deque[tuple[str, int | None]] | None = None
        self._prefetched: dict[tuple[str, int | None], bytearray] = {}
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
    def lazy_first_dim_keys(self) -> frozenset[str]:
        return frozenset(self._lazy_first_dim_keys)

    def plan_prefetch(self, items: Iterable[tuple[str, int | None]]) -> None:
        """Plan requested tensor ranges in consumption order for bounded prefetch windows."""
        self._prefetch_plan = deque(dict.fromkeys(item for item in items if item[0] in self._weight_map))

    @property
    def read_stats(self) -> ReadStats:
        return ReadStats(
            self._opens,
            self._reads,
            self._bytes_read,
            self._read_seconds,
            self._gets,
            self._prefetch_misses,
            self._prefetch_unused + len(self._prefetched) + len(self._prefetch_plan or ()),
        )

    def _handle(self, shard: str) -> BinaryIO:
        source = self._handles.get(shard)
        if source is None:
            started = time.monotonic()
            uri = join_resource_path(self.source_uri, shard)
            filesystem, path = io.filesystem_and_path(uri)
            self._locations[shard] = filesystem, path
            source = filesystem.open(path, "rb", cache_type="none")
            self._handles[shard] = source
            self._opens += 1
            self._read_seconds += time.monotonic() - started
        return source

    def _take_prefetch_window(self) -> list[_PrefetchItem]:
        items = []
        budget = 0
        while self._prefetch_plan:
            key, index = self._prefetch_plan[0]
            shard = self._weight_map[key]
            _dtype, _shape, start, end = self._tensor_range(key, index)
            cost = end - start + _PREFETCH_MAX_GAP
            if cost > _PREFETCH_WINDOW_BYTES:
                self._prefetch_plan.popleft()
                continue
            if budget + cost > _PREFETCH_WINDOW_BYTES:
                break
            self._prefetch_plan.popleft()
            budget += cost
            items.append(_PrefetchItem(key, index, shard, start, end))
        return items

    def _fill_window(self) -> None:
        items = self._take_prefetch_window()
        if not items:
            return
        filesystem = self._locations[items[0].shard][0]
        paths, starts, ends = [], [], []
        for _key, _index, shard, start, end in items:
            path = self._locations[shard][1]
            for part_start in range(start, end, _PREFETCH_MAX_BLOCK):
                paths.append(path)
                starts.append(part_start)
                ends.append(min(part_start + _PREFETCH_MAX_BLOCK, end))
        paths, starts, ends = merge_offset_ranges(
            paths, starts, ends, max_gap=_PREFETCH_MAX_GAP, max_block=_PREFETCH_MAX_BLOCK
        )
        started = time.monotonic()
        payloads = filesystem.cat_ranges(paths, starts, ends, batch_size=_PREFETCH_BATCH_SIZE)
        self._gets += len(paths)
        self._read_seconds += time.monotonic() - started
        ranges = defaultdict(list)
        for path, start, end, payload in zip(paths, starts, ends, payloads, strict=True):
            if isinstance(payload, Exception):
                raise payload
            if len(payload) != end - start:
                raise ValueError("Truncated prefetched safetensors range")
            ranges[path].append((start, end, payload))
        for key, index, shard, start, end in items:
            fetched = ranges[self._locations[shard][1]]
            value = bytearray()
            for part_start in range(start, end, _PREFETCH_MAX_BLOCK):
                part_end = min(part_start + _PREFETCH_MAX_BLOCK, end)
                block = bisect_right(fetched, part_start, key=lambda entry: entry[0]) - 1
                low, high, payload = fetched[block]
                if block < 0 or not low <= part_start < part_end <= high:
                    raise ValueError(f"Prefetch did not cover {key!r}")
                value.extend(payload[part_start - low : part_end - low])
            self._prefetched[key, index] = value

    def _tensor_range(self, key: str, index: int | None) -> tuple[torch.dtype, tuple[int, ...], int, int]:
        shard = self._weight_map[key]
        offset, header = self._header(shard)
        dtype, shape, start, end = self._tensor_descriptor(key, shard, header)
        if index is not None:
            if not shape:
                raise IndexError(f"Cannot slice scalar safetensors tensor {key!r}")
            index = index + shape[0] if index < 0 else index
            if index < 0 or index >= shape[0]:
                raise IndexError(f"First-dimension index {index} is out of bounds for {key!r} with shape {shape}")
            shape = shape[1:]
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            start += index * size
            end = start + size
        return dtype, shape, offset + start, offset + end

    def _read(self, key: str, index: int | None = None) -> torch.Tensor:
        dtype, shape, start, end = self._tensor_range(key, index)
        payload = None
        if self._prefetch_plan is not None:
            if not self._prefetched:
                self._fill_window()
            payload = self._prefetched.pop((key, index), None)
            if payload is None:
                self._prefetch_misses += 1
        started = time.monotonic()
        if payload is None:
            source = self._handle(self._weight_map[key])
            source.seek(start)
            payload = bytearray(source.read(end - start))
            self._gets += 1
        self._record_read(len(payload), started)
        if len(payload) != end - start:
            raise ValueError(f"Truncated tensor {key!r} in {self.source_uri}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def close(self) -> None:
        """Release buffered ranges and source handles, recording unconsumed plan items."""
        self._prefetch_unused += len(self._prefetched) + len(self._prefetch_plan or ())
        self._prefetched.clear()
        self._prefetch_plan = None
        for source in self._handles.values():
            source.close()
        self._handles.clear()
        self._headers.clear()
        self._locations.clear()

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
        return self._read(key, index)

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        missing = [key for key in keys if key not in self._weight_map]
        if missing:
            raise KeyError(f"Tensors are absent from the safetensors index: {missing}")
        return {
            key: _LazyFirstDimensionTensor(self, key) if key in self._lazy_first_dim_keys else self._read(key)
            for key in keys
        }


class _LazyFirstDimensionTensor:
    """Defer a stacked tensor read until a mapping selects its rank-owned slice."""

    def __init__(self, store: RemoteSafetensorsTensorStore, key: str) -> None:
        self._store = store
        self._key = key

    def __getitem__(self, index: int) -> torch.Tensor:
        if not isinstance(index, int):
            raise TypeError(f"Remote stacked tensor slices require an integer index, got {type(index).__name__}")
        return self._store.load_first_dim_slice(self._key, index)
