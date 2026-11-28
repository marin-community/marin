# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for the three checkpoint formats the measurement touches.

All access is read-only. Byte volume is bounded by the fixed sample in
``registry.SAMPLE``; nothing is copied through the Mac (jobs read in region).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import pickle
import posixpath
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import fsspec
import numpy as np
import tensorstore as ts
from registry import Checkpoint
from rigging.filesystem.cluster_config import StoreType, data_config
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials, s3_endpoint


def _build_kvstore_spec(path: str) -> dict:
    """Tensorstore kvstore spec for an s3:// URI (virtual-hosted endpoint, tensorstore #285)."""
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme != "s3":
        raise ValueError(f"only s3:// checkpoint URIs are supported, got {path}")
    bucket = parsed.netloc
    spec: dict = {"driver": "s3", "path": parsed.path.lstrip("/")}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        scheme, _, host = endpoint.partition("://")
        spec["endpoint"] = f"{scheme}://{bucket}.{host}"
    else:
        spec["bucket"] = bucket
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if region:
        spec["aws_region"] = region
    return spec


def configure_object_storage() -> None:
    """Point fsspec and tensorstore at CoreWeave object storage without printing secrets."""
    configure_coreweave_s3()
    credentials = s3_credentials(StoreType.COREWEAVE)
    if credentials is None:
        raise RuntimeError("CoreWeave object-storage credentials are not available in this environment")
    key, secret = credentials
    # tensorstore reads the standard AWS env chain; never log these values.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)
    os.environ.setdefault("AWS_ENDPOINT_URL", s3_endpoint(StoreType.COREWEAVE))


def bf16_view_to_f32(view: np.ndarray) -> np.ndarray:
    """Widen a BF16 buffer (uint16 elements) to float32."""
    return (view.astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---------------------------------------------------------------------------
# Levanter / Grug native checkpoints (OCDBT + zarr3, FP32)
# ---------------------------------------------------------------------------


class LevanterReader:
    """Chunk-cached slice reader over an OCDBT checkpoint.

    OCDBT stores one blob per zarr3 chunk; the chunk is the read unit (up to
    ~1.4 GB for the expert banks). The reader fetches whole chunks, slices in
    memory, and keeps a small LRU so many requests over one chunk cost one
    fetch. Slices may span chunk boundaries; covered chunks are concatenated.
    """

    CHUNK_CACHE_BYTES: ClassVar[int] = 8 * 2**30

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self._stores: dict[str, object] = {}
        self._chunk_shapes: dict[str, tuple[int, ...]] | None = None
        self._shapes: dict[str, tuple[int, ...]] | None = None
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self.bytes_read = 0
        self.chunk_fetches = 0

    # -- storage plumbing ---------------------------------------------------

    def _kvstore_spec(self) -> dict:
        return _build_kvstore_spec(self.checkpoint.uri)

    def _apply_signing_region(self, spec: dict) -> dict:
        """CoreWeave buckets sign against their declared signing region, not AWS_DEFAULT_REGION.

        ``build_kvstore_spec`` omits the bucket under a virtual-hosted endpoint, so
        resolve the bucket from the checkpoint URI instead.
        """
        bucket = urllib.parse.urlparse(self.checkpoint.uri).netloc
        spec_entry = data_config().buckets.get(bucket)
        if spec_entry is not None and spec_entry.signing_region:
            spec["kvstore"]["base"]["aws_region"] = spec_entry.signing_region
        return spec

    async def _store(self, array_path: str):
        if array_path not in self._stores:
            spec = self._apply_signing_region(
                {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": self._kvstore_spec(), "path": array_path}}
            )
            self._stores[array_path] = await ts.open(spec, read=True)
        return self._stores[array_path]

    # -- chunk plan ----------------------------------------------------------

    def _manifest(self) -> dict:
        manifest_uri = posixpath.join(self.checkpoint.uri, "manifest.json")
        fs, _ = fsspec.url_to_fs(manifest_uri)
        with fs.open(manifest_uri, "rb") as handle:
            return json.load(handle)

    def chunk_shapes(self) -> dict[str, tuple[int, ...]]:
        if self._chunk_shapes is None:
            self._chunk_shapes = {e["path"]: tuple(e["chunk_shape"]) for e in self._manifest()["arrays"]}
        return self._chunk_shapes

    def shapes(self) -> dict[str, tuple[int, ...]]:
        if self._shapes is None:
            self._shapes = {e["path"]: tuple(e["shape"]) for e in self._manifest()["arrays"]}
        return self._shapes

    def _cache_put(self, key: tuple, arr: np.ndarray) -> None:
        while self._cache and self._cache_bytes + arr.nbytes > self.CHUNK_CACHE_BYTES:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
        self._cache[key] = arr
        self._cache_bytes += arr.nbytes

    async def _fetch_chunk(self, array_path: str, chunk_index: tuple[int, ...]) -> np.ndarray:

        store = await self._store(array_path)
        chunks = self.chunk_shapes()[array_path]
        slices = tuple(slice(i * c, (i + 1) * c) for i, c in zip(chunk_index, chunks, strict=False))
        chunk = await store[slices].read()
        self._cache_put((array_path, chunk_index), chunk)
        self.bytes_read += chunk.nbytes
        self.chunk_fetches += 1
        return chunk

    # -- reads ---------------------------------------------------------------

    def _covered_chunks(self, array_path: str, slices: tuple) -> tuple[list[list[tuple[int, int]]], tuple[slice, ...]]:
        """Per-dim (chunk_index, within-chunk slice) coverage; returns ranges + normalized slices."""
        chunks = self.chunk_shapes()[array_path]
        shape = self.shapes()[array_path]
        ranges: list[list[tuple[int, int]]] = []
        normalized: list[slice] = []
        for dim in range(len(chunks)):
            spec = slices[dim] if dim < len(slices) else slice(None)
            size = shape[dim]
            chunk = chunks[dim]
            if isinstance(spec, int):
                spec = slice(spec, spec + 1)
            start, stop, step = spec.indices(size)
            if step != 1:
                raise ValueError(f"strided slices unsupported: {spec}")
            first, last = start // chunk, max(stop - 1, start) // chunk
            ranges.append(
                [
                    (i, slice(max(start, i * chunk) - i * chunk, min(stop, (i + 1) * chunk) - i * chunk))
                    for i in range(first, last + 1)
                ]
            )
            normalized.append(slice(start, stop))
        return ranges, tuple(normalized)

    async def _read_impl(self, array_path: str, slices: tuple) -> np.ndarray:
        ranges, _ = self._covered_chunks(array_path, slices)
        # Cartesian product of covered chunks (a dim spans >1 chunk only for band reads).
        choices = [[(index, within) for index, within in dim_ranges] for dim_ranges in ranges]
        combos = list(product(*choices))
        pieces = []
        for combo in combos:
            prefix = tuple(index for index, _ in combo)
            key = (array_path, prefix)
            if key not in self._cache:
                await self._fetch_chunk(array_path, prefix)
            piece = self._cache[key]
            for dim, (_, within) in enumerate(combo):
                if within != slice(0, piece.shape[dim]):
                    piece = piece[(slice(None),) * dim + (within,)]
            pieces.append(piece)
        if len(pieces) == 1:
            return pieces[0]
        # Chunks cross on exactly one dim in this sample (embedding bands): the
        # per-dim trims above already shaped every piece identically, so order the
        # pieces by the crossing dim's chunk index and concatenate along it.
        crossing = [dim for dim, dim_ranges in enumerate(ranges) if len(dim_ranges) > 1]
        if len(crossing) != 1:
            raise ValueError(f"multi-dim chunk crossing unsupported: {ranges}")
        dim = crossing[0]
        order = sorted(range(len(pieces)), key=lambda i: combos[i][dim][0])
        return np.concatenate([pieces[i] for i in order], axis=dim)

    def _array_path(self, tensor: str) -> str:
        """Canonical registry names map to ``params/<name>`` inside the OCDBT store."""
        return tensor if tensor.startswith("params/") else f"params/{tensor}"

    def read(self, tensor: str, slices: tuple) -> np.ndarray:
        """Read one hyper-rect; ``slices`` are per-dim ints/slices; missing dims read fully."""
        return asyncio.run(self._read_impl(self._array_path(tensor), slices))

    def read_full(self, tensor: str) -> np.ndarray:
        return asyncio.run(self._read_impl(self._array_path(tensor), ()))


# ---------------------------------------------------------------------------
# Hugging Face safetensors exports (BF16)
# ---------------------------------------------------------------------------


@dataclass
class SafetensorsTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # byte offset of the data
    length: int  # byte length
    shard: str


class SafetensorsReader:
    """Byte-range reader for a sharded safetensors export (never reads a whole shard)."""

    _DTYPES: ClassVar[dict] = {
        "F64": (np.dtype("<f8"), 8),
        "F32": (np.dtype("<f4"), 4),
        "F16": (np.dtype("<f2"), 2),
        "BF16": (np.dtype("uint16"), 2),
        "I64": (np.dtype("<i8"), 8),
        "I32": (np.dtype("<i4"), 4),
    }

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self._fs, _ = fsspec.url_to_fs(checkpoint.uri.rstrip("/"))
        self._root = checkpoint.uri.rstrip("/")
        self._index: dict[str, SafetensorsTensor] | None = None
        self._shard_headers: dict[str, dict[str, tuple[tuple[int, ...], str, int, int]]] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self.bytes_read = 0

    def _load_index(self) -> dict[str, SafetensorsTensor]:
        if self._index is not None:
            return self._index
        # The export ships model.safetensors.index.json mapping tensor -> shard file;
        # use it instead of scanning every shard header (193 shards).
        index_uri = posixpath.join(self._root, "model.safetensors.index.json")
        with self._fs.open(index_uri, "rb") as handle:
            index_doc = json.load(handle)
        self._index = {
            name: SafetensorsTensor(name=name, dtype="", shape=(), start=0, length=0, shard=shard)
            for name, shard in index_doc["weight_map"].items()
        }
        return self._index

    def _shard_header(self, shard: str) -> dict[str, tuple[tuple[int, ...], str, int, int]]:
        """Per-shard header: tensor name -> (shape, dtype, data start, length)."""
        if shard not in self._shard_headers:
            path = posixpath.join(self._root, shard)
            with self._fs.open(path, "rb") as handle:
                header_size = int(np.frombuffer(handle.read(8), dtype="<u8")[0])
                header = json.loads(handle.read(header_size).decode("utf-8"))
            base = 8 + header_size
            parsed = {}
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                relative = meta["data_offsets"]
                parsed[name] = (tuple(meta["shape"]), meta["dtype"], base + relative[0], relative[1] - relative[0])
            self._shard_headers[shard] = parsed
        return self._shard_headers[shard]

    def names(self) -> list[str]:
        return sorted(self._load_index())

    def shape_of(self, name: str) -> tuple[int, ...]:
        entry = self._load_index()[name]
        header = self._shard_header(entry.shard)
        return header[name][0]

    def _read_bytes(self, name: str, element_start: int, count: int) -> np.ndarray:
        entry = self._load_index()[name]
        _, dtype, data_start, _ = self._shard_header(entry.shard)[name]
        itemsize = self._DTYPES[dtype][1]
        with self._fs.open(posixpath.join(self._root, entry.shard), "rb") as handle:
            handle.seek(data_start + element_start * itemsize)
            payload = handle.read(count * itemsize)
        self.bytes_read += count * itemsize
        return np.frombuffer(payload, dtype=self._DTYPES[dtype][0]), dtype

    def read_f32(self, name: str, slices: tuple = ()) -> np.ndarray:
        """Read a hyper-rect as float32 (HF orientation).

        A leading-prefix slice (everything after the first sliced dim full) is one
        byte range; any other slice reads the whole tensor and narrows in memory.
        """
        index = self._load_index()
        entry = index[name]
        shape, _, _, _ = self._shard_header(entry.shard)[name]
        resolved = []
        for dim in range(len(shape)):
            spec = slices[dim] if dim < len(slices) else slice(None)
            if isinstance(spec, int):
                spec = slice(spec, spec + 1)
            start, stop, step = spec.indices(shape[dim])
            if step != 1:
                raise ValueError(f"strided slices unsupported: {spec}")
            resolved.append(slice(start, stop))
        first_sliced = next((d for d, s in enumerate(resolved) if s != slice(None, shape[d])), None)
        contiguous = first_sliced is None or all(
            resolved[d] == slice(None, shape[d]) for d in range(first_sliced + 1, len(shape))
        )
        if contiguous:
            flat = 0
            count = 1
            sliced_through = first_sliced if first_sliced is not None else len(shape) - 1
            for dim in range(sliced_through + 1):
                s = resolved[dim]
                if s == slice(None, shape[dim]):
                    flat *= shape[dim]
                    count *= shape[dim]
                else:
                    flat = flat * shape[dim] + s.start
                    count = (s.stop - s.start) * int(np.prod(shape[dim + 1 :], dtype=np.int64))
                    break
            payload, dtype_name = self._read_bytes(name, flat, count)
            out_shape = tuple(s.stop - s.start for s in resolved[: sliced_through + 1])
            arr = payload.reshape(out_shape)
        else:
            arr, dtype_name = self._read_bytes(name, 0, int(np.prod(shape, dtype=np.int64)))
            arr = arr.reshape(shape)
        if dtype_name == "BF16":
            arr = bf16_view_to_f32(arr.astype(np.uint16))
        else:
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr[tuple(resolved)])


# ---------------------------------------------------------------------------
# Megatron native checkpoints (torch distributed checkpoint)
# ---------------------------------------------------------------------------


# torch is Linux-only in this repo's dependency set; guard it as an optional dep.
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - macOS dev checkouts
    torch = None


def _require_torch():
    if torch is None:
        raise RuntimeError("torch is required for Megatron checkpoints but is not installed " "(Linux-only dependency)")


_MEGATRON_METADATA_CACHE: dict[str, object] = {}


def _megatron_metadata(checkpoint: Checkpoint):
    """Unpickle the DCP ``.metadata`` over fsspec.

    SkyRL's direct-S3 writer stores ``.metadata`` as a raw pickle of torch's
    Metadata object (not a torch.save zip), and torch's FileSystemReader cannot
    open s3:// URIs anyway (it collapses the scheme).
    """
    _require_torch()
    if checkpoint.uri in _MEGATRON_METADATA_CACHE:
        return _MEGATRON_METADATA_CACHE[checkpoint.uri]
    uri = checkpoint.uri.rstrip("/") + "/.metadata"
    fs, _ = fsspec.url_to_fs(uri)
    with fs.open(uri, "rb") as handle:
        metadata = pickle.load(handle)
    _MEGATRON_METADATA_CACHE[checkpoint.uri] = metadata
    return metadata


class MegatronInventory:
    """Parse a policy DCP ``.metadata`` into a compact key inventory (no tensor IO)."""

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self.metadata = _megatron_metadata(checkpoint)

    def entries(self) -> list[dict]:
        rows = []
        for key, meta in self.metadata.state_dict_metadata.items():
            if not hasattr(meta, "size"):
                continue  # BytesStorageMetadata (_extra_state) has no tensor geometry
            dtype = str(getattr(getattr(meta, "property", None), "dtype", "unknown"))
            size = [int(s) for s in meta.size]
            chunks = []
            for chunk in getattr(meta, "chunks", []):
                chunks.append(
                    {
                        "offsets": [int(o) for o in chunk.offsets],
                        "sizes": [int(s) for s in chunk.sizes],
                    }
                )
            rows.append({"key": key, "dtype": dtype, "size": size, "n_chunks": len(chunks), "chunks": chunks})
        return rows


def load_megatron_block(checkpoint: Checkpoint, key: str, meg_slices: tuple) -> np.ndarray:
    """Read one hyper-rect of a DCP tensor by streaming only intersecting chunks.

    Each chunk is one torch.save payload at a recorded (offset, length) byte
    range; chunk offsets are multi-dim and match ``storage_data`` fqns exactly.
    torch's own FileSystemReader cannot open s3:// URIs (it collapses the
    scheme), hence this reader.
    """
    _require_torch()
    metadata = _megatron_metadata(checkpoint)
    meta = metadata.state_dict_metadata[key]
    shape = tuple(int(s) for s in meta.size)
    resolved = []
    for dim in range(len(shape)):
        spec = meg_slices[dim] if dim < len(meg_slices) else slice(None)
        if isinstance(spec, int):
            spec = slice(spec, spec + 1)
        start, stop, step = spec.indices(shape[dim])
        if step != 1:
            raise ValueError(f"strided slices unsupported: {spec}")
        resolved.append(slice(start, stop))

    by_offsets: dict[tuple, object] = {}
    for index, info in metadata.storage_data.items():
        if getattr(index, "fqn", getattr(index, "key", None)) == key and index.offset is not None:
            by_offsets[tuple(int(o) for o in index.offset)] = info

    root = checkpoint.uri.rstrip("/")
    fs, _ = fsspec.url_to_fs(root)
    shard_cache: dict[str, object] = {}

    def read_shard_range(relative_path: str, offset: int, length: int) -> bytes:
        if relative_path not in shard_cache:
            shard_cache[relative_path] = fs.open(posixpath.join(root, relative_path), "rb")
        handle = shard_cache[relative_path]
        handle.seek(offset)
        return handle.read(length)

    buffer = np.empty(tuple(s.stop - s.start for s in resolved), dtype=np.float32)
    placed = 0
    for chunk in meta.chunks:
        chunk_offsets = tuple(int(o) for o in chunk.offsets)
        info = by_offsets.get(chunk_offsets)
        if info is None:
            continue
        overlap = []
        touches = True
        for c_off, c_size, req in zip(chunk_offsets, [int(s) for s in chunk.sizes], resolved, strict=False):
            lo = max(c_off, req.start)
            hi = min(c_off + c_size, req.stop)
            if lo >= hi:
                touches = False
                break
            overlap.append((lo, hi))
        if not touches:
            continue
        payload = read_shard_range(info.relative_path, info.offset, info.length)
        piece = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        piece_np = piece.numpy()
        src = tuple(slice(lo - c_off, hi - c_off) for (lo, hi), c_off in zip(overlap, chunk_offsets, strict=False))
        dst = tuple(slice(lo - req.start, hi - req.start) for (lo, hi), req in zip(overlap, resolved, strict=False))
        buffer[dst] = piece_np[src]
        placed += 1
    if placed == 0:
        raise KeyError(f"no DCP chunks intersect {key} {meg_slices}")
    return buffer


def load_megatron_keys(checkpoint: Checkpoint, keys: dict[str, tuple[int, ...]], dtype=None) -> dict[str, np.ndarray]:
    """Assemble full tensors for the given DCP keys via chunk streaming."""
    outputs = {}
    for key, shape in keys.items():
        outputs[key] = load_megatron_block(checkpoint, key, tuple(slice(None, s) for s in shape))
    return outputs
