# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Iterable

import fsspec
import humanfriendly
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
from fsspec import AbstractFileSystem
from fsspec.asyn import AsyncFileSystem

logger = logging.getLogger(__name__)

_SAFETENSOR_DTYPE_MAP: Dict[str, np.dtype] = {
    "F16": np.dtype("float16"),
    "BF16": np.dtype(jnp.bfloat16),
    "F32": np.dtype("float32"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"),
    "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"),
    "BOOL": np.dtype("bool"),
}


ShardingFunction = Callable[[Tuple[int, ...]], Optional[jax.sharding.Sharding]]

DEFAULT_CHUNK_SIZE_BYTES = int(os.environ.get("LEVANTER_FSSPEC_CHUNK_BYTES", 2 * 1024**3))
MAX_CONCURRENT_CHUNKS = int(os.environ.get("LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS", "4"))


@dataclass(frozen=True)
class TensorRecord:
    key: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    file_path: str
    byte_start: int
    byte_end: int


class _AsyncifyingFileSystemWrapper(AsyncFileSystem):
    """Wrap a synchronous AbstractFileSystem to provide async methods using a thread pool."""

    def __init__(self, fs: AbstractFileSystem):
        super().__init__()
        self._fs = fs
        import concurrent.futures

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS)

    async def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._fs.cat_file(path, start=start, end=end, **kwargs),
        )

    async def _info(self, path, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._fs.info(path, **kwargs),
        )

    async def _size(self, path):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._fs.size(path),
        )


# See https://huggingface.co/docs/safetensors/en/index#format for format spec
# It's pretty simple:
# # - 8 bytes: little-endian uint64 header length N
# # - N bytes: UTF-8 JSON header of shapes/dtypes/data offsets
# # - remaining bytes: raw tensor data blobs
async def _read_metadata_async(fs: AsyncFileSystem, path: str) -> Dict[str, TensorRecord]:
    header_len_bytes = await fs._cat_file(path, start=0, end=8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    metadata_bytes = await fs._cat_file(path, start=8, end=8 + header_len)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    tensors: Dict[str, TensorRecord] = {}
    data_offset_base = 8 + header_len

    for key, meta in metadata.items():
        if key == "__metadata__":
            continue
        dtype_name: str = meta["dtype"]
        dtype = _SAFETENSOR_DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")

        rel_start, rel_end = meta["data_offsets"]
        tensors[key] = TensorRecord(
            key=key,
            dtype=dtype,
            shape=tuple(meta["shape"]),
            file_path=path,
            byte_start=data_offset_base + rel_start,
            byte_end=data_offset_base + rel_end,
        )

    return tensors


@dataclass(frozen=True)
class ChunkSpec:
    file_path: str
    byte_start: int
    byte_end: int
    tensors: Tuple[TensorRecord, ...]

    @property
    def size(self) -> int:
        return self.byte_end - self.byte_start


def _build_chunks(tensors: Iterable[TensorRecord], chunk_limit: int) -> List[ChunkSpec]:
    if chunk_limit <= 0:
        raise ValueError("chunk_limit must be positive")

    sorted_records = sorted(tensors, key=lambda t: (t.file_path, t.byte_start))
    chunks: List[ChunkSpec] = []

    current: List[TensorRecord] = []
    current_start = 0
    current_end = 0
    current_path: Optional[str] = None

    for record in sorted_records:
        if not current:
            current = [record]
            current_start = record.byte_start
            current_end = record.byte_end
            current_path = record.file_path
            continue

        same_file = record.file_path == current_path
        proposed_end = max(current_end, record.byte_end)
        proposed_size = proposed_end - current_start

        if (not same_file) or (proposed_size > chunk_limit):
            chunks.append(
                ChunkSpec(
                    file_path=current_path or record.file_path,
                    byte_start=current_start,
                    byte_end=current_end,
                    tensors=tuple(current),
                )
            )
            current = [record]
            current_start = record.byte_start
            current_end = record.byte_end
            current_path = record.file_path
        else:
            current.append(record)
            current_end = proposed_end

    if current:
        chunks.append(
            ChunkSpec(
                file_path=current_path or sorted_records[-1].file_path,
                byte_start=current_start,
                byte_end=current_end,
                tensors=tuple(current),
            )
        )

    return chunks


def _materialize_sharded_tensor_from_host_array(
    array: np.ndarray,
    sharding: jax.sharding.Sharding,
) -> jax.Array:
    indices_map = sharding.devices_indices_map(array.shape)
    local_devices = sharding.addressable_devices
    per_device_arrays = []
    for device in local_devices:
        indices = tuple(indices_map[device])
        per_device_arrays.append(jax.device_put(array[indices], device))
    return jax.make_array_from_single_device_arrays(array.shape, sharding, per_device_arrays)


async def read_safetensors_fsspec(
    path: str,
    *,
    dtype_override: Optional[jnp.dtype] = None,
    sharding_fn: Optional[ShardingFunction] = None,
    fs: Optional[AbstractFileSystem] = None,
) -> Dict[str, jax.Array]:
    """
    Stream tensors from a safetensors file using fsspec, optionally sharding the outputs.
    """

    protocol, fs_path = fsspec.core.split_protocol(path)
    if protocol is None:
        protocol = "file"

    if fs is None:
        fs = fsspec.filesystem(protocol, asynchronous=True, anon=False)

    if isinstance(fs, AsyncFileSystem):
        async_fs = fs
    else:
        async_fs = _AsyncifyingFileSystemWrapper(fs)

    records = await _read_metadata_async(async_fs, path)
    chunk_specs = _build_chunks(records.values(), DEFAULT_CHUNK_SIZE_BYTES)

    sharding_fn = sharding_fn or (lambda _: None)
    sharding_map: Dict[str, Optional[jax.sharding.Sharding]] = {}

    for record in records.values():
        sharding = sharding_fn(record.shape)
        sharding_map[record.key] = sharding

    total_file_size = await async_fs._size(path)

    human_total = humanfriendly.format_size(total_file_size)
    logger.info(
        "Reading %d tensors from %s in %d chunk(s) (total size: %s)",
        len(records),
        path,
        len(chunk_specs),
        human_total,
    )

    pbar = tqdm.tqdm(total=total_file_size, unit="B", unit_scale=True, desc=f"Reading {path}")
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, min(MAX_CONCURRENT_CHUNKS, len(chunk_specs))))

    async def _materialize_chunk(chunk: ChunkSpec) -> Dict[str, jax.Array]:
        async with semaphore:
            raw = await async_fs._cat_file(chunk.file_path, start=chunk.byte_start, end=chunk.byte_end)
            chunk_view = memoryview(raw)
            chunk_results: Dict[str, jax.Array] = {}

            for record in chunk.tensors:
                offset = record.byte_start - chunk.byte_start
                count = int(np.prod(record.shape, dtype=int))
                tensor_np = np.frombuffer(
                    chunk_view,
                    dtype=record.dtype,
                    count=count,
                    offset=offset,
                ).reshape(record.shape)

                if dtype_override is not None and np.issubdtype(tensor_np.dtype, np.floating):
                    tensor_np = tensor_np.astype(np.dtype(dtype_override), copy=False)

                sharding = sharding_map[record.key]
                if sharding is not None:
                    array = _materialize_sharded_tensor_from_host_array(tensor_np, sharding)
                else:
                    array = jnp.asarray(tensor_np)

                chunk_results[record.key] = array

            async with progress_lock:
                pbar.update(chunk.size)

            return chunk_results

    chunk_dicts = await asyncio.gather(*(_materialize_chunk(chunk) for chunk in chunk_specs))
    pbar.close()

    result: Dict[str, jax.Array] = {}
    for chunk_dict in chunk_dicts:
        result.update(chunk_dict)
    return result
