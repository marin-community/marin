# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import copy
import dataclasses
import gc
import logging as pylogging
import operator
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union, cast

import deepdiff
import jax
import jax.tree_util as jtu
from rigging.filesystem import open_url, url_to_fs
import numpy as np
import pyarrow as pa
import tensorstore as ts
from dataclasses_json import dataclass_json
from fray import ResourceConfig
from fsspec import AbstractFileSystem
from jaxtyping import PyTree
from tqdm_loggable.tqdm_logging import tqdm_logging
from zephyr import Dataset, ZephyrContext
from zephyr import counters as zephyr_counters
from zephyr.writers import ThreadedBatchWriter, atomic_rename, batchify, ensure_parent_dir

from levanter.data.dataset import AsyncDataset
from levanter.utils.jax_utils import broadcast_one_to_all
from levanter.utils.thread_utils import blocking_wait

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.sharded_datasource import ShardedDataSource
from .jagged_array import JaggedArrayStore, _no_cache_read_context
from .tree_store import TreeStore

T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"
CONSOLIDATE_DATA_SIZE_WORKERS = 32
CACHE_LAYOUT_CONSOLIDATED = "consolidated"
CACHE_LAYOUT_SHARDED = "sharded"

DEFAULT_LOG_LEVEL = pylogging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


@dataclass(frozen=True)
class CacheOptions:
    batch_size: int = 128

    @staticmethod
    def default():
        return CacheOptions()


def build_or_load_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions = CacheOptions.default(),
) -> "TreeCache[U]":
    """
    Build or load a TreeCache from a sharded data source using a Zephyr backend.
    """
    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    try:
        return TreeCache.load(cache_dir, processor.output_exemplar, metadata)
    except FileNotFoundError:
        logger.info(f"Cache not found at {cache_dir}. Building with zephyr pipeline.")

    # Distributed coordination: only process 0 builds; others wait and then load.
    if jax.distributed.is_initialized() and jax.process_count() > 1:
        _distributed_build_cache(cache_dir, source, processor, options, metadata, is_leader=jax.process_index() == 0)
    else:
        build_cache(cache_dir, source, processor, options, metadata)

    return TreeCache.load(cache_dir, processor.output_exemplar, metadata)


class TreeCache(AsyncDataset[T_co]):
    ledger: "CacheLedger"

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: "CacheLedger",
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._exemplar = exemplar
        self._shard_stores: Dict[str, TreeStore[T_co]] = {}
        self._shard_row_offsets: Optional[np.ndarray] = None
        self._shard_field_stores: Dict[Tuple[str, str], Any] = {}
        self._shard_field_offsets: Dict[str, np.ndarray] = {}
        self._flat_field_offsets: Dict[str, np.ndarray] = {}
        self._flat_field_offset_futures: Dict[str, concurrent.futures.Future[np.ndarray]] = {}
        self._flat_field_offsets_lock = threading.Lock()

        if not ledger.is_finished:
            raise RuntimeError(f"Cache at {cache_dir} is not finished.")

        if self.is_sharded:
            _validate_sharded_ledger(ledger)
            self._reader: _TreeCacheReader[T_co] = _ShardedTreeCacheReader(self)
        else:
            store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=False)
            self._reader = _MaterializedTreeCacheReader(store)

    @property
    def store(self) -> TreeStore[T_co]:
        return self._reader.store

    @property
    def is_sharded(self) -> bool:
        return self.ledger.layout == CACHE_LAYOUT_SHARDED

    def _shard_path(self, shard_name: str) -> str:
        return os.path.join(self.cache_dir, shard_name)

    async def async_len(self) -> int:
        return await self._reader.async_len()

    def __len__(self):
        return len(self._reader)

    def is_finite(self) -> bool:
        return True

    def __getitem__(self, item):
        return self._reader[item]

    def __iter__(self):
        yield from self._reader

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self._reader.get_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None) -> Sequence[T_co]:
        return self._reader.get_batch_sync(indices_or_slice, timeout=timeout)

    def flat_field_length(self, field: str) -> int:
        return self._reader.flat_field_length(field)

    async def async_flat_field_length(self, field: str) -> int:
        return await self._reader.async_flat_field_length(field)

    def flat_field_num_rows(self, field: str) -> int:
        return self._reader.flat_field_num_rows(field)

    def jagged_array_tree(self):
        return self._reader.jagged_array_tree()

    async def get_flat_field_batch(self, field: str, offsets: Sequence[int], length: int) -> Sequence[np.ndarray]:
        return await self._reader.get_flat_field_batch(field, offsets, length)

    def _ensure_shard_row_offsets(self) -> Tuple[List[str], np.ndarray]:
        if self._shard_row_offsets is not None:
            return self.ledger.finished_shards, self._shard_row_offsets

        counts = [self.ledger.shard_rows[shard_name] for shard_name in self.ledger.finished_shards]
        self._shard_row_offsets = np.cumsum(np.array(counts, dtype=np.int64))
        return self.ledger.finished_shards, self._shard_row_offsets

    def _ensure_shard_field_offsets(self, field: str) -> Tuple[List[str], np.ndarray]:
        offsets = self._shard_field_offsets.get(field)
        if offsets is not None:
            return self.ledger.finished_shards, offsets

        counts = []
        for shard_name in self.ledger.finished_shards:
            field_count = self.ledger.field_counts_by_shard.get(shard_name, {}).get(field)
            if field_count is None:
                if self.ledger.shard_rows[shard_name] == 0:
                    field_count = 0
                else:
                    raise ValueError(f"Sharded cache ledger missing {field} count for shard {shard_name}")
            counts.append(field_count)

        offsets = np.cumsum(np.array(counts, dtype=np.int64))
        self._shard_field_offsets[field] = offsets
        return self.ledger.finished_shards, offsets

    def _ensure_flat_field_offsets(self, field: str) -> np.ndarray:
        return blocking_wait(self._ensure_flat_field_offsets_async(field))

    async def _ensure_flat_field_offsets_async(self, field: str) -> np.ndarray:
        with self._flat_field_offsets_lock:
            offsets = self._flat_field_offsets.get(field)
            if offsets is not None:
                return offsets

            future = self._flat_field_offset_futures.get(field)
            if future is None:
                future = concurrent.futures.Future()
                self._flat_field_offset_futures[field] = future
                should_build = True
            else:
                should_build = False

        if not should_build:
            return await asyncio.shield(asyncio.wrap_future(future))

        try:
            offsets = await self._build_flat_field_offsets_async(field)
        except BaseException as exc:
            with self._flat_field_offsets_lock:
                self._flat_field_offset_futures.pop(field, None)
            future.set_exception(exc)
            raise

        with self._flat_field_offsets_lock:
            self._flat_field_offsets[field] = offsets
            self._flat_field_offset_futures.pop(field, None)
        future.set_result(offsets)
        return offsets

    async def _build_flat_field_offsets_async(self, field: str) -> np.ndarray:
        async def read_shard_offsets(shard_name: str, row_count: int):
            field_store = await self._shard_field_store_async(shard_name, field)
            return await field_store.offsets[1 : row_count + 1].read()

        read_tasks = []
        shard_row_counts = []
        for shard_name in self.ledger.finished_shards:
            row_count = self.ledger.shard_rows[shard_name]
            if row_count == 0:
                continue

            if field not in self.ledger.field_counts_by_shard.get(shard_name, {}):
                raise ValueError(f"Sharded cache ledger missing {field} count for shard {shard_name}")

            read_tasks.append(read_shard_offsets(shard_name, row_count))
            shard_row_counts.append(row_count)

        shard_offset_arrays = await asyncio.gather(*read_tasks)

        pieces = [np.array([self.ledger.total_num_rows], dtype=np.int64)]
        data_offset = 0
        for shard_offsets, row_count in zip(shard_offset_arrays, shard_row_counts, strict=True):
            shard_offsets = np.asarray(shard_offsets, dtype=np.int64)
            assert len(shard_offsets) == row_count
            pieces.append(shard_offsets + data_offset)
            data_offset += int(shard_offsets[-1]) if len(shard_offsets) else 0

        offsets = np.concatenate(pieces)
        self._flat_field_offsets[field] = offsets
        return offsets

    async def _read_sharded_flat_field_slice(self, field: str, item: slice) -> np.ndarray:
        start, stop, step = item.indices(self.flat_field_length(field))
        if step != 1:
            data = await self._get_sharded_flat_field(field, start, max(stop - start, 0))
            return data[::step]
        return await self._get_sharded_flat_field(field, start, max(stop - start, 0))

    def _read_sharded_flat_field_slice_sync(self, field: str, item: slice) -> np.ndarray:
        return blocking_wait(self._read_sharded_flat_field_slice(field, item))

    def _shard_store(self, shard_name: str) -> TreeStore[T_co]:
        store = self._shard_stores.get(shard_name)
        if store is None:
            store = TreeStore.open(self._exemplar, self._shard_path(shard_name), mode="r", cache_metadata=True)
            self._shard_stores[shard_name] = store
        return store

    def _shard_field_store(self, shard_name: str, field: str):
        key = (shard_name, field)
        store = self._shard_field_stores.get(key)
        if store is None:
            tree_store = TreeStore.open(
                _field_exemplar(self._exemplar, field),
                self._shard_path(shard_name),
                mode="r",
                cache_metadata=True,
            )
            store = _tree_field(tree_store.tree, field)
            self._shard_field_stores[key] = store
        return store

    async def _shard_field_store_async(self, shard_name: str, field: str):
        key = (shard_name, field)
        store = self._shard_field_stores.get(key)
        if store is None:
            tree_store = await TreeStore.open_async(
                _field_exemplar(self._exemplar, field),
                self._shard_path(shard_name),
                mode="r",
                cache_metadata=True,
            )
            store = _tree_field(tree_store.tree, field)
            self._shard_field_stores[key] = store
        return store

    async def _get_sharded_batch(self, indices: Sequence[int]) -> List[T_co]:
        if len(indices) == 0:
            return []

        shard_names, shard_offsets = self._ensure_shard_row_offsets()
        shard_batches: Dict[int, List[Tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            index = int(index)
            if index < 0 or index >= self.ledger.total_num_rows:
                raise ValueError("Requested indices beyond the end of the dataset")

            shard_index = int(np.searchsorted(shard_offsets, index, side="right"))
            shard_start = int(shard_offsets[shard_index - 1]) if shard_index > 0 else 0
            shard_batches.setdefault(shard_index, []).append((output_index, index - shard_start))

        output: List[Optional[T_co]] = [None] * len(indices)

        async def read_shard(shard_index: int, batch: List[Tuple[int, int]]) -> None:
            local_indices = [local_index for _, local_index in batch]
            shard_batch = await self._shard_store(shard_names[shard_index]).get_batch(local_indices)
            for (output_index, _), row in zip(batch, shard_batch, strict=True):
                output[output_index] = row

        await asyncio.gather(*[read_shard(shard_index, batch) for shard_index, batch in shard_batches.items()])
        rows = []
        for row in output:
            assert row is not None
            rows.append(row)
        return rows

    def _get_sharded_batch_sync(self, indices: Sequence[int]) -> List[T_co]:
        if len(indices) == 0:
            return []

        shard_names, shard_offsets = self._ensure_shard_row_offsets()
        shard_batches: Dict[int, List[Tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            index = int(index)
            if index < 0 or index >= self.ledger.total_num_rows:
                raise ValueError("Requested indices beyond the end of the dataset")

            shard_index = int(np.searchsorted(shard_offsets, index, side="right"))
            shard_start = int(shard_offsets[shard_index - 1]) if shard_index > 0 else 0
            shard_batches.setdefault(shard_index, []).append((output_index, index - shard_start))

        output: List[Optional[T_co]] = [None] * len(indices)
        for shard_index, batch in shard_batches.items():
            local_indices = [local_index for _, local_index in batch]
            shard_batch = self._shard_store(shard_names[shard_index]).get_batch_sync(local_indices)
            for (output_index, _), row in zip(batch, shard_batch, strict=True):
                output[output_index] = row

        rows = []
        for row in output:
            assert row is not None
            rows.append(row)
        return rows

    async def _get_sharded_flat_field(self, field: str, offset: int, length: int) -> np.ndarray:
        if length == 0:
            return np.array([], dtype=np.asarray(_tree_field(self._exemplar, field)).dtype)

        shard_names, shard_offsets = self._ensure_shard_field_offsets(field)
        remaining = length
        position = offset
        reads = []

        while remaining > 0:
            shard_index = int(np.searchsorted(shard_offsets, position, side="right"))
            if shard_index >= len(shard_names):
                raise ValueError("Requested field offsets beyond the end of the dataset")

            shard_start = int(shard_offsets[shard_index - 1]) if shard_index > 0 else 0
            local_start = position - shard_start
            available = int(shard_offsets[shard_index] - position)
            take = min(remaining, available)
            field_store = self._shard_field_store(shard_names[shard_index], field)
            reads.append(field_store.data[local_start : local_start + take].read())
            position += take
            remaining -= take

        chunks = await asyncio.gather(*reads)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks)

    @staticmethod
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        logger.info(f"Loading cache from {cache_dir}")
        ledger = CacheLedger.load(cache_dir, options)

        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: Optional["CacheOptions"] = None,
    ) -> "TreeCache[U]":
        if options is None:
            options = CacheOptions.default()
        return build_or_load_cache(cache_dir, shard_source, processor, options=options)

    @property
    def is_finished(self):
        return True


class _TreeCacheReader(Protocol[T_co]):
    @property
    def store(self) -> TreeStore[T_co]: ...

    async def async_len(self) -> int: ...

    def __len__(self) -> int: ...

    def __getitem__(self, item) -> Union[T_co, Sequence[T_co]]: ...

    def __iter__(self) -> Iterator[T_co]: ...

    async def get_batch(self, indices: Union[Sequence[int], slice]) -> Sequence[T_co]: ...

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None) -> Sequence[T_co]: ...

    def flat_field_length(self, field: str) -> int: ...

    async def async_flat_field_length(self, field: str) -> int: ...

    def flat_field_num_rows(self, field: str) -> int: ...

    def jagged_array_tree(self) -> Any: ...

    async def get_flat_field_batch(self, field: str, offsets: Sequence[int], length: int) -> Sequence[np.ndarray]: ...


class _MaterializedTreeCacheReader:
    def __init__(self, store: TreeStore[T_co]):
        self._store = store

    @property
    def store(self) -> TreeStore[T_co]:
        return self._store

    async def async_len(self) -> int:
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, item) -> Union[T_co, Sequence[T_co]]:
        return self._store[item]

    def __iter__(self) -> Iterator[T_co]:
        for row in self._store:
            yield cast(T_co, row)

    async def get_batch(self, indices: Union[Sequence[int], slice]) -> Sequence[T_co]:
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = range(start, stop, step)
        return await self._store.get_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None) -> Sequence[T_co]:
        if isinstance(indices_or_slice, slice):
            start, stop, step = indices_or_slice.indices(len(self))
            indices_or_slice = range(start, stop, step)
        return self._store.get_batch_sync(indices_or_slice)

    def flat_field_length(self, field: str) -> int:
        return _tree_field(self._store.tree, field).data_size

    async def async_flat_field_length(self, field: str) -> int:
        return await _tree_field(self._store.tree, field).data_size_async()

    def flat_field_num_rows(self, field: str) -> int:
        return _tree_field(self._store.tree, field).num_rows

    def jagged_array_tree(self) -> Any:
        return self._store.tree

    async def get_flat_field_batch(self, field: str, offsets: Sequence[int], length: int) -> Sequence[np.ndarray]:
        if len(offsets) == 0:
            return []

        field_store = _tree_field(self._store.tree, field)
        with ts.Batch():
            futures = [field_store.data[int(offset) : int(offset) + length].read() for offset in offsets]
        return await asyncio.gather(*futures)


class _ShardedTreeCacheReader:
    def __init__(self, cache: TreeCache[T_co]):
        self._cache = cache

    @property
    def store(self) -> TreeStore[T_co]:
        raise RuntimeError(f"Cache at {self._cache.cache_dir} is sharded and has no top-level TreeStore.")

    async def async_len(self) -> int:
        return self._cache.ledger.total_num_rows

    def __len__(self) -> int:
        return self._cache.ledger.total_num_rows

    def __getitem__(self, item) -> Union[T_co, Sequence[T_co]]:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return self.get_batch_sync(range(start, stop, step))
        return self.get_batch_sync([item])[0]

    def __iter__(self) -> Iterator[T_co]:
        for index in range(len(self)):
            yield cast(T_co, self.get_batch_sync([index])[0])

    async def get_batch(self, indices: Union[Sequence[int], slice]) -> Sequence[T_co]:
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = range(start, stop, step)
        return await self._cache._get_sharded_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None) -> Sequence[T_co]:
        if isinstance(indices_or_slice, slice):
            start, stop, step = indices_or_slice.indices(len(self))
            indices_or_slice = range(start, stop, step)
        return self._cache._get_sharded_batch_sync(indices_or_slice)

    def flat_field_length(self, field: str) -> int:
        field_count = self._cache.ledger.field_counts.get(field)
        if field_count is not None:
            return field_count

        if self._cache.ledger.total_num_rows == 0:
            return 0
        raise ValueError(f"Sharded cache ledger missing aggregate {field} count")

    async def async_flat_field_length(self, field: str) -> int:
        field_count = self._cache.ledger.field_counts.get(field)
        if field_count is not None:
            return field_count

        if self._cache.ledger.total_num_rows == 0:
            return 0
        raise ValueError(f"Sharded cache ledger missing aggregate {field} count")

    def flat_field_num_rows(self, field: str) -> int:
        return self._cache.ledger.total_num_rows

    def jagged_array_tree(self) -> Any:
        def field_store(path, _):
            field = "/".join(_render_path_elem(part) for part in path)
            return _ShardedJaggedArrayStore(self._cache, field)

        return jtu.tree_map_with_path(field_store, self._cache._exemplar)

    async def get_flat_field_batch(self, field: str, offsets: Sequence[int], length: int) -> Sequence[np.ndarray]:
        if len(offsets) == 0:
            return []
        return await asyncio.gather(
            *[self._cache._get_sharded_flat_field(field, int(offset), length) for offset in offsets]
        )


@dataclass_json
@dataclass
class CacheLedger:
    total_num_rows: int
    shard_rows: Dict[str, int]
    is_finished: bool = False
    finished_shards: List[str] = dataclasses.field(default_factory=list)
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    field_counts_by_shard: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)
    layout: str = CACHE_LAYOUT_CONSOLIDATED
    metadata: "CacheMetadata" = dataclasses.field(default_factory=lambda: CacheMetadata({}))

    @staticmethod
    def load_or_initialize(cache_dir: str, source: ShardedDataSource, processor: BatchProcessor):
        metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
        try:
            return CacheLedger.load(cache_dir, metadata)
        except FileNotFoundError:
            return CacheLedger(
                total_num_rows=0,
                shard_rows={shard: 0 for shard in source.shard_names},
                is_finished=False,
                metadata=metadata,
            )

    @staticmethod
    def load(cache_dir: str, metadata: Optional["CacheMetadata"] = None) -> "CacheLedger":
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        try:
            logger.info(f"Attempting to load cache ledger from {ledger_path}")
            with open_url(ledger_path) as file:
                cache_ledger = CacheLedger.from_json(file.read())  # type: ignore[arg-type]
            if metadata:
                diff = cache_ledger.metadata.compare_to(metadata)
                if diff:
                    logger.warning(f"Metadata mismatch: {diff}")
            return cache_ledger
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Cache ledger not found at {ledger_path}") from exc

    def _serialize_and_commit(self, cache_dir):
        path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        return _serialize_json_and_commit(path, self)  # type: ignore[arg-type]


def _validate_sharded_ledger(ledger: CacheLedger) -> None:
    seen_shards: set[str] = set()
    total_num_rows = 0
    for shard_name in ledger.finished_shards:
        if shard_name in seen_shards:
            raise ValueError(f"Sharded cache ledger contains duplicate shard {shard_name}")
        seen_shards.add(shard_name)

        if "://" in shard_name or os.path.isabs(shard_name):
            raise ValueError(f"Sharded cache ledger shard path must be relative: {shard_name}")

        if shard_name not in ledger.shard_rows:
            raise ValueError(f"Sharded cache ledger missing row count for shard {shard_name}")

        num_rows = ledger.shard_rows[shard_name]
        if num_rows < 0:
            raise ValueError(f"Sharded cache ledger has negative row count for shard {shard_name}: {num_rows}")
        total_num_rows += num_rows

    if total_num_rows != ledger.total_num_rows:
        raise ValueError(
            "Sharded cache ledger row count mismatch: "
            f"sum(finished shard rows)={total_num_rows}, total_num_rows={ledger.total_num_rows}"
        )

    for shard_name in ledger.field_counts_by_shard:
        if shard_name not in seen_shards:
            raise ValueError(f"Sharded cache ledger has field counts for unknown shard {shard_name}")

    field_counts: Dict[str, int] = {}
    for shard_name in ledger.finished_shards:
        for field, count in ledger.field_counts_by_shard.get(shard_name, {}).items():
            field_counts[field] = field_counts.get(field, 0) + count

    if field_counts != ledger.field_counts:
        raise ValueError(
            "Sharded cache ledger field count mismatch: "
            f"sum(finished shard field counts)={field_counts}, field_counts={ledger.field_counts}"
        )


def _tree_field(tree, field: str):
    value = tree
    for part in field.split("/"):
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, (list, tuple)):
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def _field_exemplar(exemplar, field: str):
    if "/" not in field and isinstance(exemplar, Mapping):
        return {field: exemplar[field]}
    return exemplar


class _ArrayRead:
    def __init__(self, sync_reader, async_reader=None):
        self._sync_reader = sync_reader
        self._async_reader = async_reader

    def read(self):
        return _ArrayReadFuture(self._sync_reader, self._async_reader)


class _ArrayReadFuture:
    def __init__(self, sync_reader, async_reader=None):
        self._sync_reader = sync_reader
        self._async_reader = async_reader

    def result(self):
        return self._sync_reader()

    def __await__(self):
        async def read_async():
            if self._async_reader is not None:
                return await self._async_reader()
            return self._sync_reader()

        return read_async().__await__()


class _ShardedJaggedArrayData:
    def __init__(self, cache: TreeCache, field: str):
        self._cache = cache
        self._field = field

    def __getitem__(self, item):
        if not isinstance(item, slice):
            item = slice(item, item + 1)
        return _ArrayRead(
            lambda: self._cache._read_sharded_flat_field_slice_sync(self._field, item),
            lambda: self._cache._read_sharded_flat_field_slice(self._field, item),
        )


class _ShardedJaggedArrayOffsets:
    def __init__(self, cache: TreeCache, field: str):
        self._cache = cache
        self._field = field

    def __getitem__(self, item):
        async def read_offsets():
            return (await self._cache._ensure_flat_field_offsets_async(self._field))[item]

        return _ArrayRead(
            lambda: self._cache._ensure_flat_field_offsets(self._field)[item],
            read_offsets,
        )


class _ShardedJaggedArrayStore:
    def __init__(self, cache: TreeCache, field: str):
        self._cache = cache
        self._field = field
        self.offsets = _ShardedJaggedArrayOffsets(cache, field)
        self.data = _ShardedJaggedArrayData(cache, field)
        self.shapes = None
        self.item_rank = 1

    @property
    def num_rows(self) -> int:
        return self._cache.flat_field_num_rows(self._field)

    @property
    def data_size(self) -> int:
        return self._cache.flat_field_length(self._field)


@dataclass_json
@dataclass(frozen=True)
class CacheMetadata:
    preprocessor_metadata: Optional[dict[str, Any]] = None

    def compare_to(self, other: "CacheMetadata") -> deepdiff.DeepDiff:
        if other.preprocessor_metadata is None:
            sorta_self = dataclasses.replace(self, preprocessor_metadata=None)
        else:
            sorta_self = self
        return deepdiff.DeepDiff(sorta_self, other)

    @staticmethod
    def empty():
        return CacheMetadata()


class SerialCacheWriter:
    """
    Writes TreeCache-compatible caches to disk directly. Mostly for scripts and debugging.
    """

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        metadata: Optional["CacheMetadata"] = None,
        shard_name: str = "",
        mode: str = "w",
    ):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._exemplar = exemplar
        self._shard_name = shard_name
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode=mode, cache_metadata=True)
        self._is_closed = False

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ledger = CacheLedger(
            total_num_rows=len(self._tree_store),
            is_finished=True,
            shard_rows={self._shard_name: len(self._tree_store)},
            finished_shards=[self._shard_name],
            field_counts=_field_counts_from_store(self._tree_store),
            metadata=self.metadata or CacheMetadata.empty(),
        )

        if exc_type is None:
            ledger._serialize_and_commit(self.cache_dir)
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self) -> "TreeCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until TreeCacheWriter is closed")
        return TreeCache.load(self.cache_dir, self._exemplar, self.metadata)

    def write_batch(self, batch: BatchResult):
        if isinstance(batch, pa.RecordBatch):
            batch = dict_from_record_batch(batch)

        cbatch = _canonicalize_batch(batch)  # type: ignore[arg-type]
        self._tree_store.extend(cbatch)


# Fixed batch size for Levanter cache writes (2^14).
_LEVANTER_BATCH_SIZE = 16384


def write_levanter_cache(
    records: Iterable[dict[str, Any]],
    output_path: str,
    *,
    metadata: dict[str, Any],
    batch_size: int = _LEVANTER_BATCH_SIZE,
) -> dict:
    """Write tokenized records to Levanter cache format.

    Args:
        records: Tokenized records (iterable of dicts with array values)
        output_path: Path to output cache directory
        metadata: Metadata for the cache
        batch_size: Number of records to accumulate before flushing to disk.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    ensure_parent_dir(output_path)
    record_iter = iter(records)

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 0
    logger.info("write_levanter_cache: starting write to %s (batch_size=%d)", output_path, batch_size)

    with atomic_rename(output_path) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata)) as writer:

            def _drain_batches(batches: Iterable) -> None:
                for batch in batches:
                    writer.write_batch(batch)

            with ThreadedBatchWriter(_drain_batches) as threaded:
                threaded.submit([exemplar])
                count += 1
                zephyr_counters.increment("zephyr/records_out")
                for batch in batchify(record_iter, n=batch_size):
                    threaded.submit(batch)
                    count += len(batch)
                    zephyr_counters.increment("zephyr/records_out", len(batch))
                    logger.info("write_levanter_cache: %s — %d records so far", output_path, count)

    logger.info("write_levanter_cache: finished %s — %d records", output_path, count)

    # write success sentinel
    with open_url(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def _serialize_json_and_commit(path: str, obj):
    fs: AbstractFileSystem = url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        fs.copy(path, f"{path}.bak")

    for _ in range(10):
        try:
            with open_url(path, "w") as file:
                file.write(obj.to_json())
            break
        except FileNotFoundError:
            logger.exception(f"Failed to write {path}")


def build_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions,
    metadata: CacheMetadata,
) -> CacheLedger:
    """
    Build a cache from a sharded data source using a Zephyr backend.
    """
    shard_names = list(source.shard_names)

    if len(shard_names) == 0:
        logger.info("No shards to process. Writing empty cache.")
        TreeStore.open(processor.output_exemplar, cache_dir, mode="w", cache_metadata=True)
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            finished_shards=[],
            field_counts={},
            metadata=metadata,
        )
        ledger._serialize_and_commit(cache_dir)
        return ledger

    temp_root = cache_dir
    shard_jobs = [{"shard_name": name, "index": idx} for idx, name in enumerate(shard_names)]

    def process_shard(job: dict):
        return _build_single_shard_cache(
            shard_name=job["shard_name"],
            shard_index=job["index"],
            temp_root=temp_root,
            source=source,
            processor=processor,
            options=options,
            metadata=metadata,
        )

    ctx = ZephyrContext(
        resources=ResourceConfig(ram="32g", disk="16g"),
        max_workers=min(128, len(shard_jobs)),
        name="levanter-cache-build",
    )
    shard_results = ctx.execute(Dataset.from_list(shard_jobs).map(process_shard), verbose=False).results
    shard_results = sorted(shard_results, key=lambda r: r["index"])

    shard_cache_paths = [s["path"] for s in shard_results]
    ledger = consolidate_shard_caches(
        shard_cache_paths=shard_cache_paths,
        output_path=cache_dir,
        exemplar=processor.output_exemplar,
        metadata=metadata,
    )
    return ledger


def _build_single_shard_cache(
    shard_name: str,
    shard_index: int,
    temp_root: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    options: CacheOptions,
    metadata: CacheMetadata,
):
    shard_path = os.path.join(temp_root, f"{shard_index:05d}_{_sanitize_shard_name(shard_name)}")
    existing = _try_load(shard_path, metadata)
    if existing is not None:
        logger.info(f"Found existing shard cache for {shard_name} at {shard_path}. Skipping build.")
        return {"shard_name": shard_name, "path": shard_path, "ledger": existing, "index": shard_index}

    logger.info(f"Building shard {shard_name} -> {shard_path}")

    def records():
        batch = []
        pbar = tqdm_logging(desc=f"Shard {shard_name}")
        for example in source.open_shard_at_row(shard_name, 0):
            batch.append(example)
            if len(batch) >= options.batch_size:
                processed = processor(batch)
                yield from _canonicalize_batch(processed)
                batch.clear()
            pbar.update(1)
        if batch:
            processed = processor(batch)
            yield from _canonicalize_batch(processed)

    result = write_levanter_cache(records(), shard_path, metadata=metadata.preprocessor_metadata or {})

    if result.get("count", 0) == 0:
        TreeStore.open(processor.output_exemplar, shard_path, mode="w", cache_metadata=True)
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={shard_name: 0},
            is_finished=True,
            finished_shards=[shard_name],
            field_counts={},
            metadata=metadata,
        )
        ledger._serialize_and_commit(shard_path)
    else:
        ledger = CacheLedger.load(shard_path, metadata)

    return {"shard_name": shard_name, "path": shard_path, "ledger": ledger, "index": shard_index}


def consolidate_shard_caches(
    shard_cache_paths: list[str],
    output_path: str,
    exemplar,
    metadata: CacheMetadata | None = None,
    copy_max_workers: int = 128,
) -> CacheLedger:
    """
    Consolidate multiple shard caches into a single materialized cache directory.

    Args:
        shard_cache_paths: List of shard cache directories.
        output_path: Destination cache directory.
        exemplar: Output exemplar structure.
        metadata: CacheMetadata to use for the final ledger.
        copy_max_workers: Maximum Zephyr fanout for the cache copy phase.
    """
    if metadata is None:
        metadata = CacheMetadata.empty()
    if copy_max_workers < 1:
        raise ValueError(f"copy_max_workers must be positive, got {copy_max_workers}")

    if not shard_cache_paths:
        TreeStore.open(exemplar, output_path, mode="w", cache_metadata=True)
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            finished_shards=[],
            field_counts={},
            metadata=metadata,
        )
        ledger._serialize_and_commit(output_path)
        return ledger

    logger.info(f"Consolidating {len(shard_cache_paths)} shard caches into {output_path}")

    with _no_cache_read_context():
        first_cache = TreeStore.open(exemplar, shard_cache_paths[0], mode="r", cache_metadata=True)
        data_offset_tree = jax.tree.map(lambda x: 0, first_cache.tree)

    shard_info: list[dict] = []
    total_rows = 0

    def _probe_shard(shard_path):
        ledger = CacheLedger.load(shard_path, metadata)
        with _no_cache_read_context():
            store = TreeStore.open(exemplar, shard_path, mode="r", cache_metadata=True)
            data_sizes = jax.tree.map(lambda x: x.data_size, store.tree)
        return (data_sizes, ledger)

    probe_ctx = ZephyrContext(
        resources=ResourceConfig(ram="5g", cpu=2),
        max_workers=min(CONSOLIDATE_DATA_SIZE_WORKERS, len(shard_cache_paths)),
        name="levanter-cache-probe",
    )
    probe_results = probe_ctx.execute(
        Dataset.from_list(shard_cache_paths).map(_probe_shard),
    ).results
    per_shard_sizes = [r[0] for r in probe_results]
    shard_ledgers = [r[1] for r in probe_results]
    per_shard_field_counts = [_field_counts_from_data_sizes(data_sizes) for data_sizes in per_shard_sizes]

    for shard_path, ledger, this_offsets in zip(shard_cache_paths, shard_ledgers, per_shard_sizes, strict=True):
        shard_name = os.path.basename(shard_path)
        shard_info.append(
            {
                "path": shard_path,
                "shard_name": shard_name,
                "row_offset": total_rows,
                "data_offset_tree": copy.deepcopy(data_offset_tree),
                "ledger": ledger,
            }
        )
        total_rows += ledger.total_num_rows
        data_offset_tree = jax.tree.map(operator.add, data_offset_tree, this_offsets)

    TreeStore.open(exemplar, output_path, mode="w", cache_metadata=True)

    def _copy_shard(info: dict):
        asyncio.run(
            _extend_cache_with_other_cache(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )

    ctx = ZephyrContext(
        resources=ResourceConfig(ram="10g", disk="16g"),
        max_workers=min(copy_max_workers, len(shard_info)),
        name="levanter-cache-copy",
    )
    ctx.execute(
        Dataset.from_list(shard_info).map(_copy_shard),
        verbose=False,
    )

    asyncio.run(_consolidate_metadata(output_path, exemplar, shard_info))

    final_ledger = _merge_materialized_ledgers(
        output_path, shard_cache_paths, shard_ledgers, per_shard_field_counts, metadata
    )
    _expose_cache_rows(output_path, exemplar, final_ledger.total_num_rows)
    return final_ledger


def consolidate_shard_cache_ledgers(
    shard_cache_paths: List[str],
    output_path: str,
    exemplar,
    metadata: Optional[CacheMetadata] = None,
) -> CacheLedger:
    """
    Consolidate multiple shard cache ledgers into one sharded cache ledger.

    Shard directories must be under output_path. The output ledger stores their
    relative paths instead of copying their arrays into a top-level TreeStore.
    """
    if metadata is None:
        metadata = CacheMetadata.empty()

    if not shard_cache_paths:
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            finished_shards=[],
            field_counts={},
            field_counts_by_shard={},
            layout=CACHE_LAYOUT_SHARDED,
            metadata=metadata,
        )
        ledger._serialize_and_commit(output_path)
        return ledger

    for shard_path in shard_cache_paths:
        _relative_shard_path(output_path, shard_path)

    logger.info(f"Consolidating {len(shard_cache_paths)} shard cache ledgers into {output_path}")

    # Distributed: load ledger + read data_size for each shard in parallel.
    # Both operations are S3 I/O-bound; distributing across zephyr workers
    # avoids serializing thousands of S3 calls in the coordinator process.
    def _probe_shard(shard_path):
        ledger = CacheLedger.load(shard_path, metadata)
        if ledger.field_counts:
            field_counts = ledger.field_counts
        else:
            with _no_cache_read_context():
                store = TreeStore.open(exemplar, shard_path, mode="r", cache_metadata=True)
                field_counts = _field_counts_from_store(store)
        return (field_counts, ledger)

    probe_ctx = ZephyrContext(
        resources=ResourceConfig(ram="5g", cpu=2),
        max_workers=min(CONSOLIDATE_DATA_SIZE_WORKERS, len(shard_cache_paths)),
        name="levanter-cache-probe",
    )
    probe_results = probe_ctx.execute(
        Dataset.from_list(shard_cache_paths).map(_probe_shard),
    ).results
    per_shard_field_counts = [r[0] for r in probe_results]
    shard_ledgers = [r[1] for r in probe_results]

    return _merge_sharded_ledgers(output_path, shard_cache_paths, shard_ledgers, per_shard_field_counts, metadata)


def _merge_materialized_ledgers(
    output_path: str,
    shard_cache_paths: List[str],
    shard_ledgers: List[CacheLedger],
    per_shard_field_counts: List[Dict[str, int]],
    metadata: CacheMetadata,
) -> CacheLedger:
    final_ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        finished_shards=[],
        field_counts={},
        metadata=metadata,
    )
    for shard_path, ledger, field_counts in zip(shard_cache_paths, shard_ledgers, per_shard_field_counts, strict=True):
        shard_name = os.path.basename(shard_path)
        final_ledger.shard_rows[shard_name] = ledger.total_num_rows
        final_ledger.finished_shards.append(shard_name)
        final_ledger.total_num_rows += ledger.total_num_rows
        for field, count in field_counts.items():
            final_ledger.field_counts[field] = final_ledger.field_counts.get(field, 0) + count

    final_ledger.is_finished = True
    final_ledger._serialize_and_commit(output_path)
    return final_ledger


def _merge_sharded_ledgers(
    output_path: str,
    shard_cache_paths: List[str],
    shard_ledgers: List[CacheLedger],
    per_shard_field_counts: List[Dict[str, int]],
    metadata: CacheMetadata,
) -> CacheLedger:
    final_ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        finished_shards=[],
        field_counts={},
        field_counts_by_shard={},
        layout=CACHE_LAYOUT_SHARDED,
        metadata=metadata,
    )
    for shard_path, ledger, field_counts in zip(shard_cache_paths, shard_ledgers, per_shard_field_counts, strict=True):
        shard_name = _relative_shard_path(output_path, shard_path)
        if shard_name in final_ledger.shard_rows:
            raise ValueError(f"Multiple shard cache paths resolve to the same ledger shard path: {shard_name}")

        final_ledger.shard_rows[shard_name] = ledger.total_num_rows
        final_ledger.finished_shards.append(shard_name)
        final_ledger.total_num_rows += ledger.total_num_rows
        final_ledger.field_counts_by_shard[shard_name] = field_counts
        for field, count in field_counts.items():
            final_ledger.field_counts[field] = final_ledger.field_counts.get(field, 0) + count

    final_ledger.is_finished = True
    _validate_sharded_ledger(final_ledger)
    final_ledger._serialize_and_commit(output_path)
    return final_ledger


def _distributed_build_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions,
    metadata: CacheMetadata,
    is_leader: bool,
) -> CacheLedger:
    status = {"val": np.array(0, dtype=np.int32)}
    lock = threading.Lock()

    def broadcaster():
        while True:
            with lock:
                received = broadcast_one_to_all(status["val"], is_source=is_leader)
                status["val"] = received
                if received != 0:
                    break

            time.sleep(10)

        return status["val"]

    if is_leader:
        b_thread = threading.Thread(target=broadcaster, daemon=True)
        b_thread.start()

        try:
            ledger = build_cache(cache_dir, source, processor, options, metadata)
            with lock:
                status["val"] = np.array(1, dtype=np.int32)
        except Exception:
            with lock:
                status["val"] = np.array(-1, dtype=np.int32)
            raise
        finally:
            b_thread.join()
        return ledger
    else:
        status_out = broadcaster()
        if status_out == 1:
            return CacheLedger.load(cache_dir, metadata)
        elif status_out == -1:
            raise RuntimeError("Cache build failed on leader process.")
        else:
            raise RuntimeError("Unexpected status received during distributed cache build.")


def _expose_cache_rows(cache_path: str, exemplar: T, num_rows: int) -> None:
    cache = TreeStore.open(exemplar, cache_path, mode="a", cache_metadata=False)
    futures = jax.tree.leaves(jax.tree.map(lambda x: x.offsets[0].write(num_rows), cache.tree))
    for future in futures:
        future.result()


async def _extend_cache_with_other_cache(
    dest_path: str, source_path: str, exemplar: dict, data_offset_tree: PyTree[int], row_offset
) -> int:
    try:
        logger.info(f"Copying data from {source_path} to {dest_path}.")
        with _no_cache_read_context():
            dest = TreeStore.open(exemplar, dest_path, mode="a", cache_metadata=False)
            source = TreeStore.open(exemplar, source_path, mode="r", cache_metadata=True)

            source_num_rows = await source.async_len()

            async def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
                data_size = source_array.data_size
                data = source_array.data
                max_elems = 64 * 1024 * 1024
                await _copy_in_batches(dest_array.data, data_offset, data, data_size, max_elems)

            futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)
            await asyncio.gather(*jax.tree.leaves(futures))
            del dest, source
        gc.collect()
        logger.info(f"Finished copying data from {source_path} to {dest_path}.")
        return source_num_rows
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to copy data from {source_path} to {dest_path}: {e}")
        raise


async def _copy_in_batches(dest_array, dest_offset, src_array, src_len, elems_per_batch):
    start = 0
    out_start = dest_offset
    while start < src_len:
        num_to_copy = min(elems_per_batch, src_len - start)
        end = start + num_to_copy
        out_end = out_start + num_to_copy

        chunk = await src_array[start:end].read()
        await dest_array[out_start:out_end].write(chunk)
        del chunk

        start += num_to_copy
        out_start += num_to_copy


async def _consolidate_metadata(dest_path: str, exemplar: dict, shard_infos: list[dict]) -> None:
    dest = TreeStore.open(exemplar, dest_path, mode="a")
    start = time.monotonic()

    delay = 4
    while True:
        write_futures = []
        try:
            async with ts.Transaction() as txn:
                for info in shard_infos:
                    with _no_cache_read_context():
                        source = TreeStore.open(exemplar, info["path"], mode="r", cache_metadata=True)
                    source_num_rows = info["ledger"].total_num_rows
                    row_offset = info["row_offset"]

                    for dest_array, source_array, data_offset in zip(
                        jax.tree.leaves(dest.tree),
                        jax.tree.leaves(source.tree),
                        jax.tree.leaves(info["data_offset_tree"]),
                        strict=True,
                    ):
                        if source_array.shapes is not None:
                            assert dest_array.shapes is not None
                            source_shapes = await source_array.shapes[:source_num_rows].read()
                            out_end = row_offset + source_num_rows
                            write_futures.append(
                                dest_array.shapes.with_transaction(txn)[row_offset:out_end].write(source_shapes)
                            )

                        source_offsets = await source_array.offsets[1 : source_num_rows + 1].read()
                        source_offsets = np.asarray(source_offsets) + data_offset
                        out_end = 1 + row_offset + source_num_rows
                        write_futures.append(
                            dest_array.offsets.with_transaction(txn)[row_offset + 1 : out_end].write(source_offsets)
                        )

            await asyncio.gather(*write_futures)
            elapsed = time.monotonic() - start
            logger.info(f"Metadata consolidation complete: {len(shard_infos)} shards in {elapsed:.1f}s")
            break
        except ValueError as e:
            if "Please reduce your request rate." not in str(e):
                raise
            logger.info(f"Rate limit exceeded during metadata consolidation. Retrying in {delay}s.")
            await asyncio.sleep(delay)
            delay *= 2
            if delay > 120:
                raise


def _relative_shard_path(output_path: str, shard_path: str) -> str:
    if "://" in output_path or "://" in shard_path:
        prefix = output_path.rstrip("/") + "/"
        if shard_path.startswith(prefix):
            return shard_path[len(prefix) :]
        raise ValueError(f"Sharded cache path {shard_path} is not under output path {output_path}")

    output_abs = os.path.abspath(output_path)
    shard_abs = os.path.abspath(shard_path)
    try:
        common_path = os.path.commonpath([output_abs, shard_abs])
    except ValueError as exc:
        raise ValueError(f"Sharded cache path {shard_path} is not under output path {output_path}") from exc

    if shard_abs == output_abs or common_path != output_abs:
        raise ValueError(f"Sharded cache path {shard_path} is not under output path {output_path}")

    relative_path = os.path.relpath(shard_abs, output_abs)
    return relative_path


def _field_counts_from_store(store: TreeStore) -> Dict[str, int]:
    return _field_counts_from_data_sizes(jax.tree.map(lambda array: array.data_size, store.tree))


def _field_counts_from_data_sizes(data_sizes) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path, value in jtu.tree_leaves_with_path(data_sizes):
        field = "/".join(_render_path_elem(part) for part in path)
        counts[field] = int(value)
    return counts


def _render_path_elem(path_elem) -> str:
    match path_elem:
        case jtu.DictKey(key):
            return str(key)
        case jtu.GetAttrKey(key):
            return str(key)
        case jtu.SequenceKey(i):
            return str(i)
        case jtu.FlattenedIndexKey(i):
            return str(i)
        case _:
            return str(path_elem)


def _sanitize_shard_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return safe or "shard"


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        keys = list(batch.keys())
        values = list(batch.values())
        num_rows = len(values[0]) if values else 0
        return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]
    else:
        return list(batch)


def _try_load(path, metadata):
    try:
        ledger = CacheLedger.load(path, metadata)
        if ledger.is_finished:
            return ledger
        logger.debug(f"Cache exists but is not finished at {path}.")
        return None
    except FileNotFoundError:
        return None


__all__ = [
    "TreeCache",
    "build_or_load_cache",
    "SerialCacheWriter",
    "CacheLedger",
    "CacheMetadata",
    "CacheOptions",
    "consolidate_shard_cache_ledgers",
    "consolidate_shard_caches",
    "write_levanter_cache",
]
