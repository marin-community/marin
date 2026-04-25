# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import bisect
import copy
import dataclasses
import gc
import logging as pylogging
import operator
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import deepdiff
import jax
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
from zephyr.writers import write_levanter_cache

from levanter.data.dataset import AsyncDataset
from levanter.utils.jax_utils import broadcast_one_to_all
from levanter.utils.thread_utils import blocking_wait

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.sharded_datasource import ShardedDataSource
from ..utils.fsspec_utils import exists as fsspec_exists
from ..utils.fsspec_utils import remove as fsspec_remove
from .jagged_array import JaggedArrayStore, _no_cache_read_context
from .tree_store import TreeStore

T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"
CONSOLIDATE_DATA_SIZE_WORKERS = 32

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

        if not ledger.is_finished:
            raise RuntimeError(f"Cache at {cache_dir} is not finished.")

        self._store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=False)

    @property
    def store(self) -> TreeStore[T_co]:
        return self._store

    async def async_len(self) -> int:
        return len(self.store)

    def __len__(self):
        return len(self.store)

    def is_finite(self) -> bool:
        return True

    def __getitem__(self, item):
        return self.store[item]

    async def get_batch(self, indices: Sequence[int] | slice):
        if isinstance(indices, slice):
            indices = range(indices.start or 0, indices.stop or len(self), indices.step or 1)
        return await self.store.get_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        if isinstance(indices_or_slice, slice):
            indices_or_slice = range(
                indices_or_slice.start or 0,
                indices_or_slice.stop or len(self),
                indices_or_slice.step or 1,
            )
        return self.store.get_batch_sync(indices_or_slice)

    @staticmethod
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        logger.info(f"Loading cache from {cache_dir}")
        ledger = CacheLedger.load(cache_dir, options)

        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        if ledger.shard_paths is not None:
            return ShardedTreeCache(ledger.shard_paths, exemplar, ledger)  # type: ignore[return-value]
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


class _VirtualRead:
    def __init__(self, read_async):
        self._read_async = read_async

    def read(self):
        return self

    def __await__(self):
        return self._read_async().__await__()

    def result(self):
        return blocking_wait(self._read_async())


class _ShardedArray:
    def __init__(self, arrays, sizes: list[int]):
        self._arrays = arrays
        self._sizes = sizes
        self._boundaries = _cumulative_offsets(sizes)

    def __getitem__(self, item):
        return _VirtualRead(lambda: self._read(item))

    async def _read(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._boundaries[-1])
            if step != 1:
                values = await self._read(slice(start, stop))
                return values[::step]
            pieces = []
            for shard_index, local_slice in _split_slice_by_boundaries(start, stop, self._boundaries):
                pieces.append(await self._arrays[shard_index][local_slice].read())
            return _concatenate_or_empty(pieces)

        index = item
        if index < 0:
            index += self._boundaries[-1]
        if index < 0 or index >= self._boundaries[-1]:
            raise IndexError("Index out of bounds")
        shard_index = bisect.bisect_right(self._boundaries, index) - 1
        local_index = index - self._boundaries[shard_index]
        return await self._arrays[shard_index][local_index].read()


class _ShardedOffsets:
    def __init__(self, stores: list[JaggedArrayStore]):
        self._stores = stores
        self._num_rows = sum(store.num_rows for store in stores)
        self._data_sizes = [store.data_size for store in stores]

    def __getitem__(self, item):
        return _VirtualRead(lambda: self._read(item))

    async def _read(self, item):
        offsets = await self._full_offsets()
        return offsets[item]

    async def _full_offsets(self):
        offset_reads = [store.offsets[0 : store.num_rows + 1].read() for store in self._stores]
        per_shard_offsets = await asyncio.gather(*offset_reads)
        adjusted_offsets = [np.asarray([self._num_rows], dtype=np.int64)]
        data_base = 0
        for offsets, data_size in zip(per_shard_offsets, self._data_sizes):
            offsets = np.asarray(offsets, dtype=np.int64)
            offsets[0] = 0
            adjusted_offsets.append(offsets[1:] + data_base)
            data_base += data_size
        return np.concatenate(adjusted_offsets)


class _ShardedShapes:
    def __init__(self, stores: list[JaggedArrayStore]):
        self._stores = stores
        self._sizes = [store.num_rows for store in stores]
        self._boundaries = _cumulative_offsets(self._sizes)

    def __getitem__(self, item):
        return _VirtualRead(lambda: self._read(item))

    async def _read(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._boundaries[-1])
            if step != 1:
                values = await self._read(slice(start, stop))
                return values[::step]
            pieces = []
            for shard_index, local_slice in _split_slice_by_boundaries(start, stop, self._boundaries):
                shapes = self._stores[shard_index].shapes
                assert shapes is not None
                pieces.append(await shapes[local_slice].read())
            return _concatenate_or_empty(pieces)

        index = item
        if index < 0:
            index += self._boundaries[-1]
        if index < 0 or index >= self._boundaries[-1]:
            raise IndexError("Index out of bounds")
        shard_index = bisect.bisect_right(self._boundaries, index) - 1
        local_index = index - self._boundaries[shard_index]
        shapes = self._stores[shard_index].shapes
        assert shapes is not None
        return await shapes[local_index].read()


class ShardedJaggedArrayStore:
    """Virtual JaggedArrayStore backed by multiple shard-local stores."""

    def __init__(self, stores: list[JaggedArrayStore]):
        if not stores:
            raise ValueError("ShardedJaggedArrayStore requires at least one store")
        self._stores = stores
        self.item_rank = stores[0].item_rank
        self.offsets = _ShardedOffsets(stores)
        self.data = _ShardedArray([store.data for store in stores], [store.data_size for store in stores])
        self.shapes = _ShardedShapes(stores) if stores[0].shapes is not None else None

    @property
    def num_rows(self):
        return sum(store.num_rows for store in self._stores)

    async def num_rows_async(self):
        return self.num_rows

    @property
    def data_size(self):
        return sum(store.data_size for store in self._stores)

    async def data_size_async(self):
        return self.data_size

    def __len__(self):
        return self.num_rows

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return self.get_batch_sync(list(range(start, stop, step)))
        shard_index, local_index = self._resolve_row(item)
        return self._stores[shard_index][local_index]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[np.ndarray]:
        shard_groups = _group_indices_by_shard(indices, self._row_boundaries())

        results: list[None | np.ndarray] = [None] * len(indices)

        async def fetch_shard(shard_index: int, items: list[tuple[int, int]]):
            local_indices = [local_index for _, local_index in items]
            batch = await self._stores[shard_index].get_batch(local_indices)
            for (position, _), value in zip(items, batch):
                results[position] = value

        await asyncio.gather(*[fetch_shard(shard_index, items) for shard_index, items in shard_groups.items()])
        return results

    def get_batch_sync(self, indices: Sequence[int]) -> Sequence[np.ndarray]:
        shard_groups = _group_indices_by_shard(indices, self._row_boundaries())
        results: list[None | np.ndarray] = [None] * len(indices)
        for shard_index, items in shard_groups.items():
            local_indices = [local_index for _, local_index in items]
            batch = self._stores[shard_index].get_batch_sync(local_indices)
            for (position, _), value in zip(items, batch):
                results[position] = value
        return results

    def _resolve_row(self, index: int) -> tuple[int, int]:
        boundaries = self._row_boundaries()
        if index < 0:
            index += boundaries[-1]
        if index < 0 or index >= boundaries[-1]:
            raise IndexError("Index out of bounds")
        shard_index = bisect.bisect_right(boundaries, index) - 1
        return shard_index, index - boundaries[shard_index]

    def _row_boundaries(self):
        return _cumulative_offsets([store.num_rows for store in self._stores])


class ShardedTreeStore:
    """Virtual TreeStore backed by multiple shard-local TreeStores."""

    def __init__(self, stores: list[TreeStore]):
        if not stores:
            raise ValueError("ShardedTreeStore requires at least one store")
        self.path = stores[0].path
        self.mode = "r"
        self._stores = stores
        self.tree = jax.tree.map(
            lambda *leaves: ShardedJaggedArrayStore(list(leaves)), *[store.tree for store in stores]
        )

    def __len__(self):
        return len(jax.tree.leaves(self.tree)[0])

    async def async_len(self):
        return len(self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return self.get_batch_sync(list(range(start, stop, step)))
        return jax.tree.map(lambda reader: reader[item], self.tree)

    async def get_batch(self, indices) -> List[T_co]:
        grouped = jax.tree.map(lambda reader: reader.get_batch(indices), self.tree)
        leaves, structure = jax.tree.flatten(grouped)
        awaited_leaves = await asyncio.gather(*leaves)
        return [jax.tree.unflatten(structure, [leaf[i] for leaf in awaited_leaves]) for i in range(len(indices))]

    def get_batch_sync(self, indices) -> List[T_co]:
        grouped = jax.tree.map(lambda reader: reader.get_batch_sync(indices), self.tree)
        return [jax.tree.map(lambda _, leaf: leaf[i], self.tree, grouped) for i in range(len(indices))]


class ShardedTreeCache(AsyncDataset[T_co]):
    """Reads across multiple shard caches without requiring a consolidation step.

    Each shard is an independent TreeStore directory. This class maintains a
    cumulative row index to translate global indices into (shard, local_index).
    """

    def __init__(self, shard_paths: list[str], exemplar: T_co, ledger: "CacheLedger"):
        super().__init__()
        self._exemplar = exemplar
        self.ledger = ledger
        self._shard_paths = shard_paths

        # Build cumulative row boundaries: [0, rows_shard0, rows_shard0+rows_shard1, ...]
        self._cum_rows: list[int] = [0]
        self._stores: list[TreeStore] = []
        for path in shard_paths:
            shard_name = os.path.basename(path)
            rows = ledger.shard_rows.get(shard_name, 0)
            self._cum_rows.append(self._cum_rows[-1] + rows)
            self._stores.append(TreeStore.open(exemplar, path, mode="r", cache_metadata=False))
        self._store = ShardedTreeStore(self._stores)

    @property
    def store(self) -> ShardedTreeStore:
        return self._store

    def _resolve_index(self, global_idx: int) -> tuple[int, int]:
        """Return (shard_index, local_row) for a global row index."""
        if global_idx < 0 or global_idx >= self._cum_rows[-1]:
            raise IndexError(f"Index {global_idx} out of range [0, {self._cum_rows[-1]})")
        shard_idx = bisect.bisect_right(self._cum_rows, global_idx) - 1
        local_idx = global_idx - self._cum_rows[shard_idx]
        return shard_idx, local_idx

    def __len__(self):
        return self._cum_rows[-1]

    async def async_len(self) -> int:
        return len(self)

    def is_finite(self) -> bool:
        return True

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.get_batch_sync(item)
        shard_idx, local_idx = self._resolve_index(item)
        return self._stores[shard_idx][local_idx]

    async def get_batch(self, indices: Sequence[int] | slice) -> Sequence:
        if isinstance(indices, slice):
            indices = range(indices.start or 0, indices.stop or len(self), indices.step or 1)

        # Group indices by shard, preserving original order
        shard_groups: dict[int, list[tuple[int, int]]] = {}  # shard_idx -> [(position_in_output, local_idx)]
        for pos, global_idx in enumerate(indices):
            shard_idx, local_idx = self._resolve_index(global_idx)
            shard_groups.setdefault(shard_idx, []).append((pos, local_idx))

        # Fetch per-shard batches concurrently
        results: list[None | Any] = [None] * len(indices)

        async def _fetch_shard(shard_idx: int, items: list[tuple[int, int]]):
            local_indices = [local_idx for _, local_idx in items]
            batch = await self._stores[shard_idx].get_batch(local_indices)
            for (pos, _), value in zip(items, batch):
                results[pos] = value

        await asyncio.gather(*[_fetch_shard(si, items) for si, items in shard_groups.items()])
        return results

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        if isinstance(indices_or_slice, slice):
            indices_or_slice = range(
                indices_or_slice.start or 0,
                indices_or_slice.stop or len(self),
                indices_or_slice.step or 1,
            )
        # Group by shard, fetch per-shard, reassemble
        shard_groups: dict[int, list[tuple[int, int]]] = {}
        for pos, global_idx in enumerate(indices_or_slice):
            shard_idx, local_idx = self._resolve_index(global_idx)
            shard_groups.setdefault(shard_idx, []).append((pos, local_idx))

        results: list[None | Any] = [None] * len(indices_or_slice)
        for shard_idx, items in shard_groups.items():
            local_indices = [local_idx for _, local_idx in items]
            batch = self._stores[shard_idx].get_batch_sync(local_indices)
            for (pos, _), value in zip(items, batch):
                results[pos] = value
        return results

    @property
    def is_finished(self):
        return True


def _cumulative_offsets(sizes: Sequence[int]) -> list[int]:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return offsets


def _split_slice_by_boundaries(start: int, stop: int, boundaries: Sequence[int]) -> list[tuple[int, slice]]:
    if start >= stop:
        return []
    pieces = []
    shard_index = bisect.bisect_right(boundaries, start) - 1
    while shard_index < len(boundaries) - 1 and start < stop:
        shard_start = boundaries[shard_index]
        shard_stop = boundaries[shard_index + 1]
        piece_stop = min(stop, shard_stop)
        if start < piece_stop:
            pieces.append((shard_index, slice(start - shard_start, piece_stop - shard_start)))
        start = piece_stop
        shard_index += 1
    return pieces


def _concatenate_or_empty(pieces: Sequence[np.ndarray]) -> np.ndarray:
    if not pieces:
        return np.asarray([])
    if len(pieces) == 1:
        return np.asarray(pieces[0])
    return np.concatenate(pieces)


def _group_indices_by_shard(indices: Sequence[int], boundaries: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    shard_groups: dict[int, list[tuple[int, int]]] = {}
    total_rows = boundaries[-1]
    for position, index in enumerate(indices):
        if index < 0:
            index += total_rows
        if index < 0 or index >= total_rows:
            raise IndexError("Index out of bounds")
        shard_index = bisect.bisect_right(boundaries, index) - 1
        local_index = index - boundaries[shard_index]
        shard_groups.setdefault(shard_index, []).append((position, local_index))
    return shard_groups


@dataclass_json
@dataclass
class CacheLedger:
    total_num_rows: int
    shard_rows: Dict[str, int]
    is_finished: bool = False
    finished_shards: List[str] = dataclasses.field(default_factory=list)
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    metadata: "CacheMetadata" = dataclasses.field(default_factory=lambda: CacheMetadata({}))
    shard_paths: Optional[List[str]] = None

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
    Writes TreeCache-compatible caches to disk without Ray. Mostly for scripts and debugging.
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
            field_counts={},
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

    temp_root = os.path.join(cache_dir, "__shards__")
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
    shard_ledgers = [s["ledger"] for s in shard_results]
    ledger = _merge_ledgers(cache_dir, shard_cache_paths, shard_ledgers, metadata)
    # Store shard paths so the reader can find them without consolidation
    ledger.shard_paths = shard_cache_paths
    ledger._serialize_and_commit(cache_dir)
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
    Consolidate multiple shard caches into a single cache directory.

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

    # Distributed: load ledger + read data_size for each shard in parallel.
    # Both operations are S3 I/O-bound; distributing across zephyr workers
    # avoids serializing thousands of S3 calls in the coordinator process.
    def _probe_shard(shard_path):
        ledger = CacheLedger.load(shard_path, metadata)
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

    # Serial: accumulate row_offset and data_offset_tree (order-dependent)
    for shard_path, ledger, this_offsets in zip(shard_cache_paths, shard_ledgers, per_shard_sizes):
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

    # Single shared transaction to coalesce metadata writes (see #4100, tensorstore#202)
    asyncio.run(_consolidate_metadata(output_path, exemplar, shard_info))

    final_ledger = _merge_ledgers(output_path, shard_cache_paths, shard_ledgers, metadata)
    # as a final step, set the total num rows in the final cache
    _expose_cache_rows(output_path, exemplar, final_ledger.total_num_rows)
    return final_ledger


def _merge_ledgers(
    output_path: str, shard_cache_paths: list[str], shard_ledgers: list[CacheLedger], metadata: CacheMetadata
) -> CacheLedger:
    final_ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        finished_shards=[],
        field_counts={},
        metadata=metadata,
    )
    for shard_path, ledger in zip(shard_cache_paths, shard_ledgers):
        shard_name = os.path.basename(shard_path)
        final_ledger.shard_rows[shard_name] = ledger.total_num_rows
        final_ledger.finished_shards.append(shard_name)
        final_ledger.total_num_rows += ledger.total_num_rows
        for field, count in ledger.field_counts.items():
            final_ledger.field_counts[field] = final_ledger.field_counts.get(field, 0) + count

    final_ledger.is_finished = True
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


def _safe_remove(path: str):
    try:
        if fsspec_exists(path):
            fsspec_remove(path, recursive=True)
    except Exception:  # noqa: BLE001
        logger.exception(f"Failed to remove temporary cache path {path}")


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
                MAX_ELEMS = 64 * 1024 * 1024
                await _copy_in_batches(dest_array.data, data_offset, data, data_size, MAX_ELEMS)

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

        # Materialize into numpy to avoid holding TensorStore internal references
        # across iterations. Direct ts-to-ts copy leaks ~14 MiB/shard (#4196).
        chunk = await src_array[start:end].read()
        await dest_array[out_start:out_end].write(chunk)
        del chunk

        start += num_to_copy
        out_start += num_to_copy


async def _consolidate_metadata(dest_path: str, exemplar: dict, shard_infos: list[dict]) -> None:
    """Copy metadata (offsets + shapes) from all shards into dest using a single shared transaction.

    Replaces the old per-shard loop that committed a transaction per shard, causing
    O(num_shards) read-modify-write cycles on the same zarr3 chunks (tensorstore#202).
    """
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
    "ShardedTreeCache",
    "build_or_load_cache",
    "SerialCacheWriter",
    "CacheLedger",
    "CacheMetadata",
    "CacheOptions",
    "consolidate_shard_caches",
]
