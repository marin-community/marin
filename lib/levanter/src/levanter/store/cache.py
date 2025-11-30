# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import copy
import dataclasses
import logging as pylogging
import operator
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import deepdiff
import fsspec.core
import jax
import numpy as np
import pyarrow as pa
import tensorstore as ts
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from jaxtyping import PyTree
from zephyr import Dataset, flow_backend

from levanter.data import batched
from levanter.data.dataset import AsyncDataset

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data.sharded_datasource import ShardedDataSource
from ..utils.fsspec_utils import exists as fsspec_exists
from ..utils.fsspec_utils import remove as fsspec_remove
from .jagged_array import JaggedArrayStore
from .tree_store import TreeStore

T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"

DEFAULT_LOG_LEVEL = pylogging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


@dataclass_json
@dataclass(frozen=True)
class CacheOptions:
    """
    Configuration for a cache build.

    The legacy implementation allowed exposing prefixes of partially built caches.
    The simplified zephyr-based pipeline always materializes the full cache before it
    is readable, but we keep the same options structure for compatibility.
    """

    num_shard_groups: Optional[int] = 128
    target_size_per_flush: int | str = "512MB"
    batch_size: int = 128

    @property
    def target_bytes_per_flush(self):
        if isinstance(self.target_size_per_flush, int):
            return self.target_size_per_flush
        import humanfriendly

        return humanfriendly.parse_size(self.target_size_per_flush)

    @staticmethod
    def default():
        return CacheOptions()

    @staticmethod
    def no_fanciness(batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = 128
        return CacheOptions(num_shard_groups=None, batch_size=batch_size)

    @staticmethod
    def one_group():
        return CacheOptions(num_shard_groups=1, batch_size=128)


def build_or_load_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    options: CacheOptions = CacheOptions.default(),
) -> "TreeCache[U]":
    """
    Build a sharded cache of a dataset using a simplified zephyr-based pipeline.

    The new pipeline processes each shard independently, writes temporary per-shard
    caches, and then consolidates them into a single TreeStore once all shards finish.
    Unlike the legacy implementation, partial caches are not exposed for reading.
    """
    if monitors is None:
        monitors = [LoggerMetricsMonitor()]

    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    try:
        return TreeCache.load(cache_dir, processor.output_exemplar, metadata)
    except FileNotFoundError:
        logger.info(f"Cache not found at {cache_dir}. Building with zephyr pipeline.")

    if await_finished:
        ledger = _build_cache_with_zephyr(cache_dir, source, processor, options, metadata, monitors)
        return TreeCache(cache_dir, processor.output_exemplar, ledger, None)

    build_future: concurrent.futures.Future[CacheLedger] = concurrent.futures.Future()

    def _runner():
        try:
            ledger = _build_cache_with_zephyr(cache_dir, source, processor, options, metadata, monitors)
            build_future.set_result(ledger)
        except Exception as exc:  # noqa: BLE001
            _safe_remove(os.path.join(cache_dir, "__shards__"))
            if not build_future.done():
                build_future.set_exception(exc)

    threading.Thread(target=_runner, daemon=True).start()
    return TreeCache(cache_dir=cache_dir, exemplar=processor.output_exemplar, ledger=None, _build_future=build_future)


class TreeCache(AsyncDataset[T_co]):
    ledger: Optional["CacheLedger"]

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: Optional["CacheLedger"],
        _build_future: Optional[concurrent.futures.Future["CacheLedger"]],
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._exemplar = exemplar
        self._build_future = _build_future
        self._store_future: concurrent.futures.Future[TreeStore] = concurrent.futures.Future()
        self._metrics_monitors: list[MetricsMonitor] = []

        if ledger is not None and ledger.is_finished:
            store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=False)
            self._store_future.set_result(store)

    @property
    def store(self) -> TreeStore[T_co]:
        self._ensure_store_ready()
        return self._store_future.result()

    async def store_async(self) -> TreeStore[T_co]:
        if self._store_future.done():
            return self._store_future.result()
        await asyncio.wrap_future(self._store_future)
        return self._store_future.result()

    def _ensure_store_ready(self, timeout: Optional[float] = None):
        if self._store_future.done():
            return
        if self._build_future is None:
            raise FileNotFoundError("Cache is not finished building.")
        ledger = self._build_future.result(timeout=timeout)
        self.ledger = ledger
        store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=False)
        self._store_future.set_result(store)
        for monitor in self._metrics_monitors:
            monitor(_ledger_to_metrics(ledger))

    async def async_len(self) -> int:
        self._ensure_store_ready()
        return len(await self.store_async())

    def __len__(self):
        self._ensure_store_ready()
        return len(self.store)

    async def final_length_is_known(self) -> bool:
        if self.ledger is None:
            return False
        if self.ledger.is_finished:
            return True
        if self._build_future is not None:
            await asyncio.wrap_future(self._build_future)
            return True
        return False

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> int:
        if self._store_future.done():
            return len(await self.store_async())
        if self._build_future is not None and self._build_future.done():
            self._ensure_store_ready()
            return len(self.store)
        if self.ledger is not None:
            return self.ledger.total_num_rows
        return 0

    def __getitem__(self, item):
        self._ensure_store_ready()
        return self.store[item]

    async def get_batch(self, indices: Sequence[int] | slice):
        if self._build_future is not None:
            await asyncio.wrap_future(self._build_future)
        self._ensure_store_ready()

        if isinstance(indices, slice):
            indices = range(indices.start or 0, indices.stop or len(self), indices.step or 1)
        max_index = max(indices)
        await self.wait_until_len_at_least(max_index + 1)
        return await self.store.get_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        if self._build_future is not None:
            self._build_future.result(timeout=timeout)
        self._ensure_store_ready(timeout=timeout)

        if isinstance(indices_or_slice, slice):
            indices_or_slice = range(
                indices_or_slice.start or 0,
                indices_or_slice.stop or len(self),
                indices_or_slice.step or 1,
            )
        max_index = max(indices_or_slice)
        if max_index >= len(self.store):
            raise IndexError(f"Index {max_index} out of bounds for cache of size {len(self.store)}")
        return self.store.get_batch_sync(indices_or_slice)

    def await_finished(self, timeout: Optional[float] = None, await_cleanup: bool = False):
        if self._build_future is not None:
            # propagate exceptions
            self._build_future.result(timeout=timeout)
            self._ensure_store_ready(timeout=timeout)

    async def finished(self):
        if self._build_future is not None:
            await asyncio.wrap_future(self._build_future)
            self._ensure_store_ready()

    def finished_sentinel(self):
        if self._build_future is None:
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut
        return asyncio.wrap_future(self._build_future)

    def attach_metrics_monitor(self, monitor: MetricsMonitor):
        if self._store_future.done():
            monitor(_ledger_to_metrics(self.ledger))  # type: ignore[arg-type]
            return
        self._metrics_monitors.append(monitor)

    @staticmethod
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        logger.info(f"Loading cache from {cache_dir}")
        time_in = os.times()[4]
        ledger = CacheLedger.load(cache_dir, options)
        time_out = os.times()[4]
        if time_out - time_in > 4:
            logger.info(f"Loaded cache ledger in {time_out - time_in:.2f}s")

        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: Optional["CacheOptions"] = None,
    ) -> "TreeCache[U]":
        if options is None:
            options = CacheOptions.default()
        return build_or_load_cache(cache_dir, shard_source, processor, await_finished=True, options=options)

    @property
    def is_finished(self):
        if self.ledger is not None and self.ledger.is_finished:
            return True
        if self._build_future is not None and self._build_future.done() and not self._build_future.exception():
            return True
        return False


@dataclass_json
@dataclass
class CacheLedger:
    total_num_rows: int
    shard_rows: Dict[str, int]
    is_finished: bool = False
    finished_shards: List[str] = dataclasses.field(default_factory=list)
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
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
            with fsspec.open(ledger_path) as file:
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
    ):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._exemplar = exemplar
        self._shard_name = shard_name
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="w", cache_metadata=True)
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
    fs: AbstractFileSystem = fsspec.core.url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        fs.copy(path, f"{path}.bak")

    for _ in range(10):
        try:
            with fsspec.open(path, "w") as file:
                file.write(obj.to_json())
            break
        except FileNotFoundError:
            logger.exception(f"Failed to write {path}")


def _build_cache_with_zephyr(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions,
    metadata: CacheMetadata,
    monitors: Sequence[MetricsMonitor],
) -> CacheLedger:
    pylogging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    backend = flow_backend()
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
        _notify_monitors(monitors, ledger)
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

    shard_results = list(backend.execute(Dataset.from_list(shard_jobs).map(process_shard), verbose=False))
    shard_results = sorted(shard_results, key=lambda r: r["index"])

    ledger = _consolidate_shard_caches(shard_results, cache_dir, processor.output_exemplar, metadata)
    _safe_remove(temp_root)
    _notify_monitors(monitors, ledger)
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
    existing = _try_load(shard_path)
    if existing is not None:
        logger.info(f"Found existing shard cache for {shard_name} at {shard_path}. Skipping build.")
        return {"shard_name": shard_name, "path": shard_path, "ledger": existing, "index": shard_index}

    logger.info(f"Building shard {shard_name} -> {shard_path}")
    iterator = source.open_shard_at_row(shard_name, 0)
    with SerialCacheWriter(shard_path, processor.output_exemplar, metadata=metadata, shard_name=shard_name) as writer:
        for batch in batched(iterator, options.batch_size):
            processed = processor(batch)
            writer.write_batch(_canonicalize_batch(processed))

    ledger = CacheLedger.load(shard_path, metadata)
    return {"shard_name": shard_name, "path": shard_path, "ledger": ledger, "index": shard_index}


def _consolidate_shard_caches(
    shard_results: list[dict],
    output_path: str,
    exemplar,
    metadata: CacheMetadata,
) -> CacheLedger:
    if not shard_results:
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

    logger.info(f"Consolidating {len(shard_results)} shard caches into {output_path}")

    first_cache = TreeStore.open(exemplar, shard_results[0]["path"], mode="r", cache_metadata=True)
    data_offset_tree = jax.tree.map(lambda x: 0, first_cache.tree)

    shard_info: list[dict] = []
    total_rows = 0
    for shard in shard_results:
        ledger: CacheLedger = shard["ledger"]
        shard_info.append(
            {
                "path": shard["path"],
                "shard_name": shard["shard_name"],
                "row_offset": total_rows,
                "data_offset_tree": copy.deepcopy(data_offset_tree),
                "ledger": ledger,
            }
        )
        total_rows += ledger.total_num_rows
        this_cache = TreeStore.open(exemplar, shard["path"], mode="r", cache_metadata=True)
        this_offsets = jax.tree.map(lambda x: x.data_size, this_cache.tree)
        data_offset_tree = jax.tree.map(operator.add, data_offset_tree, this_offsets)

    TreeStore.open(exemplar, output_path, mode="w", cache_metadata=True)

    for shard in shard_info:
        asyncio.run(
            extend_cache_with_other_cache(
                output_path, shard["path"], exemplar, shard["data_offset_tree"], shard["row_offset"]
            )
        )
        asyncio.run(
            extend_cache_metadata_with_other(
                output_path, shard["path"], exemplar, shard["data_offset_tree"], shard["row_offset"]
            )
        )

    final_ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        finished_shards=[],
        field_counts={},
        metadata=metadata,
    )
    for shard in shard_info:
        shard_ledger: CacheLedger = shard["ledger"]
        final_ledger.shard_rows[shard["shard_name"]] = shard_ledger.total_num_rows
        final_ledger.finished_shards.append(shard["shard_name"])
        final_ledger.total_num_rows += shard_ledger.total_num_rows
        for field, count in shard_ledger.field_counts.items():
            final_ledger.field_counts[field] = final_ledger.field_counts.get(field, 0) + count

    final_ledger.is_finished = True
    final_ledger._serialize_and_commit(output_path)
    expose_cache_rows(output_path, exemplar, final_ledger.total_num_rows)
    return final_ledger


def _safe_remove(path: str):
    try:
        if fsspec_exists(path):
            fsspec_remove(path, recursive=True)
    except Exception:  # noqa: BLE001
        logger.exception(f"Failed to remove temporary cache path {path}")


def expose_cache_rows(cache_path: str, exemplar: T, num_rows: int) -> None:
    cache = TreeStore.open(exemplar, cache_path, mode="a", cache_metadata=False)
    _expose_available_rows(cache, num_rows)


def merge_ledgers(dest: CacheLedger, source: CacheLedger) -> CacheLedger:
    assert not dest.is_finished
    dest.total_num_rows += source.total_num_rows
    for shard, rows in source.shard_rows.items():
        current_value = dest.shard_rows.get(shard, 0)
        if current_value != 0:
            raise ValueError(f"Shard {shard} already has {current_value} rows")
        dest.shard_rows[shard] = rows

    dest.finished_shards.extend(source.finished_shards)
    for field, count in source.field_counts.items():
        dest.field_counts[field] = dest.field_counts.get(field, 0) + count

    return dest


def _expose_available_rows(permanent_cache, num_available_rows):
    futures = jax.tree.leaves(jax.tree.map(lambda x: x.offsets[0].write(num_available_rows), permanent_cache.tree))
    for future in futures:
        future.result()


async def extend_cache_with_other_cache(
    dest_path: str, source_path: str, exemplar: dict, data_offset_tree: PyTree[int], row_offset
) -> int:
    try:
        logger.info(f"Copying data from {source_path} to {dest_path}.")
        dest = TreeStore.open(exemplar, dest_path, mode="a", cache_metadata=False)
        source = TreeStore.open(exemplar, source_path, mode="r", cache_metadata=True)

        source_num_rows = await source.async_len()

        async def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
            data_size = source_array.data_size
            data = source_array.data
            MAX_ELEMS = 1024 * 1024 * 1024
            await _copy_in_batches(dest_array.data, data_offset, data, data_size, MAX_ELEMS)

        futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)
        await asyncio.gather(*jax.tree.leaves(futures))
        logger.info(f"Finished copying data from {source_path} to {dest_path}.")
        return source_num_rows
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to copy data from {source_path} to {dest_path}: {e}")
        raise


async def _copy_in_batches(dest_array, dest_offset, src_array, src_len, elems_per_batch):
    last_future: ts.Future | None = None
    start = 0
    out_start = dest_offset
    while start < src_len:
        if last_future is not None:
            await last_future
        async with ts.Transaction() as txn:
            num_to_copy = min(elems_per_batch, src_len - start)
            end = start + num_to_copy
            out_end = out_start + num_to_copy

            last_future = dest_array.with_transaction(txn)[out_start:out_end].write(src_array[start:end])
            start += num_to_copy
            out_start += num_to_copy

    if last_future is not None:
        await last_future


async def extend_cache_metadata_with_other(
    dest_path: str, source_path: str, exemplar: dict, data_offset_tree: PyTree[int], row_offset
) -> int:
    try:
        logger.info(f"Copying metadata from {source_path} to {dest_path}.")
        dest = TreeStore.open(exemplar, dest_path, mode="a")
        source = TreeStore.open(exemplar, source_path, mode="r", cache_metadata=True)

        source_num_rows = await source.async_len()

        async def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
            if source_array.shapes is not None:
                source_shapes = source_array.shapes
                async with ts.Transaction() as txn:
                    dest_shapes = dest_array.shapes
                    assert dest_shapes is not None
                    out_end = row_offset + source_num_rows
                    shape_future = dest_shapes.with_transaction(txn)[row_offset:out_end].write(source_shapes)

            source_offsets = source_array.offsets[1 : source_num_rows + 1][ts.d[:].translate_to[0]]
            source_offsets = _virtual_offset(source_offsets, data_offset)

            delay = 4
            while True:
                try:
                    async with ts.Transaction() as txn:
                        dest_offsets = dest_array.offsets
                        out_end = 1 + row_offset + source_num_rows
                        offset_future = dest_offsets.with_transaction(txn)[row_offset + 1 : out_end].write(
                            source_offsets
                        )
                    break
                except ValueError as e:
                    if "Please reduce your request rate." in str(e):
                        logger.info("Rate limit exceeded. Retrying.")
                        await asyncio.sleep(delay)
                        delay *= 2
                        if delay > 120:
                            raise
            await offset_future
            if source_array.shapes is not None:
                await shape_future

        futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)

        await asyncio.gather(*jax.tree.leaves(futures))
        logger.info(f"Finished copying metadata from {source_path} to {dest_path}.")
        return source_num_rows
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to copy metadata from {source_path} to {dest_path}: {e}")
        raise


def _virtual_offset(base: ts.TensorStore, offset_amount):
    async def do_read(domain: ts.IndexDomain, array: np.ndarray, read_params: ts.VirtualChunkedReadParameters):
        array[...] = (await base[domain].read()) + offset_amount

    return ts.virtual_chunked(do_read, dtype=base.dtype, domain=base.domain, shape=base.shape)


def _sanitize_shard_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return safe or "shard"


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        return _to_list_of_dicts(batch)
    else:
        return list(batch)


def _to_list_of_dicts(batch: dict) -> List[dict]:
    keys = list(batch.keys())
    values = list(batch.values())
    num_rows = len(values[0]) if values else 0
    return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]


def _ledger_to_metrics(ledger: CacheLedger) -> InProgressCacheMetrics:
    return InProgressCacheMetrics(
        rows_finished=ledger.total_num_rows,
        is_finished=ledger.is_finished,
        shards_finished=len(ledger.finished_shards),
        field_counts=ledger.field_counts,
    )


def _try_load(path):
    try:
        ledger = CacheLedger.load(path)
        if ledger.is_finished:
            return ledger
        logger.debug(f"Cache exists but is not finished at {path}.")
        return None
    except FileNotFoundError:
        return None


def _notify_monitors(monitors: Sequence[MetricsMonitor], ledger: CacheLedger):
    metrics = _ledger_to_metrics(ledger)
    for monitor in monitors:
        monitor(metrics)


__all__ = [
    "TreeCache",
    "build_or_load_cache",
    "SerialCacheWriter",
    "CacheLedger",
    "CacheMetadata",
    "CacheOptions",
    "merge_ledgers",
    "expose_cache_rows",
    "extend_cache_with_other_cache",
    "extend_cache_metadata_with_other",
]
