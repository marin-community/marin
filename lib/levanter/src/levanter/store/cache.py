# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import dataclasses
import logging as pylogging
import operator
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import deepdiff
import fsspec.core
import jax
import numpy as np
import pyarrow as pa
import tensorstore as ts
from dataclasses_json import dataclass_json
from fray.v2 import ResourceConfig
from fsspec import AbstractFileSystem
from jaxtyping import PyTree
from tqdm_loggable.tqdm_logging import tqdm_logging
from zephyr import Dataset, ZephyrContext
from zephyr.writers import write_levanter_cache

from levanter.data.dataset import AsyncDataset
from levanter.utils.jax_utils import broadcast_one_to_all

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
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

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> int:
        return len(self.store)

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

    with ZephyrContext(
        resources=ResourceConfig(ram="32g", disk="16g"),
        num_workers=min(128, len(shard_jobs)),
        name="levanter-cache-build",
    ) as ctx:
        shard_results = ctx.execute(Dataset.from_list(shard_jobs).map(process_shard), verbose=False)
    shard_results = sorted(shard_results, key=lambda r: r["index"])

    shard_cache_paths = [s["path"] for s in shard_results]
    ledger = consolidate_shard_caches(
        shard_cache_paths=shard_cache_paths,
        output_path=cache_dir,
        exemplar=processor.output_exemplar,
        metadata=metadata,
    )
    _safe_remove(temp_root)
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
) -> CacheLedger:
    """
    Consolidate multiple shard caches into a single cache directory.

    Args:
        shard_cache_paths: List of shard cache directories.
        output_path: Destination cache directory.
        exemplar: Output exemplar structure.
        metadata: CacheMetadata to use for the final ledger.
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
            metadata=metadata,
        )
        ledger._serialize_and_commit(output_path)
        return ledger

    logger.info(f"Consolidating {len(shard_cache_paths)} shard caches into {output_path}")

    first_cache = TreeStore.open(exemplar, shard_cache_paths[0], mode="r", cache_metadata=True)
    data_offset_tree = jax.tree.map(lambda x: 0, first_cache.tree)

    shard_info: list[dict] = []
    total_rows = 0

    shard_ledgers = [CacheLedger.load(p, metadata) for p in shard_cache_paths]

    for shard_path, ledger in zip(shard_cache_paths, shard_ledgers):
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

        this_cache = TreeStore.open(exemplar, shard_path, mode="r", cache_metadata=True)
        this_offsets = jax.tree.map(lambda x: x.data_size, this_cache.tree)
        data_offset_tree = jax.tree.map(operator.add, data_offset_tree, this_offsets)

    TreeStore.open(exemplar, output_path, mode="w", cache_metadata=True)

    def _copy_shard(info: dict):
        asyncio.run(
            _extend_cache_with_other_cache(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )

    with ZephyrContext(
        resources=ResourceConfig(ram="32g", disk="16g"),
        num_workers=min(128, len(shard_info)),
        name="levanter-cache-copy",
    ) as ctx:
        ctx.execute(
            Dataset.from_list(shard_info).map(_copy_shard),
            verbose=True,
        )

    # do metadata serially b/c of write amplification concerns
    for info in shard_info:
        asyncio.run(
            _extend_cache_metadata_with_other(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )

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


async def _extend_cache_metadata_with_other(
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
    "consolidate_shard_caches",
]
