# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent
import copy
import dataclasses
import logging as pylogging
import operator
import os
import pprint
import random
import threading
import time
from asyncio import InvalidStateError
from concurrent.futures import Future as threading_Future
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

import deepdiff
import fsspec.core
import jax
import numpy as np
import pyarrow as pa
import ray
import tensorstore as ts
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from jaxtyping import PyTree
from ray.actor import ActorHandle
from ray.runtime_env import RuntimeEnv
from tqdm_loggable.auto import tqdm

from levanter.data import batched
from levanter.data.dataset import AsyncDataset

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data.sharded_datasource import ShardedDataSource
from ..utils.fsspec_utils import exists as fsspec_exists
from ..utils.fsspec_utils import remove as fsspec_remove
from ..utils.ray_utils import ExceptionInfo, SnitchRecipient, current_actor_handle, log_failures_to, ser_exc_info
from .jagged_array import JaggedArrayStore, PreparedBatch
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
    Configuration for a cache. This is used to configure a few parts of the cache creation process and to
    store metadata that can be checked to ensure that the cache being loaded was created with the expected
    configuration. It combined with the [[BatchProcessor]] metadata to form the [[CacheMetadata]].

    It is intended that caching it deterministic conditional on the input data, processor, and these options.
    """

    num_shard_groups: Optional[int] = 128

    # the below options don't actually impact the cache's result, but do impact construction
    target_size_per_flush: int | str = "512MB"
    """The number of bytes to buffer before flushing to disk. This is used to control the memory usage of the cache
    building process. Lower values will use less memory but could take somewhat longer to build the cache."""

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
        """
        For testing, disables all the fancy features of the cache. This makes it easier to predict the behavior
        """
        if batch_size is None:
            batch_size = 128
        return CacheOptions(num_shard_groups=None, batch_size=batch_size)

    @staticmethod
    def one_group():
        """
        For testing, disables all the fancy features of the cache. This makes it easier to predict the behavior
        """
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
    Produces a sharded cache of the dataset using Ray for distributed processing. The cache can be any path
    on any file system understood by fsspec.

    This system is designed with tokenization and similar processes in mind, but it can potentially be used for any kind
    of preprocessing that converts input batches to output batches. The main design goal is to make it easy to
    parallelize preprocessing across multiple machines while maintaining reproducibility and fault tolerance.
    Usually the machines in question are the ones doing the training, but they could be separate machines as well.

    See the [Dataloader Design Doc](https://github.com/stanford-crfm/levanter/blob/main/docs/design/Data-Loader-Design.md)
    for a somewhat out of date overview of the design.

    Args:
        cache_dir: The directory to write the cache to. This can be any path understood by fsspec.
        source: A ShardedDataset that will be used to read the input data. Conceptually, it's just a mapping
                    from shard names to iterators over the data in that shard.
        processor: A BatchProcessor that will be used to process batches of data. This is the main place where
                    you can customize the preprocessing pipeline.
        await_finished: If True, this function will block until the cache is finished. If False, it will return
                    immediately.
        monitors: a list of MetricsMonitors to attach to the cache. These will be called periodically with
            metrics about the cache build process. If None, will add a LoggerMetricsMonitor.

        options: Configuration for the cache. This is used to configure a few parts of the cache creation process

    Returns:
       (TreeCache) A TreeCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = TreeCache.build_or_load(
        cache_dir=cache_dir,
        shard_source=source,
        processor=processor,
        options=options,
    )

    if cache.is_finished:
        logger.info("Cache already finished. Skipping.")
        return cache

    if monitors is None:
        monitors = [LoggerMetricsMonitor()]

    for monitor in monitors:
        cache.attach_metrics_monitor(monitor)

    while await_finished:
        try:
            cache.await_finished(None, await_cleanup=True)
            break
        except TimeoutError:
            pass

    return cache


class TreeCache(AsyncDataset[T_co]):
    ledger: Optional["CacheLedger"]
    _builder: Optional[ActorHandle]  # handle of _TreeStoreCacheBuilder
    # monitor_thread waits for new metrics and also periodically reloads the cache
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: Optional["CacheLedger"],
        _broker,  # handle of _TreeStoreCacheBuilder
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._was_already_finished = ledger is not None and ledger.is_finished
        self._builder = _broker
        self._exemplar = exemplar

        self._metrics_monitors = []
        name = os.path.join(*cache_dir.split("/")[-2:])
        self.logger = pylogging.getLogger(f"TreeCache.{name}")
        self._store_future: threading_Future[TreeStore] = threading_Future()
        self._stop = False
        # assert _broker is None

        if self._builder is not None:
            self._monitor_thread = threading.Thread(target=self._monitor_metrics, daemon=True)
            self._monitor_thread.start()
        else:
            self._attempt_to_load_store(cache_metadata=True)
            assert self._store_future.done()

    @property
    def store(self) -> TreeStore[T_co]:
        return self._store_future.result()

    async def store_async(self) -> TreeStore[T_co]:
        if self._builder is not None:
            return await asyncio.wrap_future(self._store_future)
        else:
            return self.store

    async def async_len(self) -> int:
        if self._builder is not None:
            self.await_finished()

        return len(await self.store_async())

    def __len__(self):
        self.await_finished()

        return len(self.store)

    async def final_length_is_known(self) -> bool:
        return self.ledger is not None and self.ledger.is_finished

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> int:
        if not self._store_future.done():
            return 0

        return len(await self.store_async())

    async def get_batch(self, indices: Sequence[int] | slice):
        # this is tricky: we want to wait until either the cache is finished or we have the max index
        if isinstance(indices, slice):
            start, step, stop = await self._get_start_stops_async(indices)
            await self._wait_for_len(max(stop, start))
            indices = range(start, stop, step)

        max_index = max(indices)
        await self._wait_for_len(max_index + 1)

        return await self.store.get_batch(indices)

    async def _wait_for_len(self, needed_len: int):
        if self._builder is not None:
            while needed_len > await self.current_len():
                new_ledger: CacheLedger = await self._builder.updated_ledger.remote()

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.total_num_rows:
                        raise IndexError(
                            f"Index {needed_len} out of bounds for cache of size {new_ledger.total_num_rows}"
                        )
                    break
        else:
            if needed_len > len(self.store):
                raise IndexError(f"Index {needed_len} out of bounds for cache of size {len(self.store)}")

    def _wait_for_len_sync(self, needed_len, timeout: Optional[float] = None):
        time_in = time.time()
        t_max = time_in + (timeout or 1e6)
        if self._builder is not None:
            while needed_len > len(self.store):
                cur_time = time.time()
                if cur_time > t_max:
                    raise TimeoutError(f"Timed out waiting for cache to reach {needed_len}")
                try:
                    new_ledger: CacheLedger = ray.get(
                        self._builder.updated_ledger.remote(), timeout=max(t_max - cur_time, 10)
                    )
                except TimeoutError:
                    continue

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.total_num_rows:
                        raise IndexError(
                            f"Index {needed_len} out of bounds for cache of size {new_ledger.total_num_rows}"
                        )
                    break
        else:
            if needed_len > len(self.store):
                raise IndexError(f"Index {needed_len} out of bounds for cache of size {len(self.store)}")

    @staticmethod
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        """Loads a cache from disk or an object store. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        time_in = time.time()
        ledger = CacheLedger.load(cache_dir, options)
        time_out = time.time()
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
        metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
        try:
            return TreeCache.load(cache_dir, processor.output_exemplar, metadata)
        except FileNotFoundError:
            logger.info(f"Cache not found at {cache_dir}. Building.")
            broker = _get_builder_actor(
                cache_dir=cache_dir,
                shard_source=shard_source,
                processor=processor,
                options=options,
            )
            return TreeCache(cache_dir=cache_dir, exemplar=processor.output_exemplar, ledger=None, _broker=broker)

    def finished_sentinel(self):
        """Returns a Ray-awaitable object that will be set when the cache is finished"""
        if self._builder is None:
            return ray.remote(num_cpus=0)(lambda: None).remote()
        else:
            return self._builder.finished_sentinel.remote()

    @property
    def is_finished(self):
        return self.ledger is not None and self.ledger.is_finished

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, step, stop = self._get_start_stops(item)
            # TODO: wait for store to be set
            return self.store[start:stop:step]
        else:
            if item < 0:
                item += len(self)
            if item < 0 or item >= len(self):
                raise IndexError(f"Index {item} out of bounds for cache of size {len(self)}")
            return self.store[item]

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        store = self.store
        if isinstance(indices_or_slice, slice):
            start, step, stop = self._get_start_stops(indices_or_slice)
            indices_or_slice = range(start, stop, step)

        max_index = max(indices_or_slice)

        self._wait_for_len_sync(max_index + 1, timeout=timeout)

        return store.get_batch_sync(indices_or_slice)

    def _get_start_stops(self, slice):
        start = slice.start or 0
        if slice.stop is None:
            stop = len(self)
        elif slice.stop < 0:
            stop = len(self) + slice.stop
        else:
            stop = slice.stop
        if start < 0:
            start = len(self) + slice.start
        step = slice.step or 1
        return start, step, stop

    async def _get_start_stops_async(self, slice):
        start = slice.start or 0
        if slice.stop is None:
            stop = await self.async_len()
        elif slice.stop < 0:
            stop = (await self.async_len()) + slice.stop
        else:
            stop = slice.stop
        if start < 0:
            start = (await self.async_len()) + slice.start

        step = slice.step or 1
        return start, step, stop

    def await_finished(self, timeout: Optional[float] = None, await_cleanup: bool = False):
        if self._builder is None:
            return
        x = ray.get(self.finished_sentinel(), timeout=timeout)
        if await_cleanup:
            ray.get(self._builder.await_cleanup.remote(), timeout=timeout)
        self._attempt_to_load_store(cache_metadata=False)
        return x

    async def finished(self):
        if self._builder is None:
            return
        x = await self.finished_sentinel()
        # TODO: make an async version of this
        self._attempt_to_load_store(cache_metadata=False)
        return x

    def _attempt_to_load_store(self, cache_metadata):
        if self._store_future.done():
            return

        try:
            store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=cache_metadata)
        except FileNotFoundError:
            assert self._builder is not None
            ledger = ray.get(self._builder.current_ledger.remote())
            metrics = _ledger_to_metrics(ledger)
            if metrics.rows_finished == 0 and metrics.is_finished:
                # this means we built an empty cache. go with it
                store = TreeStore.open(self._exemplar, self.cache_dir, mode="a")
            else:
                raise
        try:
            self._store_future.set_result(store)
        except concurrent.futures.InvalidStateError:
            pass

    def attach_metrics_monitor(self, monitor: MetricsMonitor):
        if self._builder is None:
            logger.warning("Cannot attach metrics monitor to finished cache.")
            # TODO: decide what to do about attaching if the cache is already finished
            # maybe get the final metrics?
            return

        self._metrics_monitors.append(monitor)

    def _monitor_metrics(self):
        while not self._stop:
            try:
                try:
                    # it's better to let the Ray actor handle the timeout
                    ledger_or_timeout = ray.get(self._builder.updated_ledger.remote(timeout=4.0), timeout=10.0)
                    if isinstance(ledger_or_timeout, Exception):
                        raise ledger_or_timeout
                    self.ledger = ledger_or_timeout
                    metrics = _ledger_to_metrics(self.ledger)
                    for monitor in self._metrics_monitors:
                        monitor(metrics)
                    if metrics.is_finished:
                        break
                except TimeoutError:
                    pass
                except Exception as e:
                    if str(e).startswith("Failed to submit task to actor"):
                        logger.warning("Cache builder actor is gone. Stopping monitoring.")
                        break
                    else:
                        raise
                try:
                    self._attempt_to_load_store(cache_metadata=False)
                except FileNotFoundError:
                    pass
            except Exception as e:
                if str(e).startswith("Failed to submit task to actor"):
                    logger.warning("Cache builder actor is gone. Stopping monitoring.")
                    break
                else:
                    self.logger.exception("Error while reading metrics from shard cache.")
                    raise e


@dataclass_json
@dataclass
class CacheLedger:
    # NB: unlike the old cache, the mere existence of a ledger doesn't mean the cache is finished
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
            logger.debug(f"Attempting to load cache ledger from {ledger_path}")
            with fsspec.open(ledger_path) as file:
                cache_ledger = CacheLedger.from_json(file.read())  # type: ignore
            if metadata:
                diff = cache_ledger.metadata.compare_to(metadata)
                if not diff:
                    logger.debug("Metadata matches")
                else:
                    logger.warning(f"Metadata mismatch: {pprint.pformat(diff, indent=2)}")
            return cache_ledger
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache ledger not found at {ledger_path}")

    def _serialize_and_commit(self, cache_dir):
        path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        return _serialize_json_and_commit(path, self)  # type: ignore


@dataclass_json
@dataclass(frozen=True)
class CacheMetadata:
    preprocessor_metadata: Optional[dict[str, Any]] = None

    def compare_to(self, other: "CacheMetadata") -> deepdiff.DeepDiff:
        """
        Compare this metadata to another set of metadata. This is used to check if the cache being loaded
        was created with the expected configuration.

        if other.preprocessor_metadata is None, we ignore it for the purposes of comparison.
        """
        if other.preprocessor_metadata is None:
            sorta_self = dataclasses.replace(self, preprocessor_metadata=None)
        else:
            sorta_self = self
        return deepdiff.DeepDiff(sorta_self, other)

    @staticmethod
    def empty():
        return CacheMetadata()


class SerialCacheWriter(AbstractContextManager):
    """
    Writes TreeCache-compatible caches to disk. This is a serial version of TreeCacheWriter that doesn't use Ray.
    Mostly for scripts and debugging.

    Examples:
        >>> with SerialCacheWriter(cache_dir,exemplar) as writer:
        ...     for batch in process_batches():
        ...         writer.write_batch(batch)
    """

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        metadata: Optional["CacheMetadata"] = None,
    ):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._exemplar = exemplar
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="w", cache_metadata=True)
        self._is_closed = False

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if successful, write the ledger
        # TODO: store field counts in the ledger
        ledger = CacheLedger(
            total_num_rows=len(self._tree_store),
            is_finished=True,
            shard_rows={"": len(self._tree_store)},
            finished_shards=[""],
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
            raise NotImplementedError("Only non-RecordBatch batches are supported for now")

        cbatch = _canonicalize_batch(batch)  # type: ignore

        self._tree_store.extend(cbatch)


def _serialize_json_and_commit(path: str, obj):
    # just to be paranoid, we write to a temp file and then rename it
    # TODO: probably we could do better here
    fs: AbstractFileSystem = fsspec.core.url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        # copy the old file to a backup
        fs.copy(path, f"{path}.bak")

    for i in range(10):

        try:
            with fsspec.open(path, "w") as file:
                file.write(obj.to_json())
            break
        except FileNotFoundError:
            # this happens for some reason sometimes. It makes no sense.
            # FileNotFoundError: b/levanter-data/o/scratch%2Fdlwh%2Fpile-YYY%2Fpubmed_abs%2Ftrain%2Fshard_ledger.json.tmp/rewriteTo/b/levanter-data/o/scratch%2Fdlwh%2Fpile-YYY%2Fpubmed_abs%2Ftrain%2Fshard_ledger.json
            logger.exception(f"Failed to rename {path}.tmp to {path}")
            pass


@ray.remote(
    num_cpus=0.1, runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"})
)  # keep this small b/c it doesn't do a lot
class _TreeStoreCacheBuilder(SnitchRecipient):
    """
    Actor that coordinates the building of a cache. It spins up a bunch of workers to read from each shard
    and write to the cache.

    """

    def __init__(
        self,
        cache_dir: str,
        name: str,
        source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: CacheOptions,
    ):
        pylogging.basicConfig(format=LOG_FORMAT)
        self.logger = pylogging.getLogger(f"{__name__}.{name}")
        self.logger.setLevel(DEFAULT_LOG_LEVEL)
        self._finished_promise: asyncio.Future[None] = asyncio.Future()
        try:
            self.source = source
            self._cache_dir = cache_dir
            self._options = options
            self._updated_ledger_condition = asyncio.Condition()  # used to subscribe to metrics updates

            self._ledger = CacheLedger.load_or_initialize(cache_dir, source, processor)

            if self._ledger.is_finished:
                self._finished_promise.set_result(None)

            path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
            name = f"broker::{path_for_name}"
            self.logger = pylogging.getLogger(f"{name}")

            if self._ledger.is_finished:
                self.logger.info("Cache already finished. Nothing to do.")
                return
            self._cache_writer = _core_writer_task.options(
                name=f"writer::{path_for_name}",
                scheduling_strategy="SPREAD",
                # memory needed for the writer is twice the options' target size per flush
                # (we get twice from we need to concatenate prepared batches into the accumulator)
                # TODO: measure.
                memory=2 * self._options.target_bytes_per_flush,
            ).remote(current_actor_handle(), cache_dir, source, options, processor)

            self._tokenize_pbar = tqdm(
                total=len(source.shard_names), desc=f"{path_for_name}: tokenizing", unit="shard"
            )
            self._copy_pbar = tqdm(total=len(source.shard_names), desc=f"{path_for_name}: copying", unit="shard")
            self._report_totals = _ProgressReport(0, 0, 0)
            self._copy_report_totals = _ProgressReport(0, 0, 0)
            self._last_update = time.time()

        except Exception:
            # Ray behaves poorly if the constructor of an actor fails, so we catch and log here
            # this also propagates to the finished promise, so we can handle it there
            self._writer_exception(None, ser_exc_info())

    def current_ledger(self):
        if self._finished_promise.done() and self._finished_promise.exception() is not None:
            raise self._finished_promise.exception()
        return self._ledger

    def is_finished(self):
        if self.failed():
            return False
        return self._ledger.is_finished

    def failed(self):
        return self._finished_promise.done() and self._finished_promise.exception() is not None

    async def finished_sentinel(self):
        await self._finished_promise

    async def await_cleanup(self):
        if self._cache_writer is not None:
            await self._cache_writer

    async def updated_ledger(self, timeout: float | None = None) -> CacheLedger | TimeoutError:
        """
        NB: we **return** a timeout error, we don't raise it. This is because we want to find real failures
        in the ray dashboard, and it's a real pain to find exceptions in the logs.
        """
        if self._finished_promise.done():
            if self._finished_promise.exception() is not None:
                raise self._finished_promise.exception()  # type: ignore
            else:
                return self._ledger

        try:
            async with self._updated_ledger_condition:
                cond = self._updated_ledger_condition.wait()
                if timeout is not None:
                    await asyncio.wait_for(cond, timeout=timeout)
                else:
                    await cond
            return self._ledger
        except asyncio.TimeoutError:
            return TimeoutError("Timed out waiting for cache to update")

    def _writer_exception(self, shard_name, exc_info: ExceptionInfo):
        info = exc_info.restore()

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)

        try:
            self._finished_promise.set_exception(info[1])
        except InvalidStateError:
            pass
        except concurrent.futures.InvalidStateError:
            pass
        self._do_notify()

    def _child_failed(self, child: ray.actor.ActorHandle | str | None, exception: ExceptionInfo):
        self._writer_exception(str(child), exception)

    def _notify_updated_ledger(self, ledger: CacheLedger):
        """
        Called by the cache writer when it has updated the ledger.
        """
        was_finished = self._ledger.is_finished
        # ensure the ledger is "monotonic" meaning that we only expect it to grow
        if ledger.total_num_rows < self._ledger.total_num_rows:
            raise RuntimeError(f"Ledger went backwards: {ledger.total_num_rows} < {self._ledger.total_num_rows}")

        for shard, rows in ledger.shard_rows.items():
            if rows < self._ledger.shard_rows.get(shard, 0):
                raise RuntimeError(f"Shard {shard} went backwards: {rows} < {self._ledger.shard_rows.get(shard, 0)}")

        if was_finished:
            raise RuntimeError("Ledger was already finished")

        self._ledger = ledger
        if self._ledger.is_finished:
            logger.info(f"Finalizing cache {self._cache_dir}...")
            # guard against invalid state errors
            if not self._finished_promise.done():
                self._finished_promise.set_result(None)

        self._do_notify()

    def _notify_cleanup_finished(self):
        self._cache_writer = None

    def _do_notify(self):
        async def _do_notify_async():
            async with self._updated_ledger_condition:
                self._updated_ledger_condition.notify_all()

        asyncio.create_task(_do_notify_async())

    def _report_progress(self, report: "_ProgressReport"):
        import humanfriendly

        if report.new_shards > 0:
            self._tokenize_pbar.update(report.new_shards)
        self._report_totals.new_shards += report.new_shards
        self._report_totals.new_rows += report.new_rows
        self._report_totals.new_bytes += report.new_bytes

        if time.time() - self._last_update > 10.0:
            self._last_update = time.time()

            mb_str = humanfriendly.format_size(self._report_totals.new_bytes)
            self._tokenize_pbar.set_postfix(
                {
                    "rows": self._report_totals.new_rows,
                    "size": mb_str,
                }
            )

    def _report_copy_progress(self, report: "_ProgressReport"):
        self._copy_pbar.update(report.new_shards)
        self._copy_report_totals.new_shards += report.new_shards
        self._copy_report_totals.new_rows += report.new_rows
        self._copy_report_totals.new_bytes += report.new_bytes

        if time.time() - self._last_update > 10.0:
            self._last_update = time.time()
            self._copy_pbar.set_postfix(
                {
                    "rows": self._copy_report_totals.new_rows,
                    # "size": humanfriendly.format_size(report.new_bytes),
                }
            )


def _get_builder_actor(cache_dir, shard_source, processor, options=CacheOptions.default()):
    name = f"lev_cache_manager::{cache_dir}"
    path_for_name = os.path.join(*os.path.split(cache_dir)[-2:])
    name_for_display = f"builder::{path_for_name}"

    return _TreeStoreCacheBuilder.options(name=name, get_if_exists=True).remote(  # type: ignore
        name=name_for_display,
        cache_dir=cache_dir,
        source=shard_source,
        processor=processor,
        options=options,
    )


#####
# Core implementation starts below.
#####
# The main idea is to tokenize each shard group in parallel, and then write the results to the cache in order.


@dataclass
class _ShardFinished:
    """
    A message indicating that a shard has finished.
    """

    shard_name: str
    total_rows: int
    path_to_shard: str


@ray.remote(num_cpus=1, runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))
def _core_writer_task(
    parent,
    cache_dir,
    source: ShardedDataSource,
    options: CacheOptions,
    processor,
):
    """
    This is the main task that processes the data and writes it to the cache.

    It receives "finished shards" messages from the reader tasks, and copies the data from temporary files
    to the cache directory.

    """
    pylogging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    logger.info("Starting writer task")

    name = str(os.path.join(*cache_dir.split("/")[-2:]))
    # append a small random number to the name to avoid collisions
    name += f"::{random.randint(0, 1000)}"

    # we want to do the following:
    # 1. write the 0th shard group to the output cache directly, updating metrics as we go
    # 2. in the background, start processing other shard groups to temporary caches
    # 3. once (1) is done, we start copying the temporary caches to the output cache (in order)

    # We notify the parent actor of progress and updates to the ledger.
    # We special-case the 0'th ledger because we commit it to the output cache directly.
    def report_fn(report: _ProgressReport, ledger: CacheLedger):
        parent._report_progress.remote(report)

    def report_fn_first_group(report: _ProgressReport, ledger: CacheLedger):
        parent._report_progress.remote(report)
        ray.get(parent._notify_updated_ledger.remote(ledger))

    with log_failures_to(parent):
        temporary_cache_path = os.path.join(cache_dir, "___temp")

        group_cache_paths: dict[str, str] = {}
        group_ledgers: dict[str, CacheLedger | None] = {}
        write_refs: dict[str, ray.ObjectRef] = {}

        if len(source.shard_names) == 0:
            logger.info("No shards to process. Writing empty ledger.")
            ledger = CacheLedger.load_or_initialize(cache_dir, source, processor)
            ledger.is_finished = True
            ledger._serialize_and_commit(cache_dir)
            ray.get(parent._notify_updated_ledger.remote(ledger))
            return

        shard_groups = _assign_shards_to_groups(source, options.num_shard_groups)

        for name, group in shard_groups.items():
            assert len(group) > 0

        logger.debug(
            f"Tokenizing {len(source.shard_names)} shards in {len(shard_groups)} groups to {temporary_cache_path}."
        )

        processor_ref = ray.put(processor)
        source_ref = ray.put(source)

        # We treat the first group specially: we tokenize it directly to the output cache (since it comes first)
        # This enables us to expose data quickly
        first_group = next(iter(shard_groups), None)

        for group_name, group_shards in shard_groups.items():
            if group_name == first_group:
                group_cache_path = cache_dir
            else:
                group_cache_path = os.path.join(temporary_cache_path, group_name)

            group_cache_paths[group_name] = group_cache_path

            ledger = _try_load(group_cache_path)
            group_ledgers[group_name] = ledger

            if ledger is not None:
                if group_name == first_group:
                    ray.get(parent._notify_updated_ledger.remote(ledger))
                continue

            report_fn_to_use = report_fn_first_group if group_name == first_group else report_fn

            ref = (
                ray.remote(_tokenize_one_shard_group)
                .options(  # type: ignore
                    num_cpus=processor.num_cpus,
                    num_gpus=processor.num_gpus,
                    resources=processor.resources,
                    memory=3 * 1024 * 1024 * 1024,  # made this up
                    name=f"tokenize::{temporary_cache_path}::{group_name}",
                    retry_exceptions=True,
                    max_retries=10,
                )
                .remote(
                    group_cache_path=group_cache_path,
                    source=source_ref,
                    shards=group_shards,
                    processor=processor_ref,
                    options=options,
                    report_fn=report_fn_to_use,
                    # don't finalize the first group b/c we write it directly to the output cache
                    force_unfinalized=group_name == first_group,
                )
            )

            write_refs[group_name] = ref

        # wait for the first group to finish
        if first_group is not None:
            logger.info(f"Waiting for first group {first_group} to finish")
            ray.get(write_refs[first_group])

        logger.info(f"First group {first_group} finished. Copying other groups into permanent cache.")

        ledger = _copy_temp_caches_to_final_cache(
            parent,
            cache_dir,
            shard_groups,
            first_group,
            write_refs,
            group_ledgers,
            group_cache_paths,
            processor,
            processor_ref,
        )

        ledger.is_finished = True
        ledger._serialize_and_commit(cache_dir)
        ray.get(parent._notify_updated_ledger.remote(ledger))

        _clean_up_temp_caches(temporary_cache_path)
        # Fire and forget
        parent._notify_cleanup_finished.remote()


def _clean_up_temp_caches(path):
    logger.info(f"Cleaning up temporary cache at {path}")
    if fsspec_exists(path):
        for i in range(10):
            # this is crashy for some reason
            try:
                fsspec_remove(path, recursive=True)
                break
            except Exception:
                logger.exception(f"Failed to remove {path} on attempt {i}")
                time.sleep(1)


def _assign_shards_to_groups(source: ShardedDataSource, num_groups: int | None) -> dict[str, Sequence[str]]:
    if num_groups is None or num_groups >= len(source.shard_names):
        return {shard_name: [shard_name] for shard_name in source.shard_names}

    if num_groups <= 0:
        raise ValueError("num_groups must be a positive integer.")

    shard_names = source.shard_names
    num_shards_per_group = len(shard_names) // num_groups
    num_groups_with_extra = len(shard_names) % num_groups

    # if we have a remainder, we want to distribute the extra shards evenly
    out_groups: dict[str, list[str]] = {}
    start = 0
    for i in range(num_groups):
        num_shards = num_shards_per_group + (1 if i < num_groups_with_extra else 0)
        out_groups[f"group_{i}"] = list(shard_names[start : start + num_shards])
        start += num_shards

    assert sum(len(shards) for shards in out_groups.values()) == len(shard_names), "Mismatch in shard assignment."

    return out_groups  # type: ignore


def _merge_ledgers(dest: CacheLedger, source: CacheLedger):
    assert not dest.is_finished
    dest.total_num_rows += source.total_num_rows
    for shard, rows in source.shard_rows.items():
        current_value = dest.shard_rows.get(shard, 0)
        assert current_value == 0, f"Shard {shard} already has {current_value} rows"
        dest.shard_rows[shard] = rows

    dest.finished_shards.extend(source.finished_shards)
    for field, count in source.field_counts.items():
        dest.field_counts[field] = dest.field_counts.get(field, 0) + count

    return dest


def _copy_temp_caches_to_final_cache(
    parent,
    cache_dir,
    shard_groups,
    first_group,
    write_refs,
    group_ledgers,
    group_cache_paths,
    processor,
    processor_ref,
):
    """
    Copy the temporary caches to the output cache, in order. (essentially concatenating them)

    Args:
        parent: the parent actor handle (_TreeStoreCacheBuilder)
        cache_dir: the output cache directory
        shard_groups: a dict mapping group names to lists of shard names
        first_group: the privileged group that is written directly to the output cache
        write_refs: a dict mapping group names to ray.ObjectRefs of the cache building tasks
        group_ledgers: a dict mapping group names to the ledgers for the groups. Mutated in place.
        group_cache_paths: a dict mapping group names to the paths of the temporary caches
        processor: the processor object
        processor_ref: a ray.ObjectRef of the processor object

    Returns:
        The final ledger
    """
    # This logic is a bit hairy thanks to resumes.
    # First, note that each TreeCache is a tree of JaggedArrayStores, and we need to copy each of these
    # separately.
    # Second, note that JaggedArrayStores have two notions of length: the number of rows, and the data size.
    # We store the number of rows in offsets[0], and the data size in offsets[offsets[0]], which is just the final offset.
    # So we can keep a cache "locked" to a particular read size until we're ready by controlling the offsets.

    # * When we load the permanent cache, we have already written some number of groups to it. In
    #   particular, we have written the 0'th group to the permanent cache.
    # * We enforce that we only commit a whole group to the ledger at a time.
    # * We need to copy the remaining groups to the permanent cache, and update the ledger as we go.
    # * To copy a group, we need to know the total number of rows in that group, as well as the "data offsets"
    #   for the data in the cache. We can get the total number of rows from the ledger, and we also calculate
    #   the data offsets for where the group goes in the permanent cache. This is just a running sum of the
    #   data sizes of the previous groups. Because we have multiple JaggedArrayStores, this can be a pytree
    #   of integers, one for each array.
    # * Once we have finished the i'th cache and all caches < 1, we can "unlock" the data for the i'th cache
    #   by updating the offset[0] of the permanent cache to the total number of rows through the i'th cache.
    # * We also need to update the ledger with the total number of rows

    # reload the ledger for the first group, which will be the sink for the other groups
    assert first_group in write_refs

    group_ledgers[first_group] = ray.get(write_refs[first_group])
    overall_ledger = group_ledgers[first_group]

    # initialize the data offset tree
    permanent_cache = TreeStore.open(processor.output_exemplar, cache_dir, mode="a", cache_metadata=False)
    data_offset_tree = jax.tree.map(lambda x: x.data_size, permanent_cache.tree)

    total_rows_from_caches = overall_ledger.total_num_rows
    copy_refs: dict[str, ray.ObjectRef] = {}

    metadata_copier = _MetadataCopier.options(name=f"metadata_copier::{cache_dir}").remote(parent)
    copy_metadata_refs: dict[str, ray.ObjectRef] = {}

    parent._report_copy_progress.remote(
        _ProgressReport(new_shards=len(overall_ledger.finished_shards), new_rows=overall_ledger.total_num_rows)
    )

    found_one_to_copy = False

    for group in shard_groups:
        # first make sure it's either done this run or already done
        if write_refs.get(group) is not None:
            this_ledger = ray.get(write_refs[group])
            group_ledgers[group] = this_ledger
        else:
            this_ledger = group_ledgers[group]

        if group == first_group:
            continue

        assert this_ledger is not None
        # see if we already copied this group, meaning all the shards are in the permanent cache
        shards_copied = [shard for shard in shard_groups[group] if shard in overall_ledger.finished_shards]

        if len(shards_copied) == len(shard_groups[group]):
            assert (
                overall_ledger.total_num_rows >= total_rows_from_caches
            ), f"{overall_ledger.total_num_rows} < {total_rows_from_caches}. {group}"
            assert not found_one_to_copy, f"Found a finished group after an unfinished group: {group}"

            logger.info(f"Group {group} already copied. Skipping.")
            continue  # nothing to do
        elif len(shards_copied) > 0:
            # In theory, we can handle this, but it's a bit tricky, so we're going to punt for now
            raise RuntimeError(
                "Some shards were copied but not all. This should never happen."
                f"Specifically the following shards were copied: {shards_copied}"
                f"And the following shards were not: {set(shard_groups[group]) - set(shards_copied)}"
            )

        # we need to copy this group
        found_one_to_copy = True

        copy_refs[group] = _copy_cache_data.remote(
            cache_dir,
            group_cache_paths[group],
            processor_ref,
            data_offset_tree,
            total_rows_from_caches,
            parent,
        )
        copy_metadata_refs[group] = metadata_copier.copy_metadata.remote(
            cache_dir,
            group_cache_paths[group],
            processor_ref,
            data_offset_tree,
            total_rows_from_caches,
        )

        this_rows = this_ledger.total_num_rows
        total_rows_from_caches += this_rows

        # update the offset information: data offsets and total rows
        this_cache = TreeStore.open(processor.output_exemplar, group_cache_paths[group], mode="r", cache_metadata=True)
        data_offset_tree = jax.tree.map(
            operator.add, data_offset_tree, jax.tree.map(lambda x: x.data_size, this_cache.tree)
        )

    # commit each group in order. We need to do two things:
    # "commit" here means:
    # 1. update the data offsets in the permanent cache with the number of rows available from all groups so far
    # 2. update the ledger with the combined information from all groups so far
    num_available_rows = overall_ledger.total_num_rows
    for group, ref in copy_refs.items():
        ray.get(ref)  # block on data copy
        ray.get(copy_metadata_refs[group])  # block on metadata copy

        group_ledger = group_ledgers[group]
        num_available_rows += group_ledger.total_num_rows

        _expose_available_rows(permanent_cache, num_available_rows)

        _merge_ledgers(overall_ledger, group_ledger)
        overall_ledger._serialize_and_commit(cache_dir)
        ray.get(parent._notify_updated_ledger.remote(overall_ledger))
        parent._report_copy_progress.remote(
            _ProgressReport(new_shards=len(group_ledger.finished_shards), new_rows=group_ledger.total_num_rows)
        )
        logger.info(f"Group {group} copied. Updating ledger.")

    return overall_ledger


def _expose_available_rows(permanent_cache, num_available_rows):
    """
    Updates the permanent cache to expose the available rows. This is done by updating the offsets[0] of the
    permanent cache to the total number of rows in the cache.
    """
    futures = jax.tree.leaves(jax.tree.map(lambda x: x.offsets[0].write(num_available_rows), permanent_cache.tree))
    for future in futures:
        future.result()


@ray.remote(
    num_cpus=4,
    memory=6 * 1024 * 1024 * 1024,
    runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}),
    scheduling_strategy="SPREAD",
)
def _copy_cache_data(dest_path, source_path, processor, data_offset_tree, rows_so_far, parent):
    """
    Copies the data from one cache to another, appending it to the end of the destination cache.

    Once the copy is done and the last_ref is set, the data is "unlocked" in the destination cache by updating the
    offsets[0] of the destination cache to the total number of rows in the cache.
    Args:
        dest_path:  The path to the destination cache.
        source_path: The path to the source cache.
        processor: The processor used to create the cache.
        data_offset_tree: The data offset tree for the destination cache.
        rows_so_far: The total number of rows in the destination cache before this copy.
    """
    with log_failures_to(parent):
        asyncio.run(_extend_cache_with_other_cache(dest_path, source_path, processor, data_offset_tree, rows_so_far))


@ray.remote(
    num_cpus=0.5,
    memory=1 * 1024 * 1024 * 1024,
    runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}),
)
class _MetadataCopier:
    """Copies the metadata from one cache to another. We use an actor because we want to impose the impliacit
    actor mutex lock on the metadata file to prevent concurrent writes. We have found that using ts Transactions here
    results in way too many retries, resulting in $$$$. If we prevent concurrent writes, we can avoid this."""

    # DO NOT ADD ASYNC METHODS TO THIS CLASS. It will remove the implicit lock and cause concurrent writes.
    def __init__(self, parent):
        self.parent = parent

    def copy_metadata(self, dest_path, source_path, processor, data_offset_tree, rows_so_far):
        """
        Copies the data from one cache to another, appending it to the end of the destination cache.

        Once the copy is done and the last_ref is set, the data is "unlocked" in the destination cache by updating the
        offsets[0] of the destination cache to the total number of rows in the cache.
        Args:
            dest_path:  The path to the destination cache.
            source_path: The path to the source cache.
            processor: The processor used to create the cache.
            data_offset_tree: The data offset tree for the destination cache.
            rows_so_far: The total number of rows in the destination cache before this copy.
        """
        with log_failures_to(self.parent):
            asyncio.run(
                _extend_cache_metadata_with_other(dest_path, source_path, processor, data_offset_tree, rows_so_far)
            )


async def _extend_cache_with_other_cache(
    dest_path: str, source_path: str, processor: BatchProcessor, data_offset_tree: PyTree[int], row_offset
) -> int:
    """
    Copies the data from one cache to another, appending it to the end of the destination cache.

    Returns:
        The number of rows in the source cache.
    """

    try:

        logger.info(f"Copying data from {source_path} to {dest_path}.")
        dest = TreeStore.open(processor.output_exemplar, dest_path, mode="a", cache_metadata=False)
        source = TreeStore.open(processor.output_exemplar, source_path, mode="r", cache_metadata=True)

        source_num_rows = await source.async_len()

        async def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
            """Copies **just the data array** from one shard to the permanent cache at a given offset."""
            # TODO: it'd be good if we just didn't expose the full data array (but only the used part)
            data_size = source_array.data_size
            data = source_array.data

            # To prevent OOM, copy in smaller batches
            MAX_ELEMS = 1024 * 1024 * 1024
            await _copy_in_batches(dest_array.data, data_offset, data, data_size, MAX_ELEMS)

        futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)

        await asyncio.gather(*jax.tree.leaves(futures))
        logger.info(f"Finished copying data from {source_path} to {dest_path}.")

        return source_num_rows
    except Exception as e:
        logger.exception(f"Failed to copy data from {source_path} to {dest_path}: {e}")
        raise


async def _copy_in_batches(dest_array, dest_offset, src_array, src_len, elems_per_batch):
    """
    Copies the data from one array to another in batches.
    """
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
    dest_path: str, source_path: str, processor: BatchProcessor, data_offset_tree: PyTree[int], row_offset
) -> int:
    """Copies just the offsets and shapes (if present)"""
    try:
        logger.info(f"Copying metadata from {source_path} to {dest_path}.")
        dest = TreeStore.open(processor.output_exemplar, dest_path, mode="a")
        source = TreeStore.open(processor.output_exemplar, source_path, mode="r", cache_metadata=True)

        source_num_rows = await source.async_len()

        async def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
            """Copies **just the data array** from one shard to the permanent cache at a given offset."""

            if source_array.shapes is not None:
                source_shapes = source_array.shapes
                async with ts.Transaction() as txn:
                    dest_shapes = dest_array.shapes
                    assert dest_shapes is not None
                    out_end = row_offset + source_num_rows
                    shape_future = dest_shapes.with_transaction(txn)[row_offset:out_end].write(source_shapes)

            # the 0th offset is the number of rows so we don't want to copy it into the destination
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
    except Exception as e:
        logger.exception(f"Failed to copy metadata from {source_path} to {dest_path}: {e}")
        raise


def _virtual_offset(base: ts.TensorStore, offset_amount):
    """
    This function creates a new tensorstore that is a virtual offset of another tensorstore.
    That is, it's y[i] = x[i] + offset_amount.
    """

    async def do_read(domain: ts.IndexDomain, array: np.ndarray, read_params: ts.VirtualChunkedReadParameters):
        array[...] = (await base[domain].read()) + offset_amount

    return ts.virtual_chunked(do_read, dtype=base.dtype, domain=base.domain, shape=base.shape)


@dataclass
class _ProgressReport:
    new_rows: int = 0
    new_bytes: int = 0
    new_shards: int = 0
    # TODO: other counts


def _tokenize_one_shard_group(
    group_cache_path: str,
    source: ShardedDataSource,
    shards: list[str],
    processor: BatchProcessor,
    options: CacheOptions,
    report_fn: Callable[[_ProgressReport, CacheLedger], None],
    force_unfinalized: bool,
) -> CacheLedger:
    # ray breaks if this is top level
    import humanfriendly

    logger = pylogging.getLogger("tokenize")
    pylogging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(DEFAULT_LOG_LEVEL)

    # restrict shards to the ones we're supposed to process
    # this is a bit hacky but when there are a lot of shards (e.g. SlimPajama 122K),
    # we encounter significant overhead just parsing the shard names from the json
    source = _RestrictedShardedDataSource(source, shards)

    ledger = CacheLedger.load_or_initialize(group_cache_path, source, processor)

    if ledger.is_finished:
        report_fn(_ProgressReport(new_rows=ledger.total_num_rows, new_shards=len(ledger.finished_shards)), ledger)
        logger.info("Shard group already processed.")
        return ledger

    writer = ShardGroupCacheWriter(group_cache_path, ledger, shards, processor.output_exemplar)

    total_rows = ledger.total_num_rows
    found_shard_with_rows = False

    for shard_name in shards:
        if shard_name in ledger.finished_shards:
            report_fn(_ProgressReport(new_rows=ledger.shard_rows[shard_name], new_shards=1), ledger)
            logger.info(f"Shard {shard_name} already processed.")
            continue

        logger.debug(f"Processing {shard_name}.")

        rows_this_shard = ledger.shard_rows.get(shard_name, 0)

        if found_shard_with_rows and rows_this_shard != 0:
            raise ValueError(
                "Found more than one partially processed shard in this group. This indicates that the"
                "number of groups has changed, which is not supported."
            )

        if rows_this_shard != 0:
            report_fn(_ProgressReport(new_rows=rows_this_shard), ledger)
            found_shard_with_rows = True

        shard_iterator = source.open_shard_at_row(shard_name, rows_this_shard)

        prepared_batch: PyTree[PreparedBatch] | None = None
        this_batch_size = 0

        for batch in batched(shard_iterator, options.batch_size):
            tokenized = processor(batch)
            tokenized = _canonicalize_batch(tokenized)  # type: ignore
            this_prepared = writer._tree_store.batch_preparer(tokenized)

            this_batch_size += len(batch)
            rows_this_shard += len(batch)
            total_rows += len(batch)

            if prepared_batch is None:
                prepared_batch = this_prepared
            else:
                prepared_batch = jax.tree.map(
                    lambda *trees: PreparedBatch.concat(trees), prepared_batch, this_prepared
                )

            batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))

            if batch_byte_size > options.target_bytes_per_flush:
                writer.write_prepared_batch(shard_name, this_batch_size, prepared_batch)
                report_fn(_ProgressReport(new_rows=this_batch_size, new_bytes=batch_byte_size), writer.ledger)

                nice_bytes = humanfriendly.format_size(batch_byte_size)
                logger.debug(
                    f"Processed {rows_this_shard} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})"
                )
                this_batch_size = 0
                prepared_batch = None

        if prepared_batch is not None:
            batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))
            nice_bytes = humanfriendly.format_size(batch_byte_size)

            report_fn(_ProgressReport(new_rows=this_batch_size, new_bytes=batch_byte_size), writer.ledger)

            writer.write_prepared_batch(shard_name, this_batch_size, prepared_batch)
            logger.debug(
                f"Processed {rows_this_shard} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})"
            )
            this_batch_size = 0
            prepared_batch = None

        writer.finish_shard(shard_name, rows_this_shard)

        report_fn(_ProgressReport(new_shards=1), writer.ledger)

    if not force_unfinalized:
        writer.finish()

    logger.info(f"Finished processing {len(shards)} shards. Wrote {total_rows} rows.")

    return writer.ledger


class ShardGroupCacheWriter:
    """
    Similar to SerialCacheWriter, but tracks shard metadata for one shard.
    """

    def __init__(self, cache_dir: str, initial_ledger: CacheLedger, shards: list[str], exemplar: T):
        self.cache_dir = cache_dir

        self._ledger = copy.deepcopy(initial_ledger)
        self.shards = shards

        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="a")  # type: ignore
        self._tree_store.trim_to_size(self._ledger.total_num_rows)

    @property
    def ledger(self):
        return self._ledger

    # we have both versions b/c we need this one for actors
    def get_ledger(self):
        return self._ledger

    @property
    def is_finished(self):
        return self._ledger.is_finished

    def finish_shard(self, shard_name: str, num_rows: int):
        if shard_name not in self.shards:
            raise ValueError(f"Shard {shard_name} not in tracked shards")

        current_rows = self._ledger.shard_rows.get(shard_name, 0)
        if current_rows != num_rows:
            raise ValueError(f"Expected {num_rows} rows in finished shard {shard_name}, but found {current_rows}")

        self._ledger.finished_shards.append(shard_name)
        self._ledger._serialize_and_commit(self.cache_dir)

    def write_prepared_batch(self, shard_name: str, row_count: int, batch: PyTree[PreparedBatch]):
        if self.is_finished:
            raise RuntimeError("Cannot write to a finished cache")
        self._tree_store.extend_with_batch(batch)

        if shard_name not in self.shards:
            raise ValueError(f"Shard {shard_name} not in tracked shards")
        self._ledger.shard_rows[shard_name] += row_count
        self._ledger.total_num_rows += row_count

        self._ledger._serialize_and_commit(self.cache_dir)

    def finish(self):
        if len(self._ledger.finished_shards) != len(self.shards):
            raise ValueError("Not all shards are finished")

        self._ledger.is_finished = True
        self._ledger._serialize_and_commit(self.cache_dir)
        # ensure all tracked shards are finished

        return self._tree_store


class _RestrictedShardedDataSource(ShardedDataSource):
    def __init__(self, source: ShardedDataSource, shards: list[str]):
        self._source = source
        self._shards = shards

    @property
    def shard_names(self):
        return self._shards

    def open_shard_at_row(self, shard_name, row):
        return self._source.open_shard_at_row(shard_name, row)


def _randomize_shards(shards: Sequence[T], seed: int) -> list[T]:
    prng = random.Random(seed)
    shuffled = list(shards)
    prng.shuffle(shuffled)
    return shuffled


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        return _to_list_of_dicts(batch)
    else:
        return batch


def _to_list_of_dicts(batch: dict) -> List[dict]:
    """
    Convert a batch of dictionaries to a list of dictionaries, suitable for writing to a cache.
    """
    keys = list(batch.keys())
    values = list(batch.values())
    num_rows = len(values[0])
    return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]


def _ledger_to_metrics(ledger: CacheLedger) -> InProgressCacheMetrics:
    # TODO: remove this
    return InProgressCacheMetrics(
        rows_finished=ledger.total_num_rows,
        is_finished=ledger.is_finished,
        # shard_rows=ledger.shard_rows,
        shards_finished=len(ledger.finished_shards),
        field_counts=ledger.field_counts,
    )


def _try_load(path):
    try:
        ledger = CacheLedger.load(path)
        if ledger.is_finished:
            return ledger
        else:
            logger.debug(f"Cache exists but is not finished at {path}.")
            return None
    except FileNotFoundError:
        return None
