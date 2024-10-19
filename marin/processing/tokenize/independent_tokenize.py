"""
Alternative to Levanter's tokenization pipeline that tokenizes shards independently and then concatenates the results.
This is much more reliable on the Marin cluster. It is also easier to map tokens back to the original data.
"""

import asyncio
import copy
import logging
import os
from collections.abc import Sequence
from functools import partial
from typing import TypeVar

import fsspec
import jax.tree
import numpy as np
import ray
import tensorstore as ts
from jaxtyping import PyTree
from levanter.data import BatchProcessor, ShardedDataSource, batched
from levanter.store import TreeStore
from levanter.store.cache import CacheLedger, CacheOptions, _canonicalize_batch
from levanter.store.jagged_array import DEFAULT_WRITE_CHUNK_SIZE, JaggedArrayStore, PreparedBatch
from tqdm_loggable.auto import tqdm

from ...utilities import ray_utils
from ...utils import fsspec_exists

T = TypeVar("T")

logger = logging.getLogger(__name__)


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


def _tokenize_one_shard_group(
    temporary_cache_path: str,
    source: ShardedDataSource,
    shards: list[str],
    processor: BatchProcessor,
    options: CacheOptions,
) -> CacheLedger:
    # ray breaks if this is top level
    import humanfriendly

    logger = logging.getLogger("tokenize")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    source = _RestrictedShardedDataSource(source, shards)
    ledger = CacheLedger.load_or_initialize(temporary_cache_path, source, processor, options)

    if ledger.is_finished:
        logger.info("Shard group already processed.")
        return ledger

    # restrict shards to the ones we're supposed to process
    # this is a bit hacky but when there are a lot of shards (e.g. SlimPajama 122K),
    # we encounter significant overhead just parsing the shard names from the json

    writer = ShardGroupCacheWriter(temporary_cache_path, ledger, shards, processor.output_exemplar)

    total_rows = ledger.total_num_rows
    found_shard_with_rows = False

    for shard_name in shards:
        if shard_name in ledger.finished_shards:
            logger.info(f"Shard {shard_name} already processed.")
            continue

        logger.debug(f"Processing {shard_name}.")

        rows_this_shard = ledger.shard_rows.get(shard_name, 0)

        if found_shard_with_rows and rows_this_shard != 0:
            raise ValueError("Found more than one shard with rows to process.")

        if rows_this_shard != 0:
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

            if prepared_batch is None:
                prepared_batch = this_prepared
            else:
                prepared_batch = jax.tree.map(lambda *trees: PreparedBatch.concat(trees), prepared_batch, this_prepared)

            batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))

            if batch_byte_size > options.target_bytes_per_flush:
                writer.write_prepared_batch(shard_name, this_batch_size, prepared_batch)
                nice_bytes = humanfriendly.format_size(batch_byte_size)
                logger.debug(
                    f"Processed {rows_this_shard} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})"
                )
                this_batch_size = 0
                prepared_batch = None

        if prepared_batch is not None:
            batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))
            nice_bytes = humanfriendly.format_size(batch_byte_size)
            writer.write_prepared_batch(shard_name, this_batch_size, prepared_batch)
            logger.debug(
                f"Processed {rows_this_shard} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})"
            )
            this_batch_size = 0
            prepared_batch = None

        total_rows += rows_this_shard

        writer.finish_shard(shard_name, rows_this_shard)

    writer.finish()

    logger.info(f"Finished processing {len(shards)} shards. Wrote {total_rows} rows.")

    return writer.ledger


def _assign_shards_to_groups(source: ShardedDataSource, num_groups: int | None) -> dict[str, Sequence[str]]:
    if num_groups is None or num_groups >= len(source.shard_names):
        return {shard_name: [shard_name] for shard_name in source.shard_names}

    shard_names = source.shard_names
    num_shards_per_group = len(shard_names) // num_groups
    return {
        f"group_{i}": shard_names[i * num_shards_per_group : (i + 1) * num_shards_per_group] for i in range(num_groups)
    }


def tokenize_all_shards(
    temporary_cache_path: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    options: CacheOptions,
) -> list[tuple[str, CacheLedger]]:
    logger = logging.getLogger("tokenize_all_shards")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

    shard_groups = _assign_shards_to_groups(source, options.num_shard_groups)

    logger.info(f"Tokenizing {len(source.shard_names)} shards in {len(shard_groups)} groups to {temporary_cache_path}.")

    paths: dict[str, str] = {}
    ledgers: dict[str, CacheLedger | None] = {}
    already_finished_paths: list[str] = []
    caches_in_progress: dict[ray.ObjectRef, str] = {}
    refs: list[ray.ObjectRef] = []

    unit = "shard" if len(shard_groups) == len(source.shard_names) else "shard group"
    pbar = tqdm(total=len(shard_groups), desc="Tokenizing", unit=unit)

    processor_ref = ray.put(processor)
    source_ref = ray.put(source)

    for group_name, shards in shard_groups.items():
        path = os.path.join(temporary_cache_path, group_name)
        paths[group_name] = path

        ledger = _try_load(path)

        if ledger is not None:
            already_finished_paths.append(path)

        ledgers[group_name] = ledger

        if ledger is not None and ledger.is_finished:
            pbar.update(1)
            continue

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
            .remote(os.path.join(temporary_cache_path, group_name), source_ref, shards, processor_ref, options)
        )

        caches_in_progress[ref] = group_name
        refs.append(ref)

    while refs:
        done, refs = ray.wait(list(refs), num_returns=1)
        for ref in done:
            group_name = caches_in_progress.pop(ref)
            ledger = ray.get(ref)
            ledgers[group_name] = ledger
            pbar.update(1)

    assert all(ledger is not None for ledger in ledgers.values())

    return [(paths[group_name], ledger) for group_name, ledger in ledgers.items()]  # type: ignore


async def concatenate_stores(
    permanent_cache_path: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    caches: list[tuple[str, CacheLedger]],
) -> CacheLedger:
    """
    Args:
        permanent_cache_path: path to the permanent cache we're going to write to
        caches:  path, ledger of individual shard caches

    Returns:
        the ledger of the permanent cache
    """
    logger = logging.getLogger("concatenate_stores")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    options = caches[0][1].metadata.options

    logger.info(f"Concatenating {len(caches)} caches to {permanent_cache_path}.")

    permanent_ledger = CacheLedger.load_or_initialize(permanent_cache_path, source, processor, options)

    if permanent_ledger.is_finished:
        logger.info("Permanent cache already finished.")
        return permanent_ledger

    data_futures = _start_data_copies_for_shards(permanent_cache_path, source, processor, caches)

    # ok we try to be smart about this. We have a bunch of tree stores we want to concatenate.
    # each treestore is a PyTree of jagged arrays, and a jagged array is 2-3 tensorstore arrays (offsets, data, shapes?)
    # we want to concatenate these treestores.
    stores = [TreeStore.open(processor.output_exemplar, path, mode="r", cache_metadata=True) for path, _ in caches]

    dest = TreeStore.open(processor.output_exemplar, permanent_cache_path, mode="a", cache_metadata=False)

    metadata_futures = jax.tree.map(
        lambda d_arr, *arrs: concatenate_jagged_array_metadata(d_arr, arrs), dest.tree, *[s.tree for s in stores]
    )

    await asyncio.gather(*data_futures)
    await asyncio.gather(*jax.tree.leaves(metadata_futures))

    # ok now make the ledger
    permanent_ledger.total_num_rows = sum(ledger.total_num_rows for _, ledger in caches)
    permanent_ledger.shard_rows = {
        shard_name: sum(ledger.shard_rows.values())
        for shard_name, (_, ledger) in zip(source.shard_names, caches, strict=False)
    }
    permanent_ledger.is_finished = True

    permanent_ledger._serialize_and_commit(permanent_cache_path)

    logger.info(f"Finished concatenating {len(caches)} caches to {permanent_cache_path}.")

    return permanent_ledger


@ray.remote(retry_exceptions=True)
def concatenate_stores_remote(
    permanent_cache_path: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    caches: list[tuple[str, CacheLedger]],
) -> CacheLedger:
    # Ray doesn't let tasks be async, so we have to do this.
    return asyncio.run(concatenate_stores(permanent_cache_path, source, processor, caches))


async def concatenate_jagged_array_metadata(dest: JaggedArrayStore, arrays: Sequence[JaggedArrayStore]):
    """
    This function concatenates a sequence of jagged arrays into a single jagged array.
    It relies on knowledge of internals of the JaggedArrayStore class.
    """
    data_sizes = np.array([a.data_size for a in arrays])
    data_offsets_per_shard = np.concatenate([np.array([0], dtype=int), np.cumsum(data_sizes)])

    # # this is a virtual concatenation of the data arrays.
    # this is super slow, so we do the more complicated thing below.
    # data_concat = ts.concat(datas, axis=0)
    #
    # data_write_future = dest.data[0:total_data_size].write(data_concat)

    offsets_write_future = _write_offsets(dest, arrays, data_offsets_per_shard)

    if dest.shapes is not None:
        # this won't be set for tokenization, but for completeness
        shapes = [a.shapes[0 : a.num_rows] for a in arrays]
        shapes_concat = ts.concat(shapes, axis=0)
        shapes_write_future = dest.shapes[0 : shapes_concat.shape[0], :].write(shapes_concat)
        await shapes_write_future

    await offsets_write_future
    # write number of rows
    row_counts = [a.num_rows for a in arrays]
    await dest.offsets[0].write(np.array(sum(row_counts), dtype=int))

    return


async def _write_offsets(dest: JaggedArrayStore, arrays: Sequence[JaggedArrayStore], data_offsets_per_shard: np.ndarray):
    # we pack the number of rows into the 0'th entry of the indices array.
    # tensorstore doesn't shift indices to 0, so we need to do that ourselves.
    offsetses = [a.offsets[1 : a.num_rows + 1][ts.d[:].translate_to[0]] for a in arrays]
    adjusted_offsets = [
        _virtual_offset(offsets, offset) for offsets, offset in zip(offsetses, data_offsets_per_shard, strict=False)
    ]
    offsets_concat = ts.concat(adjusted_offsets, axis=0)
    offsets_write_future = dest.offsets[1 : offsets_concat.shape[0] + 1].write(offsets_concat)
    return await offsets_write_future


def _start_data_copies_for_shards(
    permanent_cache_path: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    caches: list[tuple[str, CacheLedger]],
):
    sources = [TreeStore.open(processor.output_exemplar, path, mode="r", cache_metadata=True) for path, _ in caches]

    def compute_data_offsets_for_shards(stores: list[JaggedArrayStore]):
        data_sizes = [a.data_size for a in stores]
        data_offsets = np.concatenate([np.array([0], dtype=int), np.cumsum(data_sizes)])
        return data_offsets

    data_offsets = jax.tree.map(lambda *trees: compute_data_offsets_for_shards(trees), *[s.tree for s in sources])

    data_offsets_for_shards = [
        jax.tree.map(partial(lambda i, offset_array: offset_array[i], i), data_offsets) for i in range(len(caches))
    ]

    if ray_utils.is_local_ray_cluster() or os.getenv("CI", "false").lower() in {"true", "1"}:
        cpus = 1
        memory = 1 * 1024 * 1024 * 1024
    else:
        cpus = 4
        memory = 16 * 1024 * 1024 * 1024

    # SPREAD to take advantage of the fact that we're copying data from different shards
    # for some reason, this uses a ton of memory
    @ray.remote(scheduling_strategy="SPREAD", num_cpus=cpus, memory=memory, retry_exceptions=True)
    def do_copy(path, data_offset_tree):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        if fsspec_exists(os.path.join(path, ".COPY_SUCCESS")):
            logger.info(f"Data already copied from {path} to {permanent_cache_path}.")
            return

        asyncio.run(
            _copy_data_from_one_shard_to_permanent_memory(permanent_cache_path, path, processor, data_offset_tree)
        )

        with fsspec.open(os.path.join(path, ".COPY_SUCCESS"), "w") as f:
            f.write("")

        return

    futures: list[ray.ObjectRef] = []

    for (path, _), data_offset_tree in zip(caches, data_offsets_for_shards, strict=False):
        if fsspec_exists(os.path.join(path, ".COPY_SUCCESS")):
            logger.info(f"Data already copied from {path} to {permanent_cache_path}.")
            continue

        futures.append(do_copy.remote(path, data_offset_tree))

    return futures


async def _copy_data_from_one_shard_to_permanent_memory(
    dest_path: str,
    source_path: str,
    processor: BatchProcessor,
    data_offset_tree: PyTree[int],
):
    """Copies **just the data array** from one shard to the permanent cache at a given offset."""
    logger.info(f"Copying data from {source_path} to {dest_path}.")
    dest = TreeStore.open(processor.output_exemplar, dest_path, mode="a", cache_metadata=False)
    source = TreeStore.open(processor.output_exemplar, source_path, mode="r", cache_metadata=True)

    def _copy_one_array(dest_array: JaggedArrayStore, source_array: JaggedArrayStore, data_offset: int):
        # TODO: it'd be good if we just didn't expose the full data array (but only the used part)
        # data = source_array.data[0 : source_array.data_size]
        # write_future = dest_array.data[data_offset : data_offset + source_array.data_size].write(data)
        #
        # return write_future
        return _chunked_copy(source_path, dest_array.data, source_array.data, data_offset, source_array.data_size)

    futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)

    await asyncio.gather(*jax.tree.leaves(futures))
    logger.info(f"Finished copying data from {source_path} to {dest_path}.")
    return


def _chunked_copy(desc: str, dest_array: ts.TensorStore, source_array: ts.TensorStore, dest_offset: int, data_size: int):
    """
    Tries to do an aligned copy of a source array to a destination array. For some reason TS is using
    a *ton* of memory to do the naive copy. We try to do something smart/chunked.
    """
    # TODO: transactions don't seem to be working?

    itemsize = dest_array.dtype.numpy_dtype.itemsize
    pbar = tqdm(total=data_size * itemsize, desc=f"Copying data ({desc})", unit="B", unit_scale=True)

    write_alignment = DEFAULT_WRITE_CHUNK_SIZE  # 500MB or so
    # write_alignment = 64
    block_size = write_alignment * 4

    # find the first aligned index
    first_aligned_index = (dest_offset + write_alignment - 1) // write_alignment * write_alignment
    src_index = 0
    dest_index = 0

    # copy the initial unaligned part
    if first_aligned_index != 0:
        write_length = min(data_size, first_aligned_index - dest_offset)
        data = source_array[0:write_length].read().result()
        dest_array[dest_offset : dest_offset + write_length].write(data).result()
        del data
        src_index = write_length
        dest_index = dest_offset + write_length
        pbar.update(write_length * itemsize)

    while src_index < data_size:
        write_length = min(block_size, data_size - src_index)
        data = source_array[src_index : src_index + write_length].read().result()
        dest_array[dest_index : dest_index + write_length].write(data).result()
        del data
        src_index += write_length
        dest_index += write_length
        pbar.update(write_length * itemsize)

    return


def _virtual_offset(base: ts.TensorStore, offset_amount):
    """
    This function creates a new tensorstore that is a virtual offset of another tensorstore.
    That is, it's y[i] = x[i] + offset_amount.
    """

    async def do_read(domain: ts.IndexDomain, array: np.ndarray, read_params: ts.VirtualChunkedReadParameters):
        array[...] = (await base[domain].read()) + offset_amount

    return ts.virtual_chunked(do_read, dtype=base.dtype, domain=base.domain, shape=base.shape)


def tokenize_and_concatenate_shards(
    source: ShardedDataSource,
    processor: BatchProcessor,
    cache_path: str,
    options: CacheOptions,
) -> CacheLedger:
    """

    Tokenizes the shards of a ShardedDataSource independently and concatenates the results into a single cache.

    Returns:
        The ledger of the concatenated cache.

    """
    # first see if we need to do anything
    try:
        ledger = CacheLedger.load(cache_path)
        logger.info(f"Cache already exists at {cache_path}.")
        return ledger
    except FileNotFoundError:
        pass

    temporary_cache_path = os.path.join(cache_path, "__temporary")
    ledgers = tokenize_all_shards(temporary_cache_path, source, processor, options)
    ledger = ray.get(concatenate_stores_remote.remote(cache_path, source, processor, ledgers))
    # delete temporary cache
    remove(temporary_cache_path, recursive=True)
    return ledger


def remove(url, *, recursive=False, **kwargs):
    """Remove a file from a remote filesystem."""
    # TODO: better to use a STS deletion policy or job for this one.
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    fs.rm(path, recursive=recursive)
