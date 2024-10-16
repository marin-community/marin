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
from levanter.store.jagged_array import JaggedArrayStore, PreparedBatch

from ...utilities import ray_utils
from ...utils import fsspec_exists

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ShardCacheWriter:
    """
    Similar to SerialCacheWriter, but tracks shard metadata for one shard.
    """

    def __init__(self, cache_dir: str, initial_ledger: CacheLedger, shard: str, exemplar: T):
        self.cache_dir = cache_dir

        self._ledger = copy.deepcopy(initial_ledger)
        self.shard = shard

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
        current_rows = self._ledger.shard_rows.get(shard_name, 0)
        if current_rows != num_rows:
            raise ValueError(f"Expected {num_rows} rows in finished shard {shard_name}, but found {current_rows}")

        self._ledger.finished_shards.append(shard_name)
        self._ledger._serialize_and_commit(self.cache_dir)

    def write_prepared_batch(self, row_count: int, batch: PyTree[PreparedBatch]):
        if self.is_finished:
            raise RuntimeError("Cannot write to a finished cache")
        self._tree_store.extend_with_batch(batch)

        self._ledger.shard_rows[self.shard] += row_count
        self._ledger.total_num_rows += row_count

        self._ledger._serialize_and_commit(self.cache_dir)

    def finish(self):
        self._ledger.is_finished = True
        self._ledger._serialize_and_commit(self.cache_dir)

        return self._tree_store


def _tokenize_one_shard(
    temporary_cache_path: str,
    source: ShardedDataSource,
    shard_name: str,
    processor: BatchProcessor,
    options: CacheOptions | None = None,
):
    # ray breaks if this is top level
    import humanfriendly

    logger = logging.getLogger(f"_tokenize::{shard_name}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ledger = CacheLedger.load_or_initialize(temporary_cache_path, source, processor, options)
    if ledger.is_finished:
        logger.info(f"Shard {shard_name} already processed.")
        return temporary_cache_path, ledger

    writer = ShardCacheWriter(temporary_cache_path, ledger, shard_name, processor.output_exemplar)

    total_rows = ledger.total_num_rows

    shard_iterator = source.open_shard_at_row(shard_name, total_rows)

    prepared_batch: PyTree[PreparedBatch] | None = None
    this_batch_size = 0

    for batch in batched(shard_iterator, options.batch_size):
        tokenized = processor(batch)
        tokenized = _canonicalize_batch(tokenized)  # type: ignore
        this_prepared = writer._tree_store.batch_preparer(tokenized)

        this_batch_size += len(batch)
        total_rows += len(batch)

        if prepared_batch is None:
            prepared_batch = this_prepared
        else:
            prepared_batch = jax.tree.map(lambda *trees: PreparedBatch.concat(trees), prepared_batch, this_prepared)

        batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))

        if batch_byte_size > options.target_bytes_per_flush:
            writer.write_prepared_batch(this_batch_size, prepared_batch)
            nice_bytes = humanfriendly.format_size(batch_byte_size)
            logger.info(f"Processed {total_rows} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})")
            this_batch_size = 0
            prepared_batch = None

    if prepared_batch is not None:
        batch_byte_size = sum(prepared_batch.byte_size for prepared_batch in jax.tree.leaves(prepared_batch))
        nice_bytes = humanfriendly.format_size(batch_byte_size)
        writer.write_prepared_batch(this_batch_size, prepared_batch)
        logger.info(f"Processed {total_rows} rows. Wrote {this_batch_size} rows to {shard_name}. ({nice_bytes})")
        this_batch_size = 0
        prepared_batch = None

    writer.finish_shard(shard_name, total_rows)

    writer.finish()

    logger.info(f"Finished processing {shard_name}. Wrote {total_rows} rows.")

    return temporary_cache_path, writer.ledger


def tokenize_all_shards(
    temporary_cache_path: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    options: CacheOptions,
):
    logger = logging.getLogger("tokenize_all_shards")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"Tokenizing {len(source.shard_names)} shards to {temporary_cache_path}.")

    futures = [
        ray.remote(_tokenize_one_shard)
        .options(  # type: ignore
            num_cpus=processor.num_cpus,
            num_gpus=processor.num_gpus,
            resources=processor.resources,
            memory=1 * 1024 * 1024 * 1024,  # made this up
            name=f"tokenize::{temporary_cache_path}::{shard_name}",
        )
        .remote(os.path.join(temporary_cache_path, shard_name), source, shard_name, processor, options)
        for shard_name in source.shard_names
    ]

    ledgers = ray.get(futures)

    return ledgers


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

    all_futures = jax.tree.map(
        lambda d_arr, *arrs: concatenate_jagged_array_metadata(d_arr, arrs), dest.tree, *[s.tree for s in stores]
    )

    await asyncio.gather(*jax.tree.leaves(all_futures))
    await asyncio.gather(*data_futures)

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


@ray.remote(num_cpus=8)
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

    # to concatenate, we need to adjust the indices, so be careful with that one.

    row_counts = [a.num_rows for a in arrays]

    # we pack the number of rows into the 0'th entry of the indices array.
    # tensorstore doesn't shift indices to 0, so we need to do that ourselves.
    offsetses = [a.offsets[1 : a.num_rows + 1][ts.d[:].translate_to[0]] for a in arrays]

    adjusted_offsets = [
        _virtual_offset(offsets, offset) for offsets, offset in zip(offsetses, data_offsets_per_shard, strict=False)
    ]

    offsets_concat = ts.concat(adjusted_offsets, axis=0)

    offsets_write_future = dest.offsets[1 : offsets_concat.shape[0] + 1].write(offsets_concat)

    futures = [offsets_write_future]

    if dest.shapes is not None:
        # this won't be set for tokenization, but for completeness
        shapes = [a.shapes[0 : a.num_rows] for a in arrays]
        shapes_concat = ts.concat(shapes, axis=0)
        shapes_write_future = dest.shapes[0 : shapes_concat.shape[0], :].write(shapes_concat)
        futures.append(shapes_write_future)

    await offsets_write_future
    # write number of rows
    await dest.offsets[0].write(np.array(sum(row_counts), dtype=int))

    return await asyncio.gather(*futures)


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
        jax.tree.map(partial(lambda i, offset_array: offset_array[i], i), data_offsets)
        for i in range(len(source.shard_names))
    ]

    # SPREAD to take advantage of  the fact that we're copying data from different shards
    # for some reason, this uses a ton of memory
    @ray.remote(scheduling_strategy="SPREAD", memory=8 * 1024 * 1024 * 1024)
    def do_copy(path, data_offset_tree):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        if fsspec_exists(os.path.join(path, ".COPY_SUCCESS")):
            logger.info(f"Data already copied from {path} to {permanent_cache_path}.")
            return

        done = asyncio.run(
            _copy_data_from_one_shard_to_permanent_memory(permanent_cache_path, path, processor, data_offset_tree)
        )

        with open(os.path.join(path, ".COPY_SUCCESS"), "w") as f:
            f.write("")

        return done

    if ray_utils.is_local_ray_cluster():
        do_copy = do_copy.options(memory=1 * 1024 * 1024 * 1024)

    futures = [
        do_copy.remote(path, data_offset_tree)
        for (path, _), data_offset_tree in zip(caches, data_offsets_for_shards, strict=False)
    ]

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
        data = source_array.data[0 : source_array.data_size]
        write_future = dest_array.data[data_offset : data_offset + source_array.data_size].write(data)

        return write_future

    futures = jax.tree.map(_copy_one_array, dest.tree, source.tree, data_offset_tree)

    out = await asyncio.gather(*jax.tree.leaves(futures))
    logger.info(f"Finished copying data from {source_path} to {dest_path}.")
    return out


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
