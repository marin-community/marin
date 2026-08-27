# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Levanter cache metadata startup and TensorStore codecs on object storage.

The benchmark writes disposable data beneath ``marin_temp_bucket``. It measures
the existing per-component ledger startup path against a merged cache catalog,
then writes and reads identical token arrays with Blosc/LZ4 level 5 and
Blosc/Zstd level 1.
"""

import argparse
import gc
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from levanter.data.text.datasets import CACHE_METADATA_WORKERS, DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.store.cache import (
    CACHE_LAYOUT_SHARDED,
    LEDGER_FILE_NAME,
    CacheLedger,
    CacheMetadata,
)
from levanter.store.jagged_array import BloscCodec, JaggedArrayStore, PreparedBatch
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

BENCHMARK_OUTPUT_TTL_DAYS = 1
COREWEAVE_DATA_PREFIX = "s3://marin-us-east-02a/marin"
DEFAULT_COMPONENTS = 300
DEFAULT_SHARDS_PER_COMPONENT = 10
DEFAULT_ROWS_PER_SHARD = 100
DEFAULT_TOKENS_PER_ROW = 2048
DEFAULT_NUM_TOKENS = 64 * 1024 * 1024
DEFAULT_STARTUP_REPETITIONS = 5
DEFAULT_CODEC_REPETITIONS = 2
TOKEN_VOCAB_SIZE = 32_768
TOKEN_ZIPF_EXPONENT = 1.2


@dataclass(frozen=True)
class SyntheticCacheShape:
    num_shards: int
    rows_per_shard: int
    tokens_per_row: int


LZ4 = BloscCodec("lz4", 5)
ZSTD = BloscCodec("zstd", 1)


def _component_ledger(shape: SyntheticCacheShape) -> CacheLedger:
    shard_names = [f"shard-{shard_index:04d}" for shard_index in range(shape.num_shards)]
    shard_rows = {name: shape.rows_per_shard for name in shard_names}
    field_counts_by_shard = {name: {"input_ids": shape.rows_per_shard * shape.tokens_per_row} for name in shard_names}
    return CacheLedger(
        total_num_rows=shape.num_shards * shape.rows_per_shard,
        shard_rows=shard_rows,
        is_finished=True,
        finished_shards=shard_names,
        field_counts={"input_ids": shape.num_shards * shape.rows_per_shard * shape.tokens_per_row},
        field_counts_by_shard=field_counts_by_shard,
        layout=CACHE_LAYOUT_SHARDED,
        metadata=CacheMetadata(preprocessor_metadata={"input_ids_key": "input_ids", "loss_weights_key": None}),
    )


def _write_component_ledgers(
    output_prefix: str,
    *,
    num_components: int,
    shape: SyntheticCacheShape,
) -> dict[str, DatasetComponent]:
    ledger = _component_ledger(shape)
    components = {
        f"component-{component_index:04d}": DatasetComponent(
            source=None,
            cache_dir=prefix_join(output_prefix, f"components/component-{component_index:04d}"),
            format=PrebuiltLmDatasetFormat(),
            flat_cache=True,
        )
        for component_index in range(num_components)
    }

    def write_one(component: DatasetComponent) -> None:
        assert component.cache_dir is not None
        StoragePath(prefix_join(component.cache_dir, LEDGER_FILE_NAME)).write_text(ledger.to_json())

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(CACHE_METADATA_WORKERS, num_components)) as pool:
        list(pool.map(write_one, components.values()))
    logger.info("Wrote %d component ledgers in %.3f seconds", num_components, time.perf_counter() - started)
    return components


def _data_config(components: dict[str, DatasetComponent], catalog_path: str | None = None) -> LmDataConfig:
    return LmDataConfig(
        components=components,
        cache_dir=None,
        cache_catalog=catalog_path,
        tokenizer="passthrough",
        vocab_size=TOKEN_VOCAB_SIZE,
    )


def _time_startup(components: dict[str, DatasetComponent], catalog_path: str | None) -> float:
    gc.collect()
    started = time.perf_counter()
    caches = _data_config(components, catalog_path).build_caches("train")
    elapsed = time.perf_counter() - started
    if len(caches) != len(components):
        raise RuntimeError(f"Loaded {len(caches)} caches for {len(components)} components")
    return elapsed


def benchmark_catalog(
    output_prefix: str,
    *,
    num_components: int,
    shape: SyntheticCacheShape,
    repetitions: int,
) -> dict:
    components = _write_component_ledgers(
        output_prefix,
        num_components=num_components,
        shape=shape,
    )
    catalog_path = prefix_join(output_prefix, "cache-catalog.json")

    started = time.perf_counter()
    catalog = _data_config(components).build_cache_catalog(catalog_path, splits=("train",))
    catalog_build_seconds = time.perf_counter() - started
    if len(catalog.splits["train"]) != num_components:
        raise RuntimeError("Catalog does not contain every component")

    individual_seconds: list[float] = []
    catalog_seconds: list[float] = []
    for repetition in range(repetitions):
        if repetition % 2 == 0:
            individual_seconds.append(_time_startup(components, None))
            catalog_seconds.append(_time_startup(components, catalog_path))
        else:
            catalog_seconds.append(_time_startup(components, catalog_path))
            individual_seconds.append(_time_startup(components, None))

    individual_median = statistics.median(individual_seconds)
    catalog_median = statistics.median(catalog_seconds)
    return {
        "components": num_components,
        "shards_per_component": shape.num_shards,
        "catalog_path": catalog_path,
        "catalog_bytes": StoragePath(catalog_path).size(),
        "catalog_build_seconds": catalog_build_seconds,
        "individual_startup_seconds": individual_seconds,
        "catalog_startup_seconds": catalog_seconds,
        "individual_startup_median_seconds": individual_median,
        "catalog_startup_median_seconds": catalog_median,
        "startup_speedup": individual_median / catalog_median,
    }


def _stored_bytes(path: str) -> int:
    fs, fs_path = url_to_fs(path)
    return int(fs.du(fs_path, total=True))


def _token_write_seconds(path: str, tokens: np.ndarray, codec: BloscCodec) -> float:
    started = time.perf_counter()
    store = JaggedArrayStore.open(
        path,
        mode="w",
        item_rank=1,
        dtype=np.int32,
        cache_metadata=True,
        write_codec=codec,
    )
    store.extend(PreparedBatch(data=tokens, offsets=np.array([tokens.size], dtype=np.int64), shapes=None))
    return time.perf_counter() - started


def _token_read_seconds(path: str, num_tokens: int, expected_edges: tuple[list[int], list[int]]) -> float:
    started = time.perf_counter()
    store = JaggedArrayStore.open(path, mode="r", item_rank=1, dtype=np.int32, cache_metadata=False)
    tokens = np.asarray(store.data[:num_tokens].read().result())
    elapsed = time.perf_counter() - started
    if tokens[:8].tolist() != expected_edges[0] or tokens[-8:].tolist() != expected_edges[1]:
        raise RuntimeError(f"Token verification failed for {path}")
    return elapsed


def benchmark_codecs(output_prefix: str, *, num_tokens: int, repetitions: int) -> dict:
    rng = np.random.default_rng(0)
    tokens = np.remainder(rng.zipf(TOKEN_ZIPF_EXPONENT, size=num_tokens), TOKEN_VOCAB_SIZE).astype(np.int32)
    expected_edges = (tokens[:8].tolist(), tokens[-8:].tolist())
    raw_bytes = tokens.nbytes
    measurements = {
        codec.compressor: {"write_seconds": [], "read_seconds": [], "stored_bytes": []} for codec in (LZ4, ZSTD)
    }
    paths: dict[tuple[int, str], str] = {}

    for repetition in range(repetitions):
        codecs = (LZ4, ZSTD) if repetition % 2 == 0 else (ZSTD, LZ4)
        for codec in codecs:
            path = prefix_join(output_prefix, f"codecs/{repetition}/{codec.compressor}")
            paths[repetition, codec.compressor] = path
            measurements[codec.compressor]["write_seconds"].append(_token_write_seconds(path, tokens, codec))
            measurements[codec.compressor]["stored_bytes"].append(_stored_bytes(path))

    for repetition in range(repetitions):
        codecs = (ZSTD, LZ4) if repetition % 2 == 0 else (LZ4, ZSTD)
        for codec in codecs:
            measurements[codec.compressor]["read_seconds"].append(
                _token_read_seconds(paths[repetition, codec.compressor], num_tokens, expected_edges)
            )

    result = {"num_tokens": num_tokens, "raw_bytes": raw_bytes, "codecs": {}}
    for codec in (LZ4, ZSTD):
        codec_measurements = measurements[codec.compressor]
        write_median = statistics.median(codec_measurements["write_seconds"])
        read_median = statistics.median(codec_measurements["read_seconds"])
        stored_median = statistics.median(codec_measurements["stored_bytes"])
        result["codecs"][codec.compressor] = {
            "compression_level": codec.compression_level,
            **codec_measurements,
            "stored_bytes_median": stored_median,
            "compression_ratio": raw_bytes / stored_median,
            "write_median_seconds": write_median,
            "write_mib_per_second": raw_bytes / write_median / 1024**2,
            "read_median_seconds": read_median,
            "read_mib_per_second": raw_bytes / read_median / 1024**2,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--shards-per-component", type=int, default=DEFAULT_SHARDS_PER_COMPONENT)
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--tokens-per-row", type=int, default=DEFAULT_TOKENS_PER_ROW)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--startup-repetitions", type=int, default=DEFAULT_STARTUP_REPETITIONS)
    parser.add_argument("--codec-repetitions", type=int, default=DEFAULT_CODEC_REPETITIONS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    output_prefix = marin_temp_bucket(
        ttl_days=BENCHMARK_OUTPUT_TTL_DAYS,
        prefix=f"levanter-cache-benchmark/{args.run_tag}",
        source_prefix=COREWEAVE_DATA_PREFIX,
    )
    cache_shape = SyntheticCacheShape(
        num_shards=args.shards_per_component,
        rows_per_shard=args.rows_per_shard,
        tokens_per_row=args.tokens_per_row,
    )
    result = {
        "run_tag": args.run_tag,
        "output_prefix": output_prefix,
        "catalog": benchmark_catalog(
            output_prefix,
            num_components=args.components,
            shape=cache_shape,
            repetitions=args.startup_repetitions,
        ),
        "codec": benchmark_codecs(
            output_prefix,
            num_tokens=args.num_tokens,
            repetitions=args.codec_repetitions,
        ),
    }
    result_path = prefix_join(output_prefix, "results.json")
    StoragePath(result_path).write_text(json.dumps(result, indent=2, sort_keys=True))
    logger.info("BENCHMARK_RESULT %s", json.dumps(result, sort_keys=True))
    logger.info("Wrote benchmark result to %s", result_path)


if __name__ == "__main__":
    main()
