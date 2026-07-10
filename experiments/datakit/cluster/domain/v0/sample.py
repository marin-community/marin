# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stratified sample across all per-source EmbeddingAttrData → centroid training input.

Map-only Zephyr pipeline, one ``ZephyrContext`` per source. Each context
spawns one task per embedding shard for its source: read the shard, sample
up to ``per_shard`` rows uniformly, write a parquet file labeled with the
source. Output schema (per shard)::

    source     string
    embedding  list<int8> length 192   (same encoding as EmbeddingAttrData)

Why one context per source instead of one global context: at ~100 sources x
~1K shards/source you get ~100K total tasks. A single Zephyr coordinator
tracking that fan-out OOMs at the default container RAM. Splitting into
per-source contexts keeps each coordinator's task table bounded.

Per-source cap: ``per_shard = max(n_per_source // num_shards, 1)`` so every
source contributes roughly ``n_per_source`` rows. The cap (not strict
proportional) keeps long-tail sources audible against the giants. At ~100
active sources x n_per_source=100_000 the result is ~10M rows.

Train consumes ``**/*.parquet`` from the output dir; dequantization to fp32
with ``dequantize_to_fp32`` happens there. Each per-source output dir is
independent so the pipeline is fully resumable across preemptions
(``skip_existing=True`` on the writer means previously-completed shards
survive a restart).
"""

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.luxical.pipeline import LUXICAL_DIM, EmbeddingAttrData

logger = logging.getLogger(__name__)


_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), LUXICAL_DIM)),
    ]
)


def _sample_shard(
    batches: Iterator[list[dict[str, Any]]],
    shard: ShardInfo,
    *,
    source_name: str,
    per_shard: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Per-shard map: collect rows, uniform-sample ``per_shard`` of them, emit with source label.

    Reads the whole shard into memory because parquet row groups can't be
    sampled randomly without first knowing row counts. Shards are MB-sized
    (~22K rows x 192-byte int8 embedding) so this is cheap.
    """
    rng = np.random.default_rng(seed + shard.shard_idx)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        rows.extend(batch)
    if per_shard < len(rows):
        idx = rng.choice(len(rows), size=per_shard, replace=False)
        rows = [rows[i] for i in idx]
    for r in rows:
        yield {"source": source_name, "embedding": r["embedding"]}


def _sample_one_source(
    source_name: str,
    attr: EmbeddingAttrData,
    output_path: str,
    n_per_source: int,
    seed: int,
    worker_resources: ResourceConfig,
    max_workers: int,
) -> None:
    """Run one ZephyrContext to sample all of one source's embedding shards."""
    shards = attr.shard_paths()
    if not shards:
        logger.warning("No embedding shards for %s under %s", source_name, attr.output_dir)
        return

    per_shard = max(n_per_source // len(shards), 1)
    # Per-source subdirectory so writers can't collide across sources.
    source_dir = f"{output_path.rstrip('/')}/{source_name.replace('/', '-')}"

    def _out(shard_idx: int, _total: int, dest: str = source_dir) -> str:
        return f"{dest}/sample-{shard_idx:06d}.parquet"

    source_specs = [InputFileSpec(path=p, columns=["embedding"]) for p in shards]

    ds = (
        Dataset.from_list(source_specs)
        .flat_map(load_file)
        .window(4096)
        .map_shard(
            lambda batches, shard, sn=source_name, ps=per_shard, sd=seed: _sample_shard(
                batches, shard, source_name=sn, per_shard=ps, seed=sd
            )
        )
        .write_parquet(_out, schema=_SAMPLE_SCHEMA, skip_existing=True)
    )

    ctx = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(shards)),
        name=f"sample-{source_name.replace('/', '-')[:32]}",
        stage_runner_factory=InlineRunner,
    )
    logger.info(
        "Sampling %s: %d shards, per_shard=%d (~%d rows)",
        source_name,
        len(shards),
        per_shard,
        per_shard * len(shards),
    )
    ctx.execute(ds, verbose=True)


def sample_centroid_inputs(
    output_path: str,
    embeddings: dict[str, EmbeddingAttrData],
    n_per_source: int,
    seed: int = 42,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
    parallel_sources: int = 8,
) -> None:
    """Stratified sample across every source via per-source Zephyr contexts.

    ``parallel_sources`` ZephyrContexts run concurrently via a thread pool;
    each spawns its own coordinator + worker pool, so total resource use is
    ``parallel_sources * (1 coord + up to max_workers workers)``. Sources
    are still chosen in sorted order; first-completed-first-reported on the
    way out.
    """
    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=2, ram="4g")

    items = sorted(embeddings.items())
    total = len(items)
    logger.info(
        "Sample pipeline: %d sources, %d concurrent Zephyr contexts, %d workers each",
        total,
        parallel_sources,
        max_workers,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=parallel_sources) as pool:
        futures = {
            pool.submit(
                _sample_one_source,
                source_name=sn,
                attr=a,
                output_path=output_path,
                n_per_source=n_per_source,
                seed=seed,
                worker_resources=worker_resources,
                max_workers=max_workers,
            ): sn
            for sn, a in items
        }
        for fut in as_completed(futures):
            source_name = futures[fut]
            try:
                fut.result()
            except Exception:
                logger.exception("Source %s failed", source_name)
                raise
            completed += 1
            logger.info("Completed %d/%d: %s", completed, total, source_name)

    logger.info("Sample pipeline complete: %d sources written under %s", total, output_path)
