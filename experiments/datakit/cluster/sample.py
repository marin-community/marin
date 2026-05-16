# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stratified sample across all per-source EmbeddingAttrData → centroid training input.

Map-only Zephyr pipeline. One task per ``(source, embedding shard)`` reads the
shard, samples up to ``per_shard`` rows uniformly, and writes those rows to a
parquet file labeled with the source. Output schema::

    source     string
    embedding  list<int8> length 192   (same encoding as EmbeddingAttrData)

Per-source cap: ``per_shard = max(n_per_source // num_shards, 1)`` so every source
contributes roughly ``n_per_source`` rows (slightly under if not divisible).
The cap (not strict proportional) keeps long-tail sources audible against the
giants. At ~100 active sources x n_per_source=100_000 the result is ~10M rows.

Train consumes ``*.parquet`` from the output dir; dequantization to fp32 with
``dequantize_to_fp32`` happens there. Each shard is independent so the pipeline
is fully parallelizable and resumable across preemptions (``skip_existing=True``
on the writer means previously-completed sample shards survive a restart).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
from fray import ResourceConfig
from zephyr import Dataset, InputFileSpec, ShardInfo, ZephyrContext, load_file
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


def sample_centroid_inputs(
    output_path: str,
    embeddings: dict[str, EmbeddingAttrData],
    n_per_source: int,
    seed: int = 42,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
) -> None:
    """Map-only Zephyr sample across every (source, shard) pair → parquet shards."""
    shard_paths: list[str] = []
    source_by_shard: list[str] = []
    per_shard_by_shard: list[int] = []

    for source_name, attr in sorted(embeddings.items()):
        shards = attr.shard_paths()
        if not shards:
            logger.warning("No embedding shards for %s under %s", source_name, attr.output_dir)
            continue
        per_shard = max(n_per_source // len(shards), 1)
        for shard_path in shards:
            shard_paths.append(shard_path)
            source_by_shard.append(source_name)
            per_shard_by_shard.append(per_shard)

    if not shard_paths:
        raise RuntimeError("No embedding shards found across any source")

    logger.info(
        "Sampling %d shards across %d sources (n_per_source=%d, ~%d total rows)",
        len(shard_paths),
        len({s for s in source_by_shard}),
        n_per_source,
        sum(per_shard_by_shard),
    )

    output_basenames = [f"{s.replace('/', '-')}-{i:06d}.parquet" for i, s in enumerate(source_by_shard)]

    def _output_path(shard_idx: int, _total: int, bn: list[str] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{bn[shard_idx]}"

    source_specs = [InputFileSpec(path=p, columns=["embedding"]) for p in shard_paths]

    ds = (
        Dataset.from_list(source_specs)
        .flat_map(load_file)
        .map_shard(
            lambda batches, shard, sn=source_by_shard, ps=per_shard_by_shard, sd=seed: _sample_shard(
                batches,
                shard,
                source_name=sn[shard.shard_idx],
                per_shard=ps[shard.shard_idx],
                seed=sd,
            )
        )
        .write_parquet(_output_path, schema=_SAMPLE_SCHEMA, skip_existing=True)
    )

    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=2, ram="4g")

    ctx = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(shard_paths)),
        name=f"sample-centroid-{os.path.basename(output_path.rstrip('/'))[:12]}",
        # InlineRunner so workers reuse process-level state across shards
        # (here mostly numpy/pyarrow warmup; modest win at this fan-out).
        stage_runner_factory=InlineRunner,
    )
    ctx.execute(ds, verbose=True)
