# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign every registered Datakit source against the pinned Harrier domain model.

One step per source. Each reads that source's complete Harrier embeddings
(:func:`experiments.datakit.hero_data.harrier`), searches them against the
registered centroids (:func:`hero_data.domain_cluster_assignment`), and writes
co-partitioned ``AssignmentAttrData`` parquet.

All sources share one Zephyr pool. Each worker is many times the size of one
task, so it runs ``WORKER_RESOURCES.cpu // TASK_RESOURCES.cpu`` shards at once.
The pool runs ``SubprocessRunner``: the per-worker centroid cache keeps the
FAISS rebuild cheap, and one process per shard keeps the Python-side decode of
18.7 B embedding rows off a single GIL.

Sizing follows the ``cpu-genoa`` pool of cw-us-east-02a, which ``max_slices``
caps at 4 nodes of 192 vCPU (``lib/iris/config/cw-us-east-02a.yaml``).

Submit through the federation, which hands the job to the cluster that owns the
store. ``--extra datakit`` is what puts faiss in the environment the pool's
actors inherit::

    uv run iris --cluster=marin job run --no-wait \\
        --job-name harrier-assign-all --target-cluster cw-us-east-02a \\
        --priority batch --cpu 2 --memory 8GB --enable-extra-resources \\
        --extra datakit -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.cluster.domain.v0.run_assign
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact, read_record
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging
from zephyr.coordinator import MAX_CONCURRENT_PIPELINES
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit import hero_data
from experiments.datakit.cluster.domain.v0.assign import (
    AssignmentAttrData,
    assign_hash_attrs,
    assign_source,
)
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM, QUANT_SCALE, EmbeddingAttrData

logger = logging.getLogger(__name__)

# The registered model trains K=5000 and coarsens to a single 40-cluster view.
# It ships no other lookup, so do not widen this without retraining.
K_TRAIN = 5000
K_VIEWS = (40,)

BATCH_SIZE = 4096

# Kueue spreads these across the GPU-class nodes, not the small cpu-genoa pool,
# so the ceiling is the cluster's free CPU (about 2,900 vCPU) rather than 768.
# Keeping a worker to a third of a 128-core node leaves those nodes room for the
# GPU work they exist for.
WORKER_RESOURCES = ResourceConfig(cpu=48, ram="192g", disk="64g")
POOL_WORKERS = 36

# Measured on a live worker: 25 concurrent shards drew 25.6 cores and 1.75 GiB
# of RSS each (2.1 GiB peak). One shard is one core. Asking for two halved the
# tasks a worker would take, so 48 fit where 24 ran.
TASK_RESOURCES = ResourceConfig(cpu=1, ram="3g", disk="2g")

# The coordinator's own ceiling. Half the sources hold a single shard, so the
# pool stays fed by taking the largest sources first, not by raising this.
MAX_CONCURRENT = MAX_CONCURRENT_PIPELINES

# The coordinator holds per-shard state for every live pipeline. Its 1 GB
# default died with exit 137 after 33 minutes here, tracking pipelines of
# 14,285 and 12,818 shards with 61,485 completed; every worker then lost the
# actor endpoint and the pool wedged with the job still reading "running".
COORDINATOR_RESOURCES = ResourceConfig(cpu=4, ram="64g", preemptible=False)

_SIZING_PARALLELISM = 32

# The focus crawl's complete embeddings come from the repartition and backfill
# step (#8270), not from ``embed_source``, so that directory holds no
# EmbeddingAttrData. Its shards carry the same schema as every other source.
_REPARTITIONED_SOURCES = {"common-crawl-focus-2026-22"}


@dataclass(frozen=True)
class _EmbeddingInputs:
    """What assign needs to read one source's embeddings."""

    directory: str
    dim: int
    quantization_scale: float
    source_key: str


def _embedding_inputs(source: str) -> _EmbeddingInputs:
    """Resolve the embedding shape for ``source``, whichever step produced it."""
    directory = hero_data.harrier(source)
    if source not in _REPARTITIONED_SOURCES:
        embedding = read_artifact(directory, EmbeddingAttrData)
        return _EmbeddingInputs(
            directory=embedding.output_dir,
            dim=embedding.embedding_dim,
            quantization_scale=embedding.quantization_scale,
            source_key=embedding.source_key,
        )

    # The repartition matched today's normalize document for document, so the
    # normalized step is the authority on the key the store checks against.
    normalized = read_artifact(hero_data.normalized(source).output_path, NormalizedData)
    return _EmbeddingInputs(
        directory=directory,
        dim=HARRIER_DIM,
        quantization_scale=QUANT_SCALE,
        source_key=datakit_source_key(normalized.main_output_dir),
    )


def _assign(
    output_path: str,
    source: str,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    zephyr_context: ZephyrContext,
) -> AssignmentAttrData:
    inputs = _embedding_inputs(source)
    return assign_source(
        output_path=output_path,
        embedding_dir=inputs.directory,
        embedding_dim=inputs.dim,
        quantization_scale=inputs.quantization_scale,
        source_key=inputs.source_key,
        centroids_uri=centroids_uri,
        lookup_uris=lookup_uris,
        batch_size=BATCH_SIZE,
        worker_resources=TASK_RESOURCES,
        zephyr_context=zephyr_context,
    )


def _document_count(source: str) -> int:
    """Documents in a source's embeddings, read from its artifact counters.

    Both artifact shapes carry the counter, so this reads the record rather
    than a typed payload.
    """
    record = read_record(hero_data.harrier(source))
    result = record.result if record else None
    counters = result.get("counters") if result else None
    if not isinstance(counters, dict):
        return 0
    return int(counters.get("zephyr/records_out") or 0)


def _sources_largest_first() -> list[str]:
    """Every source, largest first.

    ``StepRunner`` starts steps in list order, and only ``MAX_CONCURRENT`` run
    at once. Half the sources hold one shard, so alphabetical order would hand
    the pool 16 one-shard pipelines and leave it idle.
    """
    names = hero_data.source_names()
    with ThreadPoolExecutor(max_workers=_SIZING_PARALLELISM) as pool:
        counts = list(pool.map(_document_count, names))
    ordered = sorted(zip(names, counts, strict=True), key=lambda pair: pair[1], reverse=True)
    logger.info("Ordered %d sources, largest first: %s", len(ordered), ordered[:3])
    return [name for name, _ in ordered]


def build_steps(
    zephyr_context: ZephyrContext,
    partition_index: int = 0,
    partition_count: int = 1,
) -> list[StepSpec]:
    """Build one assign step per source in a deterministic partition."""
    if not 0 <= partition_index < partition_count:
        raise ValueError(f"partition index {partition_index} must be in [0, {partition_count})")

    centroids = hero_data.domain_cluster_assignment()
    model_dir = centroids.output_path.rstrip("/")
    centroids_uri = f"{model_dir}/centroids_{K_TRAIN}.npy"
    lookup_uris = {k: f"{model_dir}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS}

    return [
        StepSpec(
            name=f"datakit/cluster_assign/harrier/{source}",
            # The frozen model step carries the pin into each output's identity,
            # so a later model repoints these instead of silently reusing them.
            deps=[centroids],
            hash_attrs=assign_hash_attrs(centroids.name_with_hash, K_TRAIN, K_VIEWS, BATCH_SIZE),
            fn=lambda output_path, src=source: _assign(output_path, src, centroids_uri, lookup_uris, zephyr_context),
        )
        for source in _sources_largest_first()[partition_index::partition_count]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Datakit sources to Harrier domain clusters")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--pool-workers", type=int, default=POOL_WORKERS, help="shared Zephyr worker count")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="sources assigned at one time")
    args = parser.parse_args()

    # One step holds one pipeline for its whole run. Raising the coordinator's
    # ceiling to fit more of them cost a run: at 32 pipelines the workers lost
    # the coordinator's actor endpoint a minute in, and the coordinator then
    # read every one of them as dead while their shards requeued forever.
    if args.max_concurrent > MAX_CONCURRENT_PIPELINES:
        raise ValueError(f"--max-concurrent {args.max_concurrent} exceeds the pool limit {MAX_CONCURRENT_PIPELINES}")

    configure_logging()
    with ZephyrContext(
        name="datakit-assign",
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=args.pool_workers,
        stage_runner_factory=SubprocessRunner,
    ) as pool:
        steps = build_steps(pool, args.partition_index, args.partition_count)
        StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
