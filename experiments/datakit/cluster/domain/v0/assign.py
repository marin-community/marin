# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map-only Zephyr assign pipeline: EmbeddingAttrData -> co-partitioned AssignmentAttrData parquet.

Each embedding parquet shard becomes one Zephyr task producing one output
parquet shard with the same basename. Schema::

    id              string
    cluster_<K>     int32     # K=k_train assignment
    dist_<K>        float32   # squared L2 distance to assigned centroid
    cluster_<k>     int32     # for each coarser k in lookups (via agglomerative merge)

FAISS centroids + lookups are loaded per worker process and cached, so a
worker handles many shards with a single load. ``InlineRunner`` keeps that
cache valid across Zephyr tasks.

Counters: ``assign/docs_in``, ``assign/shards_in``.
"""

import logging
import os
import tempfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
from fray import ResourceConfig
from marin.execution.artifact import write_artifact
from pydantic import BaseModel
from rigging.filesystem import StoragePath, open_url
from zephyr import Dataset, InputFileSpec, ShardInfo, ZephyrContext, counters, load_file
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.luxical.pipeline import EmbeddingAttrData, dequantize_to_fp32

logger = logging.getLogger(__name__)


class AssignmentAttrData(BaseModel):
    """Co-partitioned per-source cluster-assignment parquet shards."""

    version: str = "v1"
    output_dir: str
    source_main_dir: str
    embedding_output_dir: str
    k_train: int
    k_views: list[int]
    counters: dict[str, int | float] = {}

    def shard_paths(self) -> list[str]:
        return sorted(str(m) for m in StoragePath(f"{self.output_dir.rstrip('/')}/*.parquet").glob())


def _read_npy(uri: str) -> np.ndarray:
    local = os.path.join(tempfile.gettempdir(), os.path.basename(uri))
    with open_url(uri, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    arr = np.load(local)
    os.remove(local)
    return arr


# Per-process FAISS index + lookups cache (one-time download + index.add per worker).
_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def _get_index(centroids_uri: str, lookup_uris: dict[int, str]) -> dict[str, Any]:
    """Build or fetch a cached FAISS index + lookups for this worker process."""
    if centroids_uri not in _INDEX_CACHE:
        import faiss  # noqa: PLC0415  # optional dep: faiss

        logger.info("Loading centroids from %s", centroids_uri)
        centroids = _read_npy(centroids_uri).astype(np.float32, copy=False)
        k_train, d = centroids.shape
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        lookups = {k: _read_npy(uri).astype(np.int32, copy=False) for k, uri in lookup_uris.items()}
        _INDEX_CACHE[centroids_uri] = {
            "index": index,
            "k_train": int(k_train),
            "dim": int(d),
            "lookups": lookups,
        }
    return _INDEX_CACHE[centroids_uri]


def _assign_shard(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
    *,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    quant_scale: float,
) -> Iterator[dict]:
    """Per-shard map: dequantize int8 embeddings, FAISS-search against centroids, emit cluster ids."""
    ctx = _get_index(centroids_uri, lookup_uris)
    index = ctx["index"]
    k_train: int = ctx["k_train"]
    d: int = ctx["dim"]
    lookups: dict[int, np.ndarray] = ctx["lookups"]
    cluster_col = f"cluster_{k_train}"
    dist_col = f"dist_{k_train}"

    n_docs = 0
    for batch in batches:
        ids = [r["id"] for r in batch]
        # embeddings come in as list[int] per record from upstream parquet read;
        # reshape into a (B, dim) int8 matrix then dequantize.
        flat_int8 = np.asarray([r["embedding"] for r in batch], dtype=np.int8)
        embeddings = dequantize_to_fp32(flat_int8.reshape(-1, d), scale=quant_scale)
        dist, cluster_train = index.search(embeddings, 1)
        cluster_train_arr = cluster_train[:, 0].astype(np.int32, copy=False)
        dist_train_arr = dist[:, 0].astype(np.float32, copy=False)
        n_docs += len(ids)

        # Precompute the coarser-K cluster ids for this batch via the lookups.
        coarser = {k: lookups[k][cluster_train_arr] for k in lookups}

        for i, did in enumerate(ids):
            rec: dict[str, Any] = {
                "id": did,
                cluster_col: int(cluster_train_arr[i]),
                dist_col: float(dist_train_arr[i]),
            }
            for k in lookups:
                rec[f"cluster_{k}"] = int(coarser[k][i])
            yield rec

    counters.pipeline.update_counter("assign/docs_in", n_docs)
    counters.pipeline.update_counter("assign/shards_in", 1)
    logger.info(
        "shard %d/%d: %d docs assigned (K=%d centroids, %d coarser views)",
        shard.shard_idx,
        shard.total_shards,
        n_docs,
        k_train,
        len(lookups),
    )


def _output_schema(k_train: int, k_views: list[int]) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("id", pa.string()),
        pa.field(f"cluster_{k_train}", pa.int32()),
        pa.field(f"dist_{k_train}", pa.float32()),
    ]
    for k in sorted(k_views):
        fields.append(pa.field(f"cluster_{k}", pa.int32()))
    return pa.schema(fields)


def assign_source(
    output_path: str,
    embedding: EmbeddingAttrData,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    *,
    window_size: int = 4096,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
) -> AssignmentAttrData:
    """Map-only Zephyr cluster of every shard in one source's EmbeddingAttrData."""
    embedding_shards = embedding.shard_paths()
    if not embedding_shards:
        raise RuntimeError(f"No embedding shards under {embedding.output_dir}")

    # Load centroids on the driver just to discover (k_train, dim) for the schema.
    # Workers do their own loads (cached) — this driver read is small (~MB).
    centroids = _read_npy(centroids_uri)
    k_train, d = int(centroids.shape[0]), int(centroids.shape[1])
    if d != embedding.embedding_dim:
        raise ValueError(f"centroid dim {d} != embedding dim {embedding.embedding_dim}")
    k_views = sorted(int(k) for k in lookup_uris)
    schema = _output_schema(k_train, k_views)

    output_basenames = tuple(os.path.basename(p) for p in embedding_shards)

    def _output_path(shard_idx: int, _total: int, bn: tuple[str, ...] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{bn[shard_idx]}"

    logger.info(
        "Assigning %d shards from %s against K=%d centroids (views: %s)",
        len(embedding_shards),
        embedding.output_dir,
        k_train,
        k_views,
    )

    source_specs = [InputFileSpec(path=p, columns=["id", "embedding"]) for p in embedding_shards]

    quant_scale = embedding.quantization_scale

    ds = (
        Dataset.from_list(source_specs)
        .flat_map(load_file)
        .window(window_size)
        .map_shard(
            lambda batches, shard, cu=centroids_uri, lu=lookup_uris, qs=quant_scale: _assign_shard(
                batches, shard, centroids_uri=cu, lookup_uris=lu, quant_scale=qs
            )
        )
        .write_parquet(_output_path, schema=schema, skip_existing=True)
    )

    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=4, ram="8g")

    ctx_z = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(embedding_shards)),
        name=f"assign-k{k_train}-{os.path.basename(embedding.output_dir)[:8]}",
        stage_runner_factory=InlineRunner,
    )
    outcome = ctx_z.execute(ds, verbose=True)

    artifact = AssignmentAttrData(
        output_dir=output_path,
        source_main_dir=embedding.source_main_dir,
        embedding_output_dir=embedding.output_dir,
        k_train=k_train,
        k_views=k_views,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
