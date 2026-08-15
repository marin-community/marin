# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map-only Zephyr assign pipeline: int8 embedding shards -> co-partitioned AssignmentAttrData parquet.

Each embedding parquet shard becomes one Zephyr task producing one output
parquet shard with the same basename. Schema::

    id              string
    cluster_<K>     int32     # K=k_train assignment
    dist_<K>        float32   # squared L2 distance to assigned centroid
    cluster_<k>     int32     # for each coarser k in lookups (via agglomerative merge)

FAISS centroids + lookups are loaded per worker process and cached, so a
worker handles many shards with a single load. ``InlineRunner`` keeps that
cache valid across Zephyr tasks. Under ``SubprocessRunner`` each task rebuilds
the index, but reads the matrix from the worker's local disk cache.

Counters: ``assign/docs_in``, ``assign/shards_in``.
"""

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import write_artifact
from pydantic import BaseModel
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

logger = logging.getLogger(__name__)
ASSIGNMENT_ATTR_DATA_VERSION = 2


class AssignmentAttrData(BaseModel):
    """Co-partitioned per-source cluster-assignment parquet shards."""

    version: str = f"v{ASSIGNMENT_ATTR_DATA_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    embedding_output_dir: DatakitArtifactPath
    k_train: int
    k_views: list[int]
    counters: dict[str, int | float] = {}

    def shard_paths(self) -> list[str]:
        return sorted(str(m) for m in StoragePath(f"{self.output_dir.rstrip('/')}/*.parquet").glob())


def _read_npy(uri: str) -> np.ndarray:
    """Load a small ``.npy``, cached on the worker's local disk.

    A subprocess pool re-enters this once per task, so the cache turns one
    download per shard into one per worker.
    """
    digest = hashlib.sha256(uri.encode()).hexdigest()[:12]
    cache_dir = os.path.join(tempfile.gettempdir(), "domain-assign")
    local = os.path.join(cache_dir, f"{digest}-{os.path.basename(uri)}")
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as staged:
            with open_url(uri, "rb") as src:
                shutil.copyfileobj(src, staged)
        # Tasks on one worker share this directory, so publish by rename: a
        # concurrent reader sees either no file or a complete one.
        os.replace(staged.name, local)
    return np.load(local)


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
    paths: Iterator[str],
    shard: ShardInfo,
    *,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    quant_scale: float,
    batch_size: int,
    schema: pa.Schema,
) -> Iterator[pa.RecordBatch]:
    """Per-shard map: dequantize int8 embeddings, FAISS-search against centroids, emit cluster ids.

    Yields ``pa.RecordBatch`` matching ``schema``, so the ``id`` column passes
    through without becoming Python objects.
    """
    ctx = _get_index(centroids_uri, lookup_uris)
    index = ctx["index"]
    k_train: int = ctx["k_train"]
    d: int = ctx["dim"]
    lookups: dict[int, np.ndarray] = ctx["lookups"]

    n_docs = 0
    for path in paths:
        # Read columnar throughout. On an 18k-row shard, ``to_pylist`` plus the
        # list-to-numpy conversion cost 7.3 s against 2.5 s of FAISS, while
        # taking the same values from the Arrow buffer costs 0.1 s.
        #
        # PyArrow's own S3 reader is rejected by the CoreWeave store, so the
        # file handle comes from rigging.
        with open_url(path, "rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(batch_size=batch_size, columns=["id", "embedding"]):
                # ``flatten`` respects the batch's offset, so a sliced column
                # cannot silently read its neighbour's values.
                values = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
                embeddings = values.reshape(batch.num_rows, d).astype(np.float32) * quant_scale
                dist, cluster_train = index.search(embeddings, 1)
                cluster_train_arr = cluster_train[:, 0].astype(np.int32, copy=False)
                n_docs += batch.num_rows

                arrays: list[pa.Array] = [
                    batch.column("id"),
                    pa.array(cluster_train_arr, type=pa.int32()),
                    pa.array(dist[:, 0].astype(np.float32, copy=False), type=pa.float32()),
                ]
                arrays.extend(pa.array(lookups[k][cluster_train_arr], type=pa.int32()) for k in sorted(lookups))
                yield pa.RecordBatch.from_arrays(arrays, schema=schema)

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


def assign_hash_attrs(centroids_identity: str, k_train: int, k_views: Sequence[int], batch_size: int) -> dict[str, Any]:
    """Return the identity of an assign step, which keys its output path.

    Shared so that the producer and any consumer resolving the same outputs
    cannot drift apart. ``centroids_identity`` must be region-independent --
    the model step's ``name_with_hash``, never its resolved path.
    """
    return {
        "centroids_dir": centroids_identity,
        "k_train": k_train,
        "k_views": list(k_views),
        "batch_size": batch_size,
        "v": ASSIGNMENT_ATTR_DATA_VERSION,
    }


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
    *,
    embedding_dir: str,
    embedding_dim: int,
    quantization_scale: float,
    source_key: str,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    batch_size: int = 4096,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
    zephyr_context: ZephyrContext | None = None,
) -> AssignmentAttrData:
    """Map-only Zephyr cluster of every embedding shard in one source.

    Takes the embedding facts rather than an artifact: the luxical and Harrier
    pipelines each define their own ``EmbeddingAttrData``, and one Harrier
    source is a repartition output that carries no such payload.

    Args:
        embedding_dir: Directory of co-partitioned int8 embedding parquet shards.
        embedding_dim: Vector width, which must equal the centroid width.
        quantization_scale: ``fp32 = int8.astype(float32) * scale``.
        source_key: Prefix-relative key of the normalized source these
            embeddings mirror. :func:`build_clustered_store` rejects an
            assignment whose key differs from the tokenize artifact's.
        zephyr_context: Shared pool. Unset builds a dedicated ``InlineRunner``
            pool, which holds one FAISS index for every task it runs. A shared
            pool that multiplexes tasks wants ``SubprocessRunner`` instead: the
            per-worker centroid cache keeps the rebuild cheap, and separate
            processes keep the Python-side decode off one GIL.
    """
    embedding_shards = sorted(str(path) for path in (StoragePath(embedding_dir) / "*.parquet").glob())
    if not embedding_shards:
        raise RuntimeError(f"No embedding shards under {embedding_dir}")

    # Load centroids on the driver just to discover (k_train, dim) for the schema.
    # Workers do their own loads (cached) — this driver read is small (~MB).
    centroids = _read_npy(centroids_uri)
    k_train, d = int(centroids.shape[0]), int(centroids.shape[1])
    if d != embedding_dim:
        raise ValueError(f"centroid dim {d} != embedding dim {embedding_dim}")
    k_views = sorted(int(k) for k in lookup_uris)
    schema = _output_schema(k_train, k_views)

    output_basenames = tuple(os.path.basename(p) for p in embedding_shards)

    def _output_path(shard_idx: int, _total: int, bn: tuple[str, ...] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{bn[shard_idx]}"

    logger.info(
        "Assigning %d shards from %s against K=%d centroids (views: %s)",
        len(embedding_shards),
        embedding_dir,
        k_train,
        k_views,
    )

    ds = (
        # One embedding file per shard, read inside the map so the rows stay in
        # Arrow rather than becoming dicts on the way in.
        Dataset.from_list(embedding_shards)
        .map_shard(
            lambda paths, shard, cu=centroids_uri, lu=lookup_uris, qs=quantization_scale, bs=batch_size: _assign_shard(
                paths, shard, centroids_uri=cu, lookup_uris=lu, quant_scale=qs, batch_size=bs, schema=schema
            )
        )
        .write_parquet(_output_path, schema=schema, skip_existing=True)
    )

    resources = worker_resources or ResourceConfig(cpu=4, ram="8g")
    ctx_z = zephyr_context or ZephyrContext(
        resources=resources,
        max_workers=min(max_workers, len(embedding_shards)),
        name=f"assign-k{k_train}-{os.path.basename(embedding_dir)[:8]}",
        stage_runner_factory=InlineRunner,
    )
    outcome = ctx_z.execute(ds, verbose=True, map_task_resources=resources)

    artifact = AssignmentAttrData(
        output_dir=output_path,
        source_key=source_key,
        embedding_output_dir=embedding_dir,
        k_train=k_train,
        k_views=k_views,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
