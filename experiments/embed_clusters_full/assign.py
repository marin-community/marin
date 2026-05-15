# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign every doc in a source's EmbeddingAttrData to its nearest centroid.

Output is itself a datakit attribute dataset (:class:`AssignmentAttrData`):
one parquet shard per input embedding shard, sharing basenames, sorted by
``id`` (row order preserved end-to-end from normalized → embedding →
assignment). Columns::

    id              string
    cluster_5000    int32     # K=k_train assignment
    dist_5000       float32   # squared L2 distance to assigned centroid
    cluster_1000    int32     # via agglomerative-merge lookup
    cluster_40      int32     # via agglomerative-merge lookup

Per-doc cost at K=5000, d=192 is ~1M FLOP; FAISS BLAS on cpu=8 hits
roughly ~1M docs/min. 15B docs / 1M docs/min / 1000 workers ~= 15 min.
"""

import logging
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.utils import fsspec_glob
from pydantic import BaseModel
from rigging.filesystem import open_url

from experiments.embed_clusters_full.embed_source import EmbeddingAttrData, dequantize_to_fp32

logger = logging.getLogger(__name__)


class AssignmentAttrData(BaseModel):
    """Co-partitioned per-source cluster-assignment parquet shards.

    Mirrors the upstream :class:`EmbeddingAttrData` (and thus the
    ``NormalizedData.main_output_dir``) shard-for-shard, with the same
    basenames and the same row order. Persisted as the step's ``.artifact``.
    """

    version: str = "v1"
    output_dir: str
    source_main_dir: str
    embedding_output_dir: str
    k_train: int
    k_views: list[int]
    counters: dict[str, int] = {}

    def shard_paths(self) -> list[str]:
        return sorted(fsspec_glob(f"{self.output_dir.rstrip('/')}/*.parquet"))


def _read_npy(uri: str) -> np.ndarray:
    local = os.path.join(tempfile.gettempdir(), os.path.basename(uri))
    with open_url(uri, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    arr = np.load(local)
    os.remove(local)
    return arr


def assign_source(
    output_path: str,
    embedding_step_output: str,
    centroids_uri: str,
    lookup_uris: dict[int, str],
) -> AssignmentAttrData:
    """Assign every shard of one source's EmbeddingAttrData; emit AssignmentAttrData."""
    import faiss

    embed_attr = Artifact.from_path(embedding_step_output, EmbeddingAttrData)
    centroids = _read_npy(centroids_uri).astype(np.float32, copy=False)
    k_train, d = centroids.shape
    if d != embed_attr.embedding_dim:
        raise ValueError(f"centroid dim {d} != embedding dim {embed_attr.embedding_dim}")
    lookups = {k: _read_npy(uri).astype(np.int32, copy=False) for k, uri in lookup_uris.items()}

    index = faiss.IndexFlatL2(d)
    index.add(centroids)

    schema_fields: list[pa.Field] = [
        pa.field("id", pa.string()),
        pa.field(f"cluster_{k_train}", pa.int32()),
        pa.field(f"dist_{k_train}", pa.float32()),
    ]
    for k in sorted(lookups):
        schema_fields.append(pa.field(f"cluster_{k}", pa.int32()))
    out_schema = pa.schema(schema_fields)

    shards = embed_attr.shard_paths()
    logger.info("Assigning %d shards from %s against K=%d centroids", len(shards), embed_attr.output_dir, k_train)

    total = 0
    for shard_uri in shards:
        basename = os.path.basename(shard_uri)
        table = pq.read_table(shard_uri)
        ids = table["id"]
        flat_int8 = table["embedding"].values.to_numpy(zero_copy_only=False)
        embeddings = dequantize_to_fp32(flat_int8.reshape(-1, d), scale=embed_attr.quantization_scale)

        dist, cluster_train = index.search(embeddings, 1)
        cluster_train_arr = cluster_train[:, 0].astype(np.int32, copy=False)
        dist_train_arr = dist[:, 0].astype(np.float32, copy=False)

        cols: dict[str, pa.Array | pa.ChunkedArray] = {
            "id": ids,
            f"cluster_{k_train}": pa.array(cluster_train_arr, type=pa.int32()),
            f"dist_{k_train}": pa.array(dist_train_arr, type=pa.float32()),
        }
        for k in sorted(lookups):
            cols[f"cluster_{k}"] = pa.array(lookups[k][cluster_train_arr], type=pa.int32())
        out_table = pa.table(cols, schema=out_schema)

        local_out = os.path.join(tempfile.gettempdir(), basename)
        pq.write_table(out_table, local_out, compression="zstd", use_dictionary=True)
        with open(local_out, "rb") as src, open_url(f"{output_path.rstrip('/')}/{basename}", "wb") as dst:
            dst.write(src.read())
        os.remove(local_out)

        total += out_table.num_rows
        logger.info("Wrote %d assignments to %s", out_table.num_rows, basename)

    artifact = AssignmentAttrData(
        output_dir=output_path,
        source_main_dir=embed_attr.source_main_dir,
        embedding_output_dir=embed_attr.output_dir,
        k_train=int(k_train),
        k_views=sorted(lookups.keys()),
        counters={"shards_out": len(shards), "docs_out": total},
    )
    Artifact.save(artifact, output_path)
    return artifact
