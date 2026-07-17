# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train K=5000 spherical K-means on the centroid sample, derive coarser views.

Writes:
- ``centroids_<k_train>.npy`` — (k_train, d) float32 unit-norm
- ``lookup_<k_train>_to_<k>.npy`` for each k in ``k_views`` (e.g. 1000, 40)
- ``train_stats.json``

Uses FAISS K-means (BLAS-backed, multi-threaded) — at K=5000 on 10M x 192
float32 it takes hours on cpu=32, not days. Agglomerative merging the
trained centroids is trivial (5000x5000 cosine distance, seconds).

Why ``method="average"`` for the merge: K-means centroids are means, so
linkage by average pairwise cosine matches the underlying geometry.
``ward`` requires Euclidean and would distort unit-vector clusters.
"""

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url

from experiments.datakit.embeddings.luxical.pipeline import LUXICAL_DIM, QUANT_SCALE, dequantize_to_fp32

logger = logging.getLogger(__name__)


_LOAD_PARALLELISM = 64
_PROGRESS_INTERVAL = 5000


def _load_sample_parquet(sample_path: str) -> np.ndarray:
    """Load all sample parquet shards, dequantize int8 → fp32, return one stacked array.

    Parquet reads parallelize over ``_LOAD_PARALLELISM`` threads — the bottleneck
    is GCS round-trips on ~100K small files, so threadpool over IO gets a
    ~30-50x speedup vs the serial list-comp.
    """
    # Per-source subdirs: {sample_path}/{source_name.replace('/','-')}/sample-NNNNNN.parquet
    shard_paths = sorted(str(m) for m in StoragePath(f"{sample_path.rstrip('/')}/**/*.parquet").glob())
    if not shard_paths:
        raise RuntimeError(f"No sample parquet shards under {sample_path}")

    logger.info(
        "Loading %d sample parquet shards from %s (%d threads)",
        len(shard_paths),
        sample_path,
        _LOAD_PARALLELISM,
    )
    t0 = time.monotonic()

    def _read(path: str) -> pa.Table:
        # fsspec handle rather than a path: pyarrow's native S3 filesystem issues
        # HeadObject probes the CW object store rejects with HTTP 400.
        with StoragePath(path).open("rb") as fh:
            return pq.read_table(fh, columns=["embedding"])

    tables: list[pa.Table] = [None] * len(shard_paths)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=_LOAD_PARALLELISM) as pool:
        for i, t in enumerate(pool.map(_read, shard_paths)):
            tables[i] = t
            if (i + 1) % _PROGRESS_INTERVAL == 0:
                logger.info("Loaded %d/%d shards", i + 1, len(shard_paths))

    table = pa.concat_tables(tables)
    fsl = table["embedding"].combine_chunks()
    flat_int8 = fsl.values.to_numpy(zero_copy_only=False)
    embeddings_int8 = flat_int8.reshape(-1, LUXICAL_DIM)
    embeddings = dequantize_to_fp32(embeddings_int8, scale=QUANT_SCALE)
    logger.info(
        "Loaded sample (%d x %d) in %.1fs",
        embeddings.shape[0],
        embeddings.shape[1],
        time.monotonic() - t0,
    )
    return embeddings


def train_centroids(
    output_path: str,
    sample_path: str,
    k_train: int = 5000,
    k_views: tuple[int, ...] = (40, 1000),
    n_iter: int = 20,
    n_redo: int = 3,
    seed: int = 42,
    n_threads: int = 32,
) -> None:
    """Train K=k_train spherical K-means, then agglomerative-merge to each k in k_views.

    Determinism: the result is pinned to ``(sample, seed, n_threads)``. ``seed``
    fixes FAISS's K-means init/restarts; ``n_threads`` fixes the OpenMP/BLAS
    thread count for both the K-means (``faiss.omp_set_num_threads``) and the
    agglomerative-merge cosine matmul + linkage (``threadpool_limits``). Without
    the thread pin, FAISS and BLAS default to the *node's* core count, so the
    float-reduction order -- and thus the centroids and the K-view lookups --
    would vary with machine placement rather than the config (marin#6798).
    """
    import faiss  # noqa: PLC0415  # optional dep: faiss
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415  # optional dep: scipy
    from scipy.spatial.distance import squareform  # noqa: PLC0415  # optional dep: scipy
    from threadpoolctl import threadpool_limits  # noqa: PLC0415  # optional dep: threadpoolctl

    faiss.omp_set_num_threads(n_threads)
    embeddings = _load_sample_parquet(sample_path)
    logger.info(
        "Running K-means K=%d on %d x %d sample (n_threads=%d, seed=%d)", k_train, *embeddings.shape, n_threads, seed
    )

    # Pin BLAS threads (numpy matmul + scipy linkage) alongside the FAISS OMP pin
    # so every float reduction in this process uses a fixed, config-derived thread
    # count regardless of the node's physical core count.
    with threadpool_limits(limits=n_threads):
        t0 = time.monotonic()
        km = faiss.Kmeans(
            d=int(embeddings.shape[1]),
            k=k_train,
            niter=n_iter,
            nredo=n_redo,
            spherical=True,  # renormalize centroids each iter — matches Luxical unit-norm output
            seed=seed,
            verbose=True,
        )
        km.train(embeddings)
        centroids = km.centroids.astype(np.float32, copy=False)
        train_s = time.monotonic() - t0
        final_obj = float(km.obj[-1])
        logger.info("K-means K=%d done in %.0fs (final obj=%.4f)", k_train, train_s, final_obj)

        _save_npy(centroids, output_path, f"centroids_{k_train}.npy")

        # Cosine distance matrix between centroids (1 - cos sim, since centroids are unit-norm).
        # squareform expects a condensed upper-triangle.
        sim = centroids @ centroids.T
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")

        for k in k_views:
            labels = fcluster(Z, t=k, criterion="maxclust") - 1
            labels = labels.astype(np.int32, copy=False)
            _save_npy(labels, output_path, f"lookup_{k_train}_to_{k}.npy")
            logger.info(
                "Agglomerative merge: K=%d → K=%d (got %d unique labels)",
                k_train,
                k,
                int(labels.max()) + 1,
            )

    stats = {
        "k_train": int(k_train),
        "k_views": list(k_views),
        "n_sample": len(embeddings),
        "n_iter": int(n_iter),
        "n_redo": int(n_redo),
        "seed": int(seed),
        "n_threads": int(n_threads),
        "train_s": float(train_s),
        "obj_final": final_obj,
    }
    with open_url(os.path.join(output_path, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def _save_npy(arr: np.ndarray, output_path: str, name: str) -> None:
    local = os.path.join(tempfile.gettempdir(), name)
    np.save(local, arr)
    with open(local, "rb") as src, open_url(os.path.join(output_path, name), "wb") as dst:
        dst.write(src.read())
    os.remove(local)
