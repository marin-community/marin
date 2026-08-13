# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and coarsen Harrier domain centroids."""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url, prefix_join

from experiments.datakit.cluster.domain.v1.coarsen import (
    ClusteringStats,
    CoarseningConfig,
    CoarseningResult,
    coarsen_centroids,
)
from experiments.datakit.cluster.domain.v1.sample import largest_remainder_quotas
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM, dequantize_to_fp32

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrainingData:
    embeddings: np.ndarray
    population: np.ndarray
    rows_by_source: dict[str, int]


class _SearchIndex(Protocol):
    def search(self, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class _TrainedCentroids:
    centroids: np.ndarray
    index: _SearchIndex
    seconds: float
    objective: float


def training_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    if target < len(counts):
        raise ValueError("target is too small for the one-document source floor")
    if target > sum(counts.values()):
        raise ValueError("target exceeds the available documents")

    sources = tuple(sorted(counts))
    remaining = target - len(counts)
    if remaining == 0:
        return {source: 1 for source in sources}
    extra = largest_remainder_quotas(tuple(counts[source] - 1 for source in sources), remaining)
    return {source: quota + 1 for source, quota in zip(sources, extra, strict=True)}


def stratified_indices(
    source_codes: np.ndarray,
    source_names: tuple[str, ...],
    target: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    counts_array = np.bincount(source_codes, minlength=len(source_names))
    counts = dict(zip(source_names, counts_array.tolist(), strict=True))
    quotas = training_quotas(counts, target)
    selected = []
    source_codes_by_name = {source: code for code, source in enumerate(source_names)}
    for source in sorted(source_names):
        code = source_codes_by_name[source]
        source_indices = np.flatnonzero(source_codes == code)
        quota = quotas[source]
        if quota == len(source_indices):
            selected.append(source_indices)
            continue
        digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        selected.append(rng.choice(source_indices, size=quota, replace=False))
    return np.sort(np.concatenate(selected)), quotas


def _stats_payload(value: ClusteringStats) -> dict[str, Any]:
    return {
        "loss": value.loss,
        "weighted_mean_cosine_distance": value.weighted_mean_cosine_distance,
        "minimum_weight_fraction": value.minimum_weight_fraction,
        "maximum_weight_fraction": value.maximum_weight_fraction,
        "weight_fractions": value.weight_fractions.tolist(),
        "fine_centroid_counts": value.fine_centroid_counts.tolist(),
    }


def _coarsening_payload(result: CoarseningResult, config: CoarseningConfig) -> dict[str, Any]:
    return {
        "method": "weighted_divisive_spherical_kmeans_then_single_centroid_hill_climb",
        "minimum_fraction": config.minimum_fraction,
        "seeds": list(config.seeds),
        "split_restarts": config.split_restarts,
        "split_iterations": config.split_iterations,
        "selected_seed": result.selected_seed,
        "divisive_runs": [{"seed": run.seed, "stats": _stats_payload(run.stats)} for run in result.divisive_runs],
        "initial": _stats_payload(result.initial),
        "final": _stats_payload(result.final),
        "moves": result.moves,
        "sweeps": result.sweeps,
    }


def _read_sample_table(path: str) -> pa.Table:
    with StoragePath(path).open("rb") as file:
        return pq.read_table(file, columns=["source", "embedding"])


def _load_training_embeddings(
    sample_path: str,
    target: int,
    seed: int,
    load_parallelism: int,
) -> _TrainingData:
    paths = sorted(str(path) for path in StoragePath(prefix_join(sample_path, "**/*.parquet")).glob())
    if not paths:
        raise FileNotFoundError(f"No centroid sample shards under {sample_path}")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=load_parallelism) as pool:
        tables = list(pool.map(_read_sample_table, paths))
    table = pa.concat_tables(tables)
    encoded_sources = table["source"].combine_chunks().dictionary_encode()
    source_names = tuple(encoded_sources.dictionary.to_pylist())
    source_codes = encoded_sources.indices.to_numpy(zero_copy_only=False)
    selected, quotas = stratified_indices(source_codes, source_names, target, seed)

    embeddings_column = table["embedding"].combine_chunks()
    if not pa.types.is_fixed_size_list(embeddings_column.type) or embeddings_column.type.list_size != HARRIER_DIM:
        raise ValueError(f"Unexpected centroid sample type {embeddings_column.type}")
    quantized = embeddings_column.values.to_numpy(zero_copy_only=False).reshape(-1, HARRIER_DIM)
    training_embeddings = np.ascontiguousarray(dequantize_to_fp32(quantized[selected]))
    logger.info(
        "Loaded %d x %d population and selected %d stratified training rows in %.1fs",
        len(quantized),
        HARRIER_DIM,
        len(training_embeddings),
        time.monotonic() - started,
    )
    return _TrainingData(training_embeddings, quantized, quotas)


def _train_fine_centroids(
    embeddings: np.ndarray,
    k_train: int,
    points_per_centroid: int,
    n_iter: int,
    n_redo: int,
    seed: int,
    n_threads: int,
) -> _TrainedCentroids:
    import faiss  # noqa: PLC0415

    faiss.omp_set_num_threads(n_threads)
    started = time.monotonic()
    kmeans = faiss.Kmeans(
        d=HARRIER_DIM,
        k=k_train,
        niter=n_iter,
        nredo=n_redo,
        spherical=True,
        seed=seed,
        verbose=True,
        max_points_per_centroid=points_per_centroid,
    )
    kmeans.train(embeddings)
    objective = kmeans.obj
    assert objective is not None
    return _TrainedCentroids(
        centroids=kmeans.centroids.astype(np.float32, copy=False),
        index=kmeans.index,
        seconds=time.monotonic() - started,
        objective=float(objective[-1]),
    )


def _cluster_weights(
    index: _SearchIndex,
    quantized: np.ndarray,
    k_train: int,
    assign_batch_rows: int,
) -> np.ndarray:
    weights = np.zeros(k_train, dtype=np.int64)
    for start in range(0, len(quantized), assign_batch_rows):
        batch = dequantize_to_fp32(quantized[start : start + assign_batch_rows])
        _, assignments = index.search(batch, 1)
        weights += np.bincount(assignments[:, 0], minlength=k_train)
    return weights


def _save_npy(array: np.ndarray, output_path: str, name: str) -> None:
    with open_url(prefix_join(output_path, name), "wb") as file:
        np.save(file, array)


def _write_training_artifact(
    output_path: str,
    sample_path: str,
    training: _TrainingData,
    trained: _TrainedCentroids,
    weights: np.ndarray,
    coarsening: CoarseningResult,
    config: CoarseningConfig,
    points_per_centroid: int,
    n_iter: int,
    n_redo: int,
    seed: int,
    n_threads: int,
) -> None:
    k_train = len(trained.centroids)
    _save_npy(trained.centroids, output_path, f"centroids_{k_train}.npy")
    _save_npy(coarsening.fine_to_coarse, output_path, f"lookup_{k_train}_to_{config.clusters}.npy")
    with open_url(prefix_join(output_path, "fine_cluster_weights.json"), "w") as file:
        json.dump({f"document_counts_{k_train}": weights.astype(int).tolist()}, file)
    with open_url(prefix_join(output_path, "train_stats.json"), "w") as file:
        json.dump(
            {
                "sample_path": sample_path,
                "embedding_dim": HARRIER_DIM,
                "k_train": k_train,
                "k_views": [config.clusters],
                "n_sample": len(training.embeddings),
                "population_rows": len(training.population),
                "training_points_per_centroid": points_per_centroid,
                "training_rows_by_source": training.rows_by_source,
                "n_iter": n_iter,
                "n_redo": n_redo,
                "seed": seed,
                "n_threads": n_threads,
                "train_seconds": trained.seconds,
                "final_objective": trained.objective,
                "coarsening": _coarsening_payload(coarsening, config),
            },
            file,
            indent=2,
        )


def train_centroids(
    output_path: str,
    sample_path: str,
    k_train: int,
    train_rows: int,
    points_per_centroid: int,
    n_iter: int,
    n_redo: int,
    seed: int,
    n_threads: int,
    assign_batch_rows: int,
    load_parallelism: int,
    coarsening_config: CoarseningConfig,
) -> None:
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    training = _load_training_embeddings(sample_path, train_rows, seed, load_parallelism)
    with threadpool_limits(limits=n_threads):
        trained = _train_fine_centroids(
            training.embeddings,
            k_train,
            points_per_centroid,
            n_iter,
            n_redo,
            seed,
            n_threads,
        )
        weights = _cluster_weights(trained.index, training.population, k_train, assign_batch_rows).astype(np.float64)
        if np.any(weights == 0):
            raise ValueError("K-means produced an empty fine cluster")
        coarsening = coarsen_centroids(trained.centroids, weights, coarsening_config)
    _write_training_artifact(
        output_path,
        sample_path,
        training,
        trained,
        weights,
        coarsening,
        coarsening_config,
        points_per_centroid,
        n_iter,
        n_redo,
        seed,
        n_threads,
    )
