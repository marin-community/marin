# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Harrier domain clusters on a proportional sample of every source."""

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.source_key import datakit_source_path
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.domain.v1.coarsen import CoarseningConfig, CoarseningResult, coarsen_centroids
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM, dequantize_to_fp32
from experiments.datakit.reference_pipeline import select_sources

logger = logging.getLogger(__name__)

TARGET_ROWS = 10_000_000
SMALL_SOURCE_MAX_ROWS = 1_000
SMALL_SOURCE_QUOTA = 1
K_TRAIN = 5_000
K_COARSE = 40
TRAIN_POINTS_PER_CENTROID = 512
TRAIN_ROWS = K_TRAIN * TRAIN_POINTS_PER_CENTROID
N_ITER = 20
N_REDO = 3
SEED = 42
N_THREADS = 96
PARALLEL_SOURCES = 16
MAX_WORKERS = 64
LOAD_PARALLELISM = 64
ASSIGN_BATCH_ROWS = 65_536
EMBEDDING_ROOT = datakit_source_path("datakit/embed/harrier")

COARSENING_CONFIG = CoarseningConfig(
    clusters=K_COARSE,
    minimum_fraction=0.01,
    seeds=(42, 43, 44),
    split_restarts=10,
    split_iterations=20,
)

DRIVER_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g", preemptible=False)
SAMPLE_WORKER_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g")
TRAIN_RESOURCES = ResourceConfig.with_cpu(cpu=N_THREADS, ram="256g", disk="32g", preemptible=False)

_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), HARRIER_DIM)),
    ]
)


@dataclass(frozen=True)
class SourceSample:
    source: str
    paths: tuple[str, ...]
    rows: tuple[int, ...]
    quota: int


def proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    if target < len(counts) * SMALL_SOURCE_QUOTA:
        raise ValueError("target is too small for the one-document source floor")
    if target > sum(counts.values()):
        raise ValueError("target exceeds the available documents")

    small = {source for source, rows in counts.items() if rows <= SMALL_SOURCE_MAX_ROWS}
    quotas = {source: SMALL_SOURCE_QUOTA for source in small}
    large = {source: rows for source, rows in counts.items() if source not in small}
    remaining = target - sum(quotas.values())
    large_rows = sum(large.values())
    numerators = {source: remaining * rows for source, rows in large.items()}
    quotas.update({source: numerator // large_rows for source, numerator in numerators.items()})
    leftover = target - sum(quotas.values())
    for source in sorted(large, key=lambda name: (-(numerators[name] % large_rows), name))[:leftover]:
        quotas[source] += 1
    if any(quota > counts[source] for source, quota in quotas.items()):
        raise ValueError("a source quota exceeds its document count")
    return quotas


def training_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    if target < len(counts):
        raise ValueError("target is too small for the one-document source floor")
    if target > sum(counts.values()):
        raise ValueError("target exceeds the available documents")

    quotas = {source: 1 for source in counts}
    remaining = target - len(counts)
    if remaining == 0:
        return quotas
    capacities = {source: rows - 1 for source, rows in counts.items()}
    total_capacity = sum(capacities.values())
    numerators = {source: remaining * capacity for source, capacity in capacities.items()}
    quotas = {source: quotas[source] + numerator // total_capacity for source, numerator in numerators.items()}
    leftover = target - sum(quotas.values())
    for source in sorted(counts, key=lambda name: (-(numerators[name] % total_capacity), name))[:leftover]:
        quotas[source] += 1
    return quotas


def _stratified_indices(
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


def _part_quotas(rows: tuple[int, ...], target: int) -> tuple[int, ...]:
    total = sum(rows)
    numerators = [target * count for count in rows]
    quotas = [numerator // total for numerator in numerators]
    leftover = target - sum(quotas)
    order = sorted(range(len(rows)), key=lambda index: (-(numerators[index] % total), index))
    for index in order[:leftover]:
        quotas[index] += 1
    return tuple(quotas)


def _row_count(path: str) -> int:
    with StoragePath(path).open("rb") as file:
        return pq.ParquetFile(file).metadata.num_rows


def _source_samples(embedding_paths: dict[str, str]) -> tuple[SourceSample, ...]:
    source_paths = {
        source: tuple(sorted(str(shard) for shard in StoragePath(f"{path}/*.parquet").glob()))
        for source, path in sorted(embedding_paths.items())
    }
    if any(not paths for paths in source_paths.values()):
        missing = [source for source, paths in source_paths.items() if not paths]
        raise ValueError(f"sources have no embedding shards: {missing}")

    all_paths = [path for paths in source_paths.values() for path in paths]
    with ThreadPoolExecutor(max_workers=LOAD_PARALLELISM) as pool:
        all_rows = list(pool.map(_row_count, all_paths))
    path_rows = dict(zip(all_paths, all_rows, strict=True))
    source_rows = {source: sum(path_rows[path] for path in paths) for source, paths in source_paths.items()}
    empty_sources = [source for source, rows in source_rows.items() if rows == 0]
    if empty_sources:
        logger.warning("Skipping %d zero-row sources: %s", len(empty_sources), empty_sources)
        source_paths = {source: paths for source, paths in source_paths.items() if source_rows[source]}
        source_rows = {source: rows for source, rows in source_rows.items() if rows}
    quotas = proportional_quotas(source_rows, TARGET_ROWS)
    return tuple(
        SourceSample(
            source=source,
            paths=paths,
            rows=tuple(path_rows[path] for path in paths),
            quota=quotas[source],
        )
        for source, paths in sorted(source_paths.items())
    )


def _completed_embedding_paths(source_names: tuple[str, ...]) -> dict[str, str]:
    records = sorted(str(path) for path in StoragePath(f"{EMBEDDING_ROOT}/**/.artifact.json").glob())
    candidates: dict[str, list[str]] = {source: [] for source in source_names}
    source_prefixes = sorted(source_names, key=len, reverse=True)
    for record in records:
        output_path = record.removesuffix("/.artifact.json")
        relative = output_path.removeprefix(f"{EMBEDDING_ROOT}/")
        for source in source_prefixes:
            if relative.startswith(f"{source}_"):
                candidates[source].append(output_path)
                break

    resolved = {}
    for source, paths in candidates.items():
        if not paths:
            raise ValueError(f"no completed Harrier artifact directory for {source}")
        if len(paths) != 1:
            raise ValueError(f"expected one Harrier artifact directory for {source}, found {paths}")
        resolved[source] = paths[0]
    if not resolved:
        raise ValueError(f"no completed Harrier artifacts under {EMBEDDING_ROOT}")
    return resolved


def _sample_shard(
    batches: Iterator[list[dict[str, Any]]],
    _shard: ShardInfo,
    *,
    source: str,
    path: str,
    rows: int,
    quota: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    digest = hashlib.sha256(f"{seed}:{source}:{path}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = np.sort(rng.choice(rows, size=quota, replace=False))
    offset = 0
    written = 0
    for batch in batches:
        begin = int(np.searchsorted(selected, offset, side="left"))
        end = int(np.searchsorted(selected, offset + len(batch), side="left"))
        for index in selected[begin:end] - offset:
            yield {"source": source, "embedding": batch[int(index)]["embedding"]}
            written += 1
        offset += len(batch)
    if offset != rows or written != quota:
        raise ValueError(f"sampled {written}/{quota} rows after reading {offset}/{rows} from {path}")
    counters.pipeline.update_counter("harrier_cluster/sample_rows", written)


def _sample_source(sample: SourceSample, output_path: str) -> None:
    quotas = _part_quotas(sample.rows, sample.quota)
    selected = [
        (path, rows, quota) for path, rows, quota in zip(sample.paths, sample.rows, quotas, strict=True) if quota
    ]
    source_dir = f"{output_path.rstrip('/')}/{sample.source.replace('/', '-')}"
    output_paths = tuple(f"{source_dir}/sample-{index:06d}.parquet" for index in range(len(selected)))
    dataset = (
        Dataset.from_list([InputFileSpec(path=path, columns=["embedding"]) for path, _, _ in selected])
        .flat_map(load_file)
        .window(4_096)
        .map_shard(
            lambda batches, shard, parts=tuple(selected), source=sample.source: _sample_shard(
                batches,
                shard,
                source=source,
                path=parts[shard.shard_idx][0],
                rows=parts[shard.shard_idx][1],
                quota=parts[shard.shard_idx][2],
                seed=SEED,
            )
        )
        .write_parquet(lambda shard_idx, _total: output_paths[shard_idx], schema=_SAMPLE_SCHEMA, skip_existing=True)
    )
    ZephyrContext(
        resources=SAMPLE_WORKER_RESOURCES,
        coordinator_resources=ResourceConfig.with_cpu(cpu=1, ram="8g", preemptible=False),
        max_workers=min(MAX_WORKERS, len(selected)),
        name=f"sample-harrier-{hashlib.sha256(sample.source.encode()).hexdigest()[:12]}",
        stage_runner_factory=InlineRunner,
    ).execute(dataset, verbose=True)


def sample_embeddings(output_path: str, source_names: tuple[str, ...]) -> None:
    embedding_paths = _completed_embedding_paths(source_names)
    samples = _source_samples(embedding_paths)
    with ThreadPoolExecutor(max_workers=PARALLEL_SOURCES) as pool:
        futures = {pool.submit(_sample_source, sample, output_path): sample.source for sample in samples}
        for future in as_completed(futures):
            future.result()
            logger.info("Sampled %s", futures[future])

    source_rows = {sample.source: sum(sample.rows) for sample in samples}
    source_quotas = {sample.source: sample.quota for sample in samples}
    with open_url(f"{output_path.rstrip('/')}/sample_stats.json", "w") as file:
        json.dump(
            {
                "target_rows": TARGET_ROWS,
                "source_count": len(samples),
                "source_rows": source_rows,
                "source_quotas": source_quotas,
                "embedding_paths": embedding_paths,
                "small_source_max_rows": SMALL_SOURCE_MAX_ROWS,
                "small_source_quota": SMALL_SOURCE_QUOTA,
                "seed": SEED,
            },
            file,
            indent=2,
            sort_keys=True,
        )


def _coarsening_payload(result: CoarseningResult) -> dict[str, Any]:
    def stats(value: Any) -> dict[str, Any]:
        return {
            "loss": value.loss,
            "weighted_mean_cosine_distance": value.weighted_mean_cosine_distance,
            "minimum_weight_fraction": value.minimum_weight_fraction,
            "maximum_weight_fraction": value.maximum_weight_fraction,
            "weight_fractions": value.weight_fractions.tolist(),
            "fine_centroid_counts": value.fine_centroid_counts.tolist(),
        }

    return {
        "method": "weighted_divisive_spherical_kmeans_then_single_centroid_hill_climb",
        "minimum_fraction": COARSENING_CONFIG.minimum_fraction,
        "seeds": list(COARSENING_CONFIG.seeds),
        "split_restarts": COARSENING_CONFIG.split_restarts,
        "split_iterations": COARSENING_CONFIG.split_iterations,
        "selected_seed": result.selected_seed,
        "divisive_runs": [{"seed": run.seed, "stats": stats(run.stats)} for run in result.divisive_runs],
        "initial": stats(result.initial),
        "final": stats(result.final),
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
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    paths = sorted(str(path) for path in StoragePath(f"{sample_path.rstrip('/')}/**/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No centroid sample shards under {sample_path}")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=LOAD_PARALLELISM) as pool:
        tables = list(pool.map(_read_sample_table, paths))
    table = pa.concat_tables(tables)
    encoded_sources = table["source"].combine_chunks().dictionary_encode()
    source_names = tuple(encoded_sources.dictionary.to_pylist())
    source_codes = encoded_sources.indices.to_numpy(zero_copy_only=False)
    selected, quotas = _stratified_indices(source_codes, source_names, target, seed)

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
    return training_embeddings, quantized, quotas


def _cluster_weights(index: Any, quantized: np.ndarray) -> np.ndarray:
    weights = np.zeros(K_TRAIN, dtype=np.int64)
    for start in range(0, len(quantized), ASSIGN_BATCH_ROWS):
        batch = dequantize_to_fp32(quantized[start : start + ASSIGN_BATCH_ROWS])
        _, assignments = index.search(batch, 1)
        weights += np.bincount(assignments[:, 0], minlength=K_TRAIN)
    return weights


def _save_npy(array: np.ndarray, output_path: str, name: str) -> None:
    with open_url(os.path.join(output_path, name), "wb") as file:
        np.save(file, array)


def train_centroids(output_path: str, sample_path: str) -> None:
    import faiss  # noqa: PLC0415
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    faiss.omp_set_num_threads(N_THREADS)
    embeddings, population, training_rows_by_source = _load_training_embeddings(sample_path, TRAIN_ROWS, SEED)
    with threadpool_limits(limits=N_THREADS):
        started = time.monotonic()
        kmeans = faiss.Kmeans(
            d=HARRIER_DIM,
            k=K_TRAIN,
            niter=N_ITER,
            nredo=N_REDO,
            spherical=True,
            seed=SEED,
            verbose=True,
            max_points_per_centroid=TRAIN_POINTS_PER_CENTROID,
        )
        kmeans.train(embeddings)
        centroids = kmeans.centroids.astype(np.float32, copy=False)
        train_seconds = time.monotonic() - started
        weights = _cluster_weights(kmeans.index, population).astype(np.float64)
        if np.any(weights == 0):
            raise ValueError("K-means produced an empty fine cluster")
        coarsening = coarsen_centroids(
            centroids,
            weights,
            COARSENING_CONFIG,
        )

    _save_npy(centroids, output_path, f"centroids_{K_TRAIN}.npy")
    _save_npy(coarsening.fine_to_coarse, output_path, f"lookup_{K_TRAIN}_to_{K_COARSE}.npy")
    with open_url(os.path.join(output_path, "fine_cluster_weights.json"), "w") as file:
        json.dump({"document_counts_5000": weights.astype(int).tolist()}, file)
    with open_url(os.path.join(output_path, "train_stats.json"), "w") as file:
        json.dump(
            {
                "sample_path": sample_path,
                "embedding_dim": HARRIER_DIM,
                "k_train": K_TRAIN,
                "k_views": [K_COARSE],
                "n_sample": len(embeddings),
                "population_rows": len(population),
                "training_points_per_centroid": TRAIN_POINTS_PER_CENTROID,
                "training_rows_by_source": training_rows_by_source,
                "n_iter": N_ITER,
                "n_redo": N_REDO,
                "seed": SEED,
                "n_threads": N_THREADS,
                "train_seconds": train_seconds,
                "final_objective": float(kmeans.obj[-1]),
                "coarsening": _coarsening_payload(coarsening),
            },
            file,
            indent=2,
        )


def build_steps() -> tuple[StepSpec, StepSpec]:
    source_names = tuple(sorted(select_sources()))
    sample_step = StepSpec(
        name="datakit/cluster/domain/v1/harrier-all-sources-10m/sample",
        deps=[],
        hash_attrs={
            "embedding_root": EMBEDDING_ROOT,
            "source_names": source_names,
            "target_rows": TARGET_ROWS,
            "small_source_max_rows": SMALL_SOURCE_MAX_ROWS,
            "small_source_quota": SMALL_SOURCE_QUOTA,
            "seed": SEED,
            "v": 4,
        },
        fn=remote(
            lambda output_path, sources=source_names: sample_embeddings(output_path, sources),
            resources=DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    train_step = StepSpec(
        name="datakit/cluster/domain/v1/harrier-all-sources-10m/train",
        deps=[sample_step],
        hash_attrs={
            "k_train": K_TRAIN,
            "k_coarse": K_COARSE,
            "train_rows": TRAIN_ROWS,
            "train_points_per_centroid": TRAIN_POINTS_PER_CENTROID,
            "training_sampling": "source-proportional-one-document-floor",
            "n_iter": N_ITER,
            "n_redo": N_REDO,
            "seed": SEED,
            "n_threads": N_THREADS,
            "coarsening": "divisive-k40-min1pct-seeds42-44-hill-climb",
            "v": 2,
        },
        fn=remote(
            lambda output_path, sample_path=sample_step.output_path: train_centroids(output_path, sample_path),
            resources=TRAIN_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    return sample_step, train_step


def main() -> None:
    configure_logging(logging.INFO)
    _, train_step = build_steps()
    logger.info("Final artifact: %s", train_step.output_path)
    StepRunner().run([train_step])


if __name__ == "__main__":
    main()
