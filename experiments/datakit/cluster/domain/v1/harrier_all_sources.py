# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Harrier domain clusters on a proportional sample of every source."""

import logging

from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v1.coarsen import CoarseningConfig
from experiments.datakit.cluster.domain.v1.sample import sample_centroid_inputs
from experiments.datakit.cluster.domain.v1.train import train_centroids
from experiments.datakit.embeddings.harrier.run import build_steps as build_harrier_embeddings

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
PIPELINE_VERSION = "2026.08.13"

COARSENING_CONFIG = CoarseningConfig(
    clusters=K_COARSE,
    minimum_fraction=0.01,
    seeds=(42, 43, 44),
    split_restarts=10,
    split_iterations=20,
)

DRIVER_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g", preemptible=False)
SAMPLE_WORKER_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g")
SAMPLE_COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="8g", preemptible=False)
TRAIN_RESOURCES = ResourceConfig.with_cpu(cpu=N_THREADS, ram="256g", disk="32g", preemptible=False)


def _coarsening_cache_key(config: CoarseningConfig) -> str:
    if config.seeds == tuple(range(config.seeds[0], config.seeds[-1] + 1)):
        seeds = f"{config.seeds[0]}-{config.seeds[-1]}"
    else:
        seeds = "-".join(str(seed) for seed in config.seeds)
    minimum_percent = config.minimum_fraction * 100
    return f"divisive-k{config.clusters}-min{minimum_percent:g}pct-seeds{seeds}-hill-climb"


COARSENING_CACHE_KEY = _coarsening_cache_key(COARSENING_CONFIG)


def build_steps() -> tuple[StepSpec, StepSpec]:
    embedding_steps = build_harrier_embeddings("unused")
    embedding_paths = {step.name.removeprefix("datakit/embed/harrier/"): step.output_path for step in embedding_steps}
    sample_step = StepSpec(
        name="datakit/cluster/domain/v1/harrier-all-sources-10m/sample",
        deps=embedding_steps,
        hash_attrs={
            "target_rows": TARGET_ROWS,
            "small_source_max_rows": SMALL_SOURCE_MAX_ROWS,
            "small_source_quota": SMALL_SOURCE_QUOTA,
            "seed": SEED,
            "version": PIPELINE_VERSION,
        },
        fn=remote(
            lambda output_path, paths=embedding_paths: sample_centroid_inputs(
                output_path=output_path,
                embedding_paths=paths,
                target_rows=TARGET_ROWS,
                small_source_max_rows=SMALL_SOURCE_MAX_ROWS,
                small_source_quota=SMALL_SOURCE_QUOTA,
                seed=SEED,
                worker_resources=SAMPLE_WORKER_RESOURCES,
                coordinator_resources=SAMPLE_COORDINATOR_RESOURCES,
                max_workers=MAX_WORKERS,
                parallel_sources=PARALLEL_SOURCES,
                load_parallelism=LOAD_PARALLELISM,
            ),
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
            "coarsening": COARSENING_CACHE_KEY,
            "version": PIPELINE_VERSION,
        },
        fn=remote(
            lambda output_path, sample_path=sample_step.output_path: train_centroids(
                output_path=output_path,
                sample_path=sample_path,
                k_train=K_TRAIN,
                train_rows=TRAIN_ROWS,
                points_per_centroid=TRAIN_POINTS_PER_CENTROID,
                n_iter=N_ITER,
                n_redo=N_REDO,
                seed=SEED,
                n_threads=N_THREADS,
                assign_batch_rows=ASSIGN_BATCH_ROWS,
                load_parallelism=LOAD_PARALLELISM,
                coarsening_config=COARSENING_CONFIG,
            ),
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
