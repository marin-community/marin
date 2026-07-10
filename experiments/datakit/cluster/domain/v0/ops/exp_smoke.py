# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the full clustering pipeline on one capped source.

Same code paths as :mod:`exp_full_clusters` (embed -> sample -> train ->
assign -> summarize), but restricted to the first ``MAX_SHARDS`` of one
source and scaled-down K so the whole thing finishes in well under an
hour. Verifies:

- int8 quantized parquet round-trip end-to-end
- :class:`EmbeddingAttrData` / :class:`AssignmentAttrData` artifact
  persistence and consumption
- Co-partitioned (basename, row_idx) read in summarize.py
- Agglomerative-merge K=k_train -> coarser K lookup application
- Streaming top-K-by-dist heap for reps + reservoir sample for c-TF-IDF
  in a single assignments pass

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu \\
        --job-name "embed-clusters-smoke-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.domain.v0.ops.exp_smoke
"""

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.types import ResourceConfig  # noqa: E402
from marin.datakit.normalize import NormalizedData  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.artifact import read_artifact  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402
from rigging.filesystem import marin_temp_bucket  # noqa: E402

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData, assign_source  # noqa: E402
from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs  # noqa: E402
from experiments.datakit.cluster.domain.v0.summarize import summarize_at_k  # noqa: E402
from experiments.datakit.cluster.domain.v0.train import train_centroids  # noqa: E402
from experiments.datakit.embeddings.luxical.pipeline import (  # noqa: E402
    LUXICAL_REPO,
    LUXICAL_WEIGHTS_FILE,
    EmbeddingAttrData,
    embed_source,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "nemotron_cc_v2/high_quality"
MAX_SHARDS = 2

K_TRAIN = 1000
K_VIEWS: tuple[int, ...] = (40, 200)

N_PER_SOURCE_FOR_SAMPLE = 50_000
N_SAMPLE_PER_CLUSTER_AT_K_TRAIN = 20
N_SAMPLE_PER_CLUSTER_AT_K_COARSER = 100

EMBED_WINDOW = 4096  # picked by bench_batch_size.py: throughput plateau on native API
ASSIGN_WINDOW = 4096

# Per-Zephyr-worker resources for embed/assign. ZephyrContext spawns these
# from the StepSpec coordinator. The StepSpec itself (which just runs the
# coordinator code) gets coordinator_resources below.
EMBED_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="16g", regions=[DATA_REGION])
ASSIGN_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="8g", regions=[DATA_REGION])
SAMPLE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g", regions=[DATA_REGION])
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="4g", regions=[DATA_REGION])
EMBED_MAX_WORKERS = 128
ASSIGN_MAX_WORKERS = 128
SAMPLE_MAX_WORKERS = 64

_THREAD_ENV = {
    var: "8"
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
}

# Pin to eu-west4 explicitly via ``source_prefix`` so the output path doesn't drift
# with the driver's worker region (previous smoke landed in us-central1 because
# marin_temp_bucket resolves against the runtime MARIN_PREFIX, which is set per-
# worker, not from the os.environ we set at module-load time).
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="rav/clustering-full-smoke", source_prefix="gs://marin-eu-west4")


def _build_steps() -> list[StepSpec]:
    sources = all_sources()
    source = sources[SOURCE_NAME]
    normalize_step = source.normalized

    embed_step = StepSpec(
        name=f"embed/luxical/{SOURCE_NAME}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[normalize_step],
        hash_attrs={
            "luxical_repo": LUXICAL_REPO,
            "luxical_weights": LUXICAL_WEIGHTS_FILE,
            "quant_dtype": "int8",
            "quant_range": 0.6,
            "window": EMBED_WINDOW,
            "max_shards": MAX_SHARDS,
            "v": 2,
        },
        fn=remote(
            lambda output_path, np=normalize_step.output_path: embed_source(
                output_path=output_path,
                normalized=read_artifact(np, NormalizedData),
                batch_size=EMBED_WINDOW,
                max_shards=MAX_SHARDS,
                worker_resources=EMBED_WORKER_RESOURCES,
                max_workers=EMBED_MAX_WORKERS,
            ),
            resources=COORDINATOR_RESOURCES,
            env_vars=_THREAD_ENV,
            pip_dependency_groups=["datakit"],
        ),
    )

    embed_step_outputs = {SOURCE_NAME: embed_step.output_path}

    sample_step = StepSpec(
        name="cluster/sample_centroids",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_step],
        hash_attrs={"n_per_source": N_PER_SOURCE_FOR_SAMPLE, "format": "parquet", "v": 3},
        fn=remote(
            lambda output_path, eso=embed_step_outputs: sample_centroid_inputs(
                output_path=output_path,
                embeddings={n: read_artifact(p, EmbeddingAttrData) for n, p in eso.items()},
                n_per_source=N_PER_SOURCE_FOR_SAMPLE,
                worker_resources=SAMPLE_WORKER_RESOURCES,
                max_workers=SAMPLE_MAX_WORKERS,
            ),
            resources=COORDINATOR_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )

    train_step = StepSpec(
        name="cluster/train_centroids",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[sample_step],
        hash_attrs={"k_train": K_TRAIN, "k_views": list(K_VIEWS), "v": 1},
        fn=remote(
            lambda output_path: train_centroids(
                output_path=output_path,
                sample_path=sample_step.output_path,
                k_train=K_TRAIN,
                k_views=K_VIEWS,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=4, ram="8g"),
            pip_dependency_groups=["datakit"],
        ),
    )

    centroids_uri = f"{train_step.output_path}/centroids_{K_TRAIN}.npy"
    lookup_uris = {k: f"{train_step.output_path}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS}

    assign_step = StepSpec(
        name=f"cluster/assign/{SOURCE_NAME}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_step, train_step],
        hash_attrs={"k_train": K_TRAIN, "k_views": list(K_VIEWS), "window": ASSIGN_WINDOW, "v": 2},
        fn=remote(
            lambda output_path, embed_step_output=embed_step.output_path: assign_source(
                output_path=output_path,
                embedding=read_artifact(embed_step_output, EmbeddingAttrData),
                centroids_uri=centroids_uri,
                lookup_uris=lookup_uris,
                window_size=ASSIGN_WINDOW,
                worker_resources=ASSIGN_WORKER_RESOURCES,
                max_workers=ASSIGN_MAX_WORKERS,
            ),
            resources=COORDINATOR_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )

    assign_step_outputs = {SOURCE_NAME: assign_step.output_path}

    summarize_steps: list[StepSpec] = []
    for k_view in (*K_VIEWS, K_TRAIN):
        n_sample = N_SAMPLE_PER_CLUSTER_AT_K_TRAIN if k_view == K_TRAIN else N_SAMPLE_PER_CLUSTER_AT_K_COARSER
        summarize_steps.append(
            StepSpec(
                name=f"cluster/summarize_k{k_view}",
                output_path_prefix=_OUTPUT_PREFIX,
                deps=[train_step, assign_step],
                hash_attrs={"k_train": K_TRAIN, "k_view": k_view, "n_sample": n_sample, "v": 1},
                fn=remote(
                    lambda output_path, k=k_view, n=n_sample, aso=assign_step_outputs: summarize_at_k(
                        output_path=output_path,
                        k_train=K_TRAIN,
                        k_view=k,
                        assignments={n_: read_artifact(p, AssignmentAttrData) for n_, p in aso.items()},
                        n_sample_per_cluster=n,
                    ),
                    resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=4, ram="8g"),
                    pip_dependency_groups=["datakit"],
                ),
            )
        )

    return [*source.normalize_steps, embed_step, sample_step, train_step, assign_step, *summarize_steps]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps())
