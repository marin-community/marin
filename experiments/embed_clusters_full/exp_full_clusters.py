# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DAG: embed all datakit sources with Luxical-One, cluster at K=5000, views at K=1000 + K=40.

Shape::

    for src in all_sources():
        embed_<src>     (writes EmbeddingAttrData co-partitioned with source)
            |
            +-> sample_centroids -> train_centroids
                                          |
            assign_<src>  <----------------+   (writes AssignmentAttrData
                                                co-partitioned with embedding)
                |
                +-> summarize_at_5000, summarize_at_1000, summarize_at_40

Every per-shard output mirrors its source shard's basename, so
``NormalizedData -> EmbeddingAttrData -> AssignmentAttrData`` is a chain
of co-partitioned datakit attribute datasets. Downstream joins are
(basename, row_idx) lookups instead of id-keyed hash joins.

Embeddings are stored as ``FixedSizeList<int8, 192>`` with
``quantization_scale = 0.6 / 127`` recorded on the EmbeddingAttrData
artifact. Consumers dequantize on read via
``dequantize_to_fp32(int8_arr, scale)``.

Submit::

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production \\
        --job-name "embed-clusters-full-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.embed_clusters_full.exp_full_clusters
"""

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray import ResourceConfig  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

from experiments.embed_clusters_full.assign import assign_source  # noqa: E402
from experiments.embed_clusters_full.embed_source import LUXICAL_REPO, LUXICAL_WEIGHTS_FILE, embed_source  # noqa: E402
from experiments.embed_clusters_full.sample import sample_centroid_inputs  # noqa: E402
from experiments.embed_clusters_full.summarize import summarize_at_k  # noqa: E402
from experiments.embed_clusters_full.train import train_centroids  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline knobs
# ---------------------------------------------------------------------------

K_TRAIN = 5000
K_VIEWS: tuple[int, ...] = (40, 1000)

N_PER_SOURCE_FOR_SAMPLE = 100_000  # ~100 sources x 100k = ~10M-row centroid sample
N_SAMPLE_PER_CLUSTER_AT_K_TRAIN = 200
N_SAMPLE_PER_CLUSTER_AT_K_COARSER = 2_000

EMBED_WINDOW = 4096  # bench plateau on native Luxical API at Iris cpu=8
ASSIGN_WINDOW = 4096

# Zephyr worker resources (one worker per source shard, many in parallel).
# The StepSpec itself runs only the coordinator (build dataset, submit, wait).
EMBED_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="16g", regions=[DATA_REGION])
ASSIGN_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="8g", regions=[DATA_REGION])
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="4g", regions=[DATA_REGION])
EMBED_MAX_WORKERS_PER_SOURCE = 128  # cap per-source worker fan-out; tune to cluster headroom
ASSIGN_MAX_WORKERS_PER_SOURCE = 128

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

# Hard-pinned to eu-west4 (don't rely on the driver's MARIN_PREFIX). Layout:
#   datakit/embed/luxical/<source>_<hash>/   <- one EmbeddingAttrData per source
#   datakit/cluster/sample_centroids_<hash>/
#   datakit/cluster/train_centroids_<hash>/
#   datakit/cluster/assign/<source>_<hash>/  <- one AssignmentAttrData per source
#   datakit/cluster/summarize_k{N}_<hash>/cluster_stats_{N}.json
# Embed gets its own top-level since embeddings are reusable beyond clustering
# (decon, ANN, downstream attribute joins, etc.); the cluster-specific steps
# live together under cluster/.
_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit"


def _build_steps() -> list[StepSpec]:
    sources = all_sources()

    # --- Per-source embed steps ---------------------------------------------
    embed_steps: dict[str, StepSpec] = {}
    for source_name, source in sources.items():
        normalized = source.normalized
        normalized_path = normalized.output_path
        embed_steps[source_name] = StepSpec(
            name=f"embed/luxical/{source_name}",
            output_path_prefix=_OUTPUT_PREFIX,
            deps=[normalized],
            hash_attrs={
                "luxical_repo": LUXICAL_REPO,
                "luxical_weights": LUXICAL_WEIGHTS_FILE,
                "quant_dtype": "int8",
                "quant_range": 0.6,
                "window": EMBED_WINDOW,
                "v": 2,
            },
            fn=remote(
                lambda output_path, normalized_path=normalized_path: embed_source(
                    output_path=output_path,
                    normalized_path=normalized_path,
                    window_size=EMBED_WINDOW,
                    worker_resources=EMBED_WORKER_RESOURCES,
                    max_workers=EMBED_MAX_WORKERS_PER_SOURCE,
                ),
                # Just the coordinator: builds the dataset, submits to Zephyr,
                # waits, writes the artifact. Workers come from ZephyrContext.
                resources=COORDINATOR_RESOURCES,
                env_vars=_THREAD_ENV,
                pip_dependency_groups=["embed"],
            ),
        )

    embed_step_outputs = {name: step.output_path for name, step in embed_steps.items()}

    # --- Sample step (depends on all embeds) --------------------------------
    sample_step = StepSpec(
        name="cluster/sample_centroids",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=list(embed_steps.values()),
        hash_attrs={"n_per_source": N_PER_SOURCE_FOR_SAMPLE, "v": 1},
        fn=remote(
            lambda output_path: sample_centroid_inputs(
                output_path=output_path,
                embed_step_outputs=embed_step_outputs,
                n_per_source=N_PER_SOURCE_FOR_SAMPLE,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=4, ram="32g"),
        ),
    )

    # --- Train step ---------------------------------------------------------
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
            # FAISS K=5000 on 10M x 192 wants every core it can get.
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=32, ram="64g"),
            pip_dependency_groups=["cluster"],
        ),
    )

    centroids_uri = f"{train_step.output_path}/centroids_{K_TRAIN}.npy"
    lookup_uris = {k: f"{train_step.output_path}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS}

    # --- Per-source assign steps -------------------------------------------
    assign_steps: dict[str, StepSpec] = {}
    for source_name, embed_step in embed_steps.items():
        assign_steps[source_name] = StepSpec(
            name=f"cluster/assign/{source_name}",
            output_path_prefix=_OUTPUT_PREFIX,
            deps=[embed_step, train_step],
            hash_attrs={"k_train": K_TRAIN, "k_views": list(K_VIEWS), "window": ASSIGN_WINDOW, "v": 2},
            fn=remote(
                lambda output_path, embed_step_output=embed_step.output_path: assign_source(
                    output_path=output_path,
                    embedding_step_output=embed_step_output,
                    centroids_uri=centroids_uri,
                    lookup_uris=lookup_uris,
                    window_size=ASSIGN_WINDOW,
                    worker_resources=ASSIGN_WORKER_RESOURCES,
                    max_workers=ASSIGN_MAX_WORKERS_PER_SOURCE,
                ),
                resources=COORDINATOR_RESOURCES,
                pip_dependency_groups=["cluster"],
            ),
        )

    assign_step_outputs = {name: step.output_path for name, step in assign_steps.items()}

    # --- Summarize: one StepSpec per K view --------------------------------
    summarize_steps: list[StepSpec] = []
    for k_view in (*K_VIEWS, K_TRAIN):
        n_sample = N_SAMPLE_PER_CLUSTER_AT_K_TRAIN if k_view == K_TRAIN else N_SAMPLE_PER_CLUSTER_AT_K_COARSER
        summarize_steps.append(
            StepSpec(
                name=f"cluster/summarize_k{k_view}",
                output_path_prefix=_OUTPUT_PREFIX,
                deps=[train_step, *assign_steps.values()],
                hash_attrs={"k_train": K_TRAIN, "k_view": k_view, "n_sample": n_sample, "v": 1},
                fn=remote(
                    lambda output_path, k=k_view, n=n_sample: summarize_at_k(
                        output_path=output_path,
                        k_train=K_TRAIN,
                        k_view=k,
                        assignment_step_outputs=assign_step_outputs,
                        n_sample_per_cluster=n,
                    ),
                    resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=8, ram="32g"),
                    pip_dependency_groups=["probe"],
                ),
            )
        )

    # Include every source's normalize chain so StepRunner can re-materialize
    # anything not yet cached. Most will already exist in cache.
    upstream_steps: list[StepSpec] = []
    for src in sources.values():
        upstream_steps.extend(src.normalize_steps)

    return [
        *upstream_steps,
        *embed_steps.values(),
        sample_step,
        train_step,
        *assign_steps.values(),
        *summarize_steps,
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps(), max_concurrent=32)
