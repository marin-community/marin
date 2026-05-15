# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed 10k ``nemotron_cc_v2/high_quality`` docs with Luxical-One and K-means them into 40 clusters.

Three-step DAG:

    source.normalized  ──►  sample_docs (N=10000)  ──►  embed_docs (Luxical)  ──►  cluster_docs (K=40)

Outputs land in ``gs://marin-eu-west4/tmp/ttl=7d/rav/clustering/<step>-<hash>/``.

Submitted via Iris::

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production \\
        --job-name "embed-clusters-nemotron-hq-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.embed_clusters.exp_nemotron_clusters
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
from rigging.filesystem import marin_temp_bucket  # noqa: E402

from experiments.embed_clusters.cluster import cluster_and_summarize  # noqa: E402
from experiments.embed_clusters.embed import LUXICAL_MODEL, embed_documents  # noqa: E402
from experiments.embed_clusters.sample import sample_normalized  # noqa: E402

logger = logging.getLogger(__name__)

SOURCE_NAME = "nemotron_cc_v2/high_quality"
N_DOCS = 10_000
K_CLUSTERS = 40

# Per-worker thread caps so cpu=N requests actually use N threads (sentence-
# transformers + numpy + numba all otherwise auto-detect host cores).
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

_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="rav/clustering")


def _build_steps() -> list[StepSpec]:
    source = all_sources()[SOURCE_NAME]
    normalized = source.normalized
    normalized_path = normalized.output_path

    sample_step = StepSpec(
        name=f"sample_{SOURCE_NAME.replace('/', '_')}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[normalized],
        hash_attrs={"n_docs": N_DOCS, "v": 1},
        fn=remote(
            lambda output_path: sample_normalized(
                output_path=output_path,
                normalized_path=normalized_path,
                n_docs=N_DOCS,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        ),
    )

    embed_step = StepSpec(
        name=f"embed_luxical_{SOURCE_NAME.replace('/', '_')}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[sample_step],
        hash_attrs={"model": LUXICAL_MODEL, "v": 1},
        fn=remote(
            lambda output_path: embed_documents(
                output_path=output_path,
                samples_path=sample_step.output_path,
                model_name=LUXICAL_MODEL,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=8, ram="8g"),
            env_vars=_THREAD_ENV,
            pip_dependency_groups=["embed"],
        ),
    )

    cluster_step = StepSpec(
        name=f"cluster_k{K_CLUSTERS}_{SOURCE_NAME.replace('/', '_')}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[sample_step, embed_step],
        hash_attrs={"k": K_CLUSTERS, "v": 1},
        fn=remote(
            lambda output_path: cluster_and_summarize(
                output_path=output_path,
                samples_path=sample_step.output_path,
                embeddings_path=embed_step.output_path,
                k=K_CLUSTERS,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], cpu=4, ram="8g"),
            pip_dependency_groups=["probe"],
        ),
    )

    return [*source.normalize_steps, sample_step, embed_step, cluster_step]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps())
