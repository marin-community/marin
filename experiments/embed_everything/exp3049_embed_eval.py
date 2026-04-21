# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: embedding capacity ladder for quality filtering and topic clustering.

Parent issue: https://github.com/marin-community/marin/issues/3049
Experiment issue: https://github.com/marin-community/marin/issues/3535

Evaluates a ladder of sentence-transformer embedding models (Luxical, Arctic,
BGE-large) against the same oracle labels + same test split, plus fasttext
baselines from the Dolma-3 pipeline. The headline question is whether the
Luxical 192d plateau at ~0.75 Spearman is a capacity issue (breaks when we
give the probe more dims to work with) or an oracle/signal issue (stays flat
regardless of capacity).

Step DAG::

    sample_quality(binary)  ─┬─ oracle_quality ──────────┐
                             ├─ embed_quality_<model> ───┼→ eval_quality_<model>
                             └─ fasttext_quality ────────┴→ eval_fasttext_quality
    sample_topic            ─┬─ oracle_topic ────────────┐
                             ├─ embed_topic_<model> ─────┼→ eval_topic_{,supervised,reduced}_<model>
                             └─ fasttext_topic ──────────┴→ eval_fasttext_topic

``<model>`` fans out over ``EMBED_MODELS``.  Oracle + fasttext steps are
shared across the ladder (each pays once).  All steps run in europe-west4.
"""

import logging
import os
from dataclasses import dataclass, field

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_prefix, marin_temp_bucket  # noqa: E402

from experiments.embed_everything.embed import (  # noqa: E402
    ARCTIC_MODEL,
    BGE_LARGE_MODEL,
    LUXICAL_MODEL,
    embed_documents,
)
from experiments.embed_everything.evaluate import (  # noqa: E402
    evaluate_fasttext_quality,
    evaluate_fasttext_topic,
    evaluate_quality_mlp,
    evaluate_quality_probe,
    evaluate_topic_clusters,
    evaluate_topic_reduced,
    evaluate_topic_supervised,
)
from experiments.embed_everything.fasttext_baseline import (  # noqa: E402
    classify_documents_fasttext_topic,
    score_documents_fasttext_quality,
)
from experiments.embed_everything.oracle import OracleBackend, label_quality, label_topics  # noqa: E402
from experiments.embed_everything.sample import (  # noqa: E402
    sample_quality_documents_binary,
    sample_quality_documents_nemotron,
    sample_topic_documents,
)
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PER_BUCKET = 1000
N_PER_SOURCE = 67
# WebOrganizer's 24-class topic taxonomy (see experiments.embed_everything.oracle.TopicLiteral).
N_TOPIC_CLUSTERS = 24
ORACLE_BACKEND = OracleBackend.CLAUDE
# Bumped to v2 when the oracle switched to structured outputs and the topic
# taxonomy moved to the WebOrganizer 24-class set. Invalidates v1 cached labels.
PROMPT_VERSION = "v2"

# Known relative paths for input data (relative to marin_prefix()).
# The actual GCS prefix is resolved at runtime inside each lambda.
NEMOTRON_REL_PATH = "raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
DOLMA_REL_PATH = "raw/dolma/v1.7"

# Output prefix: temp bucket with 30-day TTL.
# marin_temp_bucket resolves to gs://marin-tmp-{region}/ttl=30d/... on GCP,
# or {MARIN_PREFIX}/tmp/... locally.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=30, prefix="embed-everything")


# ---------------------------------------------------------------------------
# Capacity ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedModel:
    """One rung of the embedding-capacity ladder.

    slug is used in StepSpec names and GCS output paths; keep it short and
    filesystem-safe (``[a-z0-9_]+``). ram/disk apply to the embed steps for
    this model; larger sentence-transformers (Arctic, BGE) need more headroom
    than the Luxical defaults.
    """

    slug: str
    model_name: str
    ram: str = "4g"
    disk: str = "5g"
    pip_groups: list[str] = field(default_factory=lambda: ["cpu", "embed"])


EMBED_MODELS: list[EmbedModel] = [
    EmbedModel(slug="luxical", model_name=LUXICAL_MODEL),
    EmbedModel(slug="arctic", model_name=ARCTIC_MODEL, ram="8g", disk="10g"),
    EmbedModel(slug="bge_large", model_name=BGE_LARGE_MODEL, ram="8g", disk="10g"),
]


# ---------------------------------------------------------------------------
# Step 1: Sample documents (model-independent)
# ---------------------------------------------------------------------------

sample_quality = StepSpec(
    name="sample_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_bucket": N_PER_BUCKET, "seed": 42, "v": 5},
    fn=remote(
        lambda output_path: sample_quality_documents_nemotron(
            output_path=output_path,
            nemotron_base_path=f"{marin_prefix()}/{NEMOTRON_REL_PATH}",
            n_per_bucket=N_PER_BUCKET,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

sample_quality_binary = StepSpec(
    name="sample_quality_binary",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_bucket": N_PER_BUCKET, "seed": 42, "v": 1},
    fn=remote(
        lambda output_path: sample_quality_documents_binary(
            output_path=output_path,
            dolma_base_path=f"{marin_prefix()}/{DOLMA_REL_PATH}",
            n_per_bucket=N_PER_BUCKET,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

sample_topic = StepSpec(
    name="sample_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_source": N_PER_SOURCE, "seed": 42, "v": 5},
    fn=remote(
        lambda output_path: sample_topic_documents(
            output_path=output_path,
            dolma_base_path=f"{marin_prefix()}/{DOLMA_REL_PATH}",
            n_per_source=N_PER_SOURCE,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

# ---------------------------------------------------------------------------
# Step 2a: Oracle labeling (model-independent)
# ---------------------------------------------------------------------------

oracle_quality = StepSpec(
    name="oracle_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_quality],
    hash_attrs={"backend": str(ORACLE_BACKEND), "prompt_version": PROMPT_VERSION, "v": 4},
    fn=remote(
        lambda output_path: label_quality(
            output_path=output_path,
            input_path=sample_quality.output_path,
            backend=ORACLE_BACKEND,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["oracle"],
    ),
)

oracle_topic = StepSpec(
    name="oracle_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_topic],
    hash_attrs={"backend": str(ORACLE_BACKEND), "prompt_version": PROMPT_VERSION, "v": 4},
    fn=remote(
        lambda output_path: label_topics(
            output_path=output_path,
            input_path=sample_topic.output_path,
            backend=ORACLE_BACKEND,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["oracle"],
    ),
)


# ---------------------------------------------------------------------------
# Step 2b / Step 3: Per-model embedding + evaluation fanout
# ---------------------------------------------------------------------------


def _build_model_steps(em: EmbedModel) -> list[StepSpec]:
    """Return the seven per-model StepSpecs (embed + eval) for one ladder rung."""
    embed_q = StepSpec(
        name=f"embed_quality_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[sample_quality],
        hash_attrs={"model": em.model_name},
        fn=remote(
            lambda output_path, em=em: embed_documents(
                output_path=output_path,
                input_path=sample_quality.output_path,
                model_name=em.model_name,
                input_filename="quality_samples.parquet",
                output_filename="quality_embeddings.npz",
                label_field="quality_bucket",
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram=em.ram, disk=em.disk),
            pip_dependency_groups=em.pip_groups,
        ),
    )

    embed_t = StepSpec(
        name=f"embed_topic_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[sample_topic],
        hash_attrs={"model": em.model_name},
        fn=remote(
            lambda output_path, em=em: embed_documents(
                output_path=output_path,
                input_path=sample_topic.output_path,
                model_name=em.model_name,
                input_filename="topic_samples.parquet",
                output_filename="topic_embeddings.npz",
                label_field="source_label",
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram=em.ram, disk=em.disk),
            pip_dependency_groups=em.pip_groups,
        ),
    )

    eval_q = StepSpec(
        name=f"eval_quality_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_q, oracle_quality],
        fn=remote(
            lambda output_path, embed_q=embed_q: evaluate_quality_probe(
                output_path=output_path,
                embeddings_path=embed_q.output_path,
                oracle_path=oracle_quality.output_path,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        ),
    )

    eval_q_mlp = StepSpec(
        name=f"eval_quality_mlp_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_q, oracle_quality],
        hash_attrs={"v": 1},
        fn=remote(
            lambda output_path, embed_q=embed_q: evaluate_quality_mlp(
                output_path=output_path,
                embeddings_path=embed_q.output_path,
                oracle_path=oracle_quality.output_path,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        ),
    )

    eval_t = StepSpec(
        name=f"eval_topic_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_t, oracle_topic],
        hash_attrs={"n_clusters": N_TOPIC_CLUSTERS},
        fn=remote(
            lambda output_path, embed_t=embed_t: evaluate_topic_clusters(
                output_path=output_path,
                embeddings_path=embed_t.output_path,
                oracle_path=oracle_topic.output_path,
                n_clusters=N_TOPIC_CLUSTERS,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        ),
    )

    eval_t_red = StepSpec(
        name=f"eval_topic_reduced_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_t, oracle_topic],
        hash_attrs={"n_clusters": N_TOPIC_CLUSTERS, "v": 1},
        fn=remote(
            lambda output_path, embed_t=embed_t: evaluate_topic_reduced(
                output_path=output_path,
                embeddings_path=embed_t.output_path,
                oracle_path=oracle_topic.output_path,
                n_clusters=N_TOPIC_CLUSTERS,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
            pip_dependency_groups=["dimred"],
        ),
    )

    eval_t_sup = StepSpec(
        name=f"eval_topic_supervised_{em.slug}",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=[embed_t, oracle_topic],
        hash_attrs={"v": 1},
        fn=remote(
            lambda output_path, embed_t=embed_t: evaluate_topic_supervised(
                output_path=output_path,
                embeddings_path=embed_t.output_path,
                oracle_path=oracle_topic.output_path,
            ),
            resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        ),
    )

    return [embed_q, embed_t, eval_q, eval_q_mlp, eval_t, eval_t_red, eval_t_sup]


MODEL_STEPS: list[StepSpec] = []
for _em in EMBED_MODELS:
    MODEL_STEPS.extend(_build_model_steps(_em))


# ---------------------------------------------------------------------------
# Fasttext baselines (Allen AI Dolma 3 classifiers) — Helw150's skill-issue bar.
# ---------------------------------------------------------------------------

fasttext_quality = StepSpec(
    name="fasttext_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_quality],
    hash_attrs={"model": "allenai/dolma3-fasttext-quality-classifier", "v": 1},
    fn=remote(
        lambda output_path: score_documents_fasttext_quality(
            output_path=output_path,
            input_path=sample_quality.output_path,
        ),
        # 4 GB model + fasttext internal state peaks at ~3.4 GB resident.
        # Default 4 GB RAM / 5 GB disk OOM-kills; bump both.
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram="8g", disk="20g"),
        pip_dependency_groups=["fasttext"],
    ),
)

fasttext_topic = StepSpec(
    name="fasttext_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_topic],
    hash_attrs={"model": "allenai/dolma3-fasttext-weborganizer-topic-classifier", "v": 1},
    fn=remote(
        lambda output_path: classify_documents_fasttext_topic(
            output_path=output_path,
            input_path=sample_topic.output_path,
        ),
        # 4 GB model + fasttext internal state peaks at ~3.4 GB resident.
        # Default 4 GB RAM / 5 GB disk OOM-kills; bump both.
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram="8g", disk="20g"),
        pip_dependency_groups=["fasttext"],
    ),
)

eval_fasttext_quality = StepSpec(
    name="eval_fasttext_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[fasttext_quality, oracle_quality],
    hash_attrs={"v": 1},
    fn=remote(
        lambda output_path: evaluate_fasttext_quality(
            output_path=output_path,
            fasttext_path=fasttext_quality.output_path,
            oracle_path=oracle_quality.output_path,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

eval_fasttext_topic = StepSpec(
    name="eval_fasttext_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[fasttext_topic, oracle_topic],
    hash_attrs={"v": 1},
    fn=remote(
        lambda output_path: evaluate_fasttext_topic(
            output_path=output_path,
            fasttext_path=fasttext_topic.output_path,
            oracle_path=oracle_topic.output_path,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)


# ---------------------------------------------------------------------------
# All steps in topological order for StepRunner
# ---------------------------------------------------------------------------

ALL_STEPS: list[StepSpec] = [
    sample_quality,
    sample_quality_binary,
    sample_topic,
    oracle_quality,
    oracle_topic,
    *MODEL_STEPS,
    fasttext_quality,
    fasttext_topic,
    eval_fasttext_quality,
    eval_fasttext_topic,
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(ALL_STEPS)
