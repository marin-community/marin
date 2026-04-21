# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: Evaluate Luxical embeddings for quality filtering and topic clustering.

Parent issue: https://github.com/marin-community/marin/issues/3049
Experiment issue: https://github.com/marin-community/marin/issues/3535

Step DAG:
    sample_quality → ┬─ oracle_quality ─┐
                     └─ embed_quality  ─┤→ eval_quality
    sample_topic   → ┬─ oracle_topic  ─┐
                     └─ embed_topic   ─┤→ eval_topic

All steps run on CPU via Iris (@remote). Output goes to temp buckets
(gs://marin-tmp-*/ttl=7d/embed-everything/).
"""

import logging
import os

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_prefix, marin_temp_bucket  # noqa: E402

from experiments.embed_everything.embed import LUXICAL_MODEL, embed_documents  # noqa: E402
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

EMBEDDING_MODEL = LUXICAL_MODEL
N_PER_BUCKET = 200
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
# Step 1: Sample documents
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
# Step 2a: Oracle labeling (depends on sampling)
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
# Step 2b: Compute embeddings (depends on sampling, parallel with oracle)
# ---------------------------------------------------------------------------

embed_quality = StepSpec(
    name="embed_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_quality],
    hash_attrs={"model": EMBEDDING_MODEL},
    fn=remote(
        lambda output_path: embed_documents(
            output_path=output_path,
            input_path=sample_quality.output_path,
            model_name=EMBEDDING_MODEL,
            input_filename="quality_samples.parquet",
            output_filename="quality_embeddings.npz",
            label_field="quality_bucket",
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["cpu", "embed"],
    ),
)

embed_topic = StepSpec(
    name="embed_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_topic],
    hash_attrs={"model": EMBEDDING_MODEL},
    fn=remote(
        lambda output_path: embed_documents(
            output_path=output_path,
            input_path=sample_topic.output_path,
            model_name=EMBEDDING_MODEL,
            input_filename="topic_samples.parquet",
            output_filename="topic_embeddings.npz",
            label_field="source_label",
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["cpu", "embed"],
    ),
)

# ---------------------------------------------------------------------------
# Step 3: Evaluation (depends on both oracle and embeddings)
# ---------------------------------------------------------------------------

eval_quality = StepSpec(
    name="eval_quality",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[embed_quality, oracle_quality],
    fn=remote(
        lambda output_path: evaluate_quality_probe(
            output_path=output_path,
            embeddings_path=embed_quality.output_path,
            oracle_path=oracle_quality.output_path,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

eval_topic = StepSpec(
    name="eval_topic",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[embed_topic, oracle_topic],
    hash_attrs={"n_clusters": N_TOPIC_CLUSTERS},
    fn=remote(
        lambda output_path: evaluate_topic_clusters(
            output_path=output_path,
            embeddings_path=embed_topic.output_path,
            oracle_path=oracle_topic.output_path,
            n_clusters=N_TOPIC_CLUSTERS,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

# ---------------------------------------------------------------------------
# Step 3b: Investigation steps (MLP probe, dimensionality reduction)
# ---------------------------------------------------------------------------

eval_quality_mlp = StepSpec(
    name="eval_quality_mlp",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[embed_quality, oracle_quality],
    hash_attrs={"v": 1},
    fn=remote(
        lambda output_path: evaluate_quality_mlp(
            output_path=output_path,
            embeddings_path=embed_quality.output_path,
            oracle_path=oracle_quality.output_path,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

eval_topic_reduced = StepSpec(
    name="eval_topic_reduced",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[embed_topic, oracle_topic],
    hash_attrs={"n_clusters": N_TOPIC_CLUSTERS, "v": 1},
    fn=remote(
        lambda output_path: evaluate_topic_reduced(
            output_path=output_path,
            embeddings_path=embed_topic.output_path,
            oracle_path=oracle_topic.output_path,
            n_clusters=N_TOPIC_CLUSTERS,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["dimred"],
    ),
)

# ---------------------------------------------------------------------------
# Supervised topic probe (logistic regression on embeddings) — apples-to-apples
# counterpart to the fasttext topic classifier.
# ---------------------------------------------------------------------------

eval_topic_supervised = StepSpec(
    name="eval_topic_supervised",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[embed_topic, oracle_topic],
    hash_attrs={"v": 1},
    fn=remote(
        lambda output_path: evaluate_topic_supervised(
            output_path=output_path,
            embeddings_path=embed_topic.output_path,
            oracle_path=oracle_topic.output_path,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

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
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
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
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
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

ALL_STEPS = [
    sample_quality,
    sample_quality_binary,
    sample_topic,
    oracle_quality,
    oracle_topic,
    embed_quality,
    embed_topic,
    eval_quality,
    eval_topic,
    eval_quality_mlp,
    eval_topic_reduced,
    eval_topic_supervised,
    fasttext_quality,
    fasttext_topic,
    eval_fasttext_quality,
    eval_fasttext_topic,
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(ALL_STEPS)
