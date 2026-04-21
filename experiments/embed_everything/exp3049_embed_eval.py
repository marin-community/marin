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

from fray.v2 import ResourceConfig
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.embed_everything.embed import LUXICAL_MODEL, embed_documents
from experiments.embed_everything.evaluate import (
    evaluate_quality_mlp,
    evaluate_quality_probe,
    evaluate_topic_clusters,
    evaluate_topic_reduced,
)
from experiments.embed_everything.oracle import OracleBackend, label_quality, label_topics
from experiments.embed_everything.sample import sample_quality_documents_nemotron, sample_topic_documents
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = LUXICAL_MODEL
N_PER_BUCKET = 200
N_PER_SOURCE = 67
N_TOPIC_CLUSTERS = 15
ORACLE_BACKEND = OracleBackend.CLAUDE
PROMPT_VERSION = "v1"

# Known relative paths for input data (relative to marin_prefix()).
# The actual GCS prefix is resolved at runtime inside each lambda.
NEMOTRON_REL_PATH = "raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
DOLMA_REL_PATH = "raw/dolma/v1.7"

# Output prefix: temp bucket with 7-day TTL.
# marin_temp_bucket resolves to gs://marin-tmp-{region}/ttl=7d/... on GCP,
# or {MARIN_PREFIX}/tmp/... locally.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="embed-everything")

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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
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
        resources=ResourceConfig.with_cpu(),
        pip_dependency_groups=["dimred"],
    ),
)

# ---------------------------------------------------------------------------
# All steps in topological order for StepRunner
# ---------------------------------------------------------------------------

ALL_STEPS = [
    sample_quality,
    sample_topic,
    oracle_quality,
    oracle_topic,
    embed_quality,
    embed_topic,
    eval_quality,
    eval_topic,
    eval_quality_mlp,
    eval_topic_reduced,
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(ALL_STEPS)
