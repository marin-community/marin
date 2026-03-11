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

All steps run on CPU via Iris (@remote). Intermediate data is stored in
temp buckets (gs://marin-tmp-*/ttl=7d/embed-everything/).
"""

import logging

from fray.v2 import ResourceConfig

from experiments.embed_everything.embed import LUXICAL_MODEL, embed_documents
from experiments.embed_everything.evaluate import evaluate_quality_probe, evaluate_topic_clusters
from experiments.embed_everything.oracle import OracleBackend, label_quality, label_topics
from experiments.embed_everything.sample import sample_quality_documents, sample_topic_documents
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

EXPERIMENT_PREFIX = "embed_everything/exp3049"
EMBEDDING_MODEL = LUXICAL_MODEL
N_PER_BUCKET = 25
N_PER_SOURCE = 25
N_TOPIC_CLUSTERS = 15
ORACLE_BACKEND = OracleBackend.CLAUDE
PROMPT_VERSION = "v1"

# Resolve the Nemotron CC and Dolma base paths from existing pipeline definitions.
# These are InputName objects that resolve to GCS paths at runtime.
# We import them lazily at module level so the DAG can be inspected without
# triggering heavy imports.
from experiments.pretraining_datasets.nemotron import _nemotron_cc_path  # noqa: E402
from experiments.pretraining_datasets.dolma import _DOLMA_V1_7_PATH  # noqa: E402

# For StepSpec, we need concrete string paths. These InputName objects resolve
# to strings via the executor framework, but for StepSpec we hardcode the
# known GCS paths that the executor would resolve to.
# The nemotron path is: {MARIN_PREFIX}/raw/nemotro-cc-{hash}/contrib/Nemotron/Nemotron-CC/data-jsonl/
# The dolma path is: {MARIN_PREFIX}/raw/dolma/v1.7
# We use the InputName string representations at runtime.
NEMOTRON_BASE_PATH = str(_nemotron_cc_path)
DOLMA_BASE_PATH = str(_DOLMA_V1_7_PATH)

# ---------------------------------------------------------------------------
# Step 1: Sample documents
# ---------------------------------------------------------------------------

sample_quality = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/sample_quality",
    hash_attrs={"n_per_bucket": N_PER_BUCKET, "seed": 42},
    fn=remote(
        lambda output_path: sample_quality_documents(
            output_path=output_path,
            nemotron_base_path=NEMOTRON_BASE_PATH,
            n_per_bucket=N_PER_BUCKET,
        ),
        resources=ResourceConfig.with_cpu(),
    ),
)

sample_topic = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/sample_topic",
    hash_attrs={"n_per_source": N_PER_SOURCE, "seed": 42},
    fn=remote(
        lambda output_path: sample_topic_documents(
            output_path=output_path,
            dolma_base_path=DOLMA_BASE_PATH,
            n_per_source=N_PER_SOURCE,
        ),
        resources=ResourceConfig.with_cpu(),
    ),
)

# ---------------------------------------------------------------------------
# Step 2a: Oracle labeling (depends on sampling)
# ---------------------------------------------------------------------------

oracle_quality = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/oracle_quality",
    deps=[sample_quality],
    hash_attrs={"backend": str(ORACLE_BACKEND), "prompt_version": PROMPT_VERSION},
    fn=remote(
        lambda output_path: label_quality(
            output_path=output_path,
            input_path=sample_quality.output_path,
            backend=ORACLE_BACKEND,
        ),
        resources=ResourceConfig.with_cpu(),
    ),
)

oracle_topic = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/oracle_topic",
    deps=[sample_topic],
    hash_attrs={"backend": str(ORACLE_BACKEND), "prompt_version": PROMPT_VERSION},
    fn=remote(
        lambda output_path: label_topics(
            output_path=output_path,
            input_path=sample_topic.output_path,
            backend=ORACLE_BACKEND,
        ),
        resources=ResourceConfig.with_cpu(),
    ),
)

# ---------------------------------------------------------------------------
# Step 2b: Compute embeddings (depends on sampling, parallel with oracle)
# ---------------------------------------------------------------------------

embed_quality = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/embed_quality",
    deps=[sample_quality],
    hash_attrs={"model": EMBEDDING_MODEL},
    fn=remote(
        lambda output_path: embed_documents(
            output_path=output_path,
            input_path=sample_quality.output_path,
            model_name=EMBEDDING_MODEL,
            input_filename="quality_samples.jsonl",
            output_filename="quality_embeddings.npz",
            label_field="quality_bucket",
        ),
        resources=ResourceConfig.with_cpu(),
        pip_dependency_groups=["embed"],
    ),
)

embed_topic = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/embed_topic",
    deps=[sample_topic],
    hash_attrs={"model": EMBEDDING_MODEL},
    fn=remote(
        lambda output_path: embed_documents(
            output_path=output_path,
            input_path=sample_topic.output_path,
            model_name=EMBEDDING_MODEL,
            input_filename="topic_samples.jsonl",
            output_filename="topic_embeddings.npz",
            label_field="source_label",
        ),
        resources=ResourceConfig.with_cpu(),
        pip_dependency_groups=["embed"],
    ),
)

# ---------------------------------------------------------------------------
# Step 3: Evaluation (depends on both oracle and embeddings)
# ---------------------------------------------------------------------------

eval_quality = StepSpec(
    name=f"{EXPERIMENT_PREFIX}/eval_quality",
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
    name=f"{EXPERIMENT_PREFIX}/eval_topic",
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
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(ALL_STEPS)
