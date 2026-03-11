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

from fray.v2 import ResourceConfig

from experiments.embed_everything.embed import EmbedConfig, embed_documents
from experiments.embed_everything.evaluate import (
    EvalQualityConfig,
    EvalTopicConfig,
    evaluate_quality_probe,
    evaluate_topic_clusters,
)
from experiments.embed_everything.oracle import (
    LabelQualityConfig,
    LabelTopicConfig,
    OracleBackend,
    label_quality,
    label_topics,
)
from experiments.embed_everything.sample import (
    SampleQualityConfig,
    SampleTopicConfig,
    sample_quality_documents,
    sample_topic_documents,
)
from experiments.pretraining_datasets.dolma import _DOLMA_V1_7_PATH
from experiments.pretraining_datasets.nemotron import _nemotron_cc_path
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned, executor_main
from marin.execution.remote import remote

EXPERIMENT_PREFIX = "embed_everything/exp3049"
EMBEDDING_MODEL = "DatologyAI/luxical-one"

# ---------------------------------------------------------------------------
# Step 1: Sample documents
# ---------------------------------------------------------------------------

sample_quality = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/sample_quality",
    description="Sample ~25 docs per Nemotron CC quality bucket",
    fn=remote(sample_quality_documents, resources=ResourceConfig.with_cpu()),
    config=SampleQualityConfig(
        nemotron_base_path=_nemotron_cc_path,
        output_path=this_output_path(),
        n_per_bucket=versioned(25),
    ),
)

sample_topic = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/sample_topic",
    description="Sample ~25 docs per Dolma source split",
    fn=remote(sample_topic_documents, resources=ResourceConfig.with_cpu()),
    config=SampleTopicConfig(
        dolma_base_path=_DOLMA_V1_7_PATH,
        output_path=this_output_path(),
        n_per_source=versioned(25),
    ),
)

# ---------------------------------------------------------------------------
# Step 2a: Oracle labeling (depends on sampling)
# ---------------------------------------------------------------------------

oracle_quality = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/oracle_quality",
    description="Label quality with Claude (0-5 rubric)",
    fn=remote(label_quality, resources=ResourceConfig.with_cpu()),
    config=LabelQualityConfig(
        input_path=output_path_of(sample_quality),
        output_path=this_output_path(),
        backend=versioned(OracleBackend.CLAUDE),
        prompt_version=versioned("v1"),
    ),
)

oracle_topic = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/oracle_topic",
    description="Label topics with Claude from fixed taxonomy",
    fn=remote(label_topics, resources=ResourceConfig.with_cpu()),
    config=LabelTopicConfig(
        input_path=output_path_of(sample_topic),
        output_path=this_output_path(),
        backend=versioned(OracleBackend.CLAUDE),
        prompt_version=versioned("v1"),
    ),
)

# ---------------------------------------------------------------------------
# Step 2b: Compute embeddings (depends on sampling, parallel with oracle)
# ---------------------------------------------------------------------------

embed_quality = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/embed_quality",
    description="Compute Luxical embeddings for quality docs",
    fn=remote(embed_documents, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["embed"]),
    config=EmbedConfig(
        input_path=output_path_of(sample_quality),
        output_path=this_output_path(),
        model_name=versioned(EMBEDDING_MODEL),
        input_filename="quality_samples.jsonl",
        output_filename="quality_embeddings.npz",
        label_field="quality_bucket",
    ),
)

embed_topic = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/embed_topic",
    description="Compute Luxical embeddings for topic docs",
    fn=remote(embed_documents, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["embed"]),
    config=EmbedConfig(
        input_path=output_path_of(sample_topic),
        output_path=this_output_path(),
        model_name=versioned(EMBEDDING_MODEL),
        input_filename="topic_samples.jsonl",
        output_filename="topic_embeddings.npz",
        label_field="source_label",
    ),
)

# ---------------------------------------------------------------------------
# Step 3: Evaluation (depends on both oracle and embeddings)
# ---------------------------------------------------------------------------

eval_quality = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/eval_quality",
    description="Linear probe: embeddings vs oracle quality scores",
    fn=remote(evaluate_quality_probe, resources=ResourceConfig.with_cpu()),
    config=EvalQualityConfig(
        embeddings_path=output_path_of(embed_quality),
        oracle_path=output_path_of(oracle_quality),
        output_path=this_output_path(),
    ),
)

eval_topic = ExecutorStep(
    name=f"{EXPERIMENT_PREFIX}/eval_topic",
    description="K-Means clustering vs oracle topic labels",
    fn=remote(evaluate_topic_clusters, resources=ResourceConfig.with_cpu()),
    config=EvalTopicConfig(
        embeddings_path=output_path_of(embed_topic),
        oracle_path=output_path_of(oracle_topic),
        output_path=this_output_path(),
        n_clusters=versioned(15),
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=[
            sample_quality,
            sample_topic,
            oracle_quality,
            oracle_topic,
            embed_quality,
            embed_topic,
            eval_quality,
            eval_topic,
        ],
        description="Evaluate Luxical embeddings for quality filtering and topic clustering (issue #3535)",
    )
