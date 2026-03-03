# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test Luxical as a General Tool for Data Integration Pipelines (issue #3049).

This experiment evaluates whether Luxical embeddings encode sufficient signal for:
1. Quality classification — linear probe on embeddings vs Nemotron quality labels
2. Topic clustering — K-Means on embeddings vs Dolma source labels

The DAG:
    sample_quality → ┬─ oracle_quality ─┐
                     └─ embed_quality  ─┤→ eval_quality
    sample_topic   → ┬─ oracle_topic  ─┐
                     └─ embed_topic   ─┤→ eval_topic
"""

from experiments.embed_everything.embed import embed_documents
from experiments.embed_everything.evaluate import evaluate_quality_probe, evaluate_topic_clusters
from experiments.embed_everything.oracle import label_quality, label_topics
from experiments.embed_everything.sample import sample_documents
from experiments.pretraining_datasets.dolma import DOLMA_DATASETS, _DOLMA_V1_7_PATH
from experiments.pretraining_datasets.nemotron import _nemotron_cc_path
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

# -- Config -------------------------------------------------------------------

EMBEDDING_MODEL = "DatologyAI/luxical-one"
N_PER_SOURCE = 25
SEED = 42
ORACLE_BACKEND = "claude"

# -- Nemotron quality bucket paths (actual only, skip synthetic) ---------------
# Maps quality label → glob pattern for JSONL files
NEMOTRON_QUALITY_BUCKETS = {
    "high": "quality=high/kind=actual/**/*.jsonl.gz",
    "medium_high": "quality=medium-high/**/*.jsonl.gz",
    "medium": "quality=medium/**/*.jsonl.gz",
    "medium_low": "quality=medium-low/**/*.jsonl.gz",
    "low": "quality=low/kind=actual/**/*.jsonl.gz",
}

# -- Dolma source paths -------------------------------------------------------
DOLMA_SOURCE_SPLITS = list(DOLMA_DATASETS.keys())


def _nemotron_quality_paths() -> dict[str, str]:
    """Build {label: glob_pattern} for Nemotron quality buckets."""
    base = _nemotron_cc_path
    return {label: f"{base}/{pattern}" for label, pattern in NEMOTRON_QUALITY_BUCKETS.items()}


def _dolma_topic_paths() -> dict[str, str]:
    """Build {label: glob_pattern} for Dolma source splits."""
    base = str(_DOLMA_V1_7_PATH)
    paths = {}
    for source, file_patterns in DOLMA_DATASETS.items():
        # Use first pattern per source as representative
        paths[source] = f"{base}/{file_patterns[0]}"
    return paths


def build_steps() -> list[StepSpec]:
    """Build the full StepSpec DAG for the Luxical embedding experiment."""
    nemotron_paths = _nemotron_quality_paths()
    dolma_paths = _dolma_topic_paths()

    # ---- Quality classification pipeline ----

    sample_quality = StepSpec(
        name="embed_everything/sample_quality",
        hash_attrs={"n_per_source": N_PER_SOURCE, "buckets": sorted(nemotron_paths.keys()), "seed": SEED},
        fn=lambda output_path: sample_documents(
            output_path=output_path,
            source_paths=nemotron_paths,
            n_per_source=N_PER_SOURCE,
            seed=SEED,
        ),
    )

    oracle_quality = StepSpec(
        name="embed_everything/oracle_quality",
        hash_attrs={"backend": ORACLE_BACKEND, "prompt_version": "v1"},
        deps=[sample_quality],
        fn=lambda output_path: label_quality(
            output_path=output_path,
            input_path=Artifact.load(sample_quality.output_path).path,
            backend=ORACLE_BACKEND,
            prompt_version="v1",
        ),
    )

    embed_quality = StepSpec(
        name="embed_everything/embed_quality",
        hash_attrs={"model": EMBEDDING_MODEL},
        deps=[sample_quality],
        fn=lambda output_path: embed_documents(
            output_path=output_path,
            input_path=Artifact.load(sample_quality.output_path).path,
            model_name=EMBEDDING_MODEL,
        ),
    )

    eval_quality = StepSpec(
        name="embed_everything/eval_quality",
        deps=[oracle_quality, embed_quality],
        fn=lambda output_path: evaluate_quality_probe(
            output_path=output_path,
            embeddings_path=Artifact.load(embed_quality.output_path).path,
        ),
    )

    # ---- Topic clustering pipeline ----

    sample_topic = StepSpec(
        name="embed_everything/sample_topic",
        hash_attrs={"n_per_source": N_PER_SOURCE, "sources": sorted(dolma_paths.keys()), "seed": SEED},
        fn=lambda output_path: sample_documents(
            output_path=output_path,
            source_paths=dolma_paths,
            n_per_source=N_PER_SOURCE,
            seed=SEED,
        ),
    )

    oracle_topic = StepSpec(
        name="embed_everything/oracle_topic",
        hash_attrs={"backend": ORACLE_BACKEND, "prompt_version": "v1"},
        deps=[sample_topic],
        fn=lambda output_path: label_topics(
            output_path=output_path,
            input_path=Artifact.load(sample_topic.output_path).path,
            backend=ORACLE_BACKEND,
            prompt_version="v1",
        ),
    )

    embed_topic = StepSpec(
        name="embed_everything/embed_topic",
        hash_attrs={"model": EMBEDDING_MODEL},
        deps=[sample_topic],
        fn=lambda output_path: embed_documents(
            output_path=output_path,
            input_path=Artifact.load(sample_topic.output_path).path,
            model_name=EMBEDDING_MODEL,
        ),
    )

    n_topics = len(DOLMA_SOURCE_SPLITS)
    eval_topic = StepSpec(
        name="embed_everything/eval_topic",
        hash_attrs={"n_clusters": n_topics},
        deps=[oracle_topic, embed_topic],
        fn=lambda output_path: evaluate_topic_clusters(
            output_path=output_path,
            embeddings_path=Artifact.load(embed_topic.output_path).path,
            n_clusters=n_topics,
        ),
    )

    return [
        sample_quality,
        oracle_quality,
        embed_quality,
        eval_quality,
        sample_topic,
        oracle_topic,
        embed_topic,
        eval_topic,
    ]


if __name__ == "__main__":
    steps = build_steps()
    runner = StepRunner()
    runner.run(steps)
