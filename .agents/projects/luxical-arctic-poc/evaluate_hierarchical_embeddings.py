# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate saved embeddings with accepted hierarchical semantic labels."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from evaluate_semantic_embeddings import (
    FAST_STUDENT_REPORT_URL,
    MANIFEST_URL,
    NEIGHBOR_COUNT,
    SEMANTIC_REFERENCE_MODELS,
    arctic_vectors,
    best_reference_metrics,
    candidate_vectors,
    local_model_vectors,
    semantic_sample,
)
from evaluate_teacher_candidate import CANDIDATES
from glm_hierarchical_labels import OUTPUT_ROOT, HierarchicalAssignment, parse_hierarchy
from glm_semantic_labels import SampleDocument, read_json, read_jsonl
from rigging.filesystem import atomic_rename
from semantic_embedding_metrics import cosine_order_fidelity, semantic_metrics, student_gates

SEED = 42
DEFAULT_RUN_ID = "hierarchy-1000-20260802-001"
DEFAULT_VARIANTS = ("compact", "balanced")
RESULT_FILE = Path("/tmp/luxical-hierarchical-embedding-screen")

logger = logging.getLogger(__name__)


def hierarchical_assignments(run_id: str, variant: str, documents: list[SampleDocument]) -> list[HierarchicalAssignment]:
    """Load and align one complete hierarchy assignment table."""
    root = OUTPUT_ROOT / run_id / variant
    paths = sorted((root / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    assignments.sort(key=lambda row: row.sample_index)
    expected = [row.sample_index for row in documents]
    if [row.sample_index for row in assignments] != expected:
        raise ValueError(f"The {variant} hierarchy assignments are not complete and aligned")
    return assignments


def label_levels(assignments: list[HierarchicalAssignment]) -> dict[str, tuple[np.ndarray, list[frozenset[str]]]]:
    """Return primary and multi-label arrays for each hierarchy level."""
    return {
        "parent": (
            np.asarray([row.primary_parent_id for row in assignments]),
            [frozenset((row.primary_parent_id, *row.secondary_parent_ids)) for row in assignments],
        ),
        "leaf": (
            np.asarray([row.primary_leaf_id for row in assignments]),
            [frozenset((row.primary_leaf_id, *row.secondary_leaf_ids)) for row in assignments],
        ),
        "form": (
            np.asarray([row.form_id for row in assignments]),
            [frozenset((row.form_id,)) for row in assignments],
        ),
    }


def embedding_models(documents: list[SampleDocument]) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load all saved student, baseline, and teacher vectors once."""
    manifest = read_json(MANIFEST_URL)
    models, metadata = local_model_vectors(documents)
    models["arctic_medium"] = arctic_vectors(manifest, documents)
    models["qwen3_embedding_0.6b"] = candidate_vectors("qwen3-embedding-0.6b", manifest, documents)
    models["lfm2.5_embedding_350m"] = candidate_vectors("lfm2.5-embedding-350m", manifest, documents)
    metadata["arctic_medium"] = {"vector_root": "teacher-arctic-v1"}
    metadata["qwen3_embedding_0.6b"] = asdict(CANDIDATES["qwen3-embedding-0.6b"])
    metadata["lfm2.5_embedding_350m"] = asdict(CANDIDATES["lfm2.5-embedding-350m"])
    return models, metadata, manifest


def variant_metrics(
    models: dict[str, np.ndarray],
    assignments: list[HierarchicalAssignment],
    sources: np.ndarray,
    speed_ratio: float,
) -> dict[str, Any]:
    """Return per-level metrics and student gates for one hierarchy."""
    levels = {}
    for level, (primary_labels, label_sets) in label_levels(assignments).items():
        cluster_count = len(set(primary_labels.tolist()))
        model_metrics = {}
        for name, vectors in models.items():
            logger.info("Measuring %s coherence for %s", level, name)
            metrics, _ = semantic_metrics(
                vectors,
                primary_labels,
                label_sets,
                neighbor_count=NEIGHBOR_COUNT,
                cluster_count=cluster_count,
                seed=SEED,
                exclusion_groups=sources,
            )
            model_metrics[name] = metrics
        qwen_vectors = models["qwen3_embedding_0.6b"]
        for name, vectors in models.items():
            model_metrics[name]["qwen_cosine_order_fidelity"] = cosine_order_fidelity(vectors, qwen_vectors)
        reference = best_reference_metrics(model_metrics)
        gates = student_gates(model_metrics["fast_arctic_3m"], reference, speed_ratio)
        levels[level] = {
            "label_count": cluster_count,
            "models": model_metrics,
            "best_semantic_reference_metrics": reference,
            "fast_arctic_3m_gates_against_best_teacher": gates,
            "fast_arctic_3m_all_gates_passed": all(gates.values()),
        }
    return levels


def write_report(run_id: str, report: dict[str, Any]) -> str:
    """Write one private hierarchy embedding report."""
    url = str(OUTPUT_ROOT / run_id / "embedding-screen-v1" / "report.json")
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
    return url


def main() -> None:
    """Parse arguments and run the hierarchy embedding screen."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    documents, _, _ = semantic_sample()
    models, metadata, manifest = embedding_models(documents)
    speed_report = read_json(FAST_STUDENT_REPORT_URL)
    speed_ratio = float(speed_report["comparison"]["speed_ratio"])
    sources = np.asarray([row.source for row in documents])
    variants = {}
    for variant in args.variants:
        taxonomy = read_json(str(OUTPUT_ROOT / args.run_id / variant / "taxonomy.json"))
        parse_hierarchy(taxonomy)
        assignments = hierarchical_assignments(args.run_id, variant, documents)
        variants[variant] = variant_metrics(models, assignments, sources, speed_ratio)
    report = {
        "documents": len(documents),
        "hierarchy_run_root": str(OUTPUT_ROOT / args.run_id),
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_metadata_usage": ["align_saved_vectors", "exclude_same_source_neighbors"],
        "source_metadata_used_as_quality_target": False,
        "semantic_reference_models": list(SEMANTIC_REFERENCE_MODELS),
        "fast_arctic_3m_cpu_speed_ratio": speed_ratio,
        "model_metadata": metadata,
        "variants": variants,
    }
    url = write_report(args.run_id, report)
    result = {"report_url": url, "variants": variants}
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("HIERARCHICAL_EMBEDDING_SCREEN=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
