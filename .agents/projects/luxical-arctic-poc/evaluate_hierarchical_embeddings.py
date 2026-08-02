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
from rigging.filesystem import StoragePath, atomic_rename
from semantic_embedding_metrics import cosine_order_fidelity, semantic_metrics, student_gates

SEED = 42
DEFAULT_RUN_ID = "hierarchy-1000-20260802-001"
DEFAULT_VARIANTS = ("compact", "balanced")
DEFAULT_MAXIMUM_PAIR_COUNT = 1_000_000
MINIMUM_GROUP_SUPPORT = 30
MAXIMUM_GROUP_F1_LOSS = 0.03
EXACT_SPEED_REPORT_URL = (
    "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/"
    "fast-student/speed/cpu-trained-full-full-3m.json"
)
RESULT_FILE = Path("/tmp/luxical-hierarchical-embedding-screen")

logger = logging.getLogger(__name__)


def hierarchical_assignments(root: StoragePath, documents: list[SampleDocument]) -> list[HierarchicalAssignment]:
    """Load and align one complete hierarchy assignment table."""
    paths = sorted((root / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    assignments.sort(key=lambda row: row.sample_index)
    expected = [row.sample_index for row in documents]
    if [row.sample_index for row in assignments] != expected:
        raise ValueError(f"The hierarchy assignments at {root} are not complete and aligned")
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


def group_f1_gates(model_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare student F1 with the best teacher for each large label group."""
    metric = "cross_group_nearest_primary_per_label"
    student_groups = model_metrics["fast_arctic_3m"][metric]
    output = {}
    for label, student in student_groups.items():
        support = int(student["support"])
        if support < MINIMUM_GROUP_SUPPORT:
            continue
        teacher_values = {model: float(model_metrics[model][metric][label]["f1"]) for model in SEMANTIC_REFERENCE_MODELS}
        best_model = max(teacher_values, key=lambda name: teacher_values[name])
        best_value = teacher_values[best_model]
        student_value = float(student["f1"])
        output[label] = {
            "support": support,
            "student_f1": student_value,
            "best_teacher": best_model,
            "best_teacher_f1": best_value,
            "delta": student_value - best_value,
            "passed": student_value >= best_value - MAXIMUM_GROUP_F1_LOSS,
        }
    return output


def variant_metrics(
    models: dict[str, np.ndarray],
    assignments: list[HierarchicalAssignment],
    sources: np.ndarray,
    speed_ratio: float,
    maximum_pair_count: int | None,
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
                maximum_pair_count=maximum_pair_count,
            )
            model_metrics[name] = metrics
        qwen_vectors = models["qwen3_embedding_0.6b"]
        for name, vectors in models.items():
            model_metrics[name]["qwen_cosine_order_fidelity"] = cosine_order_fidelity(
                vectors,
                qwen_vectors,
                maximum_pair_count=maximum_pair_count,
                seed=SEED,
            )
        reference = best_reference_metrics(model_metrics)
        gates = student_gates(model_metrics["fast_arctic_3m"], reference, speed_ratio)
        groups = group_f1_gates(model_metrics)
        student_metrics = model_metrics["fast_arctic_3m"]
        health_gates = {
            "finite": float(student_metrics["finite_fraction"]) == 1.0,
            "unique": float(student_metrics["unique_fraction_4dp"]) >= 0.99,
            "effective_rank_fraction": float(student_metrics["effective_rank_fraction"]) >= 0.25,
            "total_variance": float(student_metrics["total_variance"]) >= 0.50,
            "cpu_speed": speed_ratio >= 0.85,
        }
        levels[level] = {
            "label_count": cluster_count,
            "models": model_metrics,
            "best_semantic_reference_metrics": reference,
            "fast_arctic_3m_gates_against_best_teacher": gates,
            "fast_arctic_3m_large_group_f1": groups,
            "fast_arctic_3m_large_group_gates_passed": all(row["passed"] for row in groups.values()),
            "fast_arctic_3m_health_gates": health_gates,
            "fast_arctic_3m_all_gates_passed": (
                all(gates.values()) and all(row["passed"] for row in groups.values()) and all(health_gates.values())
            ),
        }
    return levels


def write_report(root: StoragePath, report: dict[str, Any]) -> str:
    """Write one private hierarchy embedding report."""
    url = str(root / "embedding-screen-v1" / "report.json")
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
    parser.add_argument("--evaluation-run-id")
    parser.add_argument("--maximum-pair-count", type=int, default=DEFAULT_MAXIMUM_PAIR_COUNT)
    args = parser.parse_args()
    if args.maximum_pair_count < 1:
        parser.error("--maximum-pair-count must be positive")
    if args.evaluation_run_id is not None and len(args.variants) != 1:
        parser.error("A held-out evaluation requires exactly one accepted hierarchy variant")
    logging.basicConfig(level=logging.INFO)

    variant = args.variants[0]
    variant_root = OUTPUT_ROOT / args.run_id / variant
    evaluation_root = variant_root / args.evaluation_run_id if args.evaluation_run_id is not None else None
    if evaluation_root is None:
        documents, _, _ = semantic_sample()
        report_root = OUTPUT_ROOT / args.run_id
    else:
        documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
        documents.sort(key=lambda row: row.sample_index)
        if [row.sample_index for row in documents] != list(range(len(documents))):
            raise ValueError("The held-out evaluation documents are not complete")
        report_root = evaluation_root
    models, metadata, manifest = embedding_models(documents)
    speed_report = read_json(EXACT_SPEED_REPORT_URL)
    speed_ratio = float(speed_report["student_to_baseline_ratio"])
    sources = np.asarray([row.source for row in documents])
    variants = {}
    for variant_name in args.variants:
        current_variant_root = OUTPUT_ROOT / args.run_id / variant_name
        taxonomy = read_json(str(current_variant_root / "taxonomy.json"))
        parse_hierarchy(taxonomy)
        assignment_root = (
            current_variant_root / args.evaluation_run_id if args.evaluation_run_id is not None else current_variant_root
        )
        assignments = hierarchical_assignments(assignment_root, documents)
        variants[variant_name] = variant_metrics(
            models,
            assignments,
            sources,
            speed_ratio,
            args.maximum_pair_count,
        )
    report = {
        "documents": len(documents),
        "hierarchy_run_root": str(OUTPUT_ROOT / args.run_id),
        "evaluation_run_id": args.evaluation_run_id,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_metadata_usage": ["align_saved_vectors", "exclude_same_source_neighbors"],
        "source_metadata_used_as_quality_target": False,
        "semantic_reference_models": list(SEMANTIC_REFERENCE_MODELS),
        "fast_arctic_3m_cpu_speed_ratio": speed_ratio,
        "speed_report_url": EXACT_SPEED_REPORT_URL,
        "maximum_pair_count": args.maximum_pair_count,
        "model_metadata": metadata,
        "variants": variants,
    }
    url = write_report(report_root, report)
    result = {"report_url": url, "variants": variants}
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("HIERARCHICAL_EMBEDDING_SCREEN=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
