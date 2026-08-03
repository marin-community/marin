# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate saved embeddings with accepted hierarchical semantic labels."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from benchmark_trained_fast_student import rate_stability
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
from glm_hierarchical_labels import (
    FORMS,
    OUTPUT_ROOT,
    VARIANTS,
    HierarchicalAssignment,
    parse_hierarchy,
    validate_hierarchy,
)
from glm_semantic_labels import SampleDocument, read_json, read_jsonl, stable_order
from rigging.filesystem import StoragePath, atomic_rename
from semantic_embedding_metrics import (
    cosine_order_fidelity,
    fixed_bucket_metrics,
    nearest_neighbors,
    nearest_neighbors_outside_groups,
    normalize_embeddings,
    semantic_metrics,
    student_gates,
)
from verify_glm_hierarchy_with_claude import validate_claude_rows

SEED = 42
DEFAULT_RUN_ID = "hierarchy-1000-20260802-001"
DEFAULT_VARIANTS = ("compact", "balanced")
DEFAULT_MAXIMUM_PAIR_COUNT = 1_000_000
MINIMUM_GROUP_SUPPORT = 30
MAXIMUM_GROUP_F1_LOSS = 0.03
REVIEW_QUERY_COUNT = 200
REVIEW_NEIGHBOR_COUNT = 5
REVIEW_TEXT_CHARACTERS = 600
PRODUCTION_BUCKET_COUNT = 40
MAXIMUM_PRODUCTION_BUCKET_METRIC_LOSS = 0.02
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


def validated_speed_ratio(
    speed_report: dict[str, Any],
    student_metadata: dict[str, Any],
    baseline_metadata: dict[str, Any],
    student_config: str,
    student_training_name: str,
    student_rung: str,
) -> float:
    """Return the ratio from a stable speed report for the exact model."""
    expected_identity = {
        "mode": "cpu",
        "jax_backend": "cpu",
        "config_name": student_config,
        "teacher": student_training_name,
        "rung": student_rung,
    }
    if any(speed_report.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("The speed report does not identify the evaluated CPU model")
    if speed_report.get("baseline") != baseline_metadata:
        raise ValueError("The speed report does not identify the evaluated Luxical baseline")
    speed_training_report = speed_report.get("training_report")
    if not isinstance(speed_training_report, dict):
        raise ValueError("The speed report has no training report")
    if speed_training_report.get("final_model_sha256") != student_metadata.get("final_model_sha256"):
        raise ValueError("The speed report model hash differs from the evaluated model")
    student_stability = rate_stability([float(value) for value in speed_report.get("student_rates", [])])
    baseline_stability = rate_stability([float(value) for value in speed_report.get("baseline_rates", [])])
    measurement_valid = bool(student_stability["passed"] and baseline_stability["passed"])
    if speed_report.get("measurement_valid") is not True or not measurement_valid:
        raise ValueError("The CPU speed measurement is not stable")
    ratio = float(speed_report["student_to_baseline_ratio"])
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("The CPU speed ratio is not positive and finite")
    return ratio


def adjudicated_assignments(
    assignments: list[HierarchicalAssignment],
    taxonomy: dict[str, Any],
    review: dict[str, Any],
) -> list[HierarchicalAssignment]:
    """Replace the reviewed low-confidence rows with valid Claude labels."""
    rows = review.get("claude_assignments")
    metrics = review.get("adjudication")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise ValueError("The adjudication review has no complete assignment result")
    if int(metrics.get("documents", -1)) != len(rows):
        raise ValueError("The adjudication review count differs from its assignments")
    indices = [int(row["sample_index"]) for row in rows]
    assignment_by_index = {row.sample_index: row for row in assignments}
    if len(assignment_by_index) != len(assignments):
        raise ValueError("The full assignment table has duplicate sample indices")
    if not set(indices).issubset(assignment_by_index):
        raise ValueError("The adjudication review contains an unknown sample index")
    validation_package = {
        "taxonomy": taxonomy | {"forms": [asdict(row) for row in FORMS]},
        "documents": [{"sample_index": index} for index in indices],
    }
    validate_claude_rows(validation_package, rows)
    for row in rows:
        assignment_by_index[int(row["sample_index"])] = HierarchicalAssignment(**row)
    return [assignment_by_index[row.sample_index] for row in assignments]


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


def embedding_models(
    documents: list[SampleDocument],
    student_model: str,
    student_config: str,
    student_training_name: str,
    student_rung: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load all saved student, baseline, and teacher vectors once."""
    manifest = read_json(MANIFEST_URL)
    models, metadata = local_model_vectors(
        documents,
        student_model,
        student_config,
        student_training_name,
        student_rung,
    )
    models["arctic_medium"] = arctic_vectors(manifest, documents)
    models["qwen3_embedding_0.6b"] = candidate_vectors("qwen3-embedding-0.6b", manifest, documents)
    models["lfm2.5_embedding_350m"] = candidate_vectors("lfm2.5-embedding-350m", manifest, documents)
    metadata["arctic_medium"] = {"vector_root": "teacher-arctic-v1"}
    metadata["qwen3_embedding_0.6b"] = asdict(CANDIDATES["qwen3-embedding-0.6b"])
    metadata["lfm2.5_embedding_350m"] = asdict(CANDIDATES["lfm2.5-embedding-350m"])
    return models, metadata, manifest


def group_f1_gates(
    model_metrics: dict[str, dict[str, Any]], student_model: str = "fast_arctic_3m"
) -> dict[str, dict[str, Any]]:
    """Compare student F1 with the best teacher for each large label group."""
    metric = "cross_group_nearest_primary_per_label"
    student_groups = model_metrics[student_model][metric]
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


def production_bucket_gates(model_metrics: dict[str, dict[str, Any]], student_model: str) -> dict[str, dict[str, Any]]:
    """Compare fixed-bucket semantic quality with the best saved teacher."""
    output = {}
    for level in ("parent", "leaf", "form"):
        for metric in ("cluster_nmi", "cluster_purity"):
            teacher_values = {
                model: float(model_metrics[model]["levels"][level][metric]) for model in SEMANTIC_REFERENCE_MODELS
            }
            best_model = max(teacher_values, key=lambda name: teacher_values[name])
            best_value = teacher_values[best_model]
            student_value = float(model_metrics[student_model]["levels"][level][metric])
            output[f"{level}_{metric}"] = {
                "student": student_value,
                "best_teacher": best_model,
                "best_teacher_value": best_value,
                "delta": student_value - best_value,
                "passed": student_value >= best_value - MAXIMUM_PRODUCTION_BUCKET_METRIC_LOSS,
            }
    return output


def production_bucket_report(
    models: dict[str, np.ndarray], assignments: list[HierarchicalAssignment], student_model: str
) -> dict[str, Any]:
    """Return semantic gates for one shared 40-bucket clustering task."""
    primary_labels_by_level = {level: primary_labels for level, (primary_labels, _) in label_levels(assignments).items()}
    model_metrics = {
        name: fixed_bucket_metrics(vectors, primary_labels_by_level, PRODUCTION_BUCKET_COUNT, SEED)
        for name, vectors in models.items()
    }
    gates = production_bucket_gates(model_metrics, student_model)
    return {
        "source_metadata_used": False,
        "cluster_count": PRODUCTION_BUCKET_COUNT,
        "models": model_metrics,
        "student_model": student_model,
        "student_gates_against_best_teacher": gates,
        "student_all_gates_passed": all(row["passed"] for row in gates.values()),
    }


def variant_metrics(
    models: dict[str, np.ndarray],
    assignments: list[HierarchicalAssignment],
    sources: np.ndarray,
    speed_ratio: float,
    maximum_pair_count: int | None,
    student_model: str = "fast_arctic_3m",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return per-level metrics and student gates for one hierarchy."""
    neighbor_cache = {}
    cross_group_neighbor_cache = {}
    for name, vectors in models.items():
        normalized = normalize_embeddings(vectors)
        neighbor_cache[name] = nearest_neighbors(normalized, NEIGHBOR_COUNT)
        cross_group_neighbor_cache[name] = nearest_neighbors_outside_groups(normalized, sources, NEIGHBOR_COUNT)
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
                precomputed_neighbors=neighbor_cache[name],
                precomputed_cross_group_neighbors=cross_group_neighbor_cache[name],
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
        gates = student_gates(model_metrics[student_model], reference, speed_ratio)
        groups = group_f1_gates(model_metrics, student_model)
        student_metrics = model_metrics[student_model]
        health_gates = {
            "finite": float(student_metrics["finite_fraction"]) == 1.0,
            "unique": float(student_metrics["unique_fraction_4dp"]) >= 0.99,
            "effective_rank_fraction": float(student_metrics["effective_rank_fraction"]) >= 0.25,
            "total_variance": float(student_metrics["total_variance"]) >= 0.50,
            "cpu_speed": speed_ratio >= 0.85,
        }
        level_report = {
            "label_count": cluster_count,
            "models": model_metrics,
            "best_semantic_reference_metrics": reference,
            "student_model": student_model,
            "student_gates_against_best_teacher": gates,
            "student_large_group_f1": groups,
            "student_large_group_gates_passed": all(row["passed"] for row in groups.values()),
            "student_health_gates": health_gates,
            "student_all_gates_passed": (
                all(gates.values()) and all(row["passed"] for row in groups.values()) and all(health_gates.values())
            ),
        }
        if student_model == "fast_arctic_3m":
            level_report.update(
                {
                    "fast_arctic_3m_gates_against_best_teacher": gates,
                    "fast_arctic_3m_large_group_f1": groups,
                    "fast_arctic_3m_large_group_gates_passed": level_report["student_large_group_gates_passed"],
                    "fast_arctic_3m_health_gates": health_gates,
                    "fast_arctic_3m_all_gates_passed": level_report["student_all_gates_passed"],
                }
            )
        levels[level] = level_report
    levels["production_buckets"] = production_bucket_report(models, assignments, student_model)
    return levels, cross_group_neighbor_cache


def strongest_reference_model(levels: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Select one teacher by its mean across fixed hierarchy metrics."""
    metric_names = (
        "cross_group_neighbor_any_label_fraction",
        "cross_group_neighbor_label_jaccard",
        "cross_group_nearest_primary_macro_f1",
        "cluster_nmi",
    )
    scores = {
        model: float(
            np.mean(
                [
                    float(levels[level]["models"][model][metric])
                    for level in ("parent", "leaf", "form")
                    for metric in metric_names
                ]
            )
        )
        for model in SEMANTIC_REFERENCE_MODELS
    }
    return max(scores, key=lambda name: scores[name]), scores


def neighborhood_review_indices(assignments: list[HierarchicalAssignment], count: int) -> list[int]:
    """Return a stable review sample with up to 30 central-code queries."""
    if count < 1 or count > len(assignments):
        raise ValueError("The neighborhood review size is invalid")
    code = sorted(
        (row.sample_index for row in assignments if row.form_id == "CODE"),
        key=lambda value: stable_order(f"neighborhood-review-code:{value}"),
    )[: min(30, count)]
    selected = set(code)
    remaining = sorted(
        (row.sample_index for row in assignments if row.sample_index not in selected),
        key=lambda value: stable_order(f"neighborhood-review-population:{value}"),
    )
    return [*code, *remaining[: count - len(code)]]


def blind_neighborhood_package(
    documents: list[SampleDocument],
    assignments: list[HierarchicalAssignment],
    cross_group_neighbors: dict[str, np.ndarray],
    reference_model: str,
    student_model: str = "fast_arctic_3m",
) -> dict[str, Any]:
    """Return randomized model-blind neighbor sets for independent review."""
    document_by_index = {row.sample_index: row for row in documents}
    assignment_by_index = {row.sample_index: row for row in assignments}
    items = []
    for sample_index in neighborhood_review_indices(assignments, REVIEW_QUERY_COUNT):
        student_first = stable_order(f"neighborhood-review-side:{sample_index}")[0] % 2 == 0
        model_order = (student_model, reference_model) if student_first else (reference_model, student_model)
        sets = {}
        for side, model in zip(("A", "B"), model_order, strict=True):
            sets[side] = [
                document_by_index[int(index)].text[:REVIEW_TEXT_CHARACTERS]
                for index in cross_group_neighbors[model][sample_index, :REVIEW_NEIGHBOR_COUNT]
            ]
        assignment = assignment_by_index[sample_index]
        items.append(
            {
                "sample_index": sample_index,
                "query": document_by_index[sample_index].text[:REVIEW_TEXT_CHARACTERS],
                "sets": sets,
                "student_side": "A" if student_first else "B",
                "glm_primary_parent_id": assignment.primary_parent_id,
                "glm_form_id": assignment.form_id,
            }
        )
    return {
        "reference_model": reference_model,
        "student_model": student_model,
        "items": items,
        "source_metadata_in_review": False,
        "same_source_neighbors_excluded": True,
        "selection": "30 central-code queries when available, then a stable uniform sample",
    }


def write_blind_neighborhood_package(root: StoragePath, package: dict[str, Any]) -> str:
    """Write one private compressed blind-review package."""
    url = str(root / "blind-neighborhood-review-v1" / "package.json.gz")
    StoragePath(url).write_text(json.dumps(package, ensure_ascii=False, sort_keys=True), compression="gzip")
    return url


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
    parser.add_argument("--adjudication-review-url")
    parser.add_argument("--maximum-pair-count", type=int, default=DEFAULT_MAXIMUM_PAIR_COUNT)
    parser.add_argument("--student-model", default="fast_arctic_3m")
    parser.add_argument("--student-config", default="full")
    parser.add_argument("--student-training-name", default="full")
    parser.add_argument("--student-rung", default="3m")
    parser.add_argument("--speed-report-url", default=EXACT_SPEED_REPORT_URL)
    args = parser.parse_args()
    if args.maximum_pair_count < 1:
        parser.error("--maximum-pair-count must be positive")
    if args.evaluation_run_id is not None and len(args.variants) != 1:
        parser.error("A held-out evaluation requires exactly one accepted hierarchy variant")
    if args.adjudication_review_url is not None and args.evaluation_run_id is None:
        parser.error("An adjudication review requires a held-out evaluation")
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
        label_root = "adjudicated-v1" if args.adjudication_review_url else "raw-v1"
        report_root = (
            evaluation_root / label_root
            if args.student_model == "fast_arctic_3m"
            else evaluation_root / f"student-{args.student_model}" / label_root
        )
    models, metadata, manifest = embedding_models(
        documents,
        args.student_model,
        args.student_config,
        args.student_training_name,
        args.student_rung,
    )
    speed_report = read_json(args.speed_report_url)
    speed_ratio = validated_speed_ratio(
        speed_report,
        metadata[args.student_model],
        metadata["luxical_one"],
        args.student_config,
        args.student_training_name,
        args.student_rung,
    )
    sources = np.asarray([row.source for row in documents])
    variants = {}
    evaluation_assignments = None
    evaluation_neighbors = None
    adjudication_review = None
    adjudication_sha256 = None
    if args.adjudication_review_url is not None:
        adjudication_text = StoragePath(args.adjudication_review_url).read_text()
        adjudication_sha256 = hashlib.sha256(adjudication_text.encode()).hexdigest()
        adjudication_review = json.loads(adjudication_text)
    for variant_name in args.variants:
        current_variant_root = OUTPUT_ROOT / args.run_id / variant_name
        taxonomy = read_json(str(current_variant_root / "taxonomy.json"))
        validate_hierarchy(parse_hierarchy(taxonomy), VARIANTS[variant_name])
        assignment_root = (
            current_variant_root / args.evaluation_run_id if args.evaluation_run_id is not None else current_variant_root
        )
        assignments = hierarchical_assignments(assignment_root, documents)
        if adjudication_review is not None:
            assignments = adjudicated_assignments(assignments, taxonomy, adjudication_review)
        metrics, cross_group_neighbors = variant_metrics(
            models,
            assignments,
            sources,
            speed_ratio,
            args.maximum_pair_count,
            args.student_model,
        )
        variants[variant_name] = metrics
        evaluation_assignments = assignments
        evaluation_neighbors = cross_group_neighbors
    report = {
        "documents": len(documents),
        "hierarchy_run_root": str(OUTPUT_ROOT / args.run_id),
        "evaluation_run_id": args.evaluation_run_id,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_metadata_usage": ["align_saved_vectors", "exclude_same_source_neighbors"],
        "source_metadata_used_as_quality_target": False,
        "semantic_reference_models": list(SEMANTIC_REFERENCE_MODELS),
        "student_model": args.student_model,
        "student_config": args.student_config,
        "student_training_name": args.student_training_name,
        "student_rung": args.student_rung,
        "student_cpu_speed_ratio": speed_ratio,
        "speed_report_url": args.speed_report_url,
        "maximum_pair_count": args.maximum_pair_count,
        "label_version": "adjudicated" if adjudication_review is not None else "raw_glm",
        "adjudication_review_url": args.adjudication_review_url,
        "adjudication_review_sha256": adjudication_sha256,
        "model_metadata": metadata,
        "variants": variants,
    }
    if args.student_model == "fast_arctic_3m":
        report["fast_arctic_3m_cpu_speed_ratio"] = speed_ratio
    if args.evaluation_run_id is not None:
        assert evaluation_assignments is not None and evaluation_neighbors is not None
        accepted_variant = args.variants[0]
        reference_model, reference_scores = strongest_reference_model(variants[accepted_variant])
        package = blind_neighborhood_package(
            documents,
            evaluation_assignments,
            evaluation_neighbors,
            reference_model,
            args.student_model,
        )
        report["blind_neighborhood_reference_model"] = reference_model
        report["blind_neighborhood_reference_scores"] = reference_scores
        report["blind_neighborhood_package_url"] = write_blind_neighborhood_package(report_root, package)
    url = write_report(report_root, report)
    result = {"report_url": url, "variants": variants}
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("HIERARCHICAL_EMBEDDING_SCREEN=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
