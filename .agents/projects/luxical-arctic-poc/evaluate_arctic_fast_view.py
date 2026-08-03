# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Arctic on the fast student text view with full-window teachers."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from evaluate_hierarchical_embeddings import (
    adjudicated_assignments,
    embedding_models,
    hierarchical_assignments,
    variant_metrics,
)
from fast_student import MAX_TOKENS, fast_document_view
from glm_hierarchical_labels import OUTPUT_ROOT, VARIANTS, parse_hierarchy, validate_hierarchy
from glm_semantic_labels import SampleDocument, read_json, read_jsonl
from rigging.filesystem import StoragePath, atomic_rename
from semantic_embedding_metrics import normalize_embeddings
from teacher_shard import EMBEDDING_DIMENSION, INFERENCE_BATCH_SIZE, MAX_TEACHER_TOKENS, new_teacher

CANDIDATE_NAME = "arctic_fast_view"
TABLE_BATCH_SIZE = 512
RESULT_FILE = Path("/tmp/luxical-arctic-fast-view-diagnostic")

logger = logging.getLogger(__name__)


def fast_view_vectors(teacher: Any, documents: list[SampleDocument]) -> np.ndarray:
    """Embed the exact character view that the fast student tokenizes."""
    batches = []
    for start in range(0, len(documents), TABLE_BATCH_SIZE):
        rows = documents[start : start + TABLE_BATCH_SIZE]
        views = [fast_document_view(row.text) for row in rows]
        vectors = teacher.embed_texts(
            views,
            is_query=False,
            batch_size=INFERENCE_BATCH_SIZE,
            mrl=True,
            progress_bar=False,
        )
        if vectors.shape != (len(rows), EMBEDDING_DIMENSION):
            raise ValueError(f"Arctic returned an invalid fast-view shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Arctic returned non-finite fast-view vectors at row {start}")
        batches.append(vectors)
        logger.info("Embedded Arctic fast views: %d/%d", start + len(rows), len(documents))
    if not batches:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    return normalize_embeddings(np.concatenate(batches))


def write_report(url: str, report: dict[str, Any]) -> None:
    """Write one private diagnostic report."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)


def main() -> None:
    """Run the fixed held-out diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--adjudication-review-url", required=True)
    parser.add_argument("--maximum-pair-count", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.maximum_pair_count < 1:
        parser.error("--maximum-pair-count must be positive")
    logging.basicConfig(level=logging.INFO)

    variant_root = OUTPUT_ROOT / args.run_id / args.variant
    evaluation_root = variant_root / args.evaluation_run_id
    documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
    documents.sort(key=lambda row: row.sample_index)
    if [row.sample_index for row in documents] != list(range(len(documents))):
        raise ValueError("The held-out evaluation documents are not complete")
    taxonomy = read_json(str(variant_root / "taxonomy.json"))
    validate_hierarchy(parse_hierarchy(taxonomy), VARIANTS[args.variant])
    assignments = hierarchical_assignments(evaluation_root, documents)
    adjudication_text = StoragePath(args.adjudication_review_url).read_text()
    adjudication_review = json.loads(adjudication_text)
    assignments = adjudicated_assignments(assignments, taxonomy, adjudication_review)

    models, metadata, manifest = embedding_models(documents, "fast_arctic_3m", "full", "full", "3m")
    teacher = new_teacher()
    models[CANDIDATE_NAME] = fast_view_vectors(teacher, documents)
    metadata[CANDIDATE_NAME] = {
        "teacher_max_tokens": MAX_TEACHER_TOKENS,
        "student_max_tokens": MAX_TOKENS,
        "input_view": "fast_document_view",
    }
    sources = np.asarray([row.source for row in documents])
    metrics, _ = variant_metrics(
        models,
        assignments,
        sources,
        speed_ratio=1.0,
        maximum_pair_count=args.maximum_pair_count,
        student_model=CANDIDATE_NAME,
    )
    report = {
        "diagnostic_only": True,
        "candidate": CANDIDATE_NAME,
        "documents": len(documents),
        "manifest_sha256": manifest["sha256"],
        "run_id": args.run_id,
        "variant": args.variant,
        "evaluation_run_id": args.evaluation_run_id,
        "adjudication_review_url": args.adjudication_review_url,
        "adjudication_review_sha256": hashlib.sha256(adjudication_text.encode()).hexdigest(),
        "student_input_tokens": MAX_TOKENS,
        "full_teacher_input_tokens": 3 * MAX_TEACHER_TOKENS,
        "fast_view_teacher_input_tokens": MAX_TEACHER_TOKENS,
        "source_metadata_used_as_quality_target": False,
        "source_metadata_usage": ["exclude_same_source_neighbors"],
        "metadata": metadata,
        "metrics": metrics,
    }
    report_url = str(evaluation_root / "teacher-diagnostics" / "arctic-fast-view-v1" / "report.json")
    write_report(report_url, report)
    summary = {
        "report_url": report_url,
        "candidate": CANDIDATE_NAME,
        "documents": len(documents),
        "metrics": {
            level: {
                "cluster_nmi": values["models"][CANDIDATE_NAME]["cluster_nmi"],
                "cross_group_nearest_primary_macro_f1": values["models"][CANDIDATE_NAME][
                    "cross_group_nearest_primary_macro_f1"
                ],
                "large_group_gates_passed": values["student_large_group_gates_passed"],
            }
            for level, values in metrics.items()
        },
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("ARCTIC_FAST_VIEW_DIAGNOSTIC=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
