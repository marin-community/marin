# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the fixed low-confidence tail from held-out GLM hierarchy labels."""

import argparse
import base64
import gzip
import json
import math
from dataclasses import asdict
from typing import Any

from glm_hierarchical_labels import FORMS, OUTPUT_ROOT, HierarchicalAssignment, parse_hierarchy
from glm_semantic_labels import SampleDocument, read_json, read_jsonl, stable_order
from verify_glm_hierarchy_with_claude import REVIEW_CHUNK_MARKER

REVIEW_CHUNK_SIZE = 8_000
DEFAULT_TAIL_FRACTION = 0.05


def low_confidence_indices(
    assignments: list[HierarchicalAssignment],
    tail_fraction: float,
) -> list[int]:
    """Return the stable lowest-confidence fraction of assignments."""
    if not assignments:
        raise ValueError("The assignment table is empty")
    if not 0 < tail_fraction <= 1:
        raise ValueError("The tail fraction must be greater than 0 and at most 1")
    indices = [row.sample_index for row in assignments]
    if len(set(indices)) != len(indices):
        raise ValueError("The assignment table has duplicate sample indices")
    count = max(1, math.ceil(len(assignments) * tail_fraction))
    ordered = sorted(
        assignments,
        key=lambda row: (row.confidence, stable_order(f"adjudication:{row.sample_index}")),
    )
    return [row.sample_index for row in ordered[:count]]


def adjudication_package(
    documents: list[SampleDocument],
    assignments: list[HierarchicalAssignment],
    taxonomy: dict[str, Any],
    tail_fraction: float,
) -> dict[str, Any]:
    """Return a source-blind package for the fixed low-confidence tail."""
    document_by_index = {row.sample_index: row for row in documents}
    assignment_by_index = {row.sample_index: row for row in assignments}
    if len(document_by_index) != len(documents) or len(assignment_by_index) != len(assignments):
        raise ValueError("The adjudication inputs have duplicate sample indices")
    if set(document_by_index) != set(assignment_by_index):
        raise ValueError("The document and assignment indices differ")
    selected = low_confidence_indices(assignments, tail_fraction)
    return {
        "taxonomy": taxonomy | {"forms": [asdict(row) for row in FORMS]},
        "documents": [{"sample_index": index, "text": document_by_index[index].text} for index in sorted(selected)],
        "glm_assignments": [asdict(assignment_by_index[index]) for index in sorted(selected)],
        "samples": {"adjudication": selected},
        "selection": {
            "population": "all fixed held-out evaluation documents",
            "tail_fraction": tail_fraction,
            "rule": "lowest GLM confidence, with stable hash tie breaks",
        },
        "source_metadata_in_package": False,
    }


def export_adjudication(
    pilot_run_id: str,
    variant: str,
    evaluation_run_id: str,
    tail_fraction: float,
) -> None:
    """Load a complete held-out run and stream its adjudication package."""
    variant_root = OUTPUT_ROOT / pilot_run_id / variant
    evaluation_root = variant_root / evaluation_run_id
    summary = read_json(str(evaluation_root / "summary.json"))
    if not summary.get("complete"):
        raise ValueError("The held-out GLM run is not complete")
    documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
    taxonomy = read_json(str(variant_root / "taxonomy.json"))
    parse_hierarchy(taxonomy)
    paths = sorted((evaluation_root / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    if len(assignments) != len(documents):
        raise ValueError("The held-out hierarchy assignments are not complete")
    package = adjudication_package(documents, assignments, taxonomy, tail_fraction)
    compressed = gzip.compress(json.dumps(package, ensure_ascii=False, sort_keys=True).encode())
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start : start + REVIEW_CHUNK_SIZE] for start in range(0, len(encoded), REVIEW_CHUNK_SIZE)]
    for index, chunk in enumerate(chunks):
        print(f"{REVIEW_CHUNK_MARKER}{index:04d}/{len(chunks):04d}:{chunk}")


def main() -> None:
    """Parse arguments and export the low-confidence adjudication package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--variant", choices=("compact", "balanced"), required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    args = parser.parse_args()
    export_adjudication(args.pilot_run_id, args.variant, args.evaluation_run_id, args.tail_fraction)


if __name__ == "__main__":
    main()
