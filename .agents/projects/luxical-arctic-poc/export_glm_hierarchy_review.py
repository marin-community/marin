# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export private review samples from a completed GLM hierarchy run."""

import argparse
import json
from dataclasses import asdict
from typing import Any

from glm_hierarchical_labels import FORMS, OUTPUT_ROOT, HierarchicalAssignment, parse_hierarchy
from glm_semantic_labels import SampleDocument, read_json, read_jsonl
from verify_glm_hierarchy_with_claude import review_indices


def review_package(
    documents: list[SampleDocument],
    assignments: list[HierarchicalAssignment],
    taxonomy: dict[str, Any],
    representative_size: int,
    stress_size: int,
) -> dict[str, Any]:
    """Return a source-blind package with separate review populations."""
    assignment_rows = [asdict(row) for row in assignments]
    samples = review_indices(assignment_rows, representative_size, stress_size)
    selected = set(samples["representative"]) | set(samples["stress"])
    document_by_index = {row.sample_index: row for row in documents}
    assignment_by_index = {row.sample_index: row for row in assignments}
    return {
        "taxonomy": taxonomy | {"forms": [asdict(row) for row in FORMS]},
        "documents": [{"sample_index": index, "text": document_by_index[index].text} for index in sorted(selected)],
        "glm_assignments": [asdict(assignment_by_index[index]) for index in sorted(selected)],
        "samples": samples,
        "selection": {
            "representative": "stable uniform sample from all fixed evaluation documents",
            "stress": "lowest GLM confidence outside the representative sample",
        },
        "source_metadata_in_package": False,
    }


def export_review(
    run_id: str,
    variant: str,
    representative_size: int,
    stress_size: int,
    output: str,
) -> None:
    """Load one complete hierarchy and write its private review package."""
    root = OUTPUT_ROOT / run_id
    documents = [SampleDocument(**row) for row in read_jsonl(root.parent.parent / "sample-private.jsonl.gz")]
    taxonomy = read_json(str(root / variant / "taxonomy.json"))
    parse_hierarchy(taxonomy)
    paths = sorted((root / variant / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    if len(documents) != 1_000 or len(assignments) != len(documents):
        raise ValueError("The hierarchy review inputs are not complete")
    package = review_package(documents, assignments, taxonomy, representative_size, stress_size)
    with open(output, "w") as file:
        json.dump(package, file, ensure_ascii=False, sort_keys=True)


def main() -> None:
    """Parse arguments and export one private review package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=("compact", "balanced"), required=True)
    parser.add_argument("--representative-size", type=int, default=100)
    parser.add_argument("--stress-size", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_review(args.run_id, args.variant, args.representative_size, args.stress_size, args.output)


if __name__ == "__main__":
    main()
