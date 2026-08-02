# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a source-blind Claude review package from a completed GLM run."""

import argparse
import json

from glm_semantic_labels import (
    OUTPUT_ROOT,
    Assignment,
    Bucket,
    SampleDocument,
    claude_review_package,
    read_jsonl,
)
from ladder_config import read_json
from rigging.filesystem import StoragePath

CLAUDE_REVIEW_CHUNK_MARKER = "CLAUDE_REVIEW_CHUNK="
CLAUDE_REVIEW_CHUNK_SIZE = 8_000


def export_review(run_id: str, output_root: StoragePath) -> None:
    """Write a compressed source-blind review package to the private job log."""
    run_root = output_root / run_id
    summary = read_json(str(run_root / "summary.json"))
    if not summary.get("complete"):
        raise ValueError("The GLM semantic-label run is not complete")
    documents = [SampleDocument(**row) for row in read_jsonl(run_root / "sample-private.jsonl.gz")]
    taxonomy = read_json(str(run_root / "taxonomy.json"))
    buckets = [Bucket(**row) for row in taxonomy["buckets"]]
    assignment_paths = sorted((run_root / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [Assignment(**row) for path in assignment_paths for row in read_jsonl(path)]
    if len(documents) != summary["sample_size"] or len(assignments) != summary["assignment_count"]:
        raise ValueError("The completed run counts do not match its review inputs")
    package = claude_review_package(documents, assignments, buckets)
    chunks = [
        package[start : start + CLAUDE_REVIEW_CHUNK_SIZE] for start in range(0, len(package), CLAUDE_REVIEW_CHUNK_SIZE)
    ]
    for index, chunk in enumerate(chunks):
        print(f"{CLAUDE_REVIEW_CHUNK_MARKER}{index:04d}/{len(chunks):04d}:{chunk}")
    print(f"GLM_SEMANTIC_SUMMARY={json.dumps(summary, sort_keys=True)}")
    print(
        "GLM_CLAUDE_REVIEW_EXPORT="
        + json.dumps({"run_id": run_id, "documents": len(documents), "assignments": len(assignments)}, sort_keys=True)
    )


def main() -> None:
    """Parse arguments and export the review package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    export_review(args.run_id, StoragePath(args.output_root))


if __name__ == "__main__":
    main()
