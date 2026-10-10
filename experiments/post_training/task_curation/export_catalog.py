# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the source registry as the RL Data Atlas build input."""

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from experiments.post_training.task_curation.grading_catalog import annotate_catalog_grading
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources


def source_row(source: RlDataSource) -> dict[str, Any]:
    """Project a declaration into the Atlas wire format without executing it."""
    info, pipeline = source.info, source.pipeline
    dataset = source.dataset
    invocation = None
    if pipeline is not None:
        invocation = {
            "name": source.name,
            "version": source.version,
            "source": dataset.name if dataset else None,
            "revision": dataset.revision if dataset else None,
            "files": source.files,
        }
    verifier = info.verifier
    tags = info.tags
    row: dict[str, Any] = {
        "id": info.id,
        "name": source.name,
        "display_name": info.title,
        "origin": info.origin,
        "family": info.family,
        "tags": tags,
        "task_count": info.count,
        "notes": info.notes,
        "dataset_id": dataset.name if dataset else None,
        "dataset_revision": dataset.revision if dataset else None,
        "url": dataset.url if dataset else None,
        "canonical_source": dataset.name if dataset else None,
        "canonical_url": dataset.url if dataset else None,
        "verifier_revision": verifier.revision if verifier else None,
        "verifier_url": verifier.url if verifier else None,
        "verifier_name": verifier.name if verifier else None,
        "type": next(
            (
                label
                for tag, label in (("rlvr", "RLVR"), ("agentic", "Agentic"), ("alignment", "Alignment"))
                if tag in tags
            ),
            "",
        ),
        "turns": next(
            (label for tag, label in (("single-turn", "Single-turn"), ("multi-turn", "Multi-turn")) if tag in tags), ""
        ),
        "is_benchmark": "benchmark" in tags,
        "kind": "Generator" if "generator" in tags else "Dataset",
        "status": "Excluded" if "excluded" in tags else "Available",
        "pipeline": invocation,
    }
    if verifier is not None and verifier.grading is not None:
        grading = verifier.grading
        row["grading_selection"] = {
            "mode": grading.mode,
            "agents": grading.agents,
            "marinskyrl_revision": grading.marinskyrl_revision,
            "harbor_revision": grading.harbor_revision,
        }
    review = source.review
    row.update(
        quality=review.grade,
        review_url=review.evidence_url,
        review_date=review.reviewed_at,
        review_source_revision=review.dataset_revision,
        review_verifier_revision=review.verifier_revision,
        difficulty=None,
        traces=None,
    )
    return row


def catalog_document(sources: Iterable[RlDataSource]) -> dict[str, Any]:
    """Produce deterministic source JSON; its digest identifies the complete inventory."""
    rows = sorted((source_row(source) for source in sources), key=lambda row: row["id"])
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate RL source identifiers")
    revision = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "revision": revision, "sources": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = catalog_document(all_sources().values())
    with httpx.Client(timeout=60) as client:
        annotate_catalog_grading(document, client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
