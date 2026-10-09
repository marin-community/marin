# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the source registry as the RL Data Atlas build input."""

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources


def source_row(source: RlDataSource) -> dict[str, Any]:
    """Project a declaration into the Atlas wire format without executing it."""
    row = asdict(source.metadata)
    pipeline = source.pipeline
    if pipeline is not None:
        upstream = pipeline.source
        if isinstance(upstream, HfSource):
            input_source, input_revision = upstream.repo, upstream.revision
            row["dataset_id"] = row["dataset_id"] or upstream.repo
            row["dataset_revision"] = row["dataset_revision"] or upstream.revision
            if not row["url"]:
                row["url"] = f"https://huggingface.co/datasets/{upstream.repo}"
        else:
            input_source, input_revision = upstream.url, upstream.sha256
            row["dataset_revision"] = row["dataset_revision"] or upstream.sha256
            if not row["url"]:
                row["url"] = upstream.url
        row["pipeline"] = {
            "name": pipeline.name,
            "version": pipeline.version,
            "source": input_source,
            "revision": input_revision,
        }
    else:
        row["pipeline"] = None
    row["display_name"] = row["display_name"] or row["dataset_id"] or source.name
    row["canonical_source"] = row["canonical_source"] or row["display_name"]
    row["canonical_url"] = row["canonical_url"] or row["url"]
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
