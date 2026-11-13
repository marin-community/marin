# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute sampled content receipts from immutable task-level observations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
from marin.execution.artifact import RECORD_FILENAME, read_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus

from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC
from experiments.post_training.task_curation.full_qc_review import quality_result
from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.sample_qc import panel_tasks
from experiments.post_training.task_curation.sample_qc_artifacts import PINNED_REVIEWER_SHA256, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt


def sample_qc_receipt(path: str) -> SampleQCReceipt:
    """Recompute counts from immutable observations with the exact pinned panel contracts."""
    record = read_record(path)
    if record is None or record.result is None:
        raise ValueError(f"Missing sampled review artifact: {path}")
    result = SampleQCReview.model_validate({**record.result, "path": path})
    reviewer_spec = {key: value for key, value in result.reviewer.items() if key != "mode"}
    if (
        result.reviewer.get("mode") != "chat"
        or canonical_sha256(reviewer_spec) != PINNED_REVIEWER_SHA256
        or canonical_sha256(result.rubric) != canonical_sha256(asdict(REVIEW_RUBRIC))
    ):
        raise ValueError("Sample Result uses another reviewer or rubric")
    records_path = StoragePath(result.records_path)
    if file_sha256(records_path) != result.records_sha256:
        raise ValueError("Sample observations changed after review")
    _, contracts = panel_tasks(result.panel)
    expected = {row["task_id"]: row for row in contracts}
    with records_path.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    if len(rows) != len(expected) or {row["task_id"] for row in rows} != set(expected):
        raise ValueError("Sample Result does not cover its selected panel exactly once")
    reviews = []
    for row in rows:
        contract = {key: row[key] for key in ("task_id", "task_sha256", "runtime_sha256")}
        if contract != expected[row["task_id"]] or row["batch_receipt"] not in result.batch_receipts:
            raise ValueError("Sample observation differs from its frozen task/runtime contract")
        review = ReviewRecord.model_validate_json(row["review"])
        if review.task_id != row["task_id"]:
            raise ValueError("Sample observation belongs to another task")
        reviews.append(review)
    if (
        quality_result(reviews) != result.result
        or result.attempted_count != len(reviews)
        or result.panel_coverage_complete != all(row.status is ReviewStatus.REVIEWED for row in reviews)
    ):
        raise ValueError("Sample Result differs from its task-level observations")
    return SampleQCReceipt(
        source_id=result.panel.source_id,
        review=result,
        artifact_sha256=file_sha256(StoragePath(path) / RECORD_FILENAME),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-artifacts", required=True, help="JSON array of sampled review artifact paths")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    paths = json.loads(StoragePath(args.review_artifacts).read_text())
    receipts = [sample_qc_receipt(path) for path in paths]
    if len({row.source_id for row in receipts}) != len(receipts):
        raise ValueError("Bind one selected review artifact per source")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_directory, {str(row.source_id): row.model_dump(mode="json") for row in receipts})


if __name__ == "__main__":
    main()
