# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry completed old-campaign reviews into a new FULL receipt under a typed marker.

An earlier FULL-QC campaign completed reviews whose reviewer identity no longer
matches the current pin. Those completed reviews count as FULL reviews for exactly
the content-hash-matching tasks: only rows whose retained task and runtime bytes
match the current contract index, and whose status is REVIEWED, are carried. Old
INVALID or truncated rows are not grandfathered and still need review. Carried rows
keep their original review payload and old-campaign batch receipt, and the merged
receipt records the old reviewer identity under a typed marker, so they are never
relabeled as current-reviewer output (user ruling 2026-11-15). Fresh rows go
through the current reviewer and the shared content-addressed review cache.
"""

import argparse
import json
import logging
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from marin.execution.artifact import RECORD_FILENAME
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRecord, ReviewRubric, ReviewStatus
from taskcompendium.pipeline.review import Reviewer
from taskcompendium.pipeline.reviewer_spec import REVIEW_PAYLOAD_REVISION, SUPPORTED_REVIEW_PAYLOAD_REVISIONS

from experiments.post_training.task_curation.full_qc import (
    REVIEW_REQUEST_TIMEOUTS,
    REVIEW_RUBRIC,
    ReadyChatReviewer,
    ReviewerVersion,
    local_reviewer,
)
from experiments.post_training.task_curation.full_qc_artifacts import FullQCReview, GrandfatheredReview
from experiments.post_training.task_curation.full_qc_inputs import FullQCInputs
from experiments.post_training.task_curation.full_qc_review import (
    RECORD_SCHEMA,
    contract_batches,
    load_inputs,
    review_batch,
    write_source_result,
)
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, PinnedLocalJudgeEndpoint

logger = logging.getLogger(__name__)


def _load_old_review(review_receipt: ContentEvidenceReference) -> FullQCReview:
    receipt_path = StoragePath(review_receipt.uri)
    if receipt_path.name != RECORD_FILENAME or file_sha256(receipt_path) != review_receipt.sha256:
        raise ValueError("Grandfathered review receipt changed")
    review = FullQCReview.raw_load(str(receipt_path.parent))
    if file_sha256(StoragePath(review.records_path)) != review.records_sha256:
        raise ValueError("Grandfathered review observations changed")
    return review


def _carried_rows(old: FullQCReview) -> dict[str, dict]:
    """Completed old-campaign observations keyed by task ID; invalid rows stay behind."""
    with StoragePath(old.records_path).open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    carried: dict[str, dict] = {}
    for row in rows:
        review = ReviewRecord.model_validate_json(row["review"])
        if review.status is ReviewStatus.REVIEWED:
            carried[row["task_id"]] = row
    return carried


def grandfathered_review(
    inputs: FullQCInputs,
    review_receipt: ContentEvidenceReference,
    reviewer: Reviewer,
    rubric: ReviewRubric,
    evidence: ContentEvidenceReference,
    output_path: str,
) -> FullQCReview:
    """Merge content-matching old-campaign reviews under a marker and review the remainder."""
    old = _load_old_review(review_receipt)
    if old.source_id != inputs.source_id:
        raise ValueError(f"Grandfathered review belongs to another source: {old.source_id}")
    carried = _carried_rows(old)
    if not carried:
        raise ValueError("Grandfathered review has no completed rows to carry")
    batches: list[list[dict]] = []
    reviews: list[ReviewRecord] = []
    receipts: list[str] = []
    carried_count = 0
    output = StoragePath(output_path)
    for batch in contract_batches(inputs):
        pending_tasks: list[TaskSpec] = []
        pending_contracts: list[dict] = []
        rows: list[dict] = []
        for task, contract in zip(batch.tasks, batch.contracts, strict=True):
            row = carried.pop(contract["task_id"], None)
            if row is None:
                pending_tasks.append(task)
                pending_contracts.append(contract)
                continue
            if (row["task_sha256"], row["runtime_sha256"]) != (contract["task_sha256"], contract["runtime_sha256"]):
                # Not an exactly content-hash-matching task: review it fresh instead.
                logger.info("Grandfathered row differs from the retained contract: %s", task.id)
                pending_tasks.append(task)
                pending_contracts.append(contract)
                continue
            rows.append(dict(row))
            reviews.append(ReviewRecord.model_validate_json(row["review"]))
            carried_count += 1
        if pending_tasks:
            fresh_records, receipt = review_batch(pending_tasks, pending_contracts, rubric, reviewer, output)
            receipts.append(receipt)
            fresh_by_id = {record.task_id: record for record in fresh_records}
            rows.extend(
                {**contract, "review": fresh_by_id[contract["task_id"]].model_dump_json(), "batch_receipt": receipt}
                for contract in pending_contracts
            )
            reviews.extend(fresh_records)
        batches.append(rows)
    if carried:
        logger.info("Grandfathered tasks outside the retained population stay behind: %d", len(carried))
    records_out = output / "attempts" / uuid4().hex / "reviews.parquet"
    with records_out.open("wb", auto_mkdir=True) as stream:
        with pq.ParquetWriter(stream, RECORD_SCHEMA, compression="zstd") as writer:
            for rows in batches:
                writer.write_table(
                    pa.Table.from_pylist(sorted(rows, key=lambda row: row["task_id"]), schema=RECORD_SCHEMA)
                )
    marker = GrandfatheredReview(reviewer=old.reviewer, evidence=evidence, accepted_count=carried_count)
    logger.info(
        "Grandfathered %d completed reviews from reviewer %s into %s",
        carried_count,
        old.reviewer.get("model_revision"),
        inputs.source_id,
    )
    return write_source_result(inputs, reviewer.identity, rubric, records_out, reviews, receipts, grandfathered=marker)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Prepared FullQCInputs artifact directory")
    parser.add_argument("--review", required=True, help="Earlier FULL campaign receipt record file")
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), required=True)
    parser.add_argument(
        "--review-payload-revision", choices=sorted(SUPPORTED_REVIEW_PAYLOAD_REVISIONS), default=REVIEW_PAYLOAD_REVISION
    )
    parser.add_argument("--evidence", required=True, help="Recorded decision authorizing the carried reviews")
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--review-cache", required=True)
    parser.add_argument("--output", required=True, help="Merged FULL receipt output directory")
    args = parser.parse_args()
    configure_coreweave_s3()
    inputs = load_inputs(args.inputs)
    spec = local_reviewer(args.reviewer_version, payload_revision=args.review_payload_revision)
    logger.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    assert isinstance(spec.endpoint, (LocalJudgeEndpoint, PinnedLocalJudgeEndpoint))
    with spec.endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            query_cache_root=args.review_cache,
            max_concurrent=32,
            request_timeout=REVIEW_REQUEST_TIMEOUTS[args.reviewer_version],
        )
        result = grandfathered_review(
            inputs,
            ContentEvidenceReference(uri=args.review, sha256=args.review_sha256),
            reviewer,
            REVIEW_RUBRIC,
            ContentEvidenceReference(uri=args.evidence, sha256=args.evidence_sha256),
            args.output,
        )
        logger.info("Merged FULL receipt: %s", result.path)


if __name__ == "__main__":
    main()
