# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review every retained contract, preserving resumable task-level evidence."""

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import ArtifactRecord, read_record, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import (
    Confidence,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewRubric,
    ReviewStatus,
    ReviewVerdict,
)
from taskcompendium.pipeline.review import Reviewer, review_tasks

from experiments.post_training.task_curation.full_qc_artifacts import (
    BatchReviewReceipt,
    FullQCInputs,
    FullQCReview,
    FullQCReviewShard,
)
from experiments.post_training.task_curation.published_census import population_documents

logger = logging.getLogger(__name__)
REVIEW_BATCH_SIZE = 32
MAX_SUMMARY_ISSUES = 50
RECORD_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("task_sha256", pa.string()),
        ("runtime_sha256", pa.string()),
        ("review", pa.large_string()),
        ("batch_receipt", pa.string()),
    ]
)


def load_inputs(path: str) -> FullQCInputs:
    record = read_record(path)
    if record is None or record.result is None:
        raise ValueError(f"Missing prepared input record: {path}")
    return FullQCInputs.model_validate({**record.result, "path": path})


def has_review_issues(verdict: ReviewVerdict) -> bool:
    return (
        verdict.quality in {Quality.ISSUES, Quality.BAD}
        or bool(verdict.defects)
        or verdict.reference_status is ReferenceStatus.CONFLICT
    )


def quality_result(reviews: Sequence[ReviewRecord]) -> SourceQualityResult:
    """Compute informational Result without treating missing observations as clean."""
    reviewed = unavailable = uncertain = 0
    issues = []
    issue_count = 0
    for review in reviews:
        verdict = review.verdict
        if review.status is not ReviewStatus.REVIEWED or verdict is None:
            unavailable += 1
            continue
        if verdict.confidence is Confidence.LOW or (
            verdict.quality is Quality.UNKNOWN and not has_review_issues(verdict)
        ):
            uncertain += 1
            continue
        reviewed += 1
        if has_review_issues(verdict):
            issue_count += 1
            if len(issues) < MAX_SUMMARY_ISSUES:
                issues.append(f"{review.task_id}: {verdict.evidence}")
    recommendation = (
        "Inspect all task-level issue evidence and qualify repairs with renewed QA."
        if issue_count
        else "No repairs indicated by conclusive reviews."
    )
    if unavailable or uncertain:
        recommendation += " Resolve unavailable and uncertain observations before source-wide conclusions."
    return SourceQualityResult(
        reviewed=reviewed,
        issue_count=issue_count,
        issues=issues,
        repair_recommendation=recommendation,
        unavailable_count=unavailable,
        uncertain_count=uncertain,
    )


def review_batch(
    tasks: list[TaskSpec], contracts: list[dict], rubric: ReviewRubric, reviewer: Reviewer, output: StoragePath
) -> tuple[list[ReviewRecord], str]:
    identity = {"contracts": contracts, "rubric": asdict(rubric), "reviewer": reviewer.identity}
    digest = canonical_sha256(identity)
    root = output / "batches" / digest
    complete = root / "complete.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        return load_batch_reviews(receipt["path"], receipt["sha256"], identity), receipt["path"]
    latest = root / "latest.json"
    previous = {}
    attempts = []
    if latest.exists():
        receipt = json.loads(latest.read_text())
        previous = {
            row.task_id: row
            for row in load_batch_reviews(receipt["path"], receipt["sha256"], identity)
            if row.status is ReviewStatus.REVIEWED
        }
        attempts = json.loads(StoragePath(receipt["path"]).read_text())["attempts"]
    pending = [task for task in tasks if task.id not in previous]
    if pending:
        result = review_tasks(pending, rubric, reviewer, cached=None)
        if len(result.reviews) != len(pending) or {row.task_id for row in result.reviews} != {
            task.id for task in pending
        }:
            raise ValueError("Review must account for every supplied task exactly once")
        previous.update({row.task_id: row for row in result.reviews})
        attempts.extend(asdict(attempt) for attempt in result.attempts)
    reviews = [previous[task.id] for task in tasks]
    path = root / "attempts" / f"{uuid4().hex}.json"
    payload = {
        "identity": identity,
        "reviews": [row.model_dump(mode="json") for row in reviews],
        "attempts": attempts,
    }
    with path.open("w", auto_mkdir=True) as stream:
        json.dump(payload, stream, ensure_ascii=False)
    receipt = {"path": str(path), "sha256": file_sha256(path)}
    with latest.open("w", auto_mkdir=True) as stream:
        json.dump(receipt, stream)
    if all(row.status is ReviewStatus.REVIEWED for row in reviews):
        with complete.open("w", auto_mkdir=True) as stream:
            json.dump(receipt, stream)
    return reviews, str(path)


def load_batch_reviews(path: str, sha256: str, identity: dict) -> list[ReviewRecord]:
    """Read immutable evidence for the supplied exact reviewer and task contracts."""
    location = StoragePath(path)
    if file_sha256(location) != sha256:
        raise ValueError(f"Completed review evidence changed: {path}")
    payload = json.loads(location.read_text())
    if canonical_sha256(payload["identity"]) != canonical_sha256(identity):
        raise ValueError(f"Completed review belongs to another contract: {path}")
    reviews = [ReviewRecord.model_validate_json(json.dumps(row)) for row in payload["reviews"]]
    expected = {row["task_id"] for row in identity["contracts"]}
    if len(reviews) != len(expected) or {row.task_id for row in reviews} != expected:
        raise ValueError("Review receipt must cover its retained contracts exactly once")
    return reviews


@dataclass(frozen=True)
class ContractBatch:
    index: int
    tasks: list[TaskSpec]
    contracts: list[dict]


def contract_batches(inputs: FullQCInputs) -> Iterator[ContractBatch]:
    """Yield the original review batches after checking exact retained task and runtime bytes."""
    index_path = StoragePath(inputs.index_path)
    if file_sha256(index_path) != inputs.index_sha256:
        raise ValueError("Prepared contract index changed")
    with index_path.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    index = {row["task_id"]: row for row in rows}
    if len(index) != len(rows) or len(index) != inputs.retained_count:
        raise ValueError("Prepared contract index is incomplete")
    seen = set()
    tasks: list[TaskSpec] = []
    contracts = []
    batch_index = 0
    for document, deployment in population_documents(inputs.population):
        identifier = document["id"]
        if identifier not in index:
            continue
        task = TaskSpec.model_validate(document)
        contract = {
            "task_id": task.id,
            "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            "runtime_sha256": canonical_sha256(deployment),
        }
        if task.id in seen or contract != index[task.id]:
            raise ValueError(f"Task differs from prepared contract: {task.id}")
        seen.add(task.id)
        tasks.append(task)
        contracts.append(contract)
        if len(tasks) == REVIEW_BATCH_SIZE:
            yield ContractBatch(batch_index, tasks, contracts)
            tasks, contracts = [], []
            batch_index += 1
    if tasks:
        yield ContractBatch(batch_index, tasks, contracts)
    if seen != set(index):
        raise ValueError("Review input does not cover every prepared retained task")


def review_inputs(inputs: FullQCInputs, reviewer: Reviewer, rubric: ReviewRubric, output_path: str) -> FullQCReview:
    """Review the complete proposed remainder without making admission decisions."""
    output = StoragePath(output_path)

    def observations():
        for batch in contract_batches(inputs):
            reviews, receipt = review_batch(batch.tasks, batch.contracts, rubric, reviewer, output)
            yield batch, reviews, receipt

    return write_review_result(inputs, reviewer.identity, rubric, output_path, observations())


def write_review_result(
    inputs: FullQCInputs,
    reviewer: dict,
    rubric: ReviewRubric,
    output_path: str,
    batches: Iterable[tuple[ContractBatch, list[ReviewRecord], str]],
) -> FullQCReview:
    """Persist complete retained observations and compute the informational source Result."""
    records_path = StoragePath(output_path) / "attempts" / uuid4().hex / "reviews.parquet"
    reviews: list[ReviewRecord] = []
    receipts = []

    with records_path.open("wb", auto_mkdir=True) as stream:
        with pq.ParquetWriter(stream, RECORD_SCHEMA, compression="zstd") as writer:

            for batch, records, receipt in batches:
                observations = {row.task_id: row for row in records}
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                **contract,
                                "review": observations[contract["task_id"]].model_dump_json(),
                                "batch_receipt": receipt,
                            }
                            for contract in batch.contracts
                        ],
                        schema=RECORD_SCHEMA,
                    )
                )
                reviews.extend(records)
                receipts.append(receipt)
                logger.info(
                    "%s: %d / %d retained contracts reviewed", inputs.source_id, len(reviews), inputs.retained_count
                )
    return write_source_result(inputs, reviewer, rubric, records_path, reviews, receipts)


def write_source_result(
    inputs: FullQCInputs,
    reviewer: dict,
    rubric: ReviewRubric,
    records_path: StoragePath,
    reviews: Sequence[ReviewRecord],
    receipts: list[str],
) -> FullQCReview:
    """Record an immutable source summary for persisted retained observations."""
    if len(reviews) != inputs.retained_count or len({row.task_id for row in reviews}) != inputs.retained_count:
        raise ValueError("Source review does not cover every retained task exactly once")
    result = FullQCReview(
        path=str(records_path.parent),
        source_id=inputs.source_id,
        inputs=inputs,
        reviewer=reviewer,
        rubric=asdict(rubric),
        coverage_complete=bool(reviews) and all(row.status is ReviewStatus.REVIEWED for row in reviews),
        qualification_status="pending",
        attempted_count=len(reviews),
        result=quality_result(reviews),
        top_issues=dict(
            Counter(
                str(defect)
                for row in reviews
                if row.status is ReviewStatus.REVIEWED
                and row.verdict is not None
                and row.verdict.confidence is not Confidence.LOW
                and has_review_issues(row.verdict)
                for defect in row.verdict.defects
            )
        ),
        records_path=str(records_path),
        records_sha256=file_sha256(records_path),
        batch_receipts=receipts,
    )
    # Each final summary is an immutable attempt; complete batch receipts are shared across retries.
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    return result


def review_shard(
    inputs: FullQCInputs,
    reviewer: Reviewer,
    rubric: ReviewRubric,
    batch_output_path: str,
    output_path: str,
    *,
    shard_index: int,
    shard_count: int,
    batch_offset: int,
    max_passes: int,
    worker_jobs: Sequence[str],
) -> FullQCReviewShard:
    """Review one disjoint assignment of original batches, retaining every unsuccessful attempt."""
    if not 0 <= shard_index < shard_count or max_passes < 1:
        raise ValueError("Invalid review shard assignment or retry count")
    receipts = []
    attempted = reviewed = 0
    first_statuses = Counter()
    statuses = Counter()
    for batch in contract_batches(inputs):
        if (batch_offset + batch.index) % shard_count != shard_index:
            continue
        records, path = review_batch(batch.tasks, batch.contracts, rubric, reviewer, StoragePath(batch_output_path))
        first_statuses.update(str(row.status) for row in records)
        for _ in range(1, max_passes):
            if all(row.status is ReviewStatus.REVIEWED for row in records):
                break
            records, path = review_batch(batch.tasks, batch.contracts, rubric, reviewer, StoragePath(batch_output_path))
        statuses.update(str(row.status) for row in records)
        attempted += len(records)
        reviewed += sum(row.status is ReviewStatus.REVIEWED for row in records)
        receipts.append(BatchReviewReceipt(batch_index=batch.index, path=path, sha256=file_sha256(StoragePath(path))))
    result = FullQCReviewShard(
        path=str(StoragePath(output_path) / "attempts" / uuid4().hex),
        inputs=inputs,
        reviewer=reviewer.identity,
        worker_jobs=list(worker_jobs),
        rubric=asdict(rubric),
        shard_index=shard_index,
        shard_count=shard_count,
        batch_offset=batch_offset,
        attempted_count=attempted,
        reviewed_count=reviewed,
        first_pass_status_counts=dict(first_statuses),
        status_counts=dict(statuses),
        receipts=receipts,
    )
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    return result


def assemble_shards(
    inputs: FullQCInputs,
    shards: Sequence[FullQCReviewShard],
    reviewer: dict,
    rubric: ReviewRubric,
    output_path: str,
) -> FullQCReview:
    """Assemble a source from pinned shard receipts, rejecting gaps, overlap and contract changes."""
    if not shards:
        raise ValueError("Source assembly requires review shards")
    first = shards[0]
    if len(shards) != first.shard_count or {s.shard_index for s in shards} != set(range(first.shard_count)):
        raise ValueError("Source assembly requires every distinct shard")
    receipts = {}
    for shard in shards:
        if (
            shard.inputs != inputs
            or shard.reviewer != reviewer
            or canonical_sha256(shard.rubric) != canonical_sha256(asdict(rubric))
            or shard.shard_count != first.shard_count
            or shard.batch_offset != first.batch_offset
        ):
            raise ValueError("Review shards do not share the exact source and reviewer contract")
        for receipt in shard.receipts:
            if (
                receipt.batch_index in receipts
                or (shard.batch_offset + receipt.batch_index) % shard.shard_count != shard.shard_index
            ):
                raise ValueError("Review shard batch assignments overlap or differ")
            receipts[receipt.batch_index] = receipt

    def observations():
        seen = set()
        statuses = Counter()
        for batch in contract_batches(inputs):
            if batch.index not in receipts:
                raise ValueError(f"Missing retained review batch: {batch.index}")
            receipt = receipts[batch.index]
            identity = {"contracts": batch.contracts, "rubric": asdict(rubric), "reviewer": reviewer}
            records = load_batch_reviews(receipt.path, receipt.sha256, identity)
            statuses.update(str(row.status) for row in records)
            seen.add(batch.index)
            yield batch, records, receipt.path
        if seen != set(receipts):
            raise ValueError("Review shards contain unexpected batches")
        expected_statuses = Counter()
        for shard in shards:
            expected_statuses.update(shard.status_counts)
        if statuses != expected_statuses:
            raise ValueError("Review shard status counts differ from immutable observations")

    return write_review_result(inputs, reviewer, rubric, output_path, observations())
