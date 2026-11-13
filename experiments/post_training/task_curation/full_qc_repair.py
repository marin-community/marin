# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair incomplete retained-task reviews without resampling completed observations."""

import argparse
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import batched
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from iris.client.client import iris_ctx
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import ReviewRecord, ReviewRubric, ReviewStatus
from taskcompendium.pipeline.review import ChatReviewer, Reviewer

from experiments.post_training.task_curation.full_qc import (
    LONG_REQUEST_TIMEOUT,
    REVIEW_RUBRIC,
    ReviewerVersion,
    local_reviewer,
)
from experiments.post_training.task_curation.full_qc_artifacts import FullQCReview
from experiments.post_training.task_curation.full_qc_review import (
    RECORD_SCHEMA,
    ContractBatch,
    contract_batches,
    review_batch,
    write_source_result,
)

REPAIR_MAX_TOKENS = 8192
REPAIR_RETRY_MAX_TOKENS = 16384
REPAIR_REQUEST_CONCURRENCY = 4
MAX_REPAIR_BATCH_WORKERS = 16
OVERLAY_SCHEMA = RECORD_SCHEMA.append(pa.field("reviewer", pa.large_string())).append(
    pa.field("origin", pa.large_string())
)


def repair_identity(primary: dict, repair: dict) -> dict:
    """Bind unchanged reviewer behavior to a separately identified token-budget repair."""
    budget_fields = {"max_tokens", "retry_max_tokens"}
    if {k: v for k, v in primary.items() if k not in budget_fields} != {
        k: v for k, v in repair.items() if k not in budget_fields
    }:
        raise ValueError("Repair must preserve the primary model, rubric, payload and sampling identity")
    if (repair["max_tokens"], repair["retry_max_tokens"]) != (REPAIR_MAX_TOKENS, REPAIR_RETRY_MAX_TOKENS):
        raise ValueError("Repair requires explicit 8192/16384 token budgets")
    if primary["max_tokens"] > repair["max_tokens"] or primary["retry_max_tokens"] > repair["retry_max_tokens"]:
        raise ValueError("Repair cannot reduce a primary token budget")
    return {
        "mode": "repair_overlay",
        "version": 1,
        "primary": primary,
        "repair": repair,
        "selection": "retain_primary_reviewed_else_repair",
    }


def primary_records(primary: FullQCReview, rubric: ReviewRubric) -> dict[str, dict]:
    """Validate the frozen primary observations against their prepared contract index."""
    if canonical_sha256(primary.rubric) != canonical_sha256(asdict(rubric)):
        raise ValueError("Repair rubric differs from the primary review")
    records_path = StoragePath(primary.records_path)
    if file_sha256(records_path) != primary.records_sha256:
        raise ValueError("Primary review observations changed")
    with records_path.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    records = {r["task_id"]: r for r in rows}
    index_path = StoragePath(primary.inputs.index_path)
    if file_sha256(index_path) != primary.inputs.index_sha256:
        raise ValueError("Prepared contract index changed")
    with index_path.open("rb") as stream:
        contracts = pq.read_table(stream).to_pylist()
    index = {r["task_id"]: r for r in contracts}
    expected = primary.inputs.retained_count
    if len(rows) != expected or len(records) != expected or len(contracts) != expected or set(records) != set(index):
        raise ValueError("Primary review must account for every retained contract exactly once")
    for task_id, row in records.items():
        if {k: row[k] for k in ("task_id", "task_sha256", "runtime_sha256")} != index[task_id]:
            raise ValueError("Primary review differs from the exact retained contract")
        review = ReviewRecord.model_validate_json(row["review"])
        if review.task_id != task_id:
            raise ValueError("Primary review identity differs from its retained task")
    return records


@dataclass(frozen=True)
class RepairBatchResult:
    selected: dict[str, ReviewRecord]
    repaired_task_ids: frozenset[str]
    receipt: dict[str, str] | None


def repair_batch(
    batch: ContractBatch,
    *,
    original: dict[str, dict],
    reviewer: Reviewer,
    rubric: ReviewRubric,
    output: StoragePath,
) -> RepairBatchResult:
    """Repair only incomplete contracts in one original batch."""
    selected = {task.id: ReviewRecord.model_validate_json(original[task.id]["review"]) for task in batch.tasks}
    pending = [task for task in batch.tasks if selected[task.id].status is not ReviewStatus.REVIEWED]
    if not pending:
        return RepairBatchResult(selected, frozenset(), None)
    pending_ids = frozenset(task.id for task in pending)
    contracts = [contract for contract in batch.contracts if contract["task_id"] in pending_ids]
    observations, path = review_batch(pending, contracts, rubric, reviewer, output / "repairs")
    selected.update((record.task_id, record) for record in observations)
    receipt = {"kind": "batch_reviews", "path": path, "sha256": file_sha256(StoragePath(path))}
    return RepairBatchResult(selected, pending_ids, receipt)


def repair_source(
    primary: FullQCReview,
    reviewer: Reviewer,
    rubric: ReviewRubric,
    output_path: str,
    *,
    batch_workers: int,
) -> FullQCReview:
    """Retry only incomplete primary observations and persist their separate reviewer origins."""
    if not 1 <= batch_workers <= MAX_REPAIR_BATCH_WORKERS:
        raise ValueError(f"Repair batch workers must be between one and {MAX_REPAIR_BATCH_WORKERS}")
    identity = repair_identity(primary.reviewer, reviewer.identity)
    original = primary_records(primary, rubric)
    output = StoragePath(output_path)
    attempt = output / "attempts" / uuid4().hex
    records_path = attempt / "reviews.parquet"
    primary_origin = {
        "kind": "source_records",
        "path": primary.records_path,
        "sha256": primary.records_sha256,
        "artifact": primary.path,
        "artifact_sha256": file_sha256(StoragePath(primary.path) / ".artifact.json"),
    }
    all_reviews = []
    receipts = []
    repair = partial(repair_batch, original=original, reviewer=reviewer, rubric=rubric, output=output)
    pending_ids = {
        task_id
        for task_id, row in original.items()
        if ReviewRecord.model_validate_json(row["review"]).status is not ReviewStatus.REVIEWED
    }
    pending_batches = (
        batch for batch in contract_batches(primary.inputs) if any(task.id in pending_ids for task in batch.tasks)
    )
    repaired_batches: dict[int, RepairBatchResult] = {}
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        for group in batched(pending_batches, batch_workers):
            repaired_batches.update(
                (batch.index, result) for batch, result in zip(group, executor.map(repair, group), strict=True)
            )
    with (
        records_path.open("wb", auto_mkdir=True) as stream,
        pq.ParquetWriter(stream, OVERLAY_SCHEMA, compression="zstd") as writer,
    ):
        for batch in contract_batches(primary.inputs):
            result = repaired_batches.pop(batch.index) if batch.index in repaired_batches else repair(batch)
            rows = []
            origins = {}
            for contract in batch.contracts:
                task_id = contract["task_id"]
                from_repair = task_id in result.repaired_task_ids
                origin = result.receipt if from_repair else primary_origin
                actual_reviewer = reviewer.identity if from_repair else primary.reviewer
                review_json = result.selected[task_id].model_dump_json() if from_repair else original[task_id]["review"]
                origins[task_id] = {"reviewer": actual_reviewer, "receipt": origin}
                rows.append(
                    {
                        **contract,
                        "review": review_json,
                        "reviewer": json.dumps(actual_reviewer),
                        "origin": json.dumps(origin),
                    }
                )
            receipt_path = attempt / "batches" / f"{batch.index:06}.json"
            payload = {
                "identity": {"contracts": batch.contracts, "rubric": asdict(rubric), "reviewer": identity},
                "reviews": [result.selected[task.id].model_dump(mode="json") for task in batch.tasks],
                "origins": origins,
                "primary_review": primary_origin,
                "repair_receipt": result.receipt,
            }
            with receipt_path.open("w", auto_mkdir=True) as receipt_stream:
                json.dump(payload, receipt_stream, ensure_ascii=False)
            for row in rows:
                row["batch_receipt"] = str(receipt_path)
            writer.write_table(pa.Table.from_pylist(rows, schema=OVERLAY_SCHEMA))
            all_reviews.extend(result.selected[task.id] for task in batch.tasks)
            receipts.append(str(receipt_path))
    return write_source_result(primary.inputs, identity, rubric, records_path, all_reviews, receipts)


@dataclass(frozen=True)
class OwnedServingClient:
    """Route requests across current registrations of explicitly owned serving workers."""

    serving_roots: tuple[str, ...]
    model: str

    def complete(self, body: Mapping[str, Any]) -> dict[str, Any]:
        endpoints = [
            endpoint
            for endpoint in iris_ctx().client.list_endpoints("/serve/")
            if any(endpoint.task_id.startswith(root + "/inference-") for root in self.serving_roots)
        ]
        if not endpoints:
            raise ConnectionError("Owned review worker has no current serving registration")
        for root in self.serving_roots:
            if sum(endpoint.task_id.startswith(root + "/inference-") for endpoint in endpoints) > 1:
                raise ValueError("Owned review worker has more than one current serving registration")
        if any(
            endpoint.metadata.get("kind") != "marin-serve" or endpoint.metadata.get("model") != self.model
            for endpoint in endpoints
        ):
            raise ValueError("Owned serving registration differs from the pinned reviewer model")
        endpoints.sort(key=lambda endpoint: endpoint.task_id)
        endpoint = endpoints[int(canonical_sha256(body), 16) % len(endpoints)]
        return OpenAIChatClient(endpoint.address.rstrip("/") + "/v1", "unused", LONG_REQUEST_TIMEOUT).complete(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Immutable strict source assembly after its primary pass")
    parser.add_argument("--primary-sha256", required=True, help="SHA-256 of the primary .artifact.json")
    parser.add_argument(
        "--serving-root",
        action="append",
        required=True,
        help="Owned primary worker to borrow; repeat for a serving pool",
    )
    parser.add_argument(
        "--batch-workers", type=int, required=True, help="Number of original batches repaired concurrently"
    )
    parser.add_argument("--output", required=True, help="New repair-overlay prefix, separate from primary outputs")
    parser.add_argument("--review-cache", required=True, help="Separate exact-request cache for the repair spec")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    roots = tuple(args.serving_root)
    if len(set(roots)) != len(roots):
        raise ValueError("Serving roots must be distinct")
    if not 1 <= args.batch_workers <= MAX_REPAIR_BATCH_WORKERS:
        raise ValueError(f"Repair batch workers must be between one and {MAX_REPAIR_BATCH_WORKERS}")
    configure_coreweave_s3()
    if file_sha256(StoragePath(args.primary) / ".artifact.json") != args.primary_sha256:
        raise ValueError("Pinned primary artifact changed")
    primary = FullQCReview.raw_load(args.primary)
    spec = replace(
        local_reviewer(ReviewerVersion.V2),
        max_tokens=REPAIR_MAX_TOKENS,
        retry_max_tokens=REPAIR_RETRY_MAX_TOKENS,
        prompt_budget=primary.reviewer["prompt_budget"],
        retry_prompt_budget=primary.reviewer["retry_prompt_budget"],
    )
    reviewer = ChatReviewer(
        OwnedServingClient(roots, spec.model),
        spec,
        query_cache_root=args.review_cache,
        max_concurrent=REPAIR_REQUEST_CONCURRENCY,
    )
    identity = repair_identity(primary.reviewer, reviewer.identity)
    rows = primary_records(primary, REVIEW_RUBRIC)
    pending = sum(
        ReviewRecord.model_validate_json(row["review"]).status is not ReviewStatus.REVIEWED for row in rows.values()
    )
    print(
        json.dumps(
            {
                "source_id": primary.source_id,
                "retained": len(rows),
                "pending": pending,
                "primary_identity_sha256": canonical_sha256(primary.reviewer),
                "repair_identity_sha256": canonical_sha256(reviewer.identity),
                "repair_cache_revision": spec.cache_revision,
                "composite_identity_sha256": canonical_sha256(identity),
            }
        ),
        flush=True,
    )
    if not args.run:
        return
    result = repair_source(primary, reviewer, REVIEW_RUBRIC, args.output, batch_workers=args.batch_workers)
    deployment = StoragePath(result.path) / "deployment.json"
    with deployment.open("w", auto_mkdir=True) as stream:
        json.dump(
            {
                "serving_roots": roots,
                "batch_workers": args.batch_workers,
                "request_concurrency_per_batch": REPAIR_REQUEST_CONCURRENCY,
                "primary": args.primary,
                "primary_sha256": args.primary_sha256,
                "repair_identity": reviewer.identity,
                "repair_cache_revision": spec.cache_revision,
                "result": result.path,
            },
            stream,
            indent=2,
        )
    print(
        json.dumps(
            {
                "artifact": result.path,
                "coverage_complete": result.coverage_complete,
                "conclusive": result.result.reviewed,
                "issues": result.result.issue_count,
                "uncertain": result.result.uncertain_count,
                "unavailable": result.result.unavailable_count,
            }
        )
    )


if __name__ == "__main__":
    main()
