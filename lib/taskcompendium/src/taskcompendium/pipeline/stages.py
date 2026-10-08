# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage-backed curation stages; experiment modules bind their artifact graph."""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import msgspec
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import DEFAULT_TARGET_BUFFER_BYTES, write_jsonl_file

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import (
    CheckStatus,
    Decision,
    Disposition,
    FilterPolicy,
    QualityBasis,
    ReviewRecord,
    ReviewRubric,
    SourceRecipe,
    TaskAudit,
)
from taskcompendium.pipeline.review import (
    BASE_RUBRIC,
    DEFAULT_PROMPT_CHARACTERS,
    DEFAULT_REVIEW_MAX_ATTEMPTS,
    DEFAULT_REVIEW_MAX_TOKENS,
    DEFAULT_REVIEW_RETRY_MAX_TOKENS,
    BatchReviewer,
    DirectReviewer,
    Reviewer,
)
from taskcompendium.pipeline.review_transport import DEFAULT_MAX_BATCH_BYTES
from taskcompendium.pipeline.source_quality import (
    QualitySampleCoverage,
    SourceQualityPolicy,
    SourceQualityReport,
    SourceQualityStatus,
    merge_quality_samples,
    quality_exclusion,
    sample_quality_rows,
    source_quality_report,
    unreviewed_quality_report,
)
from taskcompendium.pipeline.sources import staged_file_rows, staged_files, staged_inputs
from taskcompendium.pipeline.transforms import (
    deduplicate_group,
    filter_row,
    is_accepted,
    normalize_row,
    public_group_key,
    review_record,
    source_locator_order,
)
from taskcompendium.pipeline.verification import verify_task

AUDIT_SHARDS = 64
AUDIT_INPUT_PATTERN = "audit/*.parquet"
AUDIT_SHARD_TEMPLATE = "audit/part-{shard:05d}.parquet"
REVIEW_INPUT_PATTERN = "review-inputs/batch-*.jsonl.gz"
ACCEPTED_SHARD_TEMPLATE = "accepted/part-{shard:05d}.parquet"


class ReviewTransport(StrEnum):
    PROVIDER_BATCH = "provider-batch"
    DIRECT_CHAT = "direct-chat"
    MANUAL = "manual"


@dataclass(frozen=True)
class ReviewConfig:
    model: str
    model_revision: str
    transport: ReviewTransport = field(kw_only=True)
    prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    max_tokens: int = DEFAULT_REVIEW_MAX_TOKENS
    max_attempts: int = DEFAULT_REVIEW_MAX_ATTEMPTS
    retry_max_tokens: int = DEFAULT_REVIEW_RETRY_MAX_TOKENS
    retry_prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    base_rubric_sha256: str = hashlib.sha256(BASE_RUBRIC.encode()).hexdigest()
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES


@dataclass(frozen=True)
class AuditExecution:
    """Execution choices and transport; batch size determines the resumable layout."""

    max_workers: int = 1
    review_batch_size: int = 100
    review_input_bytes: int = DEFAULT_TARGET_BUFFER_BYTES
    reviewer: Reviewer | None = None
    worker_resources: ResourceConfig | None = None
    review_task_resources: ResourceConfig | None = None


def _write_json(path: StoragePath, value: Any) -> None:
    with path.open("wt", auto_mkdir=True) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_json(path: StoragePath) -> Any:
    with path.open("rt") as stream:
        return json.load(stream)


def persist_evidence(local_path: Path, remote_path: StoragePath) -> None:
    """Copy one attempt's evidence tree to its unique durable path."""
    copy(str(local_path), str(remote_path), recursive=True)


def _persist_review_batch(records: list[dict[str, Any]], output: StoragePath) -> str:
    # A file is a schedulable source shard. Keep each model wait independent
    # instead of pinning many sequential requests to one preparation partition.
    batch_id = canonical_sha256({"task_ids": [record["task_id"] for record in records]})
    path = output / "review-inputs" / f"batch-{batch_id}.jsonl.gz"
    if not path.exists():
        write_jsonl_file([{"records": records}], str(path))
    return str(path)


def _review_input_window(
    state: tuple[int, int], record: dict[str, Any], *, max_records: int, max_bytes: int
) -> tuple[bool, tuple[int, int]]:
    count, size = state
    # Match the persisted JSONL encoding, without keeping a second payload copy.
    next_size = size + len(msgspec.json.encode(record)) + int(count > 0)
    return count < max_records and next_size <= max_bytes, (count + 1, next_size)


def _check_prepared_audit(record: dict[str, Any]) -> dict[str, Any]:
    audit = TaskAudit.model_validate(record)
    if audit.decision is not None or audit.normalized is None:
        return record
    metrics = counters.current_stage()
    started = time.monotonic()
    try:
        checks = verify_task(audit.normalized)
    finally:
        metrics.update_counter("prepare/check_seconds", time.monotonic() - started)
    for check in checks:
        metrics.update_counter(f"prepare/check/{check.check}/{check.status.value}", 1)
    failed = [f"check:{check.check}" for check in checks if check.status == CheckStatus.FAIL]
    return audit.model_copy(
        update={
            "checks": checks,
            "decision": (
                Decision(task_id=audit.task_id, disposition=Disposition.REJECT, reasons=failed) if failed else None
            ),
        }
    ).model_dump(mode="json")


def _audit_batch(
    records: list[dict[str, Any]], rubric: ReviewRubric, reviewer: Reviewer, output_path: StoragePath
) -> Iterator[dict[str, Any]]:
    audits = [TaskAudit.model_validate(record) for record in records]
    candidates = [audit.normalized for audit in audits if audit.decision is None and audit.normalized is not None]
    if not candidates:
        yield from (audit_columns(audit) for audit in audits)
        return
    batch_id = canonical_sha256({"task_ids": [task.id for task in candidates]})
    evidence = output_path / "evidence" / batch_id / f"attempt-{uuid4().hex}"
    with TemporaryDirectory(prefix="task-curation-review-") as directory:
        local = Path(directory)
        try:
            reviews_path = local / "reviews.json"
            reviews = reviewer.review(candidates, rubric, local / "review")
            reviews_path.write_text(json.dumps([review.model_dump(mode="json") for review in reviews]))
            expected = {task.id for task in candidates}
            if len(reviews) != len(candidates) or {review.task_id for review in reviews} != expected:
                raise ValueError("Audit observations do not match the eligible task membership")
            for status, count in Counter(review.status for review in reviews).items():
                counters.current_stage().update_counter(f"review/final/{status}", count)
            reviews_by_id = {review.task_id: review for review in reviews}
            for audit in audits:
                if audit.task_id in reviews_by_id:
                    audit = audit.model_copy(
                        update={
                            "review": reviews_by_id[audit.task_id],
                        }
                    )
                yield audit_columns(audit)
        finally:
            # Each attempt retains its transport evidence, including failed attempts.
            started = time.monotonic()
            try:
                persist_evidence(local, evidence)
            finally:
                counters.current_stage().update_counter("review/evidence_seconds", time.monotonic() - started)


def _count_manifest_rows(rows: Iterator[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    quality_bases: Counter[str] = Counter()
    for row in rows:
        counts["input_rows"] += 1
        counts["normalized_rows"] += row["normalization_reason"] is None
        counts["reviewed_rows"] += row["review_status"] == "reviewed"
        if row["filter_status"] is not None:
            dispositions[row["filter_status"]] += 1
        reasons.update(row["filter_reasons"])
        if row["quality_basis"] is not None:
            quality_bases[row["quality_basis"]] += 1
    return {**counts, "dispositions": dict(dispositions), "reasons": dict(reasons), "quality_bases": dict(quality_bases)}


def _combine_manifest_counts(partials: Iterator[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    quality_bases: Counter[str] = Counter()
    for part in partials:
        counts.update({name: part[name] for name in ("input_rows", "normalized_rows", "reviewed_rows")})
        dispositions.update(part["dispositions"])
        reasons.update(part["reasons"])
        quality_bases.update(part["quality_bases"])
    return {**counts, "dispositions": dict(dispositions), "reasons": dict(reasons), "quality_bases": dict(quality_bases)}


def manifest_counts(
    path: StoragePath, context: ZephyrContext, *, telemetry: PhaseTelemetry | None = None
) -> dict[str, Any]:
    """Reduce audit row counts and decisions across completed Parquet shards."""
    dataset = (
        Dataset.from_files(str(path / AUDIT_INPUT_PATTERN))
        .load_parquet(
            columns=[
                "task_id",
                "normalization_reason",
                "review_status",
                "quality_basis",
                "filter_status",
                "filter_reasons",
            ]
        )
        .reduce(_count_manifest_rows, _combine_manifest_counts)
    )
    return execute_phase(context, dataset, telemetry=telemetry, operation="manifest_count").results[0]


def _executing_reviewer(execution: AuditExecution, review: ReviewConfig) -> Reviewer:
    reviewer = execution.reviewer
    if reviewer is None:
        raise ValueError("Audit execution requires a reviewer transport")
    if execution.max_workers < 1 or execution.review_batch_size < 1 or execution.review_input_bytes < 1:
        raise ValueError("Audit worker and batch counts must be positive")
    if isinstance(reviewer, (BatchReviewer, DirectReviewer)):
        actual = ReviewConfig(
            reviewer.model,
            reviewer.model_revision,
            reviewer.max_prompt_characters,
            reviewer.max_tokens,
            reviewer.max_attempts,
            reviewer.retry_max_tokens,
            reviewer.retry_max_prompt_characters,
            transport=ReviewTransport(reviewer.identity["transport"]),
            max_batch_bytes=reviewer.max_batch_bytes,
        )
        if actual != review:
            raise ValueError("Review configuration differs from the executing reviewer")
    return reviewer


def _prepared_records(prepared: StoragePath) -> Dataset:
    return Dataset.from_files(str(prepared / REVIEW_INPUT_PATTERN)).load_jsonl().flat_map(lambda batch: batch["records"])


def prepare_source(
    source_path: str,
    output_path: str,
    recipe: SourceRecipe,
    limit: int | None,
    execution: AuditExecution,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
    raw_rows: Dataset | None = None,
    normalized_rows: Dataset | None = None,
) -> dict[str, Any]:
    """Normalize, deduplicate, and run recipe checks before quality review."""
    if execution.max_workers < 1 or execution.review_batch_size < 1 or execution.review_input_bytes < 1:
        raise ValueError("Audit worker and batch counts must be positive")
    source = StoragePath(source_path)
    output = StoragePath(output_path)
    if raw_rows is None:
        relative_files = staged_files(str(source), recipe.source)
        selected = (
            Dataset.from_list(list(relative_files))
            .flat_map(partial(staged_file_rows, str(source), spec=recipe.source, inputs=staged_inputs(recipe.inputs)))
            .reshard(1)
        )
    else:
        selected = raw_rows
    if limit is not None:
        selected = selected.take_per_shard(limit)
    normalized = (
        normalized_rows
        if normalized_rows is not None
        else (
            selected.group_by(
                lambda row: row["locator"], reducer=lambda _key, rows: rows, num_output_shards=AUDIT_SHARDS
            ).map(partial(normalize_row, recipe=recipe))
        )
    )
    prepared = (
        normalized.group_by(
            public_group_key, reducer=deduplicate_group, sort_by=source_locator_order, num_output_shards=AUDIT_SHARDS
        )
        .map(_check_prepared_audit)
        # Persist inside each dedup reducer: reference-only resharding would
        # first collect whole audits into count-only pickle chunks. An indivisible
        # oversized audit remains lossless in its own review-input file.
        .window_by(
            partial(
                _review_input_window,
                max_records=execution.review_batch_size,
                max_bytes=execution.review_input_bytes,
            ),
            (0, len(b'{"records":[]}\n')),
        )
        .map(partial(_persist_review_batch, output=output))
    )
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(
            max_workers=execution.max_workers, resources=execution.worker_resources, name=f"prepare-{recipe.name}"
        )
    ) as context:
        # Reduce resources apply to every shuffle in a plan. Finish preparation
        # with the full worker budget before sharing workers across model waits.
        execute_phase(context, prepared, telemetry=telemetry, operation="prepare")
        counts = execute_phase(
            context,
            _prepared_records(output)
            .map(lambda record: audit_columns(TaskAudit.model_validate(record)))
            .reduce(_count_manifest_rows, _combine_manifest_counts),
            telemetry=telemetry,
            operation="manifest_count",
        ).results[0]
    manifest = {
        **counts,
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review_batch_size": execution.review_batch_size,
        "review_input_bytes": execution.review_input_bytes,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def assess_source_quality(
    prepared_path: str,
    output_path: str,
    recipe: SourceRecipe,
    review: ReviewConfig,
    policy: SourceQualityPolicy,
    execution: AuditExecution,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
    coverage: QualitySampleCoverage | None = None,
) -> SourceQualityReport:
    """Review a fixed panel of eligible tasks and persist its source-level decision."""
    rubric = recipe.rubric
    if rubric is None:
        raise ValueError("Quality review requires a rubric")
    reviewer = _executing_reviewer(execution, review)
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(
            max_workers=execution.max_workers, resources=execution.worker_resources, name=f"quality-{recipe.name}"
        )
    ) as context:
        sample = execute_phase(
            context,
            _prepared_records(prepared).reduce(
                partial(sample_quality_rows, policy=policy), partial(merge_quality_samples, policy=policy)
            ),
            telemetry=telemetry,
            operation="select",
        ).results[0]
        ids = set(sample.task_ids)
        # Sampling scatters IDs across the original batches. Repack before the
        # provider call so a large source cannot turn this panel into tiny requests.
        execute_phase(
            context,
            _prepared_records(prepared)
            .filter(lambda record: record["task_id"] in ids)
            .reshard(1)
            .window(execution.review_batch_size)
            .group_by(
                lambda batch: batch[0]["task_id"], reducer=lambda _key, batches: batches, num_output_shards=AUDIT_SHARDS
            )
            .map(partial(_persist_review_batch, output=output)),
            telemetry=telemetry,
            operation="persist_review_inputs",
        )
        execute_phase(
            context,
            Dataset.from_files(str(output / REVIEW_INPUT_PATTERN), empty_glob_ok=not ids)
            .load_jsonl()
            .map(lambda batch: batch["records"])
            .flat_map(partial(_audit_batch, rubric=rubric, reviewer=reviewer, output_path=output))
            .write_parquet(str(output / AUDIT_SHARD_TEMPLATE), schema=TASK_SCHEMA),
            map_task_resources=execution.review_task_resources,
            telemetry=telemetry,
            operation="review",
        )
        reviews = execute_phase(
            context,
            Dataset.from_files(str(output / AUDIT_INPUT_PATTERN), empty_glob_ok=not ids)
            .load_parquet()
            .map(review_record),
            telemetry=telemetry,
            operation="read_reviews",
        ).results
    sample_coverage = coverage or (
        QualitySampleCoverage.CENSUS if sample.eligible_count <= policy.sample_size else QualitySampleCoverage.SAMPLE
    )
    report = source_quality_report(sample, reviews, policy, coverage=sample_coverage)
    _write_json(output / "report.json", report.model_dump(mode="json"))
    _write_json(
        output / "manifest.json",
        {
            "source_quality": report.model_dump(mode="json"),
            "review": asdict(review),
            "reviewer": reviewer.identity,
            "rubric": asdict(rubric),
            "prepared_source": str(prepared),
        },
    )
    return report


def skip_source_review(
    prepared_path: str,
    output_path: str,
    policy: SourceQualityPolicy,
    execution: AuditExecution,
    *,
    coverage: QualitySampleCoverage,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
) -> SourceQualityReport:
    """Gate a source that declares no rubric on its conversion and check failures alone."""
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(max_workers=execution.max_workers, resources=execution.worker_resources, name="unreviewed")
    ) as context:
        sample = execute_phase(
            context,
            _prepared_records(prepared).reduce(
                partial(sample_quality_rows, policy=policy), partial(merge_quality_samples, policy=policy)
            ),
            telemetry=telemetry,
            operation="select",
        ).results[0]
    report = unreviewed_quality_report(sample, policy, coverage=coverage)
    _write_json(output / "report.json", report.model_dump(mode="json"))
    _write_json(
        output / "manifest.json",
        {"source_quality": report.model_dump(mode="json"), "review": None, "prepared_source": str(prepared)},
    )
    return report


def _complete_audit_batch(
    records: list[dict[str, Any]],
    *,
    sampled: dict[str, ReviewRecord],
    report: SourceQualityReport | None,
    quality_path: str | None,
    rubric: ReviewRubric | None,
    reviewer: Reviewer | None,
    output: StoragePath,
) -> Iterator[dict[str, Any]]:
    remainder = []
    for record in records:
        audit = TaskAudit.model_validate(record)
        if audit.task_id in sampled:
            row = audit_columns(
                audit.model_copy(
                    update={
                        "review": sampled[audit.task_id],
                        "source_quality_report": quality_path,
                    }
                )
            )
            if report is not None and report.status == SourceQualityStatus.REJECT:
                row.update(
                    quality_basis=QualityBasis.SOURCE_REJECTED.value,
                    filter_status="reject",
                    filter_reasons=["source_quality:reject"],
                )
            elif report is not None and report.status == SourceQualityStatus.INCOMPLETE:
                decision = task_decision(audit.task_id, audit.checks, review_record(row), FilterPolicy())
                if decision.disposition != Disposition.REJECT:
                    row.update(
                        quality_basis=QualityBasis.SOURCE_INCOMPLETE.value,
                        filter_status="defer",
                        filter_reasons=["source_quality:incomplete"],
                    )
            yield row
            continue
        if quality_exclusion(audit) is not None:
            yield audit_columns(audit)
            continue
        if report is None or report.status == SourceQualityStatus.FULL_REVIEW:
            remainder.append(record)
            continue
        if report.status == SourceQualityStatus.CENSUS:
            raise ValueError("A census quality report omitted an eligible task")
        disposition, basis = {
            SourceQualityStatus.UNREVIEWED: (Disposition.KEEP, QualityBasis.UNREVIEWED),
            SourceQualityStatus.TRUST: (Disposition.KEEP, QualityBasis.INFERRED_FROM_SOURCE),
            SourceQualityStatus.REJECT: (Disposition.REJECT, QualityBasis.SOURCE_REJECTED),
            SourceQualityStatus.INCOMPLETE: (Disposition.DEFER, QualityBasis.SOURCE_INCOMPLETE),
        }[report.status]
        yield audit_columns(
            audit.model_copy(
                update={
                    "quality_basis": basis,
                    "source_quality_report": quality_path,
                    "decision": Decision(
                        task_id=audit.task_id, disposition=disposition, reasons=[f"source_quality:{report.status}"]
                    ),
                }
            )
        )
    if not remainder:
        return
    if rubric is None or reviewer is None:
        raise ValueError("Reviewing remaining tasks requires a rubric and a reviewer")
    yield from _audit_batch(remainder, rubric, reviewer, output)


def audit_prepared_source(
    prepared_path: str,
    quality_path: str | None,
    output_path: str,
    recipe: SourceRecipe,
    review: ReviewConfig | None,
    execution: AuditExecution,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
) -> dict[str, Any]:
    """Reuse sampled reviews and review or classify the remaining prepared records.

    ``review=None`` requires a source without a rubric, whose eligible rows are kept unreviewed.
    """
    if (review is None) != (recipe.rubric is None):
        raise ValueError("A review configuration applies exactly to sources with a rubric")
    reviewer = _executing_reviewer(execution, review) if review is not None else None
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    report = (
        SourceQualityReport.model_validate(_read_json(StoragePath(quality_path) / "report.json"))
        if quality_path
        else None
    )
    if quality_path:
        quality_manifest = _read_json(StoragePath(quality_path) / "manifest.json")
        if StoragePath(quality_manifest["prepared_source"]) != prepared or quality_manifest["review"] != (
            asdict(review) if review is not None else None
        ):
            raise ValueError("Source quality evidence belongs to different prepared data or review configuration")
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(
            max_workers=execution.max_workers, resources=execution.worker_resources, name=f"audit-{recipe.name}"
        )
    ) as context:
        sampled = {}
        if quality_path:
            assert report is not None
            sampled_ids = report.population.task_ids if review is not None else ()
            review_rows = Dataset.from_files(
                str(StoragePath(quality_path) / AUDIT_INPUT_PATTERN), empty_glob_ok=not sampled_ids
            ).load_parquet()
            reviews = execute_phase(
                context,
                review_rows.map(review_record),
                telemetry=telemetry,
                operation="read_sample_reviews",
            ).results
            sampled = {record.task_id: record for record in reviews}
            if len(sampled) != len(reviews) or set(sampled) != set(sampled_ids):
                raise ValueError("Saved quality reviews do not match the declared sample")
        execute_phase(
            context,
            Dataset.from_files(str(prepared / REVIEW_INPUT_PATTERN))
            .load_jsonl()
            .map(lambda row: row["records"])
            .flat_map(
                partial(
                    _complete_audit_batch,
                    sampled=sampled,
                    report=report,
                    quality_path=quality_path,
                    rubric=recipe.rubric,
                    reviewer=reviewer,
                    output=output,
                )
            )
            .write_parquet(str(output / AUDIT_SHARD_TEMPLATE), schema=TASK_SCHEMA, skip_existing=True),
            map_task_resources=execution.review_task_resources,
            telemetry=telemetry,
            operation="review",
        )
        counts = manifest_counts(output, context, telemetry=telemetry)
    if counts["input_rows"] != _read_json(prepared / "manifest.json")["input_rows"]:
        raise ValueError("Quality assessment lost prepared source rows")
    manifest = {
        **counts,
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review": asdict(review) if review is not None else None,
        "reviewer": reviewer.identity if reviewer is not None else None,
        "rubric": asdict(recipe.rubric) if recipe.rubric is not None else None,
        "source_quality": report.model_dump(mode="json") if report else None,
        "prepared_source": str(prepared),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def filter_source(
    audit_path: str,
    output_path: str,
    policy: FilterPolicy,
    max_workers: int,
    worker_resources: ResourceConfig | None = None,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
) -> dict[str, Any]:
    """Keep, reject, or defer from saved observations, preserving the complete audit."""
    source = StoragePath(audit_path)
    output = StoragePath(output_path)
    annotated = (
        Dataset.from_files(str(source / AUDIT_INPUT_PATTERN))
        .load_parquet()
        .map(partial(filter_row, policy=policy))
        .write_parquet(str(output / AUDIT_SHARD_TEMPLATE), schema=TASK_SCHEMA)
    )
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(max_workers=max_workers, resources=worker_resources, name="filter-tasks")
    ) as context:
        execute_phase(context, annotated, telemetry=telemetry, operation="filter")
        accepted = (
            Dataset.from_files(str(output / AUDIT_INPUT_PATTERN))
            .load_parquet()
            .filter(is_accepted)
            .write_parquet(str(output / ACCEPTED_SHARD_TEMPLATE), schema=TASK_SCHEMA)
        )
        execute_phase(context, accepted, telemetry=telemetry, operation="accepted")
        counts = manifest_counts(output, context, telemetry=telemetry)
    manifest: dict[str, Any] = {**counts, "policy": asdict(policy), "audited_source": str(source)}
    audited_manifest = _read_json(source / "manifest.json")
    manifest["source_quality"] = audited_manifest.get("source_quality")
    if manifest["input_rows"] != audited_manifest["input_rows"]:
        raise ValueError("Filtering lost rows from the complete audit ledger")
    if sum(manifest["dispositions"].values()) != manifest.get("input_rows", 0):
        raise ValueError("Every final row must have a keep, reject, or defer decision")
    _write_json(output / "manifest.json", manifest)
    return manifest
