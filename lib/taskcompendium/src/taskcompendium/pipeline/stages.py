# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage-backed curation stages; experiment modules bind their artifact graph."""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import partial
from itertools import batched
from typing import Any

import msgspec
import pyarrow as pa
from fray.types import ResourceConfig
from pydantic import TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo, format_shard_path
from zephyr.input_file import InputFileSpec
from zephyr.plan import make_windows
from zephyr.readers import load_jsonl, load_parquet
from zephyr.writers import DEFAULT_TARGET_BUFFER_BYTES, write_jsonl_file

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import (
    REJECTING_CHECK_STATUSES,
    CheckResult,
    Decision,
    Disposition,
    FilterPolicy,
    QualityBasis,
    ReviewRecord,
    ReviewRubric,
    SourceRecipe,
    TaskAudit,
)
from taskcompendium.pipeline.query_cache import CachedRequests
from taskcompendium.pipeline.review import (
    BASE_RUBRIC,
    DEFAULT_PROMPT_CHARACTERS,
    DEFAULT_REVIEW_MAX_ATTEMPTS,
    DEFAULT_REVIEW_MAX_TOKENS,
    DEFAULT_REVIEW_RETRY_MAX_TOKENS,
    BatchReviewer,
    ChatReviewer,
    ReviewBatchResult,
    Reviewer,
    review_batch_id,
    review_tasks,
)
from taskcompendium.pipeline.review_requests import DEFAULT_MAX_BATCH_BYTES
from taskcompendium.pipeline.shard_outputs import ShardOutput, write_shard_outputs
from taskcompendium.pipeline.source_quality import (
    QualitySampleCoverage,
    SourceQualityPolicy,
    SourceQualityReport,
    SourceQualityStatus,
    quality_exclusion,
    sample_quality_rows,
    source_quality_report,
    unreviewed_quality_report,
)
from taskcompendium.pipeline.sources import conversion_context, source_shards, staged_file_rows
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
REVIEW_EVIDENCE_TEMPLATE = "evidence/part-{shard:05d}.parquet"
REVIEW_EVIDENCE_SCHEMA = pa.schema([("batch_id", pa.string()), ("evidence_json", pa.string())])
REVIEW_BATCH_ADAPTER = TypeAdapter(ReviewBatchResult)
ACCEPTED_SHARD_TEMPLATE = "accepted/part-{shard:05d}.parquet"
UNAVAILABLE_REVIEW_STATUSES = frozenset({"invalid", "unavailable"})
"""Review statuses that leave a task without a usable verdict."""
COUNT_COLUMNS = ["normalization_reason", "review_status", "quality_basis", "filter_status", "filter_reasons"]


class ReviewMode(StrEnum):
    BATCH = "batch"
    CHAT = "chat"
    MANUAL = "manual"


@dataclass(frozen=True)
class ReviewConfig:
    model: str
    model_revision: str
    mode: ReviewMode = field(kw_only=True)
    prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    max_tokens: int = DEFAULT_REVIEW_MAX_TOKENS
    max_attempts: int = DEFAULT_REVIEW_MAX_ATTEMPTS
    retry_max_tokens: int = DEFAULT_REVIEW_RETRY_MAX_TOKENS
    retry_prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    base_rubric_sha256: str = hashlib.sha256(BASE_RUBRIC.encode()).hexdigest()
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES


@dataclass(frozen=True)
class AuditExecution:
    """Execution choices and review mode; batch size determines the resumable layout."""

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


def _persist_review_batch(records: list[dict[str, Any]], output: StoragePath) -> str:
    # A file is a schedulable source shard. Keep each model wait independent
    # instead of pinning many sequential requests to one preparation partition.
    path = output / "review-inputs" / f"batch-{review_batch_id(record['task_id'] for record in records)}.jsonl.gz"
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


def _review_input_batches(
    records: Iterable[dict[str, Any]], execution: AuditExecution
) -> Iterator[list[dict[str, Any]]]:
    """Window prepared records by the review batch size and the persisted input byte bound."""
    return make_windows(
        records,
        partial(
            _review_input_window,
            max_records=execution.review_batch_size,
            max_bytes=execution.review_input_bytes,
        ),
        (0, len(b'{"records":[]}\n')),
    )


def task_checks(task: TaskSpec) -> list[CheckResult]:
    """Run the grader checks of one normalized task and count their outcomes."""
    metrics = counters.current_stage()
    started = time.monotonic()
    try:
        checks = verify_task(task)
    finally:
        metrics.update_counter("prepare/check_seconds", time.monotonic() - started)
    for check in checks:
        metrics.update_counter(f"prepare/check/{check.check}/{check.status.value}", 1)
    return checks


def _with_checks(audit: TaskAudit, checks: list[CheckResult]) -> dict[str, Any]:
    failed = [f"check:{check.check}" for check in checks if check.status in REJECTING_CHECK_STATUSES]
    return audit.model_copy(
        update={
            "checks": checks,
            "decision": (
                Decision(task_id=audit.task_id, disposition=Disposition.REJECT, reasons=failed) if failed else None
            ),
        }
    ).model_dump(mode="json")


def _check_prepared_audit(record: dict[str, Any]) -> dict[str, Any]:
    audit = TaskAudit.model_validate(record)
    if audit.decision is not None or audit.normalized is None:
        return record
    return _with_checks(audit, task_checks(audit.normalized))


def _audit_batch(
    records: list[dict[str, Any]],
    rubric: ReviewRubric,
    reviewer: Reviewer,
    cached: Mapping[str, CachedRequests | None],
) -> Iterator[dict[str, Any]]:
    """Review the undecided tasks of one batch; ``cached`` holds cache entries read ahead, by batch ID."""
    audits = [TaskAudit.model_validate(record) for record in records]
    candidates = [audit.normalized for audit in audits if audit.decision is None and audit.normalized is not None]
    if not candidates:
        yield from (audit_columns(audit) for audit in audits)
        return
    batch_id = review_batch_id(task.id for task in candidates)
    result = review_tasks(candidates, rubric, reviewer, cached=cached.get(batch_id))
    reviews_by_id = {review.task_id: review for review in result.reviews}
    for index, audit in enumerate(audits):
        if audit.task_id in reviews_by_id:
            audit = audit.model_copy(
                update={
                    "review": reviews_by_id[audit.task_id],
                }
            )
        row = audit_columns(audit)
        if index == 0:
            row["review_batch"] = (batch_id, result)
        yield row


def _batch_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    batch = row.get("review_batch")
    if batch is None:
        return None
    batch_id, result = batch
    return {"batch_id": batch_id, "evidence_json": REVIEW_BATCH_ADAPTER.dump_json(result).decode()}


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "review_batch"}


@dataclass
class _ManifestTally:
    """Running manifest counts over audit rows."""

    counts: Counter[str] = field(default_factory=lambda: Counter(input_rows=0, normalized_rows=0, reviewed_rows=0))
    dispositions: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    quality_bases: Counter[str] = field(default_factory=Counter)
    unavailable_reviews: int = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.counts["input_rows"] += 1
        self.counts["normalized_rows"] += row["normalization_reason"] is None
        self.counts["reviewed_rows"] += row["review_status"] == "reviewed"
        # Invalid responses count as unavailable, as in the quality gate.
        self.unavailable_reviews += row["review_status"] in UNAVAILABLE_REVIEW_STATUSES
        if row["filter_status"] is not None:
            self.dispositions[row["filter_status"]] += 1
        self.reasons.update(row["filter_reasons"])
        if row["quality_basis"] is not None:
            self.quality_bases[row["quality_basis"]] += 1

    def counted(self, rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield ``rows`` unchanged while counting them."""
        for row in rows:
            self.add(row)
            yield row

    def manifest_counts(self) -> dict[str, Any]:
        return {
            **self.counts,
            "dispositions": dict(self.dispositions),
            "reasons": dict(self.reasons),
            "quality_bases": dict(self.quality_bases),
        }


def _count_manifest_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    tally = _ManifestTally()
    for row in rows:
        tally.add(row)
    return tally.manifest_counts()


def combine_manifest_counts(partials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum manifest counts and disposition totals returned by independent shards."""
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
        .load_parquet(columns=["task_id", *COUNT_COLUMNS])
        .reduce(_count_manifest_rows, combine_manifest_counts)
    )
    return execute_phase(context, dataset, telemetry=telemetry, operation="manifest_count").results[0]


def _executing_reviewer(execution: AuditExecution, review: ReviewConfig) -> Reviewer:
    reviewer = execution.reviewer
    if reviewer is None:
        raise ValueError("Audit execution requires a reviewer")
    _check_execution(execution)
    if isinstance(reviewer, (BatchReviewer, ChatReviewer)):
        actual = ReviewConfig(
            reviewer.model,
            reviewer.model_revision,
            reviewer.max_prompt_characters,
            reviewer.max_tokens,
            reviewer.max_attempts,
            reviewer.retry_max_tokens,
            reviewer.retry_max_prompt_characters,
            mode=ReviewMode(reviewer.identity["mode"]),
            max_batch_bytes=reviewer.max_batch_bytes,
        )
        if actual != review:
            raise ValueError("Review configuration differs from the executing reviewer")
    return reviewer


def _check_execution(execution: AuditExecution) -> None:
    if execution.max_workers < 1 or execution.review_batch_size < 1 or execution.review_input_bytes < 1:
        raise ValueError("Audit worker and batch counts must be positive")


def _execution_context(context: ZephyrContext | None, execution: AuditExecution, name: str):
    """The caller's entered pool, or a pool of ``execution``'s workers for one standalone stage."""
    if context is not None:
        return nullcontext(context)
    return ZephyrContext(max_workers=execution.max_workers, resources=execution.worker_resources, name=name)


def _read_prepared_records(prepared: StoragePath) -> list[dict[str, Any]]:
    """Read a bounded prepared panel on the driver, in review-input file order."""
    return [
        record
        for path in sorted(str(path) for path in (prepared / REVIEW_INPUT_PATTERN).glob())
        for batch in load_jsonl(path)
        for record in batch["records"]
    ]


def _prepared_manifest(
    output: StoragePath, counts: dict[str, Any], recipe: SourceRecipe, execution: AuditExecution
) -> dict[str, Any]:
    manifest = {
        **counts,
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review_batch_size": execution.review_batch_size,
        "review_input_bytes": execution.review_input_bytes,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _persist_counted_batch(records: list[dict[str, Any]], output: StoragePath) -> dict[str, Any]:
    """Persist one review-input batch and return its manifest counts."""
    _persist_review_batch(records, output)
    return _count_manifest_rows(audit_columns(TaskAudit.model_validate(record)) for record in records)


def prepare_source(
    source_path: str,
    output_path: str,
    recipe: SourceRecipe,
    limit: int | None,
    execution: AuditExecution,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
    normalized_rows: Dataset | None = None,
) -> dict[str, Any]:
    """Normalize, deduplicate, and run recipe checks before quality review.

    ``normalized_rows`` supplies rows already passed through ``normalize_row`` in place of the source.
    """
    _check_execution(execution)
    source = StoragePath(source_path)
    output = StoragePath(output_path)
    if normalized_rows is None:
        selected = (
            Dataset.from_list(list(source_shards(str(source), recipe.source)))
            .flat_map(partial(staged_file_rows, str(source), spec=recipe.source, context=conversion_context(recipe)))
            .reshard(1)
        )
        if limit is not None:
            selected = selected.take_per_shard(limit)
        normalized_rows = selected.group_by(
            lambda row: row["locator"], reducer=lambda _key, rows: rows, num_output_shards=AUDIT_SHARDS
        ).map(partial(normalize_row, recipe=recipe))
    elif limit is not None:
        raise ValueError("A row limit applies only to rows read from the source")
    prepared = (
        normalized_rows.group_by(
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
        .map(partial(_persist_counted_batch, output=output))
    )
    with _execution_context(context, execution, f"prepare-{recipe.name}") as context:
        # Reduce resources apply to every shuffle in a plan. Finish preparation
        # with the full worker budget before sharing workers across model waits.
        batch_counts = execute_phase(context, prepared, telemetry=telemetry, operation="prepare").results
    return _prepared_manifest(output, combine_manifest_counts(batch_counts), recipe, execution)


@dataclass(frozen=True)
class CheckedRow:
    """A ``normalize_row`` result and the grader checks of its task, empty when it has no task to check."""

    normalized: dict[str, Any]
    checks: list[CheckResult]


def checked_row(normalized: dict[str, Any]) -> CheckedRow:
    """Check the task of one undecided normalized row."""
    audit = normalized["audit"]
    if audit["decision"] is not None or audit["normalized"] is None:
        return CheckedRow(normalized, [])
    return CheckedRow(normalized, task_checks(TaskSpec.model_validate(audit["normalized"])))


def prepare_panel(
    rows: Sequence[CheckedRow], output_path: str, recipe: SourceRecipe, execution: AuditExecution
) -> dict[str, Any]:
    """Deduplicate a checked panel on the driver and persist it in the layout of ``prepare_source``.

    A panel holds at most a quality sample of rows, so it needs no distributed shuffle.
    """
    _check_execution(execution)
    output = StoragePath(output_path)
    checks = {row.normalized["audit"]["task_id"]: row.checks for row in rows}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in sorted((row.normalized for row in rows), key=source_locator_order):
        groups.setdefault(public_group_key(row), []).append(row)
    records = []
    for key, group in groups.items():
        for record in deduplicate_group(key, iter(group)):
            # Deduplication decides only rows that converted, so an undecided row was checked.
            if record["decision"] is None:
                record = _with_checks(TaskAudit.model_validate(record), checks[record["task_id"]])
            records.append(record)
    for batch in _review_input_batches(records, execution):
        _persist_review_batch(batch, output)
    counts = _count_manifest_rows(audit_columns(TaskAudit.model_validate(record)) for record in records)
    return _prepared_manifest(output, counts, recipe, execution)


def _review_shard(
    batches: Iterator[dict[str, Any]],
    shard: ShardInfo,
    *,
    template: str,
    rubric: ReviewRubric,
    reviewer: Reviewer,
    output: StoragePath,
    cached: Mapping[str, CachedRequests | None],
) -> Iterator[ReviewRecord]:
    """Review one panel batch, write its audit rows, and return their reviews."""
    rows = [row for batch in batches for row in _audit_batch(batch["records"], rubric, reviewer, cached)]
    write_shard_outputs(
        rows,
        shard,
        [
            ShardOutput(str(output / REVIEW_EVIDENCE_TEMPLATE), REVIEW_EVIDENCE_SCHEMA, _batch_evidence),
            ShardOutput(template, TASK_SCHEMA, _audit_row),
        ],
    )
    yield from (review_record(row) for row in rows)


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
    """Review a fixed panel of eligible tasks and persist its source-level decision.

    The driver reads the bounded prepared panel, selects the sample, and reads the review
    cache once for every batch; one execution then reviews the batches in parallel.
    """
    rubric = recipe.rubric
    if rubric is None:
        raise ValueError("Quality review requires a rubric")
    reviewer = _executing_reviewer(execution, review)
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    records = _read_prepared_records(prepared)
    sample = sample_quality_rows(iter(records), policy=policy)
    ids = set(sample.task_ids)
    # Sampling scatters IDs across the prepared batches. Repack before the provider
    # call so a large source cannot turn this panel into tiny requests.
    batches = [
        list(batch) for batch in batched((r for r in records if r["task_id"] in ids), execution.review_batch_size)
    ]
    reviews: list[ReviewRecord] = []
    if batches:
        tasks = [[TaskSpec.model_validate(record["normalized"]) for record in batch] for batch in batches]
        # One cache read covers the panel; each review batch receives its own entries.
        cached = {
            review_batch_id(task.id for task in batch): entries
            for batch, entries in zip(tasks, reviewer.read_cache(tasks, rubric), strict=True)
        }
        paths = [_persist_review_batch(batch, output) for batch in batches]
        with _execution_context(context, execution, f"quality-{recipe.name}") as context:
            reviews = execute_phase(
                context,
                Dataset.from_list(paths)
                .flat_map(load_jsonl)
                .map_shard(
                    partial(
                        _review_shard,
                        template=str(output / AUDIT_SHARD_TEMPLATE),
                        rubric=rubric,
                        reviewer=reviewer,
                        output=output,
                        cached=cached,
                    )
                ),
                map_task_resources=execution.review_task_resources,
                telemetry=telemetry,
                operation="review",
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
    prepared_path: str, output_path: str, policy: SourceQualityPolicy, *, coverage: QualitySampleCoverage
) -> SourceQualityReport:
    """Gate a source that declares no rubric on its conversion and check failures alone.

    The driver reads the bounded prepared panel.
    """
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    sample = sample_quality_rows(iter(_read_prepared_records(prepared)), policy=policy)
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
        # Without a quality report every eligible row is reviewed. A report's decision covers
        # the rows outside its panel, which are never reviewed individually.
        if report is None:
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
    yield from _audit_batch(remainder, rubric, reviewer, cached={})


def _audit_shard(
    batches: Iterator[dict[str, Any]],
    shard: ShardInfo,
    *,
    template: str,
    evidence_template: str,
    complete: Callable[[list[dict[str, Any]]], Iterator[dict[str, Any]]],
) -> Iterator[_ManifestTally]:
    """Complete one review-input file's audit rows unless an earlier attempt wrote them; count them."""
    tally = _ManifestTally()
    path = format_shard_path(template, shard.shard_idx, shard.total_shards)
    if StoragePath(path).exists():
        counters.current_stage().update_counter(counters.PARTITIONS_SKIPPED, 1)
        for row in load_parquet(InputFileSpec(path=path, columns=COUNT_COLUMNS)):
            tally.add(row)
    else:
        rows = (row for batch in batches for row in complete(batch["records"]))
        write_shard_outputs(
            tally.counted(rows),
            shard,
            [
                ShardOutput(evidence_template, REVIEW_EVIDENCE_SCHEMA, _batch_evidence),
                ShardOutput(template, TASK_SCHEMA, _audit_row),
            ],
        )
    yield tally


@dataclass(frozen=True)
class AuditOutcome:
    """The audit manifest and how many reviews are invalid or unavailable."""

    manifest: dict[str, Any]
    unavailable_reviews: int


def _sampled_reviews(quality: StoragePath, sampled_ids: Sequence[str]) -> dict[str, ReviewRecord]:
    """Read the quality panel's saved reviews on the driver and require exactly the sampled tasks."""
    columns = [field.name for field in TASK_SCHEMA if field.name.startswith("review_")]
    reviews = [
        review_record(row)
        for path in sorted(str(path) for path in (quality / AUDIT_INPUT_PATTERN).glob())
        for row in load_parquet(InputFileSpec(path=path, columns=["task_id", *columns]))
    ]
    sampled = {record.task_id: record for record in reviews}
    if len(sampled) != len(reviews) or set(sampled) != set(sampled_ids):
        raise ValueError("Saved quality reviews do not match the declared sample")
    return sampled


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
) -> AuditOutcome:
    """Reuse sampled reviews and classify the remaining prepared records by the source decision.

    Without ``quality_path``, every eligible record is reviewed. ``review=None`` requires a source
    without a rubric, whose eligible rows are kept unreviewed.
    One execution writes and counts each audit shard; a shard an earlier attempt wrote is kept.
    """
    if (review is None) != (recipe.rubric is None):
        raise ValueError("A review configuration applies exactly to sources with a rubric")
    reviewer = _executing_reviewer(execution, review) if review is not None else None
    prepared, output = StoragePath(prepared_path), StoragePath(output_path)
    report = None
    sampled = {}
    if quality_path:
        quality = StoragePath(quality_path)
        report = SourceQualityReport.model_validate(_read_json(quality / "report.json"))
        quality_manifest = _read_json(quality / "manifest.json")
        if StoragePath(quality_manifest["prepared_source"]) != prepared or quality_manifest["review"] != (
            asdict(review) if review is not None else None
        ):
            raise ValueError("Source quality evidence belongs to different prepared data or review configuration")
        sampled = _sampled_reviews(quality, report.population.task_ids if review is not None else ())
    complete = partial(
        _complete_audit_batch,
        sampled=sampled,
        report=report,
        quality_path=quality_path,
        rubric=recipe.rubric,
        reviewer=reviewer,
    )
    with _execution_context(context, execution, f"audit-{recipe.name}") as context:
        tallies = execute_phase(
            context,
            Dataset.from_files(str(prepared / REVIEW_INPUT_PATTERN))
            .load_jsonl()
            .map_shard(
                partial(
                    _audit_shard,
                    template=str(output / AUDIT_SHARD_TEMPLATE),
                    evidence_template=str(output / REVIEW_EVIDENCE_TEMPLATE),
                    complete=complete,
                )
            ),
            map_task_resources=execution.review_task_resources,
            telemetry=telemetry,
            operation="review",
        ).results
    counts = combine_manifest_counts(tally.manifest_counts() for tally in tallies)
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
    return AuditOutcome(manifest, sum(tally.unavailable_reviews for tally in tallies))


def _accepted(row: dict[str, Any]) -> dict[str, Any] | None:
    return row if is_accepted(row) else None


def write_filtered_shard(
    rows: Iterator[dict[str, Any]], shard: ShardInfo, *, output: StoragePath
) -> Iterator[dict[str, Any]]:
    """Write one shard's complete filtered audit and its accepted rows, and count them."""
    tally = _ManifestTally()
    write_shard_outputs(
        tally.counted(rows),
        shard,
        [
            ShardOutput(str(output / AUDIT_SHARD_TEMPLATE), TASK_SCHEMA, lambda row: row),
            ShardOutput(str(output / ACCEPTED_SHARD_TEMPLATE), TASK_SCHEMA, _accepted),
        ],
    )
    yield tally.manifest_counts()


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
    filtered = (
        Dataset.from_files(str(source / AUDIT_INPUT_PATTERN))
        .load_parquet()
        .map(partial(filter_row, policy=policy))
        .map_shard(partial(write_filtered_shard, output=output))
    )
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(max_workers=max_workers, resources=worker_resources, name="filter-tasks")
    ) as context:
        counts = combine_manifest_counts(
            execute_phase(context, filtered, telemetry=telemetry, operation="filter").results
        )
    manifest: dict[str, Any] = {**counts, "policy": asdict(policy), "audited_source": str(source)}
    audited_manifest = _read_json(source / "manifest.json")
    manifest["source_quality"] = audited_manifest.get("source_quality")
    if manifest["input_rows"] != audited_manifest["input_rows"]:
        raise ValueError("Filtering lost rows from the complete audit ledger")
    if sum(manifest["dispositions"].values()) != manifest.get("input_rows", 0):
        raise ValueError("Every final row must have a keep, reject, or defer decision")
    _write_json(output / "manifest.json", manifest)
    return manifest
