# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage-backed curation stages; experiment modules bind their artifact graph."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.inputs import SourceFiles
from taskcompendium.pipeline.models import (
    CheckResult,
    DatasetRecipe,
    FilterPolicy,
    NormalizationChange,
    ReviewRubric,
    TaskAudit,
)
from taskcompendium.pipeline.review import BASE_RUBRIC, DEFAULT_PROMPT_CHARACTERS, BatchReviewer, Reviewer
from taskcompendium.pipeline.rewriting import BatchRewriter
from taskcompendium.pipeline.sources import staged_file_rows, staged_files
from taskcompendium.pipeline.transforms import (
    canonical_merge_record,
    canonical_representative_order,
    canonicalize_group,
    deduplicate_group,
    filter_row,
    is_accepted,
    normalize_row,
    public_group_key,
    selected_view,
    source_locator_order,
)
from taskcompendium.pipeline.verification import verify_task

AUDIT_SHARDS = 64
OUTPUT_SHARD_ROWS = 100000


@dataclass(frozen=True)
class ReviewConfig:
    model: str
    model_revision: str
    prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    max_tokens: int = 4096
    max_attempts: int = 2
    retry_max_tokens: int = 8192
    retry_prompt_budget: int = DEFAULT_PROMPT_CHARACTERS
    base_rubric_sha256: str = hashlib.sha256(BASE_RUBRIC.encode()).hexdigest()


@dataclass(frozen=True)
class AuditExecution:
    """Execution choices and injected transport, excluded from artifact identity."""

    max_workers: int = 1
    review_batch_size: int = 100
    reviewer: Reviewer | None = None


def _write_json(path: StoragePath, value: Any) -> None:
    with path.open("wt", auto_mkdir=True) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_json(path: StoragePath) -> Any:
    with path.open("rt") as stream:
        return json.load(stream)


def canonicalize_sources(merged_path: str, output_path: str) -> dict[str, Any]:
    """Deduplicate a merged audit, exclude evaluation overlap, and export curated views."""
    source, output = StoragePath(merged_path), StoragePath(output_path)
    expected = _read_json(source / "manifest.json")["input_rows"]
    dataset = (
        Dataset.from_files(str(source / "data/*.parquet"))
        .load_parquet()
        .map(canonical_merge_record)
        .group_by(
            public_group_key,
            reducer=canonicalize_group,
            sort_by=canonical_representative_order,
            num_output_shards=max(1, ceil(expected / OUTPUT_SHARD_ROWS)),
        )
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
    )
    with ZephyrContext(name="canonical-task-merge") as context:
        context.execute(dataset)
        for view in ("accepted", "train", "eval", "executable"):
            context.execute(
                Dataset.from_files(str(output / "audit/*.parquet"))
                .load_parquet()
                .filter(partial(selected_view, view=view))
                .write_parquet(str(output / view / "part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
            )
    manifest = {
        **manifest_counts(output),
        "merged_source": str(source),
        "deduplication_scope": "cross-source exact public and verifier semantics",
        "representative_policy": "evaluation first, then source dataset, revision, row and task ID",
        "conflict_policy": "reject competing accepted verifier contracts; preserve prior rejections",
    }
    if manifest["input_rows"] != expected:
        raise ValueError("Canonical merge lost source audit rows")
    _write_json(output / "manifest.json", manifest)
    return manifest


def persist_evidence(local_path: Path, remote_path: StoragePath) -> None:
    """Copy one attempt's evidence tree to its unique durable path."""
    copy(str(local_path), str(remote_path), recursive=True)


def _audit_batch(
    records: list[dict[str, Any]], recipe: DatasetRecipe, reviewer: Reviewer, output_path: StoragePath
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
            checks_path = local / "checks.json"
            checks = {}
            for task in candidates:
                if recipe.check_suite is None:
                    report_checks, rollouts = verify_task(task), ()
                else:
                    report = recipe.check_suite.run(task)
                    report_checks, rollouts = report.checks, report.rollouts
                checks[task.id] = {
                    "checks": [check.model_dump(mode="json") for check in report_checks],
                    "rollouts": [rollout.model_dump(mode="json") for rollout in rollouts],
                }
            checks_path.write_text(json.dumps(checks))
            reviews_path = local / "reviews.json"
            reviews = reviewer.review(candidates, recipe.rubric, local / "review")
            reviews_path.write_text(json.dumps([review.model_dump(mode="json") for review in reviews]))
            expected = {task.id for task in candidates}
            if (
                len(reviews) != len(candidates)
                or {review.task_id for review in reviews} != expected
                or set(checks) != expected
            ):
                raise ValueError("Audit observations do not match the eligible task membership")
            reviews_by_id = {review.task_id: review for review in reviews}
            for audit in audits:
                if audit.task_id in checks:
                    audit = audit.model_copy(
                        update={
                            "checks": [CheckResult.model_validate(check) for check in checks[audit.task_id]["checks"]],
                            "review": reviews_by_id[audit.task_id],
                        }
                    )
                yield audit_columns(audit)
        finally:
            # Each attempt retains its transport evidence, including failed attempts.
            persist_evidence(local, evidence)


def _count_manifest_rows(rows: Iterator[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0, rewritten_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        counts["input_rows"] += 1
        counts["normalized_rows"] += row["normalization_reason"] is None
        counts["reviewed_rows"] += row["review_status"] == "reviewed"
        counts["rewritten_rows"] += row["parent_id"] is not None and row["task_id"] != row["parent_id"]
        if row["filter_status"] is not None:
            dispositions[row["filter_status"]] += 1
        reasons.update(row["filter_reasons"])
    return {**counts, "dispositions": dict(dispositions), "reasons": dict(reasons)}


def _combine_manifest_counts(partials: Iterator[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0, rewritten_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for part in partials:
        counts.update(
            {name: part[name] for name in ("input_rows", "normalized_rows", "reviewed_rows", "rewritten_rows")}
        )
        dispositions.update(part["dispositions"])
        reasons.update(part["reasons"])
    return {**counts, "dispositions": dict(dispositions), "reasons": dict(reasons)}


def manifest_counts(path: StoragePath) -> dict[str, Any]:
    """Reduce audit row counts and decisions across completed Parquet shards."""
    dataset = (
        Dataset.from_files(str(path / "audit/*.parquet"))
        .load_parquet(
            columns=[
                "task_id",
                "parent_id",
                "normalization_reason",
                "review_status",
                "filter_status",
                "filter_reasons",
            ]
        )
        .reduce(_count_manifest_rows, _combine_manifest_counts)
    )
    with ZephyrContext(name="task-audit-manifest") as context:
        return context.execute(dataset).results[0]


def audit_source(
    source_path: str,
    output_path: str,
    recipe: DatasetRecipe,
    review: ReviewConfig,
    execution: AuditExecution,
    files: SourceFiles,
    limit: int | None,
) -> dict[str, Any]:
    """Normalize, deduplicate, verify, and review each row with Zephyr workers."""
    reviewer = execution.reviewer
    if reviewer is None:
        raise ValueError("Audit execution requires a reviewer transport")
    if execution.max_workers < 1 or execution.review_batch_size < 1:
        raise ValueError("Audit worker and batch counts must be positive")
    if isinstance(reviewer, BatchReviewer):
        actual = ReviewConfig(
            reviewer.model,
            reviewer.model_revision,
            reviewer.max_prompt_characters,
            reviewer.max_tokens,
            reviewer.max_attempts,
            reviewer.retry_max_tokens,
            reviewer.retry_max_prompt_characters,
        )
        if actual != review:
            raise ValueError("Review configuration differs from the executing reviewer")
    source = StoragePath(source_path)
    output = StoragePath(output_path)
    relative_files = staged_files(str(source), files)
    selected = (
        Dataset.from_list(list(relative_files)).flat_map(partial(staged_file_rows, str(source), spec=files)).reshard(1)
    )
    if limit is not None:
        selected = selected.take_per_shard(limit)
    dataset = (
        selected.reshard(AUDIT_SHARDS)
        .map(partial(normalize_row, recipe=recipe))
        .group_by(
            public_group_key, reducer=deduplicate_group, sort_by=source_locator_order, num_output_shards=AUDIT_SHARDS
        )
        .window(execution.review_batch_size)
        .flat_map(partial(_audit_batch, recipe=recipe, reviewer=reviewer, output_path=output))
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA, skip_existing=True)
    )
    with ZephyrContext(max_workers=execution.max_workers, name=f"audit-{recipe.name}") as context:
        context.execute(dataset)
    manifest = {
        **manifest_counts(output),
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review": asdict(review),
        "reviewer": reviewer.identity,
        "rubric": asdict(recipe.rubric),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def filter_source(audit_path: str, output_path: str, policy: FilterPolicy) -> dict[str, Any]:
    """Commit binary decisions from saved observations, preserving the complete audit."""
    source = StoragePath(audit_path)
    output = StoragePath(output_path)
    annotated = (
        Dataset.from_files(str(source / "audit/*.parquet"))
        .load_parquet()
        .map(partial(filter_row, policy=policy))
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
    )
    with ZephyrContext(name="filter-tasks") as context:
        context.execute(annotated)
        accepted = (
            Dataset.from_files(str(output / "audit/*.parquet"))
            .load_parquet()
            .filter(is_accepted)
            .write_parquet(str(output / "accepted/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
        )
        context.execute(accepted)
    manifest: dict[str, Any] = {**manifest_counts(output), "policy": asdict(policy), "audited_source": str(source)}
    audited_manifest = _read_json(source / "manifest.json")
    if manifest["input_rows"] != audited_manifest["input_rows"]:
        raise ValueError("Filtering lost rows from the complete audit ledger")
    if sum(manifest["dispositions"].values()) != manifest.get("input_rows", 0):
        raise ValueError("Every final row must have a keep or reject decision")
    _write_json(output / "manifest.json", manifest)
    return manifest


def concat_sources(input_paths: Sequence[str], output_path: str, view: Literal["audit", "accepted"]) -> dict[str, Any]:
    """Stream one selected view from per-source artifacts into a merged dataset."""
    if not input_paths:
        raise ValueError("At least one source is required")
    files = []
    expected = 0
    for path in input_paths:
        source = StoragePath(path)
        manifest = _read_json(source / "manifest.json")
        expected += manifest["input_rows"] if view == "audit" else manifest["dispositions"].get("keep", 0)
        files.extend(str(file) for file in sorted((source / view / "*.parquet").glob(), key=str))
    output = StoragePath(output_path)
    dataset = (
        Dataset.from_list(files)
        .load_parquet()
        .reshard(max(1, ceil(expected / OUTPUT_SHARD_ROWS)))
        .write_parquet(str(output / "data/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
    )
    with ZephyrContext(name=f"concat-{view}") as context:
        context.execute(dataset)
        actual = context.execute(
            Dataset.from_files(str(output / "data/*.parquet")).load_parquet(columns=["task_id"]).count()
        ).results[0]
    if actual != expected:
        raise ValueError(f"Merged output contains {actual} rows; source manifests declare {expected}")
    manifest = {
        "input_sources": list(input_paths),
        "view": view,
        "input_rows": expected,
        "deduplication_scope": "within each source",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _selected_rewrite_row(row: dict[str, Any], selected: frozenset[str]) -> bool:
    return row["task_id"] in selected


def _rewrite_window(
    rows: list[dict[str, Any]],
    *,
    selected: frozenset[str],
    recipe: DatasetRecipe,
    policy: FilterPolicy,
    rewrite_rubric: ReviewRubric,
    rewriter: BatchRewriter,
    reviewer: Reviewer,
    output: StoragePath,
) -> Iterator[dict[str, Any]]:
    selected_rows = [row for row in rows if row["task_id"] in selected]
    if not selected_rows:
        yield from rows
        return
    originals = {row["task_id"]: TaskSpec.model_validate_json(row["task_json"]) for row in selected_rows}
    evidence_id = canonical_sha256({"task_ids": sorted(originals)})
    evidence = output / "evidence" / evidence_id / f"attempt-{uuid4().hex}"
    with TemporaryDirectory(prefix="task-curation-rewrite-") as directory:
        work = Path(directory)
        try:
            result = rewriter.rewrite(list(originals.values()), rewrite_rubric, work)
            proposals = {record.task_id: record for record in result.records}
            lineage = {record.parent_id: record for record in result.lineage}
            candidates_by_id = {task.id: task for task in result.candidates}
            candidates = {parent: candidates_by_id[item.task_id] for parent, item in lineage.items()}
            if len(result.records) != len(originals) or set(proposals) != set(originals):
                raise ValueError("Cleanup records do not account for every selected task")
            checks = {}
            for parent, candidate in candidates.items():
                checks[parent] = (
                    verify_task(candidate) if recipe.check_suite is None else recipe.check_suite.run(candidate).checks
                )
            reviews = (
                reviewer.review(
                    list(candidates.values()),
                    recipe.rubric,
                    work / "candidate-review",
                    originals={candidate.id: originals[parent] for parent, candidate in candidates.items()},
                )
                if candidates
                else []
            )
            reviews_by_id = {record.task_id: record for record in reviews}
            if len(reviews_by_id) != len(reviews) or set(reviews_by_id) != {task.id for task in candidates.values()}:
                raise ValueError("Candidate reviews do not account for every rewritten task")
            for row in rows:
                parent = row["task_id"]
                proposal = proposals.get(parent)
                if proposal is None:
                    yield row
                    continue
                candidate = candidates.get(parent)
                if candidate is None:
                    yield {
                        **row,
                        "original_task_json": row["task_json"],
                        "parent_id": parent,
                        "cleanup_status": proposal.status.value,
                        "cleanup_action": proposal.proposal.action.value if proposal.proposal else None,
                        "cleanup_reason": proposal.proposal.reason if proposal.proposal else None,
                        "cleanup_edits": (
                            [edit.model_dump(mode="json") for edit in proposal.proposal.edits]
                            if proposal.proposal
                            else []
                        ),
                        "cleanup_detail": proposal.detail,
                    }
                    continue
                original = originals[parent]
                candidate_checks = checks[parent]
                review = reviews_by_id[candidate.id]
                audit = TaskAudit(
                    task_id=candidate.id,
                    source=original.source,
                    raw=json.loads(row["raw_json"]),
                    original=original,
                    normalized=candidate,
                    normalization_rejection=None,
                    cleanup=proposal,
                    lineage=lineage[parent].model_copy(update={"original_audit": row}),
                    checks=list(candidate_checks),
                    review=review,
                    decision=task_decision(candidate.id, candidate_checks, review, policy),
                    intended_use=recipe.intended_use,
                    normalization_changes=tuple(
                        NormalizationChange.model_validate(change) for change in row["normalization_changes"]
                    ),
                )
                yield audit_columns(audit)
        finally:
            persist_evidence(work, evidence)


def rewrite_audit_source(
    source_path: str,
    output_path: str,
    recipe: DatasetRecipe,
    policy: FilterPolicy,
    rewrite_rubric: ReviewRubric,
    rewriter: BatchRewriter,
    reviewer: Reviewer,
    selected_task_ids: tuple[str, ...],
    review_batch_size: int,
    max_workers: int,
) -> dict[str, Any]:
    """Rewrite selected audit rows in resumable Zephyr windows and export final decisions."""
    if review_batch_size < 1 or max_workers < 1:
        raise ValueError("Rewrite worker and batch counts must be positive")
    selected = frozenset(selected_task_ids)
    if len(selected) != len(selected_task_ids):
        raise ValueError("Selected rewrite task IDs must be unique")
    source, output = StoragePath(source_path), StoragePath(output_path)
    input_pattern = str(source / "audit/*.parquet")
    membership = (
        Dataset.from_files(input_pattern)
        .load_parquet(columns=["task_id", "task_json"])
        .filter(partial(_selected_rewrite_row, selected=selected))
    )
    with ZephyrContext(max_workers=max_workers, name="rewrite-selection") as context:
        selected_rows = context.execute(membership).results
    if len(selected_rows) != len(selected) or {row["task_id"] for row in selected_rows} != selected:
        raise ValueError("Selected rewrite tasks must all occur exactly once in the source audit")
    for row in selected_rows:
        if row["task_json"] is None:
            raise ValueError(f"Selected task {row['task_id']} has no normalized task")
    dataset = (
        Dataset.from_files(input_pattern)
        .load_parquet()
        .window(review_batch_size)
        .flat_map(
            partial(
                _rewrite_window,
                selected=selected,
                recipe=recipe,
                policy=policy,
                rewrite_rubric=rewrite_rubric,
                rewriter=rewriter,
                reviewer=reviewer,
                output=output,
            )
        )
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA, skip_existing=True)
    )
    with ZephyrContext(max_workers=max_workers, name=f"rewrite-{recipe.name}") as context:
        context.execute(dataset)
        context.execute(
            Dataset.from_files(str(output / "audit/*.parquet"))
            .load_parquet()
            .filter(is_accepted)
            .write_parquet(str(output / "accepted/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
        )
    manifest: dict[str, Any] = {**manifest_counts(output), "selected_rows": len(selected)}
    if manifest["input_rows"] != _read_json(source / "manifest.json")["input_rows"]:
        raise ValueError("Rewriting lost rows from the complete audit ledger")
    if sum(manifest["dispositions"].values()) != manifest["input_rows"]:
        raise ValueError("Every rewritten audit row must retain a final decision")
    _write_json(output / "manifest.json", manifest)
    return manifest
