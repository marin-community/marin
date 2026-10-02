# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage-backed curation stages; experiment modules bind their artifact graph."""

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from math import ceil
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.models import (
    CheckResult,
    DatasetRecipe,
    Decision,
    Disposition,
    FilterPolicy,
    ImportRejection,
    NormalizedTask,
    RawRow,
    ReviewRecord,
    TaskAudit,
)
from taskcompendium.pipeline.parquet import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.review import BASE_RUBRIC, DEFAULT_PROMPT_CHARACTERS, BatchReviewer, Reviewer
from taskcompendium.pipeline.sources import SourceFiles, staged_file_rows, staged_files
from taskcompendium.pipeline.verification import verify_task

GROUP_MEMORY_BYTES = 1024 * 1024
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


def _merge_record(row: dict[str, Any]) -> dict[str, Any]:
    task = TaskSpec.model_validate_json(row["task_json"]) if row["task_json"] is not None else None
    return {
        "public_key": deduplication_key(task) if task is not None else row["task_id"],
        "semantic_key": semantic_digest(task, include_reference=True) if task is not None else row["task_id"],
        "row": row,
    }


def _representative_order(record: dict[str, Any]) -> str:
    row = record["row"]
    return json.dumps(
        [
            int(row["intended_use"] != "eval"),
            row["source_dataset"],
            row["source_revision"],
            row["source_row"],
            row["task_id"],
        ]
    )


def _canonical_group(_: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Choose a deterministic accepted representative and retain every audit row."""
    references = set()
    representative = None
    has_eval = False
    with SpooledTemporaryFile(max_size=GROUP_MEMORY_BYTES, mode="w+t") as spool:
        for record in records:
            row = record["row"]
            has_eval |= row["intended_use"] == "eval"
            if row["filter_status"] == "keep":
                if len(references) < 2:
                    references.add(record["semantic_key"])
                if representative is None:
                    representative = row["task_id"]
            spool.write(json.dumps(record) + "\n")
        spool.seek(0)
        for line in spool:
            row = json.loads(line)["row"]
            if row["filter_status"] == "keep":
                reason = None
                if len(references) > 1:
                    reason = "cross_source_conflicting_verifier_contracts"
                elif has_eval and row["intended_use"] != "eval":
                    reason = "evaluation_overlap"
                elif row["task_id"] != representative:
                    reason = "cross_source_exact_duplicate"
                    row["duplicate_of"] = representative
                if reason is not None:
                    row["filter_status"] = "reject"
                    row["filter_reasons"] = [*row["filter_reasons"], reason]
            yield row


def _selected_view(row: dict[str, Any], view: str) -> bool:
    if row["filter_status"] != "keep":
        return False
    if view == "executable":
        return row["grader_readiness"] == "ready"
    return view == "accepted" or row["intended_use"] == view


def canonicalize_sources(merged_path: str, output_path: str) -> dict[str, Any]:
    """Deduplicate a merged audit, exclude evaluation overlap, and export curated views."""
    source, output = StoragePath(merged_path), StoragePath(output_path)
    expected = _read_json(source / "manifest.json")["input_rows"]
    dataset = (
        Dataset.from_files(str(source / "data/*.parquet"))
        .load_parquet()
        .map(_merge_record)
        .group_by(
            _public_key,
            reducer=_canonical_group,
            sort_by=_representative_order,
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
                .filter(partial(_selected_view, view=view))
                .write_parquet(str(output / view / "part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
            )
    manifest = {
        **_manifest(output),
        "merged_source": str(source),
        "deduplication_scope": "cross-source exact public and verifier semantics",
        "representative_policy": "evaluation first, then source dataset, revision, row and task ID",
        "conflict_policy": "reject competing accepted verifier contracts; preserve prior rejections",
    }
    if manifest["input_rows"] != expected:
        raise ValueError("Canonical merge lost source audit rows")
    _write_json(output / "manifest.json", manifest)
    return manifest


def _normalize(record: dict[str, Any], recipe: DatasetRecipe) -> dict[str, Any]:
    source = Source(
        dataset=recipe.source.dataset,
        revision=recipe.source.revision,
        row=f"{recipe.source.config}:{recipe.source.split}:{record['locator']}",
        importer_revision=recipe.version,
    )
    task_id = f"{recipe.name}-{canonical_sha256(source.model_dump())}"
    raw = {
        "task_id": task_id,
        "source": source.model_dump(),
        "raw_sha256": canonical_sha256(record["data"]),
        "data": record["data"],
    }
    result = recipe.normalize(RawRow(task_id, source, record["data"]))
    audit = TaskAudit(
        task_id=task_id,
        source=source,
        raw=raw,
        normalized=None,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=None,
        intended_use=recipe.intended_use,
    )
    public_key, semantic_key = task_id, task_id
    if isinstance(result, NormalizedTask):
        audit = audit.model_copy(update={"normalization_changes": result.changes})
        result = result.task
    if isinstance(result, ImportRejection):
        audit = audit.model_copy(
            update={
                "normalization_rejection": result,
                "decision": Decision(
                    task_id=task_id,
                    disposition=Disposition.REJECT,
                    reasons=[f"normalize:{result.reason}", result.detail],
                ),
            }
        )
    else:
        if result.id != task_id or result.source != source:
            raise ValueError("A converter must retain its supplied task identity and source provenance")
        audit = audit.model_copy(update={"normalized": result})
        public_key = deduplication_key(result)
        semantic_key = semantic_digest(result, include_reference=True)
    return {
        "locator": record["locator"],
        "public_key": public_key,
        "semantic_key": semantic_key,
        "audit": audit.model_dump(mode="json"),
    }


def _public_key(record: dict[str, Any]) -> str:
    return record["public_key"]


def _locator(record: dict[str, Any]) -> str:
    path, index = record["locator"].rsplit(":", 1)
    return f"{path}:{int(index):020d}"


def _deduplicate(_: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Cut every conflicting reference and keep the first identical row in source order."""
    # Spooling prevents a frequently repeated prompt from filling worker memory.
    references = set()
    with SpooledTemporaryFile(max_size=GROUP_MEMORY_BYTES, mode="w+t") as spool:
        for record in records:
            if len(references) < 2:
                references.add(record["semantic_key"])
            spool.write(json.dumps(record) + "\n")
        spool.seek(0)
        first_id = None
        for line in spool:
            record = json.loads(line)
            audit = record["audit"]
            if audit["decision"] is None:
                if len(references) > 1:
                    audit["decision"] = Decision(
                        task_id=audit["task_id"], disposition=Disposition.REJECT, reasons=["conflicting_references"]
                    ).model_dump(mode="json")
                elif first_id is not None:
                    audit["decision"] = Decision(
                        task_id=audit["task_id"],
                        disposition=Disposition.REJECT,
                        reasons=["exact_semantic_duplicate"],
                        duplicate_of=first_id,
                    ).model_dump(mode="json")
                else:
                    first_id = audit["task_id"]
            yield audit


def _persist_evidence(local_path: Path, remote_path: StoragePath) -> None:
    for file in local_path.rglob("*"):
        if file.is_file():
            with (
                file.open("rb") as source,
                (remote_path / file.relative_to(local_path).as_posix()).open("wb", auto_mkdir=True) as destination,
            ):
                shutil.copyfileobj(source, destination)


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
            _persist_evidence(local, evidence)


def _manifest(path: StoragePath) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for file in (path / "audit/*.parquet").glob():
        for row in load_parquet(
            InputFileSpec(
                str(file),
                columns=["normalization_reason", "review_status", "filter_status", "filter_reasons"],
            )
        ):
            counts["input_rows"] += 1
            counts["normalized_rows"] += row["normalization_reason"] is None
            counts["reviewed_rows"] += row["review_status"] == "reviewed"
            if row["filter_status"] is not None:
                dispositions[row["filter_status"]] += 1
            reasons.update(row["filter_reasons"])
    return {**counts, "dispositions": dict(dispositions), "reasons": dict(reasons)}


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
        .map(partial(_normalize, recipe=recipe))
        .group_by(_public_key, reducer=_deduplicate, sort_by=_locator, num_output_shards=AUDIT_SHARDS)
        .window(execution.review_batch_size)
        .flat_map(partial(_audit_batch, recipe=recipe, reviewer=reviewer, output_path=output))
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA, skip_existing=True)
    )
    with ZephyrContext(max_workers=execution.max_workers, name=f"audit-{recipe.name}") as context:
        context.execute(dataset)
    manifest = {
        **_manifest(output),
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review": asdict(review),
        "reviewer": reviewer.identity,
        "rubric": asdict(recipe.rubric),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _filter_row(row: dict[str, Any], policy: FilterPolicy) -> dict[str, Any]:
    if (
        row["normalization_reason"] is not None
        or row["duplicate_of"] is not None
        or "conflicting_references" in row["filter_reasons"]
    ):
        return row
    verdict = None
    if row["review_quality"] is not None:
        verdict = {
            "task_id": row["task_id"],
            "quality": row["review_quality"],
            "confidence": row["review_confidence"],
            "reference_status": row["review_reference_status"],
            "defects": row["review_defects"],
            "evidence": row["review_evidence"],
        }
    review = ReviewRecord.model_validate_json(
        json.dumps(
            {
                "task_id": row["task_id"],
                "status": row["review_status"] or "unavailable",
                "verdict": verdict,
                "detail": row["review_detail"] or "No quality assessment available",
            }
        )
    )
    decision = task_decision(
        row["task_id"], [CheckResult.model_validate(check) for check in row["checks"]], review, policy
    )
    return {
        **row,
        "filter_status": decision.disposition.value,
        "filter_reasons": decision.reasons,
        "duplicate_of": decision.duplicate_of,
    }


def _is_accepted(row: dict[str, Any]) -> bool:
    return row["filter_status"] == Disposition.KEEP.value


def filter_source(audit_path: str, output_path: str, policy: FilterPolicy) -> dict[str, Any]:
    """Commit binary decisions from saved observations, preserving the complete audit."""
    source = StoragePath(audit_path)
    output = StoragePath(output_path)
    annotated = (
        Dataset.from_files(str(source / "audit/*.parquet"))
        .load_parquet()
        .map(partial(_filter_row, policy=policy))
        .write_parquet(str(output / "audit/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
    )
    with ZephyrContext(name="filter-tasks") as context:
        context.execute(annotated)
        accepted = (
            Dataset.from_files(str(output / "audit/*.parquet"))
            .load_parquet()
            .filter(_is_accepted)
            .write_parquet(str(output / "accepted/part-{shard:05d}.parquet"), schema=TASK_SCHEMA)
        )
        context.execute(accepted)
    manifest: dict[str, Any] = {**_manifest(output), "policy": asdict(policy), "audited_source": str(source)}
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
    actual = 0
    for file in (output / "data/*.parquet").glob():
        with file.open("rb") as stream:
            actual += pq.ParquetFile(stream).metadata.num_rows
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
