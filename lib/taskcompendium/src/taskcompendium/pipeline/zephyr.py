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
from typing import Any, Literal, cast

import fsspec
import pyarrow.parquet as pq
from fsspec.core import OpenFile
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.models import (
    CheckResult,
    DatasetRecipe,
    Decision,
    Disposition,
    FilterPolicy,
    GeneratedSource,
    HFSource,
    ImportRejection,
    NormalizedTask,
    RawRow,
    ReviewRecord,
    SnapshotSource,
    TaskAudit,
)
from taskcompendium.pipeline.parquet import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.review import BASE_RUBRIC, BatchReviewer, Reviewer
from taskcompendium.pipeline.sources import source_rows
from taskcompendium.pipeline.verification import verify_task

ACQUISITION_SHARD_ROWS = 1000
GROUP_MEMORY_BYTES = 1024 * 1024
AUDIT_SHARD_ROWS = 100
OUTPUT_SHARD_ROWS = 100000


@dataclass(frozen=True)
class SourceAcquisition:
    source: HFSource | GeneratedSource | SnapshotSource
    limit: int
    sample_sha256: str | None = None


@dataclass(frozen=True)
class ReviewConfig:
    model: str
    model_revision: str
    prompt_budget: int = 128000
    max_tokens: int = 4096
    max_attempts: int = 2
    retry_max_tokens: int = 8192
    retry_prompt_budget: int = 128000
    base_rubric_sha256: str = hashlib.sha256(BASE_RUBRIC.encode()).hexdigest()


@dataclass(frozen=True)
class AuditExecution:
    """Execution choices and injected transport, excluded from artifact identity."""

    max_workers: int = 1
    review_batch_size: int = 100
    reviewer: Reviewer | None = None


def _write_json(path: str, value: Any) -> None:
    with cast(OpenFile, fsspec.open(path, "wt", auto_mkdir=True)) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_json(path: str) -> Any:
    with cast(OpenFile, fsspec.open(path, "rt")) as stream:
        return json.load(stream)


def acquire_source(acquisition: SourceAcquisition, output_path: str) -> dict[str, Any]:
    """Copy an exact bounded sample into durable, independently cached raw shards."""
    if acquisition.limit <= 0:
        raise ValueError("A positive sample limit is required")
    source = acquisition.source
    if isinstance(source, SnapshotSource) and acquisition.sample_sha256 is not None:
        digest = hashlib.sha256()
        with cast(OpenFile, fsspec.open(source.path, "rb")) as stream:
            for chunk in iter(lambda: stream.read(GROUP_MEMORY_BYTES), b""):
                digest.update(chunk)
        if digest.hexdigest() != acquisition.sample_sha256:
            raise ValueError("Snapshot bytes do not match the pinned sample digest")
    digest = hashlib.sha256()
    count = 0
    rows = iter(source_rows(source, acquisition.limit))
    while count < acquisition.limit:
        path = f"{output_path}/raw/part-{count // ACQUISITION_SHARD_ROWS:05d}.jsonl"
        with cast(OpenFile, fsspec.open(path, "wt", auto_mkdir=True)) as stream:
            for _ in range(min(ACQUISITION_SHARD_ROWS, acquisition.limit - count)):
                data = next(rows, None)
                if data is None:
                    raise ValueError(f"Source yielded {count} rows; expected {acquisition.limit}")
                record = json.dumps({"index": count, "data": data}, ensure_ascii=False, allow_nan=False) + "\n"
                stream.write(record)
                digest.update(record.encode())
                count += 1
    manifest = {"acquisition": asdict(acquisition), "input_rows": count, "raw_sample_sha256": digest.hexdigest()}
    _write_json(f"{output_path}/manifest.json", manifest)
    return manifest


def _normalize(record: dict[str, Any], recipe: DatasetRecipe) -> dict[str, Any]:
    source = Source(
        dataset=recipe.source.dataset,
        revision=recipe.source.revision,
        row=f"{recipe.source.config}:{recipe.source.split}:{record['index']}",
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
        public_key, semantic_key = semantic_digest(result, False), semantic_digest(result, True)
    return {
        "index": record["index"],
        "public_key": public_key,
        "semantic_key": semantic_key,
        "audit": audit.model_dump(mode="json"),
    }


def _public_key(record: dict[str, Any]) -> str:
    return record["public_key"]


def _index(record: dict[str, Any]) -> int:
    return record["index"]


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


def _persist_evidence(local_path: Path, remote_path: str) -> None:
    for file in local_path.rglob("*"):
        if file.is_file():
            with (
                file.open("rb") as source,
                cast(
                    OpenFile,
                    fsspec.open(f"{remote_path}/{file.relative_to(local_path).as_posix()}", "wb", auto_mkdir=True),
                ) as destination,
            ):
                shutil.copyfileobj(source, destination)


def _audit_batch(
    records: list[dict[str, Any]], recipe: DatasetRecipe, reviewer: Reviewer, output_path: str
) -> Iterator[dict[str, Any]]:
    audits = [TaskAudit.model_validate(record) for record in records]
    candidates = [audit.normalized for audit in audits if audit.decision is None and audit.normalized is not None]
    if not candidates:
        yield from (audit_columns(audit) for audit in audits)
        return
    batch_id = canonical_sha256({"task_ids": [task.id for task in candidates]})
    evidence = f"{output_path}/evidence/{batch_id}"
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
            # Raw transport records are evidence; completed Parquet shards are the cache.
            _persist_evidence(local, evidence)


def _manifest(path: str) -> dict[str, Any]:
    counts: Counter[str] = Counter(input_rows=0, normalized_rows=0, reviewed_rows=0)
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    filesystem, root = fsspec.core.url_to_fs(f"{path}/audit/*.parquet")
    for file in filesystem.glob(root):
        for row in load_parquet(
            InputFileSpec(
                filesystem.unstrip_protocol(file),
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
    source_path: str, output_path: str, recipe: DatasetRecipe, review: ReviewConfig, execution: AuditExecution
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
    acquired = _read_json(f"{source_path}/manifest.json")
    # Partition identity depends on the sample, never on the executing worker count.
    shards = max(1, ceil(acquired["input_rows"] / AUDIT_SHARD_ROWS))
    dataset = (
        Dataset.from_files(f"{source_path}/raw/*.jsonl")
        .load_jsonl()
        .map(partial(_normalize, recipe=recipe))
        .group_by(_public_key, reducer=_deduplicate, sort_by=_index, num_output_shards=shards)
        .window(execution.review_batch_size)
        .flat_map(partial(_audit_batch, recipe=recipe, reviewer=reviewer, output_path=output_path))
        .write_parquet(f"{output_path}/audit/part-{{shard:05d}}.parquet", schema=TASK_SCHEMA, skip_existing=True)
    )
    with ZephyrContext(max_workers=execution.max_workers, name=f"audit-{recipe.name}") as context:
        context.execute(dataset)
    manifest = {
        **_manifest(output_path),
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "review": asdict(review),
        "reviewer": reviewer.identity,
        "rubric": asdict(recipe.rubric),
    }
    if manifest.get("input_rows", 0) != acquired["input_rows"]:
        raise ValueError("Audit ledger does not account for every acquired source row")
    _write_json(f"{output_path}/manifest.json", manifest)
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
    annotated = (
        Dataset.from_files(f"{audit_path}/audit/*.parquet")
        .load_parquet()
        .map(partial(_filter_row, policy=policy))
        .write_parquet(f"{output_path}/audit/part-{{shard:05d}}.parquet", schema=TASK_SCHEMA)
    )
    with ZephyrContext(name="filter-tasks") as context:
        context.execute(annotated)
        accepted = (
            Dataset.from_files(f"{output_path}/audit/*.parquet")
            .load_parquet()
            .filter(_is_accepted)
            .write_parquet(f"{output_path}/accepted/part-{{shard:05d}}.parquet", schema=TASK_SCHEMA)
        )
        context.execute(accepted)
    manifest: dict[str, Any] = {**_manifest(output_path), "policy": asdict(policy), "audited_source": audit_path}
    audited_manifest = _read_json(f"{audit_path}/manifest.json")
    if manifest["input_rows"] != audited_manifest["input_rows"]:
        raise ValueError("Filtering lost rows from the complete audit ledger")
    if sum(manifest["dispositions"].values()) != manifest.get("input_rows", 0):
        raise ValueError("Every final row must have a keep or reject decision")
    _write_json(f"{output_path}/manifest.json", manifest)
    return manifest


def concat_sources(input_paths: Sequence[str], output_path: str, view: Literal["audit", "accepted"]) -> dict[str, Any]:
    """Stream one selected view from per-source artifacts into a merged dataset."""
    if not input_paths:
        raise ValueError("At least one source is required")
    files = []
    expected = 0
    for path in input_paths:
        manifest = _read_json(f"{path}/manifest.json")
        expected += manifest["input_rows"] if view == "audit" else manifest["dispositions"].get("keep", 0)
        filesystem, pattern = fsspec.core.url_to_fs(f"{path}/{view}/*.parquet")
        files.extend(filesystem.unstrip_protocol(file) for file in sorted(filesystem.glob(pattern)))
    dataset = (
        Dataset.from_list(files)
        .load_parquet()
        .reshard(max(1, ceil(expected / OUTPUT_SHARD_ROWS)))
        .write_parquet(f"{output_path}/data/part-{{shard:05d}}.parquet", schema=TASK_SCHEMA)
    )
    with ZephyrContext(name=f"concat-{view}") as context:
        context.execute(dataset)
    filesystem, pattern = fsspec.core.url_to_fs(f"{output_path}/data/*.parquet")
    actual = 0
    for file in filesystem.glob(pattern):
        with filesystem.open(file, "rb") as stream:
            actual += pq.ParquetFile(stream).metadata.num_rows
    if actual != expected:
        raise ValueError(f"Merged output contains {actual} rows; source manifests declare {expected}")
    manifest = {
        "input_sources": list(input_paths),
        "view": view,
        "input_rows": expected,
        "deduplication_scope": "within each source",
    }
    _write_json(f"{output_path}/manifest.json", manifest)
    return manifest
