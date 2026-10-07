# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded raw sampling, source review, and gated conversion on an entered worker pool."""

import hashlib
import heapq
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import partial
from itertools import batched
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.execution_telemetry import SourceTelemetry, execute_phase
from taskcompendium.pipeline.inputs import SourceFiles
from taskcompendium.pipeline.models import CheckSuite, DatasetRecipe, FilterPolicy, VerificationReport
from taskcompendium.pipeline.sampling import merge_sample_rows
from taskcompendium.pipeline.source_quality import (
    SOURCE_QUALITY_REVISION,
    QualitySampleCoverage,
    SourceQualityPolicy,
    SourceQualityReport,
    SourceQualityStatus,
)
from taskcompendium.pipeline.source_verification import (
    SOURCE_VERIFICATION_REVISION,
    SourceVerificationPolicy,
    SourceVerificationStatus,
    verify_source,
)
from taskcompendium.pipeline.sources import (
    decode_staged_row,
    source_files_identity,
    staged_files,
    staged_raw_file_rows,
)
from taskcompendium.pipeline.stages import (
    AUDIT_INPUT_PATTERN,
    AuditExecution,
    ReviewConfig,
    _read_json,
    _write_json,
    assess_source_quality,
    audit_prepared_source,
    filter_source,
    prepare_source,
)
from taskcompendium.pipeline.transforms import UNBOUND_CONTROLS_REASON, normalize_row
from taskcompendium.pipeline.verification import verify_task

SOURCE_PIPELINE_REVISION = "7"
PANEL_ROWS_PER_SHARD = 16


def _answer_report(task: TaskSpec) -> VerificationReport:
    return VerificationReport(verify_task(task))


def answer_check_suite() -> CheckSuite:
    """Return the existing VerifyIT answer controls for sources without custom controls."""
    return CheckSuite("answer-controls", "1", {}, _answer_report)


class SourceProcessingMode(StrEnum):
    SAMPLE = "sample"
    FULL = "full"
    NORMALIZE_ONLY = "normalize_only"


@dataclass(frozen=True)
class SourcePipelineConfig:
    mode: SourceProcessingMode
    quality_policy: SourceQualityPolicy
    verification_policy: SourceVerificationPolicy
    review: ReviewConfig
    execution: AuditExecution
    filter_policy: FilterPolicy
    normalized_shards: int


@dataclass(frozen=True)
class SourcePipelineResult:
    hf_path: str
    normalized_path: str
    analysis_path: str
    verification_path: str
    report_path: str
    accepted_path: str
    status: str


@dataclass(frozen=True)
class RawSample:
    population_count: int
    rows: list[dict[str, Any]]


def _raw_order(row: dict[str, Any], seed: int) -> tuple[str, str]:
    locator = row["locator"]
    return hashlib.sha256(f"{seed}:{locator}".encode()).hexdigest(), locator


def sample_raw_rows(rows: Iterator[dict[str, Any]], *, size: int, seed: int) -> RawSample:
    """Select raw source locators before invoking any task converter."""
    count = 0

    def counted() -> Iterator[dict[str, Any]]:
        nonlocal count
        for row in rows:
            count += 1
            yield row

    return_rows = heapq.nsmallest(size, counted(), key=partial(_raw_order, seed=seed))
    return RawSample(count, return_rows)


def merge_raw_samples(samples: Iterator[RawSample], *, size: int, seed: int) -> RawSample:
    count, rows = merge_sample_rows(
        ((sample.population_count, sample.rows) for sample in samples), size=size, key=partial(_raw_order, seed=seed)
    )
    metrics = counters.current_stage()
    metrics.update_counter("source/sample/population_rows", count)
    metrics.update_counter("source/sample/panel_rows", len(rows))
    return RawSample(count, rows)


def _raw_dataset(source_input: str, files: SourceFiles) -> Dataset:
    return Dataset.from_list(list(staged_files(source_input, files))).flat_map(
        partial(staged_raw_file_rows, source_input, spec=files)
    )


def _binary_hash_identity(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        return {"binary_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    raise TypeError(f"Unsupported raw source value: {type(value).__name__}")


def _raw_input_sha256(data: dict[str, Any]) -> str:
    document = json.dumps(
        data, default=_binary_hash_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(document.encode()).hexdigest()


def _decode_and_normalize(
    row: dict[str, Any], *, recipe: DatasetRecipe, source_input: str, files: SourceFiles
) -> dict[str, Any]:
    metrics = counters.current_stage()
    raw_input_sha256 = _raw_input_sha256(row["data"])
    started = time.monotonic()
    try:
        decoded = decode_staged_row(row, source_input, files)
    finally:
        metrics.update_counter("source/decode/seconds", time.monotonic() - started)
        metrics.update_counter("source/decode/attempts", 1)
    metrics.update_counter("source/decode/completed_rows", 1)
    started = time.monotonic()
    try:
        result = normalize_row(decoded, recipe)
    finally:
        metrics.update_counter("source/normalize/seconds", time.monotonic() - started)
        metrics.update_counter("source/normalize/attempts", 1)
    result["audit"]["raw"]["raw_input_sha256"] = raw_input_sha256
    result["audit"]["raw"]["source_locator"] = row["locator"]
    audit = result["audit"]
    metrics.update_counter("source/normalize/completed_rows", 1)
    metrics.update_counter("source/normalize/task_rows", int(audit["normalized"] is not None))
    rejection = audit["normalization_rejection"]
    if rejection is not None:
        metrics.update_counter(f"source/normalize/{rejection['kind']}", 1)
    return result


def _source_identity(row: dict[str, Any], recipe: DatasetRecipe) -> dict[str, Any]:
    source = Source(
        dataset=recipe.source.dataset,
        revision=recipe.source.revision,
        row=f"{recipe.source.config}:{recipe.source.split}:{row['locator']}",
        importer_revision=recipe.version,
    )
    return {
        "task_id": f"{recipe.name}-{canonical_sha256(source.model_dump())}",
        "source_locator": row["locator"],
        "raw_input_sha256": _raw_input_sha256(row["data"]),
        "raw_sha256": None,
    }


def _staged_rows_with_ledger(
    relative_file: str, *, source_input: str, files: SourceFiles, recipe: DatasetRecipe, output: StoragePath
) -> Iterator[dict[str, Any]]:
    """Read selected rows without decoding and retain their original content identities."""
    metrics = counters.current_stage()
    started = time.monotonic()
    filename = hashlib.sha256(relative_file.encode()).hexdigest()
    path = output / "hf" / "locators" / f"part-{filename}.parquet"
    try:
        with path.open("wb", auto_mkdir=True) as stream:
            with pq.ParquetWriter(stream, RAW_SCHEMA) as writer:
                batch = []
                for row in staged_raw_file_rows(source_input, relative_file, files):
                    batch.append(_source_identity(row, recipe))
                    metrics.update_counter("source/raw/selected_rows", 1)
                    metrics.update_counter("source/output/hf/rows", 1)
                    metrics.update_counter(
                        "source/raw/binary_bytes",
                        sum(len(value) for value in row["data"].values() if isinstance(value, bytes)),
                    )
                    if len(batch) >= 1000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=RAW_SCHEMA))
                        batch.clear()
                    yield row
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=RAW_SCHEMA))
            metrics.update_counter("source/output/hf/parquet_bytes", stream.tell())
    finally:
        metrics.update_counter("source/raw/read_seconds", time.monotonic() - started)


def _reuse_normalized(
    row: dict[str, Any],
    *,
    recipe: DatasetRecipe,
    cached: dict[str, dict[str, Any]],
    source_input: str,
    files: SourceFiles,
) -> dict[str, Any]:
    previous = cached.get(row["locator"])
    return (
        previous
        if previous is not None
        else _decode_and_normalize(row, recipe=recipe, source_input=source_input, files=files)
    )


IDENTITY_FIELDS = [
    ("task_id", pa.string()),
    ("source_locator", pa.string()),
    ("raw_input_sha256", pa.string()),
    ("raw_sha256", pa.string()),
]
RAW_SCHEMA = pa.schema(IDENTITY_FIELDS)
NORMALIZED_COLUMNS = (
    "task_id",
    "source_dataset",
    "source_revision",
    "source_row",
    "task_json",
    "normalization_kind",
    "normalization_reason",
    "normalization_detail",
    "normalization_changes",
)
ANALYSIS_COLUMNS = tuple(
    field.name
    for field in TASK_SCHEMA
    if field.name not in {"task_json", "raw_json", "original_task_json", "cleanup_lineage_json", "checks"}
)


def _project_sidecar(row: dict[str, Any], columns: tuple[str, ...], view: str) -> dict[str, Any]:
    counters.current_stage().update_counter(f"source/output/{view}/rows", 1)
    raw = json.loads(row["raw_json"])
    return {
        **{column: row[column] for column in columns},
        "source_locator": raw["source_locator"],
        "raw_input_sha256": raw["raw_input_sha256"],
        "raw_sha256": raw["raw_sha256"],
    }


def _unprocessed_analysis(
    identity: dict[str, Any], *, recipe: DatasetRecipe, processed: frozenset[str], disposition: str
) -> Iterator[dict[str, Any]]:
    if identity["task_id"] in processed:
        return
    counters.current_stage().update_counter("source/output/analysis/rows", 1)
    yield {
        **{column: None for column in ANALYSIS_COLUMNS},
        **identity,
        "source_dataset": recipe.source.dataset,
        "source_revision": recipe.source.revision,
        "source_row": f"{recipe.source.config}:{recipe.source.split}:{identity['source_locator']}",
        "intended_use": recipe.intended_use.value,
        "filter_status": disposition,
        "filter_reasons": ["source_gate:not_expanded"],
        "review_defects": [],
        "normalization_changes": [],
        "cleanup_edits": [],
        "grader_readiness": "unverified",
    }


def _written_output(path: str, view: str) -> str:
    metrics = counters.current_stage()
    metrics.update_counter(f"source/output/{view}/parquet_bytes", StoragePath(path).size())
    metrics.update_counter(f"source/output/{view}/shards", 1)
    return path


def _quality_gate(decision: SourceQualityReport, *, census: bool, panel_size: int) -> SourceQualityReport:
    metrics = counters.current_stage()
    started = time.monotonic()
    if not census and decision.population.input_count != panel_size:
        decision = decision.model_copy(
            update={
                "status": SourceQualityStatus.INCOMPLETE,
                "reason": "The raw panel does not contain the required sampled rows; no source trust",
            }
        )
    metrics.update_counter(f"source/gate/{decision.status.value}", 1)
    metrics.update_counter("source/gate/eligible_panel_rows", len(decision.population.task_ids))
    metrics.update_counter("source/gate/seconds", time.monotonic() - started)
    return decision


def _sidecar_schema(columns: tuple[str, ...]) -> pa.Schema:
    return pa.schema([*(TASK_SCHEMA.field(column) for column in columns), *IDENTITY_FIELDS[1:]])


def _remove_completed_scratch(scratch: StoragePath) -> None:
    # Keep request/response evidence and decision manifests. Only the completed
    # procedure's redundant task payloads are disposable; failures retain them.
    for phase in ("sample", "full", "quality", "audited", "filtered", "verified"):
        for view in ("audit", "accepted", "review-inputs"):
            path = scratch / phase / view
            if path.exists():
                path.rmtree()


def _run_source_pipeline(
    recipe: DatasetRecipe,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    files: SourceFiles,
    config: SourcePipelineConfig,
    verification_suite: CheckSuite,
    *,
    previous_verification_report: str | None = None,
    previous_sample_path: str | None = None,
    telemetry: SourceTelemetry,
) -> SourcePipelineResult:
    """Review bounded raw tasks, gate full conversion, and persist joined source sidecars.

    The caller owns the entered context and reviewer transport. Reading the raw
    population scans selected raw records with bounded memory, but does not
    execute task converters or golden controls. Successful outputs retain request
    evidence and decisions while disposing of redundant intermediate task payloads.
    """
    if config.normalized_shards < 1:
        raise ValueError("Canonical normalized shard count must be positive")
    if config.quality_policy.sample_size > 100:
        raise ValueError("The initial raw task panel is capped at 100")
    output = StoragePath(output_path)
    scratch = output / "work"
    with telemetry.phase("raw_sample") as phase:
        sample = execute_phase(
            context,
            Dataset.from_list(list(staged_files(source_input, files)))
            .flat_map(
                partial(
                    _staged_rows_with_ledger,
                    source_input=source_input,
                    files=files,
                    recipe=recipe,
                    output=output,
                )
            )
            .reduce(
                partial(sample_raw_rows, size=config.quality_policy.sample_size, seed=config.quality_policy.seed),
                partial(merge_raw_samples, size=config.quality_policy.sample_size, seed=config.quality_policy.seed),
            ),
            telemetry=phase,
        ).results[0]
    if not sample.rows:
        raise ValueError("No selected source rows are available for the quality panel")
    # Runtime controls belong after the quality gate. Structural verification
    # remains part of preparation, without invoking the recipe's golden suite.
    structural_recipe = replace(recipe, policy=replace(recipe.policy, check_suite=None))
    with telemetry.phase("panel_normalize") as phase:
        normalized = execute_phase(
            context,
            Dataset.from_list(list(batched(sample.rows, PANEL_ROWS_PER_SHARD)))
            .flat_map(iter)
            .map(partial(_decode_and_normalize, recipe=structural_recipe, source_input=source_input, files=files)),
            telemetry=phase,
        ).results
    prepared = scratch / "sample"
    quality = scratch / "quality"
    with telemetry.phase("sample_prepare") as phase:
        prepare_source(
            source_input,
            str(prepared),
            structural_recipe,
            files,
            None,
            config.execution,
            context=context,
            raw_rows=Dataset.from_list(sample.rows),
            normalized_rows=Dataset.from_list(list(batched(normalized, PANEL_ROWS_PER_SHARD))).flat_map(iter),
            telemetry=phase,
        )
    census = sample.population_count <= config.quality_policy.sample_size
    if config.mode == SourceProcessingMode.NORMALIZE_ONLY:
        if previous_sample_path is None:
            raise ValueError("Normalize-only processing requires admitted sample evidence")
        decision = SourceQualityReport.model_validate(
            _read_json(StoragePath(previous_sample_path) / "analysis/report.json")
        )
        _write_json(quality / "manifest.json", {"prepared_source": str(prepared), "review": asdict(config.review)})
    else:
        with telemetry.phase("quality_review") as phase:
            decision = assess_source_quality(
                str(prepared),
                str(quality),
                structural_recipe,
                config.review,
                config.quality_policy,
                config.execution,
                context=context,
                coverage=QualitySampleCoverage.CENSUS if census else QualitySampleCoverage.RAW_SAMPLE,
                telemetry=phase,
            )
    with telemetry.phase("quality_gate") as phase:
        decision = execute_phase(
            context,
            Dataset.from_list([decision]).map(
                partial(
                    _quality_gate,
                    census=census,
                    panel_size=config.quality_policy.sample_size,
                )
            ),
            telemetry=phase,
        ).results[0]
    _write_json(quality / "report.json", decision.model_dump(mode="json"))
    expanded = config.mode in {SourceProcessingMode.FULL, SourceProcessingMode.NORMALIZE_ONLY} and decision.status in {
        SourceQualityStatus.TRUST,
        SourceQualityStatus.CENSUS,
        SourceQualityStatus.FULL_REVIEW,
    }
    if expanded and not census:
        cached = {result["locator"]: result for result in normalized}
        prepared = scratch / "full"
        with telemetry.phase("full_prepare") as phase:
            prepare_source(
                source_input,
                str(prepared),
                structural_recipe,
                files,
                None,
                config.execution,
                context=context,
                normalized_rows=_raw_dataset(source_input, files).map(
                    partial(
                        _reuse_normalized,
                        recipe=structural_recipe,
                        cached=cached,
                        source_input=source_input,
                        files=files,
                    )
                ),
                telemetry=phase,
            )
        manifest = _read_json(quality / "manifest.json")
        manifest["prepared_source"] = str(prepared)
        _write_json(quality / "manifest.json", manifest)
    audited, filtered, verified = (scratch / name for name in ("audited", "filtered", "verified"))
    # Rebuild incomplete audit shards while retaining exact-request journals and
    # evidence. Successful requests are reconciled by the reviewer cache.
    previous_audit = audited / "audit"
    if previous_audit.exists():
        previous_audit.rmtree()
    with telemetry.phase("audit_review") as phase:
        audit_prepared_source(
            str(prepared),
            str(quality),
            str(audited),
            structural_recipe,
            config.review,
            config.execution,
            context=context,
            telemetry=phase,
            sampled_analysis_path=(
                str(StoragePath(previous_sample_path) / "analysis")
                if config.mode == SourceProcessingMode.NORMALIZE_ONLY and previous_sample_path is not None
                else None
            ),
            unreviewed_reason=(UNBOUND_CONTROLS_REASON if config.mode == SourceProcessingMode.NORMALIZE_ONLY else None),
        )
    with telemetry.phase("audit_review_count") as phase:
        incomplete_reviews = execute_phase(
            context,
            Dataset.from_files(str(audited / AUDIT_INPUT_PATTERN))
            .load_parquet(columns=["review_status"])
            .filter(lambda row: row["review_status"] in {"invalid", "unavailable"})
            .count(),
            telemetry=phase,
        ).results[0]
    with telemetry.phase("filter") as phase:
        filter_source(
            str(audited),
            str(filtered),
            config.filter_policy,
            config.execution.max_workers,
            config.execution.worker_resources,
            context=context,
            telemetry=phase,
        )
    with telemetry.phase("verification") as phase:
        verification = verify_source(
            str(filtered),
            str(verified),
            config.verification_policy,
            verification_suite,
            config.execution.max_workers,
            config.execution.worker_resources,
            context=context,
            previous_report_path=previous_verification_report,
            telemetry=phase,
        )
    raw = Dataset.from_files(str(output / "hf/locators/*.parquet")).load_parquet()
    _write_json(
        output / "hf/manifest.json",
        {
            "source_input": source_input,
            "source": {
                "dataset": recipe.source.dataset,
                "revision": recipe.source.revision,
                "config": recipe.source.config,
                "split": recipe.source.split,
            },
            "files": source_files_identity(files),
            "staged_files": staged_files(source_input, files),
            "input_identity_sha256": canonical_sha256(
                {
                    "source_input": source_input,
                    "source": {
                        "dataset": recipe.source.dataset,
                        "revision": recipe.source.revision,
                        "config": recipe.source.config,
                        "split": recipe.source.split,
                    },
                    "files": source_files_identity(files),
                }
            ),
            "population_count": sample.population_count,
            "locator_sidecars": str(output / "hf/locators/*.parquet"),
            "raw_payloads": "Retained at the immutable source input",
            "raw_input_sha256": "Canonical source JSON with binary values represented by their SHA256 and byte size",
            "raw_sha256": "SHA256 of decoded canonical JSON; absent until the row is converted",
        },
    )
    audit = Dataset.from_files(str(verified / AUDIT_INPUT_PATTERN)).load_parquet()
    # A recovered panel can expand a formerly deferred source. Remove old
    # derived shards, including unprocessed locators, before publishing that view.
    for name in ("normalized", "analysis", "verification", "accepted"):
        previous_output = output / name
        if previous_output.exists():
            previous_output.rmtree()
    for name, columns in (("normalized", NORMALIZED_COLUMNS), ("analysis", ANALYSIS_COLUMNS)):
        projected = audit.map(partial(_project_sidecar, columns=columns, view=name))
        if name == "normalized":
            # Scatter bounds serialized bytes; reshard only moves existing
            # pickle chunks and cannot subdivide a partition of wide tasks.
            projected = projected.group_by(
                lambda row: row["task_id"],
                reducer=lambda _key, rows: rows,
                num_output_shards=config.normalized_shards,
            )
        else:
            projected = projected.reshard(config.normalized_shards)
        with telemetry.phase(f"export_{name}") as phase:
            execute_phase(
                context,
                projected.write_parquet(
                    str(output / name / "part-{shard:05d}.parquet"),
                    schema=_sidecar_schema(columns),
                ).map(partial(_written_output, view=name)),
                telemetry=phase,
            )
    if not expanded and not census:
        processed = frozenset(_source_identity(row, recipe)["task_id"] for row in sample.rows)
        with telemetry.phase("export_unprocessed_analysis") as phase:
            execute_phase(
                context,
                raw.flat_map(
                    partial(
                        _unprocessed_analysis,
                        recipe=recipe,
                        processed=processed,
                        disposition="reject" if decision.status == SourceQualityStatus.REJECT else "defer",
                    )
                )
                .write_parquet(
                    str(output / "analysis/unprocessed-{shard:05d}.parquet"), schema=_sidecar_schema(ANALYSIS_COLUMNS)
                )
                .map(partial(_written_output, view="analysis")),
                telemetry=phase,
            )
    checks_columns = ("task_id", "checks", "grader_readiness", "filter_status", "filter_reasons")
    with telemetry.phase("export_verification") as phase:
        execute_phase(
            context,
            audit.map(partial(_project_sidecar, columns=checks_columns, view="verification"))
            .write_parquet(
                str(output / "verification/part-{shard:05d}.parquet"),
                schema=_sidecar_schema(checks_columns),
            )
            .map(partial(_written_output, view="verification")),
            telemetry=phase,
        )
    with telemetry.phase("export_accepted") as phase:
        execute_phase(
            context,
            audit.filter(lambda row: row["filter_status"] == "keep")
            .map(partial(_project_sidecar, columns=NORMALIZED_COLUMNS, view="accepted"))
            .write_parquet(str(output / "accepted/part-{shard:05d}.parquet"), schema=_sidecar_schema(NORMALIZED_COLUMNS))
            .map(partial(_written_output, view="accepted")),
            telemetry=phase,
        )
    verification["verification"]["source_path"] = str(output / "normalized")
    verification["verification"]["analysis_path"] = str(output / "analysis")
    retryable_verification = verification["verification"]["counts"]["infra_error"] > 0
    if decision.status == SourceQualityStatus.INCOMPLETE or retryable_verification:
        status = "incomplete"
    elif (
        decision.status == SourceQualityStatus.REJECT
        or verification["verification"]["status"] == SourceVerificationStatus.REJECTED
    ):
        status = "gated"
    elif expanded or census:
        status = "completed"
    else:
        status = "sampled" if config.mode == SourceProcessingMode.SAMPLE else "gated"
    report = {
        "telemetry": str(output / "telemetry.json"),
        "implementation_revision": SOURCE_PIPELINE_REVISION,
        "quality_revision": SOURCE_QUALITY_REVISION,
        "verification_revision": SOURCE_VERIFICATION_REVISION,
        "normalized_shards": config.normalized_shards,
        "datasets": {name: str(output / name) for name in ("hf", "normalized", "analysis", "verification", "accepted")},
        "source_revision": recipe.source.revision,
        "recipe_revision": recipe.version,
        "status": status,
        "mode": config.mode.value,
        "source": recipe.name,
        "raw_population_count": sample.population_count,
        "raw_sample_count": len(sample.rows),
        "raw_population_census": census,
        "full_expansion": expanded,
        "incomplete_reviews": incomplete_reviews,
        "quality": decision.model_dump(mode="json"),
        "verification": verification["verification"],
        "processed_rows": verification["input_rows"],
        "unprocessed_rows": sample.population_count - verification["input_rows"],
    }
    _write_json(output / "analysis/report.json", decision.model_dump(mode="json"))
    _write_json(
        output / "analysis/manifest.json",
        {
            "telemetry": str(output / "telemetry.json"),
            "normalized_source": str(output / "normalized"),
            "quality_report": str(output / "analysis/report.json"),
            "review_evidence": [str(quality / "evidence"), str(audited / "evidence")],
        },
    )
    _write_json(output / "verification/report.json", verification["verification"])
    _write_json(
        output / "verification/manifest.json",
        {
            "telemetry": str(output / "telemetry.json"),
            "normalized_source": str(output / "normalized"),
            "report": str(output / "verification/report.json"),
            "policy": {
                "sample_size": config.verification_policy.sample_size,
                "seed": config.verification_policy.seed,
                "attempts": config.verification_policy.attempts,
                "minimum_pass_fraction": config.verification_policy.minimum_pass_fraction,
            },
        },
    )
    _write_json(output / "report.json", report)
    if status != "incomplete":
        _remove_completed_scratch(scratch)
    for phase in ("sample", "full", "quality", "audited", "filtered", "verified"):
        path = scratch / phase / "manifest.json"
        if path.exists():
            manifest = _read_json(path)
            manifest.pop("prepared_source", None)
            manifest.pop("audited_source", None)
            manifest["normalized_source"] = str(output / "normalized")
            _write_json(path, manifest)
    return SourcePipelineResult(
        *(str(output / name) for name in ("hf", "normalized", "analysis", "verification", "report.json", "accepted")),
        status=status,
    )


def run_source_pipeline(
    recipe: DatasetRecipe,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    files: SourceFiles,
    config: SourcePipelineConfig,
    verification_suite: CheckSuite,
    *,
    previous_verification_report: str | None = None,
    previous_sample_path: str | None = None,
    canonical_source: str,
) -> SourcePipelineResult:
    """Run one source and persist final execution counters and partial phase evidence."""
    telemetry = SourceTelemetry(canonical_source, output_path)
    with telemetry.record():
        return _run_source_pipeline(
            recipe,
            context,
            source_input,
            output_path,
            files,
            config,
            verification_suite,
            previous_verification_report=previous_verification_report,
            previous_sample_path=previous_sample_path,
            telemetry=telemetry,
        )
