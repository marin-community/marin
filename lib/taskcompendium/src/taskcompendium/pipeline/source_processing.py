# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded raw sampling, source review, and gated conversion on an entered worker pool."""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from itertools import batched
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import Mode
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo, format_shard_path
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Grader, NoGrader, TaskSpec, VerifyitGrader, grades_in_process
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.controls import GradingMachines, control_suite
from taskcompendium.pipeline.execution_telemetry import TELEMETRY_FILENAME, SourceTelemetry, execute_phase
from taskcompendium.pipeline.models import Admission, Disposition, FilterPolicy, SourceRecipe, SourceStatus
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample
from taskcompendium.pipeline.shard_outputs import ShardOutput, write_shard_outputs
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
    VerificationCounts,
    verify_source,
)
from taskcompendium.pipeline.sources import (
    SourceShard,
    conversion_context,
    decode_staged_row,
    source_files_identity,
    source_shards,
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
    checked_row,
    filter_source,
    prepare_panel,
    prepare_source,
    skip_source_review,
)
from taskcompendium.pipeline.transforms import normalize_row, row_source, row_task_id

SOURCE_PIPELINE_REVISION = "9"
PANEL_ROWS_PER_SHARD = 16
OUTPUT_VIEWS = ("download", "normalize", "review", "verify", "final")
SIDECAR_SHARD = "part-{shard:05d}.parquet"
UNPROCESSED_REVIEW_TEMPLATE = "review/unprocessed-{shard:05d}.parquet"
SCRATCH_PHASES = ("sample", "full", "quality", "audited", "filtered", "verified")
VERIFY_REPORT_PATH = "verify/report.json"
"""A source's verification report, relative to its output; a later run of the source reuses its trials."""
NO_CONTROLS_REASON = "The source declares no controls"
JUDGE_GRADED_REASON = "judge grader; no control path yet"
EXPANDED_QUALITY = frozenset(
    {
        SourceQualityStatus.UNREVIEWED,
        SourceQualityStatus.TRUST,
        SourceQualityStatus.CENSUS,
        SourceQualityStatus.FULL_REVIEW,
    }
)
"""Panel decisions that let a full-mode run convert every row."""


class SourceProcessingMode(StrEnum):
    SAMPLE = "sample"
    FULL = "full"


@dataclass(frozen=True)
class SourcePipelineConfig:
    """Campaign settings shared by every source.

    ``machines`` runs sandbox grader controls; ``None`` permits only in-process graders.
    """

    mode: SourceProcessingMode
    quality_policy: SourceQualityPolicy
    verification_policy: SourceVerificationPolicy
    review: ReviewConfig
    execution: AuditExecution
    filter_policy: FilterPolicy
    normalized_shards: int
    machines: GradingMachines | None


@dataclass(frozen=True)
class SourcePipelineResult:
    download_path: str
    normalize_path: str
    review_path: str
    verify_path: str
    final_path: str
    manifest_path: str
    status: SourceStatus


@dataclass(frozen=True)
class RawSample:
    population_count: int
    rows: list[dict[str, Any]]


def _raw_order(row: dict[str, Any], seed: int) -> tuple[str, str]:
    return seeded_order(row["locator"], seed)


def sample_raw_rows(rows: Iterator[dict[str, Any]], *, size: int, seed: int) -> RawSample:
    """Select raw source locators before invoking any task converter."""
    count, selected = seeded_sample(rows, size=size, key=partial(_raw_order, seed=seed))
    return RawSample(count, selected)


def merge_raw_samples(samples: Iterator[RawSample], *, size: int, seed: int) -> RawSample:
    count, rows = merge_sample_rows(
        ((sample.population_count, sample.rows) for sample in samples), size=size, key=partial(_raw_order, seed=seed)
    )
    metrics = counters.current_stage()
    metrics.update_counter("source/sample/population_rows", count)
    metrics.update_counter("source/sample/panel_rows", len(rows))
    return RawSample(count, rows)


def _raw_dataset(source_input: str, recipe: SourceRecipe) -> Dataset:
    return Dataset.from_list(list(source_shards(source_input, recipe.source))).flat_map(
        partial(staged_raw_file_rows, source_input, spec=recipe.source, context=conversion_context(recipe))
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


def _decode_and_normalize(row: dict[str, Any], *, recipe: SourceRecipe) -> dict[str, Any]:
    metrics = counters.current_stage()
    raw_input_sha256 = _raw_input_sha256(row["data"])
    started = time.monotonic()
    try:
        decoded = decode_staged_row(row, recipe.source, conversion_context(recipe))
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


def _source_identity(row: dict[str, Any], recipe: SourceRecipe) -> dict[str, Any]:
    return {
        "task_id": row_task_id(recipe, row_source(recipe, row["locator"])),
        "source_locator": row["locator"],
        "raw_input_sha256": _raw_input_sha256(row["data"]),
        "raw_sha256": None,
    }


def _staged_rows_with_ledger(
    shard: SourceShard, *, source_input: str, recipe: SourceRecipe, output: StoragePath
) -> Iterator[dict[str, Any]]:
    """Read one shard's selected rows without decoding and retain their original content identities."""
    metrics = counters.current_stage()
    started = time.monotonic()
    filename = hashlib.sha256(shard.name.encode()).hexdigest()
    path = output / "download" / "locators" / f"part-{filename}.parquet"
    try:
        with path.open("wb", auto_mkdir=True) as stream:
            with pq.ParquetWriter(stream, RAW_SCHEMA) as writer:
                batch = []
                for row in staged_raw_file_rows(source_input, shard, recipe.source, conversion_context(recipe)):
                    batch.append(_source_identity(row, recipe))
                    metrics.update_counter("source/raw/selected_rows", 1)
                    metrics.update_counter("source/output/download/rows", 1)
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
            metrics.update_counter("source/output/download/parquet_bytes", stream.tell())
    finally:
        metrics.update_counter("source/raw/read_seconds", time.monotonic() - started)


def _reuse_normalized(row: dict[str, Any], *, recipe: SourceRecipe, cached: dict[str, dict[str, Any]]) -> dict[str, Any]:
    previous = cached.get(row["locator"])
    return previous if previous is not None else _decode_and_normalize(row, recipe=recipe)


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
REVIEW_COLUMNS = tuple(
    field.name for field in TASK_SCHEMA if field.name not in {"task_json", "raw_json", "checks", "admission"}
)
VERIFY_COLUMNS = ("task_id", "checks", "grader_readiness", "filter_status", "filter_reasons", "admission")


def _sidecar_row(row: dict[str, Any], *, columns: tuple[str, ...], view: str) -> dict[str, Any]:
    counters.current_stage().update_counter(f"source/output/{view}/rows", 1)
    return {
        **{column: row[column] for column in columns},
        **{name: row[name] for name, _ in IDENTITY_FIELDS[1:]},
    }


def _final_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row["admission"] != Admission.ADMITTED.value:
        return None
    return _sidecar_row(row, columns=NORMALIZED_COLUMNS, view="final")


def _unprocessed_review(
    identity: dict[str, Any], *, recipe: SourceRecipe, processed: frozenset[str], disposition: str
) -> Iterator[dict[str, Any]]:
    if identity["task_id"] in processed:
        return
    counters.current_stage().update_counter("source/output/review/rows", 1)
    yield {
        **{column: None for column in REVIEW_COLUMNS},
        **identity,
        "source_dataset": recipe.source.dataset,
        "source_revision": recipe.source.revision,
        "source_row": identity["source_locator"],
        "intended_use": recipe.intended_use.value,
        "filter_status": disposition,
        "filter_reasons": ["source_gate:not_expanded"],
        "review_defects": [],
        "normalization_changes": [],
        "grader_readiness": "unverified",
    }


def _judge_grader(grader: Grader) -> bool:
    return isinstance(grader, VerifyitGrader) and grader.mode == Mode.JUDGE


def row_admission(row: dict[str, Any], verification: SourceVerificationStatus) -> Admission:
    """Admit kept rows whose grader runs in process, is a judge, or passed source verification."""
    if row["filter_status"] == Disposition.REJECT.value:
        return Admission.REJECTED
    if row["filter_status"] == Disposition.DEFER.value:
        return Admission.DEFERRED
    grader = TaskSpec.model_validate_json(row["task_json"]).grader
    if isinstance(grader, NoGrader):
        return Admission.NO_GRADER
    if grades_in_process(grader) or _judge_grader(grader) or verification == SourceVerificationStatus.PASSED:
        return Admission.ADMITTED
    return Admission.UNVERIFIED


def _admit(row: dict[str, Any], *, verification: SourceVerificationStatus) -> dict[str, Any]:
    admission = row_admission(row, verification)
    counters.current_stage().update_counter(f"source/admission/{admission.value}", 1)
    return {**row, "admission": admission.value}


def _merge_admissions(parts: Iterator[Counter[str]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for part in parts:
        total.update(part)
    return total


def source_admission(counts: Mapping[str, int]) -> str:
    """Summarize whether a source reaches the final export."""
    return "admitted" if counts.get(Admission.ADMITTED.value) else "none"


def _written_output(path: str, view: str) -> str:
    metrics = counters.current_stage()
    metrics.update_counter(f"source/output/{view}/parquet_bytes", StoragePath(path).size())
    metrics.update_counter(f"source/output/{view}/shards", 1)
    return path


def _quality_gate(
    decision: SourceQualityReport, *, coverage: QualitySampleCoverage, panel_size: int
) -> SourceQualityReport:
    """Withhold source trust when a sampled raw panel lacks rows the sample should hold."""
    if coverage != QualitySampleCoverage.CENSUS and decision.population.input_count != panel_size:
        return decision.model_copy(
            update={
                "status": SourceQualityStatus.INCOMPLETE,
                "reason": "The raw panel does not contain the required sampled rows; no source trust",
            }
        )
    return decision


def _sidecar_schema(columns: tuple[str, ...]) -> pa.Schema:
    return pa.schema([*(TASK_SCHEMA.field(column) for column in columns), *IDENTITY_FIELDS[1:]])


def _skipped_verification(filtered: StoragePath, reason: str) -> dict[str, Any]:
    manifest = _read_json(filtered / "manifest.json")
    return {
        **manifest,
        "verification": {
            "status": SourceVerificationStatus.SKIPPED.value,
            "reason": reason,
            "counts": asdict(VerificationCounts()),
            "results": [],
        },
    }


@dataclass(frozen=True)
class _SourceRun:
    """One source invocation: its recipe, the caller's entered pool, the staged input and the output root."""

    recipe: SourceRecipe
    context: ZephyrContext
    source_input: str
    output: StoragePath
    config: SourcePipelineConfig
    telemetry: SourceTelemetry

    def scratch(self, phase: str) -> StoragePath:
        """The working directory of one of ``SCRATCH_PHASES``."""
        return self.output / "work" / phase


@dataclass(frozen=True)
class _Panel:
    """The bounded raw panel, its normalized rows, and whether it covers the whole population."""

    sample: RawSample
    normalized: list[dict[str, Any]]
    coverage: QualitySampleCoverage

    @property
    def census(self) -> bool:
        return self.coverage == QualitySampleCoverage.CENSUS


def _sample_panel(run: _SourceRun) -> _Panel:
    """Draw the raw panel while writing the locator ledger, then normalize, check and prepare only the panel."""
    policy = run.config.quality_policy
    with run.telemetry.phase("raw_sample") as phase:
        sample = execute_phase(
            run.context,
            Dataset.from_list(list(source_shards(run.source_input, run.recipe.source)))
            .flat_map(
                partial(_staged_rows_with_ledger, source_input=run.source_input, recipe=run.recipe, output=run.output)
            )
            .reduce(
                partial(sample_raw_rows, size=policy.sample_size, seed=policy.seed),
                partial(merge_raw_samples, size=policy.sample_size, seed=policy.seed),
            ),
            telemetry=phase,
        ).results[0]
    if not sample.rows:
        raise ValueError("No selected source rows are available for the quality panel")
    with run.telemetry.phase("panel_normalize") as phase:
        checked = execute_phase(
            run.context,
            Dataset.from_list(list(batched(sample.rows, PANEL_ROWS_PER_SHARD)))
            .flat_map(iter)
            .map(partial(_decode_and_normalize, recipe=run.recipe))
            .map(checked_row),
            telemetry=phase,
        ).results
    # The panel is a few dozen rows: deduplicating it on the driver costs less than an execution.
    with run.telemetry.phase("sample_prepare"):
        prepare_panel(checked, str(run.scratch("sample")), run.recipe, run.config.execution)
    census = sample.population_count <= policy.sample_size
    return _Panel(
        sample,
        [row.normalized for row in checked],
        QualitySampleCoverage.CENSUS if census else QualitySampleCoverage.RAW_SAMPLE,
    )


def _gate_quality(run: _SourceRun, panel: _Panel, review: ReviewConfig | None) -> SourceQualityReport:
    """Review the prepared panel, or skip review without a rubric, and decide the source's quality."""
    prepared, quality = run.scratch("sample"), run.scratch("quality")
    config = run.config
    with run.telemetry.phase("quality_review") as phase:
        if review is None:
            decision = skip_source_review(str(prepared), str(quality), config.quality_policy, coverage=panel.coverage)
        else:
            decision = assess_source_quality(
                str(prepared),
                str(quality),
                run.recipe,
                review,
                config.quality_policy,
                config.execution,
                context=run.context,
                coverage=panel.coverage,
                telemetry=phase,
            )
    decision = _quality_gate(decision, coverage=panel.coverage, panel_size=config.quality_policy.sample_size)
    _write_json(quality / "report.json", decision.model_dump(mode="json"))
    return decision


def _expand_full(run: _SourceRun, panel: _Panel) -> StoragePath:
    """Prepare every raw row, reusing the panel's normalized rows, and point the quality manifest at them."""
    cached = {result["locator"]: result for result in panel.normalized}
    prepared, quality = run.scratch("full"), run.scratch("quality")
    with run.telemetry.phase("full_prepare") as phase:
        prepare_source(
            run.source_input,
            str(prepared),
            run.recipe,
            None,
            run.config.execution,
            context=run.context,
            normalized_rows=_raw_dataset(run.source_input, run.recipe).map(
                partial(_reuse_normalized, recipe=run.recipe, cached=cached)
            ),
            telemetry=phase,
        )
    manifest = _read_json(quality / "manifest.json")
    manifest["prepared_source"] = str(prepared)
    _write_json(quality / "manifest.json", manifest)
    return prepared


def _audit(run: _SourceRun, prepared: StoragePath, review: ReviewConfig | None) -> int:
    """Review the prepared rows and return how many reviews are invalid or unavailable."""
    audited = run.scratch("audited")
    # Rebuild incomplete audit shards while retaining exact-request journals and
    # evidence. Successful requests are reconciled by the reviewer cache.
    previous_audit = audited / "audit"
    if previous_audit.exists():
        previous_audit.rmtree()
    with run.telemetry.phase("audit_review") as phase:
        return audit_prepared_source(
            str(prepared),
            str(run.scratch("quality")),
            str(audited),
            run.recipe,
            review,
            run.config.execution,
            context=run.context,
            telemetry=phase,
        ).unavailable_reviews


def _filter(run: _SourceRun) -> None:
    with run.telemetry.phase("filter") as phase:
        filter_source(
            str(run.scratch("audited")),
            str(run.scratch("filtered")),
            run.config.filter_policy,
            run.config.execution.max_workers,
            run.config.execution.worker_resources,
            context=run.context,
            telemetry=phase,
        )


def _judge_graded(panel: _Panel) -> bool:
    """Whether a verifyit judge grades every task the panel converted."""
    tasks = [row["audit"]["normalized"] for row in panel.normalized if row["audit"]["normalized"] is not None]
    return bool(tasks) and all(_judge_grader(TaskSpec.model_validate(task).grader) for task in tasks)


def _verify(
    run: _SourceRun, panel: _Panel, previous_verification_report: str | None
) -> tuple[dict[str, Any], StoragePath]:
    """Run the recipe's controls on the filtered rows; return the verified manifest and its stage directory."""
    filtered = run.scratch("filtered")
    # TODO(rl-data): judge test path. A judge control would verify judge-graded sources here
    # instead of skipping them; see the rl-data judge test path issue.
    if _judge_graded(panel):
        return _skipped_verification(filtered, JUDGE_GRADED_REASON), filtered
    if run.recipe.controls is None:
        return _skipped_verification(filtered, NO_CONTROLS_REASON), filtered
    verified = run.scratch("verified")
    with run.telemetry.phase("verification") as phase:
        verification = verify_source(
            str(filtered),
            str(verified),
            run.config.verification_policy,
            control_suite(run.recipe.controls, run.config.machines),
            run.config.execution.max_workers,
            run.config.execution.worker_resources,
            context=run.context,
            previous_report_path=previous_verification_report,
            telemetry=phase,
        )
    return verification, verified


def _write_download_manifest(run: _SourceRun, population_count: int) -> None:
    recipe, source_input, output = run.recipe, run.source_input, run.output
    source_files = source_files_identity(recipe.source)
    _write_json(
        output / "download/manifest.json",
        {
            "source_input": source_input,
            "inputs": dict(recipe.inputs),
            "files": source_files,
            "staged_files": staged_files(source_input, recipe.source),
            "input_identity_sha256": canonical_sha256(
                {"source_input": source_input, "inputs": dict(recipe.inputs), "files": source_files}
            ),
            "population_count": population_count,
            "locator_sidecars": str(output / "download/locators/*.parquet"),
            "raw_payloads": "Retained at the immutable source input",
            "raw_input_sha256": "Canonical source JSON with binary values represented by their SHA256 and byte size",
            "raw_sha256": "SHA256 of decoded canonical JSON; absent until the row is converted",
        },
    )


def _export_row(row: dict[str, Any], *, verification: SourceVerificationStatus) -> dict[str, Any]:
    """Admit one checked audit row and carry its raw identity in place of the raw payload no view publishes."""
    admitted = _admit(row, verification=verification)
    raw = json.loads(admitted.pop("raw_json"))
    return {**admitted, **{name: raw[name] for name, _ in IDENTITY_FIELDS[1:]}}


@dataclass(frozen=True)
class _CheckedAudit:
    """One checked audit file, whose admitted rows feed every derived view."""

    path: str
    verification: SourceVerificationStatus

    def export_rows(self) -> Iterator[dict[str, Any]]:
        for row in load_parquet(self.path):
            yield _export_row(row, verification=self.verification)


@dataclass(frozen=True)
class _UnprocessedLocators:
    """One locator file, whose rows outside the processed panel form one ``review/unprocessed-*`` shard."""

    path: str
    output: str
    shard: ShardInfo
    unprocessed: Callable[[dict[str, Any]], Iterator[dict[str, Any]]]

    def export_rows(self) -> Iterator[dict[str, Any]]:
        template = str(StoragePath(self.output) / UNPROCESSED_REVIEW_TEMPLATE)
        path = format_shard_path(template, self.shard.shard_idx, self.shard.total_shards)
        rows = (review for identity in load_parquet(self.path) for review in self.unprocessed(identity))
        write_parquet_file(rows, path, schema=_sidecar_schema(REVIEW_COLUMNS))
        _written_output(path, "review")
        return iter(())


def _write_export_shard(rows: Iterator[dict[str, Any]], shard: ShardInfo, *, output: str) -> Iterator[Counter[str]]:
    """Write one shard of the normalize, review, verify and final views; return its admission counts."""
    admissions: Counter[str] = Counter()

    def counted() -> Iterator[dict[str, Any]]:
        for row in rows:
            admissions[row["admission"]] += 1
            yield row

    root = StoragePath(output)
    outputs = {
        view: ShardOutput(
            str(root / view / SIDECAR_SHARD), _sidecar_schema(columns), partial(_sidecar_row, columns=columns, view=view)
        )
        for view, columns in (("normalize", NORMALIZED_COLUMNS), ("review", REVIEW_COLUMNS), ("verify", VERIFY_COLUMNS))
    }
    outputs["final"] = ShardOutput(str(root / "final" / SIDECAR_SHARD), _sidecar_schema(NORMALIZED_COLUMNS), _final_row)
    for view, path in zip(outputs, write_shard_outputs(counted(), shard, list(outputs.values())), strict=True):
        _written_output(path, view)
    yield admissions


def _export_sidecars(
    run: _SourceRun,
    panel: _Panel,
    decision: SourceQualityReport,
    checked: StoragePath,
    verification: SourceVerificationStatus,
    *,
    expanded: bool,
) -> dict[str, int]:
    """Publish the normalize, review, verify and final views in one execution; return admission counts.

    Every view takes its rows from one shuffle by task ID into the canonical normalized shards.
    Locator files of an unexpanded sample write their unprocessed review rows alongside.
    """
    output = run.output
    # A recovered panel can expand a formerly deferred source. Remove old
    # derived shards, including unprocessed locators, before publishing that view.
    for name in OUTPUT_VIEWS[1:]:
        previous_output = output / name
        if previous_output.exists():
            previous_output.rmtree()
    audits = sorted(str(path) for path in (checked / AUDIT_INPUT_PATTERN).glob())
    inputs: list[_CheckedAudit | _UnprocessedLocators] = [_CheckedAudit(path, verification) for path in audits]
    if not expanded and not panel.census:
        locators = sorted(str(path) for path in (output / "download/locators/*.parquet").glob())
        unprocessed = partial(
            _unprocessed_review,
            recipe=run.recipe,
            processed=frozenset(_source_identity(row, run.recipe)["task_id"] for row in panel.sample.rows),
            disposition="reject" if decision.status == SourceQualityStatus.REJECT else "defer",
        )
        inputs.extend(
            _UnprocessedLocators(path, str(output), ShardInfo(index, len(locators)), unprocessed)
            for index, path in enumerate(locators)
        )
    with run.telemetry.phase("export") as phase:
        shard_admissions = execute_phase(
            run.context,
            Dataset.from_list(inputs).flat_map(lambda item: item.export_rows())
            # Scatter bounds serialized bytes; reshard only moves existing
            # pickle chunks and cannot subdivide a partition of wide tasks.
            .group_by(
                lambda row: row["task_id"],
                reducer=lambda _key, rows: rows,
                num_output_shards=run.config.normalized_shards,
            ).map_shard(partial(_write_export_shard, output=str(output))),
            telemetry=phase,
            operation="export",
        ).results
    return dict(_merge_admissions(iter(shard_admissions)))


def _source_status(
    mode: SourceProcessingMode,
    quality: SourceQualityStatus,
    verification: SourceVerificationStatus,
    *,
    verification_infra_errors: int,
    processed_all: bool,
) -> SourceStatus:
    # Unavailable reviews only defer their rows. The source is incomplete when the
    # quality gate could not decide or a control trial hit an infrastructure error.
    if quality == SourceQualityStatus.INCOMPLETE or verification_infra_errors > 0:
        return SourceStatus.INCOMPLETE
    if quality == SourceQualityStatus.REJECT or verification == SourceVerificationStatus.REJECTED:
        return SourceStatus.GATED
    if processed_all:
        return SourceStatus.COMPLETED
    return SourceStatus.SAMPLED if mode == SourceProcessingMode.SAMPLE else SourceStatus.GATED


def _write_manifest(
    run: _SourceRun,
    panel: _Panel,
    decision: SourceQualityReport,
    verification: dict[str, Any],
    admissions: dict[str, int],
    *,
    expanded: bool,
    unavailable_reviews: int,
    status: SourceStatus,
) -> None:
    """Write the review and verify reports and manifests, then the source manifest that joins every view."""
    output, recipe, config = run.output, run.recipe, run.config
    report = verification["verification"]
    telemetry = str(output / TELEMETRY_FILENAME)
    manifest = {
        "telemetry": telemetry,
        "implementation_revision": SOURCE_PIPELINE_REVISION,
        "quality_revision": SOURCE_QUALITY_REVISION,
        "verification_revision": SOURCE_VERIFICATION_REVISION,
        "normalized_shards": config.normalized_shards,
        "datasets": {name: str(output / name) for name in OUTPUT_VIEWS},
        "source_dataset": recipe.source.dataset,
        "source_revision": recipe.source.revision,
        "recipe_revision": recipe.version,
        "status": status,
        "mode": config.mode.value,
        "source": recipe.name,
        "intended_use": recipe.intended_use.value,
        "raw_population_count": panel.sample.population_count,
        "raw_sample_count": len(panel.sample.rows),
        "raw_population_census": panel.census,
        "full_expansion": expanded,
        "unavailable_reviews": unavailable_reviews,
        "quality": decision.model_dump(mode="json"),
        "verification": report,
        "admission": source_admission(admissions),
        "admission_counts": admissions,
        "processed_rows": verification["input_rows"],
        "unprocessed_rows": panel.sample.population_count - verification["input_rows"],
    }
    _write_json(output / "review/report.json", decision.model_dump(mode="json"))
    _write_json(
        output / "review/manifest.json",
        {
            "telemetry": telemetry,
            "normalized_source": str(output / "normalize"),
            "quality_report": str(output / "review/report.json"),
            "rubric": asdict(recipe.rubric) if recipe.rubric is not None else None,
            "review_evidence": [str(run.scratch("quality") / "evidence"), str(run.scratch("audited") / "evidence")],
        },
    )
    _write_json(output / VERIFY_REPORT_PATH, report)
    _write_json(
        output / "verify/manifest.json",
        {
            "telemetry": telemetry,
            "normalized_source": str(output / "normalize"),
            "report": str(output / VERIFY_REPORT_PATH),
            "controls": recipe.controls is not None,
            "policy": asdict(config.verification_policy),
        },
    )
    _write_json(output / "manifest.json", manifest)


def _finish_scratch(run: _SourceRun, status: SourceStatus) -> None:
    """Dispose of a finished run's redundant task payloads and point phase manifests at the normalize view."""
    # Keep request/response evidence and decision manifests. Only the completed
    # procedure's redundant task payloads are disposable; failures retain them.
    if status != SourceStatus.INCOMPLETE:
        for phase in SCRATCH_PHASES:
            for view in ("audit", "accepted", "review-inputs"):
                path = run.scratch(phase) / view
                if path.exists():
                    path.rmtree()
    for phase in SCRATCH_PHASES:
        path = run.scratch(phase) / "manifest.json"
        if path.exists():
            phase_manifest = _read_json(path)
            phase_manifest.pop("prepared_source", None)
            phase_manifest.pop("audited_source", None)
            phase_manifest["normalized_source"] = str(run.output / "normalize")
            _write_json(path, phase_manifest)


def _run_source_pipeline(run: _SourceRun, *, previous_verification_report: str | None) -> SourcePipelineResult:
    """Review bounded raw tasks, gate full conversion, and persist joined source sidecars.

    The caller owns the entered context and reviewer. Reading the raw
    population scans selected raw records with bounded memory, but does not
    execute task converters or controls. Successful outputs retain request
    evidence and decisions while disposing of redundant intermediate task payloads.
    """
    config = run.config
    if config.normalized_shards < 1:
        raise ValueError("Canonical normalized shard count must be positive")
    if config.quality_policy.sample_size > 100:
        raise ValueError("The initial raw task panel is capped at 100")
    panel = _sample_panel(run)
    review = config.review if run.recipe.rubric is not None else None
    decision = _gate_quality(run, panel, review)
    expanded = config.mode == SourceProcessingMode.FULL and decision.status in EXPANDED_QUALITY
    prepared = _expand_full(run, panel) if expanded and not panel.census else run.scratch("sample")
    unavailable_reviews = _audit(run, prepared, review)
    _filter(run)
    verification, checked = _verify(run, panel, previous_verification_report)
    verification_status = SourceVerificationStatus(verification["verification"]["status"])
    _write_download_manifest(run, panel.sample.population_count)
    admissions = _export_sidecars(run, panel, decision, checked, verification_status, expanded=expanded)
    verification["verification"]["source_path"] = str(run.output / "normalize")
    verification["verification"]["review_path"] = str(run.output / "review")
    status = _source_status(
        config.mode,
        decision.status,
        verification_status,
        verification_infra_errors=verification["verification"]["counts"]["infra_error"],
        processed_all=expanded or panel.census,
    )
    _write_manifest(
        run,
        panel,
        decision,
        verification,
        admissions,
        expanded=expanded,
        unavailable_reviews=unavailable_reviews,
        status=status,
    )
    _finish_scratch(run, status)
    return SourcePipelineResult(
        *(str(run.output / name) for name in OUTPUT_VIEWS),
        manifest_path=str(run.output / "manifest.json"),
        status=status,
    )


def run_source_pipeline(
    recipe: SourceRecipe,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    config: SourcePipelineConfig,
    *,
    previous_verification_report: str | None = None,
    canonical_source: str,
) -> SourcePipelineResult:
    """Run one source and persist final execution counters and partial phase evidence.

    ``previous_verification_report`` names an earlier source's ``VERIFY_REPORT_PATH`` whose
    matching control trials are reused.
    """
    telemetry = SourceTelemetry(canonical_source, output_path)
    run = _SourceRun(recipe, context, source_input, StoragePath(output_path), config, telemetry)
    with telemetry.record():
        return _run_source_pipeline(run, previous_verification_report=previous_verification_report)
