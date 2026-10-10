# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mechanical row conversion shared by quick, sampled, and full curation."""

import time
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError
from zephyr import counters

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import (
    Converter,
    ImportFailureKind,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    SourceRecipe,
)
from taskcompendium.pipeline.sources import conversion_context, decode_staged_row


def row_source(recipe: SourceRecipe, locator: str) -> Source:
    return Source(
        dataset=recipe.source.dataset,
        revision=recipe.source.revision,
        row=locator,
        importer_revision=recipe.version,
    )


def row_task_id(recipe: SourceRecipe, source: Source) -> str:
    return f"{recipe.name}-{canonical_sha256(source.model_dump())}"


def convert_row(row: RawRow, convert: Converter, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Convert one source row, retaining rewrites and enforcing its supplied identity.

    Resource admission and content fingerprints belong to the reviewed pipeline,
    so mechanical conversion can call this without performing those checks.
    """
    try:
        result = convert(row, context)
    except ValidationError as error:
        return ImportRejection(kind=ImportFailureKind.CONVERTER_ERROR, reason="invalid_task_spec", detail=str(error))
    if isinstance(result, ImportRejection):
        return result
    normalized = result if isinstance(result, NormalizedTask) else NormalizedTask(result, ())
    if normalized.task.id != row.id or normalized.task.source != row.source:
        raise ValueError("A converter must retain its supplied task identity and source provenance")
    return normalized


@dataclass(frozen=True)
class ConvertedRow:
    """A decoded source row and its conversion, before review or resource admission."""

    raw: RawRow
    original_data: dict[str, Any]
    original_path: str | None
    result: NormalizedTask | ImportRejection


def convert_record(record: dict[str, Any], recipe: SourceRecipe) -> ConvertedRow:
    source = row_source(recipe, record["locator"])
    raw = RawRow(row_task_id(recipe, source), source, record["data"])
    converted = convert_raw_row(raw, recipe.convert, conversion_context(recipe))
    return replace(converted, original_path=record.get("original_path", converted.original_path))


def convert_raw_row(row: RawRow, convert: Converter, context: ConversionContext) -> ConvertedRow:
    """Convert a caller-owned row and retain its identity, payload, rewrites and rejection.

    The caller supplies decoded data and source provenance. No ingestion, review,
    resource admission, deduplication, mechanical checks or grader controls run.
    """
    metrics = counters.current_stage()
    started = time.monotonic()
    try:
        result = convert_row(row, convert, context)
    finally:
        metrics.update_counter("source/normalize/seconds", time.monotonic() - started)
        metrics.update_counter("source/normalize/attempts", 1)
    metrics.update_counter("source/normalize/completed_rows", 1)
    metrics.update_counter("source/normalize/task_rows", int(isinstance(result, NormalizedTask)))
    if isinstance(result, ImportRejection):
        metrics.update_counter(f"source/normalize/{result.kind.value}", 1)
    return ConvertedRow(row, dict(row.data), row.data.get("path"), result)


def convert_source_row(record: dict[str, Any], recipe: SourceRecipe) -> ConvertedRow:
    """Decode and convert once, retaining the archive identity before decoder rewrites."""
    metrics = counters.current_stage()
    started = time.monotonic()
    try:
        decoded = decode_staged_row(record, recipe.source, conversion_context(recipe))
    finally:
        metrics.update_counter("source/decode/seconds", time.monotonic() - started)
        metrics.update_counter("source/decode/attempts", 1)
    metrics.update_counter("source/decode/completed_rows", 1)
    converted = convert_record(decoded, recipe)
    return ConvertedRow(converted.raw, record["data"], converted.original_path, converted.result)


def normalization_columns(
    task_id: str,
    source: Source,
    task: TaskSpec | None,
    rejection: ImportRejection | None,
    changes: tuple[NormalizationChange, ...],
    original_path: str | None,
) -> dict[str, Any]:
    """The common mechanical output columns of every curation mode."""
    return {
        "task_id": task_id,
        "source_dataset": source.dataset,
        "source_revision": source.revision,
        "source_row": source.row,
        "original_path": original_path,
        "task_json": task.model_dump_json() if task is not None else None,
        "normalization_kind": rejection.kind.value if rejection is not None else None,
        "normalization_reason": rejection.reason if rejection is not None else None,
        "normalization_detail": rejection.detail if rejection is not None else None,
        "normalization_changes": [change.model_dump(mode="json") for change in changes],
    }


def converted_columns(row: ConvertedRow) -> dict[str, Any]:
    result = row.result
    rejection = result if isinstance(result, ImportRejection) else None
    return {
        **normalization_columns(
            row.raw.id,
            row.raw.source,
            result.task if isinstance(result, NormalizedTask) else None,
            rejection,
            result.changes if isinstance(result, NormalizedTask) else (),
            row.original_path,
        ),
        "source_locator": row.raw.source.row,
        "raw_input_sha256": None,
        "raw_sha256": None,
    }
