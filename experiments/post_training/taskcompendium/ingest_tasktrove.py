# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import supported TaskTrove answer tasks into bounded Parquet artifacts."""

import hashlib
import json
import os
import signal
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST, read_archive
from taskcompendium.importers.tasktrove.mathematical import import_task as import_math
from taskcompendium.importers.tasktrove.mcqa import import_task as import_mcqa
from taskcompendium.importers.tasktrove.models import IMPORTER_REVISION
from taskcompendium.importers.tasktrove.numeric import import_task as import_numeric
from taskcompendium.models import TaskSpec

from experiments.post_training.taskcompendium.records import (
    Disposition,
    catalog_record,
    ledger_record,
    public_task_record,
)

RELEASE_URI = "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3"
SPLITS = ("tasks", "sft")
SUPPORTED_MODES = frozenset({"mcq", "math", "numeric"})
PUBLIC_CANDIDATE_COHORTS = {
    "laion__nemotron-gym-knowledge-mcqa-v2": "mcq",
    "laion__nemotron-gym-math-openmathreasoning-v2": "math",
    "laion__nemo-prism-math-v3": "math",
}
SOURCE_METADATA_KEYS = (
    "source",
    "source_dataset",
    "source_uuid",
    "row_index",
    "license",
    "license_url",
    "source_license",
    "attribution",
    "copyright",
    "author",
    "authors",
    "url",
    "source_url",
)


def _source_terms(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        terms = {}
        for key, nested in value.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if any(token in key.lower() for token in ("license", "attribution", "copyright")):
                terms[full_key] = nested
            else:
                terms.update(_source_terms(nested, full_key))
        return terms
    if isinstance(value, list):
        return {
            key: nested
            for index, item in enumerate(value)
            for key, nested in _source_terms(item, f"{prefix}[{index}]").items()
        }
    return {}


READ_COLUMNS = (
    "path",
    "source",
    "family",
    "template_id",
    "converter",
    "mode",
    "route",
    "tags",
    "task_binary",
)
ROWS_PER_WRITE = 1_000


def _schema(fields: dict[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype) for name, dtype in fields.items()])


def _info_value(info: dict[str, Any], name: str) -> Any:
    return next((value for key, value in info.items() if key.lower() == name.lower()), None)


def _convert(row: dict[str, Any], release_uri: str, release_revision: str) -> tuple[TaskSpec, str, str]:
    mode = row.get("mode")
    archive_bytes = row.get("task_binary")
    if not isinstance(archive_bytes, bytes):
        raise ValueError("source row has no task archive bytes")
    path = row.get("path")
    source_subset = row.get("source")
    if not isinstance(path, str) or not isinstance(source_subset, str):
        raise ValueError("source row is missing path or subset identity")
    archive = read_archive(archive_bytes, source_subset, path, release_uri, release_revision)
    metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
    for field in ("family", "converter", "template_id", "mode", "tags"):
        row_value = tuple(row[field]) if field == "tags" and isinstance(row.get(field), list) else row.get(field)
        archive_value = (
            tuple(metadata[field]) if field == "tags" and isinstance(metadata.get(field), list) else metadata.get(field)
        )
        if row_value != archive_value:
            raise ValueError(f"Parquet {field} does not match the task archive metadata")
    if mode == "mcq":
        specification = import_mcqa(archive)
    elif mode == "math":
        specification = import_math(archive).specification
    else:
        specification = import_numeric(archive)
    source_metadata = {key: metadata[key] for key in SOURCE_METADATA_KEYS if key in metadata}
    return specification, archive.archive_sha256, json.dumps(source_metadata, sort_keys=True)


def ingest(release_uri: str, output_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Stream both release splits, preserving one ledger row per input row."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    configure_coreweave_s3()
    release = StoragePath(release_uri)
    manifest_bytes = (release / "manifest.json").read_bytes()
    source_manifest = json.loads(manifest_bytes)
    source_terms = _source_terms(source_manifest)
    revision = release.segments[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_uri = os.environ["TASKTROVE_OUTPUT_URI"].rstrip("/")
    output_prefix = StoragePath(output_uri)
    output_prefix.mkdirs(exist_ok=True)
    catalog_path = output_prefix / "private-catalog.parquet"
    ledger_path = output_prefix / "ingestion-ledger.parquet"
    candidates_path = output_prefix / "public-candidates.jsonl"
    catalog_schema = _schema(
        {
            "id": pa.string(),
            "source": pa.string(),
            "path": pa.string(),
            "route": pa.string(),
            "mode": pa.string(),
            "family": pa.string(),
            "converter": pa.string(),
            "template_id": pa.string(),
            "tags": pa.list_(pa.string()),
            "archive_sha256": pa.string(),
            "source_metadata_json": pa.string(),
            "specification_json": pa.string(),
        }
    )
    ledger_schema = _schema(
        {
            "input_split": pa.string(),
            "input_file": pa.string(),
            "input_row": pa.int64(),
            "source": pa.string(),
            "path": pa.string(),
            "route": pa.string(),
            "input_object_pin": pa.string(),
            "archive_sha256": pa.string(),
            "disposition": pa.string(),
            "imported_id": pa.string(),
            "reason": pa.string(),
        }
    )
    buffers: dict[str, list[dict[str, Any]]] = {"catalog": [], "ledger": []}
    schemas = {"catalog": catalog_schema, "ledger": ledger_schema}
    writers: dict[str, pq.ParquetWriter | None] = {"catalog": None, "ledger": None}
    writer_handles: dict[str, Any] = {}
    candidate_handle: Any | None = None
    counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    source_metadata_counts: Counter[str] = Counter()
    public_candidate_counts: Counter[str] = Counter()
    seen_ids: dict[tuple[str, str], str] = {}
    input_ordinal = 0
    input_files: list[dict[str, Any]] = []
    archive_payload_bytes_materialized = 0
    candidate_archive_bytes_parsed = 0
    failure: Exception | None = None

    def flush() -> None:
        for name in buffers:
            if buffers[name]:
                if writers[name] is None:
                    target = {"catalog": catalog_path, "ledger": ledger_path}[name]
                    handle = target.open("wb").open()
                    writer_handles[name] = handle
                    writers[name] = pq.ParquetWriter(handle, schemas[name])
                writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
                buffers[name].clear()

    try:
        stop = False
        for split in SPLITS:
            for path in sorted((release / split / "part-*.parquet").glob(), key=str):
                opened = path.open("rb")
                info = opened.fs.info(opened.path)
                parquet_size = _info_value(info, "size")
                parquet_etag = _info_value(info, "etag")
                parquet_version_id = _info_value(info, "versionid")
                pinned_revision = (
                    f"{revision}#manifest-sha256={hashlib.sha256(manifest_bytes).hexdigest()}"
                    f"#parquet-size={parquet_size}#parquet-etag={parquet_etag}#parquet-version-id={parquet_version_id}"
                )
                input_object_pin = pinned_revision
                file_rows = 0
                with opened as stream:
                    parquet = pq.ParquetFile(stream)
                    available = set(parquet.schema_arrow.names)
                    metadata_columns = [
                        column for column in READ_COLUMNS if column != "task_binary" and column in available
                    ]
                    row_group_offset = 0
                    for row_group in range(parquet.metadata.num_row_groups):
                        metadata_rows = parquet.read_row_group(row_group, columns=metadata_columns).to_pylist()
                        if limit is not None:
                            remaining = limit - input_ordinal
                            if remaining <= 0:
                                stop = True
                                break
                            metadata_rows = metadata_rows[:remaining]
                        read_archives = any(row.get("mode") in SUPPORTED_MODES for row in metadata_rows)
                        archive_batches = (
                            parquet.iter_batches(row_groups=[row_group], columns=["task_binary"], batch_size=128)
                            if read_archives and "task_binary" in available
                            else None
                        )
                        for row_start in range(0, len(metadata_rows), 128):
                            metadata_batch = metadata_rows[row_start : row_start + 128]
                            binary_batch = (
                                next(archive_batches).column(0).to_pylist()
                                if archive_batches
                                else [None] * len(metadata_batch)
                            )
                            archive_payload_bytes_materialized += sum(
                                len(value) for value in binary_batch if isinstance(value, bytes)
                            )
                            for index, (row, archive_bytes) in enumerate(zip(metadata_batch, binary_batch, strict=True)):
                                row_number = row_group_offset + row_start + index
                                mode = row.get("mode")
                                route = row.get("route") or split
                                source = row.get("source")
                                source_path = row.get("path")
                                counts[f"metadata:split:{split}:mode:{mode or '<missing>'}"] += 1
                                specification = None
                                archive_sha256_for_ledger = None
                                if mode not in SUPPORTED_MODES:
                                    disposition = Disposition.OUT_OF_SCOPE
                                    reason = f"unsupported mode: {mode or '<missing>'}"
                                    counts["out_of_scope"] += 1
                                else:
                                    try:
                                        if not isinstance(archive_bytes, bytes):
                                            raise ValueError("source row has no task archive bytes")
                                        archive_sha256_for_ledger = hashlib.sha256(archive_bytes).hexdigest()
                                        candidate_archive_bytes_parsed += len(archive_bytes)
                                        row["task_binary"] = archive_bytes
                                        specification, archive_sha256, source_metadata_json = _convert(
                                            row, release_uri, pinned_revision
                                        )
                                        source_identity = (str(source), str(source_path))
                                        previous_digest = seen_ids.get(source_identity)
                                        if previous_digest == archive_sha256:
                                            disposition = Disposition.DUPLICATE
                                            reason = "duplicate source subset and path"
                                            counts["duplicates"] += 1
                                        elif previous_digest is not None:
                                            raise ValueError("repeated source identity has conflicting archive bytes")
                                        else:
                                            seen_ids[source_identity] = archive_sha256
                                            disposition = Disposition.IMPORTED
                                            reason = None
                                            counts["imported"] += 1
                                            accepted_counts[f"split:{split}"] += 1
                                            accepted_counts[f"mode:{mode}"] += 1
                                            accepted_counts[f"family:{row.get('family') or '<missing>'}"] += 1
                                            accepted_counts[f"converter:{row.get('converter') or '<missing>'}"] += 1
                                            metadata = {
                                                "source": str(source),
                                                "path": str(source_path),
                                                "route": str(route),
                                                "mode": str(mode),
                                                "family": str(row.get("family") or ""),
                                                "converter": str(row.get("converter") or ""),
                                                "template_id": str(row.get("template_id") or ""),
                                                "archive_sha256": archive_sha256,
                                                "source_metadata_json": source_metadata_json,
                                            }
                                            buffers["catalog"].append(catalog_record(specification, **metadata))
                                            for key in json.loads(source_metadata_json):
                                                source_metadata_counts[key] += 1
                                            if split == "tasks" and PUBLIC_CANDIDATE_COHORTS.get(str(source)) == mode:
                                                if candidate_handle is None:
                                                    candidate_handle = candidates_path.open("wb").open()
                                                candidate = public_task_record(specification, family=metadata["family"])
                                                candidate_handle.write(
                                                    json.dumps(candidate, sort_keys=True).encode() + b"\n"
                                                )
                                                public_candidate_counts[f"source:{source}"] += 1
                                                public_candidate_counts[f"mode:{mode}"] += 1
                                    except (ValueError, KeyError, UnicodeDecodeError) as error:
                                        disposition = Disposition.REJECTED
                                        reason = str(error)[:500]
                                        rejected_reasons[reason] += 1
                                        counts["rejected"] += 1
                                buffers["ledger"].append(
                                    ledger_record(
                                        input_split=split,
                                        input_file=str(path),
                                        input_row=row_number,
                                        source=source,
                                        path=source_path,
                                        route=str(route),
                                        input_object_pin=input_object_pin,
                                        archive_sha256=archive_sha256_for_ledger,
                                        disposition=disposition,
                                        imported_id=specification.id if disposition == Disposition.IMPORTED else None,
                                        reason=reason,
                                    )
                                )
                                file_rows += 1
                                input_ordinal += 1
                                counts["input"] += 1
                                counts[f"disposition:{disposition.value}"] += 1
                                if len(buffers["ledger"]) >= ROWS_PER_WRITE:
                                    flush()
                        row_group_offset += parquet.metadata.row_group(row_group).num_rows
                        if stop:
                            break
                input_files.append(
                    {
                        "path": str(path),
                        "rows_read": file_rows,
                        "size_bytes": parquet_size,
                        "etag": parquet_etag,
                        "version_id": parquet_version_id,
                    }
                )
                flush()
                if stop:
                    break
            if stop:
                break
    except Exception as error:
        failure = error
    finally:
        flush()
        for writer in writers.values():
            if writer is not None:
                writer.close()
        for handle in writer_handles.values():
            handle.close()
        if candidate_handle is not None:
            candidate_handle.close()

    report = {
        "status": "partial" if failure is not None else "complete",
        "failure": f"{type(failure).__name__}: {failure}" if failure is not None else None,
        "release_uri": release_uri,
        "release_revision": revision,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "upstream_tasktrove": source_manifest.get("tasktrove"),
        "source_manifest_license_attribution_fields": source_terms,
        "importer_revision": IMPORTER_REVISION,
        "input_rows_processed": input_ordinal,
        "archive_payload_bytes_materialized": archive_payload_bytes_materialized,
        "candidate_archive_bytes_parsed": candidate_archive_bytes_parsed,
        "counts": dict(sorted(counts.items())),
        "accepted_counts": dict(sorted(accepted_counts.items())),
        "public_candidate_counts": dict(sorted(public_candidate_counts.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "source_metadata_field_counts": dict(sorted(source_metadata_counts.items())),
        "upstream_license_attribution_fields_found": sorted(
            key
            for key in source_metadata_counts
            if key in {"license", "license_url", "source_license", "attribution", "copyright"}
        ),
        "input_files": input_files,
        "artifacts": {
            "private_catalog": {"uri": str(catalog_path)},
            "ledger": {"uri": str(ledger_path)},
            "public_candidates": {"uri": str(candidates_path), "present": candidate_handle is not None},
        },
        "public_candidate_projection_materialized": candidate_handle is not None,
        "public_candidate_schema": "PublicTask-v1",
        "public_projection_published": False,
        "public_candidate_routes": ["tasks"],
        "public_candidate_cohorts": PUBLIC_CANDIDATE_COHORTS,
        "public_allowlist_fields": [
            "id",
            "context",
            "environment_requirements",
            "tool_providers",
            "final_tools",
            "answer_type",
            "source",
            "submission_instruction",
            "tags",
            "source_category",
        ],
        "selection": {"limit": limit, "supported_modes": sorted(SUPPORTED_MODES), "splits": list(SPLITS)},
    }
    artifact_paths = {"private_catalog": catalog_path, "ledger": ledger_path}
    if candidate_handle is not None:
        artifact_paths["public_candidates"] = candidates_path
    for key, path in artifact_paths.items():
        opened = path.open("rb")
        info = opened.fs.info(opened.path)
        report["artifacts"][key].update(
            {
                "size_bytes": _info_value(info, "size"),
                "etag": _info_value(info, "etag"),
                "version_id": _info_value(info, "versionid"),
            }
        )
        opened.close()
    manifest_path = output_prefix / "ingestion-manifest.json"
    with manifest_path.open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "ingestion-summary.json").open("w") as summary_stream:
        summary_stream.write(
            json.dumps(
                {
                    "release_uri": release_uri,
                    "release_revision": revision,
                    "source_manifest_sha256": report["source_manifest_sha256"],
                    "input_rows_processed": input_ordinal,
                    "counts": report["counts"],
                    "accepted_counts": report["accepted_counts"],
                    "public_candidate_counts": report["public_candidate_counts"],
                    "rejected_reasons": report["rejected_reasons"],
                    "artifacts": report["artifacts"],
                    "manifest_uri": str(manifest_path),
                    "selection": report["selection"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if failure is not None:
        raise failure
    return report


def main() -> None:
    def stop_at_runtime_limit(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("TaskTrove ingestion reached its internal runtime limit")

    release_uri = os.environ.get("TASKTROVE_RELEASE_URI", RELEASE_URI)
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    limit = int(os.environ["TASKTROVE_INGEST_LIMIT"]) if os.environ.get("TASKTROVE_INGEST_LIMIT") else None
    runtime_limit = int(os.environ.get("TASKTROVE_INGEST_RUNTIME_SECONDS", "840"))
    signal.signal(signal.SIGTERM, stop_at_runtime_limit)
    signal.signal(signal.SIGALRM, stop_at_runtime_limit)
    signal.setitimer(signal.ITIMER_REAL, runtime_limit)
    try:
        report = ingest(release_uri, output_dir, limit=limit)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "manifest_uri": (
                    report["artifacts"]["private_catalog"]["uri"].rsplit("/", 1)[0] + "/ingestion-manifest.json"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
