# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write packaging-ready TaskTrove rows with source proof, without archives."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import SCHEMA_VERSION

from experiments.post_training.taskcompendium.audit_tasktrove_ingest import (
    PROOF_FIELDS,
    PUBLIC_FIELDS,
    audit_artifacts,
)
from experiments.post_training.taskcompendium.ingest_tasktrove import PUBLIC_CANDIDATE_COHORTS

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REGIONAL_PIN_RE = re.compile(
    r"[^#]+#manifest-sha256=[0-9a-f]{64}#parquet-size=\d+#parquet-etag=([^#]*)#parquet-version-id=([^#]*)\Z"
)
LEDGER_COLUMNS = (
    "input_split",
    "input_file",
    "input_row",
    "source",
    "path",
    "mode",
    "family",
    "converter",
    "template_id",
    "tags",
    "reason",
    "input_object_pin",
    "archive_sha256",
    "disposition",
    "imported_id",
)
HOLDOUT_KEYS = frozenset({"holdout", "holdouts", "validation", "validation_split", "test_split"})
RIGHTS_TERM_KEYS = (
    "license",
    "source_license",
    "license_url",
    "attribution",
    "copyright",
    "author",
    "url",
    "authors",
    "source_url",
)


def _valid_object_pin(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("sha256:") and SHA256_RE.fullmatch(value[7:]):
        return True
    match = REGIONAL_PIN_RE.fullmatch(value)
    return match is not None and bool(match[1] or match[2])


def _jsonl_rows(path: StoragePath, digest: Any) -> Iterator[dict[str, Any]]:
    with path.open("rb") as opened:
        with opened as stream:
            for line in stream:
                digest.update(line)
                yield json.loads(line)


def _sha256_file(path: StoragePath) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as opened:
        with opened as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    return digest.hexdigest(), size


def _declared_holdouts(value: Any, path: str = "upstream_tasktrove") -> list[str]:
    if isinstance(value, dict):
        declared = []
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if key.lower() in HOLDOUT_KEYS and nested not in (None, False, 0, "", [], {}):
                declared.append(child_path)
            declared.extend(_declared_holdouts(nested, child_path))
        return declared
    if isinstance(value, list):
        return [
            declaration
            for index, nested in enumerate(value)
            for declaration in _declared_holdouts(nested, f"{path}[{index}]")
        ]
    return []


def _rights_terms(source_metadata_json: str) -> list[tuple[str, str]]:
    source_metadata = json.loads(source_metadata_json)
    if not isinstance(source_metadata, dict):
        raise ValueError("Catalog source metadata must be a JSON object")
    values_by_key = {str(key).strip().lower(): value for key, value in source_metadata.items()}
    terms = []
    for key in RIGHTS_TERM_KEYS:
        value = (
            json.dumps(
                values_by_key[key].strip() if isinstance(values_by_key[key], str) else values_by_key[key],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in values_by_key
            else "<absent>"
        )
        terms.append((key, value))
    return terms


def _stream_rows(path: StoragePath) -> Iterator[pa.RecordBatch]:
    with path.open("rb") as opened:
        with opened as stream:
            parquet = pq.ParquetFile(stream)
            if set(LEDGER_COLUMNS) - set(parquet.schema_arrow.names):
                raise ValueError("TaskTrove ledger is missing required export fields")
            yield from parquet.iter_batches(columns=list(LEDGER_COLUMNS), batch_size=65_536)


def _stream_catalog_metadata(path: StoragePath) -> Iterator[pa.RecordBatch]:
    columns = ("id", "source", "path", "tags", "source_metadata_json")
    with path.open("rb") as opened:
        with opened as stream:
            parquet = pq.ParquetFile(stream)
            if set(columns) - set(parquet.schema_arrow.names):
                raise ValueError("TaskTrove catalog is missing source-rights metadata fields")
            yield from parquet.iter_batches(columns=list(columns), batch_size=65_536)


def export_accepted_records(
    ledger_path: StoragePath,
    catalog_path: StoragePath,
    candidate_path: StoragePath,
    proof_path: StoragePath,
    destination: StoragePath,
    *,
    ingestion_manifest: dict[str, Any],
    builder_revision: str,
) -> dict[str, Any]:
    """Join accepted rows and write one proof-wrapped JSONL per eligible source."""
    if ingestion_manifest.get("status") != "complete":
        raise ValueError("Cannot export accepted records from an incomplete ingestion run")
    declared_holdouts = _declared_holdouts(ingestion_manifest.get("upstream_tasktrove"))
    if declared_holdouts:
        raise ValueError(f"Pinned TaskTrove metadata declares holdouts requiring split review: {declared_holdouts}")
    if not re.fullmatch(r"[0-9a-f]{40}", builder_revision):
        raise ValueError("Projection builder revision must be a full Git commit")
    audit = audit_artifacts(ledger_path, candidate_path, proof_path)
    if audit["status"] != "passed":
        raise ValueError(f"TaskTrove candidate proof audit failed: {audit['validation_errors']}")

    counters: Counter[tuple[str, str, str, str, str, str, str, str]] = Counter()
    tag_dispositions: Counter[tuple[str, str, str, str, str, str, str, str, str]] = Counter()
    rejection_reasons: Counter[tuple[str, str, str, str, str, str, str]] = Counter()
    math_sample_pointers: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    cohort_dispositions: Counter[tuple[str, str, str]] = Counter()
    cohort_rows: Counter[tuple[str, str]] = Counter()
    cohort_eligible_rows: Counter[str] = Counter()
    cohort_converted_rows: Counter[str] = Counter()
    cohort_archive_rows: Counter[str] = Counter()
    rights_term_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    rights_examples: dict[tuple[str, str, str, str, str], tuple[str, str]] = {}
    source_assets: dict[str, set[tuple[str, str]]] = {}
    candidate_digest = hashlib.sha256()
    proof_digest = hashlib.sha256()
    expected_artifacts = ingestion_manifest["artifacts"]
    ledger_digest, ledger_bytes_hashed = _sha256_file(ledger_path)
    if ledger_digest != expected_artifacts["ledger"]["sha256"]:
        raise ValueError("Ledger digest differs from the ingestion manifest")
    catalog_digest, catalog_bytes_hashed = _sha256_file(catalog_path)
    if catalog_digest != expected_artifacts["private_catalog"]["sha256"]:
        raise ValueError("Private catalog digest differs from the ingestion manifest")
    if destination.exists():
        raise FileExistsError(f"Accepted-record output prefix already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix="tasktrove-accepted-export-") as temporary_directory:
        with sqlite3.connect(Path(temporary_directory) / "joins.sqlite3") as connection:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE expected (
                    id TEXT PRIMARY KEY,
                    input_split TEXT,
                    input_file TEXT,
                    input_row INTEGER,
                    source TEXT,
                    path TEXT,
                    mode TEXT,
                    family TEXT,
                    converter TEXT,
                    template_id TEXT,
                    tags_json TEXT,
                    input_object_pin TEXT,
                    archive_sha256 TEXT
                );
                CREATE TABLE proofs (
                    id TEXT PRIMARY KEY,
                    input_file TEXT,
                    input_row INTEGER,
                    input_object_pin TEXT,
                    source TEXT,
                    path TEXT,
                    archive_sha256 TEXT,
                    disposition TEXT
                );
                CREATE TABLE catalog_terms (
                    id TEXT PRIMARY KEY, source TEXT, path TEXT, tags_json TEXT, source_metadata_json TEXT
                );
                CREATE TABLE rights_rejected (id TEXT PRIMARY KEY, source TEXT, reason TEXT);
                """
            )

            for batch in _stream_rows(ledger_path):
                for row in batch.to_pylist():
                    split = str(row.get("input_split") or "<missing>")
                    source = str(row.get("source") or "<missing>")
                    disposition = str(row.get("disposition") or "<missing>")
                    mode = str(row.get("mode") or "<missing>")
                    family = str(row.get("family") or "<missing>")
                    converter = str(row.get("converter") or "<missing>")
                    template_id = str(row.get("template_id") or "<missing>")
                    reason = str(row.get("reason") or "<none>")
                    counters[(source, split, mode, family, converter, template_id, disposition, reason)] += 1
                    if disposition == "rejected":
                        rejection_reasons[(source, mode, family, converter, template_id, reason, split)] += 1
                        sample_key = (source, mode, family, converter, reason)
                        samples = math_sample_pointers.setdefault(sample_key, [])
                        if mode == "math" and len(samples) < 5:
                            samples.append(
                                {
                                    "input_split": split,
                                    "input_file": row.get("input_file"),
                                    "input_row": row.get("input_row"),
                                    "input_object_pin": row.get("input_object_pin"),
                                    "path": row.get("path"),
                                }
                            )
                    tags = row.get("tags") or []
                    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                        raise ValueError("Ledger tags must preserve an ordered string list")
                    for tag in set(tags):
                        tag_dispositions[
                            (source, split, mode, family, converter, template_id, tag, disposition, reason)
                        ] += 1
                    expected_mode = PUBLIC_CANDIDATE_COHORTS.get(source)
                    if split == "tasks" and expected_mode is not None:
                        cohort_rows[(source, disposition)] += 1
                        if row.get("mode") == expected_mode:
                            cohort_eligible_rows[source] += 1
                            cohort_dispositions[(source, disposition, split)] += 1
                            if row.get("archive_sha256"):
                                cohort_archive_rows[source] += 1
                            if disposition in {"imported", "duplicate"}:
                                cohort_converted_rows[source] += 1
                    if disposition != "imported" or split != "tasks" or expected_mode != row.get("mode"):
                        continue
                    task_id = row.get("imported_id")
                    archive_sha256 = row.get("archive_sha256")
                    if not isinstance(task_id, str) or not task_id:
                        raise ValueError("Eligible imported ledger row has no task ID")
                    if not _valid_object_pin(row.get("input_object_pin")):
                        raise ValueError(f"Eligible row {task_id} has no valid immutable object pin")
                    if not isinstance(archive_sha256, str) or not SHA256_RE.fullmatch(archive_sha256):
                        raise ValueError(f"Eligible row {task_id} has no archive SHA256")
                    if (
                        not isinstance(row.get("input_file"), str)
                        or not row["input_file"]
                        or type(row.get("input_row")) is not int
                        or not isinstance(row.get("path"), str)
                        or not row["path"]
                    ):
                        raise ValueError(f"Eligible row {task_id} has incomplete source coordinates")
                    cursor.execute(
                        "INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            task_id,
                            split,
                            row.get("input_file"),
                            row.get("input_row"),
                            source,
                            row.get("path"),
                            row.get("mode"),
                            row.get("family"),
                            row.get("converter"),
                            row.get("template_id"),
                            json.dumps(tags, ensure_ascii=False, separators=(",", ":")),
                            row.get("input_object_pin"),
                            archive_sha256,
                        ),
                    )
                    source_assets.setdefault(source, set()).add((row["input_file"], row["input_object_pin"]))

            for batch in _stream_catalog_metadata(catalog_path):
                for row in batch.to_pylist():
                    if cursor.execute("SELECT 1 FROM expected WHERE id = ?", (row.get("id"),)).fetchone() is None:
                        continue
                    cursor.execute(
                        "INSERT INTO catalog_terms VALUES (?, ?, ?, ?, ?)",
                        (
                            row.get("id"),
                            row.get("source"),
                            row.get("path"),
                            json.dumps(row.get("tags") or [], ensure_ascii=False, separators=(",", ":")),
                            row.get("source_metadata_json"),
                        ),
                    )

            for proof_row in _jsonl_rows(proof_path, proof_digest):
                if not isinstance(proof_row, dict) or set(proof_row) != PROOF_FIELDS:
                    raise ValueError("Candidate proof row does not match its private schema")
                cursor.execute(
                    "INSERT INTO proofs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        proof_row.get("candidate_id"),
                        proof_row.get("input_file"),
                        proof_row.get("input_row"),
                        proof_row.get("input_object_pin"),
                        proof_row.get("source"),
                        proof_row.get("path"),
                        proof_row.get("archive_sha256"),
                        proof_row.get("disposition"),
                    ),
                )

            for candidate in _jsonl_rows(candidate_path, candidate_digest):
                if not isinstance(candidate, dict) or set(candidate) != PUBLIC_FIELDS:
                    raise ValueError("Public candidate does not match the PublicTask-v1 allowlist")
                if type(candidate.get("record_version")) is not int or candidate["record_version"] != 1:
                    raise ValueError("Public candidate is not PublicTask-v1")
                task_id = candidate.get("id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("Public candidate has no task ID")
                ledger = cursor.execute("SELECT * FROM expected WHERE id = ?", (task_id,)).fetchone()
                proof = cursor.execute("SELECT * FROM proofs WHERE id = ?", (task_id,)).fetchone()
                if ledger is None or proof is None:
                    raise ValueError(f"Candidate {task_id} is missing a unique accepted ledger/proof join")
                (
                    _,
                    split,
                    input_file,
                    input_row,
                    source,
                    archive_path,
                    mode,
                    family,
                    converter,
                    template_id,
                    ledger_tags_json,
                    object_pin,
                    archive_sha256,
                ) = ledger
                (
                    _,
                    proof_file,
                    proof_row,
                    proof_pin,
                    proof_source,
                    joined_proof_path,
                    proof_sha,
                    proof_disposition,
                ) = proof
                expected_row = f"{source}:{archive_path}"
                task_source = candidate.get("source")
                if not isinstance(task_source, dict) or task_source.get("row") != expected_row:
                    raise ValueError(f"Candidate {task_id} source row differs from its accepted ledger row")
                if task_source.get("dataset") != ingestion_manifest["release_uri"]:
                    raise ValueError(f"Candidate {task_id} source release differs from the ingestion pin")
                if task_source.get("revision") != object_pin or task_source.get(
                    "importer_revision"
                ) != ingestion_manifest.get("importer_revision"):
                    raise ValueError(f"Candidate {task_id} source object pin or importer revision differs")
                if candidate.get("source_category") != family:
                    raise ValueError(f"Candidate {task_id} source category differs from its accepted ledger row")
                candidate_tags = candidate.get("tags")
                if not isinstance(candidate_tags, list) or any(not isinstance(tag, str) for tag in candidate_tags):
                    raise ValueError(f"Candidate {task_id} has malformed ordered tags")
                if json.dumps(candidate_tags, ensure_ascii=False, separators=(",", ":")) != ledger_tags_json:
                    raise ValueError(f"Candidate {task_id} tags differ from the ordered ledger tags")
                if (proof_file, proof_row, proof_pin, proof_source, joined_proof_path, proof_sha, proof_disposition) != (
                    input_file,
                    input_row,
                    object_pin,
                    source,
                    archive_path,
                    archive_sha256,
                    "imported",
                ):
                    raise ValueError(f"Candidate {task_id} source proof differs from its accepted ledger row")
                catalog_row = cursor.execute(
                    "SELECT source, path, tags_json, source_metadata_json FROM catalog_terms WHERE id = ?",
                    (task_id,),
                ).fetchone()
                if catalog_row is None or catalog_row[:2] != (source, archive_path):
                    raise ValueError(f"Candidate {task_id} is missing matching private catalog source metadata")
                if catalog_row[2] != ledger_tags_json:
                    raise ValueError(f"Candidate {task_id} tags differ from the ordered catalog tags")
                terms = _rights_terms(catalog_row[3])
                for term_field, term_value in terms:
                    rights_term_counts[(source, str(mode), str(family), str(converter), term_field, term_value)] += 1
                    rights_examples.setdefault(
                        (source, str(mode), str(family), str(converter), term_field),
                        (archive_path, archive_sha256),
                    )
                # Public release remains held until packaging binds an explicit
                # clearance to this exact source/converter/template cohort.
                cursor.execute(
                    "INSERT INTO rights_rejected VALUES (?, ?, ?)",
                    (task_id, source, "awaiting-packaging-cohort-clearance"),
                )
                continue
            connection.commit()

            expected_count = cursor.execute("SELECT COUNT(*) FROM expected").fetchone()[0]
            proof_count = cursor.execute("SELECT COUNT(*) FROM proofs").fetchone()[0]
            rights_rejected_count = cursor.execute("SELECT COUNT(*) FROM rights_rejected").fetchone()[0]
            rights_rejected_counts: Counter[tuple[str, str]] = Counter()
            for source, reason, count in cursor.execute(
                "SELECT source, reason, COUNT(*) FROM rights_rejected GROUP BY source, reason"
            ):
                rights_rejected_counts[(str(source), str(reason))] = count
            if expected_count == 0:
                raise ValueError("No accepted TaskTrove candidates are available for export")
            if expected_count != rights_rejected_count or expected_count != proof_count:
                raise ValueError(
                    f"Candidate joins are incomplete: ledger={expected_count}, "
                    f"rights_rejected={rights_rejected_count}, proof={proof_count}"
                )
            unexpected_proofs = cursor.execute(
                "SELECT COUNT(*) FROM proofs p LEFT JOIN expected e ON e.id=p.id WHERE e.id IS NULL"
            ).fetchone()[0]
            if unexpected_proofs:
                raise ValueError(f"Found {unexpected_proofs} proof rows without an accepted candidate")
            if candidate_digest.hexdigest() != expected_artifacts["public_candidates"]["sha256"]:
                raise ValueError("Candidate JSONL digest differs from the ingestion manifest")
            if proof_digest.hexdigest() != expected_artifacts["candidate_proof"]["sha256"]:
                raise ValueError("Candidate proof digest differs from the ingestion manifest")

            sources = [row[0] for row in cursor.execute("SELECT DISTINCT source FROM expected ORDER BY source")]
            rights_held_sources = set(sources)
    output_files: dict[str, Any] = {}
    output_counts = Counter({source: 0 for source in rights_held_sources})
    cohort_counts: dict[str, Any] = {}
    for source in sorted(rights_held_sources):
        cohort_counts[source] = {
            "source_subset_rows_in_tasks_split": sum(
                count for (candidate_source, _disposition), count in cohort_rows.items() if candidate_source == source
            ),
            "eligible_mode_rows_in_tasks_split": cohort_eligible_rows[source],
            "archive_rows_hashed_before_conversion": cohort_archive_rows[source],
            "converter_completed_rows": cohort_converted_rows[source],
            "accepted_public_records": 0,
            "ledger_dispositions": {
                disposition: count
                for (candidate_source, disposition, _split), count in sorted(cohort_dispositions.items())
                if candidate_source == source
            },
        }
    report = {
        "status": "complete",
        "format": "AcceptedPublicRecord-v1",
        "task_spec_schema": SCHEMA_VERSION,
        "builder_revision": builder_revision,
        "importer_revision": ingestion_manifest["importer_revision"],
        "ingestion_manifest_uri": ingestion_manifest.get("manifest_uri"),
        "release_uri": ingestion_manifest["release_uri"],
        "release_revision": ingestion_manifest["release_revision"],
        "source_manifest_sha256": ingestion_manifest["source_manifest_sha256"],
        "input_ledger_sha256": ingestion_manifest["artifacts"]["ledger"]["sha256"],
        "private_catalog_sha256": ingestion_manifest["artifacts"]["private_catalog"]["sha256"],
        "input_candidate_sha256": ingestion_manifest["artifacts"]["public_candidates"]["sha256"],
        "input_proof_sha256": ingestion_manifest["artifacts"]["candidate_proof"]["sha256"],
        "candidate_sha256_verified": (
            candidate_digest.hexdigest() == ingestion_manifest["artifacts"]["public_candidates"]["sha256"]
        ),
        "proof_sha256_verified": (
            proof_digest.hexdigest() == ingestion_manifest["artifacts"]["candidate_proof"]["sha256"]
        ),
        "archive_payload_bytes_read": 0,
        "ledger_bytes_hashed": ledger_bytes_hashed,
        "private_catalog_bytes_hashed": catalog_bytes_hashed,
        "private_catalog_metadata_columns_read": ["id", "source", "path", "tags", "source_metadata_json"],
        "private_task_specifications_loaded": False,
        "original_split_to_public_split": {"tasks": "train", "sft": None},
        "source_assets_by_source": {
            source: [{"path": path, "pin": pin} for path, pin in sorted(assets)]
            for source, assets in sorted(source_assets.items())
        },
        "input_rows_by_source_split_mode_family_converter_disposition": [
            {
                "source": source,
                "split": split,
                "mode": mode,
                "family": family,
                "converter": converter,
                "template_id": template_id,
                "disposition": disposition,
                "reason": reason,
                "rows": count,
            }
            for (source, split, mode, family, converter, template_id, disposition, reason), count in sorted(
                counters.items()
            )
        ],
        "input_rows_by_source_split_mode_family_converter_tag_disposition": [
            {
                "source": source,
                "split": split,
                "mode": mode,
                "family": family,
                "converter": converter,
                "template_id": template_id,
                "tag": tag,
                "disposition": disposition,
                "reason": reason,
                "rows": count,
            }
            for (source, split, mode, family, converter, template_id, tag, disposition, reason), count in sorted(
                tag_dispositions.items()
            )
        ],
        "rejected_math_sample_pointers": [
            {
                "source": source,
                "mode": mode,
                "family": family,
                "converter": converter,
                "reason": reason,
                "samples": samples,
            }
            for (source, mode, family, converter, reason), samples in sorted(math_sample_pointers.items())
            if mode == "math"
        ],
        "rejected_rows_by_source_mode_family_converter_template_reason_split": [
            {
                "source": source,
                "mode": mode,
                "family": family,
                "converter": converter,
                "template_id": template_id,
                "reason": reason,
                "split": split,
                "rows": count,
            }
            for (source, mode, family, converter, template_id, reason, split), count in sorted(rejection_reasons.items())
        ],
        "cohort_counts": cohort_counts,
        "rights_terms_by_source_mode_family_converter": [
            {
                "source": source,
                "mode": mode,
                "family": family,
                "converter": converter,
                "term_field": term_field,
                "value": term_value,
                "rows": count,
            }
            for (source, mode, family, converter, term_field, term_value), count in sorted(rights_term_counts.items())
        ],
        "rights_term_examples_by_source_mode_family_converter": [
            {
                "source": source,
                "mode": mode,
                "family": family,
                "converter": converter,
                "term_field": term_field,
                "path": path,
                "archive_sha256": archive_sha256,
            }
            for (source, mode, family, converter, term_field), (path, archive_sha256) in sorted(rights_examples.items())
        ],
        "rights_rejected_rows_by_source_reason": [
            {"source": source, "reason": reason, "rows": count}
            for (source, reason), count in sorted(rights_rejected_counts.items())
        ],
        "accepted_rows_by_source": dict(sorted(output_counts.items())),
        "outputs": output_files,
        "source_split_policy": (
            "Original TaskTrove tasks rows map to public tasktrove_clean/train; SFT rows are excluded."
        ),
    }
    if not report["candidate_sha256_verified"] or not report["proof_sha256_verified"]:
        raise ValueError("Candidate or proof JSONL digest differs from the ingestion manifest")
    report_path = destination / "manifest.json"
    report["manifest_uri"] = str(report_path)
    destination.mkdirs(exist_ok=True)
    with report_path.open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    configure_coreweave_s3()
    output_uri = os.environ["TASKTROVE_OUTPUT_URI"]
    accepted_uri = os.environ["TASKTROVE_ACCEPTED_OUTPUT_URI"]
    builder_revision = os.environ["TASKTROVE_PROJECTION_REVISION"]
    ingest_prefix = StoragePath(output_uri)
    manifest_path = ingest_prefix / "ingestion-manifest.json"
    ingestion_manifest = json.loads(manifest_path.read_bytes())
    export_accepted_records(
        ingest_prefix / "ingestion-ledger.parquet",
        ingest_prefix / "private-catalog.parquet",
        ingest_prefix / "public-candidates.jsonl",
        ingest_prefix / "candidate-proof.jsonl",
        StoragePath(accepted_uri),
        ingestion_manifest=ingestion_manifest,
        builder_revision=builder_revision,
    )


if __name__ == "__main__":
    main()
