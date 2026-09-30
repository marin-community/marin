# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit TaskTrove ingestion ledgers and candidate proof joins in-region."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.taskcompendium.ingest_tasktrove import PUBLIC_CANDIDATE_COHORTS


class HashDigest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


BATCH_SIZE = 65_536
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_FIELDS = frozenset(
    {
        "record_version",
        "id",
        "context",
        "environment_requirements",
        "final_tools",
        "answer_type",
        "source",
        "submission_instruction",
        "tags",
        "source_category",
    }
)
PROOF_FIELDS = frozenset(
    {
        "candidate_id",
        "input_file",
        "input_row",
        "input_object_pin",
        "source",
        "path",
        "archive_sha256",
        "disposition",
    }
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
    "tags",
    "input_object_pin",
    "archive_sha256",
    "disposition",
    "imported_id",
)


def _insert_unique(
    cursor: sqlite3.Cursor, statement: str, values: tuple[Any, ...], counter: Counter[str], name: str
) -> None:
    try:
        cursor.execute(statement, values)
    except sqlite3.IntegrityError:
        counter[name] += 1


def _count(counter: Counter[tuple[str, str, str]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for (dimension, value, disposition), count in sorted(counter.items()):
        result.setdefault(dimension, {}).setdefault(value, {})[disposition] = count
    return result


def _read_jsonl(path: StoragePath, errors: Counter[str], error_name: str, digest: HashDigest):
    if not path.exists():
        return
    with path.open("rb") as opened:
        with opened as binary_stream:
            for line in binary_stream:
                digest.update(line)
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    errors[error_name] += 1


def audit_artifacts(ledger_path: StoragePath, candidate_path: StoragePath, proof_path: StoragePath) -> dict[str, Any]:
    """Count ledger dispositions and verify candidates against accepted rows."""
    dimensions: Counter[tuple[str, str, str]] = Counter()
    disposition_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    candidate_digest = hashlib.sha256()
    proof_digest = hashlib.sha256()
    total_rows = 0

    with tempfile.TemporaryDirectory(prefix="tasktrove-ingest-audit-") as temporary_directory:
        database_path = Path(temporary_directory) / "joins.sqlite3"
        with sqlite3.connect(database_path) as connection:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE accepted_ids (id TEXT PRIMARY KEY);
                CREATE TABLE expected_candidates (
                    id TEXT PRIMARY KEY,
                    input_file TEXT,
                    input_row INTEGER,
                    input_object_pin TEXT,
                    source TEXT,
                    path TEXT,
                    archive_sha256 TEXT
                );
                CREATE TABLE candidate_ids (id TEXT PRIMARY KEY);
                CREATE TABLE proof_rows (
                    id TEXT PRIMARY KEY,
                    input_file TEXT,
                    input_row INTEGER,
                    input_object_pin TEXT,
                    source TEXT,
                    path TEXT,
                    archive_sha256 TEXT,
                    disposition TEXT
                );
                """
            )
            with ledger_path.open("rb") as opened:
                with opened as stream:
                    ledger = pq.ParquetFile(stream)
                    available_columns = set(ledger.schema_arrow.names)
                    missing_columns = set(LEDGER_COLUMNS) - available_columns
                    if missing_columns:
                        raise ValueError(f"ledger is missing required columns: {sorted(missing_columns)}")
                    for batch in ledger.iter_batches(columns=list(LEDGER_COLUMNS), batch_size=BATCH_SIZE):
                        for row in batch.to_pylist():
                            total_rows += 1
                            disposition = str(row.get("disposition") or "<missing>")
                            disposition_counts[disposition] += 1
                            values = {
                                "split": row.get("input_split"),
                                "source": row.get("source"),
                                "mode": row.get("mode"),
                                "family": row.get("family"),
                                "converter": row.get("converter"),
                            }
                            for dimension, value in values.items():
                                dimensions[
                                    (dimension, str(value) if value is not None else "<missing>", disposition)
                                ] += 1
                            tags = row.get("tags") or []
                            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                                errors["invalid_ledger_tags"] += 1
                            else:
                                for tag in set(tags):
                                    dimensions[("tag", tag, disposition)] += 1

                            if disposition != "imported":
                                continue
                            task_id = row.get("imported_id")
                            if not isinstance(task_id, str) or not task_id:
                                errors["accepted_rows_missing_id"] += 1
                                continue
                            _insert_unique(
                                cursor,
                                "INSERT INTO accepted_ids VALUES (?)",
                                (task_id,),
                                errors,
                                "duplicate_accepted_ids",
                            )
                            source = row.get("source")
                            mode = row.get("mode")
                            candidate_mode = PUBLIC_CANDIDATE_COHORTS.get(str(source))
                            if row.get("input_split") != "tasks" or candidate_mode != mode:
                                continue
                            archive_sha256 = row.get("archive_sha256")
                            if not isinstance(archive_sha256, str) or not SHA256_PATTERN.fullmatch(archive_sha256):
                                errors["eligible_rows_missing_or_invalid_archive_sha256"] += 1
                            expected = (
                                task_id,
                                row.get("input_file"),
                                row.get("input_row"),
                                row.get("input_object_pin"),
                                source,
                                row.get("path"),
                                archive_sha256,
                            )
                            if not all((expected[1], expected[2] is not None, expected[3], expected[4], expected[5])):
                                errors["eligible_rows_missing_source_proof_fields"] += 1
                            _insert_unique(
                                cursor,
                                "INSERT INTO expected_candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                                expected,
                                errors,
                                "duplicate_eligible_ledger_ids",
                            )
                        connection.commit()

            for candidate in _read_jsonl(candidate_path, errors, "malformed_candidate_json", candidate_digest):
                if not isinstance(candidate, dict) or set(candidate) != PUBLIC_FIELDS:
                    errors["candidate_allowlist_mismatch"] += 1
                    continue
                if type(candidate.get("record_version")) is not int or candidate["record_version"] != 1:
                    errors["candidate_record_version_mismatch"] += 1
                if not isinstance(candidate.get("tags"), list) or any(
                    not isinstance(tag, str) for tag in candidate["tags"]
                ):
                    errors["candidate_tags_invalid"] += 1
                if (
                    not isinstance(candidate.get("context"), dict)
                    or not isinstance(candidate.get("environment_requirements"), dict)
                    or not isinstance(candidate.get("final_tools"), list)
                    or any(not isinstance(tool, dict) for tool in candidate["final_tools"])
                    or not isinstance(candidate.get("source"), dict)
                    or not isinstance(candidate.get("submission_instruction"), str)
                    or not isinstance(candidate.get("answer_type"), str)
                    or not isinstance(candidate.get("source_category"), str)
                ):
                    errors["candidate_typed_fields_invalid"] += 1
                candidate_id = candidate.get("id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    errors["candidate_missing_id"] += 1
                    continue
                _insert_unique(
                    cursor,
                    "INSERT INTO candidate_ids VALUES (?)",
                    (candidate_id,),
                    errors,
                    "duplicate_candidate_ids",
                )

            for proof in _read_jsonl(proof_path, errors, "malformed_proof_json", proof_digest):
                if not isinstance(proof, dict) or set(proof) != PROOF_FIELDS:
                    errors["proof_schema_mismatch"] += 1
                    continue
                proof_id = proof.get("candidate_id")
                archive_sha256 = proof.get("archive_sha256")
                if not isinstance(proof_id, str) or not proof_id:
                    errors["proof_missing_id"] += 1
                    continue
                if not isinstance(archive_sha256, str) or not SHA256_PATTERN.fullmatch(archive_sha256):
                    errors["proof_missing_or_invalid_archive_sha256"] += 1
                if proof.get("disposition") != "imported":
                    errors["proof_disposition_mismatch"] += 1
                if not all(
                    (
                        proof.get("input_file"),
                        proof.get("input_row") is not None,
                        proof.get("input_object_pin"),
                        proof.get("source"),
                        proof.get("path"),
                    )
                ):
                    errors["proof_missing_source_fields"] += 1
                _insert_unique(
                    cursor,
                    "INSERT INTO proof_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        proof_id,
                        proof.get("input_file"),
                        proof.get("input_row"),
                        proof.get("input_object_pin"),
                        proof.get("source"),
                        proof.get("path"),
                        archive_sha256,
                        proof.get("disposition"),
                    ),
                    errors,
                    "duplicate_proof_ids",
                )
            connection.commit()

            join_checks = {
                "expected_candidates_missing_from_public_file": (
                    """
                    SELECT COUNT(*) FROM expected_candidates e LEFT JOIN candidate_ids c ON e.id=c.id WHERE c.id IS NULL
                """
                ),
                "unexpected_public_candidates": (
                    """
                    SELECT COUNT(*) FROM candidate_ids c LEFT JOIN expected_candidates e ON c.id=e.id WHERE e.id IS NULL
                """
                ),
                "expected_candidates_missing_proof": (
                    """
                    SELECT COUNT(*) FROM expected_candidates e LEFT JOIN proof_rows p ON e.id=p.id WHERE p.id IS NULL
                """
                ),
                "unexpected_proofs": (
                    """
                    SELECT COUNT(*) FROM proof_rows p LEFT JOIN expected_candidates e ON p.id=e.id WHERE e.id IS NULL
                """
                ),
                "proof_candidate_join_mismatches": (
                    """
                    SELECT COUNT(*) FROM expected_candidates e JOIN proof_rows p ON e.id=p.id
                    WHERE e.input_file IS NOT p.input_file OR e.input_row IS NOT p.input_row
                    OR e.input_object_pin IS NOT p.input_object_pin OR e.source IS NOT p.source
                    OR e.path IS NOT p.path OR e.archive_sha256 IS NOT p.archive_sha256
                    OR p.disposition IS NOT 'imported'
                """
                ),
                "public_candidates_missing_proof": (
                    """
                    SELECT COUNT(*) FROM candidate_ids c LEFT JOIN proof_rows p ON c.id=p.id WHERE p.id IS NULL
                """
                ),
            }
            for name, query in join_checks.items():
                count = cursor.execute(query).fetchone()[0]
                if count:
                    errors[name] += count
            accepted_unique_ids = cursor.execute("SELECT COUNT(*) FROM accepted_ids").fetchone()[0]
            expected_candidate_count = cursor.execute("SELECT COUNT(*) FROM expected_candidates").fetchone()[0]
            public_candidate_count = cursor.execute("SELECT COUNT(*) FROM candidate_ids").fetchone()[0]
            proof_count = cursor.execute("SELECT COUNT(*) FROM proof_rows").fetchone()[0]

    errors = Counter({name: count for name, count in errors.items() if count})
    return {
        "status": "passed" if not errors else "failed",
        "input_rows": total_rows,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "counts_by_dimension_and_disposition": _count(dimensions),
        "unique_accepted_ids": accepted_unique_ids,
        "eligible_ledger_candidate_ids": expected_candidate_count,
        "public_candidate_ids": public_candidate_count,
        "candidate_proof_ids": proof_count,
        "candidate_jsonl_sha256": candidate_digest.hexdigest(),
        "candidate_proof_jsonl_sha256": proof_digest.hexdigest(),
        "validation_errors": dict(sorted(errors.items())),
        "tag_counting": "A row increments once per distinct original tag; repeated tags within a row count once.",
        "archive_payload_bytes_read": 0,
    }


def audit(output_uri: str, output_dir: Path) -> dict[str, Any]:
    """Audit completed ingestion artifacts and publish a compact regional report."""
    configure_coreweave_s3()
    output_prefix = StoragePath(output_uri)
    ingest_manifest_path = output_prefix / "ingestion-manifest.json"
    ingest_manifest = json.loads(ingest_manifest_path.read_bytes())
    if ingest_manifest.get("status") != "complete":
        raise ValueError("refusing to audit a partial or failed ingestion run")
    candidate_path = output_prefix / "public-candidates.jsonl"
    proof_path = output_prefix / "candidate-proof.jsonl"
    result = audit_artifacts(
        output_prefix / "ingestion-ledger.parquet",
        candidate_path,
        proof_path,
    )
    artifacts = ingest_manifest.get("artifacts", {})
    expected_proof_count = ingest_manifest.get("public_candidate_proof_count")
    if expected_proof_count is not None and expected_proof_count != result["candidate_proof_ids"]:
        result["validation_errors"]["manifest_proof_count_mismatch"] = 1
    for artifact_name, _path, result_key in (
        ("public_candidates", candidate_path, "candidate_jsonl_sha256"),
        ("candidate_proof", proof_path, "candidate_proof_jsonl_sha256"),
    ):
        metadata = artifacts.get(artifact_name, {})
        expected_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not expected_digest and result["candidate_proof_ids"]:
            result["validation_errors"][f"manifest_{artifact_name}_sha256_missing"] = 1
        if expected_digest and expected_digest != result[result_key]:
            result["validation_errors"][f"manifest_{artifact_name}_sha256_mismatch"] = 1
    if result["validation_errors"]:
        result["status"] = "failed"
    result.update(
        {
            "ingestion_run_uri": output_uri,
            "release_uri": ingest_manifest["release_uri"],
            "release_revision": ingest_manifest["release_revision"],
            "source_manifest_sha256": ingest_manifest["source_manifest_sha256"],
            "ingestion_manifest_uri": str(ingest_manifest_path),
        }
    )
    audit_path = output_prefix / "ingestion-audit.json"
    result["audit_uri"] = str(audit_path)
    audit_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with audit_path.open("wb") as opened:
        with opened as stream:
            stream.write(audit_bytes)
    result["audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": result["status"],
        "input_rows": result["input_rows"],
        "disposition_counts": result["disposition_counts"],
        "unique_accepted_ids": result["unique_accepted_ids"],
        "eligible_ledger_candidate_ids": result["eligible_ledger_candidate_ids"],
        "public_candidate_ids": result["public_candidate_ids"],
        "candidate_proof_ids": result["candidate_proof_ids"],
        "validation_errors": result["validation_errors"],
        "audit_uri": result["audit_uri"],
        "audit_sha256": result["audit_sha256"],
    }
    (output_dir / "tasktrove-ingestion-audit-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_uri = os.environ["TASKTROVE_OUTPUT_URI"]
    result = audit(output_uri, output_dir)
    print(json.dumps({"status": result["status"], "audit_uri": result["audit_uri"]}, sort_keys=True))
    if result["status"] != "passed":
        raise ValueError(f"TaskTrove ingestion audit found validation errors: {result['validation_errors']}")


if __name__ == "__main__":
    main()
