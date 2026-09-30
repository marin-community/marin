# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write rights-cleared TaskTrove rows with source proof, without archives."""

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
RIGHTS_AUDIT_MANIFEST_SHA256 = "ab71556290d1ce54e596588b549de95ddd68366cd8fd3bb18b77ed9e99f0eed1"
RIGHTS_INVENTORY_SHA256 = "165124b2f8fa95b3c2d013136eb5be0638cf0a170142cc0b4bfebd5297de0389"
RIGHTS_CLEARANCES = {
    (
        "laion__nemotron-gym-knowledge-mcqa-v2",
        "mcq",
        "qa-short-answer",
        "nemotron_mcqa",
        "c814af4f124d",
        "tasks",
    ): {
        "source_card_revision": "5d35ead3ba07abda719b3d24f6f395fee8108efd",
        "license": "CC-BY-4.0",
        "attribution": "NVIDIA",
        "expected_rows": 23_711,
    },
    (
        "laion__nemo-prism-math-v3",
        "math",
        "math-answer",
        "nemotron_math",
        "5ee94cf985a9",
        "tasks",
    ): {
        "source_card_revision": "8a35a0602167ad1f1ec9d6db5e72281e486738a7",
        "license": "CC-BY-4.0",
        "attribution": "NVIDIA",
        "expected_rows": 2_219,
    },
}


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


def _rights_inventory_sha256(rights_audit: dict[str, Any]) -> str:
    inventory = rights_audit.get("rights_terms_by_source_mode_family_converter")
    if not isinstance(inventory, list):
        raise ValueError("Rights audit is missing its normalized rights-term inventory")
    canonical = json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


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
    clearance_audit_manifest_sha256: str | None = None,
    clearances: dict[tuple[str, str, str, str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join, rights-gate, and project candidate rows with exact source proof."""
    if ingestion_manifest.get("status") != "complete":
        raise ValueError("Cannot export accepted records from an incomplete ingestion run")
    declared_holdouts = _declared_holdouts(ingestion_manifest.get("upstream_tasktrove"))
    if declared_holdouts:
        raise ValueError(f"Pinned TaskTrove metadata declares holdouts requiring split review: {declared_holdouts}")
    if not re.fullmatch(r"[0-9a-f]{40}", builder_revision):
        raise ValueError("Projection builder revision must be a full Git commit")
    if clearance_audit_manifest_sha256 not in (None, RIGHTS_AUDIT_MANIFEST_SHA256):
        raise ValueError("Rights clearance is bound to a different audited manifest")
    clearance_manifest_verified = clearance_audit_manifest_sha256 == RIGHTS_AUDIT_MANIFEST_SHA256
    clearance_rules = RIGHTS_CLEARANCES if clearances is None else clearances
    audit = audit_artifacts(ledger_path, candidate_path, proof_path)
    if audit["status"] != "passed":
        raise ValueError(f"TaskTrove candidate proof audit failed: {audit['validation_errors']}")

    counters: Counter[tuple[str, str, str, str, str, str, str, str]] = Counter()
    disposition_counts: Counter[str] = Counter()
    tag_dispositions: Counter[tuple[str, str, str, str, str, str, str, str, str]] = Counter()
    rejection_reasons: Counter[tuple[str, str, str, str, str, str, str]] = Counter()
    math_sample_pointers: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    cohort_dispositions: Counter[tuple[str, str, str]] = Counter()
    cohort_eligible_rows: Counter[str] = Counter()
    cohort_converted_rows: Counter[str] = Counter()
    cohort_archive_rows: Counter[str] = Counter()
    rights_term_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    rights_examples: dict[tuple[str, str, str, str, str], tuple[str, str]] = {}
    source_assets: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    group_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
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
                CREATE TABLE public_records (
                    id TEXT PRIMARY KEY, source TEXT, family TEXT, split TEXT, record_json TEXT
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
                    disposition_counts[disposition] += 1
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
                    group = (source, str(row.get("family") or "<missing>"), split)
                    source_assets.setdefault(group, set()).add((row["input_file"], row["input_object_pin"]))
                    group_counts[
                        (
                            source,
                            str(row.get("family") or "<missing>"),
                            str(row.get("mode") or "<missing>"),
                            str(row.get("converter") or "<missing>"),
                            str(row.get("template_id") or "<missing>"),
                            split,
                        )
                    ] += 1

            if clearance_manifest_verified:
                observed_clearance_groups = {
                    (source, mode, family, converter, template_id, split)
                    for source, family, mode, converter, template_id, split in group_counts
                }
                expected_clearance_groups = set(clearance_rules)
                if observed_clearance_groups != expected_clearance_groups:
                    raise ValueError(
                        "Candidate source/family/mode/converter/template/split groups differ from reviewed clearance: "
                        f"observed={sorted(observed_clearance_groups)}, expected={sorted(expected_clearance_groups)}"
                    )
                for source, mode, family, converter, template_id, split in sorted(expected_clearance_groups):
                    expected_rows = clearance_rules[(source, mode, family, converter, template_id, split)][
                        "expected_rows"
                    ]
                    observed_rows = group_counts[(source, family, mode, converter, template_id, split)]
                    if observed_rows != expected_rows:
                        raise ValueError(
                            f"Clearance cohort {source}/{template_id} has {observed_rows} rows, expected {expected_rows}"
                        )

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
                clearance_key = (source, str(mode), str(family), str(converter), str(template_id), str(split))
                clearance = clearance_rules.get(clearance_key)
                if not clearance_manifest_verified or clearance is None:
                    reason = "awaiting-exact-cohort-clearance"
                elif any(value != "<absent>" for _field, value in terms):
                    reason = "source-rights-metadata-requires-review"
                else:
                    source_proof = {
                        "source_row": expected_row,
                        "input_file": input_file,
                        "input_object_pin": object_pin,
                        "archive_path": archive_path,
                        "archive_sha256": archive_sha256,
                    }
                    record = {"task": candidate, "source_proof": source_proof}
                    cursor.execute(
                        "INSERT INTO public_records VALUES (?, ?, ?, ?, ?)",
                        (task_id, source, family, split, json.dumps(record, separators=(",", ":"), sort_keys=True)),
                    )
                    continue
                cursor.execute("INSERT INTO rights_rejected VALUES (?, ?, ?)", (task_id, source, reason))
            connection.commit()

            expected_count = cursor.execute("SELECT COUNT(*) FROM expected").fetchone()[0]
            proof_count = cursor.execute("SELECT COUNT(*) FROM proofs").fetchone()[0]
            accepted_count = cursor.execute("SELECT COUNT(*) FROM public_records").fetchone()[0]
            rights_rejected_count = cursor.execute("SELECT COUNT(*) FROM rights_rejected").fetchone()[0]
            rights_rejected_counts: Counter[tuple[str, str]] = Counter()
            for source, reason, count in cursor.execute(
                "SELECT source, reason, COUNT(*) FROM rights_rejected GROUP BY source, reason"
            ):
                rights_rejected_counts[(str(source), str(reason))] = count
            if expected_count == 0:
                raise ValueError("No accepted TaskTrove candidates are available for export")
            if expected_count != accepted_count + rights_rejected_count or expected_count != proof_count:
                raise ValueError(
                    f"Candidate joins are incomplete: ledger={expected_count}, accepted={accepted_count}, "
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

            output_counts: Counter[tuple[str, str, str]] = Counter()
            output_hashes: dict[tuple[str, str, str], Any] = {}
            output_sizes: Counter[tuple[str, str, str]] = Counter()
            output_paths: dict[tuple[str, str, str], str] = {}
            output_groups = list(
                cursor.execute("SELECT DISTINCT source, family, split FROM public_records ORDER BY 1, 2, 3")
            )
            destination.mkdirs(exist_ok=True)
            for source, family, split in output_groups:
                group = (source, family, split)
                public_split = {"tasks": "train"}.get(split)
                if public_split is None:
                    raise ValueError(f"No public split mapping exists for TaskTrove split {split!r}")
                output_path = destination / "tasktrove_clean" / public_split / source / f"{family}.jsonl"
                output_path.parent.mkdirs(exist_ok=True)
                output_paths[group] = str(output_path)
                output_hashes[group] = hashlib.sha256()
                with output_path.open("wb") as opened:
                    with opened as stream:
                        rows = cursor.execute(
                            "SELECT record_json FROM public_records WHERE source=? AND family=? AND split=? ORDER BY id",
                            group,
                        )
                        for (record_json,) in rows:
                            encoded = (record_json + "\n").encode()
                            stream.write(encoded)
                            output_hashes[group].update(encoded)
                            output_sizes[group] += len(encoded)
                            output_counts[group] += 1
    output_files = {
        f"{source}/{family}/{split}": {
            "uri": output_paths[(source, family, split)],
            "rows": output_counts[(source, family, split)],
            "size_bytes": output_sizes[(source, family, split)],
            "sha256": output_hashes[(source, family, split)].hexdigest(),
        }
        for source, family, split in sorted(output_paths)
    }
    cohort_counts: dict[str, Any] = {}
    for source, family, split in sorted({(key[0], key[1], key[5]) for key in group_counts}):
        cohort_name = f"{source}/{family}/{split}"
        cohort_rows = sum(
            count
            for (
                candidate_source,
                candidate_family,
                _mode,
                _converter,
                _template,
                candidate_split,
            ), count in group_counts.items()
            if (candidate_source, candidate_family, candidate_split) == (source, family, split)
        )
        cohort_counts[cohort_name] = {
            "candidate_ledger_rows": cohort_rows,
            "eligible_mode_rows_in_tasks_split": cohort_eligible_rows[source] if split == "tasks" else 0,
            "archive_rows_hashed_before_conversion": cohort_archive_rows[source] if split == "tasks" else 0,
            "converter_completed_rows": cohort_converted_rows[source] if split == "tasks" else 0,
            "accepted_public_records": sum(
                count
                for (out_source, out_family, out_split), count in output_counts.items()
                if (out_source, out_family, out_split) == (source, family, split)
            ),
            "ledger_dispositions": {
                disposition: count
                for (candidate_source, disposition, _split), count in sorted(cohort_dispositions.items())
                if candidate_source == source and _split == split
            },
            "source_assets": [
                {"path": path, "input_object_pin": pin} for path, pin in sorted(source_assets[(source, family, split)])
            ],
        }
    accepted_by_source: Counter[str] = Counter()
    for (source, _family, _split), count in output_counts.items():
        accepted_by_source[source] += count
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
        "source_assets_by_source_family_split": {
            f"{source}/{family}/{split}": [{"path": path, "pin": pin} for path, pin in sorted(assets)]
            for (source, family, split), assets in sorted(source_assets.items())
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
        "input_rows_total": sum(counters.values()),
        "input_rows_by_disposition": [
            {"disposition": disposition, "rows": count} for disposition, count in sorted(disposition_counts.items())
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
        "accepted_rows_by_source": dict(sorted(accepted_by_source.items())),
        "accepted_rows_by_source_family_split": {
            f"{source}/{family}/{split}": count for (source, family, split), count in sorted(output_counts.items())
        },
        "clearance_audit_manifest_sha256": clearance_audit_manifest_sha256,
        "clearance_rights_inventory_sha256": RIGHTS_INVENTORY_SHA256 if clearance_manifest_verified else None,
        "rights_clearance_source_cards": {
            f"{source}/{family}/{split}": clearance
            for (source, _mode, family, _converter, _template_id, split), clearance in clearance_rules.items()
        },
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
    rights_audit_uri = os.environ["TASKTROVE_RIGHTS_AUDIT_MANIFEST_URI"]
    builder_revision = os.environ["TASKTROVE_PROJECTION_REVISION"]
    ingest_prefix = StoragePath(output_uri)
    manifest_path = ingest_prefix / "ingestion-manifest.json"
    ingestion_manifest = json.loads(manifest_path.read_bytes())
    rights_audit_bytes = StoragePath(rights_audit_uri).read_bytes()
    rights_audit_sha256 = hashlib.sha256(rights_audit_bytes).hexdigest()
    rights_audit = json.loads(rights_audit_bytes)
    if rights_audit.get("status") != "complete":
        raise ValueError("Rights clearance requires a complete regional metadata audit")
    if _rights_inventory_sha256(rights_audit) != RIGHTS_INVENTORY_SHA256:
        raise ValueError("Rights-term inventory differs from the reviewed rights clearance")
    if rights_audit.get("source_manifest_sha256") != ingestion_manifest.get("source_manifest_sha256"):
        raise ValueError("Rights audit and ingestion run use different TaskTrove source manifests")
    for artifact in ("ledger", "private_catalog", "public_candidates", "candidate_proof"):
        audit_key = {
            "ledger": "input_ledger_sha256",
            "private_catalog": "private_catalog_sha256",
            "public_candidates": "input_candidate_sha256",
            "candidate_proof": "input_proof_sha256",
        }[artifact]
        if rights_audit.get(audit_key) != ingestion_manifest["artifacts"][artifact]["sha256"]:
            raise ValueError(f"Rights audit does not verify current ingestion artifact {artifact}")
    report = export_accepted_records(
        ingest_prefix / "ingestion-ledger.parquet",
        ingest_prefix / "private-catalog.parquet",
        ingest_prefix / "public-candidates.jsonl",
        ingest_prefix / "candidate-proof.jsonl",
        StoragePath(accepted_uri),
        ingestion_manifest=ingestion_manifest,
        builder_revision=builder_revision,
        clearance_audit_manifest_sha256=rights_audit_sha256,
    )
    summary = {
        "status": report["status"],
        "release_uri": report["release_uri"],
        "release_revision": report["release_revision"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "ingestion_manifest_uri": report["ingestion_manifest_uri"],
        "audit_manifest_uri": report["manifest_uri"],
        "input_ledger_sha256": report["input_ledger_sha256"],
        "private_catalog_sha256": report["private_catalog_sha256"],
        "input_candidate_sha256": report["input_candidate_sha256"],
        "input_proof_sha256": report["input_proof_sha256"],
        "input_rows_total": report["input_rows_total"],
        "input_rows_by_disposition": report["input_rows_by_disposition"],
        "ledger_bytes_hashed": report["ledger_bytes_hashed"],
        "private_catalog_bytes_hashed": report["private_catalog_bytes_hashed"],
        "candidate_sha256_verified": report["candidate_sha256_verified"],
        "proof_sha256_verified": report["proof_sha256_verified"],
        "archive_payload_bytes_read": report["archive_payload_bytes_read"],
        "private_task_specifications_loaded": report["private_task_specifications_loaded"],
        "accepted_rows_by_source": report["accepted_rows_by_source"],
        "accepted_rows_by_source_family_split": report["accepted_rows_by_source_family_split"],
        "cohort_counts": report["cohort_counts"],
        "outputs": report["outputs"],
        "rights_clearance_audit_manifest_sha256": report["clearance_audit_manifest_sha256"],
        "rights_clearance_inventory_sha256": report["clearance_rights_inventory_sha256"],
        "rights_clearance_source_cards": report["rights_clearance_source_cards"],
        "rights_rejected_rows_by_source_reason": report["rights_rejected_rows_by_source_reason"],
        "rights_terms_by_source_mode_family_converter": report["rights_terms_by_source_mode_family_converter"],
        "input_rows_by_source_split_mode_family_converter_disposition": report[
            "input_rows_by_source_split_mode_family_converter_disposition"
        ],
        "input_rows_by_source_split_mode_family_converter_tag_disposition": report[
            "input_rows_by_source_split_mode_family_converter_tag_disposition"
        ],
        "rejected_math_sample_pointers": report["rejected_math_sample_pointers"],
        "source_split_policy": report["source_split_policy"],
    }
    output_directory = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "audit-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
