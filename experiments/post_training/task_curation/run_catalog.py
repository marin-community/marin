# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist campaign statistics and exact evidence snapshots in a FineStore catalog."""

import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum
from functools import partial
from typing import Any

import pyarrow as pa
from finestore.layout import CommitToken
from finestore.store import DataStore
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import Admission
from taskcompendium.pipeline.source_verification import VerificationCounts

CATALOG_DIRECTORY = "catalog"
ADMISSIONS = tuple(admission.value for admission in Admission)
CONTROL_COUNTS = tuple(field.name for field in fields(VerificationCounts))
SOURCE_FIELDS = (
    "source",
    "source_dataset",
    "source_revision",
    "recipe_revision",
    "implementation_revision",
    "quality_revision",
    "verification_revision",
    "raw_population_count",
    "raw_sample_count",
    "raw_population_census",
    "full_expansion",
    "processed_rows",
    "unprocessed_rows",
)
SOURCE_STRING_FIELDS = (
    "run_id",
    "name",
    "atlas_id",
    "artifact_path",
    "campaign_status",
    "error",
    "status",
    "mode",
    "source",
    "source_dataset",
    "source_revision",
    "recipe_revision",
    "implementation_revision",
    "quality_revision",
    "verification_revision",
    "manifest_path",
    "manifest_sha256",
    "evidence_snapshots_json",
    "evidence_sha256_json",
    "outputs_json",
    "result_json",
    "quality_status",
    "verification_status",
    "admission_counts_json",
    "filter_counts_json",
    "attempt_started_at",
    "telemetry_status",
    "telemetry_error_type",
    "control_suite_json",
    "verification_policy_json",
    "source_input",
    "input_overrides_json",
    "manifest_evidence",
    "evidence_errors_json",
)
SOURCE_SCHEMA = pa.schema(
    [
        *((name, pa.string()) for name in SOURCE_STRING_FIELDS),
        *(
            (name, pa.int64())
            for name in (
                "raw_population_count",
                "raw_sample_count",
                "processed_rows",
                "unprocessed_rows",
                "verification_sample_count",
                "verification_eligible_count",
                *ADMISSIONS,
                *(f"controls_{name}" for name in CONTROL_COUNTS),
            )
        ),
        *((name, pa.bool_()) for name in ("raw_population_census", "full_expansion", "rubric_enabled")),
        ("verification_pass_fraction", pa.float64()),
        ("source_wall_seconds", pa.float64()),
    ]
)
RUN_SCHEMA = pa.schema(
    [
        *(
            (name, pa.string())
            for name in (
                "run_id",
                "status",
                "mode",
                "started_at",
                "finished_at",
                "report_updated_at",
                "report_path",
                "report_snapshot",
                "report_sha256",
                "status_counts_json",
                "execution_json",
            )
        ),
        ("source_count", pa.int64()),
    ]
)
PHASE_SCHEMA = pa.schema(
    [
        *((name, pa.string()) for name in ("run_id", "name", "phase", "status", "error_type", "executions_json")),
        ("ordinal", pa.int64()),
        ("wall_seconds", pa.float64()),
    ]
)


@dataclass(frozen=True)
class SourceDocuments:
    """Small JSON evidence for an artifact, optionally supplied from a saved attempt."""

    name: str
    documents: Mapping[str, bytes]
    manifest_evidence: "ManifestEvidence"


class ManifestEvidence(StrEnum):
    CURRENT_ATTEMPT = "current_attempt"
    RETAINED_ARTIFACT = "retained_artifact"
    SAVED_SNAPSHOT = "saved_snapshot"
    UNCONFIRMED = "unconfirmed"
    MISSING = "missing"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parsed_documents(evidence: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, str]]:
    parsed, errors = {}, {}
    for name, payload in evidence.items():
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            errors[name] = str(error)
            continue
        if not isinstance(document, dict):
            errors[name] = "Expected a JSON object"
            continue
        parsed[name] = document
    return parsed, errors


def _source_documents(outcome: dict[str, Any], *, started_at: str | None, report_updated_at: str) -> SourceDocuments:
    if outcome["status"] in ("queued", "running"):
        return SourceDocuments(outcome["name"], {}, ManifestEvidence.MISSING)
    root = StoragePath(outcome["path"])
    documents = {}
    for name in (
        "manifest.json",
        "telemetry.json",
        "identity.json",
        ".artifact.json",
        "work/filtered/manifest.json",
        "review/manifest.json",
    ):
        path = root / name
        if path.exists():
            with path.open("rb") as stream:
                documents[name] = stream.read()
    parsed, errors = _parsed_documents(documents)
    telemetry = parsed.get("telemetry.json", {})
    attempt = telemetry.get("attempt_started_at")
    current = (
        started_at is not None
        and attempt is not None
        and datetime.fromisoformat(started_at)
        <= datetime.fromisoformat(attempt)
        <= datetime.fromisoformat(report_updated_at)
    )
    if "manifest.json" not in documents:
        applicability = ManifestEvidence.MISSING
    elif "manifest.json" in errors:
        applicability = ManifestEvidence.UNCONFIRMED
    elif outcome["status"] != "failed":
        applicability = ManifestEvidence.CURRENT_ATTEMPT if current else ManifestEvidence.RETAINED_ARTIFACT
    elif current and telemetry["status"] == "completed":
        applicability = ManifestEvidence.CURRENT_ATTEMPT
    else:
        applicability = ManifestEvidence.UNCONFIRMED
    return SourceDocuments(outcome["name"], documents, applicability)


def append_run_catalog(
    root: str,
    report_bytes: bytes,
    *,
    run_id: str,
    report_path: str,
    source_ids: Mapping[str, str],
    started_at: str | None = None,
    execution: Mapping[str, Any] | None = None,
    documents: tuple[SourceDocuments, ...] | None = None,
) -> CommitToken:
    """Atomically append a campaign and its source/phase statistics.

    The invocation's identity differs from each artifact's original attempt. Reusing
    an artifact retains its original timing; it is not new execution throughput.
    Saved documents permit historical imports without rereading mutable artifact paths.
    Missing evidence leaves nullable statistics, including for queued or failed sources.
    """
    report = json.loads(report_bytes)
    outcomes = report["sources"]
    if len({outcome["name"] for outcome in outcomes}) != len(outcomes):
        raise ValueError("Catalog source names must be unique within a run")
    if documents is None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            documents = tuple(
                executor.map(
                    partial(_source_documents, started_at=started_at, report_updated_at=report["updated_at"]), outcomes
                )
            )
    by_name = {item.name: item for item in documents}
    if len(by_name) != len(documents) or set(by_name) != {outcome["name"] for outcome in outcomes}:
        raise ValueError("Catalog documents must cover exactly the reported sources")
    with DataStore.open(root, writer_id=run_id) as store:
        if store.read_view().point("runs", run_id=run_id) is not None:
            raise ValueError(f"Catalog invocation already exists: {run_id}")
        store.table("runs", primary_key=("run_id",), schema=RUN_SCHEMA)
        store.table("sources", primary_key=("run_id", "name"), schema=SOURCE_SCHEMA)
        store.table("phases", primary_key=("run_id", "name", "ordinal"), schema=PHASE_SCHEMA)
        with store.transaction() as transaction:
            report_uri = transaction.write_object(f"runs/{run_id}/report.json", report_bytes)
            transaction.table("runs").add(
                {
                    "run_id": run_id,
                    "status": report["status"],
                    "mode": report["mode"],
                    "started_at": started_at,
                    "finished_at": report["updated_at"] if report["status"] != "running" else None,
                    "report_updated_at": report["updated_at"],
                    "report_path": report_path,
                    "report_snapshot": report_uri,
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "source_count": len(outcomes),
                    "status_counts_json": _json(report["counts"]),
                    "execution_json": _json(execution) if execution is not None else None,
                }
            )
            for outcome in outcomes:
                document = by_name[outcome["name"]]
                evidence = document.documents
                parsed, errors = _parsed_documents(evidence)
                manifest = (
                    parsed.get("manifest.json", {}) if document.manifest_evidence != ManifestEvidence.UNCONFIRMED else {}
                )
                telemetry = parsed.get("telemetry.json", {})
                verification = manifest.get("verification", {})
                admissions = manifest.get("admission_counts")
                counts = verification.get("counts")
                result = outcome.get("result") or {}
                snapshots = {}
                digests = {}
                for name, payload in evidence.items():
                    digest = hashlib.sha256(payload).hexdigest()
                    snapshots[name] = transaction.write_object(f"evidence/{digest}/{name}", payload)
                    digests[name] = digest
                source = {
                    "run_id": run_id,
                    "name": outcome["name"],
                    "atlas_id": source_ids.get(outcome["name"]),
                    "artifact_path": outcome["path"],
                    "campaign_status": outcome["status"],
                    "error": outcome.get("error"),
                    "status": manifest.get("status"),
                    "manifest_evidence": document.manifest_evidence.value,
                    "evidence_errors_json": _json(errors),
                    "mode": manifest.get("mode", report["mode"]),
                    **{name: manifest.get(name) for name in SOURCE_FIELDS},
                    "manifest_path": (
                        str(StoragePath(outcome["path"]) / "manifest.json") if "manifest.json" in evidence else None
                    ),
                    "manifest_sha256": digests.get("manifest.json"),
                    "evidence_snapshots_json": _json(snapshots),
                    "evidence_sha256_json": _json(digests),
                    "outputs_json": _json(manifest.get("datasets", result.get("outputs", {}))),
                    "result_json": _json(result),
                    "quality_status": manifest.get("quality", {}).get("status"),
                    "rubric_enabled": (
                        (parsed["review/manifest.json"]["rubric"] is not None)
                        if "rubric" in parsed.get("review/manifest.json", {})
                        else None
                    ),
                    "verification_status": verification.get("status"),
                    "verification_sample_count": verification.get("sample_count"),
                    "verification_eligible_count": verification.get("eligible_count"),
                    "verification_pass_fraction": verification.get("pass_fraction"),
                    "control_suite_json": _json(verification["suite"]) if "suite" in verification else None,
                    "verification_policy_json": _json(verification["policy"]) if "policy" in verification else None,
                    "source_input": manifest.get("source_input"),
                    "input_overrides_json": _json(manifest.get("source_file_overrides", {})) if manifest else None,
                    "admission_counts_json": _json(admissions) if admissions is not None else None,
                    "filter_counts_json": (
                        _json(parsed["work/filtered/manifest.json"]) if "work/filtered/manifest.json" in parsed else None
                    ),
                    "attempt_started_at": telemetry.get("attempt_started_at"),
                    "source_wall_seconds": telemetry.get("source_wall_seconds", manifest.get("elapsed_seconds")),
                    "telemetry_status": telemetry.get("status"),
                    "telemetry_error_type": telemetry.get("error_type"),
                    **{name: admissions.get(name, 0) if admissions is not None else None for name in ADMISSIONS},
                    **{
                        f"controls_{name}": counts.get(name, 0) if counts is not None else None
                        for name in CONTROL_COUNTS
                    },
                }
                if manifest.get("mode") == "quick":
                    source["processed_rows"] = manifest["input_rows"]
                transaction.table("sources").add(source)
                for ordinal, phase in enumerate(telemetry.get("phases", [])):
                    transaction.table("phases").add(
                        {
                            "run_id": run_id,
                            "name": outcome["name"],
                            "ordinal": ordinal,
                            "phase": phase["phase"],
                            "status": phase["status"],
                            "wall_seconds": phase["wall_seconds"],
                            "error_type": phase.get("error_type"),
                            "executions_json": _json(phase["executions"]),
                        }
                    )
        assert transaction.token is not None
        return transaction.token
