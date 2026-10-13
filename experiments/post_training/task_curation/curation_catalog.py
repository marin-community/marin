# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist campaign statistics and exact evidence snapshots in a FineStore catalog."""

import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from functools import partial
from typing import Any

from finestore.layout import CommitToken
from finestore.rl.curation import ATTEMPTS, AssessmentAttempt, AssessmentPaths, AssessmentStatus, register_catalog
from finestore.schema import arrow_schema
from finestore.store import DataStore
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.execution_telemetry import TELEMETRY_FILENAME

CATALOG_DIRECTORY = "catalog"
MANIFEST_FILENAME = "manifest.json"
FILTER_MANIFEST_FILENAME = "work/filtered/manifest.json"
REVIEW_MANIFEST_FILENAME = "review/manifest.json"


@dataclass(frozen=True)
class CurationRun:
    run_id: str
    status: str
    mode: str
    started_at: str | None
    finished_at: str | None
    report_updated_at: str
    report_path: str
    report_snapshot: str
    report_sha256: str
    status_counts_json: str
    execution_json: str | None
    source_count: int


@dataclass(frozen=True)
class CurationSource:
    run_id: str
    name: str
    atlas_id: str | None
    artifact_path: str
    campaign_status: str
    error: str | None
    status: str | None
    mode: str
    source: str | None
    source_dataset: str | None
    source_revision: str | None
    recipe_revision: str | None
    implementation_revision: str | None
    quality_revision: str | None
    verification_revision: str | None
    manifest_path: str | None
    manifest_sha256: str | None
    evidence_snapshots_json: str
    evidence_sha256_json: str
    outputs_json: str
    result_json: str
    quality_status: str | None
    verification_status: str | None
    admission_counts_json: str | None
    filter_counts_json: str | None
    attempt_started_at: str | None
    telemetry_status: str | None
    telemetry_error_type: str | None
    control_suite_json: str | None
    verification_policy_json: str | None
    source_input: str | None
    input_overrides_json: str | None
    manifest_evidence: str
    evidence_errors_json: str
    raw_population_count: int | None
    raw_sample_count: int | None
    processed_rows: int | None
    unprocessed_rows: int | None
    verification_sample_count: int | None
    verification_eligible_count: int | None
    admitted: int | None
    rejected: int | None
    deferred: int | None
    no_grader: int | None
    unverified: int | None
    controls_checked: int | None
    controls_passed: int | None
    controls_failed: int | None
    controls_defective: int | None
    controls_unsupported: int | None
    controls_infra_error: int | None
    controls_inconsistent: int | None
    raw_population_census: bool | None
    full_expansion: bool | None
    rubric_enabled: bool | None
    verification_pass_fraction: float | None
    source_wall_seconds: float | None


@dataclass(frozen=True)
class CurationPhase:
    run_id: str
    name: str
    phase: str
    status: str
    error_type: str | None
    executions_json: str
    ordinal: int
    wall_seconds: float


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
    if outcome["status"] == "queued":
        return SourceDocuments(outcome["name"], {}, ManifestEvidence.MISSING)
    root = _curation_root(outcome)
    documents = {}
    for name in (
        MANIFEST_FILENAME,
        TELEMETRY_FILENAME,
        "identity.json",
        ".artifact.json",
        FILTER_MANIFEST_FILENAME,
        REVIEW_MANIFEST_FILENAME,
    ):
        path = root / name
        if path.exists():
            with path.open("rb") as stream:
                documents[name] = stream.read()
    parsed, errors = _parsed_documents(documents)
    telemetry = parsed.get(TELEMETRY_FILENAME, {})
    attempt = telemetry.get("attempt_started_at")
    current = (
        started_at is not None
        and attempt is not None
        and datetime.fromisoformat(started_at)
        <= datetime.fromisoformat(attempt)
        <= datetime.fromisoformat(report_updated_at)
    )
    if MANIFEST_FILENAME not in documents:
        applicability = ManifestEvidence.MISSING
    elif MANIFEST_FILENAME in errors:
        applicability = ManifestEvidence.UNCONFIRMED
    elif outcome["status"] not in ("failed", "running"):
        applicability = ManifestEvidence.CURRENT_ATTEMPT if current else ManifestEvidence.RETAINED_ARTIFACT
    elif outcome.get("assessment") is not None and parsed.get(".artifact.json", {}).get("result") is not None:
        # A failed downstream judgment does not invalidate its completed curation dependency.
        applicability = ManifestEvidence.CURRENT_ATTEMPT if current else ManifestEvidence.RETAINED_ARTIFACT
    elif current and telemetry["status"] == "completed":
        applicability = ManifestEvidence.CURRENT_ATTEMPT
    else:
        applicability = ManifestEvidence.UNCONFIRMED
    return SourceDocuments(outcome["name"], documents, applicability)


def _curation_root(outcome: dict[str, Any]) -> StoragePath:
    assessment = outcome.get("assessment")
    return StoragePath(assessment["curated_root"] if assessment is not None else outcome["path"])


ASSESSMENT_FAILURES = {
    "queued": (AssessmentStatus.SKIPPED, "Source was not started"),
    "running": (AssessmentStatus.FAILED, "Source was interrupted"),
    "failed": (AssessmentStatus.FAILED, "Source failed before assessment completed"),
}


def append_curation_catalog(
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
        store.table("runs", primary_key=("run_id",), schema=arrow_schema(CurationRun))
        store.table("sources", primary_key=("run_id", "name"), schema=arrow_schema(CurationSource))
        store.table("phases", primary_key=("run_id", "name", "ordinal"), schema=arrow_schema(CurationPhase))
        register_catalog(store)
        with store.transaction() as transaction:
            report_uri = transaction.write_object(f"runs/{run_id}/report.json", report_bytes)
            transaction.table("runs").add(
                asdict(
                    CurationRun(
                        run_id=run_id,
                        status=report["status"],
                        mode=report["mode"],
                        started_at=started_at,
                        finished_at=report["updated_at"] if report["status"] != "running" else None,
                        report_updated_at=report["updated_at"],
                        report_path=report_path,
                        report_snapshot=report_uri,
                        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                        source_count=len(outcomes),
                        status_counts_json=_json(report["counts"]),
                        execution_json=_json(execution) if execution is not None else None,
                    )
                )
            )
            for outcome in outcomes:
                document = by_name[outcome["name"]]
                evidence = document.documents
                parsed, errors = _parsed_documents(evidence)
                manifest = (
                    parsed.get(MANIFEST_FILENAME, {})
                    if document.manifest_evidence != ManifestEvidence.UNCONFIRMED
                    else {}
                )
                telemetry = parsed.get(TELEMETRY_FILENAME, {})
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
                source = CurationSource(
                    run_id=run_id,
                    name=outcome["name"],
                    atlas_id=source_ids.get(outcome["name"]),
                    artifact_path=outcome["path"],
                    campaign_status=outcome["status"],
                    error=outcome.get("error"),
                    status=manifest.get("status"),
                    manifest_evidence=document.manifest_evidence.value,
                    evidence_errors_json=_json(errors),
                    mode=manifest.get("mode", report["mode"]),
                    source=manifest.get("source"),
                    source_dataset=manifest.get("source_dataset"),
                    source_revision=manifest.get("source_revision"),
                    recipe_revision=manifest.get("recipe_revision"),
                    implementation_revision=manifest.get("implementation_revision"),
                    quality_revision=manifest.get("quality_revision"),
                    verification_revision=manifest.get("verification_revision"),
                    raw_population_count=manifest.get("raw_population_count"),
                    raw_sample_count=manifest.get("raw_sample_count"),
                    raw_population_census=manifest.get("raw_population_census"),
                    full_expansion=manifest.get("full_expansion"),
                    processed_rows=(
                        manifest["input_rows"] if manifest.get("mode") == "quick" else manifest.get("processed_rows")
                    ),
                    unprocessed_rows=manifest.get("unprocessed_rows"),
                    manifest_path=(
                        str(_curation_root(outcome) / MANIFEST_FILENAME) if MANIFEST_FILENAME in evidence else None
                    ),
                    manifest_sha256=digests.get(MANIFEST_FILENAME),
                    evidence_snapshots_json=_json(snapshots),
                    evidence_sha256_json=_json(digests),
                    outputs_json=_json(manifest.get("datasets", result.get("outputs", {}))),
                    result_json=_json(result),
                    quality_status=manifest.get("quality", {}).get("status"),
                    rubric_enabled=(
                        (parsed[REVIEW_MANIFEST_FILENAME]["rubric"] is not None)
                        if "rubric" in parsed.get(REVIEW_MANIFEST_FILENAME, {})
                        else None
                    ),
                    verification_status=verification.get("status"),
                    verification_sample_count=verification.get("sample_count"),
                    verification_eligible_count=verification.get("eligible_count"),
                    verification_pass_fraction=verification.get("pass_fraction"),
                    control_suite_json=_json(verification["suite"]) if "suite" in verification else None,
                    verification_policy_json=_json(verification["policy"]) if "policy" in verification else None,
                    source_input=manifest.get("source_input"),
                    input_overrides_json=_json(manifest.get("source_file_overrides", {})) if manifest else None,
                    admission_counts_json=_json(admissions) if admissions is not None else None,
                    filter_counts_json=(
                        _json(parsed[FILTER_MANIFEST_FILENAME]) if FILTER_MANIFEST_FILENAME in parsed else None
                    ),
                    attempt_started_at=telemetry.get("attempt_started_at"),
                    source_wall_seconds=telemetry.get("source_wall_seconds", manifest.get("elapsed_seconds")),
                    telemetry_status=telemetry.get("status"),
                    telemetry_error_type=telemetry.get("error_type"),
                    admitted=admissions.get("admitted", 0) if admissions is not None else None,
                    rejected=admissions.get("rejected", 0) if admissions is not None else None,
                    deferred=admissions.get("deferred", 0) if admissions is not None else None,
                    no_grader=admissions.get("no_grader", 0) if admissions is not None else None,
                    unverified=admissions.get("unverified", 0) if admissions is not None else None,
                    controls_checked=counts.get("checked", 0) if counts is not None else None,
                    controls_passed=counts.get("passed", 0) if counts is not None else None,
                    controls_failed=counts.get("failed", 0) if counts is not None else None,
                    controls_defective=counts.get("defective", 0) if counts is not None else None,
                    controls_unsupported=counts.get("unsupported", 0) if counts is not None else None,
                    controls_infra_error=counts.get("infra_error", 0) if counts is not None else None,
                    controls_inconsistent=counts.get("inconsistent", 0) if counts is not None else None,
                )
                transaction.table("sources").add(asdict(source))
                paths = outcome.get("assessment")
                if paths is None:
                    assessment_status, assessment_reason = (
                        AssessmentStatus.SKIPPED,
                        f"Assessment disabled in {source.mode} mode",
                    )
                else:
                    assessment_status, assessment_reason = ASSESSMENT_FAILURES.get(
                        outcome["status"], (AssessmentStatus.COMPLETE, None)
                    )
                    if assessment_status == AssessmentStatus.FAILED:
                        assessment_reason = outcome.get("error") or assessment_reason
                attempt = AssessmentAttempt(
                    run_id=run_id,
                    name=outcome["name"],
                    source_id=source_ids.get(outcome["name"]),
                    pipeline_key=outcome.get("pipeline_key"),
                    status=assessment_status,
                    reason=assessment_reason,
                    paths=AssessmentPaths(**paths) if paths is not None else None,
                    curation_uri=(
                        snapshots.get(MANIFEST_FILENAME)
                        if document.manifest_evidence != ManifestEvidence.UNCONFIRMED
                        else None
                    ),
                )
                transaction.table(ATTEMPTS).add(attempt.model_dump(mode="json"))
                for ordinal, phase in enumerate(telemetry.get("phases", [])):
                    transaction.table("phases").add(
                        asdict(
                            CurationPhase(
                                run_id=run_id,
                                name=outcome["name"],
                                ordinal=ordinal,
                                phase=phase["phase"],
                                status=phase["status"],
                                wall_seconds=phase["wall_seconds"],
                                error_type=phase.get("error_type"),
                                executions_json=_json(phase["executions"]),
                            )
                        )
                    )
        assert transaction.token is not None
        return transaction.token
