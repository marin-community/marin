# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the complete RL inventory and terminal assessments from the curation catalog."""

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finestore.layout import BlobTables, OnConflict
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore, Transaction
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.assessment import validate_coverage
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.curation_catalog import AssessmentStatus
from experiments.post_training.task_curation.export_catalog import source_row
from experiments.post_training.task_curation.source import RlDataSource

LEGACY_REVIEW_FIELDS = {
    "quality",
    "review_url",
    "review_date",
    "review_source_revision",
    "review_verifier_revision",
    "difficulty",
    "traces",
}
VERIFICATION_SUMMARY_FIELDS = (
    "status",
    "eligible_count",
    "sample_count",
    "counts",
    "pass_fraction",
    "policy",
    "suite",
    "implementation_revision",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogAssessment:
    source_id: str
    pipeline_key: str
    report_uri: str
    assessment_json: str


@dataclass(frozen=True)
class CatalogSnapshot:
    snapshot_id: str
    created_at: str
    document_uri: str


@dataclass(frozen=True)
class CurrentCatalog:
    name: str
    snapshot_id: str
    document_uri: str


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _document(view: ReadView, uri: str) -> dict[str, Any]:
    payload = view.resolve(uri)
    assert payload is not None, f"Missing catalog document: {uri}"
    return json.loads(payload)


def _archived_files(store: DataStore, root: str, prefix: str, patterns: tuple[str, ...]) -> dict[str, str]:
    saved = {}
    root_path = StoragePath(root)
    for pattern in patterns:
        for path in (root_path / pattern).glob():
            relative = path.relative_to(root_path)
            # Commit traces independently so retries cannot overflow the packet transaction.
            with store.transaction() as transaction:
                saved[relative] = transaction.write_object(f"{prefix}/{relative}", path.read_bytes())
    return saved


def _completed_assessment(
    transaction: Transaction, view: ReadView, record: dict[str, Any], traces: dict[str, str]
) -> CatalogAssessment:
    report = json.loads(record["report_json"])
    snapshots = json.loads(record["evidence_snapshots_json"])
    inputs = _document(view, snapshots["assessment_inputs/manifest.json"])
    judgments = _document(view, snapshots["assessment_judgments/manifest.json"])
    packets = {row["task_id"]: row for path in inputs["packet_paths"] for row in load_parquet(path)}
    grades = [row for path in judgments["grade_paths"] for row in load_parquet(path)]
    validate_coverage(inputs["sample"], grades)
    key = record["pipeline_key"]
    prefix = f"assessments/{record['source_id']}/{key}"
    samples = []
    for grade in grades:
        if grade["status"] != "reviewed":
            raise ValueError(f"Completed assessment has an unavailable judgment: {grade['task_id']}")
        packet = packets[grade["task_id"]]
        uri = transaction.write_object(
            f"{prefix}/tasks/{grade['task_id']}.json",
            _json({"judgment": grade, "packet": json.loads(packet["packet_json"]), "omissions": packet["omissions"]}),
        )
        samples.append(
            {"task_id": grade["task_id"], "cohort": grade["cohort"], "verdict": grade["verdict"], "evidence_uri": uri}
        )
    report["evidence"] = {"samples": samples, "traces": traces, "snapshots": snapshots}
    report_uri = transaction.write_object(f"{prefix}/report.json", _json(report))
    assessment = {
        "status": AssessmentStatus.COMPLETE,
        "pipeline_key": key,
        "scope": report["scope"],
        "basis": report["basis"],
        "grade_rule": report["grade_rule"],
        "model": report["model"],
        "admitted": report["assessment"]["admitted"],
        "excluded": report["assessment"]["excluded"],
        "samples": samples,
        "report_uri": report_uri,
        "curation": {
            name: report["curation"].get(name)
            for name in (
                "status",
                "mode",
                "source_revision",
                "recipe_revision",
                "raw_population_count",
                "raw_sample_count",
                "processed_rows",
                "unprocessed_rows",
                "admission_counts",
                "verification",
                "source_file_overrides",
            )
        },
    }
    return CatalogAssessment(record["source_id"], key, report_uri, _json(assessment).decode())


def _unavailable_assessment(record: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    saved = json.loads(record["evidence_snapshots_json"])
    return {
        "status": record["status"],
        "pipeline_key": record["pipeline_key"],
        "reason": record["reason"],
        "diagnostics": {**saved, **files},
    }


def _inventory_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    curation = assessment.get("curation")
    if curation is None:
        return assessment
    verification = curation.get("verification")
    if verification is not None:
        verification = {name: verification.get(name) for name in VERIFICATION_SUMMARY_FIELDS}
    summary = {**assessment, "curation": {**curation, "verification": verification}}
    for cohort in ("admitted", "excluded"):
        summary[cohort] = {name: value for name, value in assessment[cohort].items() if name != "examples"}
    return summary


def _publication(
    store: DataStore, view: ReadView, run_id: str, sources: Iterable[RlDataSource], *, legacy_archive_url: str | None
) -> dict[str, Any]:
    # Stored JSON uses lists; normalize tuple fields before comparing declarations.
    declarations = [
        json.loads(
            _json({name: value for name, value in source_row(source).items() if name not in LEGACY_REVIEW_FIELDS})
        )
        for source in sources
    ]
    identifiers = {row["id"] for row in declarations}
    if len(identifiers) != len(declarations):
        raise ValueError("Catalog source IDs must be unique")
    records = list(view.iter_rows("assessment_runs", where=[("run_id", "==", run_id)]))
    run = view.point("runs", run_id=run_id)
    assert run is not None
    if len(records) != run["source_count"]:
        raise ValueError("Assessment records do not cover the terminal campaign")
    selected_ids = set()
    for record in records:
        source_id = record["source_id"]
        if source_id is None or record["pipeline_key"] is None:
            raise ValueError(f"Assessment has no source or pipeline identity: {record['name']}")
        if source_id not in identifiers:
            raise ValueError(f"Campaign source is outside the catalog inventory: {source_id}")
        if source_id in selected_ids:
            raise ValueError(f"Campaign has multiple assessments for source: {source_id}")
        selected_ids.add(source_id)
    previous = view.point("catalog_current", name="latest")
    old = {}
    if previous is not None:
        prior_document = _document(view, previous["document_uri"])
        old = {row["id"]: row for row in prior_document["sources"]}
        if legacy_archive_url is None:
            legacy_archive_url = prior_document["legacy_archive_url"]
    selected = {}
    for record in records:
        source_id = record["source_id"]
        recorded = view.point("catalog_assessments", source_id=source_id, pipeline_key=record["pipeline_key"])
        if recorded is not None:
            selected[source_id] = json.loads(recorded["assessment_json"])
            if record["status"] != AssessmentStatus.COMPLETE:
                selected[source_id]["latest_attempt_error"] = record["reason"]
            continue
        if record["status"] == AssessmentStatus.COMPLETE:
            traces = _archived_files(
                store,
                record["judgments_root"],
                f"assessments/{source_id}/{record['pipeline_key']}",
                ("requests/*.jsonl", "responses/*.jsonl"),
            )
            with store.transaction() as transaction:
                assessment = _completed_assessment(transaction, view, record, traces)
                transaction.table("catalog_assessments").add(asdict(assessment))
                selected[source_id] = json.loads(assessment.assessment_json)
        else:
            files = {}
            if record["judgments_root"] is not None:
                files = _archived_files(
                    store,
                    record["judgments_root"],
                    f"attempts/{run_id}/{record['name']}",
                    ("grades/*.parquet", "requests/*.jsonl", "responses/*.jsonl"),
                )
            selected[source_id] = _unavailable_assessment(record, files)
    rows = []
    for declaration in declarations:
        prior = old.get(declaration["id"])
        assessment = selected.get(declaration["id"])
        if assessment is None and prior is not None and prior["declaration"] == declaration:
            assessment = prior["assessment"]
        assessment = assessment or {
            "status": AssessmentStatus.SKIPPED,
            "reason": "No conversion pipeline" if declaration["pipeline"] is None else "No assessment published",
        }
        assessment = _inventory_assessment(assessment)
        row = {**declaration, "declaration": declaration, "assessment": assessment}
        row["quality"] = assessment.get("admitted", {}).get("grade")
        row["curation"] = assessment.get("curation")
        curation = row["curation"]
        if curation is not None and curation["mode"] == "full" and not curation.get("source_file_overrides"):
            row["task_count"] = curation["raw_population_count"]
        rows.append(row)
    rows.sort(key=lambda row: row["id"])
    return {
        "schema_version": 2,
        "snapshot_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "revision": hashlib.sha256(_json(rows)).hexdigest(),
        "catalog_root": store.root,
        "legacy_archive_url": legacy_archive_url,
        "sources": rows,
    }


def publish_catalog(
    root: str, run_id: str, sources: Iterable[RlDataSource], *, legacy_archive_url: str | None = None
) -> dict[str, Any]:
    """Archive cited evidence, then atomically select a terminal campaign's catalog snapshot."""
    view = ReadView(root)
    run = view.point("runs", run_id=run_id)
    if run is None or run["status"] == "running":
        raise ValueError(f"No terminal curation catalog invocation: {run_id}")
    root_path = StoragePath(root)
    lease = create_lock(str(root_path / "publication.lock"), worker_id=run_id)
    if not lease.try_acquire():
        raise RuntimeError("Another campaign is publishing the catalog; retry publication")
    try:
        with lease_refresh(lease), DataStore.open(root, writer_id=f"{run_id}-catalog") as store:
            existing = store.read_view().point("catalog_snapshots", snapshot_id=run_id)
            if existing is not None:
                return _document(store.read_view(), existing["document_uri"])
            store.table(
                "catalog_assessments", primary_key=("source_id", "pipeline_key"), schema=arrow_schema(CatalogAssessment)
            )
            store.table("catalog_snapshots", primary_key=("snapshot_id",), schema=arrow_schema(CatalogSnapshot))
            store.table(
                "catalog_current",
                primary_key=("name",),
                schema=arrow_schema(CurrentCatalog),
                on_conflict=OnConflict.SUPERSEDE,
            )
            document = _publication(store, store.read_view(), run_id, sources, legacy_archive_url=legacy_archive_url)
            # Per-file evidence transactions leave separate shards for point readers to reopen.
            result = store.compact(BlobTables.DESCRIPTORS)
            logger.info("Catalog blob descriptor compaction: %s", result)
            with store.transaction() as transaction:
                uri = transaction.write_object(f"catalog/snapshots/{run_id}.json", _json(document))
                transaction.table("catalog_snapshots").add(asdict(CatalogSnapshot(run_id, document["created_at"], uri)))
                transaction.table("catalog_current").add(asdict(CurrentCatalog("latest", run_id, uri)))
            return document
    finally:
        lease.release()


def current_catalog(root: str) -> dict[str, Any]:
    """Read the complete snapshot selected by the catalog's current pointer."""
    view = ReadView(root)
    current = view.point("catalog_current", name="latest")
    if current is None:
        raise ValueError(f"No published catalog at {root}")
    return _document(view, current["document_uri"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the curation catalog's published snapshot.")
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = current_catalog(args.catalog_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_json(document) + b"\n")


if __name__ == "__main__":
    main()
