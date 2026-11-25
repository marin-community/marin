# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill immutable receipts without reading trial archives or model weights."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import cache, partial

from finestore.rl.difficulty import Solver
from pydantic import TypeAdapter
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty.ledger import (
    IMPORTS_DIRECTORY,
    LEDGER_VERSION,
    append_rows,
    digest,
    read_rows,
)
from experiments.post_training.task_curation.difficulty.legacy import LegacyReasonError, legacy_gaps
from experiments.post_training.task_curation.difficulty.resource_deviations import ResourceDeviation
from experiments.post_training.task_curation.difficulty.state import CatalogSource, Gap, GapReason, LedgerRow, Outcome

LOGGER = logging.getLogger(__name__)
IMPORT_WORKERS = 32


def population_revision(sample: dict) -> str:
    return digest(
        {key: sample[key] for key in ("release", "converter_version", "converter_revision", "task_specs_sha256")}
    )


def declaration_population_revision(declaration: dict) -> str:
    census = declaration.get("actor_census_input") or {}
    pipeline = declaration.get("pipeline") or {}
    return population_revision(
        {
            "release": declaration.get("current_release"),
            "converter_version": pipeline.get("version"),
            "converter_revision": census.get("converter_revision"),
            "task_specs_sha256": census.get("task_specs_sha256"),
        }
    )


@cache
def catalog_sources(queue: str, queue_sha256: str) -> list[CatalogSource]:
    path = StoragePath(queue)
    if file_sha256(path) != queue_sha256:
        raise ValueError("Catalog queue differs from its digest")
    return [
        CatalogSource(
            row["sample"]["source_id"],
            population_revision(row["sample"]),
            tuple((task["task_id"], task["task_sha256"]) for task in row["sample"]["tasks"]),
            queue,
            queue_sha256,
            backend=row.get("backend"),
        )
        for row in json.loads(path.read_text())
    ]


@cache
def optional_json(path: StoragePath) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def receipt_row(reference: dict, sample: dict, solver: Solver, run: str, recorded_at: str) -> LedgerRow:
    path = StoragePath(reference["uri"])
    # Bucket routing binds each receipt's signing region; StoragePath.open uses
    # the ambient S3 backend instead of this per-bucket configuration.
    filesystem, key = filesystem_for(str(path))
    payload = filesystem.cat_file(key)
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"Attempt receipt changed: {path}")
    attempt = json.loads(payload)
    tasks = {task["task_id"]: task["task_sha256"] for task in sample["tasks"]}
    gaps = legacy_gaps(attempt.get("reason"))
    gap = gaps[0] if gaps else Gap(GapReason.COVERAGE_SHORTFALL)
    scored = attempt["status"] == "scored"
    if scored and attempt.get("reward") is None:
        raise ValueError(f"Scored receipt lacks a reward: {path}")
    return LedgerRow(
        population_revision(sample),
        sample["source_id"],
        tasks[attempt["task_id"]],
        solver,
        digest(str(path)),
        Outcome.SCORED if scored else Outcome.UNSCORED,
        None if scored else gap.reason,
        run,
        0.0,
        attempt.get("result", {}).get("finished_at") or recorded_at,
        task_id=attempt["task_id"],
        reward=attempt.get("reward") if scored else None,
        artifact_uri=gap.artifact_uri,
        judge_model=gap.judge_model,
        capability=gap.capability,
        evidence_uri=str(path),
        artifact_kind=gap.artifact_kind,
        judge_temperature=gap.judge_temperature,
        error_identity=gap.error_identity,
        secondary_gaps=gaps[1:],
        resource_deviation=(
            TypeAdapter(ResourceDeviation).validate_python(attempt["resource_deviation"])
            if attempt.get("resource_deviation") is not None
            else None
        ),
    )


def source_rows(run: str, prepared: dict, solver: Solver, parent: str | None) -> list[LedgerRow]:
    sample = prepared["sample"]
    source_id = sample["source_id"]
    key = hashlib.sha256(source_id.encode()).hexdigest()
    run_path = StoragePath(run)
    root = run_path / "trials" / key
    measurement = optional_json(root / "measurement.json")
    aggregate = optional_json(root / "attempts.json")
    # Interrupted sources lack an aggregate; early complete aggregates lack a
    # receipt index. Both need discovery of the immutable per-task receipts.
    references = (aggregate or {}).get("attempt_receipts", [])
    own = {}
    for ref in references:
        path = StoragePath(ref["uri"])
        try:
            path.relative_to(run_path)
        except ValueError:
            continue  # Linked ancestor receipts are imported with their original run.
        own[str(path)] = ref
    incomplete = (
        (root / "attempts" / "*" / "*" / "attempt.json").glob()
        if measurement is None or "attempt_receipts" not in (aggregate or {})
        else []
    )
    for path in incomplete:
        if str(path) not in own:
            own[str(path)] = {"uri": str(path), "sha256": file_sha256(path)}
    timestamp = (measurement or {}).get("created_at", "1970-01-01T00:00:00+00:00")
    rows = []
    failures = []
    for ref in own.values():
        try:
            rows.append(receipt_row(ref, sample, solver, run, timestamp))
        except LegacyReasonError as error:
            failures.append(f"{ref['uri']}: {error}")
    if failures:
        raise LegacyReasonError("\n".join(failures))
    revision = population_revision(sample)
    if measurement is not None:
        gaps = legacy_gaps(measurement.get("reason"))
        for gap in gaps or (Gap(GapReason.COVERAGE_SHORTFALL),):
            rows.append(
                LedgerRow(
                    revision,
                    source_id,
                    None,
                    solver,
                    digest([run, source_id, measurement["measurement_id"], asdict(gap)]),
                    Outcome.OBSERVATION,
                    gap.reason,
                    run,
                    0.0,
                    timestamp,
                    artifact_uri=gap.artifact_uri,
                    judge_model=gap.judge_model,
                    capability=gap.capability,
                    evidence_uri=str(root / "measurement.json"),
                    legacy_status=measurement["status"],
                    artifact_kind=gap.artifact_kind,
                    judge_temperature=gap.judge_temperature,
                    error_identity=gap.error_identity,
                )
            )
        inherited = 0.0
        cursor = parent
        while cursor:
            previous = StoragePath(cursor)
            old = optional_json(previous / "trials" / key / "measurement.json")
            if old is not None:
                inherited = old["gpu_hours"]
                break
            cursor = (optional_json(previous / "trials" / "parent-campaign.json") or {}).get("root")
        cost = measurement["gpu_hours"] - inherited
        if cost < -0.001:
            raise ValueError(f"Cumulative GPU cost regressed: {root}")
        # Historical receipts record a source window, not individual trial
        # durations. Keep that exact cost as a separate accounting event.
        rows.append(
            LedgerRow(
                revision,
                source_id,
                None,
                solver,
                digest([run, source_id, "cost", measurement["measurement_id"]]),
                Outcome.COST,
                None,
                run,
                max(0.0, cost),
                timestamp,
                evidence_uri=str(root / "measurement.json"),
            )
        )
    return rows


def import_run(run: str, ledger: str) -> dict:
    root = StoragePath(run)
    serving = optional_json(root / "serving.json")
    if serving is None:
        return {"run": run, "rows": 0, "state": "not_served"}
    prepared = json.loads((root / "prepared.json").read_text())
    parent = (optional_json(root / "trials" / "parent-campaign.json") or {}).get("root")
    load = partial(source_rows, run, solver=Solver(serving["solver"]), parent=parent)
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=IMPORT_WORKERS) as pool:
        futures = [(source["sample"]["source_id"], pool.submit(load, source)) for source in prepared["sources"]]
        for source_id, future in futures:
            try:
                batch = future.result()
            except LegacyReasonError as error:
                failures.append(f"{run} {source_id}: {error}")
                continue
            rows.extend(replace(row, queue_sha256=prepared["source_queue_sha256"]) for row in batch)
    if failures:
        raise LegacyReasonError("\n".join(failures))
    partition = append_rows(ledger, rows)
    report = {"run": run, "rows": len(rows), "partition": partition}
    LOGGER.info("Imported %s", json.dumps(report))
    return report


def import_runs(roots: list[str], ledger: str, refresh_roots: frozenset[str] = frozenset()) -> list[dict]:
    """Import roots and linked ancestors, reusing validated partitions unless refreshed."""
    optional_json.cache_clear()
    reports = []
    seen = set()
    remaining = list(roots)
    failures = []
    completed = {row.run_id for row in read_rows(ledger)}
    while remaining:
        root = str(StoragePath(remaining.pop(0)))
        if root in seen:
            continue
        seen.add(root)
        try:
            if root in completed and root not in refresh_roots:
                reports.append({"run": root, "state": "reused_validated_partition"})
            else:
                reports.append(import_run(root, ledger))
        except LegacyReasonError as error:
            failures.append(str(error))
            LOGGER.error("Reason audit failed: %s", error)
        parent = optional_json(StoragePath(root) / "trials" / "parent-campaign.json")
        if parent:
            remaining.append(parent["root"])
    if failures:
        raise LegacyReasonError("Complete reason audit failed:\n" + "\n".join(failures))
    receipt = {"roots": sorted(seen), "reports": reports}
    target = StoragePath(ledger) / LEDGER_VERSION / IMPORTS_DIRECTORY / f"{digest(receipt)}.json"
    target.parent.mkdirs()
    if not target.exists():
        target.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return reports
