# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive assessment records and select a campaign's catalog snapshot."""

import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from finestore.rl.curation import (
    ASSESSMENTS,
    ATTEMPTS,
    CURRENT,
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    INVENTORY,
    ITEMS,
    MAX_UNAVAILABLE_JUDGMENT_SHARE,
    POPULATIONS,
    SNAPSHOTS,
    AdmissionCounts,
    ArchivedFile,
    Assessment,
    AssessmentAttempt,
    AssessmentItem,
    AssessmentPopulation,
    AssessmentStatus,
    CatalogSnapshot,
    ControlCounts,
    ControlSummary,
    ConvertedExamples,
    CurationCatalog,
    CurationMode,
    CurationSummary,
    CurrentCatalog,
    DaytonaMeasurement,
    JudgmentStatus,
    SnapshotSource,
    SourceQualityResult,
    Verdict,
    register_catalog,
    validate_items,
)
from finestore.store import DataStore
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.assessment import assessment_inputs, assessment_judgments, validate_coverage
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignStatus
from experiments.post_training.task_curation.catalog_identity import canonical_attempt, canonical_snapshot_source
from experiments.post_training.task_curation.curation_catalog import RUNS
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.source_identity import canonical_source_id, canonical_source_map
from experiments.post_training.task_curation.source_policy import GRANDFATHERED_MARINSKYRL_SOURCES


def assessment_quality_result(items: Iterable[AssessmentItem]) -> SourceQualityResult:
    """Compute retained-sample quality independently of mechanical exclusions."""
    sample_items = [item for item in items if item.cohort == "admitted"]
    reviewed_items = [
        item
        for item in sample_items
        if item.status == JudgmentStatus.REVIEWED and item.has_task and item.verdict != Verdict.UNKNOWN
    ]
    issues = [item for item in reviewed_items if item.verdict != Verdict.GOOD]
    return SourceQualityResult(
        reviewed=len(reviewed_items),
        issue_count=len(issues),
        unavailable_count=sum(item.status != JudgmentStatus.REVIEWED for item in sample_items),
        uncertain_count=sum(
            item.status == JudgmentStatus.REVIEWED and (not item.has_task or item.verdict == Verdict.UNKNOWN)
            for item in sample_items
        ),
        issues=[f"{item.task_id}: {item.explanation}" for item in issues],
        repair_recommendation=(
            "Review the recorded issues and qualify a repaired release with SAMPLE or FULL QA."
            if issues
            else "No repair indicated by the reviewed sample."
        ),
    )


def curation_summary(manifest: dict[str, Any]) -> CurationSummary:
    verification = manifest.get("verification")
    controls = None
    if verification is not None:
        controls = ControlSummary(
            status=verification.get("status"),
            eligible_count=verification.get("eligible_count"),
            sample_count=verification.get("sample_count"),
            counts=ControlCounts.model_validate(verification.get("counts") or {}),
        )
    return CurationSummary(
        status=manifest["status"],
        mode=CurationMode(manifest["mode"]),
        **{
            key: manifest.get(key)
            for key in (
                "source_revision",
                "recipe_revision",
                "raw_population_count",
                "raw_sample_count",
                "processed_rows",
                "unprocessed_rows",
            )
        },
        admission_counts=AdmissionCounts.model_validate(manifest.get("admission_counts") or {}),
        verification=controls,
        quality_result=(
            SourceQualityResult.model_validate(manifest["source_quality"]["result"])
            if manifest.get("source_quality", {}).get("result") is not None
            else None
        ),
    )


def archive_files(store: DataStore, root: str, prefix: str, patterns: tuple[str, ...]) -> list[ArchivedFile]:
    files = []
    directory = StoragePath(root)
    for pattern in patterns:
        for path in (directory / pattern).glob():
            name = path.relative_to(directory)
            files.append(ArchivedFile(name=name, uri=store.write_object(f"{prefix}/{name}", path.read_bytes())))
    store.flush()
    return files


def publish_assessment(store: DataStore, catalog: CurationCatalog, attempt: AssessmentAttempt) -> Assessment:
    assert attempt.paths is not None and attempt.pipeline_key is not None and attempt.source_id is not None
    assert attempt.curation_uri is not None
    paths = attempt.paths
    inputs, judgments = assessment_inputs(paths.inputs_root), assessment_judgments(paths.judgments_root)
    packets = {row["task_id"]: row for path in inputs.packet_paths for row in load_parquet(path)}
    grades = [row for path in judgments.grade_paths for row in load_parquet(path)]
    validate_coverage(inputs.sample, grades)
    unavailable = sum(row["status"] != JudgmentStatus.REVIEWED for row in grades)
    if grades and unavailable / len(grades) > MAX_UNAVAILABLE_JUDGMENT_SHARE:
        raise ValueError(f"Assessment has {unavailable} unavailable judgments above the allowed share")
    if set(packets) != {row.task_id for row in inputs.sample}:
        raise ValueError("Packet records do not match the selected assessment sample")
    key = attempt.pipeline_key
    prefix = f"assessments/{key}"
    items = []
    for grade in grades:
        packet = packets[grade["task_id"]]
        evidence = json.loads(packet["packet_json"])
        status = JudgmentStatus(grade["status"])
        uri = store.write_object(
            f"{prefix}/tasks/{grade['task_id']}.json",
            json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode(),
        )
        items.append(
            AssessmentItem(
                pipeline_key=key,
                task_id=grade["task_id"],
                cohort=grade["cohort"],
                status=status,
                verdict=grade["verdict"],
                explanation=grade["explanation"],
                evidence_ids=grade["evidence_ids"],
                packet_uri=uri,
                has_task="task" in evidence,
                omissions=packet["omissions"],
                exclusion_reason=packet["exclusion_reason"],
            )
        )
    populations = [AssessmentPopulation(pipeline_key=key, **row.model_dump()) for row in inputs.populations]
    validate_items(populations, items)
    traces = archive_files(store, paths.judgments_root, prefix, ("requests/*.jsonl", "responses/*.jsonl"))
    assessment = Assessment(
        pipeline_key=key,
        source_id=attempt.source_id,
        scope=inputs.scope,
        model=judgments.model,
        rubric=judgments.rubric,
        curation=curation_summary(catalog.document(attempt.curation_uri)),
        curation_uri=attempt.curation_uri,
        traces=traces,
    )
    store.flush()
    with store.transaction() as transaction:
        transaction.table(ASSESSMENTS).add(assessment.model_dump(mode="json"))
        transaction.table(POPULATIONS).extend(row.model_dump(mode="json") for row in populations)
        transaction.table(ITEMS).extend(row.model_dump(mode="json") for row in items)
    return assessment


def published_source_starts(catalog: CurationCatalog) -> dict[str, datetime]:
    """Return each source's latest published campaign start from retained run history."""
    published = {key[0] for key in catalog.view.keys(SNAPSHOTS)}
    starts = {
        row["run_id"]: row["started_at"]
        for row in catalog.view.iter_rows(RUNS, columns=["run_id", "started_at"])
        if row["run_id"] in published
    }
    latest: dict[str, datetime] = {}
    for row in catalog.view.iter_rows(ATTEMPTS, columns=["run_id", "source_id", "status"]):
        if row["run_id"] not in starts or row["source_id"] is None or row["status"] == AssessmentStatus.SKIPPED:
            continue
        started_at = starts[row["run_id"]]
        if started_at is None:
            raise ValueError(f"Published campaign {row['run_id']} has no start time")
        started = datetime.fromisoformat(started_at)
        source_id = canonical_source_id(row["source_id"])
        if source_id not in latest or started > latest[source_id]:
            latest[source_id] = started
    return latest


def publish_catalog(root: str, run_id: str, sources: Iterable[RlDataSource]) -> CatalogSnapshot:
    """Archive selected outputs before publishing their typed catalog snapshot."""
    declarations = [source.declaration for source in sources]
    by_id = {row.id: row for row in declarations}
    if len(by_id) != len(declarations):
        raise ValueError("Catalog source IDs must be unique")
    lease = create_lock(str(StoragePath(root) / "publication.lock"), worker_id=run_id)
    if not lease.try_acquire():
        raise RuntimeError("Another campaign is publishing the catalog; retry publication")
    try:
        with lease_refresh(lease), DataStore.open(root, writer_id=f"{run_id}-catalog") as store:
            register_catalog(store)
            catalog = CurationCatalog(root)
            run = catalog.view.point(RUNS, run_id=run_id)
            if run is None or run["status"] == CampaignStatus.RUNNING:
                raise ValueError(f"No terminal curation campaign: {run_id}")
            existing = catalog.view.point(SNAPSHOTS, snapshot_id=run_id)
            if existing is not None:
                return CatalogSnapshot.model_validate({key: existing[key] for key in CatalogSnapshot.model_fields})
            attempts = [canonical_attempt(attempt) for attempt in catalog.attempts(run_id)]
            if len(attempts) != run["source_count"]:
                raise ValueError("Assessment attempts do not cover the terminal campaign")
            selected = {row.source_id: row for row in attempts}
            if len(selected) != len(attempts) or None in selected or not set(selected).issubset(by_id):
                raise ValueError("Campaign sources must be unique members of the catalog inventory")
            pointer = catalog.view.point(CURRENT, name="latest")
            previous = (
                {
                    canonical_source_id(row.id): canonical_snapshot_source(row)
                    for row in catalog.sources(pointer["snapshot_id"])
                }
                if pointer
                else {}
            )
            if run["started_at"] is None:
                raise ValueError(f"Campaign {run_id} has no start time")
            started = datetime.fromisoformat(run["started_at"])
            superseded = {
                source_id
                for source_id, prior_start in published_source_starts(catalog).items()
                if source_id in selected and source_id in previous and started < prior_start
            }
            declarations.extend(
                row.declaration
                for source_id, row in previous.items()
                if source_id not in by_id and row.declaration.pipeline is not None
            )
            completed = set(catalog.view.keys(ASSESSMENTS))
            rows = []
            for declaration in declarations:
                prior = previous.get(declaration.id)
                attempt = selected.get(declaration.id)
                if prior is not None and (attempt is None or declaration.id in superseded):
                    reason = prior.reason
                    if declaration.id in superseded:
                        reason = "; ".join(
                            part for part in (reason, f"Ignored superseded attempt from campaign {run_id}") if part
                        )
                    update = {"snapshot_id": run_id, "reason": reason}
                    if (
                        declaration.id in by_id
                        and prior.declaration.pipeline == declaration.pipeline
                        and prior.declaration.dataset_id == declaration.dataset_id
                        and prior.declaration.dataset_revision == declaration.dataset_revision
                    ):
                        update["declaration"] = declaration.model_copy(
                            update={
                                "quality_result": (
                                    assessment_quality_result(catalog.items(prior.assessment_key))
                                    if prior.assessment_key
                                    else declaration.quality_result
                                )
                            }
                        )
                    rows.append(prior.model_copy(update=update))
                    continue
                unchanged = (
                    prior is not None
                    and prior.declaration.pipeline == declaration.pipeline
                    and prior.declaration.dataset_revision == declaration.dataset_revision
                )
                key = prior.assessment_key if unchanged else None
                status = prior.latest_attempt_status if unchanged else AssessmentStatus.SKIPPED
                reason = (
                    prior.reason
                    if unchanged
                    else "No conversion pipeline" if declaration.pipeline is None else "No assessment published"
                )
                latest = prior.latest_attempt_key if unchanged else None
                summary = prior.curation if unchanged else None
                diagnostics = prior.diagnostics if unchanged else []
                if attempt is not None:
                    status, reason, latest = attempt.status, attempt.reason, attempt.pipeline_key
                    summary = curation_summary(catalog.document(attempt.curation_uri)) if attempt.curation_uri else None
                    if status == AssessmentStatus.COMPLETE:
                        assert latest is not None
                        if (latest,) not in completed:
                            publish_assessment(store, catalog, attempt)
                        key = latest
                        diagnostics = []
                    elif status == AssessmentStatus.FAILED:
                        key = latest if (latest,) in completed else None
                        if attempt.paths is not None:
                            diagnostics = archive_files(
                                store,
                                attempt.paths.judgments_root,
                                f"attempts/{run_id}/{attempt.name}",
                                ("grades/*.parquet", "requests/*.jsonl", "responses/*.jsonl"),
                            )
                if key is not None:
                    declaration = declaration.model_copy(
                        update={"quality_result": assessment_quality_result(catalog.items(key))}
                    )
                rows.append(
                    SnapshotSource(
                        snapshot_id=run_id,
                        id=declaration.id,
                        declaration=declaration,
                        assessment_key=key,
                        latest_attempt_key=latest,
                        latest_attempt_status=status,
                        reason=reason,
                        curation=summary,
                        diagnostics=diagnostics,
                    )
                )
            snapshot = CatalogSnapshot(
                snapshot_id=run_id, created_at=datetime.now(UTC).isoformat(), source_count=len(rows)
            )
            lease.refresh()
            if store.read_view().point(CURRENT, name="latest") != pointer:
                raise RuntimeError("Catalog selection changed during publication; retry publication")
            with store.transaction() as transaction:
                transaction.table(INVENTORY).extend(row.model_dump(mode="json") for row in rows)
                transaction.table(SNAPSHOTS).add(snapshot.model_dump(mode="json"))
                transaction.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=run_id).model_dump())
            return snapshot
    finally:
        lease.release()


def _refresh_snapshot(
    root: str, snapshot_id: str, transform: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> CatalogSnapshot:
    lease = create_lock(str(StoragePath(root) / "publication.lock"), worker_id=snapshot_id)
    if not lease.try_acquire():
        raise RuntimeError("Another writer is publishing the catalog; retry refresh")
    try:
        with lease_refresh(lease), DataStore.open(root, writer_id=f"{snapshot_id}-catalog") as store:
            register_catalog(store)
            catalog = CurationCatalog(root)
            pointer = catalog.view.point(CURRENT, name="latest")
            if pointer is None:
                raise ValueError("A refresh requires an existing catalog snapshot")
            if store.read_view().point(SNAPSHOTS, snapshot_id=snapshot_id) is not None:
                raise FileExistsError(f"Catalog snapshot already exists: {snapshot_id}")
            prior = list(
                catalog.view.iter_rows(
                    INVENTORY,
                    columns=list(SnapshotSource.model_fields),
                    where=[("snapshot_id", "==", pointer["snapshot_id"])],
                )
            )
            prior = [
                {
                    **row,
                    "id": canonical_source_id(row["id"]),
                    "declaration": {**row["declaration"], "id": canonical_source_id(row["declaration"]["id"])},
                }
                for row in prior
            ]
            rows = [
                SnapshotSource.model_validate({**row, "snapshot_id": snapshot_id}).model_dump(mode="json")
                for row in transform(prior)
            ]
            snapshot = CatalogSnapshot(
                snapshot_id=snapshot_id, created_at=datetime.now(UTC).isoformat(), source_count=len(rows)
            )
            lease.refresh()
            if store.read_view().point(CURRENT, name="latest") != pointer:
                raise RuntimeError("Catalog selection changed during refresh; retry publication")
            with store.transaction() as transaction:
                transaction.table(INVENTORY).extend(rows)
                transaction.table(SNAPSHOTS).add(snapshot.model_dump(mode="json"))
                transaction.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())
            return snapshot
    finally:
        lease.release()


def retire_catalog_sources(root: str, snapshot_id: str, retired_source_ids: frozenset[str]) -> CatalogSnapshot:
    """Retire selected entries without rewriting declarations, assessments or artifacts."""

    retired_source_ids = frozenset(map(canonical_source_id, retired_source_ids))

    def retired(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unknown = retired_source_ids - {row["id"] for row in records}
        if unknown:
            raise ValueError(f"Retirement IDs are not in the selected catalog: {sorted(unknown)}")
        return [row for row in records if row["id"] not in retired_source_ids]

    return _refresh_snapshot(root, snapshot_id, retired)


def refresh_catalog(
    root: str, snapshot_id: str, sources: Iterable[RlDataSource], *, retired_source_ids: frozenset[str]
) -> CatalogSnapshot:
    """Refresh matching-pin declarations and explicitly retire entries, retaining all evidence.

    This publishes a catalog snapshot, not a curation campaign. It cannot add an
    unconverted source or reuse an assessment under a changed source pin.
    """
    retired_source_ids = frozenset(map(canonical_source_id, retired_source_ids))
    declarations = {source.info.id: source.declaration for source in sources}
    catalog = CurationCatalog(root)
    if retired_source_ids & declarations.keys():
        raise ValueError("A retired source cannot also be declared in the refreshed catalog")

    def refreshed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unknown = retired_source_ids - {row["id"] for row in records}
        if unknown:
            raise ValueError(f"Retirement IDs are not in the selected catalog: {sorted(unknown)}")
        rows = []
        for record in records:
            source_id = record["id"]
            if source_id in retired_source_ids:
                continue
            declaration = declarations.get(source_id)
            if declaration is None:
                raise ValueError(f"Live catalog source has no registered declaration: {source_id}")
            if record["assessment_key"] is not None:
                declaration = declaration.model_copy(
                    update={"quality_result": assessment_quality_result(catalog.items(record["assessment_key"]))}
                )
            previous = record["declaration"]
            if (
                previous["pipeline"] != declaration.model_dump(mode="json")["pipeline"]
                or previous["dataset_revision"] != declaration.dataset_revision
                or previous["dataset_id"] != declaration.dataset_id
            ):
                raise ValueError(f"Source pin changed; run a conversion campaign before refreshing: {source_id}")
            rows.append(
                SnapshotSource.model_validate({**record, "declaration": declaration.model_dump(mode="json")}).model_dump(
                    mode="json"
                )
            )
        return rows

    return _refresh_snapshot(root, snapshot_id, refreshed)


def refresh_daytona_measurements(
    root: str,
    snapshot_id: str,
    measurements: Mapping[str, DaytonaMeasurement],
    *,
    retired_source_ids: frozenset[str],
    converted_examples: Mapping[str, ConvertedExamples] | None = None,
) -> CatalogSnapshot:
    """Apply measured actor fields to exact input pins, preserving other metadata and assessments."""

    measurements = canonical_source_map(measurements)
    converted_examples = canonical_source_map(converted_examples or {})
    retired_source_ids = frozenset(map(canonical_source_id, retired_source_ids))

    def measured(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prior_ids = {row["id"] for row in records}
        unknown = (measurements.keys() | retired_source_ids | (converted_examples or {}).keys()) - prior_ids
        if unknown:
            raise ValueError(f"Measurements or retirements are not in the selected catalog: {sorted(unknown)}")
        rows = []
        for record in records:
            source_id = record["id"]
            if source_id in retired_source_ids:
                continue
            declaration = dict(record["declaration"])
            measurement = measurements.get(source_id)
            if measurement is not None:
                if (declaration["dataset_id"], declaration["dataset_revision"]) != (
                    measurement.dataset_id,
                    measurement.dataset_revision,
                ):
                    raise ValueError(f"Daytona measurement belongs to a different dataset pin: {source_id}")
                declaration.update(measurement.model_dump(mode="json", exclude={"dataset_id", "dataset_revision"}))
                declaration["daytona_safety"] = measurement.safety.value
                declaration["daytona_snapshot_threshold"] = DAYTONA_SAFE_SNAPSHOT_LIMIT
            elif source_id in GRANDFATHERED_MARINSKYRL_SOURCES and declaration.get("daytona_exemption"):
                declaration["daytona_safety"] = None
            else:
                raise ValueError(f"Actor measurement is missing for live source: {source_id}")
            examples = (converted_examples or {}).get(source_id)
            if examples is not None:
                if (examples.source_dataset_id, examples.source_revision) != (
                    declaration["dataset_id"],
                    declaration["dataset_revision"],
                ):
                    raise ValueError(f"Converted examples belong to a different dataset pin: {source_id}")
                if declaration["pipeline"]["version"] != examples.converter_version:
                    raise ValueError(f"Converted examples belong to a different converter version: {source_id}")
                declaration["converted_examples"] = examples.model_dump(mode="json")
            rows.append({**record, "declaration": declaration})
        return rows

    return _refresh_snapshot(root, snapshot_id, measured)
