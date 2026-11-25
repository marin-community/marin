# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a published curation catalog into Atlas."""

import json
import time
import uuid
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import click
import pyarrow.parquet as pq
from finestore.rl.curation import (
    MAX_UNAVAILABLE_JUDGMENT_SHARE,
    AdmissionCounts,
    Assessment,
    AssessmentItem,
    AssessmentPopulation,
    CurationCatalog,
    CurationMode,
    CurationSummary,
    JudgmentStatus,
    ModelReference,
    SamplePanelCurrency,
    Verdict,
    require_typed_status_fields,
    validate_items,
)
from finestore.rl.status_taxonomy import PendingStage, SourceStatus
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import NetworkPolicy
from shellbox.requirements import Requirement
from taskcompendium.backend_requirements import BackendSourceMap, BackendSurvey, backend_schema_sha256
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus
from taskcompendium.source_fingerprints import SourceIdentitySurvey

from experiments.post_training.task_curation.catalog_identity import (
    backend_survey_payload,
    canonical_assessment,
    canonical_backend_survey,
    canonical_identity_survey,
    canonical_snapshot_source,
    identity_survey_payload,
)
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.method_variants import source_variants
from experiments.post_training.task_curation.retirement import retired_admission_source_ids
from experiments.post_training.task_curation.review_quality import quality_result as recompute_review_result
from experiments.post_training.task_curation.source_status import served_snapshot_sources
from infra.marina.apps.rl_data_catalog.historical_assessments import retain_prior_assessments
from infra.marina.apps.rl_data_catalog.load_counts import staged_row_count
from infra.marina.apps.rl_data_catalog.prepare_preview import PREVIEW_APP_ID
from infra.marina.apps.rl_data_catalog.server.admission import (
    backend_admission_review,
    has_runnable_production_support,
    require_available_lexical_clearance,
    require_content_result,
    require_declaration_admission,
    require_sample_pin_currency,
)
from infra.marina.apps.rl_data_catalog.server.cohort_summaries import materialize_cohort_summaries
from infra.marina.apps.rl_data_catalog.server.content_access import content_is_withheld
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.quality import Quality, quality_display, review_completeness
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL

MAX_BATCH_BYTES = 1_000_000
STAGING_TABLE_PREFIX = "atlas_"
# The per-task staging tables carry many small rows; their inserts maintain four
# indexes (including the GIN defects index) on the retained item volume, so their
# statements stay under the SQL gateway's per-statement timeout only with small
# batches — bounded both by bytes and by row count. Catalog rows with inline
# evidence exceed the byte budgets and keep the default.
TABLE_BATCH_BYTES = {
    "items": 35_000,
}
TABLE_BATCH_ROWS = {
    "items": 5,
    # Large evidence packets use MAX_BATCH_BYTES and one row per statement.
    "packets": 1,
}
PACKET_READ_BATCH = 512
DEFAULT_CATALOG_ROOT = "s3://marin-us-east-02a/marin/rl-data/catalog/production"
REVIEW_COMPLETENESS_VERSION = 6
FULL_REVIEW_KEY_PREFIX = "full-content"


class CatalogDisplayPolicy(StrEnum):
    STRICT = "strict"
    PREVIEW = "preview"


TABLE_COLUMNS = {
    "sources": (
        "id TEXT, declaration JSONB, assessment_key TEXT, latest_attempt_key TEXT, "
        "latest_attempt_status TEXT, reason TEXT, curation JSONB, diagnostics JSONB, "
        "source_fingerprint JSONB, duplicate_group JSONB, review_completeness JSONB, "
        "backend_verification_pending JSONB, sample_pin_stale JSONB, method_variants JSONB"
    ),
    "assessments": (
        "pipeline_key TEXT, source_id TEXT, scope TEXT, model JSONB, rubric TEXT, "
        "curation JSONB, curation_uri TEXT, traces JSONB"
    ),
    "populations": "pipeline_key TEXT, cohort TEXT, population BIGINT, sample_count BIGINT",
    "items": (
        "pipeline_key TEXT, task_id TEXT, cohort TEXT, status TEXT, verdict TEXT, explanation TEXT, "
        "evidence_ids JSONB, packet_uri TEXT, has_task BOOLEAN, omissions JSONB, exclusion_reason TEXT, "
        "confidence TEXT, reference_status TEXT, defects JSONB, "
        "provenance TEXT, replaced_task_sha256 TEXT, repair_advisories JSONB"
    ),
    "packets": "pipeline_key TEXT, task_id TEXT, packet TEXT",
    "backend_surveys": "survey_id TEXT, survey JSONB",
    "backend_sources": "source_id TEXT, capabilities JSONB",
    "backend_cells": "backend TEXT, requirement TEXT, cell JSONB",
    "identity_surveys": "survey_id TEXT, metadata JSONB",
    "duplicate_reviews": "left_source TEXT, right_source TEXT, review JSONB",
}


def batches(rows: Iterable[dict[str, Any]], budget: int = MAX_BATCH_BYTES, max_rows: int | None = None) -> Iterable[str]:
    """Bound SQL request bodies by encoded bytes, including packet escaping.

    The per-task tables take a smaller budget and a row-count cap: their statements hit
    the SQL gateway's per-statement timeout on the retained item volume, while a few
    catalog rows (declarations with inline evidence) legitimately exceed that smaller
    budget and keep the default.
    """
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # The SQL API carries the row array as a JSON string, adding another escape layer.
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > budget:
            raise ValueError("One catalog row exceeds the SQL batch budget")
        if size + length > budget or (max_rows is not None and len(batch) >= max_rows):
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def table_batch_bytes(table: str) -> int:
    """The SQL batch budget for one staging table."""
    return TABLE_BATCH_BYTES.get(table.removeprefix(STAGING_TABLE_PREFIX), MAX_BATCH_BYTES)


def table_batch_rows(table: str) -> int | None:
    """The row-count cap for one staging table, when its statements need one."""
    return TABLE_BATCH_ROWS.get(table.removeprefix(STAGING_TABLE_PREFIX))


def served_declaration_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Project one served declaration row onto what the SQL gateway can carry.

    ``oracle_filtration.pending_exclusions`` names every task excluded from a
    private oracle-qualified release (one entry per task, so a large admitted
    population makes the row exceed the SQL batch budget and the gateway's
    statement limit). No API document consumes the per-task list: the aggregate
    ``exclusion_counts`` and every pinned filter/validation artifact reference
    stay on the row, the admission-time validation reads the typed model from
    its sealed receipts, and the full evidence remains in the catalog snapshot
    and census receipts. The r34 release-manifest projection (served row
    trimmed, stored manifest unchanged) is the precedent.
    """
    filtration = declaration.get("oracle_filtration")
    if not filtration or not filtration.get("pending_exclusions"):
        return declaration
    trimmed = {k: v for k, v in filtration.items() if k != "pending_exclusions"}
    return {**declaration, "oracle_filtration": trimmed}


def ensure_tables(query: AppletQuery, names: tuple[str, ...]) -> None:
    """Create explicitly requested additive measurement tables from the app's schema."""
    statements: dict[str, str] = {
        statement.strip().split()[5]: statement
        for statement in SCHEMA_SQL.split(";")
        if statement.strip().startswith("CREATE TABLE IF NOT EXISTS ")
    }
    for name in names:
        query(statements[name], {})


def sql_column_names(columns: str) -> tuple[str, ...]:
    """Return the field names in an explicit JSON recordset declaration."""
    return tuple(column.strip().split()[0] for column in columns.split(","))


def insert_rows(
    query: AppletQuery,
    table: str,
    columns: str,
    load_id: str,
    rows: Iterable[dict[str, Any]],
    *,
    conflict_nothing: bool = False,
    conflict_updates: tuple[str, ...] = (),
) -> int:
    """Insert byte-bounded batches and return the stored row count.

    With ``conflict_nothing``, re-staged rows whose primary keys already exist are
    absorbed instead of raising — the resume path re-runs every insert and lets the
    primary keys deduplicate.
    ``conflict_updates`` instead replaces only the named fields, allowing a
    lossless transport migration within an existing unselected load.
    """
    fields = sql_column_names(columns)
    names = ", ".join(fields)
    budget = table_batch_bytes(table)
    max_rows = table_batch_rows(table)
    if conflict_updates and (conflict_nothing or not set(conflict_updates) <= set(fields)):
        raise ValueError("Conflict updates must name inserted columns and exclude conflict_nothing")
    conflict = "ON CONFLICT DO NOTHING" if conflict_nothing else ""
    if conflict_updates:
        conflict = f"ON CONFLICT ON CONSTRAINT {table}_pkey DO UPDATE SET " + ", ".join(
            f"{name}=EXCLUDED.{name}" for name in conflict_updates
        )
    count = 0
    for body in batches(rows, budget, max_rows):
        result = query(
            f"INSERT INTO {table} (load_id, {names}) SELECT :load_id, {names} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns}) {conflict}",
            {"load_id": load_id, "rows": body},
        )
        count += result["row_count"]
    return count


def full_review_pipeline_key(source_id: str) -> str:
    """The stable assessment key carrying a source's whole-population FULL review rows."""
    return f"{FULL_REVIEW_KEY_PREFIX}:{source_id}"


def read_full_review_records(uri: str, sha256: str) -> list[ReviewRecord]:
    """Read SHA-checked consolidated review records; unknown enum values fail validation loudly."""
    path = StoragePath(uri)
    if file_sha256(path) != sha256:
        raise ValueError(f"FULL review records changed: {uri}")
    with path.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    if any("task_id" not in row or "review" not in row for row in rows):
        raise ValueError(f"FULL review records are missing their task identity or review JSON: {uri}")
    return [ReviewRecord.model_validate_json(row["review"]) for row in rows]


def full_review_items(pipeline_key: str, records: list[ReviewRecord]) -> list[AssessmentItem]:
    """Project whole-population review records into typed assessment items without packets."""
    items = []
    for record in records:
        verdict = record.verdict
        reviewed = record.status is ReviewStatus.REVIEWED and verdict is not None
        items.append(
            AssessmentItem(
                pipeline_key=pipeline_key,
                task_id=record.task_id,
                cohort="admitted",
                status=JudgmentStatus.REVIEWED if reviewed else JudgmentStatus.UNAVAILABLE,
                verdict=verdict.quality if reviewed else Verdict.UNKNOWN,
                explanation=verdict.evidence if reviewed else record.detail or "Judgment unavailable",
                evidence_ids=[],
                packet_uri="",
                has_task=False,
                omissions=[],
                exclusion_reason=None,
                confidence=verdict.confidence if reviewed else None,
                reference_status=verdict.reference_status if reviewed else None,
                defects=list(verdict.defects) if reviewed else [],
            )
        )
    return items


class StagedFullReview(NamedTuple):
    """The staged records for one pinned whole-source binding."""

    assessment: Assessment
    populations: list[AssessmentPopulation]
    items: list[AssessmentItem]
    metadata: dict[str, Any]


def stage_full_review(source_id: str, binding: FullContentResultBinding) -> StagedFullReview:
    """Validate one pinned whole-source binding and stage its per-task review rows.

    Every check fails loudly: the loader never fabricates typed fields or maps
    unknown defect values to free text, and the staged rows must reconcile with
    the bound whole-source Result.
    """
    if binding.source_id != source_id:
        raise ValueError(f"FULL review binding belongs to another source: {source_id}: {binding.source_id}")
    records = read_full_review_records(binding.records.uri, binding.records.sha256)
    measured = binding.input.measured_tasks
    if len({row.task_id for row in records}) != len(records) or len(records) != measured:
        raise ValueError(
            f"FULL review records do not cover the retained population exactly once: {source_id}: "
            f"{len(records)} records for {measured} retained tasks"
        )
    recomputed = recompute_review_result(records)
    recorded = binding.result
    if (
        recomputed.reviewed,
        recomputed.issue_count,
        recomputed.uncertain_count,
        recomputed.unavailable_count,
    ) != (
        recorded.reviewed,
        recorded.issue_count,
        recorded.uncertain_count,
        recorded.unavailable_count,
    ):
        raise ValueError(f"FULL review records disagree with the bound Result: {source_id}")
    if binding.coverage_complete != all(row.status is ReviewStatus.REVIEWED for row in records):
        raise ValueError(f"FULL review coverage differs from its records: {source_id}")
    unavailable = sum(1 for row in records if row.status is not ReviewStatus.REVIEWED)
    if records and unavailable / len(records) > MAX_UNAVAILABLE_JUDGMENT_SHARE:
        raise ValueError(f"FULL review exceeds the unavailable-judgment threshold: {source_id}")
    receipt = json.loads(StoragePath(binding.review_receipt.uri).read_text())
    payload = receipt.get("result") or {}
    reviewer = payload.get("reviewer") or {}
    rubric = payload.get("rubric")
    if not rubric or not reviewer.get("model"):
        raise ValueError(f"FULL review receipt is missing its rubric or reviewer identity: {source_id}")
    pipeline_key = full_review_pipeline_key(source_id)
    assessment = Assessment(
        pipeline_key=pipeline_key,
        source_id=source_id,
        scope="full",
        model=ModelReference(
            model=str(reviewer["model"]),
            revision=str(reviewer.get("model_revision") or ""),
            max_tokens=int(reviewer.get("max_tokens") or 0),
            max_attempts=int(reviewer.get("max_attempts") or 0),
            retry_max_tokens=int(reviewer.get("retry_max_tokens") or 0),
            reviewer_spec_json=json.dumps(reviewer, sort_keys=True),
        ),
        rubric=json.dumps(rubric, sort_keys=True),
        curation=CurationSummary(
            status="complete",
            mode=CurationMode.FULL,
            source_revision=None,
            recipe_revision=None,
            raw_population_count=measured,
            raw_sample_count=measured,
            processed_rows=measured,
            unprocessed_rows=0,
            admission_counts=AdmissionCounts(),
            verification=None,
            quality_result=recorded,
        ),
        curation_uri=binding.review_receipt.uri,
        traces=[],
    )
    populations = [
        AssessmentPopulation(pipeline_key=pipeline_key, cohort="admitted", population=measured, sample_count=measured),
        AssessmentPopulation(pipeline_key=pipeline_key, cohort="excluded", population=0, sample_count=0),
    ]
    items = full_review_items(pipeline_key, records)
    validate_items(populations, items)
    return StagedFullReview(
        assessment, populations, items, {"reviewed": recomputed.reviewed, "unavailable": unavailable}
    )


def _grader_network_profile_pending(backend_row: BackendSourceMap, backend_map: BackendSurvey) -> bool:
    """Whether the source's grader needs network while the survey's grader profile denies it.

    Under the deny defaults every production backend reports unsupported for
    such a source, so the fit failure is the profile binding, not a missing
    backend; the r22 projection ships it as typed Pending.
    """
    grader_demands = (backend_row.demands or {}).get("grader") or {}
    needs_network = bool({Requirement.NETWORK, Requirement.NETWORK_ALLOW} & set(grader_demands))
    grader_profile = backend_map.runtime_defaults.get("grader")
    return needs_network and grader_profile is not None and grader_profile.network is NetworkPolicy.DENY


UNRESOLVED_DOCKER_BUILD_GATE = "Unresolved Docker build context must be built and pinned before execution"


def _docker_build_preparation_pending(backend_row: BackendSourceMap) -> bool:
    """Whether the source's grader role fails only on its own unresolved Docker builds.

    The survey's build-pinning gate is a source preparation requirement, not a backend
    capability gap: the grader images are simply not built yet, and a maintained image
    preparation re-admits the source. Callers gate this on no runnable production
    backend, so a runnable sibling never triggers it.
    """
    fits = backend_row.backends
    grader_fails = [
        (message, count)
        for fit in fits.values()
        if fit.roles.get("grader", {}).get("unsupported")
        for message, count in fit.validation_errors.items()
        if message.startswith("grader:")
    ]
    if not grader_fails:
        return False
    return all(UNRESOLVED_DOCKER_BUILD_GATE in message for message, _ in grader_fails)


def load_catalog(
    catalog: CurationCatalog,
    query: AppletQuery,
    backend_map: BackendSurvey,
    identities: SourceIdentitySurvey,
    *,
    display_policy: CatalogDisplayPolicy = CatalogDisplayPolicy.STRICT,
    full_reviews: dict[str, FullContentResultBinding] | None = None,
    resume_load_id: str | None = None,
) -> dict[str, Any]:
    """Stage and validate a snapshot, then select it if the serving pointer is unchanged.

    ``full_reviews`` carries staged whole-source FULL review bindings keyed by
    source ID. For each bound source without a catalog assessment, the loader
    stages the pinned per-task review rows as a browseable assessment.
    ``resume_load_id`` continues a previous attempt of the same load: the staged
    prefixes per table are skipped and the fixed load identifier is reused.
    """
    started = time.monotonic()
    backend_map = canonical_backend_survey(backend_map.model_dump(mode="json"))
    identities = canonical_identity_survey(identities.model_dump(mode="json"))
    snapshot = catalog.current()
    if (backend_map.catalog_root, backend_map.snapshot_id) != (catalog.root, snapshot.snapshot_id):
        raise ValueError("Backend map belongs to another catalog snapshot; survey the selected catalog")
    if backend_map.schema_sha256 != backend_schema_sha256():
        raise ValueError("Backend map uses another requirement schema; rerun the survey")
    if (identities.catalog_root, identities.snapshot_id) != (catalog.root, snapshot.snapshot_id):
        raise ValueError("Source identity map belongs to another catalog snapshot")
    identity_id = canonical_sha256(identity_survey_payload(identities))
    survey_id = canonical_sha256(backend_survey_payload(backend_map))
    ensure_tables(query, ("atlas_cohort_summaries",))
    current = query(
        """
        SELECT c.load_id, l.catalog_root, l.snapshot_id, l.review_completeness_version, l.display_policy,
            b.survey_id, i.survey_id AS identity_id FROM atlas_current c
        JOIN atlas_loads l USING (load_id)
        LEFT JOIN atlas_backend_surveys b USING (load_id)
        LEFT JOIN atlas_identity_surveys i USING (load_id) WHERE c.name = 'latest'
    """,
        {},
    )["rows"]
    if current and (
        current[0]["catalog_root"],
        current[0]["snapshot_id"],
        current[0]["survey_id"],
        current[0]["identity_id"],
        current[0]["review_completeness_version"],
        current[0]["display_policy"],
    ) == (
        catalog.root,
        snapshot.snapshot_id,
        survey_id,
        identity_id,
        REVIEW_COMPLETENESS_VERSION,
        display_policy,
    ):
        materialize_cohort_summaries(lambda sql, parameters: query(sql, parameters)["rows"], current[0]["load_id"])
        return {"changed": False, "snapshot_id": snapshot.snapshot_id}
    expected_load_id = current[0]["load_id"] if current else None
    load_id = resume_load_id or uuid.uuid4().hex
    requests = 2

    def execute(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return query(sql, parameters)

    def insert(table: str, rows: Iterable[dict[str, Any]]) -> int:
        # A resumed load skips each table's staged prefix: the rows are inserted in a
        # deterministic order and the gateway completes timed-out statements server-side,
        # so the staged row count is exactly the committed prefix. ON CONFLICT DO NOTHING
        # absorbs any overlap at the batch boundary.
        if resume_load_id is not None:
            skip = staged_row_count(execute, f"atlas_{table}", load_id)
            if skip:
                rows = list(rows)
                rows = rows[skip:]
                if not rows:
                    return skip
                return skip + insert_rows(
                    execute,
                    f"atlas_{table}",
                    TABLE_COLUMNS[table],
                    load_id,
                    rows,
                    conflict_nothing=True,
                )
        return insert_rows(execute, f"atlas_{table}", TABLE_COLUMNS[table], load_id, rows)

    if resume_load_id is not None:
        execute(
            "INSERT INTO atlas_loads"
            " (load_id, catalog_root, snapshot_id, created_at, review_completeness_version, display_policy)"
            " VALUES (:load_id, :catalog_root, :snapshot_id, :created_at, :review_completeness_version, :display_policy)"
            " ON CONFLICT (load_id) DO UPDATE SET catalog_root = EXCLUDED.catalog_root,"
            " snapshot_id = EXCLUDED.snapshot_id, created_at = EXCLUDED.created_at,"
            " review_completeness_version = EXCLUDED.review_completeness_version,"
            " display_policy = EXCLUDED.display_policy",
            {
                "load_id": load_id,
                "catalog_root": catalog.root,
                "snapshot_id": snapshot.snapshot_id,
                "created_at": snapshot.created_at,
                "review_completeness_version": REVIEW_COMPLETENESS_VERSION,
                "display_policy": display_policy,
            },
        )
    else:
        execute(
            """
            INSERT INTO atlas_loads
                (load_id, catalog_root, snapshot_id, created_at, review_completeness_version, display_policy)
            VALUES (:load_id, :catalog_root, :snapshot_id, :created_at, :review_completeness_version, :display_policy)
        """,
            {
                "load_id": load_id,
                "catalog_root": catalog.root,
                "snapshot_id": snapshot.snapshot_id,
                "created_at": snapshot.created_at,
                "review_completeness_version": REVIEW_COMPLETENESS_VERSION,
                "display_policy": display_policy,
            },
        )
    sources = [canonical_snapshot_source(source) for source in served_snapshot_sources(catalog, snapshot.snapshot_id)]
    if retired := {source.id for source in sources} & retired_admission_source_ids():
        raise ValueError(f"Retired sources remain in the selected catalog: {sorted(retired)}")
    previously_available = (
        {
            row["source_id"]
            for row in execute(
                "SELECT s.id AS source_id FROM atlas_sources s WHERE s.load_id=:load_id "
                "AND s.declaration->>'status'='Available'",
                {"load_id": expected_load_id},
            )["rows"]
        }
        if expected_load_id is not None
        else set()
    )
    verification_pending: dict[str, list[str]] = {}
    sample_pin_stale: dict[str, bool] = {}
    downgraded: dict[str, tuple[PendingStage, str]] = {}
    backend_rows = {row.source_id: row for row in backend_map.sources}
    for source in sources:
        backend_row = backend_rows[source.id]
        verdicts = {name: fit.support.value for name, fit in backend_row.backends.items()}
        blocker, pending = backend_admission_review(verdicts, source.declaration.status, backend_map.deployment_classes)
        if blocker:
            if source.declaration.status is SourceStatus.AVAILABLE and _grader_network_profile_pending(
                backend_row, backend_map
            ):
                # r22 ruling: the profile conflict ships as typed Pending —
                # unadmitted, with the unsupported evidence and the conflict
                # visible — until a maintained per-source profile binding lands.
                downgraded[source.id] = (
                    PendingStage.BACKEND_PROFILE_PENDING,
                    "Backend profile pending: grader needs network allow.",
                )
            elif (
                source.declaration.status is SourceStatus.AVAILABLE
                and not has_runnable_production_support(verdicts, backend_map.deployment_classes)
                and _docker_build_preparation_pending(backend_row)
            ):
                # The build-pinning gate is a source preparation requirement: the grader
                # images are not built yet, so the survey cannot qualify any backend.
                # Ships as typed Pending until the image-preparation lane pins them.
                downgraded[source.id] = (
                    PendingStage.BACKEND_PREPARATION_PENDING,
                    "Backend preparation pending: the grader Docker build context is unresolved on every "
                    "task (the surveyed production backends cannot run unpinned images); the image "
                    "preparation lane re-admits the source with pinned grader images.",
                )
            else:
                raise ValueError(f"Backend admission blocked: {source.id}: {blocker}")
        # Available rows must pass full population admission; every other status
        # publishes as metadata without admission or selectability.
        if (
            source.declaration.status is SourceStatus.AVAILABLE
            and filtration(source.declaration.model_dump(mode="json"))["state"] != "excluded"
        ):
            require_declaration_admission(source.declaration)
            if source.id not in previously_available:
                # ARMED (rulings-enforcement): a NEW Available admission needs
                # lexical clearance — explicit intended_use plus the train
                # private-release receipt or the eval benchmark-measured
                # receipt at the pin — and its SAMPLE panel on the current
                # final pin. Sources already Available in the previously
                # selected catalog are grandfathered with visible markers.
                # A source the backend review already downgrades to Pending
                # skips the SAMPLE gate: it cannot admit either way, and the
                # downgrade note carries its stale-panel conflict.
                require_available_lexical_clearance(source.declaration)
                if source.id in downgraded:
                    sample_pin_stale[source.id] = True
                else:
                    require_sample_pin_currency(source.declaration)
            elif (
                source.declaration.independent_qc is not None
                and source.declaration.independent_qc.panel_currency is SamplePanelCurrency.ARCHIVED_LINEAGE
            ):
                sample_pin_stale[source.id] = True
        if pending:
            # An already-admitted source keeps its status with a visible marker.
            # A new source downgrades to Pending only when NO production backend
            # satisfies admission and VERIFY is its only positive production
            # outcome; a runnable production backend admits it regardless of
            # sibling VERIFY cells.
            verification_pending[source.id] = pending
            if source.id not in previously_available and not has_runnable_production_support(
                verdicts, backend_map.deployment_classes
            ):
                downgraded[source.id] = (
                    PendingStage.BACKEND_VERIFICATION_PENDING,
                    "Backend verification pending: "
                    + ", ".join(pending)
                    + " (new admission waits on the live qualification).",
                )
    if len(sources) != snapshot.source_count:
        raise ValueError("Catalog inventory does not cover its declared source count")
    if {source.id for source in sources} != {source.source_id for source in backend_map.sources}:
        raise ValueError("Backend map does not cover the selected catalog inventory")
    if {source.id for source in sources} != set(identities.fingerprints):
        raise ValueError("Source identity map does not cover the selected catalog inventory")
    source_declarations = {source.id: source.declaration for source in sources}
    for index, source in enumerate(sources):
        if source.id not in downgraded:
            continue
        # A typed Pending projection: the registry declaration is untouched and
        # the load-time copy carries the stage and note — VERIFY-pending sources
        # wait on the live qualification, profile-conflict sources (grader
        # needs network allow) wait on the maintained profile binding, and
        # unresolved grader images wait on the image preparation lane.
        stage, note = downgraded[source.id]
        notes = (source.declaration.notes + " " if source.declaration.notes else "") + note
        downgraded_declaration = source.declaration.model_copy(
            update={"status": SourceStatus.PENDING, "stage": stage, "notes": notes}
        )
        require_typed_status_fields(downgraded_declaration)
        sources[index] = source.model_copy(update={"declaration": downgraded_declaration})
        source_declarations[source.id] = downgraded_declaration
    # The survey row's grading evidence is a copy of the declaration's taken at survey
    # time; it is metadata that no fit verdict reads. The catalog declaration is
    # current, so the load serves its evidence and records every row it rebound.
    grading_evidence_rebound = []
    rebound_sources = []
    for source in backend_map.sources:
        evidence = source_declarations[source.source_id].grading_environment
        expected = evidence.model_dump(mode="json") if evidence is not None else None
        if source.grading_environment != expected:
            grading_evidence_rebound.append(source.source_id)
        rebound_sources.append(source.model_copy(update={"grading_environment": expected}))
    backend_map = backend_map.model_copy(update={"sources": rebound_sources})
    groups = {member.source_id: group.model_dump(mode="json") for group in identities.groups for member in group.members}
    metadata: dict[str, list[dict[str, Any]]] = {
        "sources": [
            {
                **row.model_dump(mode="json", exclude={"snapshot_id"}),
                "source_fingerprint": (
                    identities.fingerprints[row.id].model_dump(mode="json")
                    if identities.fingerprints[row.id] is not None
                    else None
                ),
                "duplicate_group": groups.get(row.id),
                "backend_verification_pending": verification_pending.get(row.id),
                "sample_pin_stale": sample_pin_stale.get(row.id),
                "method_variants": source_variants(row.declaration.model_dump(mode="json")),
                "declaration": served_declaration_projection(row.declaration.model_dump(mode="json")),
            }
            for row in sources
        ],
        "assessments": [],
        "populations": [],
        "items": [],
        "identity_surveys": [
            {
                "survey_id": identity_id,
                "metadata": identities.model_dump(mode="json", exclude={"fingerprints", "reviews", "groups"}),
            }
        ],
        "duplicate_reviews": [
            {
                "left_source": row.similarity.left,
                "right_source": row.similarity.right,
                "review": row.model_dump(mode="json"),
            }
            for row in identities.reviews
        ],
        "backend_surveys": [
            {"survey_id": survey_id, "survey": backend_map.model_dump(mode="json", exclude={"sources", "matrix"})}
        ],
        "backend_cells": [
            {"backend": cell.backend, "requirement": cell.requirement.value, "cell": cell.model_dump(mode="json")}
            for cell in backend_map.matrix
        ],
        "backend_sources": [
            {"source_id": source.source_id, "capabilities": source.model_dump(mode="json")}
            for source in backend_map.sources
        ],
    }
    all_items = []
    full_counts: dict[str, dict[str, int]] = {}
    withheld_pipelines = {
        source.assessment_key
        for source in sources
        if source.assessment_key is not None and content_is_withheld(source.declaration.model_dump(mode="json"))
    }
    for source, record in zip(sources, metadata["sources"], strict=True):
        key = source.assessment_key
        record["review_completeness"] = review_completeness(source.declaration, [], [])
        binding = full_reviews.get(source.id) if full_reviews else None
        if key is None and binding is not None:
            # A staged whole-source FULL review: stage its per-task rows as the
            # source's browseable assessment. The catalog population must match
            # the pinned binding; rows are excluded from packet reads because
            # whole-population reviews carry their evidence in the row itself.
            retained = source.declaration.within_source_dedup
            if retained is None or retained.unique_count != binding.input.measured_tasks:
                raise ValueError(
                    f"FULL review binding belongs to another retained population: {source.id}: "
                    f"binding {binding.input.measured_tasks} tasks vs catalog "
                    f"{retained.unique_count if retained else None}"
                )
            assessment, populations, items, staged = stage_full_review(source.id, binding)
            record["assessment_key"] = assessment.pipeline_key
            if source.declaration.quality_result is None:
                record["review_completeness"] = review_completeness(
                    source.declaration.model_copy(
                        update={
                            "quality_result": binding.result,
                            "quality_result_finalization": binding.finalization,
                            "quality_result_waiver": binding.waiver,
                        }
                    ),
                    [],
                    [],
                )
            display = quality_display(
                source.declaration.model_dump(mode="json"),
                source.curation.model_dump(mode="json") if source.curation else None,
                key,
            )
            if display_policy is CatalogDisplayPolicy.STRICT or display.quality is not Quality.QUICK:
                require_content_result(source.declaration, source.curation, key)
            metadata["assessments"].append(assessment.model_dump(mode="json"))
            metadata["populations"].extend(row.model_dump(mode="json") for row in populations)
            metadata["items"].extend(row.model_dump(mode="json") for row in items)
            full_counts[source.id] = staged
            continue
        if key is None:
            display = quality_display(
                source.declaration.model_dump(mode="json"),
                source.curation.model_dump(mode="json") if source.curation else None,
                key,
            )
            if display_policy is CatalogDisplayPolicy.STRICT or display.quality is not Quality.QUICK:
                require_content_result(source.declaration, source.curation, key)
            continue
        assessment = canonical_assessment(catalog.assessment(key))
        if assessment.source_id != source.id:
            raise ValueError("Selected assessment belongs to another source")
        populations, items = catalog.populations(key), list(catalog.items(key))
        validate_items(populations, items)
        record["review_completeness"] = review_completeness(source.declaration, populations, items)
        unavailable = sum(1 for item in items if item.status != JudgmentStatus.REVIEWED)
        if items and unavailable / len(items) > MAX_UNAVAILABLE_JUDGMENT_SHARE:
            raise ValueError(f"Selected assessment exceeds the unavailable-judgment threshold: {key}")
        metadata["assessments"].append(assessment.model_dump(mode="json"))
        metadata["populations"].extend(row.model_dump(mode="json") for row in populations)
        metadata["items"].extend(row.model_dump(mode="json") for row in items)
        all_items.extend(items)
    counts = {table: insert(table, rows) for table, rows in metadata.items()}
    # On a resumed load the inserts absorb re-staged rows (the primary keys deduplicate),
    # so the inserted-row counts understate the staged totals; verify the completeness
    # against the database's own staged rows.
    backend_counts = {
        table: execute(f"SELECT COUNT(*) AS n FROM atlas_{table} WHERE load_id = :load_id", {"load_id": load_id})[
            "rows"
        ][0]["n"]
        for table in ("backend_sources", "backend_surveys", "backend_cells")
    }
    if (
        backend_counts["backend_sources"] != len(sources)
        or backend_counts["backend_surveys"] != 1
        or backend_counts["backend_cells"] != len(backend_map.matrix)
    ):
        raise ValueError("Backend survey import is incomplete")

    def packet_rows() -> Iterable[dict[str, Any]]:
        for offset in range(0, len(all_items), PACKET_READ_BATCH):
            for item, packet in catalog.packets(all_items[offset : offset + PACKET_READ_BATCH]):
                if ("task" in packet) != item.has_task:
                    raise ValueError(f"Packet evidence disagrees with catalog item: {item.task_id}")
                if item.pipeline_key in withheld_pipelines:
                    # A PRIVATE_ALWAYS source's packets stay in its private store:
                    # the served database holds the marker response, never the bytes.
                    continue
                yield {
                    "pipeline_key": item.pipeline_key,
                    "task_id": item.task_id,
                    "packet": json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                }

    counts["packets"] = insert("packets", packet_rows())
    if expected_load_id is not None:
        for table, copied in retain_prior_assessments(execute, TABLE_COLUMNS, expected_load_id, load_id).items():
            counts[table] += copied
    materialize_cohort_summaries(lambda sql, parameters: execute(sql, parameters)["rows"], load_id)
    packet_sizes = execute(
        """
        SELECT COALESCE(SUM(octet_length(packet)), 0) AS packet_bytes,
            MAX(octet_length(packet)) AS max_packet_bytes,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY octet_length(packet)) AS p95_packet_bytes
        FROM atlas_packets WHERE load_id = :load_id
    """,
        {"load_id": load_id},
    )["rows"][0]
    if CurationCatalog(catalog.root).current().snapshot_id != snapshot.snapshot_id:
        raise RuntimeError("Catalog selection changed during import; retry the load")
    selected = execute(
        "UPDATE atlas_current SET load_id = :load_id WHERE name = 'latest' "
        "AND load_id IS NOT DISTINCT FROM :expected_load_id",
        {"load_id": load_id, "expected_load_id": expected_load_id},
    )
    if selected["row_count"] != 1:
        raise RuntimeError("Atlas selection changed during import; retry the load")
    return {
        "changed": True,
        "load_id": load_id,
        "snapshot_id": snapshot.snapshot_id,
        "counts": counts,
        "full_reviews": full_counts,
        "grading_evidence_rebound": sorted(grading_evidence_rebound),
        **packet_sizes,
        "sql_requests": requests,
        "seconds": time.monotonic() - started,
    }


@click.command(help=__doc__)
@click.option(
    "--catalog-root", default=DEFAULT_CATALOG_ROOT, show_default=True, help="Published FineStore catalog to import."
)
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--backend-map", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--source-identities", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--full-content-reviews",
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Staged whole-source FULL review bindings (JSON source_id -> binding). Sources without a "
        "catalog assessment load their pinned per-task review rows as a browseable assessment."
    ),
)
@click.option(
    "--resume-load-id",
    default=None,
    help="Continue a previous attempt of this load: reuse the identifier and skip staged prefixes.",
)
def main(
    catalog_root: str,
    applet_id: str,
    service_url: str,
    backend_map: Path,
    source_identities: Path,
    full_content_reviews: Path | None,
    resume_load_id: str | None,
) -> None:
    configure_coreweave_s3()
    survey = canonical_backend_survey(json.loads(backend_map.read_text()))
    identities = canonical_identity_survey(json.loads(source_identities.read_text()))
    reviews = (
        {
            source_id: FullContentResultBinding.model_validate(payload)
            for source_id, payload in json.loads(full_content_reviews.read_text()).items()
        }
        if full_content_reviews is not None
        else None
    )
    click.echo(
        json.dumps(
            load_catalog(
                CurationCatalog(catalog_root),
                applet_query(service_url, applet_id),
                survey,
                identities,
                display_policy=(
                    CatalogDisplayPolicy.PREVIEW if applet_id == PREVIEW_APP_ID else CatalogDisplayPolicy.STRICT
                ),
                full_reviews=reviews,
                resume_load_id=resume_load_id,
            )
        )
    )


if __name__ == "__main__":
    main()
