# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins
import hashlib
import importlib
import json
import shutil
import subprocess
import tomllib
import uuid
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from finestore.reader import ReadView
from finestore.rl.curation import (
    ASSESSMENTS,
    CURRENT,
    INVENTORY,
    ITEMS,
    POPULATIONS,
    SNAPSHOTS,
    Assessment,
    AssessmentItem,
    AssessmentPopulation,
    CatalogSnapshot,
    Confidence,
    ConsumerDecision,
    ConsumerNeed,
    ConsumerReason,
    ContentEvidenceReference,
    CurationCatalog,
    CurationSummary,
    CurrentCatalog,
    Defect,
    GradingEnvironmentEvidence,
    IndependentQC,
    OversightLevel,
    ReferenceStatus,
    SamplePanelCurrency,
    SnapshotSource,
    SourceOrigin,
    SourceQualityResult,
    register_catalog,
    source_display_name,
)
from finestore.rl.curation import (
    QualifiedSourceDeclaration as SourceDeclaration,
)
from finestore.rl.difficulty import (
    MEASUREMENTS,
    DifficultyAttempt,
    DifficultyMeasurement,
    Solver,
    measurement_from_attempts,
    register_difficulty,
)
from finestore.store import DataStore
from marina.applets import (
    MAX_QUERY_RESPONSE_BYTES,
    AppletServices,
    load_backend_entrypoint,
    package_applet,
    read_applet_package,
    remove_backend_revision,
    validate_query,
)
from marina.client import AppletQuery, MarinaRequestError
from rigging.server_auth import VerifiedIdentity, identity_scope
from shellbox.machine import Backend, NetworkPolicy
from shellbox.requirements import Requirement, Support
from shellbox.survey import CapabilityCell
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    FitSupport,
    RuntimeProfile,
    backend_schema_sha256,
)
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage
from taskcompendium.pipeline.models import Quality, ReviewRecord, ReviewVerdict
from taskcompendium.source_fingerprints import SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation import registry, retirement
from experiments.post_training.task_curation.catalog_identity import backend_survey_payload, canonical_backend_survey
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.review_quality import quality_result
from experiments.post_training.task_curation.review_quality import quality_result as recompute_review_result
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory
from experiments.post_training.task_curation.tests.admission_fixtures import (
    actor_census_input,
    consumer_receipt,
    grading_evidence,
    with_declaration_admission_census,
)
from experiments.post_training.task_curation.tests.admission_fixtures import (
    isolated_admission_census as isolated_admission_census,
)
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from experiments.post_training.task_curation.tests.license_fixtures import isolated_license_bans as isolated_license_bans
from infra.marina.apps.rl_data_catalog.confidence import confidence_rows
from infra.marina.apps.rl_data_catalog.load import (
    CatalogDisplayPolicy,
    ensure_curriculum_analyze_settings,
    full_review_pipeline_key,
)
from infra.marina.apps.rl_data_catalog.load import (
    load_catalog as load_surveyed_catalog,
)
from infra.marina.apps.rl_data_catalog.load_curriculum import load_curriculum
from infra.marina.apps.rl_data_catalog.load_difficulty import load_measurements, read_measurements
from infra.marina.apps.rl_data_catalog.load_identity import load_identity
from infra.marina.apps.rl_data_catalog.load_reviews import load_content_reviews
from infra.marina.apps.rl_data_catalog.load_rollouts import load_rollouts
from infra.marina.apps.rl_data_catalog.prepare_preview import PREVIEW_APP_ID, PRODUCTION_APP_ID, prepare_preview
from infra.marina.apps.rl_data_catalog.release_curriculum import check_curriculum_document
from infra.marina.apps.rl_data_catalog.release_evidence import (
    HISTORY_PIPELINE_BATCH,
    catalog_state,
    projected_duplicate_inventory,
    retained_packet_checksums,
    stored_duplicate_groups,
    verify_preservation,
)
from infra.marina.apps.rl_data_catalog.release_gateway import CandidateApi
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference
from infra.marina.apps.rl_data_catalog.server import consumer as consumer_policy
from infra.marina.apps.rl_data_catalog.server.app import canonical_source_record, create_api, migrate
from infra.marina.apps.rl_data_catalog.server.curriculum import (
    TASKS_SQL,
    CoverageQuery,
    loaded_release,
    source_restrictions,
    task_parameters,
    task_response,
)
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content
from infra.marina.apps.rl_data_catalog.server.methods import rl_variant
from infra.marina.apps.rl_data_catalog.server.quality import result
from infra.marina.apps.rl_data_catalog.server.review_content import task_request_evidence
from infra.marina.apps.rl_data_catalog.server.schema import CURRICULUM_ANALYZED_TABLES, SCHEMA_SQL
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.difficulty_fixtures import difficulty_measurement
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence, filtered_declaration
from infra.marina.apps.rl_data_catalog.tests.method_fixtures import sft_variant
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import review_declarations
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

pytestmark = pytest.mark.usefixtures("isolated_admission_census")


def test_history_checksums_cover_each_confidence_load_and_detect_archived_edits(catalog_query, tmp_path):
    root = str(tmp_path / "confidence-history")
    publish_sample(root, "confidence-history")
    load_catalog(CurationCatalog(root), catalog_query)
    for load in ("archived", "current", "empty"):
        catalog_query(
            "INSERT INTO curriculum_loads(load_id,release_id,release_uri,manifest) "
            "VALUES (:load,:load,'s3://fixture/release.json','{}')",
            {"load": load},
        )
    catalog_query(
        "INSERT INTO curriculum_confidence VALUES "
        "('archived','z','placement','math','algebra',ARRAY[9,3]),"
        "('archived','a','denominator','','',ARRAY[10,10]),"
        "('current','a','denominator','','',ARRAY[11,11])",
        {},
    )
    expected = catalog_query(
        "SELECT l.load_id,"
        "(SELECT COUNT(*) FROM curriculum_confidence v WHERE v.load_id=l.load_id) AS vectors,"
        "(SELECT md5(COALESCE(string_agg(md5(to_jsonb(v)::text),'' ORDER BY v.scope_id,v.metric,"
        "v.subject_id,v.node_id),'')) FROM curriculum_confidence v WHERE v.load_id=l.load_id) AS checksum "
        "FROM curriculum_loads l ORDER BY l.load_id",
        {},
    )["rows"]
    observed_loads = []

    def bounded_query(sql, parameters):
        if "FROM curriculum_confidence" in sql:
            assert parameters.get("load_id") in {"archived", "current", "empty"}
            observed_loads.append(parameters["load_id"])
        return catalog_query(sql, parameters)

    before = catalog_state(bounded_query)
    assert before["confidence_vectors"] == expected
    assert observed_loads == ["archived", "current", "empty"]
    catalog_query("UPDATE curriculum_confidence SET instances=ARRAY[8,3] WHERE load_id='archived' AND scope_id='z'", {})
    with pytest.raises(ValueError, match="Historical confidence_vectors"):
        verify_preservation(before, catalog_state(bounded_query))


def test_release_gate_preserves_archived_packets_even_during_an_explicit_catalog_refresh(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "release-catalog")
    publish_sample(root, "before")
    load_catalog(CurationCatalog(root), catalog_query)
    before = catalog_state(catalog_query)
    verify_preservation(before, catalog_state(catalog_query))
    assert before["archives"][0]["sources"] == 1
    assert before["archives"][0]["packets"] == 3
    with catalog_database.begin() as connection:
        connection.execute(text("UPDATE atlas_sources SET duplicate_group='{}'::jsonb"))
    with pytest.raises(ValueError):
        verify_preservation(before, catalog_state(catalog_query))
    with catalog_database.begin() as connection:
        connection.execute(text("UPDATE atlas_sources SET duplicate_group=NULL"))
    verify_preservation(before, catalog_state(catalog_query))
    publish_sample(root, "after")
    load_catalog(CurationCatalog(root), catalog_query)
    after = catalog_state(catalog_query)
    retained = {row["load_id"]: row for row in after["archives"]}
    for archive in before["archives"]:
        assert retained[archive["load_id"]] == archive
    with pytest.raises(ValueError):
        verify_preservation(before, after)
    verify_preservation(before, after, refresh_snapshot="after")
    with catalog_database.begin() as connection:
        connection.execute(text("UPDATE atlas_packets SET packet=packet || ' '"))
    with pytest.raises(ValueError):
        verify_preservation(before, catalog_state(catalog_query), refresh_snapshot="after")


def test_history_packet_batches_preserve_distinct_variants_across_loads(catalog_database, catalog_query):
    packets = []
    for pipeline in range(HISTORY_PIPELINE_BATCH + 3):
        for load in ("older", "current"):
            packets.extend(
                [
                    {
                        "load_id": load,
                        "pipeline_key": f"pipeline-{pipeline:02}",
                        "task_id": "same",
                        "packet": "identical",
                    },
                    {"load_id": load, "pipeline_key": f"pipeline-{pipeline:02}", "task_id": "changed", "packet": load},
                ]
            )
    with catalog_database.begin() as connection:
        connection.execute(text("INSERT INTO atlas_packets VALUES (:load_id,:pipeline_key,:task_id,:packet)"), packets)
    statements = []

    def query(sql, parameters):
        statements.append((sql, parameters))
        return catalog_query(sql, parameters)

    actual = retained_packet_checksums(query)
    expected = []
    for pipeline in sorted({row["pipeline_key"] for row in packets}):
        variants = sorted(
            {
                (row["task_id"], hashlib.md5(row["packet"].encode()).hexdigest())
                for row in packets
                if row["pipeline_key"] == pipeline
            }
        )
        expected.append(
            {
                "pipeline_key": pipeline,
                "packets": len(variants),
                "checksum": hashlib.md5("".join(digest for _, digest in variants).encode()).hexdigest(),
            }
        )
    assert actual == expected
    checksums = [(sql, values) for sql, values in statements if "md5(packet)" in sql]
    assert len(checksums) == 2
    assert all(0 < len(values) <= HISTORY_PIPELINE_BATCH for _, values in checksums)
    assert {key for _, values in checksums for key in values.values()} == {row["pipeline_key"] for row in packets}


def catalog_backend_map(catalog: CurationCatalog) -> BackendSurvey:
    snapshot = catalog.current()
    return BackendSurvey(
        catalog_root=catalog.root,
        snapshot_id=snapshot.snapshot_id,
        generated_at="2026-11-12T00:00:00+00:00",
        git_revision="test",
        schema_sha256=backend_schema_sha256(),
        runtime_defaults={},
        deployment_limits={},
        matrix=[
            CapabilityCell(
                backend=backend,
                requirement=need,
                support=Support.PARTIAL,
                evidence=("test fixture",),
                note="Unmeasured test environment",
            )
            for backend in (backend.value for backend in Backend)
            for need in Requirement
        ],
        sources=[
            BackendSourceMap(
                source_id=source.id,
                basis="unassessed" if not (source.declaration.task_count or 0) else "converted_spec_census",
                reason=(
                    None if source.declaration.task_count else "No converted population is bound to this fixture source."
                ),
                task_count=source.declaration.task_count or 0,
                inputs={},
                demands={},
                resource_ranges={},
                backends=(
                    {
                        backend: BackendFit(
                            support=FitSupport.ROLE_ABSENT,
                            task_counts={FitSupport.ROLE_ABSENT: source.declaration.task_count or 0},
                            roles={
                                "agent": {FitSupport.ROLE_ABSENT: source.declaration.task_count or 0},
                                "grader": {FitSupport.ROLE_ABSENT: source.declaration.task_count or 0},
                            },
                            blockers={},
                            not_applicable={},
                            validation_errors={},
                        )
                        for backend in (backend.value for backend in Backend)
                    }
                    if source.declaration.task_count
                    else {}
                ),
                role_backend_assignments={},
                grading_environment=(
                    source.declaration.grading_environment.model_dump(mode="json")
                    if source.declaration.grading_environment is not None
                    else None
                ),
            )
            for source in catalog.sources(snapshot.snapshot_id)
        ],
    )


def load_catalog(catalog: CurationCatalog, query: AppletQuery) -> dict[str, Any]:
    return load_surveyed_catalog(catalog, query, catalog_backend_map(catalog), catalog_identity_map(catalog))


def catalog_identity_map(catalog: CurationCatalog) -> SourceIdentitySurvey:
    snapshot = catalog.current()
    return SourceIdentitySurvey(
        catalog_root=catalog.root,
        snapshot_id=snapshot.snapshot_id,
        generated_at="2026-11-12T00:00:00+00:00",
        fingerprints={source.id: None for source in catalog.sources(snapshot.snapshot_id)},
        reviews=[],
        groups=[],
        judge=None,
    )


@pytest.mark.parametrize("decision_status", [None, "Pending", "Excluded"])
def test_retained_rows_require_a_user_decision_before_consumer_status_changes(
    catalog_database, catalog_query, tmp_path, monkeypatch, decision_status
):
    root = str(tmp_path / "consumer-refresh")
    publish_sample(root, "standard", source_id="Marin:standard")
    publish_sample(root, "bespoke", source_id="Marin:bespoke")
    catalog = CurationCatalog(root)
    sources = [*catalog.sources("standard"), *catalog.sources("bespoke")]
    with DataStore.open(root, writer_id="combined") as store:
        register_catalog(store)
        with store.transaction() as transaction:
            transaction.table(INVENTORY).extend(
                source.model_copy(
                    update={
                        "snapshot_id": "combined",
                        "declaration": source.declaration.model_copy(update={"consumer_receipt": None}),
                    }
                ).model_dump(mode="json")
                for source in sources
            )
            transaction.table(SNAPSHOTS).add(
                CatalogSnapshot(
                    snapshot_id="combined",
                    created_at="2026-11-20T00:00:00Z",
                    source_count=2,
                ).model_dump()
            )
            transaction.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id="combined").model_dump())
    load_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.begin() as connection:
        connection.execute(
            text("UPDATE atlas_sources SET declaration = declaration - 'consumer_requirements' - 'consumer_receipt'")
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        before = {row["id"]: row for row in client.get("/sources", params={"projection": "table"}).json()["sources"]}
    receipt = consumer_receipt("Marin:bespoke", actor_census_input(FIXTURE_DATASET, FIXTURE_REVISION, "1", 3))
    failed = receipt.model_copy(
        update={
            "failed_tasks": 3,
            "failures": {ConsumerReason.TOOL_UNREGISTERED: 3},
            "requirements": {ConsumerNeed.TOOL_PROVIDER: 3},
        }
    )
    decision = (
        ConsumerDecision(
            source_id="Marin:bespoke",
            runtime_commit=receipt.runtime_commit,
            task_specs_sha256=receipt.task_specs_sha256,
            reason=ConsumerReason.TOOL_UNREGISTERED,
            status=decision_status,
            user_ruling="fixture user ruling for this source",
        ).model_dump(mode="json")
        if decision_status is not None
        else None
    )
    monkeypatch.setattr(
        consumer_policy,
        "packaged_consumer_inputs",
        lambda: {
            "Marin:bespoke": {
                "consumer_requirements": [ConsumerNeed.TOOL_PROVIDER.value],
                "consumer_receipt": failed.model_dump(mode="json"),
                "consumer_decision": decision,
            }
        },
    )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        table = {row["id"]: row for row in client.get("/sources", params={"projection": "table"}).json()["sources"]}
        compact = {row["id"]: row for row in client.get("/source-statuses").json()["sources"]}
    assert table["Marin:bespoke"]["status"] == compact["Marin:bespoke"]["status"] == (decision_status or "Available")
    assert (
        table["Marin:bespoke"]["consumer_findings"]
        == compact["Marin:bespoke"]["consumer_findings"]
        == [ConsumerReason.TOOL_UNREGISTERED.value]
    )
    if decision_status is None:
        assert table["Marin:bespoke"]["filtration"] == before["Marin:bespoke"]["filtration"]
    else:
        assert table["Marin:bespoke"]["filtration"]["admitted_count"] == (0 if decision_status == "Excluded" else None)
    assert table["Marin:bespoke"]["filtration"]["population_count"] == 3
    assert table["Marin:standard"]["status"] == compact["Marin:standard"]["status"] == "Available"
    assert table["Marin:standard"]["filtration"] == before["Marin:standard"]["filtration"]


def test_status_inventory_matches_source_statuses_without_loading_task_evidence(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "status-inventory")
    publish_sample(root, "status-inventory")
    load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        complete = client.get("/sources").json()
        compact = client.get("/source-statuses").json()
    fields = (
        "id",
        "display_name",
        "status",
        "stage",
        "replacement_source_id",
        "exclusion_claims",
        "notes",
        "consumer_findings",
        "data_restriction",
        "intended_use",
        "method_tags",
        "updated_at",
    )
    assert compact["sources"] == [{key: source.get(key) for key in fields} for source in complete["sources"]]
    assert compact["snapshot_id"] == complete["snapshot_id"]
    assert compact["synced_at"] == complete["synced_at"]


@pytest.mark.parametrize("obsolete_summary", [False, True])
def test_catalog_upgrade_materializes_summaries_and_new_load_refreshes_results(
    catalog_database, catalog_query, tmp_path, obsolete_summary
):
    root = str(tmp_path / "summary-upgrade")
    publish_sample(root, "first", admitted_verdicts=("good", "good"))
    first = load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        before = client.get("/sources").json()
        # Both missing summaries and old counting policies must upgrade before serving.
        with catalog_database.begin() as connection:
            if obsolete_summary:
                connection.execute(
                    text("UPDATE atlas_cohort_summaries SET version=0,summary=jsonb_set(summary,'{grade}','\"bad\"')")
                )
            else:
                connection.execute(text("DELETE FROM atlas_cohort_summaries"))
            migrate(connection)
        assert client.get("/sources").json() == before
        assert before["sources"][0]["result"] == "GOOD"
        retained = catalog_query(
            "SELECT cohort,summary FROM atlas_cohort_summaries WHERE load_id=:load_id ORDER BY cohort",
            {"load_id": first["load_id"]},
        )["rows"]
        assert {row["cohort"] for row in retained} == {"admitted", "excluded"}
        publish_sample(root, "second", admitted_verdicts=("bad", "bad"))
        load_catalog(CurationCatalog(root), catalog_query)
        after = client.get("/sources").json()
        assert after["snapshot_id"] == "second" and after["sources"][0]["result"] == "BAD"
        assert (
            catalog_query(
                "SELECT cohort,summary FROM atlas_cohort_summaries WHERE load_id=:load_id ORDER BY cohort",
                {"load_id": first["load_id"]},
            )["rows"]
            == retained
        )


def test_filtered_zero_is_visible_without_admission_and_positive_output_still_needs_census(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "filtered")
    for retained in (0, 1):
        declaration = filtered_declaration(retained)
        if retained:
            # A positive retained population that still needs admission stays Available;
            # Excluded rows load as metadata without population admission.
            declaration = declaration.model_copy(update={"status": "Available"})
        snapshot_id = f"filtered-{retained}"
        source = SnapshotSource(
            snapshot_id=snapshot_id,
            id=declaration.id,
            declaration=declaration,
            assessment_key=None,
            latest_attempt_key=None,
            latest_attempt_status="skipped",
            reason="Current private filtration output",
        )
        with DataStore.open(root, writer_id=snapshot_id) as store:
            register_catalog(store)
            with store.transaction() as tx:
                tx.table(INVENTORY).add(source.model_dump(mode="json"))
                tx.table(SNAPSHOTS).add(
                    CatalogSnapshot(
                        snapshot_id=snapshot_id, created_at="2026-11-14T00:00:00Z", source_count=1
                    ).model_dump()
                )
                tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())
        catalog = CurationCatalog(root)
        if retained:
            with pytest.raises(ValueError, match="Admission census belongs to another current population"):
                load_catalog(catalog, catalog_query)
            continue
        load_catalog(catalog, catalog_query)
        with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
            row = client.get("/sources").json()["sources"][0]
            assert row["filtration"]["state"] == "excluded"
            assert row["filtration"]["admitted_count"] == 0
            assert row["status"] == "Excluded"
            detail = client.get("/source", params={"source_id": declaration.id}).json()
            assert detail["filtration"] == row["filtration"]
            assert detail["declaration"]["oracle_filtration"] == declaration.oracle_filtration.model_dump(mode="json")
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/sources").json()["snapshot_id"] == "filtered-0"


def test_backend_survey_refresh_and_failed_staging_preserve_selected_map(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:a")
    catalog = CurationCatalog(root)
    first = catalog_backend_map(catalog)
    identities = catalog_identity_map(catalog)
    loaded = load_surveyed_catalog(catalog, catalog_query, first, identities)
    second = first.model_copy(update={"generated_at": "2026-11-12T01:00:00+00:00"})

    def failing_query(sql, parameters):
        if "INSERT INTO atlas_backend_sources" in sql:
            raise RuntimeError("SQL transport interrupted")
        return catalog_query(sql, parameters)

    with pytest.raises(RuntimeError, match="transport interrupted"):
        load_surveyed_catalog(catalog, failing_query, second, identities)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/backends").json()["generated_at"] == first.generated_at
        assert client.get("/sources").json()["sources"][0]["backend_capabilities"]["basis"] == "converted_spec_census"
    refreshed = load_surveyed_catalog(catalog, catalog_query, second, identities)
    assert refreshed["load_id"] != loaded["load_id"]
    assert not load_surveyed_catalog(catalog, catalog_query, second, identities)["changed"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/backends").json()["generated_at"] == second.generated_at


def test_load_serves_catalog_grading_evidence_over_a_stale_survey_copy(catalog_database, catalog_query, tmp_path):
    """A survey row carries the grading evidence measured at survey time; the load serves the catalog's.

    Regression: R40's carried backend survey held pre-re-measure grading evidence for
    31 sources and the load refused instead of serving the current declaration's.
    """
    evidence = GradingEnvironmentEvidence(
        assessed_tasks=3,
        required_counts={"shared": 3},
        declared_counts={"shared": 3},
        status_counts={"matched": 3},
        reason_counts={"submitted answer": 3},
        evidence_url="s3://fixture/grading/current.json",
    )
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", grading_environment=evidence)
    catalog = CurationCatalog(root)
    survey = catalog_backend_map(catalog)
    stale = evidence.model_copy(update={"evidence_url": "s3://fixture/grading/at-survey.json"})
    survey = survey.model_copy(
        update={
            "sources": [
                row.model_copy(update={"grading_environment": stale.model_dump(mode="json")}) for row in survey.sources
            ]
        }
    )
    result = load_surveyed_catalog(catalog, catalog_query, survey, catalog_identity_map(catalog))
    assert result["grading_evidence_rebound"] == [row.source_id for row in survey.sources]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["backend_capabilities"]["grading_environment"] == evidence.model_dump(mode="json")


def test_grading_census_survives_catalog_loader_and_backend_projection(catalog_database, catalog_query, tmp_path):
    evidence = GradingEnvironmentEvidence(
        assessed_tasks=3,
        required_counts={"shared": 1, "separate": 2},
        declared_counts={"separate": 3},
        status_counts={"matched": 2, "mismatch": 1},
        reason_counts={"workspace state": 1, "submitted answer": 2},
        evidence_url="s3://fixture/grading/receipt.json",
    )
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", grading_environment=evidence)
    catalog = CurationCatalog(root)
    load_catalog(catalog, catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["grading_environment"] == evidence.model_dump(mode="json")
        assert source["backend_capabilities"]["grading_environment"] == evidence.model_dump(mode="json")
        table = client.get("/sources", params={"projection": "table"}).json()["sources"][0]
        assert table["grading_environment"] == evidence.model_dump(mode="json")
        assert table["backend_capabilities"]["backends"] == source["backend_capabilities"]["backends"]
        assert "inputs" not in table["backend_capabilities"]
        assert table["assessment"]["status"] == source["assessment"]["status"]
        assert "admitted" not in table["assessment"]


def test_source_fingerprint_is_stored_separately_and_failed_refresh_keeps_previous(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:a")
    catalog = CurationCatalog(root)
    fixture = Path(__file__).resolve().parents[3] / (
        "experiments/post_training/task_curation/tests/fixtures/source-overlap-fingerprints.json"
    )
    fingerprint = SourceFingerprint.model_validate(next(iter(json.loads(fixture.read_text()).values())))
    identities = catalog_identity_map(catalog).model_copy(update={"fingerprints": {"Marin:a": fingerprint}})
    backends = catalog_backend_map(catalog)
    load_surveyed_catalog(catalog, catalog_query, backends, identities)
    newer = identities.model_copy(update={"generated_at": "2026-11-12T02:00:00+00:00"})

    def failing_query(sql, parameters):
        if "INSERT INTO atlas_backend_cells" in sql:
            raise RuntimeError("Import interrupted after source fingerprints")
        return catalog_query(sql, parameters)

    with pytest.raises(RuntimeError, match="Import interrupted"):
        load_surveyed_catalog(catalog, failing_query, backends, newer)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        summary = client.get("/sources").json()["sources"][0]["source_fingerprint"]
        assert summary["sample_sha256"] == fingerprint.sample_sha256 and "sample" not in summary
        assert client.get("/source-fingerprint", params={"source_id": "Marin:a"}).json()["fingerprint"]["sample"] == [
            sample.model_dump(mode="json") for sample in fingerprint.sample
        ]
        assert client.get("/source-duplicates").json()["generated_at"] == identities.generated_at


def test_source_inventory_transfers_only_fingerprint_summary_and_keeps_full_evidence(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:a")
    catalog = CurationCatalog(root)
    fixture = Path(__file__).resolve().parents[3] / (
        "experiments/post_training/task_curation/tests/fixtures/source-overlap-fingerprints.json"
    )
    fingerprint = SourceFingerprint.model_validate(next(iter(json.loads(fixture.read_text()).values())))
    identities = catalog_identity_map(catalog).model_copy(update={"fingerprints": {"Marin:a": fingerprint}})
    load_surveyed_catalog(catalog, catalog_query, catalog_backend_map(catalog), identities)
    expected = fingerprint.model_dump(
        mode="json", include={"version", "seed", "task_count", "within_source_duplicates", "sample_sha256", "lineage"}
    )
    transferred = []

    def decode_json(payload):
        value = json.loads(payload)
        if isinstance(value, dict) and "sample_sha256" in value and "within_source_duplicates" in value:
            transferred.append(value)
        return value

    with catalog_database.connect() as connection:
        schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    observed_engine = create_engine(catalog_database.url, json_deserializer=decode_json)

    @event.listens_for(observed_engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")
        connection.commit()

    try:
        with TestClient(create_api(cast(AppletServices, CatalogServices(observed_engine)))) as client:
            sources = {row["id"]: row for row in client.get("/sources").json()["sources"]}
            assert sources["Marin:a"]["source_fingerprint"] == expected
            assert transferred == [expected]
            transferred.clear()
            evidence = client.get("/source-fingerprint", params={"source_id": "Marin:a"}).json()["fingerprint"]
            assert evidence == fingerprint.model_dump(mode="json")
            assert transferred == [evidence]
            with catalog_database.begin() as connection:
                connection.execute(text("UPDATE atlas_sources SET source_fingerprint=NULL WHERE id='Marin:a'"))
            assert client.get("/sources").json()["sources"][0]["source_fingerprint"] is None
    finally:
        observed_engine.dispose()


def test_source_cache_reuses_unchanged_evidence_and_invalidates_previous_pin_counts(
    catalog_database, catalog_query, tmp_path, monkeypatch
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:cached")
    load_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.connect() as connection:
        schema = connection.execute(text("SELECT current_schema()")).scalar_one()
        declaration = SourceDeclaration.model_validate(
            connection.execute(text("SELECT declaration FROM atlas_sources WHERE id='Marin:cached'")).scalar_one()
        )
    observation = count_evidence(declaration, oracle_tasks=0, blocked=1)
    monkeypatch.setattr(
        "infra.marina.apps.rl_data_catalog.server.app.SOURCE_COUNTS",
        json.dumps({"version": 1, "sources": [observation.model_dump(mode="json")]}),
    )
    deny_evidence_read = False

    def decode_json(payload):
        value = json.loads(payload)
        if deny_evidence_read and isinstance(value, dict) and "within_source_dedup" in value:
            raise RuntimeError("The unchanged stored declaration was fetched again")
        return value

    observed_engine = create_engine(catalog_database.url, json_deserializer=decode_json)

    @event.listens_for(observed_engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")
        connection.commit()

    try:
        with TestClient(create_api(cast(AppletServices, CatalogServices(observed_engine)))) as client:
            before = client.get("/sources", params={"projection": "table"}).json()
            assert before["sources"][0]["filtration"]["admitted_count"] == 2
            deny_evidence_read = True
            assert client.get("/sources", params={"projection": "table"}).json() == before
            deny_evidence_read = False
            with catalog_database.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE atlas_sources SET declaration=jsonb_set("
                        "declaration,'{dataset_revision}',CAST(:revision AS JSONB))"
                    ),
                    {"revision": json.dumps("e" * 40)},
                )
            changed = client.get("/sources", params={"projection": "table"}).json()["sources"][0]
            assert changed["dataset_revision"] == "e" * 40
            assert changed["filtration"]["admitted_count"] is None
            assert changed["filtration"]["example_count"] is None
    finally:
        observed_engine.dispose()


@pytest.fixture
def catalog_database(database_url: str) -> Iterator[Engine]:
    admin = create_engine(database_url)
    schema = "catalog_test_" + uuid.uuid4().hex
    with admin.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")

    try:
        with engine.begin() as connection:
            migrate(connection)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


@pytest.fixture
def catalog_query(catalog_database: Engine) -> AppletQuery:
    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        validate_query(sql)
        with catalog_database.begin() as connection:
            result = connection.execute(text(sql), parameters)
            return {
                "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                "row_count": result.rowcount,
            }

    return query


FIXTURE_DATASET = "open-athena/fixture"
FIXTURE_REVISION = "b" * 40


def fixture_lexical_clearance() -> Any:
    """Train-use lexical clearance bound to the fixture pin."""
    return decontamination_evidence(FIXTURE_DATASET, FIXTURE_REVISION, 3)


def publish_sample(
    root: str,
    snapshot_id: str,
    *,
    missing_packet: bool = False,
    source_id: str = "Marin:math",
    origin: SourceOrigin = SourceOrigin.MARIN,
    duplicates: int = 0,
    unassessed_oversight: OversightLevel | None = None,
    grading_environment: GradingEnvironmentEvidence | None = None,
    admitted_verdicts: tuple[str, str] = ("good", "unknown"),
    oversight: OversightLevel = OversightLevel.NONE,
    review_types: dict[str, dict[str, Any]] | None = None,
    lexical_clearance: bool = True,
) -> None:
    declaration = SourceDeclaration(
        license_receipt=fixture_license(source_id, FIXTURE_DATASET, FIXTURE_REVISION),
        id=source_id,
        display_name=source_display_name(source_id),
        origin=origin,
        build_path="test",
        upstream_dataset=source_id.partition(":")[2],
        upstream_name="Math",
        current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture tasks contain no actor environment.",
        grading_environment=grading_environment or grading_evidence(3 - duplicates),
        family="Marin:math",
        tags=[],
        task_count=3 - duplicates,
        within_source_dedup=deduplication_evidence(FIXTURE_DATASET, FIXTURE_REVISION, "1", 3).model_copy(
            update={"unique_count": 3 - duplicates, "within_source_duplicates": duplicates}
        ),
        notes="",
        oversight=unassessed_oversight or oversight,
        dataset_id=FIXTURE_DATASET,
        dataset_revision=FIXTURE_REVISION,
        lexical_contamination=fixture_lexical_clearance() if lexical_clearance else None,
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        intended_use="train",
        kind="Dataset",
        status="Available",
        pipeline={
            "name": "fixture",
            "version": "1",
            "source": FIXTURE_DATASET,
            "revision": FIXTURE_REVISION,
            "files": [],
        },
        actor_census_input=actor_census_input(FIXTURE_DATASET, FIXTURE_REVISION, "1", 3 - duplicates),
        consumer_receipt=consumer_receipt(
            source_id, actor_census_input(FIXTURE_DATASET, FIXTURE_REVISION, "1", 3 - duplicates)
        ),
    )
    curation = CurationSummary(
        status="completed",
        mode="full",
        source_revision=None,
        recipe_revision="1",
        raw_population_count=3,
        raw_sample_count=3,
        processed_rows=3,
        unprocessed_rows=0,
        admission_counts={"admitted": 2, "rejected": 1},
        verification=None,
    )
    key = snapshot_id
    with DataStore.open(root, writer_id=snapshot_id) as store:
        register_catalog(store)
        items = []
        for task_id, cohort, verdict, has_task in (
            ("one", "admitted", admitted_verdicts[0], True),
            ("two", "admitted", admitted_verdicts[1], admitted_verdicts[1] != "unknown"),
            ("three", "excluded", "bad", True),
        ):
            packet = {"task": {"id": task_id, "prompt": "Null: \u0000; Unicode: π"}} if has_task else {}
            uri = store.write_object(f"{key}/{task_id}.json", json.dumps(packet).encode())
            if missing_packet and task_id == "one":
                uri = "finestore://blobs/absent.json"
            items.append(
                AssessmentItem(
                    pipeline_key=key,
                    task_id=task_id,
                    cohort=cohort,
                    status="reviewed",
                    verdict=verdict,
                    explanation="sample finding",
                    evidence_ids=["task"] if has_task else ["curation"],
                    packet_uri=uri,
                    has_task=has_task,
                    omissions=[] if has_task else ["TaskSpec unavailable"],
                    exclusion_reason="Reference contradiction" if cohort == "excluded" else None,
                    **(review_types or {}).get(task_id, {}),
                )
            )
        curation_uri = store.write_object(f"{key}/curation.json", curation.model_dump_json().encode())
        store.flush()
        assessment = Assessment(
            pipeline_key=key,
            source_id=source_id,
            scope="full",
            model={"model": "glm", "revision": "1", "max_tokens": 4096, "max_attempts": 3, "retry_max_tokens": 8192},
            rubric="Check answerability",
            curation=curation,
            curation_uri=curation_uri,
            traces=[],
        )
        snapshot = CatalogSnapshot(snapshot_id=snapshot_id, created_at="2026-10-13T00:00:00Z", source_count=1)
        source = SnapshotSource(
            snapshot_id=snapshot_id,
            id=source_id,
            declaration=declaration,
            assessment_key=key if unassessed_oversight is None else None,
            latest_attempt_key=key,
            latest_attempt_status="complete",
            reason=None,
            curation=curation,
        )
        with store.transaction() as tx:
            tx.table(ASSESSMENTS).add(assessment.model_dump(mode="json"))
            tx.table(ITEMS).extend(i.model_dump(mode="json") for i in items)
            tx.table(POPULATIONS).extend(
                AssessmentPopulation(pipeline_key=key, cohort=cohort, population=count, sample_count=count).model_dump(
                    mode="json"
                )
                for cohort, count in (("admitted", 2), ("excluded", 1))
            )
            tx.table(INVENTORY).add(source.model_dump(mode="json"))
            tx.table(SNAPSHOTS).add(snapshot.model_dump())
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())


@dataclass(frozen=True)
class CatalogServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


def curriculum_release(tmp_path: Path, unavailable_status: str = "insufficient_context") -> Path:
    declarations = [
        SourceDeclaration(
            id=identifier,
            display_name=source_display_name(identifier),
            origin=SourceOrigin.MARIN,
            build_path="test",
            upstream_dataset=identifier.partition(":")[2],
            upstream_name=identifier.upper(),
            current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
            daytona_safety="not_applicable",
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture tasks contain no actor environment.",
            family="Numbers",
            tags=[],
            task_count=1,
            within_source_dedup=deduplication_evidence(FIXTURE_DATASET, FIXTURE_REVISION, "1"),
            notes="",
            dataset_id=FIXTURE_DATASET,
            dataset_revision=FIXTURE_REVISION,
            url=None,
            type="RLVR",
            turns="Single-turn",
            is_benchmark=False,
            kind="Dataset",
            status="Available",
            pipeline={
                "name": "fixture",
                "version": "1",
                "source": FIXTURE_DATASET,
                "revision": FIXTURE_REVISION,
                "files": [],
            },
        )
        for identifier in ("Marin:a", "Marin:b", "Marin:inventory")
    ]
    records = []
    for source_id, task_id, content, capability_score in (
        ("Marin:a", "one", "h1", 0.9),
        ("Marin:a", "two", "h2", 0.65),
        ("Marin:a", "three", "h3", None),
        ("Marin:b", "one", "h1", 0.9),
    ):
        records.append(
            {
                "source_id": source_id,
                "source_key": source_id + "-key",
                "task_id": task_id,
                "task_hash": source_id + task_id,
                "content_hash": content,
                "intended_use": "train",
                "population": "converted",
                "description_status": "ready" if capability_score is not None else unavailable_status,
                "description": content if capability_score is not None else "",
                "reason": "",
                "task_preview": "Count public objects.",
                "preview_truncated": False,
                "instruction_characters": 21,
                "matches": (
                    {
                        "subjects": [
                            {
                                "subject_id": "D01",
                                "facet": "subject_domain",
                                "membership_similarity": 0.9,
                                "capability_id": "add",
                                "candidates": [
                                    {
                                        "subject_id": "D01",
                                        "capability_id": "add",
                                        "similarity": capability_score,
                                        "anchor_id": "addition",
                                    }
                                ],
                            }
                        ],
                        "capabilities": [],
                    }
                    if capability_score is not None
                    else None
                ),
            }
        )
    data = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(records), data)
    release = AlignmentRelease(
        release_id="first",
        created_at="2026-10-14T00:00:00Z",
        curriculum_key="curriculum-1",
        population="converted",
        sample_size=None,
        task_count=400,
        ready_count=300,
        insufficient_context_count=100 * int(unavailable_status == "insufficient_context"),
        sources=[
            {
                "source_id": identifier,
                "source_key": identifier + "-key",
                "population": "converted",
                "intended_use": "train",
                "basis": "matched",
                "task_count": count,
                "ready_count": ready,
                "insufficient_context_count": count - ready if unavailable_status == "insufficient_context" else 0,
            }
            for identifier, count, ready in (("Marin:a", 300, 200), ("Marin:b", 100, 100))
        ],
        inventory=declarations,
        nodes=[
            {
                "node_id": "subject:D01",
                "subject_id": "D01",
                "parent_id": None,
                "kind": "subject",
                "name": "Mathematics",
                "description": "Numbers",
            },
            {
                "node_id": "arithmetic",
                "subject_id": "D01",
                "parent_id": "subject:D01",
                "kind": "group",
                "name": "Arithmetic",
                "description": "Calculate",
            },
            {
                "node_id": "add",
                "subject_id": "D01",
                "parent_id": "arithmetic",
                "kind": "capability",
                "name": "Addition",
                "description": "Add numbers",
            },
        ],
        paths=[str(tmp_path / "full-corpus-not-loaded-by-atlas.parquet")],
        lineage={"alignment_root": "fixture"},
        embedding_spec={
            "model": "fixture",
            "revision": "fixture",
            "tokenizer_revision": "fixture",
            "dimensions": 2,
            "max_tokens": 512,
            "compute_dtype": "float32",
        },
    )
    path = tmp_path / "release.json"
    path.write_text(release.model_dump_json())
    statistics = []
    for source, denominator, assigned, subject_only, unavailable in (
        ("Marin:a", 300, 100, 100, 100),
        ("Marin:b", 100, 100, 0, 0),
    ):
        for metric, count in (
            ("denominator", denominator),
            ("assigned", assigned),
            ("subject_only", subject_only),
            (unavailable_status, unavailable),
        ):
            if count:
                statistics.append(dict(source_id=source, metric=metric, subject_id="", node_id="", instances=count))
        for node_id, count in (("subject:D01", assigned + subject_only), ("arithmetic", assigned), ("add", assigned)):
            statistics.append(
                dict(source_id=source, metric="placement", subject_id="D01", node_id=node_id, instances=count)
            )
        statistics.append(
            dict(source_id=source, metric="capability", subject_id="D01", node_id="subject:D01", instances=assigned)
        )
    statistics_path = tmp_path / "statistics.parquet"
    pq.write_table(pa.Table.from_pylist(statistics), statistics_path)
    coverage = CurriculumCoverage(
        coverage_id="first",
        alignment_release_id="first",
        curriculum_key="curriculum-1",
        subject_distance=0.45,
        capability_distance=0.25,
        task_count=400,
        paths=[str(statistics_path)],
        sample_paths=[str(data)],
        sample_task_count=4,
        source_sample_size=100,
        area_sample_size=10,
    )
    (tmp_path / "coverage.json").write_text(coverage.model_dump_json())
    return path


@pytest.mark.parametrize("retired_source", ["Marin:a", "Marin:inventory", "MarinSkyRL:aime24"])
def test_retired_curriculum_is_rejected_before_idempotency_or_sql(tmp_path, monkeypatch, retired_source):
    path = curriculum_release(tmp_path)
    if retired_source == "MarinSkyRL:aime24":
        release = json.loads(path.read_text())
        release["inventory"][0].update(
            id=retired_source,
            display_name=source_display_name(retired_source),
            origin=SourceOrigin.HUGGINGFACEH4,
            upstream_dataset="aime24",
        )
        release["sources"][0]["source_id"] = retired_source
        path.write_text(json.dumps(release))
    inputs = [path, tmp_path / "coverage.json", tmp_path / "tasks.parquet", tmp_path / "statistics.parquet"]
    original_bytes = {file: file.read_bytes() for file in inputs}
    ledger = tmp_path / "retirement-ledger.json"
    ledger.write_text(json.dumps({"entries": [{"id": retired_source, "disposition": "retired_admission_check"}]}))
    reference = tmp_path / "retirement-reference.json"
    reference.write_text(
        json.dumps(
            {
                "repo": "fixture/retirements",
                "revision": "a" * 40,
                "path": "retirements.json",
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(retirement, "RETIREMENTS_PATH", reference)
    monkeypatch.setattr(registry, "hf_hub_download", lambda **_kwargs: str(ledger))
    retirement.retired_admission_source_ids.cache_clear()
    queries = []

    def already_selected(sql, parameters):
        queries.append((sql, parameters))
        return {"rows": [{"release_id": "first", "release_uri": str(path)}]}

    with pytest.raises(ValueError, match=f"Retired sources.*{canonical_source_id(retired_source)}"):
        load_curriculum(str(path), str(tmp_path / "coverage.json"), already_selected)
    assert queries == []
    assert {file: file.read_bytes() for file in inputs} == original_bytes


def test_curriculum_gateway_keeps_large_archived_inventory_out_of_count_reads(catalog_database, catalog_query, tmp_path):
    path = curriculum_release(tmp_path)
    load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        expected = client.get("/curriculum/invariants").json()
    stored = catalog_query("SELECT load_id,manifest FROM curriculum_loads", {})["rows"][0]
    manifest = stored["manifest"]
    manifest["inventory"][0]["notes"] = "x" * (MAX_QUERY_RESPONSE_BYTES + 1)
    catalog_query(
        "UPDATE curriculum_loads SET manifest=CAST(:manifest AS JSONB) WHERE load_id=:load_id",
        {"manifest": json.dumps(manifest), "load_id": stored["load_id"]},
    )
    assert len(json.dumps(manifest).encode()) > MAX_QUERY_RESPONSE_BYTES

    def gateway(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        result = catalog_query(sql, parameters)
        if len(json.dumps(result, default=str).encode()) > MAX_QUERY_RESPONSE_BYTES:
            raise MarinaRequestError(413, f"query response exceeds {MAX_QUERY_RESPONSE_BYTES} bytes")
        return result

    candidate = CandidateApi(gateway)
    try:
        observed = candidate.get("api/curriculum/invariants")
    finally:
        candidate.close()
    assert observed == expected
    assert catalog_query("SELECT manifest FROM curriculum_loads WHERE load_id=:load_id", {"load_id": stored["load_id"]})[
        "rows"
    ] == [{"manifest": manifest}]


@pytest.mark.parametrize("unavailable_status", ["insufficient_context", "skipped"])
def test_curriculum_precomputed_population_samples_pagination_and_atomic_load(
    catalog_database, catalog_query, tmp_path, monkeypatch, unavailable_status
):
    monkeypatch.setattr("infra.marina.apps.rl_data_catalog.load_curriculum.DELETE_BATCH_SIZE", 1)
    engine = catalog_database
    query = catalog_query

    path = curriculum_release(tmp_path, unavailable_status)
    coverage_path = tmp_path / "coverage.json"
    with monkeypatch.context() as chunking:
        chunking.setattr("infra.marina.apps.rl_data_catalog.load.MAX_BATCH_BYTES", 3500)
        assert load_curriculum(str(path), str(coverage_path), query)["counts"]["tasks"] == 4
        assert load_curriculum(str(path), str(coverage_path), query)["changed"] is False
    catalog_root = str(tmp_path / "catalog")
    publish_sample(catalog_root, "graded", source_id="Marin:a", duplicates=1)
    load_catalog(CurationCatalog(catalog_root), query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(engine)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["task_count"] == 2
        assert source["within_source_dedup"]["raw_count"] == 3
        assert source["within_source_dedup"]["within_source_duplicates"] == 1
        assert source["coverage"] == {
            "example_count": 300,
            "intended_use": "train",
            "basis": "matched",
            "top_area": {"subject_id": "D01", "node_id": "subject:D01", "name": "Mathematics", "instances": 200},
        }
        assert source["assessment"]["admitted"]["grade"] == "good"
        first = client.get("/curriculum").json()
        invariants = client.get("/curriculum/invariants").json()
        checked = check_curriculum_document(invariants)
        assert [(case["use"], case["denominator"]) for case in checked["cases"]] == [
            ("all", 400),
            ("train", 400),
            ("eval", 0),
        ]
        assert first["totals"] == {
            "denominator": 400,
            "assigned": 200,
            "subject_only": 100,
            "out_of_scope": 0,
            "insufficient_context": 100 * int(unavailable_status == "insufficient_context"),
            "skipped": 100 * int(unavailable_status == "skipped"),
        }
        assert {row["source_id"] for row in first["sources"]} == {"Marin:a", "Marin:b", "Marin:inventory"}
        assert next(row for row in first["sources"] if row["source_id"] == "Marin:inventory")["alignment"] is None
        assert {(row["node_id"], row["count"]) for row in first["area_counts"]} == {
            ("subject:D01", 300),
            ("arithmetic", 200),
            ("add", 200),
        }
        assert first["examples"]["sample_task_count"] == 4
        default = client.get("/curriculum", params={"subject_id": "D01"}).json()
        assert default["scope_summary"] == {"subject_members": 300, "capability_matches": 200}
        area_params = {"subject_id": "D01", "node_id": "subject:D01", "scope": "unlisted"}
        unlisted = client.get("/curriculum/area", params=area_params).json()
        assert unlisted["count"] == 100
        assert [(row["source_id"], row["count"]) for row in unlisted["sources"]] == [("Marin:a", 100)]
        assert [
            (row["source_id"], row["task_id"])
            for row in client.get("/curriculum/tasks", params=area_params).json()["tasks"]
        ] == [("Marin:a", "two")]
        complete_area = client.get("/curriculum/area", params={"subject_id": "D01", "node_id": "arithmetic"}).json()
        assert complete_area["count"] == 200
        assert sum(row["count"] for row in complete_area["sources"]) == 200
        assert client.get("/curriculum/area", params={**area_params, "source_id": "Marin:b"}).json()["count"] == 0
        source_view = client.get("/curriculum", params={"source_id": "Marin:a"}).json()
        assert source_view["totals"]["denominator"] == 300
        assert {(row["node_id"], row["count"]) for row in source_view["area_counts"]} == {
            ("subject:D01", 200),
            ("arithmetic", 100),
            ("add", 100),
        }
        assert client.get("/curriculum", params={"intended_use": "eval"}).json()["totals"]["denominator"] == 0
        source_tasks = client.get("/curriculum/tasks", params={"source_id": "Marin:b"}).json()
        assert [(row["source_id"], row["task_id"]) for row in source_tasks["tasks"]] == [("Marin:b", "one")]
        diverse = client.get("/curriculum/tasks", params={"limit": 2}).json()
        remainder = client.get("/curriculum/tasks", params={"limit": 2, "cursor": diverse["next_cursor"]}).json()
        assert diverse["tasks"] + remainder["tasks"] == client.get("/curriculum/tasks").json()["tasks"]
        assert [(row["source_id"], row["task_id"]) for row in diverse["tasks"] + remainder["tasks"]] == [
            ("Marin:a", "one"),
            ("Marin:b", "one"),
            ("Marin:a", "three"),
            ("Marin:a", "two"),
        ]
        assert not remainder["next_cursor"]
        params = {"node_id": "add", "subject_id": "D01", "limit": 1}
        page = client.get("/curriculum/tasks", params=params).json()
        next_page = client.get("/curriculum/tasks", params={**params, "cursor": page["next_cursor"]}).json()
        assert [(row["source_id"], row["task_id"]) for row in page["tasks"] + next_page["tasks"]] == [
            ("Marin:a", "one"),
            ("Marin:b", "one"),
        ]
        assert not next_page["next_cursor"]
        assert client.get("/curriculum/task", params={"source_id": "Marin:a", "task_id": "one"}).json()["task"][
            "task_preview"
        ]
        # A task may match a capability elsewhere while remaining subject-only here.
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO curriculum_matches SELECT load_id,"
                    "'Marin:a','two','D02','other','subject_domain',0.9,0.9 "
                    "FROM curriculum_current WHERE name='latest'"
                )
            )
        local_remainder = client.get("/curriculum/tasks", params=area_params).json()["tasks"]
        assert [(row["task_id"], row["outcome"]) for row in local_remainder] == [("two", "subject_only")]
        assigned_params = {"subject_id": "D01", "node_id": "subject:D01", "outcome": "assigned", "limit": 2}
        assigned = client.get("/curriculum/tasks", params=assigned_params).json()
        assigned_next = client.get(
            "/curriculum/tasks", params={**assigned_params, "cursor": assigned["next_cursor"]}
        ).json()
        assert [(row["source_id"], row["task_id"]) for row in assigned["tasks"] + assigned_next["tasks"]] == [
            ("Marin:a", "one"),
            ("Marin:b", "one"),
            ("Marin:a", "two"),
        ]
        assert not assigned_next["next_cursor"]
        manifest = json.loads(path.read_text())
        coverage = json.loads(coverage_path.read_text())
        path.write_text(json.dumps({**manifest, "release_id": "bad", "task_count": 401}))
        coverage_path.write_text(
            json.dumps({**coverage, "coverage_id": "bad", "alignment_release_id": "bad", "task_count": 401})
        )
        with pytest.raises(ValueError, match="population counts"):
            load_curriculum(str(path), str(coverage_path), query)
        assert client.get("/curriculum").json()["release_id"] == "first"
        path.write_text(json.dumps({**manifest, "release_id": "next", "curriculum_key": "curriculum-2"}))
        coverage_path.write_text(
            json.dumps(
                {**coverage, "coverage_id": "next", "alignment_release_id": "next", "curriculum_key": "curriculum-2"}
            )
        )
        assert load_curriculum(str(path), str(coverage_path), query)["changed"]
        shutil.rmtree(tmp_path)
        assert client.get("/curriculum").json()["release_id"] == "next"
        assert client.get("/curriculum", params={"release": "first"}).json()["release_id"] == "first"
        releases = client.get("/curriculum/releases").json()["releases"]
        assert [(row["release_id"], row["selected"]) for row in releases] == [("next", True), ("first", False)]
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE curriculum_matches"))
            connection.execute(text("DROP TABLE curriculum_tasks"))
        assert client.get("/curriculum").json()["totals"] == first["totals"]
        assert client.get("/sources").json()["sources"][0]["coverage"] == source["coverage"]
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE atlas_sources SET declaration = "
                    "jsonb_set(declaration, '{dataset_revision}', '\"new-current-pin\"'::jsonb)"
                )
            )
        # The stale-pin projection names the last-computed revision instead of clearing:
        # the alignment exists but predates the pin, so it cannot present as current.
        stale = {
            "last_computed_dataset_revision": FIXTURE_REVISION,
            "unavailable_reason": f"Not yet computed for this pin (last computed at {FIXTURE_REVISION})",
        }
        assert client.get("/sources").json()["sources"][0]["coverage"] == stale
        assert client.get("/source", params={"source_id": "Marin:a"}).json()["coverage"] == stale
        assert client.get("/curriculum").json()["totals"] == first["totals"]


@pytest.mark.parametrize(
    ("present", "tag", "expected"),
    [
        (False, None, "open"),
        (True, "open", "open"),
        (True, "malformed", "private_always"),
        (True, None, "private_always"),
        (True, "private_always", "private_always"),
    ],
)
def test_curriculum_sample_transport_omits_oversized_unused_metadata(
    catalog_database, catalog_query, tmp_path, present, tag, expected
):
    path = curriculum_release(tmp_path)
    load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                "UPDATE curriculum_sources SET declaration=(declaration-'data_restriction') || "
                "jsonb_build_object('unused_catalog_payload',CAST(:payload AS TEXT)) WHERE source_id='Marin:a'"
            ),
            {"payload": "x" * (6 * 1024 * 1024)},
        )
        if present:
            connection.execute(
                text(
                    "UPDATE curriculum_sources SET declaration=jsonb_set(declaration,'{data_restriction}',"
                    "CAST(:tag AS JSONB)) WHERE source_id='Marin:a'"
                ),
                {"tag": json.dumps(tag)},
            )
        metadata = loaded_release(connection, "")
        parameters = task_parameters(metadata, CoverageQuery("all", "Marin:a", ""), "", "", "all", "all", "", 50)
        rows = [dict(row) for row in connection.execute(text(TASKS_SQL), parameters).mappings()]
    assert rows
    assert max(len(json.dumps(row["source_declaration"]).encode()) for row in rows) < 256
    assert all(("data_restriction" in row["source_declaration"]) is present for row in rows)
    response = task_response(metadata, rows, 50, source_restrictions(rows))
    assert all(row["data_restriction"] == expected for row in response["tasks"])
    for row in response["tasks"]:
        assert (row["task_preview"] == "Private data: content is not displayed.") is (expected == "private_always")


@pytest.mark.parametrize("upgrade", ["missing", "obsolete"])
def test_source_area_summaries_upgrade_without_rewriting_curriculum(catalog_database, catalog_query, tmp_path, upgrade):
    path = curriculum_release(tmp_path)
    coverage = tmp_path / "coverage.json"
    load_curriculum(str(path), str(coverage), catalog_query)
    with catalog_database.begin() as connection:
        expected = [
            dict(row)
            for row in connection.execute(text("SELECT * FROM curriculum_source_areas ORDER BY source_id")).mappings()
        ]
        statistics = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM curriculum_statistics ORDER BY source_id,metric,subject_id,node_id")
            ).mappings()
        ]
        selected = connection.execute(text("SELECT load_id FROM curriculum_current WHERE name='latest'")).scalar_one()
        assert next(row for row in expected if row["source_id"] == "Marin:inventory")["area"] is None
        if upgrade == "missing":
            connection.execute(text("DELETE FROM curriculum_source_areas"))
        else:
            connection.execute(text("UPDATE curriculum_source_areas SET version=0,area=NULL"))
    if upgrade == "missing":
        with catalog_database.begin() as connection:
            migrate(connection)
    else:
        assert load_curriculum(str(path), str(coverage), catalog_query)["changed"] is False
    with catalog_database.connect() as connection:
        assert [
            dict(row)
            for row in connection.execute(text("SELECT * FROM curriculum_source_areas ORDER BY source_id")).mappings()
        ] == expected
        assert [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM curriculum_statistics ORDER BY source_id,metric,subject_id,node_id")
            ).mappings()
        ] == statistics
        assert (
            connection.execute(text("SELECT load_id FROM curriculum_current WHERE name='latest'")).scalar_one()
            == selected
        )


def test_source_list_reads_materialized_areas_while_statistics_are_locked(catalog_database, catalog_query, tmp_path):
    path = curriculum_release(tmp_path)
    load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:a")
    load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        before = client.get("/sources").json()
        assert before["sources"][0]["coverage"]["top_area"] == {
            "subject_id": "D01",
            "node_id": "subject:D01",
            "name": "Mathematics",
            "instances": 200,
        }

        def bounded_lock_wait(connection):
            connection.exec_driver_sql("SET LOCAL lock_timeout='1s'")

        event.listen(catalog_database, "begin", bounded_lock_wait)
        try:
            with catalog_database.begin() as blocker:
                blocker.execute(text("LOCK TABLE curriculum_statistics IN ACCESS EXCLUSIVE MODE"))
                assert client.get("/sources").json() == before
        finally:
            event.remove(catalog_database, "begin", bounded_lock_wait)


def test_atlas_load_serves_grades_packets_and_archive_without_object_storage(
    catalog_database, catalog_query, tmp_path: Path
) -> None:
    engine = catalog_database
    root = str(tmp_path / "catalog")
    publish_sample(root, "first")

    query = catalog_query

    loaded = load_catalog(CurationCatalog(root), query)
    assert loaded["counts"] == {
        "sources": 1,
        "assessments": 1,
        "populations": 2,
        "items": 3,
        "packets": 3,
        "backend_sources": 1,
        "backend_surveys": 1,
        "backend_cells": 301,
        "identity_surveys": 1,
        "duplicate_reviews": 0,
    }
    assert load_catalog(CurationCatalog(root), query)["changed"] is False
    with engine.begin() as connection:
        connection.execute(
            text(
                """
            INSERT INTO catalog_reviews (id,source_id,collection,updated_at)
            VALUES ('legacy','Marin:math','{"quality":"good","reviews":[]}'::jsonb,NOW())
        """
            )
        )
        connection.execute(
            text(
                """
            INSERT INTO review_artifacts (review_id,path,content,sha256)
            VALUES ('legacy','nested/trace.txt','saved trace','digest')
        """
            )
        )
    services = cast(AppletServices, CatalogServices(engine))
    with TestClient(create_api(services)) as client:
        inventory = client.get("/sources").json()
        assert inventory["sources"][0]["coverage"] is None
        assessment = inventory["sources"][0]["assessment"]
        assert inventory["snapshot_id"] == "first"
        assert assessment["admitted"]["grade"] == "good"
        assert assessment["admitted"]["verdict_counts"] == {"good": 1, "some_issues": 0, "bad": 0, "unknown": 1}
        assert assessment["excluded"]["grade"] == "bad"
        report = client.get("/assessment", params={"source_id": "Marin:math"}).json()["report"]
        assert report["assessment"]["admitted"]["missing_evidence"] == 1
        assert report["evidence"]["items"][2]["exclusion_reason"] == "Reference contradiction"
        packet = client.get("/evidence", params={"source_id": "Marin:math", "task_id": "one"}).json()
        assert packet["packet"]["task"]["prompt"] == "Null: \u0000; Unicode: π"
        assert client.get("/evidence", params={"source_id": "Marin:math", "task_id": "other"}).status_code == 404
        assert client.get("/assessment", params={"source_id": "absent"}).status_code == 404
        assert client.get("/archive").json()["reviews"][0]["id"] == "legacy"
        assert client.get("/archive/reviews/legacy").json()["collection"]["quality"] == "good"
        assert client.get("/archive/reviews/legacy/artifacts/nested/trace.txt").text == "saved trace"
        publish_sample(root, "broken", missing_packet=True)
        with pytest.raises(FileNotFoundError):
            load_catalog(CurationCatalog(root), query)
        assert client.get("/sources").json()["snapshot_id"] == "first"
        assert client.get("/evidence", params={"source_id": "Marin:math", "task_id": "one"}).json() == packet
        publish_sample(root, "next")
        assert load_catalog(CurationCatalog(root), query)["changed"] is True
        shutil.rmtree(root)
        assert client.get("/sources").json()["snapshot_id"] == "next"
        assert client.get("/evidence", params={"source_id": "Marin:math", "task_id": "one"}).json() == packet
        assert client.get("/archive/reviews/legacy/artifacts/nested/trace.txt").text == "saved trace"


@pytest.mark.parametrize("oversight", [OversightLevel.SAMPLE, OversightLevel.FULL])
def test_atlas_rejects_oversight_without_result_and_preserves_selected_catalog(
    catalog_database, catalog_query, tmp_path, oversight
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "reviewed")
    load_catalog(CurationCatalog(root), catalog_query)
    publish_sample(root, "missing-result", unassessed_oversight=oversight)
    with pytest.raises(ValueError, match="SAMPLE/FULL quality requires a computed Result"):
        load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        inventory = client.get("/sources").json()
        assert inventory["snapshot_id"] == "reviewed"
        assert inventory["sources"][0]["result"] == "GOOD"
    catalog = CurationCatalog(root)
    declaration = next(iter(catalog.sources("missing-result"))).declaration.model_dump(mode="json")
    assert load_surveyed_catalog(
        catalog,
        catalog_query,
        catalog_backend_map(catalog),
        catalog_identity_map(catalog),
        display_policy=CatalogDisplayPolicy.PREVIEW,
    )["changed"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        detail = client.get("/source", params={"source_id": source["id"]}).json()
        assert source["quality"] == detail["quality"] == "QUICK"
        assert source["quality_review_status"] == detail["quality_review_status"]
        assert source["quality_review_status"]
        assert detail["declaration"] == declaration
        assert source["result"] is source["issue_rate"] is source["review_completeness"] is None
        assert source["result_provisional"] is None
    with pytest.raises(ValueError, match="SAMPLE/FULL quality requires a computed Result"):
        load_catalog(catalog, catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/sources").json()["snapshot_id"] == "missing-result"


@pytest.mark.parametrize("failure", ["missing", "stale", "no_grader"])
@pytest.mark.parametrize("display_policy", list(CatalogDisplayPolicy))
def test_atlas_blocks_invalid_admission_census_without_replacing_selected_catalog(
    catalog_database, catalog_query, tmp_path, failure, display_policy
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "reviewed")
    load_catalog(CurationCatalog(root), catalog_query)
    publish_sample(root, "next")
    source_id = "Marin:math"
    catalog = CurationCatalog(root)
    source = next(row for row in catalog.sources("next") if row.id == source_id)
    declaration = source.declaration.model_dump(mode="json")
    if failure == "missing":
        declaration["grading_environment"] = None
    elif failure == "stale":
        declaration["actor_census_input"]["dataset_revision"] = "stale"
    else:
        declaration["grading_environment"]["reason_counts"] = {"No executable grader is declared": 3}
    changed = {**source.model_dump(mode="json"), "declaration": declaration}
    with DataStore.open(root) as store:
        register_catalog(store)
        with store.transaction() as transaction:
            transaction.table(INVENTORY).add(changed)
    catalog = CurationCatalog(root)
    # The stale census is refused at declaration validation: the model binds the
    # actor census to the declaration's exact inputs, so the versioned snapshot
    # reader refuses the row before the census binding runs. The missing and
    # no-grader mutations stay schema-valid and are refused by the binding.
    with pytest.raises(
        ValueError,
        match=r"Admission census|Actor census input|without runnable graders|refusing to guess a conversion",
    ):
        load_surveyed_catalog(
            catalog,
            catalog_query,
            catalog_backend_map(catalog),
            catalog_identity_map(catalog),
            display_policy=display_policy,
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/sources").json()["snapshot_id"] == "reviewed"


def test_release_keeps_legacy_selected_catalog_readable_after_rejected_strict_reload(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "legacy")
    publish_sample(root, "reviewed")
    load_catalog(CurationCatalog(root), catalog_query)
    publish_sample(root, "pending", unassessed_oversight=OversightLevel.SAMPLE)
    catalog = CurationCatalog(root)
    pending = next(iter(catalog.sources("pending"))).declaration.model_dump(mode="json")
    catalog_query(
        "UPDATE atlas_sources SET declaration=CAST(:declaration AS JSONB), assessment_key=NULL, curation=NULL "
        "WHERE load_id=(SELECT load_id FROM atlas_current WHERE name='latest')",
        {"declaration": json.dumps(pending)},
    )
    with catalog_database.begin() as connection:
        migrate(connection)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        for reload in (False, True):
            if reload:
                with pytest.raises(ValueError, match="SAMPLE/FULL quality requires a computed Result"):
                    load_catalog(catalog, catalog_query)
            response = client.get("/sources")
            assert response.status_code == 200
            assert response.json()["snapshot_id"] == "reviewed"
            source = response.json()["sources"][0]
            detail_response = client.get("/source", params={"source_id": source["id"]})
            assert detail_response.status_code == 200
            detail = detail_response.json()
            assert source["quality"] == detail["quality"] == "QUICK"
            assert source["quality_review_status"] == detail["quality_review_status"]
            assert source["quality_review_status"]
            assert source["result"] is detail["result"] is None
            assert detail["declaration"] == pending


@pytest.mark.parametrize("verdicts, provisional", [(("good", "good"), False), (("good", "unknown"), True)])
def test_loaded_sample_result_uses_required_review_completeness(
    catalog_database, catalog_query, tmp_path, verdicts, provisional
):
    root = str(tmp_path / "sample")
    publish_sample(root, "sample", admitted_verdicts=verdicts, oversight=OversightLevel.SAMPLE)
    load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        detail = client.get("/source", params={"source_id": source["id"]}).json()
        assert source["quality"] == "SAMPLE" and source["result"] == "GOOD"
        assert source["result_provisional"] is detail["result_provisional"] is provisional
        assert (
            source["review_completeness"]
            == detail["review_completeness"]
            == {
                "required": 2,
                "reviewed": 1 if provisional else 2,
                "coverage_complete": True,
                "provisional": provisional,
            }
        )
        # The execution scope is FULL, but the required review panel is smaller than the retained source.
        assert source["assessment"]["scope"] == "full"
        assert source["within_source_dedup"]["unique_count"] == 3


def test_unchanged_catalog_refreshes_pre_completeness_load(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "sample", admitted_verdicts=("good", "good"), oversight=OversightLevel.SAMPLE)
    catalog = CurationCatalog(root)
    load_catalog(catalog, catalog_query)
    # Version 5 cached SAMPLE finality before the missing-review tolerance.
    catalog_query("UPDATE atlas_loads SET review_completeness_version=5", {})
    catalog_query("UPDATE atlas_sources SET review_completeness=NULL", {})
    assert load_catalog(catalog, catalog_query)["changed"] is True
    assert load_catalog(catalog, catalog_query)["changed"] is False
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["review_completeness"]["required"] == source["review_completeness"]["reviewed"] == 2
        assert source["result_provisional"] is False


def seed_previous_available(catalog_database, source_ids: list[str]) -> None:
    """Seed a legacy selected load whose Available rows grandfather the gate.

    The lexical-clearance gate refuses NEW Available admissions; these tests
    render display states that only exist on sources admitted before the gate,
    so the previous load's Available set is seeded explicitly.
    """
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO atlas_loads (load_id, catalog_root, snapshot_id, created_at) "
                "VALUES ('legacy', 'fixture', 'legacy', '2026-11-01T00:00:00Z')"
            )
        )
        for source_id in source_ids:
            connection.execute(
                text(
                    "INSERT INTO atlas_sources (load_id, id, declaration, latest_attempt_status, diagnostics) "
                    "VALUES ('legacy', :id, CAST(:declaration AS JSONB), 'skipped', '[]'::jsonb)"
                ),
                {"id": source_id, "declaration": json.dumps({"id": source_id, "status": "Available"})},
            )
        connection.execute(
            text(
                "INSERT INTO atlas_current (name, load_id) VALUES ('latest', 'legacy') "
                "ON CONFLICT (name) DO UPDATE SET load_id = EXCLUDED.load_id"
            )
        )


def test_loaded_lexical_evidence_drives_contamination_states(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "lexical-catalog")
    cases = [
        (expected, with_declaration_admission_census(declaration))
        for expected, declaration in contamination_declarations()
    ]
    benchmark_train = next(declaration for _, declaration in cases if declaration.id == "Marin:required")
    historical = benchmark_train.model_dump(mode="json")
    historical.pop("intended_use")
    with DataStore.open(root, writer_id="lexical") as store:
        register_catalog(store)
        with store.transaction() as tx:
            tx.table(INVENTORY).extend(
                SnapshotSource(
                    snapshot_id="lexical",
                    id=declaration.id,
                    declaration=declaration,
                    assessment_key=None,
                    latest_attempt_key=None,
                    latest_attempt_status="skipped",
                    reason=None,
                    curation=None,
                ).model_dump(mode="json")
                for _, declaration in cases
            )
            tx.table(SNAPSHOTS).add(
                CatalogSnapshot(
                    snapshot_id="lexical",
                    created_at="2026-11-13T00:00:00Z",
                    source_count=len(cases),
                ).model_dump()
            )
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id="lexical").model_dump())
    seed_previous_available(catalog_database, [declaration.id for _, declaration in cases])
    load_catalog(CurationCatalog(root), catalog_query)
    shutil.rmtree(root)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        sources = {row["id"]: row for row in client.get("/sources").json()["sources"]}
        for expected, declaration in cases:
            source = sources[declaration.id]
            detail = client.get("/source", params={"source_id": declaration.id}).json()
            assert source["contamination_status"] == detail["contamination_status"] == expected
            assert detail["declaration"]["lexical_contamination"] == source["lexical_contamination"]
        assert sources["Marin:eval"]["lexical_contamination"]["contaminated_tasks"] == 1
        assert sources["Marin:unmeasured"]["lexical_contamination"] is None
        assert sources["Marin:unmeasured"]["issue_rate"] is None
        assert sources["Marin:eval"]["issue_rate"] == 0.0
        assert sources["Marin:required"]["intended_use"] == "train"

    # A declaration without a Use no longer serves as free text: the loader
    # refuses it at admission, so no row can reach the page without a typed
    # SourceUse value.
    root = str(tmp_path / "lexical-historical")
    with DataStore.open(root, writer_id="lexical-historical") as store:
        register_catalog(store)
        with store.transaction() as tx:
            tx.table(INVENTORY).add(
                SnapshotSource(
                    snapshot_id="lexical-historical",
                    id="Marin:historical",
                    declaration=SourceDeclaration.model_validate(
                        {
                            **historical,
                            "id": "Marin:historical",
                            "display_name": source_display_name("Marin:historical"),
                            "consumer_receipt": (
                                consumer_receipt("Marin:historical", benchmark_train.actor_census_input).model_dump(
                                    mode="json"
                                )
                            ),
                            "upstream_dataset": "historical",
                            "is_benchmark": True,
                            "license_receipt": fixture_license(
                                "Marin:historical", historical["dataset_id"], historical["dataset_revision"]
                            ),
                        }
                    ),
                    assessment_key=None,
                    latest_attempt_key=None,
                    latest_attempt_status="skipped",
                    reason=None,
                    curation=None,
                ).model_dump(mode="json")
            )
            tx.table(SNAPSHOTS).add(
                CatalogSnapshot(
                    snapshot_id="lexical-historical", created_at="2026-11-13T00:00:00Z", source_count=1
                ).model_dump()
            )
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id="lexical-historical").model_dump())
    # No previously-available seeding: the row is a new Available admission and the
    # declaration-license gate must reject it before any grandfathering applies.
    with pytest.raises(ValueError, match="Admission requires an explicit intended_use"):
        load_catalog(CurationCatalog(root), catalog_query)


def test_loaded_assemblies_keep_inflight_and_uncertain_results_provisional(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "lexical-catalog")
    cases = [
        (expected, with_declaration_admission_census(declaration)) for expected, declaration in review_declarations()
    ]
    seed_previous_available(catalog_database, [declaration.id for _, declaration in cases])
    with DataStore.open(root, writer_id="lexical") as store:
        register_catalog(store)
        with store.transaction() as tx:
            tx.table(INVENTORY).extend(
                SnapshotSource(
                    snapshot_id="lexical",
                    id=declaration.id,
                    declaration=declaration,
                    assessment_key=None,
                    latest_attempt_key=None,
                    latest_attempt_status="skipped",
                    reason=None,
                    curation=None,
                ).model_dump(mode="json")
                for _, declaration in cases
            )
            tx.table(SNAPSHOTS).add(
                CatalogSnapshot(
                    snapshot_id="lexical",
                    created_at="2026-11-13T00:00:00Z",
                    source_count=len(cases),
                ).model_dump()
            )
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id="lexical").model_dump())
    load_catalog(CurationCatalog(root), catalog_query)
    shutil.rmtree(root)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        sources = {row["id"]: row for row in client.get("/sources").json()["sources"]}
        for provisional, declaration in cases:
            source = sources[declaration.id]
            detail = client.get("/source", params={"source_id": declaration.id}).json()
            assert source["result_provisional"] is detail["result_provisional"] is provisional
            assert source["review_completeness"] == detail["review_completeness"]
            assert declaration.quality_result is not None
            assert source["review_completeness"]["reviewed"] == declaration.quality_result.reviewed
            if declaration.quality_result.reviewed == 0:
                assert source["quality"] == detail["quality"] == "QUICK"
                assert source["result"] is detail["result"] is None
                assert source["issue_rate"] is detail["issue_rate"] is None
                assert source["quality_review_status"] == detail["quality_review_status"]
                assert source["quality_review_status"]
                assert detail["declaration"] == declaration.model_dump(mode="json")
        assert sources["Marin:sample-complete"]["quality"] == sources["Marin:sample-uncertain"]["quality"] == "SAMPLE"
        content_only = sources["Marin:sample-content"]
        assert content_only["quality"] == "SAMPLE" and content_only["oversight"] == "none"
        assert content_only["result"] == "GOOD" and content_only["result_provisional"] is False
        assert sources["Marin:full-uncertain"]["quality"] == "FULL"
        assert sources["Marin:full-inflight"]["review_completeness"]["coverage_complete"] is False


def test_a_competing_atlas_cutover_prevents_an_older_load_from_being_selected(catalog_query, tmp_path):
    older, newer = str(tmp_path / "older"), str(tmp_path / "newer")
    publish_sample(older, "older")
    publish_sample(newer, "newer")
    interrupted = False

    def overlapping_query(sql, parameters):
        nonlocal interrupted
        result = catalog_query(sql, parameters)
        if sql.startswith("INSERT INTO atlas_packets") and not interrupted:
            interrupted = True
            assert load_catalog(CurationCatalog(newer), catalog_query)["changed"]
        return result

    with pytest.raises(RuntimeError, match="Atlas selection changed"):
        load_catalog(CurationCatalog(older), overlapping_query)
    selected = catalog_query(
        "SELECT l.snapshot_id, count(p.*) AS packets FROM atlas_current c JOIN atlas_loads l USING(load_id) "
        "JOIN atlas_packets p USING(load_id) GROUP BY l.snapshot_id",
        {},
    )["rows"]
    assert selected == [{"snapshot_id": "newer", "packets": 3}]


def test_a_new_catalog_snapshot_during_import_leaves_the_serving_load_unchanged(catalog_query, tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "first")
    load_catalog(CurationCatalog(root), catalog_query)
    publish_sample(root, "next")

    def changing_catalog_query(sql, parameters):
        result = catalog_query(sql, parameters)
        if sql.startswith("INSERT INTO atlas_packets"):
            publish_sample(root, "later")
        return result

    with pytest.raises(RuntimeError, match="Catalog selection changed"):
        load_catalog(CurationCatalog(root), changing_catalog_query)
    assert catalog_query("SELECT l.snapshot_id FROM atlas_current c JOIN atlas_loads l USING(load_id)", {})["rows"] == [
        {"snapshot_id": "first"}
    ]
    assert load_catalog(CurationCatalog(root), catalog_query)["changed"]


def test_atlas_selection_preserves_a_load_started_after_its_cutover(catalog_query, tmp_path):
    roots = [str(tmp_path / name) for name in ("baseline", "first", "later")]
    for root, name in zip(roots, ("baseline", "first", "later"), strict=True):
        publish_sample(root, name)
    load_catalog(CurationCatalog(roots[0]), catalog_query)
    staged, continue_load = Event(), Event()
    later = None

    def later_query(sql, parameters):
        result = catalog_query(sql, parameters)
        if sql.startswith("INSERT INTO atlas_packets"):
            staged.set()
            assert continue_load.wait(timeout=15)
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:

        def first_query(sql, parameters):
            nonlocal later
            result = catalog_query(sql, parameters)
            if sql.startswith("UPDATE atlas_current SET load_id") and later is None:
                later = executor.submit(load_catalog, CurationCatalog(roots[2]), later_query)
                assert staged.wait(timeout=15)
            return result

        try:
            assert load_catalog(CurationCatalog(roots[1]), first_query)["changed"]
        finally:
            continue_load.set()
        assert later is not None
        assert later.result(timeout=30)["changed"]
    # The three snapshots retain all three assessments and their three packets each.
    assert catalog_query(
        "SELECT l.snapshot_id, count(p.*) AS packets FROM atlas_current c JOIN atlas_loads l USING(load_id) "
        "JOIN atlas_packets p USING(load_id) GROUP BY l.snapshot_id",
        {},
    )["rows"] == [{"snapshot_id": "later", "packets": 9}]
    assert catalog_query(
        "SELECT l.snapshot_id, count(p.*) AS packets FROM atlas_loads l "
        "JOIN atlas_packets p USING(load_id) GROUP BY l.snapshot_id ORDER BY l.snapshot_id",
        {},
    )["rows"] == [
        {"snapshot_id": "baseline", "packets": 3},
        {"snapshot_id": "first", "packets": 6},
        {"snapshot_id": "later", "packets": 9},
    ]


@pytest.mark.parametrize(
    "verdicts, curation_status, expected",
    [
        (["good"] * 99 + ["bad"], "completed", "good"),
        (["good"] * 90 + ["bad"] * 10, "completed", "good"),
        (["good"] * 89 + ["bad"] * 11, "completed", "some_issues"),
        (["good"] * 51 + ["some_issues"] * 49, "completed", "some_issues"),
        (["good"] * 50 + ["some_issues"] * 50, "completed", "bad"),
        (["good"] * 79 + ["some_issues"] * 21, "completed", "some_issues"),
        (["good"] * 80 + ["unknown"] * 20, "completed", "good"),
        (["good"] * 79 + ["unknown"] * 21, "completed", "good"),
        (["good", "missing_task"], "completed", "good"),
        ([], "completed", "unknown"),
        ([], "gated", "unknown"),
        ([], "incomplete", "unknown"),
    ],
)
def test_source_grade_uses_sample_fractions_and_zero_output(catalog_database, verdicts, curation_status, expected):
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO atlas_assessments VALUES
                ('load', 'pipeline', 'source', 'sample', '{}'::jsonb, 'rubric',
                 :curation, 'curation-uri', '[]'::jsonb)
            """
            ),
            {"curation": json.dumps({"status": curation_status, "processed_rows": 100})},
        )
        connection.execute(
            text("INSERT INTO atlas_populations VALUES ('load', 'pipeline', 'admitted', :count, :count)"),
            {"count": len(verdicts)},
        )
        for index, verdict in enumerate(verdicts):
            connection.execute(
                text(
                    """
                    INSERT INTO atlas_items VALUES
                    ('load', 'pipeline', :id, 'admitted', 'reviewed', :verdict, 'finding',
                     '[]'::jsonb, 'packet-uri', :has_task, '[]'::jsonb, NULL)
                """
                ),
                {
                    "id": str(index),
                    "verdict": "good" if verdict == "missing_task" else verdict,
                    "has_task": verdict != "missing_task",
                },
            )
        row = (
            connection.execute(
                text("SELECT grade, good_fraction, issue_count, issue_rate, result_reviewed FROM atlas_cohort_summary")
            )
            .mappings()
            .one()
        )
    assert row["grade"] == expected
    assert result({}, dict(row)) == (None if expected == "unknown" else expected.upper())
    assert row["issue_count"] == sum(v in {"bad", "some_issues"} for v in verdicts)
    assert row["result_reviewed"] == sum(v not in {"unknown", "missing_task"} for v in verdicts)
    if verdicts == ["good"] * 99 + ["bad"]:
        assert row["good_fraction"] == 0.99


# The cohort view a retained Atlas database already runs: audit counts were introduced in
# 09d6d89d without touching the retained column prefix.
RETAINED_COHORT_SUMMARY_VIEW = """
CREATE OR REPLACE VIEW atlas_cohort_summary AS
WITH counts AS (
SELECT p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count,
    COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed') AS reviewed,
    COUNT(i.task_id) FILTER (WHERE NOT i.has_task) AS missing_evidence,
    COUNT(i.task_id) FILTER (WHERE jsonb_array_length(i.omissions) > 0) AS packets_with_omissions,
    jsonb_build_object(
        'good', COUNT(i.task_id) FILTER (WHERE i.verdict = 'good'),
        'some_issues', COUNT(i.task_id) FILTER (WHERE i.verdict = 'some_issues'),
        'bad', COUNT(i.task_id) FILTER (WHERE i.verdict = 'bad'),
        'unknown', COUNT(i.task_id) FILTER (WHERE i.verdict = 'unknown')
    ) AS verdict_counts,
    COUNT(i.task_id) AS item_count,
    COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed' AND i.has_task AND i.verdict = 'good') AS good_count,
    COUNT(i.task_id) FILTER (
        WHERE i.status = 'reviewed' AND i.has_task AND i.verdict <> 'unknown'
    ) AS conclusive_count
FROM atlas_populations p JOIN atlas_assessments a USING (load_id, pipeline_key)
LEFT JOIN atlas_items i
    ON i.load_id = p.load_id AND i.pipeline_key = p.pipeline_key AND i.cohort = p.cohort
GROUP BY p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count, a.curation
)
SELECT load_id, pipeline_key, cohort, population, sample_count, reviewed,
    missing_evidence, packets_with_omissions, verdict_counts,
    CASE
        WHEN sample_count = 0 OR item_count <> sample_count OR conclusive_count = 0 THEN 'unknown'
        WHEN 2 * (conclusive_count - good_count) >= conclusive_count THEN 'bad'
        WHEN 10 * (conclusive_count - good_count) <= conclusive_count THEN 'good'
        ELSE 'some_issues'
    END AS grade,
    good_count::double precision / NULLIF(conclusive_count, 0) AS good_fraction,
    CASE
        WHEN sample_count = 0 THEN 'No sample available'
        WHEN item_count <> sample_count THEN 'Sample coverage is incomplete'
        WHEN conclusive_count = 0 THEN 'No conclusive reviewed tasks'
        WHEN 2 * (conclusive_count - good_count) >= conclusive_count THEN 'At least 50% of reviewed tasks have issues'
        WHEN 10 * (conclusive_count - good_count) <= conclusive_count THEN 'At least 90% of reviewed tasks are clean'
        ELSE 'More than 10% and fewer than 50% of reviewed tasks have issues'
    END AS grade_reason,
    conclusive_count AS result_reviewed,
    conclusive_count - good_count AS issue_count,
    (conclusive_count - good_count)::double precision / NULLIF(conclusive_count, 0) AS issue_rate,
    reviewed - conclusive_count AS uncertain_count,
    sample_count - reviewed AS unavailable_count
FROM counts;
"""


def test_migrate_appends_audit_counts_to_the_retained_cohort_view(database_url: str):
    """Upgrading a retained database keeps the cohort view's rows and retained column prefix.

    Postgres rejects CREATE OR REPLACE VIEW when an existing column changes name or
    position, so the audit count projections must append after the retained columns.
    """
    admin = create_engine(database_url)
    schema = "catalog_test_" + uuid.uuid4().hex
    with admin.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")

    retained = [
        statement.strip()
        for statement in SCHEMA_SQL.split(";")
        if statement.strip() and not statement.lstrip().startswith("CREATE OR REPLACE VIEW atlas_cohort_summary")
    ]
    try:
        with engine.begin() as connection:
            for statement in [*retained, RETAINED_COHORT_SUMMARY_VIEW]:
                connection.execute(text(statement))
            connection.execute(
                text(
                    """
                    INSERT INTO atlas_assessments VALUES
                    ('load', 'pipeline', 'source', 'sample', '{}'::jsonb, 'rubric',
                     :curation, 'curation-uri', '[]'::jsonb)
                    """
                ),
                {"curation": json.dumps({"status": "completed", "processed_rows": 100})},
            )
            connection.execute(text("INSERT INTO atlas_populations VALUES ('load', 'pipeline', 'admitted', 2, 2)"))
            for index, verdict in enumerate(["good", "bad"]):
                connection.execute(
                    text(
                        """
                        INSERT INTO atlas_items VALUES
                        ('load', 'pipeline', :id, 'admitted', 'reviewed', :verdict, 'finding',
                         '[]'::jsonb, 'packet-uri', TRUE, '[]'::jsonb, NULL)
                        """
                    ),
                    {"id": str(index), "verdict": verdict},
                )
            before = (
                connection.execute(
                    text(
                        "SELECT grade, good_fraction, issue_count, issue_rate, unavailable_count"
                        " FROM atlas_cohort_summary"
                    )
                )
                .mappings()
                .one()
            )
        with engine.begin() as connection:
            migrate(connection)
        with engine.begin() as connection:
            columns = [
                row["column_name"]
                for row in connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema=current_schema() AND table_name = 'atlas_cohort_summary' "
                        "ORDER BY ordinal_position"
                    )
                ).mappings()
            ]
            after = connection.execute(text("SELECT * FROM atlas_cohort_summary")).mappings().one()
        assert columns[-2:] == ["defect_counts", "confidence_counts"]
        for column, value in before.items():
            assert after[column] == value, f"{column} changed across the upgrade"
        assert after["defect_counts"] == {
            "missing_context": 0,
            "ambiguity": 0,
            "wrong_reference": 0,
            "answer_leakage": 0,
            "malformed": 0,
            "rubric_mismatch": 0,
        }
        assert after["confidence_counts"] == {"high": 0, "medium": 0, "low": 0, "unrecorded": 2}
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def test_historical_alias_urls_keep_assessments_packets_and_sql_counts(catalog_database, catalog_query, tmp_path):
    old = "MarinSkyRL:aime24"
    current = canonical_source_id(old)
    peer = "MarinSkyRL:math500"
    group = {
        "group_id": "historical-math",
        "canonical_source_id": old,
        "members": [
            {"source_id": old, "overlap_with_canonical": None},
            {"source_id": peer, "overlap_with_canonical": 0.8},
        ],
    }
    duplicate_review = {
        "similarity": {
            "left": old,
            "right": peer,
            "rank_score": 0.8,
            "instruction_similarity": 0.8,
            "instruction_overlap": 0.8,
            "reference_overlap": 0.8,
            "test_overlap": 0.8,
            "task_id_overlap": 0.8,
            "url_overlap": 0.8,
            "shared_lineage": False,
        },
        "verdict": {"label": "partial_overlap", "overlap_fraction": 0.8, "reason": "Shared math tasks"},
        "prompt_sha256": "a" * 64,
        "model": "fixture",
        "completion_id": "legacy",
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen", source_id=current, origin=SourceOrigin.HUGGINGFACEH4)
    with DataStore.open(root) as store:
        register_catalog(store)
        record = {
            key: value
            for key, value in store.read_view().point(INVENTORY, snapshot_id="frozen", id=current).items()
            if key in SnapshotSource.model_fields
        }
        record.update(
            snapshot_id="legacy",
            id=old,
            declaration={
                **record["declaration"],
                "id": old,
                "display_name": source_display_name(old),
            },
        )
        assessment = {
            key: value
            for key, value in store.read_view().point(ASSESSMENTS, pipeline_key="frozen").items()
            if key in Assessment.model_fields
        }
        assessment["source_id"] = old
        with store.transaction() as transaction:
            transaction.table(INVENTORY).add(record)
            transaction.table(ASSESSMENTS).add(assessment)
            transaction.table(SNAPSHOTS).add(
                {"snapshot_id": "legacy", "created_at": "2026-11-12T00:00:00Z", "source_count": 1}
            )
            transaction.table(CURRENT).add({"name": "latest", "snapshot_id": "legacy"})
    catalog = CurationCatalog(root)
    original_packet = catalog.packet("frozen", "one")
    loaded = load_catalog(catalog, catalog_query)
    assert {key: loaded["counts"][key] for key in ("sources", "assessments", "populations", "items", "packets")} == {
        "sources": 1,
        "assessments": 1,
        "populations": 2,
        "items": 3,
        "packets": 3,
    }
    # Exercise a retained SQL load too: its physical keys still use the historical ID.
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                "UPDATE atlas_sources SET id=:old, declaration=jsonb_set(declaration, '{id}', "
                "to_jsonb(CAST(:old AS text)))"
            ),
            {"old": old},
        )
        connection.execute(
            text(
                "UPDATE atlas_backend_sources SET source_id=:old, "
                "capabilities=jsonb_set(capabilities, '{source_id}', to_jsonb(CAST(:old AS text)))"
            ),
            {"old": old},
        )
        connection.execute(text("UPDATE atlas_assessments SET source_id=:old"), {"old": old})
        connection.execute(
            text("UPDATE atlas_sources SET duplicate_group=CAST(:duplicate_group AS JSONB)"),
            {"duplicate_group": json.dumps(group)},
        )
        connection.execute(
            text(
                "INSERT INTO atlas_duplicate_reviews SELECT load_id, :left, :right, CAST(:review AS JSONB) "
                "FROM atlas_current WHERE name='latest'"
            ),
            {"left": old, "right": peer, "review": json.dumps(duplicate_review)},
        )
    shutil.rmtree(root)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["id"] == source["backend_capabilities"]["source_id"] == current
        assert source["duplicate_group"] is None
        duplicates = client.get("/source-duplicates").json()
        assert duplicates["groups"] == []
        assert duplicates["stored_groups"][0]["canonical_source_id"] == current
        assert [member["source_id"] for member in duplicates["stored_groups"][0]["members"]] == [
            current,
            canonical_source_id(peer),
        ]
        assert duplicates["reviews"][0]["similarity"]["left"] == current
        assert duplicates["reviews"][0]["similarity"]["right"] == canonical_source_id(peer)
        for identifier in (old, current):
            report = client.get("/assessment", params={"source_id": identifier}).json()["report"]
            assert report["source_id"] == current and report["pipeline_key"] == "frozen"
            assert len(report["evidence"]["items"]) == 3
            packet = client.get("/evidence", params={"source_id": identifier, "task_id": "one"}).json()["packet"]
            assert packet == original_packet
            assert client.get("/source-fingerprint", params={"source_id": identifier}).json()["source_id"] == current
            detail = client.get("/source", params={"source_id": identifier}).json()
            assert detail["declaration"]["id"] == current
            assert detail["assessment"]["pipeline_key"] == "frozen"
            assert detail["backend_capabilities"]["basis"] == "converted_spec_census"
            assert detail["result"] == "GOOD"

    with catalog_database.connect() as connection:
        assert connection.execute(text("SELECT duplicate_group FROM atlas_sources")).scalar_one() == group
        assert connection.execute(text("SELECT review FROM atlas_duplicate_reviews")).scalar_one() == duplicate_review


def test_historical_curriculum_artifacts_load_without_rewriting_inputs(catalog_database, catalog_query, tmp_path):
    old = "MarinSkyRL:aime24"
    current = canonical_source_id(old)
    path = curriculum_release(tmp_path)
    release = json.loads(path.read_text())
    for row in release["sources"]:
        if row["source_id"] == "Marin:a":
            row["source_id"] = old
    for row in release["inventory"]:
        if row["id"] == "Marin:a":
            row.update(
                id=old,
                display_name=source_display_name(old),
                origin=SourceOrigin.HUGGINGFACEH4,
                upstream_dataset="aime24",
            )
    path.write_text(json.dumps(release))
    inputs = [path, tmp_path / "coverage.json", tmp_path / "tasks.parquet", tmp_path / "statistics.parquet"]
    for parquet in inputs[2:]:
        table = pq.read_table(parquet)
        rows = table.to_pylist()
        for row in rows:
            if row["source_id"] == "Marin:a":
                row["source_id"] = old
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), parquet)
    original_bytes = {file: file.read_bytes() for file in inputs}
    loaded = load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    assert loaded["population_tasks"] == 400
    assert {file: file.read_bytes() for file in inputs} == original_bytes
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        for identifier in (old, current):
            coverage = client.get("/curriculum", params={"source_id": identifier}).json()
            assert coverage["totals"]["denominator"] == 300
            source = next(row for row in coverage["sources"] if row["source_id"] == current)
            assert source["alignment"]["source_id"] == current
            tasks = client.get("/curriculum/tasks", params={"source_id": identifier, "limit": 3}).json()["tasks"]
            assert [(row["source_id"], row["task_id"]) for row in tasks] == [
                (current, "one"),
                (current, "three"),
                (current, "two"),
            ]


def test_historical_curriculum_alias_keeps_counts_samples_and_pagination(catalog_database, catalog_query, tmp_path):
    old = "MarinSkyRL:aime24"
    current = canonical_source_id(old)
    path = curriculum_release(tmp_path)
    load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    with catalog_database.begin() as connection:
        # Seed a retained historical population without changing its foreign-key contract.
        connection.execute(
            text(
                "INSERT INTO curriculum_sources SELECT load_id, :old, declaration, family, alignment "
                "FROM curriculum_sources WHERE source_id='Marin:a'"
            ),
            {"old": old},
        )
        connection.execute(
            text(
                "INSERT INTO curriculum_tasks SELECT load_id, :old, task_id, content_hash, "
                "description_status, description, detail FROM curriculum_tasks WHERE source_id='Marin:a'"
            ),
            {"old": old},
        )
        connection.execute(text("UPDATE curriculum_matches SET source_id=:old WHERE source_id='Marin:a'"), {"old": old})
        connection.execute(
            text("UPDATE curriculum_statistics SET source_id=:old WHERE source_id='Marin:a'"), {"old": old}
        )
        connection.execute(text("DELETE FROM curriculum_tasks WHERE source_id='Marin:a'"))
        connection.execute(text("DELETE FROM curriculum_sources WHERE source_id='Marin:a'"))
        connection.execute(
            text(
                "UPDATE curriculum_sources SET alignment=jsonb_set(alignment, '{source_id}', "
                "to_jsonb(CAST(:old AS text))) WHERE source_id=:old"
            ),
            {"old": old},
        )
        connection.execute(
            text(
                "UPDATE curriculum_tasks SET detail=jsonb_set(detail, '{source_id}', "
                "to_jsonb(CAST(:old AS text))) WHERE source_id=:old"
            ),
            {"old": old},
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        for identifier in (old, current):
            coverage = client.get("/curriculum", params={"source_id": identifier}).json()
            assert coverage["totals"]["denominator"] == 300
            source = next(row for row in coverage["sources"] if row["source_id"] == current)
            assert source["alignment"]["source_id"] == current
            first = client.get("/curriculum/tasks", params={"source_id": identifier, "limit": 1}).json()
            second = client.get(
                "/curriculum/tasks", params={"source_id": identifier, "limit": 2, "cursor": first["next_cursor"]}
            ).json()
            assert [(row["source_id"], row["task_id"]) for row in first["tasks"] + second["tasks"]] == [
                (current, "one"),
                (current, "three"),
                (current, "two"),
            ]
            assert not second["next_cursor"]
            detail = client.get("/curriculum/task", params={"source_id": identifier, "task_id": "one"}).json()["task"]
            assert detail["source_id"] == current and detail["task_preview"]


def test_difficulty_archive_reader_excludes_finestore_metadata(tmp_path):
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measurement = difficulty_measurement(declaration, Solver.SMALL)
    archive = str(tmp_path / "difficulty")
    with DataStore.open(archive) as store:
        register_difficulty(store)
        with store.transaction() as transaction:
            transaction.table(MEASUREMENTS).add(measurement.model_dump(mode="json"))

    assert read_measurements(ReadView(archive)) == [measurement]


def test_difficulty_loader_accepts_existing_measurement_without_blocked_count():
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measurement = difficulty_measurement(declaration, Solver.SMALL)
    stored = measurement.model_dump(mode="json", exclude={"blocked_tasks"})

    def query(sql: str, _parameters: dict[str, Any]) -> dict[str, Any]:
        # the batched readback selects the id with the measurement projection
        if sql.lstrip().startswith("SELECT"):
            return {"rows": [{"measurement_id": measurement.measurement_id, "measurement": stored}]}
        return {"row_count": 0}

    assert load_measurements([measurement], query) == {"inserted": 0, "records": 1}


@pytest.mark.parametrize("changed_input", ["release", "population"])
def test_difficulty_archive_load_and_source_pin_change_preserve_history(
    catalog_database, catalog_query, tmp_path, changed_input
):
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measurement = difficulty_measurement(declaration, Solver.SMALL)
    measurement = measurement.model_copy(update={"created_at": "2026-11-13T01:00:00+00:00"})
    older = difficulty_measurement(declaration, Solver.SMALL, solved=3).model_copy(
        update={"created_at": "2026-11-13T02:00:00+02:00"}
    )
    archive = str(tmp_path / "difficulty")
    with DataStore.open(archive) as store:
        register_difficulty(store)
        with store.transaction() as transaction:
            transaction.table(MEASUREMENTS).add(measurement.model_dump(mode="json"))
            transaction.table(MEASUREMENTS).add(older.model_dump(mode="json"))
    records = read_measurements(ReadView(archive))
    assert {record.measurement_id for record in records} == {measurement.measurement_id, older.measurement_id}
    assert load_measurements(records, catalog_query)["inserted"] == 2
    assert load_measurements(records, catalog_query)["inserted"] == 0
    with catalog_database.begin() as connection:
        connection.execute(text("INSERT INTO atlas_loads VALUES ('difficulty','fixture','difficulty',NOW(),NOW())"))
        connection.execute(text("UPDATE atlas_current SET load_id='difficulty' WHERE name='latest'"))
        connection.execute(
            text(
                "INSERT INTO atlas_sources (load_id,id,declaration,latest_attempt_status,diagnostics) "
                "VALUES ('difficulty',:id,CAST(:declaration AS JSONB),'skipped','{}'::jsonb)"
            ),
            {"id": declaration.id, "declaration": declaration.model_dump_json()},
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        model = source["difficulty"]["models"][0]
        assert (model["n"], model["solved"], model["solve_rate"]) == (3, 1, 1 / 3)
        assert model["infrastructure_attempts"] == 2
        assert "tasks" not in model["sample"]
        detail = client.get("/source", params={"source_id": declaration.id}).json()
        assert detail["difficulty"]["models"][0] == model
        assert model["sample"]["pre_oracle_filter"] and model["sample"]["resample_after_oracle_filter"]
        table = client.get("/sources", params={"projection": "table"}).json()["sources"][0]
        assert table["id"] == source["id"] and table["filtration"] == source["filtration"]
        assert table["backend_capabilities"] is None
        assert table["daytona"] == source["daytona"]
        assert "quality_result" not in table and "license_receipt" not in table
        assert "historical" not in table["difficulty"]
        compact_model = table["difficulty"]["models"][0]
        for key in ["solver", "status", "reason", "n", "solved", "solve_rate", "wilson_95", "protocol_id"]:
            assert compact_model.get(key) == model.get(key)
        assert compact_model["model"] == {"model": model["model"]["model"], "revision": model["model"]["revision"]}
        assert compact_model["sample"] == {"pre_oracle_filter": True}
        assert source["difficulty"]["historical"][0]["measurement_id"] == older.measurement_id
        assert detail["difficulty"]["historical"] == source["difficulty"]["historical"]
        large = difficulty_measurement(declaration, Solver.LARGE, solved=3).model_copy(
            update={"created_at": "2026-11-13T00:30:00+00:00"}
        )
        load_measurements([large], catalog_query)
        refreshed = client.get("/sources", params={"projection": "table"}).json()["sources"][0]
        assert next(row for row in refreshed["difficulty"]["models"] if row["solver"] == "large")["solve_rate"] == 1
        assert [row["solver"] for row in source["difficulty"]["models"]] == [solver.value for solver in Solver]
        assert all(row["solve_rate"] is None and row["reason"] for row in source["difficulty"]["models"][1:])
        changed = declaration.model_dump(mode="json")
        if changed_input == "release":
            changed["current_release"] = {"repo": "fixture", "revision": "f" * 40, "url": "https://example.org/release"}
        else:
            changed["actor_census_input"]["task_specs_sha256"] = "f" * 64
        with catalog_database.begin() as connection:
            connection.execute(
                text("UPDATE atlas_sources SET declaration=CAST(:value AS JSONB)"), {"value": json.dumps(changed)}
            )
        source = client.get("/sources").json()["sources"][0]
        assert all(row["solve_rate"] is None for row in source["difficulty"]["models"])
        assert source["difficulty"]["historical"][0]["measurement_id"] == measurement.measurement_id
    altered = measurement.model_copy(update={"gpu_hours": 2})
    with pytest.raises(ValueError, match="different evidence"):
        load_measurements([altered], catalog_query)


def test_difficulty_counting_excludes_infrastructure_and_keeps_partial_credit():
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measurement = difficulty_measurement(declaration, Solver.SMALL)
    attempts = [
        DifficultyAttempt(task_id="0", status="infrastructure", reward=None),
        DifficultyAttempt(task_id="0", status="scored", reward=1.0),
        DifficultyAttempt(task_id="1", status="scored", reward=0.4),
    ]
    partial = measurement_from_attempts(measurement, attempts)
    assert (partial.status, partial.n, partial.solved, partial.pending_tasks, partial.infrastructure_attempts) == (
        "pending",
        2,
        1,
        1,
        1,
    )
    complete = measurement_from_attempts(
        measurement, [*attempts, DifficultyAttempt(task_id="2", status="scored", reward=0)]
    )
    assert (complete.status, complete.n, complete.solved, complete.pending_tasks) == ("measured", 3, 1, 0)
    assert complete.wilson_95.low == pytest.approx(0.06149194472039621)
    assert complete.wilson_95.high == pytest.approx(0.7923403991979522)
    with pytest.raises(ValueError, match="count a task twice"):
        measurement_from_attempts(measurement, [*attempts, attempts[1]])


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_full_sample_at_solve_rate_endpoint_keeps_a_valid_wilson_interval(reward):
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measurement = difficulty_measurement(declaration, Solver.SMALL)
    sample = measurement.sample.model_copy(
        update={
            "population_count": 100,
            "tasks": [measurement.sample.tasks[0].model_copy(update={"task_id": str(i)}) for i in range(100)],
        }
    )
    measurement = measurement.model_copy(update={"sample": sample})
    result = measurement_from_attempts(
        measurement, [DifficultyAttempt(task_id=str(i), status="scored", reward=reward) for i in range(100)]
    )
    assert (result.status, result.n, result.solved, result.pending_tasks) == ("measured", 100, int(reward) * 100, 0)
    if reward:
        assert result.wilson_95.high == 1.0
        assert result.wilson_95.low == pytest.approx(0.9630065017930143)
    else:
        assert result.wilson_95.low == 0.0
        assert result.wilson_95.high == pytest.approx(0.03699349820698568)


def test_historical_protocol_retains_its_original_sampling_contract():
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    payload = difficulty_measurement(declaration, Solver.SMALL).model_dump(mode="json")
    payload["settings"]["temperature"] = 0.6
    with pytest.raises(ValueError, match="temperature differs"):
        DifficultyMeasurement.model_validate(payload)
    # The recorded legacy v2 contract reads: same pinned sample, model and sampling, so
    # historical measurements stay readable without reopening fail-open validation
    # (ruling 2026-11-19).
    payload["settings"]["temperature"] = 0.7
    payload["protocol_id"] = "atlas-difficulty-v2-65k16k"
    assert DifficultyMeasurement.model_validate(payload).settings.temperature == 0.7
    # Unknown protocols still fail closed.
    payload["protocol_id"] = "atlas-difficulty-v9-unknown"
    with pytest.raises(ValueError, match="Unknown difficulty protocol"):
        DifficultyMeasurement.model_validate(payload)


def test_frozen_backend_survey_digest_survives_canonical_source_projection(tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen", source_id="HuggingFaceH4:aime24", origin=SourceOrigin.HUGGINGFACEH4)
    raw = catalog_backend_map(CurationCatalog(root)).model_dump(mode="json")
    raw["sources"][0]["source_id"] = "MarinSkyRL:aime24"
    survey = canonical_backend_survey(raw)
    assert survey.sources[0].source_id == "HuggingFaceH4:aime24"
    assert canonical_sha256(backend_survey_payload(survey)) == canonical_sha256(raw)


@pytest.mark.parametrize("stale_input", [False, True])
def test_preview_packages_producer_oracle_inventory_and_rejects_mixed_pins(tmp_path, stale_input):
    binding = actor_census_input("openai/gsm8k", "a" * 40, "1", 3)
    inventory = SampleInventory(
        source_id="MarinSkyRL:gsm8k",
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        converter_version=binding.converter_version,
        input_artifact="s3://fixture/population",
        task_specs_sha256=binding.task_specs_sha256,
        scanned_task_specs_sha256="f" * 64,
        retained_tasks=3,
        sample_seed=0,
        sample_size=3,
        panel_path="s3://fixture/panel.parquet",
        panel_sha256="b" * 64,
        evidence_url="s3://fixture/inventory.json",
        oracle_tasks=0,
        execution_qualification="not_assessed",
    )
    actor = binding.model_dump(mode="json")
    if stale_input:
        actor["dataset_revision"] = "b" * 40
    ledger = {
        "schema_version": 2,
        "kind": "full_population_oracle_inventory",
        "sources": 1,
        "retained_tasks": 3,
        "rows": [
            {
                "source_id": "OpenAI:gsm8k",
                "current_input": actor,
                "inventory": inventory.model_dump(mode="json"),
                "inventory_manifest_uri": "s3://fixture/inventory-manifest.json",
                "inventory_manifest_sha256": "c" * 64,
            }
        ],
    }
    source = tmp_path / "oracle-ledger.json"
    source.write_text(json.dumps(ledger))
    checkout = Path(__file__).resolve().parents[3]
    output = tmp_path / "preview"
    if stale_input:
        with pytest.raises(ValueError, match="Oracle inventory differs from its actor input"):
            prepare_preview(checkout, output, source)
        assert not output.exists()
        return
    prepare_preview(checkout, output, source)
    observations = json.loads((output / "dist/measurements/oracle-inventory.json").read_text())
    assert observations["sources"][0]["source_id"] == "OpenAI:gsm8k"
    assert observations["sources"][0]["input"] == binding.model_dump(mode="json")
    assert observations["sources"][0]["oracle_tasks"] == 0
    assert observations["sources"][0]["execution_qualification"] == "not_assessed"
    provenance = json.loads((output / "dist/bundled-model-provenance.json").read_text())["oracle_inventory"]
    assert provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        provenance["packaged_sha256"]
        == hashlib.sha256((output / "dist/measurements/oracle-inventory.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("actorless_exempt", [False, True])
def test_preview_bundles_current_count_evidence_into_the_served_api(
    catalog_database, catalog_query, tmp_path, actorless_exempt
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "counted", source_id="Marin:counted", duplicates=1)
    load_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.connect() as connection:
        raw = connection.execute(text("SELECT declaration FROM atlas_sources")).scalar_one()
    declaration = SourceDeclaration.model_validate(raw)
    observation = count_evidence(declaration, 0, 0 if actorless_exempt else 1).model_dump(mode="json")
    if actorless_exempt:
        observation["actorless_policy"] = {
            "uri": "https://example.org/actorless.md",
            "sha256": "b02c3b3565e4baee24092fd6faa5c24da54c8712576f6a16e2b979e9e22da118",
        }
    count_file = tmp_path / "counts.json"
    count_file.write_text(json.dumps({"version": 1, "sources": [observation]}))
    output = tmp_path / "preview"
    prepare_preview(Path(__file__).resolve().parents[3], output, source_counts=count_file)
    # the sealed preview ships prebuilt assets and omits build_command: the
    # release's clean_frontend_build would otherwise rename dist away and the
    # command would regenerate only fragments over the packaged evidence
    staged_manifest = tomllib.loads((output / "applet.toml").read_text())
    assert "build_command" not in staged_manifest
    assert (output / "dist/index.html").exists()
    extras = json.loads((output / "dist/measurements/source-status-extras.json").read_text())
    assert extras["stages"], "the source build must have generated the status extras"
    provenance = json.loads((output / "dist/bundled-model-provenance.json").read_text())["build_command"]
    assert provenance["prebuilt"] is True and "omits build_command" in provenance["transformation"]
    package = read_applet_package(package_applet(output))
    # a real release dry-run must accept the staged package: no build_command
    # means clean_frontend_build never runs, so the prebuilt dist ships verbatim
    dry_run = subprocess.run(
        ["uv", "run", "--no-sync", "marina", "release", str(output), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr[-2000:] or dry_run.stdout[-2000:]
    applet_id = uuid.uuid4()
    try:
        module, factory_name = load_backend_entrypoint(applet_id, 1, package)
        api = getattr(module, factory_name)(cast(AppletServices, CatalogServices(catalog_database)))
        with TestClient(api) as client:
            source = client.get("/sources").json()["sources"][0]
            assert source["filtration"]["example_count"] == (2 if actorless_exempt else 1)
            assert source["filtration"]["admitted_count"] == (2 if actorless_exempt else 1)
            detail = client.get("/source", params={"source_id": declaration.id}).json()
            assert detail["filtration"] == source["filtration"]
            assert detail["filtration"]["count_evidence"]["policy"]["sha256"].startswith("028c7dfb")
            if actorless_exempt:
                assert detail["filtration"]["count_evidence"]["actorless_policy"] == observation["actorless_policy"]
    finally:
        remove_backend_revision(applet_id, 1, package.digest)


@pytest.mark.parametrize("preview", [False, True])
def test_atlas_packages_resolve_aliases_without_repository_imports(tmp_path, monkeypatch, preview):
    checkout = Path(__file__).resolve().parents[3]
    output = checkout / "infra/marina/apps/rl_data_catalog"
    if preview:
        output = tmp_path / "preview"
        prepare_preview(checkout, output)
    package = read_applet_package(package_applet(output))
    assert package.manifest.applet_id == (PREVIEW_APP_ID if preview else PRODUCTION_APP_ID)
    original_import = builtins.__import__

    def old_runtime_import(name, global_namespace=None, local_namespace=None, fromlist=(), level=0):
        if level == 0 and (
            name.startswith(("experiments.", "infra.marina.apps."))
            or (
                preview
                # Marina's runtime lags these modules and installs no shellbox (R40 #52:
                # the v40 publish failed backend validation on shellbox.requirements).
                and (
                    name in {"finestore.rl.curation", "finestore.rl.difficulty", "taskcompendium.source_fingerprints"}
                    or name == "shellbox"
                    or name.startswith("shellbox.")
                )
            )
        ):
            raise ModuleNotFoundError(name)
        return original_import(name, global_namespace, local_namespace, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", old_runtime_import)
    applet_id = uuid.uuid4()
    try:
        module, _ = load_backend_entrypoint(applet_id, 1, package)
        backend = cast(ModuleType, module)
        record = backend.canonical_source_record({"id": "MarinSkyRL:gsm8k", "declaration": {"id": "MarinSkyRL:gsm8k"}})
        assert record["id"] == record["declaration"]["id"] == "OpenAI:gsm8k"
        quality_module = importlib.import_module(backend.__name__.rsplit(".", 1)[0] + ".quality")
        assert (
            quality_module.result(
                {
                    "quality_result": {
                        "reviewed": 10,
                        "issue_count": 1,
                        "issues": [],
                        "repair_recommendation": "Review evidence",
                    }
                },
                None,
            )
            == "GOOD"
        )
    finally:
        remove_backend_revision(applet_id, 1, package.digest)


def test_role_projection_requires_receipt_and_preserves_frozen_sample_roles(catalog_database, catalog_query, tmp_path):
    path = curriculum_release(tmp_path, "insufficient_context")
    raw_release = json.loads(path.read_text())
    raw_release["release_id"] = canonical_sha256({key: raw_release[key] for key in ("lineage", "sources", "inventory")})
    raw_coverage = json.loads((tmp_path / "coverage.json").read_text())
    raw_coverage["alignment_release_id"] = raw_release["release_id"]
    identity_keys = (
        "alignment_release_id",
        "subject_distance",
        "capability_distance",
        "source_sample_size",
        "area_sample_size",
    )
    raw_coverage["coverage_id"] = canonical_sha256(
        {**{key: raw_coverage[key] for key in identity_keys}, "version": raw_coverage["version"]}
    )
    corrected_release = {
        **raw_release,
        "sources": [
            {**row, "intended_use": "eval"} if row["source_id"] == "Marin:a" else row for row in raw_release["sources"]
        ],
    }
    corrected_release["release_id"] = canonical_sha256(
        {key: corrected_release[key] for key in ("lineage", "sources", "inventory")}
    )
    corrected_coverage = {**raw_coverage, "alignment_release_id": corrected_release["release_id"]}
    corrected_coverage["coverage_id"] = canonical_sha256(
        {**{key: corrected_coverage[key] for key in identity_keys}, "version": corrected_coverage["version"]}
    )
    documents = {}
    for name, release, coverage in (
        ("raw", raw_release, raw_coverage),
        ("corrected", corrected_release, corrected_coverage),
    ):
        release_path, coverage_path = tmp_path / f"{name}-release.json", tmp_path / f"{name}-coverage.json"
        release_path.write_text(json.dumps(release))
        coverage_path.write_text(json.dumps(coverage))
        documents[name] = {
            "release_uri": str(release_path),
            "release_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
            "release_id": release["release_id"],
            "coverage_uri": str(coverage_path),
            "coverage_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
            "coverage_id": coverage["coverage_id"],
        }
    source = next(row for row in raw_release["sources"] if row["source_id"] == "Marin:a")
    vector_manifest = tmp_path / "vectors.json"
    vector_manifest.write_text(json.dumps({"cache_hits": 4, "cache_misses": 0}))
    note = {
        "schema_version": 1,
        "kind": "source-use-correction",
        "policy_ruling": {"session_id": "fixture", "message_id": "approval", "timestamp": "2026-11-13T00:00:00Z"},
        **documents,
        "changes": [
            {
                **{key: source[key] for key in ("source_id", "source_key", "population", "task_count")},
                "dataset_id": FIXTURE_DATASET,
                "dataset_revision": FIXTURE_REVISION,
                "recorded_use": "train",
                "effective_use": "eval",
            }
        ],
        "vector_cache": {
            "manifest_uri": str(vector_manifest),
            "manifest_sha256": hashlib.sha256(vector_manifest.read_bytes()).hexdigest(),
            "cache_hits": 4,
            "cache_misses": 0,
        },
    }
    note_path = tmp_path / "role-projection.json"
    note_path.write_text(json.dumps(note))
    reference = ManifestReference(uri=str(note_path), sha256=hashlib.sha256(note_path.read_bytes()).hexdigest())
    load_curriculum(documents["raw"]["release_uri"], documents["raw"]["coverage_uri"], catalog_query)
    root = str(tmp_path / "catalog")
    publish_sample(root, "graded", source_id="Marin:a")
    load_catalog(CurationCatalog(root), catalog_query)
    args = (documents["corrected"]["release_uri"], documents["corrected"]["coverage_uri"], catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        with pytest.raises(ValueError, match="lineage"):
            load_curriculum(*args)
        with pytest.raises(ValueError, match="current catalog policy"):
            load_curriculum(*args, role_projection=reference)
        assert client.get("/curriculum").json()["release_id"] == raw_coverage["coverage_id"]
        with catalog_database.begin() as connection:
            connection.execute(
                text("UPDATE atlas_sources SET declaration=jsonb_set(declaration, '{intended_use}', '\"eval\"')")
            )
        changed_coverage = {**corrected_coverage, "sample_paths": ["unapproved-samples.parquet"]}
        changed_path = Path(documents["corrected"]["coverage_uri"])
        changed_path.write_text(json.dumps(changed_coverage))
        changed_note = {
            **note,
            "corrected": {**note["corrected"], "coverage_sha256": hashlib.sha256(changed_path.read_bytes()).hexdigest()},
        }
        changed_note_path = tmp_path / "changed-note.json"
        changed_note_path.write_text(json.dumps(changed_note))
        changed_reference = ManifestReference(
            uri=str(changed_note_path), sha256=hashlib.sha256(changed_note_path.read_bytes()).hexdigest()
        )
        with pytest.raises(ValueError, match="changes frozen data"):
            load_curriculum(*args, role_projection=changed_reference)
        assert client.get("/curriculum").json()["release_id"] == raw_coverage["coverage_id"]
        changed_path.write_text(json.dumps(corrected_coverage))
        load_curriculum(*args, role_projection=reference)
        assert client.get("/curriculum", params={"intended_use": "eval"}).json()["totals"]["denominator"] == 300
        assert client.get("/curriculum", params={"intended_use": "train"}).json()["totals"]["denominator"] == 100
        detail = client.get("/curriculum/task", params={"source_id": "Marin:a", "task_id": "one"}).json()["task"]
        assert detail["intended_use"] == "eval" and detail["recorded_intended_use"] == "train"
        with catalog_database.connect() as connection:
            assert connection.execute(
                text("SELECT DISTINCT detail->>'intended_use' FROM curriculum_tasks")
            ).scalars().all() == ["train"]
        for ref in documents.values():
            assert hashlib.sha256(Path(ref["release_uri"]).read_bytes()).hexdigest() == ref["release_sha256"]
            assert hashlib.sha256(Path(ref["coverage_uri"]).read_bytes()).hexdigest() == ref["coverage_sha256"]


def test_frozen_catalog_projects_measured_math_folds_and_directional_notes(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen")
    load_catalog(CurationCatalog(root), catalog_query)
    fixture = Path(__file__).resolve().parents[3] / (
        "experiments/post_training/task_curation/tests/fixtures/math_overlap_reviews.json"
    )
    evidence = json.loads(fixture.read_text())
    stored = {
        "group_id": "old-partial-chain",
        "canonical_source_id": "AI-MO:numina_math",
        "members": [{"source_id": row["id"], "overlap_with_canonical": None} for row in evidence["sources"]],
    }
    with catalog_database.begin() as connection:
        template = dict(connection.execute(text("SELECT * FROM atlas_sources")).mappings().one())
        for source in evidence["sources"]:
            declaration = {
                **template["declaration"],
                "id": source["id"],
                "within_source_dedup": {
                    **template["declaration"]["within_source_dedup"],
                    "unique_count": source["task_count"],
                },
            }
            connection.execute(
                text(
                    "INSERT INTO atlas_sources SELECT (jsonb_populate_record(NULL::atlas_sources, "
                    "to_jsonb(s) || CAST(:patch AS JSONB))).* FROM atlas_sources s WHERE id = :template"
                ),
                {
                    "template": template["id"],
                    "patch": json.dumps(
                        {
                            "id": source["id"],
                            "declaration": declaration,
                            "method_variants": None,
                            "duplicate_group": stored,
                        }
                    ),
                },
            )
        connection.execute(text("DELETE FROM atlas_sources WHERE id = :id"), {"id": template["id"]})
        for review in evidence["reviews"]:
            connection.execute(
                text("INSERT INTO atlas_duplicate_reviews VALUES (:load_id, :left, :right, CAST(:review AS JSONB))"),
                {
                    "load_id": template["load_id"],
                    "left": review["similarity"]["left"],
                    "right": review["similarity"]["right"],
                    "review": json.dumps(review),
                },
            )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        rows = {row["id"]: row for row in client.get("/sources").json()["sources"]}
        assert rows["AI-MO:numina_math"]["duplicate_group"] is None
        assert rows["BytedTsinghua-SIA:dapo_math"]["duplicate_group"] is None
        assert rows["EleutherAI:hendrycks_math"]["duplicate_group"]["canonical_source_id"] == "agentica-org:deepscaler"
        notes = rows["agentica-org:deepscaler"]["partial_overlaps"]
        assert next(row for row in notes if row["other_id"] == "AI-MO:numina_math") == {
            "other_id": "AI-MO:numina_math",
            "source_fraction": 0.7294921875,
            "other_fraction": 0.042236328125,
        }
        detail = client.get("/source", params={"source_id": "agentica-org:deepscaler"}).json()
        assert detail["partial_overlaps"] == notes
        assert detail["duplicate_group"] == rows["agentica-org:deepscaler"]["duplicate_group"]
        duplicates = client.get("/source-duplicates").json()
        stale = {"sources": [{**row, "duplicate_group": stored, "partial_overlaps": []} for row in rows.values()]}
        projected = projected_duplicate_inventory(stale, catalog_query, template["load_id"])
        assert projected["sources"] == list(rows.values())
        assert stored_duplicate_groups(catalog_query, template["load_id"]) == duplicates["stored_groups"]
        assert duplicates["stored_groups"] == [stored | {"advisory": True}]
        assert {g["group_id"] for g in duplicates["groups"]} == {
            row["duplicate_group"]["group_id"] for row in rows.values() if row["duplicate_group"] is not None
        }
    with catalog_database.connect() as connection:
        assert connection.execute(text("SELECT DISTINCT duplicate_group FROM atlas_sources")).scalar_one() == stored


@pytest.mark.parametrize("corruption", [None, "aggregate", "population", "parent"])
def test_confidence_export_reconciles_real_source_vectors_at_every_level(tmp_path: Path, corruption: str | None) -> None:
    path = curriculum_release(tmp_path)
    release = AlignmentRelease.model_validate_json(path.read_text())
    coverage = CurriculumCoverage.model_validate_json((tmp_path / "coverage.json").read_text())
    levels = [
        {"label": "Broad", "subject_min_similarity": 0.55, "capability_min_similarity": 0.6},
        {"label": "Original", "subject_min_similarity": 0.55, "capability_min_similarity": 0.75},
    ]
    coverage = CurriculumCoverage.model_validate({**coverage.model_dump(), "confidence_levels": levels})
    source_rows = []
    combined = {}
    for source in release.sources:
        statistics = []
        for row in pq.read_table(coverage.paths[0]).to_pylist():
            if row["source_id"] != source.source_id:
                continue
            strict = row["instances"]
            broad = strict
            if source.source_id == "Marin:a":
                if row["metric"] in ("assigned", "capability") or row["node_id"] in ("arithmetic", "add"):
                    broad = 200
                elif row["metric"] == "subject_only":
                    broad = 0
            key = (row["metric"], row["subject_id"], row["node_id"])
            vector = [broad, strict]
            statistics.append({"metric": key[0], "subject_id": key[1], "node_id": key[2], "instances": vector})
            combined[key] = [a + b for a, b in zip(combined.get(key, [0, 0]), vector, strict=True)]
        source_rows.append({"source_id": source.source_id, "intended_use": "train", "statistics": statistics})
    aggregate = [
        {"metric": key[0], "subject_id": key[1], "node_id": key[2], "instances": vector}
        for key, vector in combined.items()
    ]
    document = {
        "schema_version": 1,
        "release": {"release_id": release.release_id},
        "coverage": {"coverage_id": coverage.coverage_id},
        "levels": levels,
        "sources": source_rows,
        "aggregates": {"all": aggregate, "train": aggregate, "eval": []},
        "samples": {"paths": coverage.sample_paths, "task_count": coverage.sample_task_count},
    }
    if corruption == "aggregate":
        aggregate[0]["instances"][0] += 1
    elif corruption == "population":
        source_rows[0]["statistics"][0]["instances"][0] += 1
    elif corruption == "parent":
        next(row for row in source_rows[0]["statistics"] if row["node_id"] == "add")["instances"][0] = 301
    if corruption:
        with pytest.raises(ValueError):
            list(confidence_rows(document, release, coverage))
    else:
        result = list(confidence_rows(document, release, coverage))
        totals = {row["metric"]: row["instances"] for row in result if row["scope_id"] == "@all" and not row["node_id"]}
        assert totals["denominator"] == [400, 400]
        assert totals["assigned"] == [300, 200]
        assert totals["subject_only"] == [0, 100]


def test_overlap_refresh_binds_frozen_inputs_and_preserves_legacy_reviews(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen")
    load_catalog(CurationCatalog(root), catalog_query)
    fixture = (
        Path(__file__).resolve().parents[3]
        / "experiments/post_training/task_curation/tests/fixtures/math_overlap_reviews.json"
    )
    evidence = json.loads(fixture.read_text())
    fingerprints = {}
    with catalog_database.begin() as connection:
        template = dict(connection.execute(text("SELECT * FROM atlas_sources")).mappings().one())
        for index, source in enumerate(evidence["sources"]):
            fingerprint = SourceFingerprint(
                version=1,
                seed=0,
                task_count=source["task_count"],
                within_source_duplicates=0,
                sample_sha256="a" * 64,
                sample=[],
                instruction_minhash=[],
                instruction_hashes=[],
                reference_hashes=[],
                test_hashes=[],
                task_id_hashes=[],
                url_hashes=[],
                lineage=[],
                inputs={f"fixture-{index}": "b" * 64},
            )
            fingerprints[source["id"]] = fingerprint.model_dump(mode="json")
            declaration = {
                **template["declaration"],
                "id": source["id"],
                "within_source_dedup": {
                    **template["declaration"]["within_source_dedup"],
                    "unique_count": source["task_count"],
                },
            }
            connection.execute(
                text(
                    "INSERT INTO atlas_sources SELECT (jsonb_populate_record(NULL::atlas_sources,to"
                    "_jsonb(s) || CAST(:patch AS JSONB))).* FROM atlas_sources s WHERE id=:template"
                ),
                {
                    "template": template["id"],
                    "patch": json.dumps(
                        {
                            "id": source["id"],
                            "declaration": declaration,
                            "method_variants": None,
                            "source_fingerprint": fingerprint.model_dump(mode="json"),
                        }
                    ),
                },
            )
        connection.execute(text("DELETE FROM atlas_sources WHERE id=:source"), {"source": template["id"]})
    # An empty Parquet population still has a pinned manifest path and digest.
    empty_fingerprint = SourceFingerprint(
        version=1,
        seed=0,
        task_count=0,
        within_source_duplicates=0,
        sample_sha256="a" * 64,
        sample=[],
        instruction_minhash=[],
        instruction_hashes=[],
        reference_hashes=[],
        test_hashes=[],
        task_id_hashes=[],
        url_hashes=[],
        lineage=[],
        inputs={"empty.parquet": "c" * 64},
    )
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO atlas_sources SELECT (jsonb_populate_record(NULL::atlas_sources, "
                "to_jsonb(s) || CAST(:patch AS JSONB))).* FROM atlas_sources s LIMIT 1"
            ),
            {
                "patch": json.dumps(
                    {
                        "id": "Marin:empty",
                        "method_variants": None,
                        "declaration": {
                            **template["declaration"],
                            "id": "Marin:empty",
                            "within_source_dedup": {**template["declaration"]["within_source_dedup"], "unique_count": 0},
                        },
                        "source_fingerprint": empty_fingerprint.model_dump(mode="json"),
                    }
                )
            },
        )
    fingerprints["Marin:empty"] = None
    survey = {
        "version": 1,
        "catalog_root": "fixture",
        "snapshot_id": "archived",
        "generated_at": "2026-11-14T00:00:00Z",
        "fingerprints": fingerprints,
        "reviews": evidence["reviews"],
        "groups": [],
        "judge": None,
    }
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(survey))
    reference = ManifestReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with catalog_database.begin() as connection:
        connection.execute(
            text(
                "UPDATE atlas_sources SET source_fingerprint=jsonb_set(source_fingerprint,'{inp"
                "uts}','{}'::jsonb) WHERE id=:source"
            ),
            {"source": evidence["sources"][0]["id"]},
        )
    with pytest.raises(ValueError, match="differs from the frozen catalog"):
        load_identity(reference, catalog_query)
    assert catalog_query("SELECT * FROM atlas_identity_measurement_current", {})["rows"] == []
    with catalog_database.begin() as connection:
        connection.execute(
            text("UPDATE atlas_sources SET source_fingerprint=CAST(:fingerprint AS JSONB) WHERE id=:source"),
            {
                "fingerprint": json.dumps(fingerprints[evidence["sources"][0]["id"]]),
                "source": evidence["sources"][0]["id"],
            },
        )
    before = catalog_state(catalog_query)
    first = load_identity(reference, catalog_query)
    assert first["excluded_empty_sources"] == ["Marin:empty"]
    assert load_identity(reference, catalog_query) == first
    after = catalog_state(catalog_query)
    assert before["archives"] == after["archives"]
    assert before["packets"] == after["packets"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        rows = {row["id"]: row for row in client.get("/sources").json()["sources"]}
        assert rows["AI-MO:numina_math"]["duplicate_group"] is None
        assert rows["EleutherAI:hendrycks_math"]["duplicate_group"]["canonical_source_id"] == "agentica-org:deepscaler"
        assert client.get("/source-duplicates").json()["survey_id"] == reference.sha256


def test_sample_list_carries_filter_scores_while_detail_retains_alternatives(catalog_database, catalog_query, tmp_path):
    path = curriculum_release(tmp_path)
    samples = tmp_path / "tasks.parquet"
    rows = pq.read_table(samples).to_pylist()
    alternative = {"subject_id": "D01", "capability_id": "add", "similarity": 0.4, "anchor_id": "alternative"}
    rows[0]["matches"]["subjects"][0]["candidates"].append(alternative)
    pq.write_table(pa.Table.from_pylist(rows), samples)
    load_curriculum(str(path), str(tmp_path / "coverage.json"), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        listed = client.get("/curriculum/tasks").json()["tasks"]
        first = next(row for row in listed if row["source_id"] == "Marin:a" and row["task_id"] == "one")
        assert first["matches"][0]["candidates"] == [{"capability_id": "add", "similarity": 0.9}]
        detail = client.get("/curriculum/task", params={"source_id": "Marin:a", "task_id": "one"}).json()["task"]
        assert detail["matches"]["subjects"][0]["candidates"][1] == alternative


def test_typed_review_fields_load_and_serve(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "typed")
    types = {
        "one": {
            "confidence": "low",
            "reference_status": "conflict",
            "defects": ["wrong_reference"],
            "provenance": "replaced",
            "replaced_task_sha256": "a" * 64,
            "repair_advisories": [
                {"signal": "pass@8-filter", "candidate_source_id": "Marin:candidate", "reason": "No pass at 8"}
            ],
        },
        "two": {"confidence": "high", "reference_status": "consistent", "defects": ["ambiguity", "malformed"]},
        "three": {"confidence": "medium", "reference_status": "consistent", "defects": ["answer_leakage"]},
    }
    publish_sample(root, "typed", admitted_verdicts=("some_issues", "bad"), review_types=types)
    load_surveyed_catalog(
        CurationCatalog(root),
        catalog_query,
        catalog_backend_map(CurationCatalog(root)),
        catalog_identity_map(CurationCatalog(root)),
    )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        report = client.get("/assessment", params={"source_id": "Marin:math"}).json()["report"]
        samples = {row["task_id"]: row for row in report["evidence"]["items"]}
        assert samples["one"]["confidence"] == "low"
        assert samples["one"]["reference_status"] == "conflict"
        assert samples["one"]["defects"] == ["wrong_reference"]
        assert samples["one"]["provenance"] == "replaced"
        assert samples["one"]["replaced_task_sha256"] == "a" * 64
        assert samples["one"]["repair_advisories"] == [
            {
                "signal": "pass@8-filter",
                "candidate_source_id": "Marin:candidate",
                "reason": "No pass at 8",
                "receipt_uri": None,
            }
        ]
        assert samples["two"]["provenance"] is None
        assert samples["two"]["repair_advisories"] == []
        admitted = report["assessment"]["admitted"]
        assert admitted["defect_counts"]["ambiguity"] == 1
        assert admitted["defect_counts"]["malformed"] == 1
        assert admitted["confidence_counts"] == {"high": 1, "medium": 0, "low": 1, "unrecorded": 0}
        evidence = client.get("/evidence", params={"source_id": "Marin:math", "task_id": "two"}).json()
        assert evidence["judgment"]["defects"] == ["ambiguity", "malformed"]
        assert evidence["judgment"]["confidence"] == "high"
        replaced = client.get("/evidence", params={"source_id": "Marin:math", "task_id": "one"}).json()
        assert replaced["judgment"]["provenance"] == "replaced"
        assert replaced["judgment"]["replaced_task_sha256"] == "a" * 64
    # A legacy snapshot without the columns keeps validating as unrecorded.
    legacy = str(tmp_path / "legacy")
    publish_sample(legacy, "legacy")
    load_surveyed_catalog(
        CurationCatalog(legacy),
        catalog_query,
        catalog_backend_map(CurationCatalog(legacy)),
        catalog_identity_map(CurationCatalog(legacy)),
    )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        report = client.get("/assessment", params={"source_id": "Marin:math"}).json()["report"]
        samples = {row["task_id"]: row for row in report["evidence"]["items"]}
        assert samples["one"]["confidence"] is None
        assert samples["one"]["defects"] == []


FULL_QC_SOURCE = "Marin:fullqc"
FULL_QC_POPULATION = 200


def full_qc_review_records(reviews: list[ReviewRecord], path: Path) -> tuple[str, str]:
    """Write one consolidated FULL review records.parquet fixture."""
    schema = pa.schema(
        [
            ("task_id", pa.string()),
            ("task_sha256", pa.string()),
            ("runtime_sha256", pa.string()),
            ("review", pa.large_string()),
            ("batch_receipt", pa.string()),
        ]
    )
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        writer.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "task_id": record.task_id,
                        "task_sha256": "b" * 64,
                        "runtime_sha256": "c" * 64,
                        "review": record.model_dump_json(),
                        "batch_receipt": "s3://fixture/receipt.json",
                    }
                    for record in reviews
                ],
                schema=schema,
            )
        )
    return str(path), file_sha256(path)


def full_qc_verdict(
    task_id: str, quality: str, confidence: str, reference: str, defects: list[str], evidence: str = "Recorded finding."
) -> ReviewVerdict:
    return ReviewVerdict(
        task_id=task_id,
        quality=Quality(quality),
        confidence=Confidence(confidence),
        reference_status=ReferenceStatus(reference),
        defects=[Defect(defect) for defect in defects],
        evidence=evidence,
    )


def full_qc_review_records_fixture() -> list[ReviewRecord]:
    """A whole-population review panel: every Defect value, all confidences, one unavailable."""
    records = [
        ReviewRecord(
            task_id="t01",
            status="reviewed",
            verdict=full_qc_verdict("t01", "some_issues", "high", "consistent", ["missing_context"]),
            detail="",
        ),
        ReviewRecord(
            task_id="t02",
            status="reviewed",
            verdict=full_qc_verdict("t02", "some_issues", "high", "conflict", ["wrong_reference"]),
            detail="",
        ),
        ReviewRecord(
            task_id="t03",
            status="reviewed",
            verdict=full_qc_verdict("t03", "bad", "medium", "conflict", ["ambiguity", "rubric_mismatch"]),
            detail="",
        ),
        ReviewRecord(
            task_id="t04",
            status="reviewed",
            verdict=full_qc_verdict("t04", "some_issues", "high", "consistent", ["answer_leakage"]),
            detail="",
        ),
        ReviewRecord(
            task_id="t05",
            status="reviewed",
            verdict=full_qc_verdict("t05", "some_issues", "high", "consistent", ["malformed"]),
            detail="",
        ),
        ReviewRecord(
            task_id="t06",
            status="reviewed",
            verdict=full_qc_verdict("t06", "good", "high", "consistent", []),
            detail="",
        ),
        # Low confidence is uncertain and stays outside the conclusive denominator.
        ReviewRecord(
            task_id="t07",
            status="reviewed",
            verdict=full_qc_verdict("t07", "some_issues", "low", "consistent", ["ambiguity"]),
            detail="",
        ),
        ReviewRecord(task_id="t08", status="unavailable", verdict=None, detail="Reviewer response truncated."),
    ]
    cycles = [
        ["missing_context"],
        ["wrong_reference"],
        ["answer_leakage"],
        ["malformed"],
        ["rubric_mismatch"],
        ["ambiguity"],
    ]
    for index in range(len(records), FULL_QC_POPULATION):
        task_id = f"t{index + 1:02d}"
        if index % 4 == 0:
            verdict = full_qc_verdict(task_id, "good", "high", "consistent", [])
        else:
            verdict = full_qc_verdict(task_id, "some_issues", "high", "consistent", cycles[index % len(cycles)])
        records.append(ReviewRecord(task_id=task_id, status="reviewed", verdict=verdict, detail=""))
    return records


def publish_full_bound_source(
    root: str,
    snapshot_id: str,
    *,
    result: SourceQualityResult,
    waiver: ContentEvidenceReference,
    population: int = FULL_QC_POPULATION,
    source_id: str = FULL_QC_SOURCE,
) -> None:
    """A catalog source whose whole-population review is bound outside the campaign."""
    declaration = SourceDeclaration(
        license_receipt=fixture_license(source_id, "fixture", "1"),
        id=source_id,
        display_name=source_display_name(source_id),
        origin=SourceOrigin.MARIN,
        build_path="test",
        upstream_dataset="fullqc",
        upstream_name="Full QC",
        current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture tasks contain no actor environment.",
        grading_environment=grading_evidence(population),
        family="Marin:fullqc",
        tags=[],
        task_count=population,
        within_source_dedup=deduplication_evidence("fixture", "1", "1", population),
        notes="",
        oversight=OversightLevel.FULL,
        dataset_id="fixture",
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        intended_use="train",
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        actor_census_input=actor_census_input("fixture", "1", "1", population),
        consumer_receipt=consumer_receipt(source_id, actor_census_input("fixture", "1", "1", population)),
        quality_result=result,
        quality_result_finalization="accepted_incomplete",
        quality_result_waiver=waiver.model_dump(mode="json"),
        independent_qc={
            "verdict": result.verdict.value,
            "method": "Whole-population review bound with the finalization waiver",
            "reviewed": result.reviewed + result.uncertain_count,
            "evidence_url": "https://example.org/review",
        },
    )
    with DataStore.open(root, writer_id=snapshot_id) as store:
        register_catalog(store)
        source = SnapshotSource(
            snapshot_id=snapshot_id,
            id=source_id,
            declaration=declaration,
            assessment_key=None,
            latest_attempt_key=None,
            latest_attempt_status="skipped",
            reason=None,
            curation=None,
        )
        with store.transaction() as tx:
            tx.table(INVENTORY).add(source.model_dump(mode="json"))
            tx.table(SNAPSHOTS).add(
                CatalogSnapshot(snapshot_id=snapshot_id, created_at="2026-11-16T00:00:00Z", source_count=1).model_dump()
            )
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())


def full_qc_binding(tmp_path: Path, result: SourceQualityResult, records: list[ReviewRecord]) -> dict[str, Any]:
    records_uri, records_sha = full_qc_review_records(records, tmp_path / "records.parquet")
    receipt_path = tmp_path / "receipt.artifact.json"
    receipt_path.write_text(
        json.dumps(
            {
                "result_type": "experiments.post_training.task_curation.full_qc_artifacts.FullQCReview",
                "result": {
                    "source_id": FULL_QC_SOURCE,
                    "reviewer": {
                        "model": "glm",
                        "model_revision": "r1",
                        "max_tokens": 8192,
                        "max_attempts": 3,
                        "retry_max_tokens": 16384,
                    },
                    "rubric": {"id": "atlas-retained-full-qc", "version": "2026.11.13.1", "criteria": ["Review tasks."]},
                    "coverage_complete": result.unavailable_count == 0,
                    "qualification_status": "pending",
                    "attempted_count": len(records),
                    "result": result.model_dump(mode="json"),
                },
            }
        )
    )
    waiver_path = tmp_path / "waiver.json"
    waiver_path.write_text(
        json.dumps({"source_id": FULL_QC_SOURCE, "qc_unreviewed": [{"task_id": "t08", "reason": "token budget"}]})
    )
    waiver = ContentEvidenceReference(uri=str(waiver_path), sha256=file_sha256(waiver_path))
    return {
        FULL_QC_SOURCE: {
            "source_id": FULL_QC_SOURCE,
            "input": actor_census_input("fixture", "1", "1", FULL_QC_POPULATION).model_dump(mode="json"),
            "contract_index": {"uri": "s3://fixture/contracts.parquet", "sha256": "0" * 64},
            "review_receipt": {"uri": str(receipt_path), "sha256": file_sha256(receipt_path)},
            "records": {"uri": records_uri, "sha256": records_sha},
            "result": result.model_dump(mode="json"),
            "coverage_complete": result.unavailable_count == 0,
            "finalization": "accepted_incomplete",
            "waiver": waiver.model_dump(mode="json"),
        }
    }


def test_full_review_rows_load_typed_and_browseable(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "fullqc")
    records = full_qc_review_records_fixture()
    result = recompute_review_result(records)
    assert result.unavailable_count == 1 and 0 < result.unavailable_count / len(records) <= 0.02
    assert len(records) == FULL_QC_POPULATION == 200
    expected_defects = Counter(
        defect.value for record in records if record.verdict is not None for defect in record.verdict.defects
    )
    assert set(expected_defects) == {defect.value for defect in Defect}
    bindings = full_qc_binding(tmp_path, result, records)
    publish_full_bound_source(
        root,
        "fullqc",
        result=result,
        waiver=ContentEvidenceReference.model_validate(bindings[FULL_QC_SOURCE]["waiver"]),
    )
    seed_previous_available(catalog_database, [FULL_QC_SOURCE, "Marin:math"])
    load_result = load_surveyed_catalog(
        CurationCatalog(root),
        catalog_query,
        catalog_backend_map(CurationCatalog(root)),
        catalog_identity_map(CurationCatalog(root)),
        full_reviews={key: FullContentResultBinding.model_validate(value) for key, value in bindings.items()},
    )
    assert load_result["full_reviews"] == {
        FULL_QC_SOURCE: {"reviewed": result.reviewed, "unavailable": result.unavailable_count}
    }
    # the cohort summary honors the review policy for FULL rows: the review
    # record is the evidence, so has_task=False does not zero the counts, and
    # conclusive/good follow review_quality.quality_result (low confidence or
    # unknown-without-issues is uncertain; defects or reference conflicts are
    # issues even under a good/unknown verdict)
    full_key = full_review_pipeline_key(FULL_QC_SOURCE)
    policy = quality_result(records)
    summary = catalog_query(
        "SELECT reviewed, missing_evidence, result_reviewed, issue_count, grade"
        " FROM atlas_cohort_summary WHERE load_id=:load_id AND pipeline_key=:key AND cohort='admitted'",
        {"load_id": load_result["load_id"], "key": full_key},
    )["rows"]
    assert len(summary) == 1
    row = summary[0]
    assert row["reviewed"] == FULL_QC_POPULATION - result.unavailable_count
    # missing_evidence keeps its packet semantics unchanged (FULL rows carry no
    # packets by design); the conclusive/good counts are the policy fix
    assert row["missing_evidence"] == FULL_QC_POPULATION
    assert row["result_reviewed"] == policy.reviewed
    assert row["issue_count"] == policy.issue_count
    assert row["grade"] == "bad"
    assert catalog_query(
        "SELECT bool_and(s.summary=to_jsonb(v)-'load_id'-'pipeline_key'-'cohort') AS identical "
        "FROM atlas_cohort_summaries s JOIN atlas_cohort_summary v USING(load_id,pipeline_key,cohort) "
        "WHERE s.load_id=:load AND s.pipeline_key=:key",
        {"load": load_result["load_id"], "key": full_key},
    )["rows"] == [{"identical": True}]
    with catalog_database.connect() as connection:
        plan = connection.execute(
            text(
                "EXPLAIN (FORMAT JSON) SELECT * FROM atlas_cohort_summary " "WHERE load_id=:load AND pipeline_key=:key"
            ),
            {"load": load_result["load_id"], "key": full_key},
        ).scalar_one()[0]["Plan"]
    nodes = [plan]
    group_keys = []
    while nodes:
        node = nodes.pop()
        group_keys.extend(node.get("Group Key", []))
        nodes.extend(node.get("Plans", []))
    assert group_keys and not any("curation" in expression for expression in group_keys)
    services = cast(AppletServices, CatalogServices(catalog_database))
    with TestClient(create_api(services)) as client:
        page = client.get("/assessment", params={"source_id": FULL_QC_SOURCE}).json()
        assert page["scope"] == "FULL"
        assert page["result"] == result.verdict.value.upper()
        assert page["finalization"] == "accepted_incomplete"
        assert page["waiver"]["sha256"] == bindings[FULL_QC_SOURCE]["waiver"]["sha256"]
        assert page["review_completeness"]["coverage_complete"] is False
        assert page["review_completeness"]["provisional"] is False
        report = page["report"]
        items = {row["task_id"]: row for row in report["evidence"]["items"]}
        assert report["evidence"]["total"] == FULL_QC_POPULATION
        assert items["t01"]["defects"] == ["missing_context"]
        assert items["t01"]["confidence"] == "high"
        assert items["t02"]["reference_status"] == "conflict"
        assert items["t03"]["defects"] == ["ambiguity", "rubric_mismatch"]
        assert items["t06"]["verdict"] == "good" and items["t06"]["defects"] == []
        assert items["t07"]["confidence"] == "low"
        assert items["t08"]["status"] == "unavailable" and items["t08"]["verdict"] == "unknown"
        assert items["t08"]["explanation"] == "Reviewer response truncated."
        # Whole-population SQL counts reconcile with the stored rows.
        counts = report["counts"]
        assert counts["population"] == FULL_QC_POPULATION and counts["item_count"] == FULL_QC_POPULATION
        assert counts["defect_counts"] == {**{defect: 0 for defect in counts["defect_counts"]}, **expected_defects}
        assert counts["confidence_counts"]["unrecorded"] == 1
        # Pagination and the defect filter narrow the same population.
        first = client.get("/assessment", params={"source_id": FULL_QC_SOURCE, "limit": 3}).json()["report"]["evidence"]
        assert [row["task_id"] for row in first["items"]] == ["t01", "t02", "t03"] and first[
            "total"
        ] == FULL_QC_POPULATION
        served_total = first["total"]
        last = client.get(
            "/assessment", params={"source_id": FULL_QC_SOURCE, "limit": 3, "offset": served_total - 3}
        ).json()["report"]["evidence"]
        # the endpoint orders by (cohort, task_id) as text, so past 99 rows the
        # three-digit ids sort before the two-digit ones
        expected_last = sorted(f"t{index:02d}" for index in range(1, served_total + 1))[-3:]
        assert [row["task_id"] for row in last["items"]] == expected_last
        ambiguous = sum(
            1 for record in records if record.verdict and "ambiguity" in [str(d) for d in record.verdict.defects]
        )
        defect_page = client.get(
            "/assessment", params={"source_id": FULL_QC_SOURCE, "defect": "ambiguity", "limit": 2, "offset": 1}
        ).json()["report"]["evidence"]
        assert defect_page["total"] == ambiguous
        assert len(defect_page["items"]) == 2
        assert (
            client.get("/assessment", params={"source_id": FULL_QC_SOURCE, "defect": "made_up_defect"}).status_code
            == 422
        )
        # The evidence view serves whole-population rows that have no archived packet.
        evidence = client.get("/evidence", params={"source_id": FULL_QC_SOURCE, "task_id": "t03"}).json()
        assert evidence["packet"] is None
        assert evidence["judgment"]["defects"] == ["ambiguity", "rubric_mismatch"]
        # Audit mode draws from the whole reviewed population.
        with identity_scope(VerifiedIdentity(user_id="auditor@example", role="user")):
            sample = client.post(
                "/audit/samples",
                json={"source_id": FULL_QC_SOURCE, "strategy": "random", "seed": "seed-full", "sample_size": 4},
            ).json()
            tasks = client.get(f"/audit/samples/{sample['sample_id']}/tasks").json()
            assert len(tasks["tasks"]) == 4
    # An unknown defect value in a receipt fails the load loudly; nothing is mapped to free text.
    unknown_root = str(tmp_path / "unknown-defect")
    verdict_json = full_qc_verdict("t1", "some_issues", "high", "consistent", ["missing_context"]).model_dump_json()
    unknown_review = verdict_json.replace('"missing_context"', '"typo_overlap"')
    (tmp_path / "unknown").mkdir()
    unknown_bindings = full_qc_binding(
        tmp_path / "unknown",
        SourceQualityResult.model_validate(bindings[FULL_QC_SOURCE]["result"]),
        [
            ReviewRecord(
                task_id="t1",
                status="reviewed",
                verdict=full_qc_verdict("t1", "some_issues", "high", "consistent", ["missing_context"]),
                detail="",
            )
        ],
    )
    # Swap the records for a receipt whose review JSON carries the unknown defect value.
    schema = pa.schema(
        [
            ("task_id", pa.string()),
            ("task_sha256", pa.string()),
            ("runtime_sha256", pa.string()),
            ("review", pa.large_string()),
            ("batch_receipt", pa.string()),
        ]
    )
    unknown_records_path = tmp_path / "unknown" / "records.parquet"
    with pq.ParquetWriter(unknown_records_path, schema, compression="zstd") as writer:
        writer.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "task_id": "t1",
                        "task_sha256": "b" * 64,
                        "runtime_sha256": "c" * 64,
                        "review": unknown_review,
                        "batch_receipt": "s3://fixture/receipt.json",
                    }
                ],
                schema=schema,
            )
        )
    unknown_bindings[FULL_QC_SOURCE]["records"] = {
        "uri": str(unknown_records_path),
        "sha256": file_sha256(unknown_records_path),
    }
    publish_full_bound_source(
        unknown_root,
        "unknown",
        result=SourceQualityResult.model_validate(unknown_bindings[FULL_QC_SOURCE]["result"]),
        waiver=ContentEvidenceReference.model_validate(unknown_bindings[FULL_QC_SOURCE]["waiver"]),
    )
    with pytest.raises(ValueError, match="typo_overlap"):
        load_surveyed_catalog(
            CurationCatalog(unknown_root),
            catalog_query,
            catalog_backend_map(CurationCatalog(unknown_root)),
            catalog_identity_map(CurationCatalog(unknown_root)),
            full_reviews={
                key: FullContentResultBinding.model_validate(value) for key, value in unknown_bindings.items()
            },
        )


def test_manual_audit_sample_judgments_and_agreement(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "audit")
    types = {
        "one": {"confidence": "low", "reference_status": "conflict", "defects": ["wrong_reference"]},
        "two": {"confidence": "high", "reference_status": "consistent", "defects": ["ambiguity"]},
        "three": {"confidence": "medium", "reference_status": "consistent", "defects": []},
    }
    publish_sample(root, "audit", admitted_verdicts=("some_issues", "bad"), review_types=types)
    load_surveyed_catalog(
        CurationCatalog(root),
        catalog_query,
        catalog_backend_map(CurationCatalog(root)),
        catalog_identity_map(CurationCatalog(root)),
    )
    services = cast(AppletServices, CatalogServices(catalog_database))
    request = {"source_id": "Marin:math", "strategy": "stratified", "seed": "seed-1", "sample_size": 2}
    with identity_scope(VerifiedIdentity(user_id="auditor@example", role="user")):
        with TestClient(create_api(services)) as client:
            first = client.post("/audit/samples", json=request).json()
            again = client.post("/audit/samples", json=request).json()
            assert first["sample_id"] == again["sample_id"]
            assert again["existing"] is True
            tasks = client.get(f"/audit/samples/{first['sample_id']}/tasks").json()
            assert len(tasks["tasks"]) == 2
            sample_id = first["sample_id"]
            assert (
                client.put(f"/audit/samples/{sample_id}/judgments", json={"task_id": "one", "agree": False}).status_code
                == 422
            )
            saved = client.put(
                f"/audit/samples/{sample_id}/judgments",
                json={
                    "task_id": "one",
                    "agree": False,
                    "corrected_verdict": "good",
                    "corrected_defects": [],
                    "note": "Reference is fine",
                },
            )
            assert saved.status_code == 200
            updated = client.put(
                f"/audit/samples/{sample_id}/judgments",
                json={
                    "task_id": "one",
                    "agree": False,
                    "corrected_verdict": "some_issues",
                    "corrected_defects": ["ambiguity"],
                },
            )
            assert updated.status_code == 200
            outside = client.put(
                f"/audit/samples/{sample_id}/judgments",
                json={
                    "task_id": "absent",
                    "agree": True,
                },
            )
            assert outside.status_code == 404
            agreement = client.get(f"/audit/samples/{sample_id}/agreement").json()
            assert agreement["overall"]["judged"] == 1
            assert agreement["per_defect"]["wrong_reference"]["judged"] == 1
    # Another user's judgment is separate; the first user's is unchanged.
    with identity_scope(VerifiedIdentity(user_id="second@example", role="user")):
        with TestClient(create_api(services)) as client:
            client.put(f"/audit/samples/{sample_id}/judgments", json={"task_id": "one", "agree": True})
            tasks = client.get(f"/audit/samples/{sample_id}/tasks").json()
            judgments = [row["judgment"] for row in tasks["tasks"] if row["task_id"] == "one"]
            assert len(judgments) == 1
            agreement = client.get(f"/audit/samples/{sample_id}/agreement").json()
            assert agreement["overall"]["judged"] == 2
            assert agreement["overall"]["agree"] == 1
    with TestClient(create_api(services)) as client:
        assert client.post("/audit/samples", json=request).status_code == 401
        assert client.get(f"/audit/samples/{sample_id}/tasks").status_code == 401


def test_backend_verification_pending_persists_and_projects(catalog_database, catalog_query, tmp_path):
    """The pending marker is persisted in the source row and survives the API projection."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "pending-verify", source_id="Marin:pending")
    catalog = CurationCatalog(root)
    load_catalog(catalog, catalog_query)
    survey = catalog_backend_map(catalog)
    flipped = survey.model_copy(
        update={
            "sources": [
                (
                    source.model_copy(
                        update={
                            "backends": {
                                **source.backends,
                                "daytona": source.backends["daytona"].model_copy(update={"support": Support.VERIFY}),
                            }
                        }
                    )
                    if source.source_id == "Marin:pending"
                    else source
                )
                for source in survey.sources
            ]
        }
    )
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    rows = catalog_query(
        "SELECT id, backend_verification_pending FROM atlas_sources WHERE load_id=:load_id",
        {"load_id": result["load_id"]},
    )["rows"]
    pending_rows = {row["id"]: row["backend_verification_pending"] for row in rows}
    assert pending_rows["Marin:pending"] == ["daytona"]
    # the API projection passes the marker through without rewriting it
    projected = canonical_source_record(
        {
            "id": "Marin:pending",
            "declaration": {"id": "Marin:pending"},
            "backend_verification_pending": pending_rows["Marin:pending"],
        }
    )
    assert projected["backend_verification_pending"] == ["daytona"]


def _verify_pending_survey(survey: BackendSurvey, source_id: str) -> BackendSurvey:
    """Flip every production backend of one source to VERIFY (pending live qualification)."""
    return survey.model_copy(
        update={
            "sources": [
                (
                    source.model_copy(
                        update={
                            "backends": {
                                **source.backends,
                                **{
                                    backend: source.backends[backend].model_copy(update={"support": Support.VERIFY})
                                    for backend in ("daytona", "iris", "local")
                                },
                            }
                        }
                    )
                    if source.source_id == source_id
                    else source
                )
                for source in survey.sources
            ]
        }
    )


def _source_load_row(catalog_query: AppletQuery, load_id: str, source_id: str) -> dict:
    row = catalog_query(
        "SELECT id, declaration->>'status' AS status, declaration->>'notes' AS notes, "
        "backend_verification_pending, sample_pin_stale"
        " FROM atlas_sources WHERE load_id=:load_id AND id=:source_id",
        {"load_id": load_id, "source_id": source_id},
    )["rows"]
    assert len(row) == 1
    return row[0]


def test_new_verify_source_materializes_as_pending_with_marker(catalog_database, catalog_query, tmp_path):
    """A new admission whose only positive production outcome is VERIFY loads as Pending."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "new-verify", source_id="Marin:newverify")
    catalog = CurationCatalog(root)
    flipped = _verify_pending_survey(catalog_backend_map(catalog), "Marin:newverify")
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    row = _source_load_row(catalog_query, result["load_id"], "Marin:newverify")
    assert row["status"] == "Pending"
    assert "Backend verification pending" in row["notes"]
    assert row["backend_verification_pending"] == ["daytona", "iris", "local"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        served = next(
            source for source in client.get("/source-statuses").json()["sources"] if source["id"] == "Marin:newverify"
        )
        assert served["stage"] == "Backend verification pending"


def test_admitted_source_stays_available_when_every_production_backend_is_verify(
    catalog_database, catalog_query, tmp_path
):
    """An already-admitted source keeps Available under VERIFY, with the marker."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "admitted-verify", source_id="Marin:admitted")
    catalog = CurationCatalog(root)
    load_catalog(catalog, catalog_query)
    flipped = _verify_pending_survey(catalog_backend_map(catalog), "Marin:admitted")
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    row = _source_load_row(catalog_query, result["load_id"], "Marin:admitted")
    assert row["status"] == "Available"
    assert row["backend_verification_pending"] == ["daytona", "iris", "local"]


def test_new_source_with_a_runnable_production_backend_stays_available(catalog_database, catalog_query, tmp_path):
    """A runnable production backend admits a new source regardless of sibling VERIFY cells."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "mixed-verify", source_id="Marin:mixed")
    catalog = CurationCatalog(root)
    survey = catalog_backend_map(catalog)
    flipped = survey.model_copy(
        update={
            "sources": [
                (
                    source.model_copy(
                        update={
                            "backends": {
                                **source.backends,
                                # one runnable production backend + a VERIFY sibling:
                                # the runnable backend admits the new source
                                "daytona": source.backends["daytona"].model_copy(update={"support": Support.VERIFY}),
                                "local": source.backends["local"].model_copy(update={"support": FitSupport.SUPPORTED}),
                            }
                        }
                    )
                    if source.source_id == "Marin:mixed"
                    else source
                )
                for source in survey.sources
            ]
        }
    )
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    row = _source_load_row(catalog_query, result["load_id"], "Marin:mixed")
    assert row["status"] == "Available"
    assert row["backend_verification_pending"] == ["daytona"]


def _unsupported_profile_survey(
    catalog: CurationCatalog, source_id: str, *, grader_demands_network: bool
) -> BackendSurvey:
    """Every backend UNSUPPORTED for one source, with the survey's grader profile denying network."""
    survey = catalog_backend_map(catalog)
    profile = RuntimeProfile(
        network=NetworkPolicy.DENY,
        cpus=1,
        memory_mb=512,
        storage_mb=1024,
        gpus=0,
        user=None,
        startup_timeout=900.0,
        cleanup_timeout=120.0,
    )
    row = next(s for s in survey.sources if s.source_id == source_id)
    count = row.task_count or 1
    backends = {
        backend: fit.model_copy(
            update={
                "support": FitSupport.UNSUPPORTED,
                "task_counts": {FitSupport.UNSUPPORTED: count},
                "roles": {
                    "agent": {FitSupport.UNSUPPORTED: count},
                    "grader": {FitSupport.UNSUPPORTED: count},
                },
                "blockers": {"agent: Machine execution requires explicit semantics and native dependencies": count},
            }
        )
        for backend, fit in row.backends.items()
    }
    demands = (
        {"grader": {Requirement.NETWORK: count}} if grader_demands_network else {"grader": {Requirement.CPUS: count}}
    )
    new_row = row.model_copy(update={"backends": backends, "demands": demands})
    return survey.model_copy(
        update={
            "sources": [new_row if s.source_id == source_id else s for s in survey.sources],
            "runtime_defaults": {"agent": profile, "grader": profile},
        }
    )


def test_grader_profile_conflict_ships_as_typed_pending(catalog_database, catalog_query, tmp_path):
    """A source whose grader needs network under deny defaults loads as Pending with the stage note."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "profile-conflict", source_id="Marin:orshape")
    catalog = CurationCatalog(root)
    flipped = _unsupported_profile_survey(catalog, "Marin:orshape", grader_demands_network=True)
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    row = _source_load_row(catalog_query, result["load_id"], "Marin:orshape")
    assert row["status"] == "Pending"
    assert "Backend profile pending: grader needs network allow." in row["notes"]
    # no VERIFY cells here: the marker stays unset; the typed stage note is the visibility
    assert row["backend_verification_pending"] is None
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        served = next(
            source for source in client.get("/source-statuses").json()["sources"] if source["id"] == "Marin:orshape"
        )
        assert served["stage"] == "Backend profile pending"


def test_all_unsupported_without_a_profile_conflict_still_blocks(catalog_database, catalog_query, tmp_path):
    """An all-unsupported Available source with no network demand keeps the hard blocker."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "plain-unsupported", source_id="Marin:plain")
    catalog = CurationCatalog(root)
    flipped = _unsupported_profile_survey(catalog, "Marin:plain", grader_demands_network=False)
    with pytest.raises(ValueError, match="Backend admission blocked"):
        load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))


def test_docker_build_preparation_gate_ships_as_typed_pending(catalog_database, catalog_query, tmp_path):
    """An all-unsupported Available source whose grader fails only on the unresolved Docker
    build gate loads as Pending: the gate is a source preparation requirement, not a
    backend capability gap, and the image preparation re-admits it."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "build-pending", source_id="Marin:buildgate")
    catalog = CurationCatalog(root)
    survey = catalog_backend_map(catalog)
    row = next(s for s in survey.sources if s.source_id == "Marin:buildgate")
    count = row.task_count or 1
    backends = {
        backend: fit.model_copy(
            update={
                "support": FitSupport.UNSUPPORTED,
                "task_counts": {FitSupport.UNSUPPORTED: count},
                "roles": {
                    "agent": {FitSupport.PARTIAL: count},
                    "grader": {FitSupport.UNSUPPORTED: count},
                },
                "blockers": {"agent:non_root_user": count},
                "validation_errors": {
                    "grader: Unresolved Docker build context must be built and pinned before execution": count
                },
            }
        )
        for backend, fit in row.backends.items()
    }
    flipped = survey.model_copy(
        update={
            "sources": [
                row.model_copy(update={"backends": backends}) if s.source_id == "Marin:buildgate" else s
                for s in survey.sources
            ]
        }
    )
    result = load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))
    served = _source_load_row(catalog_query, result["load_id"], "Marin:buildgate")
    assert served["status"] == "Pending"
    assert "Backend preparation pending: the grader Docker build context is unresolved" in served["notes"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        served_row = next(
            source for source in client.get("/source-statuses").json()["sources"] if source["id"] == "Marin:buildgate"
        )
        assert served_row["stage"] == "Backend preparation pending"


def test_docker_build_gate_with_other_validation_errors_still_blocks(catalog_database, catalog_query, tmp_path):
    """An all-unsupported source whose grader fails on something besides the build gate
    keeps the hard blocker: only the source preparation gate downgrades."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "other-gate", source_id="Marin:othergate")
    catalog = CurationCatalog(root)
    survey = catalog_backend_map(catalog)
    row = next(s for s in survey.sources if s.source_id == "Marin:othergate")
    count = row.task_count or 1
    backends = {
        backend: fit.model_copy(
            update={
                "support": FitSupport.UNSUPPORTED,
                "task_counts": {FitSupport.UNSUPPORTED: count},
                "roles": {
                    "agent": {FitSupport.UNSUPPORTED: count},
                    "grader": {FitSupport.UNSUPPORTED: count},
                },
                "validation_errors": {"grader: Network capability requires an allow network policy": count},
            }
        )
        for backend, fit in row.backends.items()
    }
    flipped = survey.model_copy(
        update={
            "sources": [
                row.model_copy(update={"backends": backends}) if s.source_id == "Marin:othergate" else s
                for s in survey.sources
            ]
        }
    )
    with pytest.raises(ValueError, match="Backend admission blocked"):
        load_surveyed_catalog(catalog, catalog_query, flipped, catalog_identity_map(catalog))


def test_new_train_admission_without_lexical_clearance_is_refused(catalog_database, catalog_query, tmp_path):
    """The armed gate: a new Available TRAIN source without clearance cannot load."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "uncleared", source_id="Marin:uncleared", lexical_clearance=False)
    catalog = CurationCatalog(root)
    with pytest.raises(ValueError, match="Decontamination"):
        load_surveyed_catalog(catalog, catalog_query, catalog_backend_map(catalog), catalog_identity_map(catalog))


def test_archived_lineage_panel_grandfathered_with_marker(catalog_database, catalog_query, tmp_path):
    """The SAMPLE pin gate: new ARCHIVED_LINEAGE admissions refuse; grandfathered rows keep the marker."""
    root = str(tmp_path / "catalog")
    publish_sample(root, "stale-panel", source_id="Marin:stalepanel")
    catalog = CurationCatalog(root)
    survey = catalog_backend_map(catalog)
    stale = next(source.declaration for source in catalog.sources(catalog.current().snapshot_id)).model_copy(
        update={
            "independent_qc": IndependentQC(
                verdict="good",
                method="Archived SAMPLE content panel",
                reviewed=100,
                evidence_url="s3://test/panel",
                panel_currency=SamplePanelCurrency.ARCHIVED_LINEAGE,
            )
        }
    )
    with DataStore.open(root, writer_id="stale-panel") as store:
        register_catalog(store)
        with store.transaction() as tx:
            tx.table(INVENTORY).add(
                SnapshotSource(
                    snapshot_id="stale-panel",
                    id=stale.id,
                    declaration=stale,
                    assessment_key=None,
                    latest_attempt_key=None,
                    latest_attempt_status="skipped",
                    reason=None,
                    curation=None,
                ).model_dump(mode="json")
            )
            tx.table(SNAPSHOTS).add(
                CatalogSnapshot(
                    snapshot_id="stale-panel", created_at="2026-11-17T00:00:00Z", source_count=1
                ).model_dump()
            )
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id="stale-panel").model_dump())
    stale_catalog = CurationCatalog(root)
    with pytest.raises(ValueError, match="older population pin"):
        load_surveyed_catalog(stale_catalog, catalog_query, survey, catalog_identity_map(stale_catalog))
    # grandfathering: seed the previous load with the same Available source, then reload
    seed_previous_available(catalog_database, ["Marin:stalepanel"])
    result = load_surveyed_catalog(stale_catalog, catalog_query, survey, catalog_identity_map(stale_catalog))
    row = _source_load_row(catalog_query, result["load_id"], "Marin:stalepanel")
    assert row["status"] == "Available"
    assert row["sample_pin_stale"] is True
    projected = canonical_source_record(
        {
            "id": "Marin:stalepanel",
            "declaration": {"id": "Marin:stalepanel"},
            "sample_pin_stale": True,
        }
    )
    assert projected["sample_pin_stale"] is True


def test_qc_inventory_keeps_its_selected_load_during_a_refresh(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "inventory-refresh")
    publish_sample(root, "inventory-refresh")
    load_catalog(CurationCatalog(root), catalog_query)
    catalog_query(
        "INSERT INTO atlas_sources SELECT (jsonb_populate_record(NULL::atlas_sources, "
        "to_jsonb(s)||jsonb_build_object('id','Marin:unreviewed','assessment_key',NULL))).* "
        "FROM atlas_sources s WHERE id='Marin:math'",
        {},
    )
    catalog_query(
        "INSERT INTO atlas_loads SELECT (jsonb_populate_record(NULL::atlas_loads, "
        "to_jsonb(l)||jsonb_build_object('load_id','next-load','snapshot_id','next'))).* "
        "FROM atlas_loads l JOIN atlas_current c USING(load_id) WHERE c.name='latest'",
        {},
    )
    switched = False

    def select_new_load(connection, _cursor, statement, _parameters, _context, _many):
        nonlocal switched
        if not switched and "atlas_review_current" in statement:
            switched = True
            connection.execute(text("UPDATE atlas_current SET load_id='next-load' WHERE name='latest'"))

    event.listen(catalog_database, "before_cursor_execute", select_new_load)
    try:
        with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
            inventory = client.get("/qc/inventory").json()
        assert switched
        assert {row["source_id"] for row in inventory["panels"]} == {"Marin:math", "Marin:unreviewed"}
    finally:
        event.remove(catalog_database, "before_cursor_execute", select_new_load)


@pytest.mark.parametrize("shared_batch", [False, True, "chunked"])
def test_additive_sample_browse_preserves_catalog_and_exposes_raw_reviews(
    catalog_database, catalog_query, tmp_path, shared_batch
):
    root = str(tmp_path / "sample-browse")
    publish_sample(root, "sample-browse")
    load_catalog(CurationCatalog(root), catalog_query)
    catalog_query(
        "UPDATE atlas_sources SET declaration=jsonb_set(declaration,'{independent_qc}',"
        "jsonb_build_object('evidence_url',CAST(:uri AS TEXT),'reviewed',3,'admitted',3,"
        "'method','llm','verdict','good'),true) "
        "WHERE id='Marin:math'",
        {"uri": "s3://fixture/panel/.artifact.json"},
    )
    before = catalog_query("SELECT load_id,declaration,assessment_key FROM atlas_sources WHERE id='Marin:math'", {})[
        "rows"
    ][0]
    old_key = before["assessment_key"]
    load_id = before["load_id"]
    catalog_query(
        "INSERT INTO atlas_sources SELECT (jsonb_populate_record(NULL::atlas_sources, "
        "to_jsonb(s)||jsonb_build_object('id','Marin:unreviewed','assessment_key',NULL))).* "
        "FROM atlas_sources s WHERE load_id=:load AND id='Marin:math'",
        {"load": load_id},
    )
    key = "qc:" + "b" * 64
    assessment = catalog_query(
        "SELECT * FROM atlas_assessments WHERE load_id=:load AND pipeline_key=:key", {"load": load_id, "key": old_key}
    )["rows"][0]
    assessment.pop("load_id")
    assessment["pipeline_key"] = key
    populations = catalog_query(
        "SELECT pipeline_key,cohort,population,sample_count FROM atlas_populations "
        "WHERE load_id=:load AND pipeline_key=:key",
        {"load": load_id, "key": old_key},
    )["rows"]
    items = catalog_query(
        "SELECT * FROM atlas_items WHERE load_id=:load AND pipeline_key=:key ORDER BY task_id",
        {"load": load_id, "key": old_key},
    )["rows"]
    for row in [*populations, *items]:
        row.pop("load_id", None)
        row["pipeline_key"] = key
    items[0]["defects"] = ["ambiguity"]
    batch = {
        "attempts": [
            {
                "number": 0,
                "query_task_ids": {row["task_id"]: [row["task_id"]] for row in items},
                "requests": {
                    "requests": [
                        {
                            "custom_id": row["task_id"],
                            "body": {"messages": [{"role": "user", "content": "exact reviewer input"}]},
                        }
                        for row in items
                    ],
                    "observations": [
                        {
                            "output": "\n".join(
                                json.dumps({"custom_id": row["task_id"], "body": "unabridged reviewer output"})
                                for row in items
                            )
                        }
                    ],
                },
            }
        ]
    }
    if shared_batch == "chunked":
        # Incompressible text crosses the real gateway request budget after compression.
        batch["attempts"][0]["requests"]["requests"][0]["body"]["messages"][0]["content"] = "".join(
            hashlib.sha256(str(index).encode()).hexdigest() for index in range(40_000)
        )
    messages = task_request_evidence(batch)
    raw = [
        {
            "pipeline_key": key,
            "task_id": row["task_id"],
            "record": {**messages[row["task_id"]], "provenance": {"sha256": "a" * 64}},
        }
        for row in items
    ]
    expected_raw = json.loads(json.dumps(raw))
    receipt = {
        "artifact": {"uri": "s3://fixture/panel/.artifact.json", "sha256": "b" * 64},
        "panel": {"dataset_id": "original", "dataset_revision": "old-pin"},
        "current_pin": False,
    }
    staged = {
        "data_restriction": "open",
        "assessment": assessment,
        "populations": populations,
        "items": items,
        "raw": raw,
        "receipt": receipt,
    }
    if shared_batch:
        staged["batches"] = [{"pipeline_key": key, "batch_key": "shared", "record": pack_content(batch)}]
        staged["batch_links"] = [
            {"pipeline_key": key, "task_id": row["task_id"], "batch_key": "shared"} for row in items
        ]
        staged["receipt"]["linked_tasks"] = len(items)
        for row in raw:
            del row["record"]["input"], row["record"]["output"]
    # Another regional panel may have populations inserted while its task rows are still loading.
    catalog_query(
        "INSERT INTO atlas_populations VALUES (:load,'incomplete-sibling','admitted',2,2)",
        {"load": load_id},
    )

    def bounded_query(sql, parameters):
        if "SELECT COUNT(*) AS matched FROM jsonb_to_recordset" in sql and len(json.loads(parameters["rows"])) > 1:
            raise MarinaRequestError(504, "deterministic multi-row deadline")
        return catalog_query(sql, parameters)

    assert load_content_reviews(staged, load_id, "Marin:math", bounded_query)["atlas_items"] == 3
    assert (
        catalog_query(
            "SELECT summary FROM atlas_cohort_summaries WHERE load_id=:load AND pipeline_key='incomplete-sibling'",
            {"load": load_id},
        )["rows"]
        == []
    )
    if shared_batch == "chunked":
        # A previous loader stored identical chunked evidence before the scalar count column existed.
        catalog_query(
            "UPDATE atlas_review_batches SET chunk_count=0 WHERE load_id=:load AND pipeline_key=:key",
            {"load": load_id, "key": key},
        )
    assert not any(load_content_reviews(staged, load_id, "Marin:math", catalog_query).values())
    after = catalog_query("SELECT load_id,declaration,assessment_key FROM atlas_sources WHERE id='Marin:math'", {})[
        "rows"
    ][0]
    assert after == before
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        page = client.get("/assessment", params={"source_id": "Marin:math", "defect": "ambiguity"}).json()
        assert page["report"]["evidence"]["total"] == 1
        assert page["report"]["evidence"]["items"][0]["task_id"] == items[0]["task_id"]
        assert page["review_input"] == receipt
        candidate = CandidateApi(catalog_query)
        try:
            assert candidate.get("api/assessment?source_id=Marin%3Amath&defect=ambiguity") == page
            assert (
                candidate.get("api/source?source_id=Marin%3Amath")
                == client.get("/source", params={"source_id": "Marin:math"}).json()
            )
        finally:
            candidate.close()
        detail = client.get("/evidence", params={"source_id": "Marin:math", "task_id": items[0]["task_id"]}).json()
        assert detail["raw_review"] == expected_raw[0]["record"]
        # An archived copy of the same pipeline must not inflate the selected panel.
        catalog_query(
            "INSERT INTO atlas_raw_reviews SELECT 'archive-'||load_id,pipeline_key,task_id,record,reviewer "
            "FROM atlas_raw_reviews WHERE load_id=:load AND pipeline_key=:key",
            {"load": load_id, "key": key},
        )
        inventory = client.get("/qc/inventory").json()
        assert (
            len(inventory["panels"])
            == catalog_query("SELECT COUNT(*) AS count FROM atlas_sources WHERE load_id=:load", {"load": load_id})[
                "rows"
            ][0]["count"]
        )
        panel = next(row for row in inventory["panels"] if row["source_id"] == "Marin:math")
        assert panel["task_count"] == panel["raw_count"] == 3
        assert panel["linked_count"] == (3 if shared_batch else 0)
        assert panel["missing_batches"] == 0
        unreviewed = next(row for row in inventory["panels"] if row["source_id"] == "Marin:unreviewed")
        assert unreviewed["pipeline_key"] is None
        assert {
            name: unreviewed[name]
            for name in ("task_count", "raw_count", "linked_count", "missing_batches", "missing_chunks")
        } == dict.fromkeys(("task_count", "raw_count", "linked_count", "missing_batches", "missing_chunks"), 0)
        legend = client.get("/qc/defects").json()
        ambiguity = next(row for row in legend["defects"] if row["id"] == "ambiguity")
        assert ambiguity["count"] == 1
        assert ambiguity["sources"] == [{"id": "Marin:math", "name": before["declaration"]["display_name"], "count": 1}]
        if not shared_batch:
            catalog_query(
                "UPDATE atlas_raw_reviews SET reviewer=CAST(:reviewer AS JSONB) "
                "WHERE load_id=:load AND pipeline_key=:key AND task_id=:task",
                {
                    "reviewer": json.dumps({"changed": "canary"}),
                    "load": load_id,
                    "key": key,
                    "task": items[0]["task_id"],
                },
            )
            with pytest.raises(ValueError, match="Stored atlas_raw_reviews differs"):
                load_content_reviews(staged, load_id, "Marin:math", bounded_query)
        if shared_batch == "chunked":
            catalog_query(
                "DELETE FROM atlas_review_batch_chunks WHERE load_id=:load AND pipeline_key=:key AND chunk_index=0",
                {"load": load_id, "key": key},
            )
            missing = client.get("/evidence", params={"source_id": "Marin:math", "task_id": items[0]["task_id"]})
            assert missing.status_code == 503
            incomplete = client.get("/qc/inventory").json()
            assert next(row for row in incomplete["panels"] if row["source_id"] == "Marin:math")["missing_chunks"] == 1
        if shared_batch:
            catalog_query(
                "UPDATE atlas_sources SET declaration=jsonb_set(declaration,'{data_restriction}',"
                "CAST(:tier AS JSONB)) WHERE load_id=:load AND id='Marin:math'",
                {"load": load_id, "tier": json.dumps("private_always")},
            )

            def refuse_private_batch(_connection, _cursor, statement, _parameters, _context, _many):
                if "SELECT record FROM atlas_review_batches" in statement:
                    raise AssertionError("Private original reviewer batch was read")

            event.listen(catalog_database, "before_cursor_execute", refuse_private_batch)
            try:
                restricted = client.get(
                    "/evidence", params={"source_id": "Marin:math", "task_id": items[0]["task_id"]}
                ).json()
                assert restricted["raw_review"]["private"] is True
                assert "exact reviewer input" not in json.dumps(restricted)
            finally:
                event.remove(catalog_database, "before_cursor_execute", refuse_private_batch)


def test_staged_rollout_pages_keep_attempt_identity_and_exact_content(catalog_database, catalog_query):
    declaration = with_declaration_admission_census(contamination_declarations()[0][1])
    measured = difficulty_measurement(declaration, Solver.SMALL)
    load_measurements([measured], catalog_query)
    with catalog_database.begin() as connection:
        connection.execute(text("INSERT INTO atlas_loads VALUES ('rollouts','fixture','rollouts',NOW(),NOW())"))
        connection.execute(text("UPDATE atlas_current SET load_id='rollouts' WHERE name='latest'"))
        connection.execute(
            text(
                "INSERT INTO atlas_sources (load_id,id,declaration,latest_attempt_status,diagnostics) "
                "VALUES ('rollouts',:source,CAST(:declaration AS JSONB),'skipped','{}')"
            ),
            {"source": declaration.id, "declaration": declaration.model_dump_json()},
        )
    records = []
    for index in range(2):
        records.append(
            {
                "attempt_id": f"attempt-{index}",
                "task_id": measured.sample.tasks[index].task_id,
                "summary": {
                    "attempt_id": f"attempt-{index}",
                    "task_id": measured.sample.tasks[index].task_id,
                    "solver": "small",
                    "score": float(index),
                    "duration": 2.5,
                    "status": "scored",
                    "exception": None,
                    "has_rollout": True,
                },
                "detail": {
                    "conversation": [
                        {
                            "role": "assistant",
                            "content": f"Exact π response {index}\nNext line",
                            "tool_calls": [{"name": "terminal", "arguments": "check.py"}],
                        }
                    ],
                    "verifier": {"reward": index},
                },
            }
        )
    metadata = {
        "data_restriction": "open",
        "attempt_count": 2,
        "recorded_attempt_count": 2,
        "attempts_receipt": measured.attempts_receipt.model_dump(mode="json"),
    }
    measurement = measured.model_dump(mode="json")
    assert load_rollouts(measurement, metadata, records, catalog_query)["inserted"] == 2
    assert load_rollouts(measurement, metadata, records, catalog_query)["inserted"] == 0
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        parameters = {"source_id": declaration.id, "measurement_id": measured.measurement_id, "limit": 1}
        first = client.get("/difficulty-attempts", params=parameters).json()
        assert first["current_pin"] is True
        assert first["total"] == 2 and first["items"] == [records[0]["summary"]]
        second = client.get("/difficulty-attempts", params={**parameters, "offset": 1}).json()
        assert second["items"] == [records[1]["summary"]]
        detail = client.get(
            "/difficulty-rollout",
            params={"source_id": declaration.id, "measurement_id": measured.measurement_id, "attempt_id": "attempt-0"},
        ).json()
        assert detail == {"summary": records[0]["summary"], "detail": records[0]["detail"]}
        assert (
            client.get(
                "/difficulty-rollout",
                params={
                    "source_id": "Fixture:another-source",
                    "measurement_id": measured.measurement_id,
                    "attempt_id": "attempt-0",
                },
            ).status_code
            == 404
        )
    assert catalog_query(
        "SELECT measurement FROM atlas_difficulty WHERE measurement_id=:id", {"id": measured.measurement_id}
    )["rows"] == [{"measurement": measurement}]


def test_quality_explanation_inventory_detail_and_cache_follow_full_receipt_inputs(
    catalog_database, catalog_query, tmp_path
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "quality-explanations")
    load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        initial = client.get("/sources").json()["sources"][0]
        assert initial["quality_explanation"]["code"] == "no_recorded_full_review"
        source_id = initial["id"]
        panel = {
            "source_id": source_id,
            "dataset_id": initial["dataset_id"],
            "dataset_revision": initial["dataset_revision"],
            "converter_version": initial["pipeline"]["version"],
            "task_specs_sha256": initial["actor_census_input"]["task_specs_sha256"],
            "retained_tasks": initial["actor_census_input"]["measured_tasks"],
        }
        receipt = {"artifact": {"uri": "s3://fixture/full-review", "sha256": "a" * 64}, "panel": panel}
        with catalog_database.begin() as connection:
            connection.execute(
                text(
                    """
                INSERT INTO atlas_review_current (load_id,source_id,pipeline_key,receipt)
                SELECT load_id,id,assessment_key,CAST(:receipt AS JSONB) FROM atlas_sources WHERE id=:source
            """
                ),
                {"source": source_id, "receipt": json.dumps(receipt)},
            )
        current = client.get("/sources", params={"projection": "table"}).json()["sources"][0]
        assert current["quality_explanation"]["code"] == "full_review_incomplete"
        detail = client.get("/source", params={"source_id": source_id}).json()
        assert detail["quality_explanation"] == current["quality_explanation"]
        assert current["quality"] == initial["quality"]
        assert current["filtration"] == initial["filtration"]
        receipt["panel"]["dataset_revision"] = "c" * 40
        with catalog_database.begin() as connection:
            connection.execute(
                text("UPDATE atlas_review_current SET receipt=CAST(:receipt AS JSONB)"), {"receipt": json.dumps(receipt)}
            )
        archived = client.get("/sources").json()["sources"][0]
        assert archived["quality_explanation"]["code"] == "full_review_other_input"
        assert archived["quality_explanation"]["full"]["different_fields"] == ["dataset_revision"]
        assert (
            archived["quality_explanation"]
            == client.get("/source", params={"source_id": source_id}).json()["quality_explanation"]
        )


def test_dataset_method_variants_survive_catalog_and_api_projection(tmp_path, catalog_database, catalog_query):
    root = str(tmp_path / "methods")
    publish_sample(root, "method-snapshot")
    load_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.begin() as connection:
        row = (
            connection.execute(text("SELECT load_id,id,declaration FROM atlas_sources ORDER BY id LIMIT 1"))
            .mappings()
            .one()
        )
        variants = [
            rl_variant(row["declaration"]).model_dump(mode="json"),
            sft_variant(row["id"]).model_dump(mode="json"),
        ]
        connection.execute(
            text("UPDATE atlas_sources SET method_variants=CAST(:variants AS jsonb) WHERE load_id=:load AND id=:source"),
            {"variants": json.dumps(variants), "load": row["load_id"], "source": row["id"]},
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        inventory = client.get("/sources", params={"projection": "table"}).json()["sources"]
        source = next(source for source in inventory if source["id"] == row["id"])
        assert source["method_tags"] == ["RL", "SFT-viable"]
        assert "method_variants" not in source
        evidence = client.get("/source", params={"source_id": row["id"]}).json()
        assert evidence["method_tags"] == ["RL", "SFT-viable"]
        assert evidence["method_variants"] == variants


def _curriculum_reloptions(query: AppletQuery) -> dict[str, list[str] | None]:
    rows = query(
        "SELECT relname, reloptions FROM pg_class WHERE relname = ANY(CAST(:names AS TEXT[]))"
        " AND relnamespace = to_regnamespace(current_schema())",
        {"names": list(CURRICULUM_ANALYZED_TABLES)},
    )["rows"]
    return {row["relname"]: row["reloptions"] for row in rows}


def test_curriculum_tables_are_analyzed_after_every_load(catalog_query):
    """Regression (R40 #58): a load adding a tenth of the rows stayed under the default
    autoanalyze threshold, so the planner estimated R40's load at a third of its size."""
    expected = {table: ["autovacuum_analyze_scale_factor=0.01"] for table in CURRICULUM_ANALYZED_TABLES}
    assert _curriculum_reloptions(catalog_query) == expected
    for table in CURRICULUM_ANALYZED_TABLES:
        catalog_query(f"ALTER TABLE {table} RESET (autovacuum_analyze_scale_factor)", {})
    assert set(_curriculum_reloptions(catalog_query).values()) == {None}
    ensure_curriculum_analyze_settings(catalog_query)
    assert _curriculum_reloptions(catalog_query) == expected
