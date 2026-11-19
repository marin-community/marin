# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end private-data tier tests: canary leak sweep over loader, API and publish paths.

A synthetic PRIVATE_ALWAYS fixture source carries unique canary strings in every
content-bearing field — packets, reviewer text, rollout traces, curriculum previews,
archived reviews, audit notes and declaration notes. The tests load the catalog, serve
every content surface through the Atlas API, run the curriculum load and publish guards,
and assert the canary never appears in any response, artifact or log, while counts,
statuses, hashes and receipts stay visible and the open fixture keeps serving content.
Tamper tests refuse derived artifacts whose restriction tag was flipped or dropped.
"""

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
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
    CurationCatalog,
    CurationSummary,
    CurrentCatalog,
    SnapshotSource,
    register_catalog,
)
from finestore.rl.status_taxonomy import DataRestriction
from finestore.store import DataStore
from marina.applets import AppletServices, validate_query
from marina.client import AppletQuery
from marina.testing import test_database
from rigging.server_auth import VerifiedIdentity, identity_scope
from shellbox.machine import Backend
from shellbox.requirements import Requirement, Support
from shellbox.survey import CapabilityCell
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    FitSupport,
    backend_schema_sha256,
)
from taskcompendium.source_fingerprints import SourceIdentitySurvey

from infra.marina.apps.rl_data_catalog.load import load_catalog as load_surveyed_catalog
from infra.marina.apps.rl_data_catalog.load_curriculum import atlas_source_restrictions, load_curriculum
from infra.marina.apps.rl_data_catalog.load_rollouts import stage_rollouts
from infra.marina.apps.rl_data_catalog.server import licensing, restriction_registry
from infra.marina.apps.rl_data_catalog.server.app import create_api, migrate
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content
from infra.marina.apps.rl_data_catalog.server.restrictions import (
    declaration_restriction,
    require_public_data,
    restriction_overrides,
)
from infra.marina.apps.rl_data_catalog.tests.private_tier_fixtures import (
    FIXTURE_DATASET,
    FIXTURE_REVISION,
    OPEN_SOURCE_ID,
    PRIVATE_CANARY,
    PRIVATE_SOURCE_ID,
    curriculum_release_fixture,
    private_canary_text,
    restricted_declaration,
    write_review_rows,
)

OPEN_NOTES = "open-source-notes-served-while-open"
OPEN_PROMPT = "Count public objects and report the total."


def catalog_backend_map(catalog: CurationCatalog) -> BackendSurvey:
    """The unassessed backend map every fixture source in the staged catalog needs."""
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


def load_tier_catalog(catalog: CurationCatalog, query: AppletQuery) -> dict[str, Any]:
    return load_surveyed_catalog(catalog, query, catalog_backend_map(catalog), catalog_identity_map(catalog))


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with test_database() as url:
        yield url


@pytest.fixture
def catalog_database(database_url: str) -> Iterator[Engine]:
    admin = create_engine(database_url)
    schema = "private_tier_" + uuid.uuid4().hex
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


@pytest.fixture(autouse=True)
def clean_override_registry(monkeypatch):
    monkeypatch.setattr(restriction_registry, "RESTRICTION_OVERRIDES", '{"version":"test","restrictions":[]}')
    restriction_overrides.cache_clear()
    yield
    restriction_overrides.cache_clear()


def force_private(monkeypatch, source_id: str) -> None:
    """Install a registry override marking one source PRIVATE_ALWAYS."""
    monkeypatch.setattr(
        restriction_registry,
        "RESTRICTION_OVERRIDES",
        json.dumps(
            {
                "version": "test-override",
                "restrictions": [
                    {
                        "source_id": source_id,
                        "restriction": "private_always",
                        "reason": "test ruling",
                        "issue_url": "https://example.test/issue/9",
                    }
                ],
            }
        ),
    )
    restriction_overrides.cache_clear()


@dataclass(frozen=True)
class CatalogServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


def stage_private_catalog(root: str, snapshot_id: str = "private-tier-1") -> dict[str, str]:
    """Stage one catalog with a PRIVATE_ALWAYS and an open synthetic source; return pipeline keys."""
    private_notes = f"Vendor provenance notes quoting {PRIVATE_CANARY}"
    private_declaration = restricted_declaration(PRIVATE_SOURCE_ID, DataRestriction.PRIVATE_ALWAYS, notes=private_notes)
    # Reviewer prose, a task-contract example and a conversion failure reason, each
    # quoting the canary: the served projection must drop them all.
    private_declaration["quality_result"] = {
        "reviewed": 3,
        "issue_count": 1,
        "issues": [f"Reviewer noted {PRIVATE_CANARY} in the prompt"],
        "repair_recommendation": f"Drop the task quoting {PRIVATE_CANARY}",
        "verdict": "some_issues",
    }
    private_declaration["integrity_contracts"] = {
        "groups": [],
        "distinct_groups": 0,
        "absent_contract_tasks": 0,
        "required_shared_tasks": 0,
        "shared_missing_contract_tasks": 0,
        "example": {"task_id": "one", "contract_json": f"contract quoting {PRIVATE_CANARY}"},
        "evidence_url": "https://example.test/integrity",
        "dataset_id": FIXTURE_DATASET,
        "dataset_revision": FIXTURE_REVISION,
        "task_specs_sha256": "d" * 64,
    }
    private_declaration["converted_examples"] = {
        "count": 3,
        "input_rows": 3,
        "rejections": {},
        "admission_rejections": {},
        "source_dataset_id": FIXTURE_DATASET,
        "source_revision": FIXTURE_REVISION,
        "converter_revision": "1",
        "converter_version": "1",
        "evidence_url": "https://example.test/conversion",
        "failure_reason": f"Rejected rows quote {PRIVATE_CANARY}",
    }
    declarations = {
        PRIVATE_SOURCE_ID: private_declaration,
        OPEN_SOURCE_ID: restricted_declaration(OPEN_SOURCE_ID, DataRestriction.OPEN, notes=OPEN_NOTES),
    }
    private_packet = {"task": {"prompt": private_canary_text("prompt")}}
    open_packet = {"task": {"prompt": OPEN_PROMPT}}
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
    snapshot = CatalogSnapshot(snapshot_id=snapshot_id, created_at="2026-11-17T00:00:00Z", source_count=2)
    with DataStore.open(root, writer_id=snapshot_id) as store:
        register_catalog(store)
        for source_id, packet in ((PRIVATE_SOURCE_ID, private_packet), (OPEN_SOURCE_ID, open_packet)):
            private = source_id == PRIVATE_SOURCE_ID
            key = snapshot_id + (":private" if private else ":open")
            explanation = f"Reviewer flagged {PRIVATE_CANARY} inside the prompt." if private else "Clean sample finding."
            traces = (
                [{"name": f"rollout-transcript-{PRIVATE_CANARY}.json", "uri": "finestore://blobs/trace.json"}]
                if private
                else []
            )
            rows = []
            for task_id, cohort, verdict in (
                ("one", "admitted", "good"),
                ("two", "admitted", "unknown"),
                ("three", "excluded", "bad"),
            ):
                uri = store.write_object(f"{key}/{task_id}.json", json.dumps(packet).encode())
                advisories = (
                    [{"signal": "drift", "candidate_source_id": "Other:x", "reason": f"dropped: {PRIVATE_CANARY}"}]
                    if private and task_id == "two"
                    else []
                )
                rows.append(
                    AssessmentItem(
                        pipeline_key=key,
                        task_id=task_id,
                        cohort=cohort,
                        status="reviewed",
                        verdict=verdict,
                        explanation=explanation,
                        evidence_ids=["task"],
                        packet_uri=uri,
                        has_task=True,
                        omissions=[],
                        repair_advisories=advisories,
                        exclusion_reason="Reference contradiction" if cohort == "excluded" else None,
                    )
                )
            store.flush()
            assessment = Assessment(
                pipeline_key=key,
                source_id=source_id,
                scope="full",
                model={
                    "model": "glm",
                    "revision": "1",
                    "max_tokens": 4096,
                    "max_attempts": 3,
                    "retry_max_tokens": 8192,
                },
                rubric=f"Reviewer rubric quoting {PRIVATE_CANARY}" if private else "Check answerability",
                curation=curation,
                curation_uri=store.write_object(f"{key}/curation.json", curation.model_dump_json().encode()),
                traces=traces,
            )
            source = SnapshotSource(
                snapshot_id=snapshot_id,
                id=source_id,
                declaration=declarations[source_id],
                assessment_key=key,
                latest_attempt_key=key,
                latest_attempt_status="complete",
                reason=None,
                curation=curation,
            )
            with store.transaction() as tx:
                tx.table(ASSESSMENTS).add(assessment.model_dump(mode="json"))
                tx.table(ITEMS).extend(item.model_dump(mode="json") for item in rows)
                tx.table(POPULATIONS).extend(
                    AssessmentPopulation(
                        pipeline_key=key, cohort=cohort, population=count, sample_count=count
                    ).model_dump(mode="json")
                    for cohort, count in (("admitted", 2), ("excluded", 1))
                )
                tx.table(INVENTORY).add(source.model_dump(mode="json"))
        with store.transaction() as tx:
            tx.table(SNAPSHOTS).add(snapshot.model_dump())
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())
    return {
        "snapshot_id": snapshot_id,
        "private_pipeline": snapshot_id + ":private",
        "open_pipeline": snapshot_id + ":open",
    }


def insert_legacy_audit_rows(engine: Engine, load_id: str, pipeline_key: str, sample_id: str) -> None:
    """Legacy audit rows for the private source, as written before the tier existed."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO atlas_audit_samples (sample_id, source_id, pipeline_key, load_id, strategy, seed, "
                "sample_size, strata, created_by) VALUES (:sample_id, :source_id, :pipeline_key, :load_id, "
                "'random', 'seed-1', 2, '{}', 'auditor@example')"
            ),
            {"sample_id": sample_id, "source_id": PRIVATE_SOURCE_ID, "pipeline_key": pipeline_key, "load_id": load_id},
        )
        connection.execute(
            text(
                "INSERT INTO atlas_audit_sample_tasks (sample_id, task_id, verdict, defects) "
                "VALUES (:sample_id, 'one', 'good', '[]')"
            ),
            {"sample_id": sample_id},
        )
        connection.execute(
            text(
                "INSERT INTO atlas_audit_judgments (sample_id, task_id, user_id, agree, note, created_at, updated_at) "
                "VALUES (:sample_id, 'one', 'auditor@example', false, :note, NOW(), NOW())"
            ),
            {"sample_id": sample_id, "note": f"I disagree: the task text says {PRIVATE_CANARY}"},
        )


def canary_in_payload(payload: Any) -> bool:
    return PRIVATE_CANARY in json.dumps(payload, default=str)


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
def test_status_projection_preserves_absent_and_invalid_restriction_tags(
    catalog_database: Engine, catalog_query: AppletQuery, tmp_path: Path, present: bool, tag: Any, expected: str
):
    root = str(tmp_path / "status-tier-catalog")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.begin() as connection:
        connection.execute(
            text("UPDATE atlas_sources SET declaration=declaration-'data_restriction' WHERE id=:source"),
            {"source": OPEN_SOURCE_ID},
        )
        if present:
            connection.execute(
                text(
                    "UPDATE atlas_sources SET declaration=jsonb_set(declaration,'{data_restriction}',"
                    "CAST(:tag AS JSONB)) WHERE id=:source"
                ),
                {"source": OPEN_SOURCE_ID, "tag": json.dumps(tag)},
            )
    services = CatalogServices(catalog_database)
    with TestClient(create_api(cast(AppletServices, services))) as client:
        for path in ("/sources?projection=table", "/source-statuses"):
            response = client.get(path)
            assert response.status_code == 200
            source = next(row for row in response.json()["sources"] if row["id"] == OPEN_SOURCE_ID)
            assert source["data_restriction"] == expected
            assert source["notes"] == (OPEN_NOTES if expected == "open" else "Private data: content is not displayed.")
    with catalog_database.connect() as connection:
        stored = connection.execute(
            text("SELECT declaration ? 'data_restriction' FROM atlas_sources WHERE id=:source"),
            {"source": OPEN_SOURCE_ID},
        ).scalar_one()
        assert stored is present


def test_private_source_content_never_serves(
    catalog_database: Engine, catalog_query: AppletQuery, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Loader, API, evidence, curriculum, difficulty, archive, audit: no canary anywhere."""
    root = str(tmp_path / "private-tier-catalog")
    keys = stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)

    # The loader keeps a PRIVATE_ALWAYS source's packets out of the served database;
    # the open source's packets load normally.
    with catalog_database.connect() as connection:
        stored = {
            row["pipeline_key"]: row["count"]
            for row in connection.execute(
                text("SELECT pipeline_key, COUNT(*) AS count FROM atlas_packets GROUP BY pipeline_key")
            ).mappings()
        }
        (load_id,) = connection.execute(text("SELECT load_id FROM atlas_current WHERE name='latest'")).fetchone()
    assert stored.get(keys["private_pipeline"], 0) == 0
    assert stored.get(keys["open_pipeline"], 0) == 3

    caplog.set_level(logging.DEBUG)
    services = CatalogServices(catalog_database)
    with TestClient(create_api(cast(AppletServices, services))) as client:
        # Evidence: the private packet is the marker; the judgment text is the marker;
        # advisory reasons are redacted. The open source still serves its real content.
        private_evidence = client.get("/evidence", params={"source_id": PRIVATE_SOURCE_ID, "task_id": "one"}).json()
        assert private_evidence["packet"] == {
            "data_restriction": "private_always",
            "private": True,
            "content": None,
            "marker": "Private data: content is not displayed.",
        }
        assert private_evidence["judgment"]["explanation"] == "Private data: content is not displayed."
        assert private_evidence["judgment"]["verdict"] == "good"
        assert not canary_in_payload(private_evidence)
        open_evidence = client.get("/evidence", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).json()
        assert open_evidence["packet"]["task"]["prompt"] == OPEN_PROMPT
        assert open_evidence["judgment"]["explanation"] == "Clean sample finding."

        # Assessment report: reviewer text redacted, rollout traces withheld for the
        # private source and served for the open one; cohort example explanations too.
        private_assessment = client.get("/assessment", params={"source_id": PRIVATE_SOURCE_ID}).json()
        assert "traces" not in private_assessment["report"]
        assert private_assessment["report"]["rubric"] == "Private data: content is not displayed."
        items = private_assessment["report"]["evidence"]["items"]
        redacted = {item["task_id"]: item for item in items}
        assert redacted["one"]["explanation"] == "Private data: content is not displayed."
        assert redacted["two"]["repair_advisories"][0]["reason"] == "Private data: content is not displayed."
        assert redacted["one"]["verdict"] == "good"
        assert not canary_in_payload(private_assessment)
        open_assessment = client.get("/assessment", params={"source_id": OPEN_SOURCE_ID}).json()
        assert open_assessment["report"]["traces"] == []
        assert open_assessment["report"]["evidence"]["items"][0]["explanation"] == "Clean sample finding."

        # Source evidence page: private declaration keeps receipts and counts, loses
        # prose; data_restriction is served for both tiers.
        private_source = client.get("/source", params={"source_id": PRIVATE_SOURCE_ID}).json()
        declaration = private_source["declaration"]
        assert declaration["data_restriction"] == "private_always"
        assert declaration["notes"] == "Private data: content is not displayed."
        assert declaration["task_count"] == 3
        assert declaration["license_receipt"]["dataset_id"] == "open-athena/private-tier-fixture"
        assert "traces" not in private_source["assessment"]
        assert private_source["assessment"]["rubric"] == "Private data: content is not displayed."
        assert not canary_in_payload(private_source)
        open_source = client.get("/source", params={"source_id": OPEN_SOURCE_ID}).json()
        assert open_source["declaration"]["data_restriction"] == "open"
        assert open_source["declaration"]["notes"] == OPEN_NOTES

        # Catalog table: the restriction key is on every row, private prose is gone.
        inventory = client.get("/sources", params={"projection": "table"}).json()["sources"]
        by_id = {row["id"]: row for row in inventory}
        assert by_id[PRIVATE_SOURCE_ID]["data_restriction"] == "private_always"
        assert by_id[OPEN_SOURCE_ID]["data_restriction"] == "open"
        assert by_id[PRIVATE_SOURCE_ID]["notes"] == "Private data: content is not displayed."
        assert not canary_in_payload(inventory)

        # Status page: private notes never render.
        statuses = client.get("/source-statuses").json()["sources"]
        private_status = next(row for row in statuses if row["id"] == PRIVATE_SOURCE_ID)
        assert private_status["notes"] == "Private data: content is not displayed."
        assert not canary_in_payload(statuses)

        # Archived reviews: the private source's collection and artifacts are markers.
        with catalog_database.begin() as connection:
            write_review_rows(
                connection,
                PRIVATE_SOURCE_ID,
                "review-private-1",
                {"reviews": [{"review": f"quotes {PRIVATE_CANARY}"}]},
                f"artifact text quoting {PRIVATE_CANARY}",
            )
            write_review_rows(
                connection,
                OPEN_SOURCE_ID,
                "review-open-1",
                {"reviews": [{"review": "open review text"}]},
                "open artifact text",
            )
        private_review = client.get("/archive/reviews/review-private-1").json()
        assert private_review["data_restriction"] == "private_always"
        assert private_review["collection"]["marker"] == "Private data: content is not displayed."
        assert not canary_in_payload(private_review)
        private_artifact = client.get("/archive/reviews/review-private-1/artifacts/review/notes.txt")
        assert "Private data: content is not displayed." in private_artifact.text
        assert PRIVATE_CANARY not in private_artifact.text
        assert private_artifact.headers["x-data-restriction"] == "private_always"
        open_artifact = client.get("/archive/reviews/review-open-1/artifacts/review/notes.txt")
        assert open_artifact.text == "open artifact text"
        assert "x-data-restriction" not in open_artifact.headers

        # Audit: creating a new sample for a private source is refused; legacy rows
        # redact reviewer text.
        with identity_scope(VerifiedIdentity(user_id="auditor@example", role="user")):
            refused = client.post(
                "/audit/samples",
                json={"source_id": PRIVATE_SOURCE_ID, "strategy": "random", "seed": "s1", "sample_size": 2},
            )
            assert refused.status_code == 403
            insert_legacy_audit_rows(catalog_database, load_id, keys["private_pipeline"], "legacy-private-sample")
            legacy = client.get("/audit/samples/legacy-private-sample/tasks").json()
            assert legacy["tasks"][0]["explanation"] == "Private data: content is not displayed."
            assert legacy["tasks"][0]["judgment"]["note"] == "Private data: content is not displayed."
            assert not canary_in_payload(legacy)

    # Difficulty: the served summary never carries per-task rows or free-text reasons
    # that could quote content; the stored measurement keeps counts only.
    with catalog_database.connect() as connection:
        measurement = connection.execute(
            text("SELECT measurement FROM atlas_difficulty WHERE source_id = :id"), {"id": PRIVATE_SOURCE_ID}
        ).scalar_one_or_none()
    assert measurement is None or not canary_in_payload(measurement)

    # Every response body and every captured log record is canary-free.
    assert PRIVATE_CANARY not in caplog.text


def test_curriculum_release_refuses_private_sources_and_tampered_tags(
    catalog_database: Engine, catalog_query: AppletQuery, tmp_path: Path, monkeypatch
):
    """A release refuses private sources; flipped and dropped tags are caught by the catalog tier."""
    root = str(tmp_path / "private-tier-catalog")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)

    open_release, open_coverage, _, _ = curriculum_release_fixture(
        tmp_path / "open-release",
        [restricted_declaration(OPEN_SOURCE_ID, DataRestriction.OPEN, notes=OPEN_NOTES)],
        instructions=OPEN_PROMPT,
    )
    load_curriculum(str(open_release), str(open_coverage), catalog_query)

    services = CatalogServices(catalog_database)
    with TestClient(create_api(cast(AppletServices, services))) as client:
        tasks = client.get("/curriculum/tasks").json()["tasks"]
        assert tasks and all(row["task_preview"] == OPEN_PROMPT for row in tasks if row["source_id"] == OPEN_SOURCE_ID)
        assert not canary_in_payload(tasks)
        detail = client.get("/curriculum/task", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).json()
        assert detail["task"]["task_preview"] == OPEN_PROMPT

    private_release_dir = tmp_path / "private-release"
    release, coverage, _, _ = curriculum_release_fixture(
        private_release_dir,
        [
            restricted_declaration(OPEN_SOURCE_ID, DataRestriction.OPEN, notes=OPEN_NOTES),
            restricted_declaration(PRIVATE_SOURCE_ID, DataRestriction.PRIVATE_ALWAYS),
        ],
    )
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS source"):
        load_curriculum(str(release), str(coverage), catalog_query)

    # Tamper 1: the release row's tag flipped to open; the catalog's authoritative
    # tier still refuses it.
    flipped = json.loads(release.read_text())
    flipped["inventory"][1]["data_restriction"] = "open"
    flipped_path = private_release_dir / "release-flipped.json"
    flipped_path.write_text(json.dumps(flipped))
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS source"):
        load_curriculum(str(flipped_path), str(coverage), catalog_query)

    # Tamper 2: the release row's tag dropped entirely; same refusal.
    dropped = json.loads(release.read_text())
    del dropped["inventory"][1]["data_restriction"]
    dropped_path = private_release_dir / "release-dropped.json"
    dropped_path.write_text(json.dumps(dropped))
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS source"):
        load_curriculum(str(dropped_path), str(coverage), catalog_query)


def test_registry_override_redacts_a_warm_table_cache(
    catalog_database: Engine, catalog_query: AppletQuery, tmp_path: Path, monkeypatch
):
    """A registry override must invalidate the /sources cache and redact immediately."""
    root = str(tmp_path / "private-tier-catalog")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)

    services = CatalogServices(catalog_database)
    with TestClient(create_api(cast(AppletServices, services))) as client:
        warm = client.get("/sources", params={"projection": "table"}).json()["sources"]
        open_row = next(row for row in warm if row["id"] == OPEN_SOURCE_ID)
        assert open_row["notes"] == OPEN_NOTES
        assert open_row["data_restriction"] == "open"

        force_private(monkeypatch, OPEN_SOURCE_ID)
        redacted = client.get("/sources", params={"projection": "table"}).json()["sources"]
        redacted_row = next(row for row in redacted if row["id"] == OPEN_SOURCE_ID)
        assert redacted_row["data_restriction"] == "private_always"
        assert redacted_row["notes"] == "Private data: content is not displayed."
        evidence = client.get("/evidence", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).json()
        assert evidence["packet"]["private"] is True
        assert evidence["packet"]["content"] is None
        assert "task" not in evidence["packet"]


def test_publication_guard_refuses_the_private_fixture():
    """The publication guard refuses the PRIVATE_ALWAYS fixture outright."""
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        require_public_data(DataRestriction.PRIVATE_ALWAYS, PRIVATE_SOURCE_ID, "hf-dataset")


def test_private_rollout_routes_never_read_attempt_content(catalog_database, catalog_query, tmp_path):
    root = str(tmp_path / "private-rollout")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)

    event.listen(catalog_database, "before_cursor_execute", reject_content_read)
    try:
        with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
            for endpoint, parameters in [
                ("/difficulty-attempts", {"source_id": PRIVATE_SOURCE_ID, "measurement_id": "unknown"}),
                (
                    "/difficulty-rollout",
                    {"source_id": PRIVATE_SOURCE_ID, "measurement_id": "unknown", "attempt_id": "unknown"},
                ),
            ]:
                response = client.get(endpoint, params=parameters)
                assert response.status_code == 200
                assert response.json() == {
                    "data_restriction": "private_always",
                    "private": True,
                    "content": None,
                    "marker": "Private data: content is not displayed.",
                }
                assert not canary_in_payload(response.json())
    finally:
        event.remove(catalog_database, "before_cursor_execute", reject_content_read)


def test_license_ban_redacts_existing_content_and_invalidates_inventory_cache(
    catalog_database, catalog_query, tmp_path, monkeypatch
):
    root = str(tmp_path / "license-ban-catalog")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)
    canary = "LICENSE-BANNED-CONTENT-CANARY-20261119"
    measurement_id = "banned-measurement"
    release, coverage, _, _ = curriculum_release_fixture(
        tmp_path / "license-open-release",
        [restricted_declaration(OPEN_SOURCE_ID, DataRestriction.OPEN, notes=OPEN_NOTES)],
        instructions=canary,
    )
    load_curriculum(str(release), str(coverage), catalog_query)
    with catalog_database.begin() as connection:
        load_id = connection.execute(text("SELECT load_id FROM atlas_current WHERE name='latest'")).scalar_one()
        connection.execute(
            text("UPDATE curriculum_sources SET declaration=declaration-'id' WHERE source_id=:source"),
            {"source": OPEN_SOURCE_ID},
        )
        capabilities = {
            "source_id": OPEN_SOURCE_ID,
            "basis": "converted_spec_census",
            "task_count": 3,
            "backends": {
                "daytona": {
                    "support": "unsupported",
                    "roles": {"agent": {"unsupported": 3}, "grader": {"role_absent": 3}},
                    "task_counts": {"unsupported": 3},
                    "blockers": {"agent:non_root_user": 3},
                    "not_applicable": {},
                    "validation_errors": {canary: 3},
                },
            },
        }
        connection.execute(
            text(
                "INSERT INTO atlas_backend_sources VALUES (:load,:source,CAST(:caps AS JSONB)) "
                "ON CONFLICT (load_id,source_id) DO UPDATE SET capabilities=EXCLUDED.capabilities"
            ),
            {"load": load_id, "source": OPEN_SOURCE_ID, "caps": json.dumps(capabilities)},
        )
        connection.execute(
            text(
                "UPDATE atlas_sources SET source_fingerprint=CAST(:fingerprint AS JSONB) "
                "WHERE load_id=:load AND id=:source"
            ),
            {
                "load": load_id,
                "source": OPEN_SOURCE_ID,
                "fingerprint": json.dumps({"task_count": 3, "sample": [{"instruction": canary}]}),
            },
        )
        connection.execute(
            text(
                "INSERT INTO atlas_rollout_attempts VALUES (:measurement,'attempt','one', '{}'::jsonb, "
                "CAST(:detail AS JSONB))"
            ),
            {
                "measurement": measurement_id,
                "detail": json.dumps(pack_content({"conversation": canary, "verifier": canary})),
            },
        )
        connection.execute(
            text("INSERT INTO catalog_reviews VALUES ('license-review',:source,CAST(:collection AS JSONB),now())"),
            {"source": OPEN_SOURCE_ID, "collection": json.dumps({"reviews": [{"prompt": canary}]})},
        )
        connection.execute(
            text("INSERT INTO review_artifacts VALUES ('license-review','raw.txt',:content,:sha)"),
            {"content": canary, "sha": "f" * 64},
        )
    services = cast(AppletServices, CatalogServices(catalog_database))
    with TestClient(create_api(services)) as client:
        # Existing content is genuinely served before the ban, including the
        # cached inventory fingerprint and the historical curriculum sample.
        assert client.get("/sources").status_code == 200
        assert canary in client.get("/source-fingerprint", params={"source_id": OPEN_SOURCE_ID}).text
        assert canary in client.get("/curriculum/tasks").text
        assert canary in client.get("/curriculum/task", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).text
        assert canary in client.get("/archive/reviews/license-review/artifacts/raw.txt").text
        registry = json.loads(licensing.LICENSE_BANS)
        ban = {
            **registry["bans"][0],
            "source_id": OPEN_SOURCE_ID,
            "upstream_id": FIXTURE_DATASET,
            "upstream_revision": FIXTURE_REVISION,
            "reason": "Recorded license exclusion",
        }
        monkeypatch.setattr(licensing, "LICENSE_BANS", json.dumps({"version": "license-ban-test-v1", "bans": [ban]}))
        licensing.license_bans.cache_clear()
        try:
            for endpoint, parameters in [
                ("/sources", {}),
                ("/source-statuses", {}),
                ("/source", {"source_id": OPEN_SOURCE_ID}),
                ("/assessment", {"source_id": OPEN_SOURCE_ID}),
                ("/evidence", {"source_id": OPEN_SOURCE_ID, "task_id": "one"}),
                ("/source-fingerprint", {"source_id": OPEN_SOURCE_ID}),
                ("/curriculum/tasks", {}),
                ("/curriculum/task", {"source_id": OPEN_SOURCE_ID, "task_id": "one"}),
                ("/archive/reviews/license-review", {}),
                ("/archive/reviews/license-review/artifacts/raw.txt", {}),
            ]:
                response = client.get(endpoint, params=parameters)
                assert response.status_code == 200, (endpoint, response.text)
                assert canary not in response.text, endpoint
                assert OPEN_PROMPT not in response.text, endpoint
            metadata = client.get("/source", params={"source_id": OPEN_SOURCE_ID}).json()["backend_capabilities"]
            assert metadata["task_count"] == 3 and metadata["basis"] == "converted_spec_census"
            fit = metadata["backends"]["daytona"]
            assert fit["roles"] == capabilities["backends"]["daytona"]["roles"]
            assert list(fit["validation_errors"].values()) == [3]
            assert all(key.startswith("withheld:") for key in fit["validation_errors"])
            evidence = client.get("/evidence", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).json()
            assert evidence["packet"]["content"] is None
            assert evidence["packet"]["content_withheld"] is True
            assert evidence["packet"]["data_restriction"] == "open"
            detail = client.get("/curriculum/task", params={"source_id": OPEN_SOURCE_ID, "task_id": "one"}).json()
            assert detail["task"]["content_withheld"] is True

            event.listen(catalog_database, "before_cursor_execute", reject_content_read)
            try:
                for endpoint, parameters in [
                    ("/difficulty-attempts", {"source_id": OPEN_SOURCE_ID, "measurement_id": measurement_id}),
                    (
                        "/difficulty-rollout",
                        {"source_id": OPEN_SOURCE_ID, "measurement_id": measurement_id, "attempt_id": "attempt"},
                    ),
                ]:
                    response = client.get(endpoint, params=parameters)
                    assert response.status_code == 200
                    assert response.json()["content_withheld"] is True
                    assert response.json()["content"] is None
                    assert canary not in response.text
            finally:
                event.remove(catalog_database, "before_cursor_execute", reject_content_read)
            with pytest.raises(ValueError):
                stage_rollouts({}, restricted_declaration(OPEN_SOURCE_ID, DataRestriction.OPEN), 1)
        finally:
            licensing.license_bans.cache_clear()


@pytest.mark.parametrize("ruling", ["decision", "exclusion"])
def test_explicit_license_ruling_withholds_attempts(catalog_database, catalog_query, tmp_path, ruling):
    root = str(tmp_path / "license-ruling-catalog")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)
    with catalog_database.begin() as connection:
        declaration = connection.execute(
            text("SELECT declaration FROM atlas_sources WHERE id=:source"), {"source": OPEN_SOURCE_ID}
        ).scalar_one()
        if ruling == "decision":
            declaration["license_receipt"]["decisions"] = [
                {
                    "kind": "ban",
                    "approved": False,
                    "who": "fixture",
                    "when": "2026-11-19T15:00:00Z",
                    "source_id": OPEN_SOURCE_ID,
                    "dataset_id": FIXTURE_DATASET,
                    "revision": FIXTURE_REVISION,
                    "scope": ["train"],
                    "rationale": "Recorded license exclusion",
                    "evidence_urls": ["https://example.org/license"],
                    "issue_url": "https://github.com/marin-community/marin/issues/11497",
                }
            ]
        else:
            declaration["exclusion_claims"] = [{"reason": "license-ban"}]
        connection.execute(
            text("UPDATE atlas_sources SET declaration=CAST(:declaration AS JSONB) WHERE id=:source"),
            {"source": OPEN_SOURCE_ID, "declaration": json.dumps(declaration)},
        )
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        response = client.get(
            "/difficulty-rollout",
            params={"source_id": OPEN_SOURCE_ID, "measurement_id": "not-read", "attempt_id": "not-read"},
        )
        assert response.status_code == 200
        assert response.json()["content"] is None
        assert response.json()["content_withheld"] is True


def reject_content_read(connection, cursor, statement, parameters, context, executemany):
    if "atlas_rollout" in statement or "FROM atlas_difficulty" in statement:
        raise AssertionError("Withheld rollout route read task-level content")


@pytest.mark.parametrize("tag", ["absent", None, "private_always"])
def test_curriculum_policy_read_is_compact_and_preserves_fail_closed_tags(catalog_query, tmp_path, tag):
    root = str(tmp_path / "compact-curriculum-policy")
    stage_private_catalog(root)
    load_tier_catalog(CurationCatalog(root), catalog_query)
    load_id = catalog_query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"][0]["load_id"]
    catalog_query(
        "UPDATE atlas_sources SET declaration=jsonb_set(declaration,'{notes}',to_jsonb(repeat('x',6000000))) "
        "WHERE load_id=:load AND id=:source",
        {"load": load_id, "source": OPEN_SOURCE_ID},
    )
    if tag == "absent":
        catalog_query(
            "UPDATE atlas_sources SET declaration=declaration-'data_restriction' WHERE load_id=:load AND id=:source",
            {"load": load_id, "source": OPEN_SOURCE_ID},
        )
    else:
        catalog_query(
            "UPDATE atlas_sources SET declaration=jsonb_set(declaration,'{data_restriction}',CAST(:tag AS JSONB)) "
            "WHERE load_id=:load AND id=:source",
            {"load": load_id, "source": OPEN_SOURCE_ID, "tag": json.dumps(tag)},
        )
    policies = atlas_source_restrictions(catalog_query, load_id)
    assert len(json.dumps(policies).encode()) < 1024
    expected = DataRestriction.OPEN if tag == "absent" else DataRestriction.PRIVATE_ALWAYS
    assert declaration_restriction(policies[OPEN_SOURCE_ID]) is expected
    assert ("data_restriction" in policies[OPEN_SOURCE_ID]) is (tag != "absent")
