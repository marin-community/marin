# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins
import hashlib
import importlib
import json
import shutil
import uuid
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
    OversightLevel,
    SnapshotSource,
    SourceDeclaration,
    SourceOrigin,
    register_catalog,
)
from finestore.rl.source_identity import canonical_source_id
from finestore.store import DataStore
from marina.applets import (
    AppletServices,
    load_backend_entrypoint,
    package_applet,
    read_applet_package,
    remove_backend_revision,
    validate_query,
)
from marina.client import AppletQuery
from shellbox.machine import Backend
from shellbox.requirements import Requirement, Support
from shellbox.survey import CapabilityCell
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from taskcompendium.backend_requirements import BackendSourceMap, BackendSurvey, backend_schema_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage
from taskcompendium.source_fingerprints import SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.datasets.admission_censuses import ADMISSION_CENSUS_RECEIPTS
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.tests.admission_fixtures import (
    isolated_admission_census as isolated_admission_census,
)
from experiments.post_training.task_curation.tests.admission_fixtures import register_admission_census
from infra.marina.apps.rl_data_catalog.load import load_catalog as load_surveyed_catalog
from infra.marina.apps.rl_data_catalog.load_curriculum import load_curriculum
from infra.marina.apps.rl_data_catalog.prepare_preview import PREVIEW_APP_ID, prepare_preview
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference
from infra.marina.apps.rl_data_catalog.server.app import create_api, migrate
from infra.marina.apps.rl_data_catalog.server.quality import result
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

pytestmark = pytest.mark.usefixtures("isolated_admission_census")


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
            for backend in [*(backend.value for backend in Backend), "iris"]
            for need in Requirement
        ],
        sources=[
            BackendSourceMap(
                source_id=source.id,
                basis="unassessed",
                task_count=0,
                inputs={},
                demands={},
                resource_ranges={},
                backends={},
                role_backend_assignments={},
                grading_environment=None,
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
        assert client.get("/sources").json()["sources"][0]["backend_capabilities"]["basis"] == "unassessed"
    refreshed = load_surveyed_catalog(catalog, catalog_query, second, identities)
    assert refreshed["load_id"] != loaded["load_id"]
    assert not load_surveyed_catalog(catalog, catalog_query, second, identities)["changed"]
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/backends").json()["generated_at"] == second.generated_at


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


def publish_sample(
    root: str,
    snapshot_id: str,
    *,
    missing_packet: bool = False,
    source_id: str = "Marin:math",
    origin: SourceOrigin = SourceOrigin.MARIN,
    duplicates: int = 0,
    unassessed_oversight: OversightLevel | None = None,
) -> None:
    declaration = SourceDeclaration(
        id=source_id,
        name="Marin:math",
        display_name="Math",
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
        family="Marin:math",
        tags=[],
        task_count=3 - duplicates,
        within_source_dedup=deduplication_evidence("fixture", "1", "1", 3).model_copy(
            update={"unique_count": 3 - duplicates, "within_source_duplicates": duplicates}
        ),
        notes="",
        dataset_id="fixture",
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        oversight=unassessed_oversight or OversightLevel.NONE,
        actor_census_input=register_admission_census(source_id, "fixture", "1", "1", 3 - duplicates),
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
            ("one", "admitted", "good", True),
            ("two", "admitted", "unknown", False),
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
            name=identifier,
            display_name=identifier.upper(),
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
            within_source_dedup=deduplication_evidence("fixture", "1", "1"),
            notes="",
            dataset_id="fixture",
            dataset_revision="1",
            url=None,
            type="RLVR",
            turns="Single-turn",
            is_benchmark=False,
            kind="Dataset",
            status="Available",
            pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
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


@pytest.mark.parametrize("unavailable_status", ["insufficient_context", "skipped"])
def test_curriculum_precomputed_population_samples_pagination_and_atomic_load(
    catalog_database, catalog_query, tmp_path, monkeypatch, unavailable_status
):
    monkeypatch.setattr("infra.marina.apps.rl_data_catalog.load.MAX_BATCH_BYTES", 3500)
    monkeypatch.setattr("infra.marina.apps.rl_data_catalog.load_curriculum.DELETE_BATCH_SIZE", 1)
    engine = catalog_database
    query = catalog_query

    path = curriculum_release(tmp_path, unavailable_status)
    coverage_path = tmp_path / "coverage.json"
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
            VALUES ('legacy','math','{"quality":"good","reviews":[]}'::jsonb,NOW())
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
        assert report["evidence"]["samples"][2]["exclusion_reason"] == "Reference contradiction"
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


@pytest.mark.parametrize("failure", ["missing", "stale", "no_grader"])
def test_atlas_blocks_invalid_admission_census_without_replacing_selected_catalog(
    catalog_database, catalog_query, tmp_path, failure
):
    root = str(tmp_path / "catalog")
    publish_sample(root, "reviewed")
    load_catalog(CurationCatalog(root), catalog_query)
    publish_sample(root, "next")
    source_id = "Marin:math"
    receipt = ADMISSION_CENSUS_RECEIPTS[source_id]
    if failure == "missing":
        del ADMISSION_CENSUS_RECEIPTS[source_id]
    elif failure == "stale":
        ADMISSION_CENSUS_RECEIPTS[source_id] = receipt.model_copy(
            update={"input": receipt.input.model_copy(update={"dataset_revision": "stale"})}
        )
    else:
        ADMISSION_CENSUS_RECEIPTS[source_id] = type(receipt).model_validate(
            {
                **receipt.model_dump(),
                "runnable_grader_tasks": 2,
                "no_grader_tasks": 1,
                "grader_kinds": {"verifyit": 2, "none": 1},
                "missing_grader_reasons": {"No reward model": 1},
            }
        )
    with pytest.raises(ValueError, match=r"Admission census|without runnable graders"):
        load_catalog(CurationCatalog(root), catalog_query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        assert client.get("/sources").json()["snapshot_id"] == "reviewed"


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


def test_atlas_cleanup_preserves_a_load_started_after_its_cutover(catalog_query, tmp_path):
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
            if sql.startswith("DELETE FROM atlas_sources") and later is None:
                later = executor.submit(load_catalog, CurationCatalog(roots[2]), later_query)
                assert staged.wait(timeout=15)
            return catalog_query(sql, parameters)

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
                ('load', 'pipeline', 'source', 'full', '{}'::jsonb, 'rubric',
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
    assert result({}, dict(row)) == expected.upper()
    assert row["issue_count"] == sum(v in {"bad", "some_issues"} for v in verdicts)
    assert row["result_reviewed"] == sum(v not in {"unknown", "missing_task"} for v in verdicts)
    if verdicts == ["good"] * 99 + ["bad"]:
        assert row["good_fraction"] == 0.99


def test_historical_alias_urls_keep_assessments_packets_and_sql_counts(catalog_database, catalog_query, tmp_path):
    old = "MarinSkyRL:aime24"
    current = canonical_source_id(old)
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen", source_id=current, origin=SourceOrigin.HUGGINGFACEH4)
    with DataStore.open(root) as store:
        register_catalog(store)
        record = {
            key: value
            for key, value in store.read_view().point(INVENTORY, snapshot_id="frozen", id=current).items()
            if key in SnapshotSource.model_fields
        }
        record.update(snapshot_id="legacy", id=old, declaration={**record["declaration"], "id": old})
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
    shutil.rmtree(root)
    with TestClient(create_api(cast(AppletServices, CatalogServices(catalog_database)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["id"] == source["backend_capabilities"]["source_id"] == current
        for identifier in (old, current):
            report = client.get("/assessment", params={"source_id": identifier}).json()["report"]
            assert report["source_id"] == current and report["pipeline_key"] == "frozen"
            assert len(report["evidence"]["samples"]) == 3
            packet = client.get("/evidence", params={"source_id": identifier, "task_id": "one"}).json()["packet"]
            assert packet == original_packet
            assert client.get("/source-fingerprint", params={"source_id": identifier}).json()["source_id"] == current
            detail = client.get("/source", params={"source_id": identifier}).json()
            assert detail["declaration"]["id"] == current
            assert detail["assessment"]["pipeline_key"] == "frozen"
            assert detail["backend_capabilities"]["basis"] == "unassessed"
            assert detail["result"] == "GOOD"


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


def test_frozen_backend_survey_digest_survives_canonical_source_projection(tmp_path):
    root = str(tmp_path / "catalog")
    publish_sample(root, "frozen", source_id="HuggingFaceH4:aime24", origin=SourceOrigin.HUGGINGFACEH4)
    raw = catalog_backend_map(CurationCatalog(root)).model_dump(mode="json")
    raw["sources"][0]["source_id"] = "MarinSkyRL:aime24"
    survey = BackendSurvey.model_validate(raw)
    assert survey.sources[0].source_id == "HuggingFaceH4:aime24"
    assert canonical_sha256(survey.identity_payload()) == canonical_sha256(raw)


def test_preview_package_imports_canonical_models_without_new_shared_runtime_modules(tmp_path, monkeypatch):
    checkout = Path(__file__).resolve().parents[3]
    output = tmp_path / "preview"
    prepare_preview(checkout, output)
    package = read_applet_package(package_applet(output))
    assert package.manifest.applet_id == PREVIEW_APP_ID
    original_import = builtins.__import__

    def old_runtime_import(name, global_namespace=None, local_namespace=None, fromlist=(), level=0):
        if level == 0 and name in {
            "finestore.rl.curation",
            "finestore.rl.source_identity",
            "taskcompendium.source_fingerprints",
        }:
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
        {**{key: raw_coverage[key] for key in identity_keys}, "version": COVERAGE_VERSION}
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
        {**{key: corrected_coverage[key] for key in identity_keys}, "version": COVERAGE_VERSION}
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
                "dataset_id": "fixture",
                "dataset_revision": "1",
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
                text("UPDATE atlas_sources SET declaration=jsonb_set(declaration, '{is_benchmark}', 'true')")
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
