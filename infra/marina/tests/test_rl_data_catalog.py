# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
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
    SnapshotSource,
    SourceDeclaration,
    register_catalog,
)
from finestore.store import DataStore
from marina.applets import AppletServices, validate_query
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage

from infra.marina.applets.rl_data_catalog.load import Query, load_catalog
from infra.marina.applets.rl_data_catalog.load_curriculum import load_curriculum
from infra.marina.applets.rl_data_catalog.server.app import create_api, migrate


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
def catalog_query(catalog_database: Engine) -> Query:
    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        validate_query(sql)
        with catalog_database.begin() as connection:
            result = connection.execute(text(sql), parameters)
            return {
                "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                "row_count": result.rowcount,
            }

    return query


def publish_sample(root: str, snapshot_id: str, *, missing_packet: bool = False, source_id: str = "math") -> None:
    declaration = SourceDeclaration(
        id=source_id,
        name="math",
        display_name="Math",
        origin="native",
        family="math",
        tags=[],
        task_count=3,
        notes="",
        dataset_id=None,
        dataset_revision=None,
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline=None,
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
            assessment_key=key,
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
            origin="fixture",
            family="Numbers",
            tags=[],
            task_count=None,
            notes="",
            dataset_id=None,
            dataset_revision=None,
            url=None,
            type="RLVR",
            turns="Single-turn",
            is_benchmark=False,
            kind="Dataset",
            status="Available",
            pipeline=None,
        )
        for identifier in ("a", "b", "inventory")
    ]
    records = []
    for source_id, task_id, content, capability_score in (
        ("a", "one", "h1", 0.9),
        ("a", "two", "h2", 0.65),
        ("a", "three", "h3", None),
        ("b", "one", "h1", 0.9),
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
                "task_count": count,
                "ready_count": ready,
                "insufficient_context_count": count - ready if unavailable_status == "insufficient_context" else 0,
            }
            for identifier, count, ready in (("a", 300, 200), ("b", 100, 100))
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
    for source, denominator, assigned, subject_only, unavailable in (("a", 300, 100, 100, 100), ("b", 100, 100, 0, 0)):
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
    monkeypatch.setattr("infra.marina.applets.rl_data_catalog.load.MAX_BATCH_BYTES", 2500)
    monkeypatch.setattr("infra.marina.applets.rl_data_catalog.load_curriculum.DELETE_BATCH_SIZE", 1)
    engine = catalog_database
    query = catalog_query

    path = curriculum_release(tmp_path, unavailable_status)
    coverage_path = tmp_path / "coverage.json"
    assert load_curriculum(str(path), str(coverage_path), query)["counts"]["tasks"] == 4
    assert load_curriculum(str(path), str(coverage_path), query)["changed"] is False
    catalog_root = str(tmp_path / "catalog")
    publish_sample(catalog_root, "graded", source_id="a")
    load_catalog(CurationCatalog(catalog_root), query)
    with TestClient(create_api(cast(AppletServices, CatalogServices(engine)))) as client:
        source = client.get("/sources").json()["sources"][0]
        assert source["task_count"] == 3
        assert source["coverage"] == {
            "example_count": 300,
            "intended_use": "train",
            "top_area": {"subject_id": "D01", "node_id": "subject:D01", "name": "Mathematics", "instances": 200},
        }
        assert source["assessment"]["admitted"]["grade"] == "unknown"
        first = client.get("/curriculum").json()
        assert first["totals"] == {
            "denominator": 400,
            "assigned": 200,
            "subject_only": 100,
            "out_of_scope": 0,
            "insufficient_context": 100 * int(unavailable_status == "insufficient_context"),
            "skipped": 100 * int(unavailable_status == "skipped"),
        }
        assert {row["source_id"] for row in first["sources"]} == {"a", "b", "inventory"}
        assert next(row for row in first["sources"] if row["source_id"] == "inventory")["alignment"] is None
        assert {(row["node_id"], row["count"]) for row in first["area_counts"]} == {
            ("subject:D01", 300),
            ("arithmetic", 200),
            ("add", 200),
        }
        assert first["examples"]["sample_task_count"] == 4
        default = client.get("/curriculum", params={"subject_id": "D01"}).json()
        assert default["scope_summary"] == {"subject_members": 300, "capability_matches": 200}
        source_view = client.get("/curriculum", params={"source_id": "a"}).json()
        assert source_view["totals"]["denominator"] == 300
        assert {(row["node_id"], row["count"]) for row in source_view["area_counts"]} == {
            ("subject:D01", 200),
            ("arithmetic", 100),
            ("add", 100),
        }
        assert client.get("/curriculum", params={"intended_use": "eval"}).json()["totals"]["denominator"] == 0
        source_tasks = client.get("/curriculum/tasks", params={"source_id": "b"}).json()
        assert [(row["source_id"], row["task_id"]) for row in source_tasks["tasks"]] == [("b", "one")]
        diverse = client.get("/curriculum/tasks", params={"limit": 2}).json()
        remainder = client.get("/curriculum/tasks", params={"limit": 2, "cursor": diverse["next_cursor"]}).json()
        assert [(row["source_id"], row["task_id"]) for row in diverse["tasks"] + remainder["tasks"]] == [
            ("a", "one"),
            ("b", "one"),
            ("a", "three"),
            ("a", "two"),
        ]
        assert not remainder["next_cursor"]
        params = {"node_id": "add", "subject_id": "D01", "limit": 1}
        page = client.get("/curriculum/tasks", params=params).json()
        next_page = client.get("/curriculum/tasks", params={**params, "cursor": page["next_cursor"]}).json()
        assert [(row["source_id"], row["task_id"]) for row in page["tasks"] + next_page["tasks"]] == [
            ("a", "one"),
            ("b", "one"),
        ]
        assert not next_page["next_cursor"]
        assert client.get("/curriculum/task", params={"source_id": "a", "task_id": "one"}).json()["task"]["task_preview"]
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
    assert loaded["counts"] == {"sources": 1, "assessments": 1, "populations": 2, "items": 3, "packets": 3}
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
        assert assessment["admitted"]["grade"] == "unknown"
        assert assessment["admitted"]["verdict_counts"] == {"good": 1, "some_issues": 0, "bad": 0, "unknown": 1}
        assert assessment["excluded"]["grade"] == "bad"
        report = client.get("/assessment", params={"source_id": "math"}).json()["report"]
        assert report["assessment"]["admitted"]["missing_evidence"] == 1
        assert report["evidence"]["samples"][2]["exclusion_reason"] == "Reference contradiction"
        packet = client.get("/evidence", params={"source_id": "math", "task_id": "one"}).json()
        assert packet["packet"]["task"]["prompt"] == "Null: \u0000; Unicode: π"
        assert client.get("/evidence", params={"source_id": "math", "task_id": "other"}).status_code == 404
        assert client.get("/assessment", params={"source_id": "absent"}).status_code == 404
        assert client.get("/archive").json()["reviews"][0]["id"] == "legacy"
        assert client.get("/archive/reviews/legacy").json()["collection"]["quality"] == "good"
        assert client.get("/archive/reviews/legacy/artifacts/nested/trace.txt").text == "saved trace"
        publish_sample(root, "broken", missing_packet=True)
        with pytest.raises(FileNotFoundError):
            load_catalog(CurationCatalog(root), query)
        assert client.get("/sources").json()["snapshot_id"] == "first"
        assert client.get("/evidence", params={"source_id": "math", "task_id": "one"}).json() == packet
        publish_sample(root, "next")
        assert load_catalog(CurationCatalog(root), query)["changed"] is True
        shutil.rmtree(root)
        assert client.get("/sources").json()["snapshot_id"] == "next"
        assert client.get("/evidence", params={"source_id": "math", "task_id": "one"}).json() == packet
        assert client.get("/archive/reviews/legacy/artifacts/nested/trace.txt").text == "saved trace"


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
    assert catalog_query(
        "SELECT l.snapshot_id, count(p.*) AS packets FROM atlas_current c JOIN atlas_loads l USING(load_id) "
        "JOIN atlas_packets p USING(load_id) GROUP BY l.snapshot_id",
        {},
    )["rows"] == [{"snapshot_id": "later", "packets": 3}]


@pytest.mark.parametrize(
    "verdicts, curation_status, expected",
    [
        (["good"] * 99 + ["bad"], "completed", "good"),
        (["good"] * 80 + ["bad"] * 20, "completed", "good"),
        (["good"] * 79 + ["bad"] * 21, "completed", "bad"),
        (["good"] * 79 + ["some_issues"] * 21, "completed", "some_issues"),
        (["good"] * 80 + ["unknown"] * 20, "completed", "good"),
        (["good"] * 79 + ["unknown"] * 21, "completed", "unknown"),
        (["good", "missing_task"], "completed", "unknown"),
        ([], "completed", "bad"),
        ([], "gated", "bad"),
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
        row = connection.execute(text("SELECT grade, good_fraction FROM atlas_cohort_summary")).mappings().one()
    assert row["grade"] == expected
    if verdicts == ["good"] * 99 + ["bad"]:
        assert row["good_fraction"] == 0.99
