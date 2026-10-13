# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
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
    ITEMS,
    POPULATIONS,
    SNAPSHOTS,
    SOURCES,
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
from marina.applets import AppletServices
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from infra.marina.applets.rl_data_catalog.load import load_catalog
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


def publish_sample(root: str, snapshot_id: str, *, missing_packet: bool = False) -> None:
    declaration = SourceDeclaration(
        id="math",
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
        cur_uri = store.write_object(f"{key}/curation.json", curation.model_dump_json().encode())
        store.flush()
        assessment = Assessment(
            pipeline_key=key,
            source_id="math",
            scope="full",
            model={"model": "glm", "revision": "1", "max_tokens": 4096, "max_attempts": 3, "retry_max_tokens": 8192},
            rubric="Check answerability",
            curation=curation,
            curation_uri=cur_uri,
            traces=[],
        )
        snapshot = CatalogSnapshot(snapshot_id=snapshot_id, created_at="2026-10-13T00:00:00Z", source_count=1)
        source = SnapshotSource(
            snapshot_id=snapshot_id,
            id="math",
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
            tx.table(SOURCES).add(source.model_dump(mode="json"))
            tx.table(SNAPSHOTS).add(snapshot.model_dump())
            tx.table(CURRENT).add(CurrentCatalog(name="latest", snapshot_id=snapshot_id).model_dump())


@dataclass(frozen=True)
class CatalogServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


def test_atlas_load_serves_grades_packets_and_archive_without_object_storage(catalog_database, tmp_path: Path) -> None:
    engine = catalog_database
    root = str(tmp_path / "catalog")
    publish_sample(root, "first")

    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as connection:
            result = connection.execute(text(sql), parameters)
            return {
                "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                "row_count": result.rowcount,
            }

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
        assert client.post("/refresh").status_code == 404
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
