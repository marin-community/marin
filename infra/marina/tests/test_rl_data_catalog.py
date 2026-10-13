# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from finestore.store import DataStore
from marina.applets import AppletServices
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from infra.marina.applets.rl_data_catalog.server.app import (
    create_api,
    initialize_catalog,
    save_catalog,
    saved_catalog,
    source_record,
)


def catalog_document(root: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    revision = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 2,
        "snapshot_id": "fixture",
        "created_at": "2026-10-12T00:00:00Z",
        "revision": revision,
        "catalog_root": root,
        "sources": rows,
    }


@pytest.fixture
def catalog_database(database_url: str, tmp_path: Path) -> Iterator[tuple[Engine, Path]]:
    admin = create_engine(database_url)
    schema = "catalog_test_" + uuid.uuid4().hex
    with admin.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog_document(str(tmp_path / "evidence"), [])))
    try:
        with engine.begin() as connection:
            initialize_catalog(connection)
            save_catalog(connection, json.loads(path.read_text()))
        yield engine, path
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def test_catalog_mirror_keeps_legacy_reviews_separate(catalog_database) -> None:
    engine, _path = catalog_database
    source = {
        "id": "native:math",
        "origin": "native",
        "display_name": "Math\u202fsource",
        "quality": "bad",
        "assessment": {"status": "complete", "admitted": {"grade": "bad", "sample_count": 1}},
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                """
            INSERT INTO catalog_reviews (id, source_id, collection, updated_at)
            VALUES ('legacy', 'native:math', '{"quality":"good","reviews":[]}'::jsonb, NOW())
        """
            )
        )
        connection.execute(
            text(
                """
            INSERT INTO review_artifacts (review_id, path, content, sha256)
            VALUES ('legacy', 'difficulty.json', 'original evidence', 'digest')
        """
            )
        )
        save_catalog(connection, catalog_document("/catalog", [source]))
        assert source_record(connection, source["id"]) == (source, "/catalog")
        assert saved_catalog(connection)["sources"] == [source]
        assert connection.execute(text("SELECT collection->>'quality' FROM catalog_reviews")).scalar_one() == "good"
        assert connection.execute(text("SELECT content FROM review_artifacts")).scalar_one() == "original evidence"
        save_catalog(connection, catalog_document("/catalog", []))
        assert saved_catalog(connection)["sources"] == []
        assert connection.execute(text("SELECT COUNT(*) FROM catalog_reviews")).scalar_one() == 1


@pytest.mark.parametrize("corruption", ["duplicate", "stale_revision"])
def test_malformed_catalog_preserves_saved_mirror(catalog_database, corruption) -> None:
    engine, _path = catalog_database
    original = catalog_document("/catalog", [{"id": "native:math", "origin": "native", "task_count": 7}])
    with engine.begin() as connection:
        save_catalog(connection, original)
        rows = original["sources"] * 2 if corruption == "duplicate" else original["sources"]
        damaged = catalog_document("/catalog", rows)
        if corruption == "stale_revision":
            damaged["sources"][0]["task_count"] = 11
        with pytest.raises(ValueError, match=r"Duplicate catalog source|Catalog revision does not match"):
            save_catalog(connection, damaged)
        assert saved_catalog(connection)["revision"] == original["revision"]


@dataclass(frozen=True)
class CatalogServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


def test_atlas_routes_serve_catalog_evidence_and_archive(catalog_database) -> None:
    engine, path = catalog_database
    root = json.loads(path.read_text())["catalog_root"]
    packet = {
        "judgment": {"task_id": "task:one", "verdict": "good", "status": "reviewed"},
        "packet": {"task": {"id": "task:one"}},
        "omissions": [],
    }
    report = {"scope": "full", "assessment": {"admitted": {"grade": "good", "sample_count": 1}}}
    with DataStore.open(root, writer_id="fixture") as store, store.transaction() as transaction:
        evidence_uri = transaction.write_object("task.json", json.dumps(packet).encode())
        report_uri = transaction.write_object("report.json", json.dumps(report).encode())
    source = {
        "id": "native:math",
        "origin": "native",
        "display_name": "Math",
        "quality": "good",
        "assessment": {
            "status": "complete",
            "admitted": {"grade": "good", "sample_count": 1},
            "report_uri": report_uri,
            "samples": [{"task_id": "task:one", "evidence_uri": evidence_uri}],
        },
    }
    with DataStore.open(root, writer_id="snapshot") as store:
        store.table(
            "catalog_current",
            primary_key=("name",),
            schema=pa.schema([(name, pa.string()) for name in ("name", "snapshot_id", "document_uri")]),
        )
        with store.transaction() as transaction:
            document_uri = transaction.write_object(
                "catalog.json", json.dumps(catalog_document(root, [source])).encode()
            )
            transaction.table("catalog_current").add(
                {"name": "latest", "snapshot_id": "fixture", "document_uri": document_uri}
            )
    with engine.begin() as connection:
        save_catalog(connection, catalog_document(root, [source]))
        connection.execute(
            text(
                """
            INSERT INTO catalog_reviews (id, source_id, collection, updated_at)
            VALUES ('legacy', 'native:math', '{"reviews":[]}'::jsonb, NOW())
        """
            )
        )
        connection.execute(
            text(
                """
            INSERT INTO review_artifacts (review_id, path, content, sha256)
            VALUES ('legacy', 'nested/trace.txt', 'saved trace', 'digest')
        """
            )
        )
    services = cast(AppletServices, CatalogServices(engine))
    with TestClient(create_api(services)) as client:
        assert client.post("/refresh").json() == {"busy": False, "changed": False, "snapshot_id": "fixture"}
        rows = client.get("/sources").json()["sources"]
        assert rows[0]["quality"] == "good" and "samples" not in rows[0]["assessment"]
        assert client.get("/assessment", params={"source_id": source["id"]}).json()["report"] == report
        assert client.get("/evidence", params={"source_id": source["id"], "task_id": "task:one"}).json() == packet
        assert client.get("/evidence", params={"source_id": source["id"], "task_id": "other"}).status_code == 404
        assert client.get("/assessment", params={"source_id": "absent"}).status_code == 404
        assert client.get("/archive").json()["reviews"][0]["id"] == "legacy"
        archived = client.get("/archive/reviews/legacy").json()
        assert archived["collection"] == {"reviews": []}
        assert archived["artifacts"] == [{"path": "nested/trace.txt", "sha256": "digest"}]
        assert client.get("/archive/reviews/legacy/artifacts/nested/trace.txt").text == "saved trace"
