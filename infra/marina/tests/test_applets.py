# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from marina.applets import AppletForbidden, AppletStore, package_applet, read_applet_package
from marina.cli import cli
from marina.database_setup import APPLET_READER_ROLE, ensure_applet_provisioning
from marina.db import UrlDatabase, grant_read
from marina.server import MarinaConfig, create_app
from marina.table_load import read_table, table_statements
from sqlalchemy import text
from starlette.testclient import TestClient

DEMO_APPLET = Path(__file__).parents[1] / "examples" / "problem-set-applet"


def applet_client(tmp_path: Path, database_url: str) -> tuple[TestClient, AppletStore]:
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    database = UrlDatabase(database_url)
    store = AppletStore(database)
    ensure_applet_provisioning(store.engine)
    store.migrate()
    config = MarinaConfig(apps_dir=apps_dir, data_root=str(tmp_path / "data"), iap_audience=None, database=database)
    return TestClient(create_app(config), client=("127.0.0.1", 40000)), store


def test_publish_demo_serves_files_schema_query_and_python_api(tmp_path: Path, database_url: str) -> None:
    client, store = applet_client(tmp_path, database_url)
    response = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET))
    assert response.status_code == 201
    published = response.json()
    applet_id = published["id"]

    listed = client.get("/api/marina/apps").json()["apps"]
    assert listed == [
        {
            "name": applet_id,
            "title": "Problem sets",
            "description": "Browse a small collection of generated arithmetic problems.",
            "path": f"/a/{applet_id}/",
            "kind": "applet",
            "published_by": "anonymous",
            "version": 1,
        }
    ]
    current = client.get(f"/a/{applet_id}/", follow_redirects=False)
    assert current.status_code == 307
    assert current.headers["location"] == f"/a/{applet_id}/v/1/"

    page = client.get(published["path"])
    assert page.status_code == 200
    assert "Problem sets" in page.text
    assert page.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert client.get(f"/a/{applet_id}/v/1/missing.js").status_code == 404
    assert client.get(f"/a/{applet_id}/v/1/problem/17").status_code == 200

    problems = client.get(f"/a/{applet_id}/v/1/api/problems")
    assert problems.status_code == 200
    assert problems.json() == [
        {"id": 1, "prompt": "12 * 7"},
        {"id": 2, "prompt": "144 ÷ 12"},
        {"id": 3, "prompt": "19 + 28"},
    ]
    assert client.get(f"/a/{applet_id}/api/revision").json() == {"version": 1}

    created = client.post(
        f"/a/{applet_id}/query",
        json={"sql": "CREATE TABLE observations (problem_id INTEGER, note TEXT)", "parameters": {}},
    )
    assert created.status_code == 200
    inserted = client.post(
        f"/a/{applet_id}/query",
        json={
            "sql": "INSERT INTO observations (problem_id, note) VALUES (:id, :note)",
            "parameters": {"id": 1, "note": "checked"},
        },
    )
    assert inserted.status_code == 200
    rows = client.post(
        f"/a/{applet_id}/v/1/query",
        json={"sql": "SELECT problem_id, note FROM observations", "parameters": {}},
    )
    assert rows.json() == {
        "columns": ["problem_id", "note"],
        "rows": [{"problem_id": 1, "note": "checked"}],
        "row_count": 1,
    }

    with store.engine.connect() as connection:
        schemas = connection.execute(
            text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'applet_%'")
        ).scalars()
        assert list(schemas) == [f"applet_{applet_id.replace('-', '')}"]
        owner = connection.execute(
            text("SELECT tableowner FROM pg_tables WHERE schemaname = :schema AND tablename = 'problems'"),
            {"schema": f"applet_{applet_id.replace('-', '')}"},
        ).scalar_one()
        assert owner == f"applet_{applet_id.replace('-', '')}"
    identity = client.post(f"/a/{applet_id}/query", json={"sql": "SELECT current_user AS role", "parameters": {}}).json()
    assert identity["rows"] == [{"role": f"applet_{applet_id.replace('-', '')}"}]
    with store.engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA shared_read"))
        connection.execute(text("CREATE TABLE shared_read.values (value INTEGER)"))
        connection.execute(text("INSERT INTO shared_read.values VALUES (17)"))
    grant_read(store.engine, "shared_read", APPLET_READER_ROLE)
    shared = client.post(
        f"/a/{applet_id}/query",
        json={"sql": "SELECT value FROM shared_read.values", "parameters": {}},
    )
    assert shared.json()["rows"] == [{"value": 17}]
    catalog = client.post(
        f"/a/{applet_id}/query",
        json={
            "sql": (
                "SELECT table_name, column_name FROM marina.applet_catalog "
                "WHERE applet_id = :id AND table_name = 'problems' ORDER BY ordinal_position"
            ),
            "parameters": {"id": applet_id},
        },
    )
    assert catalog.json()["rows"] == [
        {"table_name": "problems", "column_name": "id"},
        {"table_name": "problems", "column_name": "prompt"},
        {"table_name": "problems", "column_name": "answer"},
    ]
    assert (
        client.post(
            f"/a/{applet_id}/query",
            json={"sql": "INSERT INTO shared_read.values VALUES (18)", "parameters": {}},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/a/{applet_id}/query",
            json={"sql": "RESET ROLE", "parameters": {}},
        ).status_code
        == 400
    )
    second = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    cross_schema = f"applet_{applet_id.replace('-', '')}"
    cross_read = client.post(
        f"/a/{second['id']}/query",
        json={"sql": f"SELECT note FROM {cross_schema}.observations", "parameters": {}},
    )
    assert cross_read.json()["rows"] == [{"note": "checked"}]
    assert (
        client.post(
            f"/a/{second['id']}/query",
            json={"sql": f"DELETE FROM {cross_schema}.observations", "parameters": {}},
        ).status_code
        == 422
    )


def test_update_keeps_old_files_and_routes_each_python_revision(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    first = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    applet_id = first["id"]
    changed = tmp_path / "changed"
    shutil.copytree(DEMO_APPLET, changed)
    old_page = (changed / "dist" / "index.html").read_text()
    (changed / "dist" / "index.html").write_text(old_page.replace("Problem sets", "Revised problems"))

    second = client.post(
        f"/api/marina/applets/{applet_id}",
        params={"base_version": 1},
        content=package_applet(changed),
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert "Problem sets" in client.get(f"/a/{applet_id}/v/1/").text
    assert "Revised problems" in client.get(f"/a/{applet_id}/v/2/").text
    assert client.get(f"/a/{applet_id}/v/1/api/revision").json() == {"version": 1}
    assert client.get(f"/a/{applet_id}/v/2/api/revision").json() == {"version": 2}
    assert (
        client.post(
            f"/api/marina/applets/{applet_id}",
            params={"base_version": 1},
            content=package_applet(changed),
        ).status_code
        == 409
    )

    details = client.get(f"/api/marina/applets/{applet_id}").json()
    assert details["current_version"] == 2
    assert [revision["version"] for revision in details["versions"]] == [2, 1]
    rolled_back = client.put(
        f"/api/marina/applets/{applet_id}/current",
        json={"version": 1, "base_version": 2},
    )
    assert rolled_back.status_code == 200
    assert client.get(f"/a/{applet_id}/api/revision").json() == {"version": 1}
    assert client.delete(f"/api/marina/applets/{applet_id}").status_code == 204
    assert client.get(f"/a/{applet_id}/v/1/").status_code == 404
    assert (
        client.post(
            f"/api/marina/applets/{applet_id}",
            params={"base_version": 1},
            content=package_applet(changed),
        ).status_code
        == 404
    )
    listed_ids = {applet["name"] for applet in client.get("/api/marina/applets").json()["applets"]}
    assert applet_id not in listed_ids


def test_applet_host_exposes_only_applet_routes(tmp_path: Path, database_url: str) -> None:
    _client, store = applet_client(tmp_path, database_url)
    config = MarinaConfig(
        apps_dir=tmp_path / "apps",
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        applet_origin="https://applets.example",
    )
    client = TestClient(create_app(config), client=("127.0.0.1", 40000))
    published = client.post(
        "/api/marina/applets",
        headers={"host": "marina.example"},
        content=package_applet(DEMO_APPLET),
    ).json()
    applet_id = published["id"]
    assert published["url"] == f"https://applets.example/a/{applet_id}/v/1/"
    listed = client.get("/api/marina/applets", headers={"host": "marina.example"}).json()["applets"]
    published_entry = next(applet for applet in listed if applet["name"] == applet_id)
    assert published_entry["path"] == f"https://applets.example/a/{applet_id}/"

    redirected = client.get(f"/a/{applet_id}/v/1/", headers={"host": "marina.example"}, follow_redirects=False)
    assert redirected.status_code == 307
    assert redirected.headers["location"] == f"https://applets.example/a/{applet_id}/v/1/"
    assert client.get(f"/a/{applet_id}/v/1/", headers={"host": "applets.example"}).status_code == 200
    assert client.get("/api/marina/apps", headers={"host": "applets.example"}).status_code == 404
    assert (
        client.post(
            f"/a/{applet_id}/query",
            headers={"host": "marina.example"},
            json={"sql": "SELECT 1", "parameters": {}},
        ).status_code
        == 404
    )


def test_store_rejects_non_owner_and_allows_operator(tmp_path: Path, database_url: str) -> None:
    _client, store = applet_client(tmp_path, database_url)
    published = store.publish(package=read_applet_package(package_applet(DEMO_APPLET)), owner="owner@example")
    with pytest.raises(AppletForbidden):
        store.rollback(published.applet_id, 1, "other@example", 1)
    store.rollback(
        published.applet_id,
        1,
        "operator@example",
        1,
        frozenset({"operator@example"}),
    )


def test_publish_rejects_archive_escape(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../outside")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    response = client.post("/api/marina/applets", content=buffer.getvalue())
    assert response.status_code == 400


def test_publish_rejects_backend_that_cannot_import(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    broken = tmp_path / "broken-import"
    shutil.copytree(DEMO_APPLET, broken)
    (broken / "server" / "app.py").write_text('raise RuntimeError("broken import")\n')
    before = client.get("/api/marina/applets").json()
    response = client.post("/api/marina/applets", content=package_applet(broken))
    assert response.status_code == 400
    assert "Python backend validation failed" in response.json()["detail"]
    assert client.get("/api/marina/applets").json() == before


def test_backend_factory_failure_returns_503(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    broken = tmp_path / "broken-factory"
    shutil.copytree(DEMO_APPLET, broken)
    (broken / "server" / "app.py").write_text(
        """def create_api(_services):
    raise RuntimeError("factory failed")
"""
    )
    published = client.post("/api/marina/applets", content=package_applet(broken)).json()
    first = client.get(f"/a/{published['id']}/v/1/api/test")
    second = client.get(f"/a/{published['id']}/v/1/api/test")
    assert first.status_code == 503
    assert second.status_code == 503


def test_static_cache_encoding_and_html_fallback(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    packaged = tmp_path / "static"
    shutil.copytree(DEMO_APPLET, packaged)
    (packaged / "dist" / "sample.txt").write_bytes(b"plain")
    (packaged / "dist" / "sample.txt.gz").write_bytes(gzip.compress(b"compressed"))
    published = client.post("/api/marina/applets", content=package_applet(packaged)).json()
    prefix = f"/a/{published['id']}/v/1"

    plain = client.get(f"{prefix}/sample.txt", headers={"accept-encoding": "gzip;q=0"})
    assert plain.headers.get("content-encoding") is None
    compressed = client.get(f"{prefix}/sample.txt", headers={"accept-encoding": "gzip"})
    assert compressed.headers["content-encoding"] == "gzip"
    unchanged = client.get(
        f"{prefix}/sample.txt",
        headers={"if-none-match": plain.headers["etag"], "accept-encoding": "gzip;q=0"},
    )
    assert unchanged.status_code == 304
    assert client.get(f"{prefix}/client-route", headers={"accept": "application/json"}).status_code == 404
    assert client.get(f"{prefix}/client-route", headers={"accept": "text/html"}).status_code == 200


def test_publish_prunes_old_revisions_and_allocates_after_rollback(tmp_path: Path, database_url: str) -> None:
    _client, store = applet_client(tmp_path, database_url)
    package_dir = tmp_path / "revisions"
    shutil.copytree(DEMO_APPLET, package_dir)
    package = read_applet_package(package_applet(package_dir))
    published = store.publish(package, "owner@example")
    current = 1
    for revision in range(2, 7):
        page = (package_dir / "dist" / "index.html").read_text()
        (package_dir / "dist" / "index.html").write_text(page + f"<!-- revision {revision} -->")
        package = read_applet_package(package_applet(package_dir))
        store.publish(package, "owner@example", published.applet_id, current)
        current = revision
    assert [item.version for item in store.versions(published.applet_id)] == [6, 5, 4, 3, 2]

    store.rollback(published.applet_id, 2, "owner@example", 6)
    next_revision = store.publish(package, "owner@example", published.applet_id, 2)
    assert next_revision.version == 7
    assert [item.version for item in store.versions(published.applet_id)] == [7, 6, 5, 4, 3]


def test_table_load_statements_populate_applet_schema(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client(tmp_path, database_url)
    published = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    source = tmp_path / "observations.csv"
    source.write_text("problem_id,note\n1,checked\n3,tricky\n")
    table = read_table(source)
    schema_sql, inserts = table_statements("observations", table, replace=True)
    endpoint = f"/a/{published['id']}/query"
    for statement in schema_sql:
        assert client.post(endpoint, json={"sql": statement, "parameters": {}}).status_code == 200
    for statement, parameters in inserts:
        assert client.post(endpoint, json={"sql": statement, "parameters": parameters}).status_code == 200
    rows = client.post(
        endpoint,
        json={"sql": "SELECT problem_id, note FROM observations ORDER BY problem_id", "parameters": {}},
    )
    assert rows.json()["rows"] == [
        {"problem_id": 1, "note": "checked"},
        {"problem_id": 3, "note": "tricky"},
    ]


def test_query_limits_roll_back_and_timeout(tmp_path: Path, database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store = applet_client(tmp_path, database_url)
    published = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    endpoint = f"/a/{published['id']}/query"
    assert client.post(endpoint, content=b"{").status_code == 400
    too_many = client.post(
        endpoint,
        json={
            "sql": (
                "WITH inserted AS ("
                "INSERT INTO problems (id, prompt, answer) "
                "SELECT value + 100, 'bulk', value FROM generate_series(1, 10001) AS value RETURNING id"
                ") SELECT id FROM inserted"
            ),
            "parameters": {},
        },
    )
    assert too_many.status_code == 413
    count = client.post(endpoint, json={"sql": "SELECT count(*) AS count FROM problems", "parameters": {}})
    assert count.json()["rows"] == [{"count": 3}]

    monkeypatch.setattr("marina.applets.QUERY_TIMEOUT_MS", 20)
    timed_out = client.post(endpoint, json={"sql": "SELECT pg_sleep(0.2)", "parameters": {}})
    assert timed_out.status_code == 504


def test_failed_migration_leaves_no_applet_or_schema(tmp_path: Path, database_url: str) -> None:
    client, store = applet_client(tmp_path, database_url)
    applets_before = client.get("/api/marina/apps").json()["apps"]
    with store.engine.connect() as connection:
        schemas_before = set(
            connection.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'applet_%'")
            ).scalars()
        )
    broken = tmp_path / "broken"
    shutil.copytree(DEMO_APPLET, broken)
    (broken / "server" / "app.py").write_text(
        """from sqlalchemy import text


def migrate(connection):
    connection.execute(text("CREATE TABLE partial (id INTEGER)"))
    raise RuntimeError("migration failed")


def create_api(_services):
    raise RuntimeError("unreachable")
"""
    )
    failing = TestClient(client.app, client=("127.0.0.1", 40000), raise_server_exceptions=False)
    response = failing.post("/api/marina/applets", content=package_applet(broken))
    assert response.status_code == 500
    assert client.get("/api/marina/apps").json()["apps"] == applets_before
    with store.engine.connect() as connection:
        schemas_after = set(
            connection.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'applet_%'")
            ).scalars()
        )
        assert schemas_after == schemas_before


def test_publish_dry_run_reports_validated_package() -> None:
    result = CliRunner().invoke(cli, ["publish", str(DEMO_APPLET), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["file_count"] == 5
    assert report["files"] == [
        "applet.toml",
        "dist/app.js",
        "dist/index.html",
        "server/__init__.py",
        "server/app.py",
    ]
