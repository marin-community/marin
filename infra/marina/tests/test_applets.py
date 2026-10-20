# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import json
import shutil
import subprocess
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from click.testing import CliRunner
from marina.applets import AppletNotFound, AppletStore, applet_role, package_applet, read_applet_package
from marina.cli import cli
from marina.database_setup import APPLET_READER_ROLE, ensure_applet_provisioning
from marina.db import UrlDatabase, grant_read
from marina.releases import require_current_main
from marina.server import MarinaConfig, MarinaSurface, create_app
from marina.table_load import read_table, table_statements
from marina.testing import isolated_database
from requests import Response as RequestsResponse
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from starlette.testclient import TestClient

DEMO_APPLET = Path(__file__).parents[1] / "examples" / "problem-set-applet"


@pytest.mark.parametrize("invalid_graph", ["duplicate_id", "missing_parent", "cycle", "cooldown_mismatch"])
def test_snowball_release_rejects_invalid_lineage_and_preserves_local_output(tmp_path: Path, invalid_graph: str) -> None:
    source = Path(__file__).parents[1] / "applets" / "snowball_family_tree"
    applet = tmp_path / "snowball"
    shutil.copytree(source, applet, ignore=shutil.ignore_patterns("dist", "__pycache__"))
    valid_release = CliRunner().invoke(cli, ["release", str(applet), "--dry-run", "--json"])
    assert valid_release.exit_code == 0, valid_release.output
    graph = json.loads((applet / "graph-data.json").read_text())
    nodes = graph["nodes"]
    if invalid_graph == "duplicate_id":
        nodes.append(nodes[-1].copy())
    elif invalid_graph == "missing_parent":
        nodes[-1]["parents"] = ["missing-checkpoint"]
    elif invalid_graph == "cycle":
        nodes[-1]["parents"] = [nodes[-1]["id"]]
    else:
        nodes[-1]["lane"] = "2.7T"
    (applet / "graph-data.json").write_text(json.dumps(graph))
    (applet / "dist").mkdir()
    previous_output = applet / "dist" / "index.html"
    previous_output.write_text("previous local output")

    result = CliRunner().invoke(cli, ["release", str(applet), "--dry-run", "--json"])

    assert result.exit_code != 0
    assert previous_output.read_text() == "previous local output"


@pytest.fixture(params=["static", "build"])
def release_checkout(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    repo = tmp_path / "checkout"
    app = repo / "demo"
    (app / "dist").mkdir(parents=True)
    (app / "dist/index.html").write_text("<h1>Released app</h1>")
    (repo / "shared").mkdir()
    (repo / "shared/component.js").write_text("export const version = 1")
    manifest = (
        'applet_id = "00000000-0000-0000-0000-000000000001"\nsource_paths = ["shared"]\n'
        'title = "Release test"\ndescription = "Test"\n'
    )
    if request.param == "build":
        manifest += 'build_command = "python build.py"\n'
        (app / ".gitignore").write_text("dist/\n")
        (app / "dist/stale.js").write_text("stale()")
        (app / "build.py").write_text(
            "from pathlib import Path\n"
            "Path('dist').mkdir(exist_ok=True)\n"
            "Path('dist/index.html').write_text('<h1>Built app</h1>')\n"
        )
    (app / "applet.toml").write_text(manifest)
    for arguments in (
        ["init", "-b", "main"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "."],
        ["commit", "-m", "initial"],
        ["init", "--bare", str(tmp_path / "origin.git")],
        ["remote", "add", "origin", str(tmp_path / "origin.git")],
        ["push", "origin", "HEAD:main"],
    ):
        subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)
    return app


@pytest.mark.parametrize(
    ("release_checkout", "expected_html"),
    [
        ("static", b"<h1>Released app</h1>"),
        ("build", b"<h1>Built app</h1>"),
    ],
    indirect=["release_checkout"],
)
def test_release_records_source_and_status_ignores_unrelated_commits(
    release_checkout: Path, expected_html: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = None

    def http_request(method: str, url: str, **kwargs: object) -> RequestsResponse:
        nonlocal metadata
        if method == "POST":
            assert url.endswith("/api/marina/applets/00000000-0000-0000-0000-000000000001")
            assert kwargs["params"] == {"base_version": 7}
            package = read_applet_package(kwargs["data"])
            assert "dist/stale.js" not in package.files
            metadata = json.loads(package.files["dist/marina-release.json"])
            assert package.files["dist/index.html"] == expected_html
            return json_response({"id": "00000000-0000-0000-0000-000000000001", "version": 8, "path": "/a/id/v/8/"})
        if url.endswith("marina-release.json"):
            response = json_response(metadata if metadata is not None else {"detail": "applet file not found"})
            if metadata is None:
                response.status_code = 404
            return response
        return json_response({"current_version": 8, "authenticated_url": "http://localhost:8080/a/id/"})

    monkeypatch.setattr("marina.client.requests.request", http_request)
    runner = CliRunner()
    status_args = ["release-status", str(release_checkout), "--url", "http://localhost:8080", "--json"]
    before = runner.invoke(cli, status_args)
    assert before.exit_code == 0, before.output
    assert json.loads(before.output)["state"] == "unrecorded"

    result = runner.invoke(
        cli, ["release", str(release_checkout), "--base-version", "7", "--url", "http://localhost:8080", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == metadata
    assert not (release_checkout / "dist/marina-release.json").exists()
    assert (release_checkout / "dist/index.html").read_text() == "<h1>Released app</h1>"
    if (release_checkout / "build.py").exists():
        assert (release_checkout / "dist/stale.js").read_text() == "stale()"
    status = runner.invoke(cli, status_args)
    assert json.loads(status.output)["state"] == "current"

    (release_checkout.parent / "unrelated.txt").write_text("another app")
    for arguments in (["add", "."], ["commit", "-m", "unrelated"]):
        subprocess.run(["git", *arguments], cwd=release_checkout.parent, check=True, capture_output=True)
    status = runner.invoke(cli, status_args)
    report = json.loads(status.output)
    assert report["state"] == "current"
    assert report["checkout"]["commit"] != report["released"]["commit"]

    (release_checkout.parent / "shared/component.js").write_text("export const version = 2")
    assert json.loads(runner.invoke(cli, status_args).output)["state"] == "modified"
    for arguments in (["add", "."], ["commit", "-m", "shared component"]):
        subprocess.run(["git", *arguments], cwd=release_checkout.parent, check=True, capture_output=True)
    assert json.loads(runner.invoke(cli, status_args).output)["state"] == "different"

    manifest_path = release_checkout / "applet.toml"
    manifest_path.write_text(manifest_path.read_text().replace('title = "Release test"', 'title = "Changed"'))
    status = runner.invoke(cli, status_args)
    assert json.loads(status.output)["state"] == "modified"
    for arguments in (["add", "."], ["commit", "-m", "app change"]):
        subprocess.run(["git", *arguments], cwd=release_checkout, check=True, capture_output=True)
    status = runner.invoke(cli, status_args)
    assert json.loads(status.output)["state"] == "different"


def test_release_requires_expected_revision_and_supports_validation(release_checkout: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(cli, ["release", str(release_checkout)])
    assert missing.exit_code == 2
    preview = runner.invoke(cli, ["release", str(release_checkout), "--dry-run", "--json"])
    assert preview.exit_code == 0, preview.output
    assert "package" in json.loads(preview.output)["checks"]


@pytest.mark.parametrize("release_checkout", ["static"], indirect=True)
@pytest.mark.parametrize("committed", [False, True])
def test_release_allow_dirty_uploads_local_source_without_claiming_verified_provenance(
    release_checkout: Path, committed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = b"<h1>Branch preview</h1>"
    (release_checkout / "dist/index.html").write_bytes(html)
    (release_checkout / "dist/extra.js").write_text("branch()")
    if committed:
        for args in (["add", "."], ["commit", "-m", "branch work"]):
            subprocess.run(["git", *args], cwd=release_checkout, check=True, capture_output=True)
    uploads = []

    def http_request(method: str, url: str, **kwargs: object) -> RequestsResponse:
        if method == "POST":
            package = read_applet_package(kwargs["data"])
            uploads.append(package)
            assert kwargs["params"] == {"base_version": 7}
            assert package.files["dist/index.html"] == html
            assert package.files["dist/extra.js"] == b"branch()"
            assert "dist/marina-release.json" not in package.files
            return json_response({"id": "id", "version": 8, "path": "/a/id/v/8/"})
        if url.endswith("marina-release.json"):
            response = json_response({"detail": "applet file not found"})
            response.status_code = 404
            return response
        return json_response({"current_version": 8, "authenticated_url": "http://localhost:8080/a/id/"})

    monkeypatch.setattr("marina.client.requests.request", http_request)
    runner = CliRunner()
    args = ["release", str(release_checkout), "--base-version", "7", "--url", "http://localhost:8080", "--json"]
    assert runner.invoke(cli, args).exit_code == 1
    assert not uploads
    released = runner.invoke(cli, [*args, "--allow-dirty"])
    assert released.exit_code == 0, released.output
    assert len(uploads) == 1
    assert "source" not in json.loads(released.output)
    if not committed:
        dirty_status = runner.invoke(
            cli, ["release-status", str(release_checkout), "--url", "http://localhost:8080", "--json"]
        )
        assert dirty_status.exit_code == 0, dirty_status.output
        assert json.loads(dirty_status.output)["state"] == "modified"
        subprocess.run(["git", "add", "."], cwd=release_checkout, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "capture local source"], cwd=release_checkout, check=True, capture_output=True
        )
    status = runner.invoke(cli, ["release-status", str(release_checkout), "--url", "http://localhost:8080", "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["state"] == "unrecorded"


@pytest.mark.parametrize("release_checkout", ["static"], indirect=True)
def test_release_rejects_ignored_files_in_static_package(
    release_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    def http_request(method: str, url: str, **_kwargs: object) -> RequestsResponse:
        requests.append((method, url))
        return json_response({"path": "/a/id/v/8/"})

    monkeypatch.setattr("marina.client.requests.request", http_request)
    (release_checkout / ".gitignore").write_text("dist/extra.js\n")
    for arguments in (["add", "."], ["commit", "-m", "static"], ["push", "origin", "HEAD:main"]):
        subprocess.run(["git", *arguments], cwd=release_checkout, check=True, capture_output=True)
    args = ["release", str(release_checkout), "--base-version", "7", "--url", "http://localhost:8080"]
    clean = CliRunner().invoke(cli, args)
    assert clean.exit_code == 0, clean.output
    assert len(requests) == 1

    (release_checkout / "dist/extra.js").write_text("unreviewed()")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "absent from Git" in result.output
    assert len(requests) == 1


@pytest.mark.parametrize("release_checkout", ["static"], indirect=True)
def test_require_current_main_allows_detached_ci_checkout(release_checkout: Path) -> None:
    subprocess.run(["git", "checkout", "--detach"], cwd=release_checkout, check=True, capture_output=True)
    commit = require_current_main(release_checkout)
    assert subprocess.check_output(["git", "show", f"{commit}:demo/dist/index.html"], cwd=release_checkout) == (
        b"<h1>Released app</h1>"
    )


@pytest.mark.parametrize("release_checkout", ["static"], indirect=True)
@pytest.mark.parametrize("change", ["untracked", "modified", "staged", "unmerged", "old_ci_rerun"])
def test_require_current_main_rejects_unreleased_source(release_checkout: Path, change: str) -> None:
    if change == "untracked":
        (release_checkout / "extra.txt").write_text("unreviewed")
    else:
        (release_checkout / "applet.toml").write_text('title = "Changed"\n')
        arguments = []
        if change != "modified":
            arguments.append(["add", "."])
        if change in {"unmerged", "old_ci_rerun"}:
            arguments.append(["commit", "-m", "change"])
        if change == "old_ci_rerun":
            arguments.extend(
                [
                    ["push", "origin", "HEAD:main"],
                    ["checkout", "--detach", "HEAD^"],
                    ["update-ref", "refs/remotes/origin/main", "HEAD"],
                ]
            )
        for command in arguments:
            subprocess.run(["git", *command], cwd=release_checkout, check=True, capture_output=True)
    with pytest.raises(RuntimeError):
        require_current_main(release_checkout)


@pytest.mark.parametrize("release_checkout", ["build"], indirect=True)
def test_failed_release_build_restores_existing_output(release_checkout: Path) -> None:
    (release_checkout / "build.py").write_text("from pathlib import Path\nPath('dist').mkdir()\nraise SystemExit(7)\n")
    for arguments in (["add", "."], ["commit", "-m", "failing build"], ["push", "origin", "HEAD:main"]):
        subprocess.run(["git", *arguments], cwd=release_checkout, check=True, capture_output=True)

    result = CliRunner().invoke(cli, ["release", str(release_checkout), "--base-version", "7"])

    assert result.exit_code == 1
    assert "exit status 7" in result.output
    assert (release_checkout / "dist/index.html").read_text() == "<h1>Released app</h1>"
    assert (release_checkout / "dist/stale.js").read_text() == "stale()"
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=release_checkout) == b""


@pytest.mark.parametrize("release_checkout", ["static"], indirect=True)
@pytest.mark.parametrize("failure", ["unreachable", "timeout"])
def test_release_reports_remote_git_failures(
    release_checkout: Path, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if failure == "unreachable":
        subprocess.run(
            ["git", "remote", "set-url", "origin", str(release_checkout / "missing.git")],
            cwd=release_checkout,
            check=True,
            capture_output=True,
        )
    else:
        run = subprocess.run

        def remote_timeout(arguments, **kwargs):
            if arguments[1] == "ls-remote":
                raise subprocess.TimeoutExpired(arguments, 30)
            return run(arguments, **kwargs)

        monkeypatch.setattr("rigging.provenance.subprocess.run", remote_timeout)

    result = CliRunner().invoke(cli, ["release", str(release_checkout), "--base-version", "7"])

    assert result.exit_code == 1
    assert "Error: git ls-remote" in result.output


def json_response(value: object) -> RequestsResponse:
    response = RequestsResponse()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(value).encode()
    return response


def applet_client_and_store(tmp_path: Path, database_url: str) -> tuple[TestClient, AppletStore]:
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    database = UrlDatabase(database_url)
    store = AppletStore(database)
    ensure_applet_provisioning(store.engine)
    store.migrate()
    config = MarinaConfig(data_root=str(tmp_path / "data"), iap_audience=None, database=database)
    return TestClient(create_app(config), client=("127.0.0.1", 40000)), store


def test_release_without_git_creates_and_updates_the_returned_uuid(
    tmp_path: Path, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adhoc"
    (source / "dist").mkdir(parents=True)
    (source / "applet.toml").write_text('title = "Report"\ndescription = "One-off report"\n')
    index = source / "dist/index.html"
    index.write_text("first report")

    with isolated_database(database_url) as local_url:
        client, store = applet_client_and_store(tmp_path, local_url)

        def http_request(method: str, url: str, **kwargs: object) -> RequestsResponse:
            response = client.request(method, url, params=kwargs.get("params"), content=kwargs.get("data"))
            result = json_response(response.json())
            result.status_code = response.status_code
            return result

        monkeypatch.setattr("marina.client.requests.request", http_request)
        runner = CliRunner()
        arguments = ["release", str(source), "--url", "http://localhost:8080", "--json"]
        first = runner.invoke(cli, [*arguments, "--base-version", "0"])
        assert first.exit_code == 0, first.output
        selected = json.loads(first.output)
        index.write_text("second report")
        second = runner.invoke(cli, [*arguments, "--id", selected["id"], "--base-version", "1"])
        assert second.exit_code == 0, second.output
        assert json.loads(second.output)["id"] == selected["id"]
        assert client.get(f"/a/{selected['id']}/").text == "second report"
        assert client.get(f"/a/{selected['id']}/v/1/").text == "first report"

        manifest = source / "applet.toml"
        manifest.write_text(f'applet_id = "{selected["id"]}"\nsource_paths = ["shared"]\n' + manifest.read_text())
        (source / "shared").mkdir()
        index.write_text("declared report")
        declared = runner.invoke(cli, [*arguments, "--base-version", "2", "--allow-dirty"])
        assert declared.exit_code == 0, declared.output
        assert json.loads(declared.output)["id"] == selected["id"]
        assert client.get(f"/a/{selected['id']}/").text == "declared report"
        stored = store.version(uuid.UUID(selected["id"]), 3).manifest
        assert stored.applet_id == selected["id"]
        assert stored.source_paths == ("shared",)
        rejected = client.post(
            f"/api/marina/applets/{uuid.uuid4()}", params={"base_version": 0}, content=package_applet(source)
        )
        assert rejected.status_code == 400
        assert store.current_version(uuid.UUID(selected["id"])) == 3
        store.engine.dispose()


def test_publish_demo_serves_files_schema_query_and_python_api(tmp_path: Path, database_url: str) -> None:
    client, store = applet_client_and_store(tmp_path, database_url)
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
            "mode": "private",
        }
    ]
    current = client.get(f"/a/{applet_id}/", follow_redirects=False)
    assert current.status_code == 307
    assert current.headers["location"] == f"/a/{applet_id}/v/1/"

    page = client.get(published["path"])
    assert page.status_code == 200
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
    assert client.get(f"/a/{applet_id}/api/identity").json() == {"user": "anonymous"}
    assert client.get(f"/a/{applet_id}/v/1/api/identity").json() == {"user": "anonymous"}

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
    client, _store = applet_client_and_store(tmp_path, database_url)
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
    _client, store = applet_client_and_store(tmp_path, database_url)
    config = MarinaConfig(
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
    assert published["mode"] == "private"
    listed = client.get("/api/marina/applets", headers={"host": "marina.example"}).json()["applets"]
    published_entry = next(applet for applet in listed if applet["name"] == applet_id)
    assert published_entry["path"] == f"https://applets.example/a/{applet_id}/"
    assert published_entry["mode"] == "private"

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


def test_public_applet_surface_serves_only_public_reads(tmp_path: Path, database_url: str) -> None:
    _publisher, store = applet_client_and_store(tmp_path, database_url)
    publisher_config = MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        public_applet_origin="https://public.applets.example",
    )
    publisher = TestClient(create_app(publisher_config), client=("127.0.0.1", 40000))
    public_config = MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        public_applet_origin="https://public.applets.example",
        surface=MarinaSurface.PUBLIC_APPLETS,
    )
    public_client = TestClient(
        create_app(public_config),
        base_url="https://public.applets.example",
        client=("10.0.0.7", 40000),
    )
    published = publisher.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    applet_id = published["id"]
    revision_path = f"/a/{applet_id}/v/1/"

    assert published["mode"] == "private"
    assert public_client.get(revision_path).status_code == 404
    assert public_client.get(f"{revision_path}api/problems").status_code == 404

    changed = publisher.put(
        f"/api/marina/applets/{applet_id}/mode",
        json={"mode": "public", "base_version": 1},
    )
    assert changed.status_code == 200
    assert changed.json() == {
        "id": applet_id,
        "version": 1,
        "mode": "public",
        "url": f"https://public.applets.example/a/{applet_id}/",
    }

    current = public_client.get(f"/a/{applet_id}/", follow_redirects=False)
    assert current.status_code == 307
    assert current.headers["location"] == revision_path
    page = public_client.get(revision_path)
    assert page.status_code == 200
    assert page.headers["cache-control"] == "public, no-cache"
    assert public_client.head(f"{revision_path}app.js").status_code == 200
    assert public_client.get(f"{revision_path}api/problems").status_code == 200
    assert public_client.get(f"{revision_path}api/identity").json() == {"user": None}

    assert public_client.post(f"/a/{applet_id}/query", json={"sql": "SELECT 1"}).status_code == 404
    assert public_client.post(f"{revision_path}api/problems").status_code == 404
    assert public_client.get("/api/marina/applets").status_code == 404

    updated = publisher.post(
        f"/api/marina/applets/{applet_id}",
        params={"base_version": 1},
        content=package_applet(DEMO_APPLET),
    )
    assert updated.status_code == 201
    assert updated.json()["mode"] == "public"

    revoked = publisher.put(
        f"/api/marina/applets/{applet_id}/mode",
        json={"mode": "private", "base_version": 2},
    )
    assert revoked.status_code == 200
    assert public_client.get(revision_path).status_code == 404


def test_named_applet_host_pins_revisions_and_isolates_routes(tmp_path: Path, database_url: str) -> None:
    publisher, store = applet_client_and_store(tmp_path, database_url)
    package = package_applet(DEMO_APPLET)
    first = publisher.post("/api/marina/applets", content=package).json()
    other = publisher.post("/api/marina/applets", content=package).json()
    applet_id = first["id"]
    store.bind_host("zephyr.example", uuid.UUID(applet_id))
    config = MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        applet_origin="https://applets.example",
        applet_domains=frozenset({"zephyr.example"}),
    )
    named = TestClient(create_app(config), base_url="https://zephyr.example", client=("127.0.0.1", 40000))
    root = named.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/v/1/"
    assert root.headers["cache-control"] == "no-cache"
    page = named.get("/")
    assert page.url == "https://zephyr.example/v/1/"
    assert page.headers["cache-control"] == "private, no-cache"
    assert "Problem sets" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    asset = named.get("/v/1/app.js")
    assert asset.content == (DEMO_APPLET / "dist" / "app.js").read_bytes()
    assert named.get("/v/1/api/identity").json() == {"user": "anonymous"}
    query = named.post(
        "/v/1/query", json={"sql": "SELECT CAST(:value AS INTEGER) AS value", "parameters": {"value": 17}}
    )
    assert query.json()["rows"] == [{"value": 17}]
    assert named.get("/api/marina/apps").status_code == 404
    assert named.delete(f"/api/marina/applets/{other['id']}").status_code == 404
    assert named.get(f"/a/{other['id']}/api/identity").status_code == 404
    assert named.get(f"/a/{applet_id}/v/1/").status_code == 404
    shared = named.get(f"https://applets.example/a/{applet_id}/v/1/api/revision")
    assert shared.json() == {"version": 1}

    updated = publisher.post(f"/api/marina/applets/{applet_id}?base_version=1", content=package)
    assert updated.status_code == 201
    assert named.get("/", follow_redirects=False).headers["location"] == "/v/2/"
    assert named.get("/v/1/api/revision").json() == {"version": 1}
    assert named.get("/v/2/api/revision").json() == {"version": 2}
    assert publisher.delete(f"/api/marina/applets/{applet_id}").status_code == 204
    assert named.get("/").status_code == 404
    assert named.get("/v/1/").status_code == 404


@pytest.mark.parametrize(
    "method,path",
    [("GET", "/"), ("GET", "/v/1/"), ("GET", "/v/1/app.js"), ("GET", "/v/1/api/identity"), ("POST", "/v/1/query")],
)
def test_named_applet_host_authenticates_rewritten_routes(
    tmp_path: Path, database_url: str, method: str, path: str
) -> None:
    publisher, store = applet_client_and_store(tmp_path, database_url)
    published = publisher.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    store.bind_host("zephyr.example", uuid.UUID(published["id"]))
    config = MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience="test-audience",
        database=store.database,
        applet_origin="https://applets.example",
        applet_domains=frozenset({"zephyr.example"}),
    )
    remote = TestClient(create_app(config), base_url="https://zephyr.example", client=("10.0.0.7", 40000))
    assert remote.request(method, path).status_code == 401


def test_host_binding_changes_reach_running_servers_and_require_operator(tmp_path: Path, database_url: str) -> None:
    publisher, store = applet_client_and_store(tmp_path, database_url)
    package = package_applet(DEMO_APPLET)
    first = publisher.post("/api/marina/applets", content=package).json()
    second = publisher.post("/api/marina/applets", content=package).json()
    config = MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        canonical_origin="https://marina.example",
        applet_origin="https://applets.example",
        applet_operators=frozenset({"anonymous"}),
        applet_domains=frozenset({"live.example"}),
    )
    operator = TestClient(create_app(config), base_url="https://marina.example", client=("127.0.0.1", 40000))
    servers = [
        TestClient(create_app(config), base_url="https://live.example", client=("127.0.0.1", 40000)) for _ in range(2)
    ]
    binding_path = "/api/marina/applet-hosts/live.example"
    body = {"applet_id": first["id"]}
    for server in servers:
        assert server.get("/").status_code == 404
        assert server.get("/api/marina/apps").status_code == 404
    assert operator.delete(binding_path).status_code == 404
    assert publisher.put(binding_path, json=body).status_code == 403
    assert operator.put(binding_path, json=body).status_code == 200
    assert operator.get("/api/marina/applet-hosts").json()["hosts"]["live.example"] == first["id"]
    for server in servers:
        assert server.get("/api/revision").json() == {"version": 1}
        assert server.get("/api/marina/apps").status_code == 404

    publisher.post(f"/api/marina/applets/{second['id']}?base_version=1", content=package)
    rebound = {"applet_id": second["id"]}
    assert operator.put(binding_path, json=rebound).status_code == 200
    for server in servers:
        assert server.get("/api/revision").json() == {"version": 2}
        assert server.get("/", follow_redirects=False).headers["location"] == "/v/2/"
    assert publisher.delete(binding_path).status_code == 403
    assert operator.put("/api/marina/applet-hosts/marina.example", json=body).status_code == 400
    assert operator.put("/api/marina/applet-hosts/unprovisioned.example", json=body).status_code == 400
    assert operator.delete("/api/marina/applet-hosts/typo.example").status_code == 404
    assert operator.delete(binding_path).status_code == 204
    for server in servers:
        assert server.get("/").status_code == 404
        assert server.get("/api/marina/apps").status_code == 404


def test_host_migration_preserves_filer_and_does_not_restore_changed_bindings(database_url: str) -> None:
    store = AppletStore(UrlDatabase(database_url))
    ensure_applet_provisioning(store.engine)
    store.migrate()
    filer_id = uuid.UUID("7f536ce3-0562-40f1-b209-e6110d8ab4e8")
    with store.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO applets (id, title, description, owner, current_version) "
                "VALUES (:id, 'Filer', '', 'owner', 4) ON CONFLICT DO NOTHING"
            ),
            {"id": filer_id},
        )
        # Model the pre-registry schema while retaining the existing production applet.
        connection.execute(text("DROP TABLE applet_hosts"))
        connection.execute(text("DELETE FROM schema_migrations WHERE name = 'm0001_applet_hosts'"))
    store.migrate()
    assert store.host_applet("filer.marina.oa.dev") == filer_id
    replacement = store.publish(read_applet_package(package_applet(DEMO_APPLET)), "owner")
    store.bind_host("filer.marina.oa.dev", replacement.applet_id)
    store.migrate()
    assert store.host_applet("filer.marina.oa.dev") == replacement.applet_id
    store.unbind_host("filer.marina.oa.dev")
    store.migrate()
    assert store.host_bindings()["filer.marina.oa.dev"] is None
    # A deliberately removed row must not be recreated by the seed on later deployments.
    with store.engine.begin() as connection:
        connection.execute(text("DELETE FROM applet_hosts WHERE hostname = 'filer.marina.oa.dev'"))
    store.migrate()
    assert "filer.marina.oa.dev" not in store.host_bindings()


@pytest.mark.parametrize("named", [False, True])
def test_authenticated_users_share_applet_lifecycle(
    tmp_path: Path, database_url: str, monkeypatch: pytest.MonkeyPatch, named: bool
) -> None:
    # Replace only external signature verification; exercise Marina's real auth
    # middleware and mutation routes with a remote peer and distinct identities.
    def verify_assertion(assertion: str, *_args: object, **_kwargs: object) -> dict[str, str]:
        return {"email": assertion}

    monkeypatch.setattr("google.oauth2.id_token.verify_token", verify_assertion)
    with isolated_database(database_url) as local_url:
        database = UrlDatabase(local_url)
        store = AppletStore(database)
        ensure_applet_provisioning(store.engine)
        store.migrate()
        config = MarinaConfig(
            data_root=str(tmp_path / "data"),
            iap_audience="test-audience",
            database=database,
            applet_operators=frozenset({"first@example"}),
        )
        client = TestClient(create_app(config), client=("10.0.0.7", 40000))
        source = tmp_path / "shared"
        shutil.copytree(DEMO_APPLET, source)
        if named:
            with (source / "applet.toml").open("a") as manifest:
                manifest.write('\nname = "shared-test"\n')
        package = package_applet(source)
        applet_id = str(uuid.uuid4())
        endpoint = f"/api/marina/applets/{applet_id}"
        first = {"x-goog-iap-jwt-assertion": "first@example"}
        second = {"x-goog-iap-jwt-assertion": "second@example"}
        created = client.post(endpoint, params={"base_version": 0}, headers=first, content=package)
        assert created.status_code == 201, created.text
        updated = client.post(endpoint, params={"base_version": 1}, headers=second, content=package)
        assert updated.status_code == 201, updated.text
        assert updated.json()["version"] == 2
        assert [item.published_by for item in store.versions(uuid.UUID(applet_id))] == [
            "second@example",
            "first@example",
        ]
        listed = client.get("/api/marina/apps", headers=first).json()["apps"]
        assert listed[0]["published_by"] == "second@example"

        mode = "private" if named else "public"
        mutations = [
            ("POST", endpoint, {"params": {"base_version": 1}, "content": package}),
            ("PUT", f"{endpoint}/mode", {"json": {"mode": mode, "base_version": 1}}),
            ("PUT", f"{endpoint}/current", {"json": {"version": 1, "base_version": 1}}),
        ]
        for method, path, arguments in mutations:
            assert client.request(method, path, headers=second, **arguments).status_code == 409
            assert client.request(method, path, **arguments).status_code == 401
        assert client.delete(endpoint).status_code == 401
        assert store.current_version(uuid.UUID(applet_id)) == 2
        assert client.put(f"{endpoint}/mode", headers=second, json={"mode": mode, "base_version": 2}).status_code == 200
        assert store.access_mode(uuid.UUID(applet_id)).value == mode
        assert (
            client.put(f"{endpoint}/current", headers=second, json={"version": 1, "base_version": 2}).status_code == 200
        )
        assert store.current_version(uuid.UUID(applet_id)) == 1
        assert client.delete(endpoint, headers=second).status_code == 204
        assert store.active_applets() == []
        store.engine.dispose()


def test_publish_rejects_archive_escape(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client_and_store(tmp_path, database_url)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../outside")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    response = client.post("/api/marina/applets", content=buffer.getvalue())
    assert response.status_code == 400


def test_publish_rejects_backend_that_cannot_import(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client_and_store(tmp_path, database_url)
    broken = tmp_path / "broken-import"
    shutil.copytree(DEMO_APPLET, broken)
    (broken / "server" / "app.py").write_text('raise RuntimeError("broken import")\n')
    before = client.get("/api/marina/applets").json()
    response = client.post("/api/marina/applets", content=package_applet(broken))
    assert response.status_code == 400
    assert client.get("/api/marina/applets").json() == before


def test_backend_factory_failure_returns_503(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client_and_store(tmp_path, database_url)
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
    client, _store = applet_client_and_store(tmp_path, database_url)
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
    _client, store = applet_client_and_store(tmp_path, database_url)
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

    store.rollback(published.applet_id, 2, 6)
    next_revision = store.publish(package, "owner@example", published.applet_id, 2)
    assert next_revision.version == 7
    assert [item.version for item in store.versions(published.applet_id)] == [7, 6, 5, 4, 3]


def test_table_load_statements_populate_applet_schema(tmp_path: Path, database_url: str) -> None:
    client, _store = applet_client_and_store(tmp_path, database_url)
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
    client, _store = applet_client_and_store(tmp_path, database_url)
    published = client.post("/api/marina/applets", content=package_applet(DEMO_APPLET)).json()
    endpoint = f"/a/{published['id']}/query"
    assert client.post(endpoint, content=b"{").status_code == 400
    too_many = client.post(
        endpoint,
        json={
            "sql": "SELECT value FROM generate_series(1, 10001) AS value",
            "parameters": {},
        },
    )
    assert too_many.status_code == 413
    returning = client.post(
        endpoint,
        json={
            "sql": "INSERT INTO problems (id, prompt, answer) VALUES (100, 'new', 1) RETURNING id",
            "parameters": {},
        },
    )
    assert returning.status_code == 400
    count = client.post(endpoint, json={"sql": "SELECT count(*) AS count FROM problems", "parameters": {}})
    assert count.json()["rows"] == [{"count": 3}]

    monkeypatch.setattr("marina.applets.MAX_QUERY_RESPONSE_BYTES", 200)
    oversized = client.post(endpoint, json={"sql": "SELECT repeat('x', 10000) AS value", "parameters": {}})
    assert oversized.status_code == 413

    monkeypatch.setattr("marina.applets.QUERY_TIMEOUT_MS", 20)
    timed_out = client.post(endpoint, json={"sql": "SELECT pg_sleep(0.2)", "parameters": {}})
    assert timed_out.status_code == 504


def test_query_pool_reuses_connections_without_leaking_request_state(tmp_path: Path, database_url: str) -> None:
    _client, store = applet_client_and_store(tmp_path, database_url)
    try:
        package = read_applet_package(package_applet(DEMO_APPLET))
        first = store.publish(package, "owner").applet_id
        second = store.publish(package, "owner").applet_id
        store.query(first, "INSERT INTO problems VALUES (4, 'first app only', 17)", {})
        identity_sql = "SELECT pg_backend_pid() AS pid, current_user AS role, count(*) AS count FROM problems"
        before = cast(list[dict[str, object]], store.query(first, identity_sql, {})["rows"])[0]
        store.query(first, "CREATE TEMP TABLE problems (id INTEGER)", {})
        store.query(
            first,
            "SELECT set_config('search_path', 'public', false), set_config('role', :role, false)",
            {"role": APPLET_READER_ROLE},
        )
        with pytest.raises(DBAPIError):
            store.query(first, "SELECT 1 / 0", {})
        after = cast(list[dict[str, object]], store.query(first, identity_sql, {})["rows"])[0]
        assert after == before
        assert after["role"] == applet_role(first)
        assert after["count"] == 4
        other = cast(list[dict[str, object]], store.query(second, identity_sql, {})["rows"])[0]
        assert other["role"] == applet_role(second)
        assert other["pid"] != after["pid"]
        assert other["count"] == 3
        store.archive(first)
        with pytest.raises(AppletNotFound):
            store.query(first, identity_sql, {})
    finally:
        store.close()


def test_concurrent_applet_queries_keep_writes_in_each_schema(tmp_path: Path, database_url: str) -> None:
    _client, store = applet_client_and_store(tmp_path, database_url)
    try:
        package = read_applet_package(package_applet(DEMO_APPLET))
        applets = [store.publish(package, "owner").applet_id for _ in range(2)]
        ready = Barrier(6)

        def write(index: int) -> None:
            ready.wait()
            store.query(
                applets[index % 2],
                "INSERT INTO problems VALUES (:id, :prompt, :answer)",
                {"id": index + 4, "prompt": f"problem {index}", "answer": index},
            )

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(write, range(6)))
        for index, applet_id in enumerate(applets):
            rows = store.query(applet_id, "SELECT answer FROM problems WHERE id > 3 ORDER BY id", {})["rows"]
            assert rows == [{"answer": value} for value in range(index, 6, 2)]
    finally:
        store.close()


def test_failed_migration_leaves_no_applet_or_schema(tmp_path: Path, database_url: str) -> None:
    client, store = applet_client_and_store(tmp_path, database_url)
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


def test_release_dry_run_reports_checks_and_runtime_omissions() -> None:
    result = CliRunner().invoke(cli, ["release", str(DEMO_APPLET), "--dry-run", "--json"])
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
    assert report["checks"] == ["package", "inline_scripts", "backend_import"]
    assert report["not_checked"] == [
        "backend_factory_execution",
        "migration_execution",
        "browser",
    ]


def test_release_cli_sends_public_mode(release_checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def http_request(method: str, url: str, **kwargs: object) -> RequestsResponse:
        params = kwargs.get("params")
        requests.append((method, url, params if isinstance(params, dict) else None))
        return json_response(
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "version": 1,
                "mode": "public",
                "path": "/a/00000000-0000-0000-0000-000000000001/v/1/",
                "url": "https://public.applets.example/a/00000000-0000-0000-0000-000000000001/v/1/",
            }
        )

    monkeypatch.setattr("marina.client.requests.request", http_request)
    result = CliRunner().invoke(
        cli,
        [
            "release",
            str(release_checkout),
            "--url",
            "http://localhost:8080",
            "--base-version",
            "0",
            "--mode",
            "public",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert requests == [
        (
            "POST",
            "http://localhost:8080/api/marina/applets/00000000-0000-0000-0000-000000000001",
            {"base_version": 0, "mode": "public"},
        )
    ]
    assert json.loads(result.output)["mode"] == "public"


def test_mode_cli_updates_current_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str, object | None]] = []

    def http_request(method: str, url: str, **kwargs: object) -> RequestsResponse:
        requests.append((method, url, kwargs.get("json")))
        if method == "GET":
            return json_response({"current_version": 9})
        return json_response({"url": "https://public.applets.example/a/example/"})

    monkeypatch.setattr("marina.client.requests.request", http_request)
    result = CliRunner().invoke(
        cli,
        ["applets", "mode", "example", "public", "--url", "http://localhost:8080"],
    )

    assert result.exit_code == 0
    assert requests == [
        ("GET", "http://localhost:8080/api/marina/applets/example", None),
        ("PUT", "http://localhost:8080/api/marina/applets/example/mode", {"mode": "public", "base_version": 9}),
    ]
    assert result.output.strip() == "https://public.applets.example/a/example/"


def test_validate_reports_backend_import_check() -> None:
    result = CliRunner().invoke(cli, ["validate", str(DEMO_APPLET), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["checks"] == ["package", "inline_scripts", "backend_import"]
    assert report["not_checked"] == ["backend_factory_execution", "migration_execution", "browser"]


def test_validate_rejects_import_outside_runtime_module_path(tmp_path: Path) -> None:
    broken = tmp_path / "broken-runtime-import"
    shutil.copytree(DEMO_APPLET, broken)
    (broken / "local_helper.py").write_text("VALUE = 1\n")
    (broken / "server" / "app.py").write_text("from local_helper import VALUE\n")

    result = CliRunner().invoke(cli, ["validate", str(broken), "--json"])

    assert result.exit_code == 2


def test_package_rejects_executable_inline_script(tmp_path: Path) -> None:
    package_dir = tmp_path / "inline-script"
    shutil.copytree(DEMO_APPLET, package_dir)
    (package_dir / "dist" / "index.html").write_text(
        "<!doctype html><html><body><script>document.body.textContent = 'blocked'</script></body></html>"
    )

    with pytest.raises(ValueError):
        package_applet(package_dir)


def test_package_rejects_legacy_javascript_mime_alias(tmp_path: Path) -> None:
    package_dir = tmp_path / "legacy-inline-script"
    shutil.copytree(DEMO_APPLET, package_dir)
    (package_dir / "dist" / "index.html").write_text(
        '<!doctype html><html><body><script type="application/x-javascript">alert(1)</script></body></html>'
    )

    with pytest.raises(ValueError):
        package_applet(package_dir)


def test_package_allows_non_executable_inline_script_data(tmp_path: Path) -> None:
    package_dir = tmp_path / "inline-data"
    shutil.copytree(DEMO_APPLET, package_dir)
    html = '<!doctype html><html><body><script type="application/json">{"page": 1}</script></body></html>'
    (package_dir / "dist" / "index.html").write_text(html)

    package = read_applet_package(package_applet(package_dir))

    assert package.files["dist/index.html"] == html.encode()


def test_package_allows_script_type_with_parameters_as_data(tmp_path: Path) -> None:
    package_dir = tmp_path / "inline-typed-data"
    shutil.copytree(DEMO_APPLET, package_dir)
    html = '<!doctype html><html><body><script type="text/javascript; charset=utf-8">data</script></body></html>'
    (package_dir / "dist" / "index.html").write_text(html)

    package = read_applet_package(package_applet(package_dir))

    assert package.files["dist/index.html"] == html.encode()


def test_package_ignores_generated_python_bytecode(tmp_path: Path) -> None:
    package_dir = tmp_path / "generated-bytecode"
    shutil.copytree(DEMO_APPLET, package_dir)
    cache_dir = package_dir / "server" / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "app.cpython-312.pyc").write_bytes(b"generated")
    (package_dir / "server" / "helper.pyc").write_bytes(b"generated")

    package = read_applet_package(package_applet(package_dir))

    assert sorted(package.files) == [
        "applet.toml",
        "dist/app.js",
        "dist/index.html",
        "server/__init__.py",
        "server/app.py",
    ]
