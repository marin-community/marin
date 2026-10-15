# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real Postgres persistence and CPython execution through the playground API."""

import base64
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import requests
from click.testing import CliRunner
from fastapi.testclient import TestClient
from marina.applets import AppletServices
from marina.db import UrlDatabase, engine_for
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.marina.applets.python_playground.cli import APPLET_ID, cli, directory_archive
from infra.marina.applets.python_playground.server import app as playground


@dataclass
class PlaygroundServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


@pytest.fixture
def services(database_url):
    schema = "playground_test_" + uuid4().hex
    engine = engine_for(UrlDatabase(database_url), schema)
    with engine.begin() as connection:
        playground.migrate(connection)
    try:
        yield cast(AppletServices, PlaygroundServices(engine))
    finally:
        engine.dispose()
        admin = create_engine(database_url)
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def client(services):
    with TestClient(playground.create_api(services).app) as client:
        yield client


def create_project(client, content, *, title="Experiment", extra_files=()):
    response = client.post(
        "/projects",
        json={
            "title": title,
            "entrypoint": "main.py",
            "packages": "",
            "files": [{"path": "main.py", "content": content}, *extra_files],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def run_project(client, project, timeout=120):
    response = client.post(f"/projects/{project['id']}/run", json={"revision": project["revision"], "timeout": timeout})
    assert response.status_code == 200, response.text
    return response.json()


def blob_sizes(services):
    with services.engine().connect() as connection:
        return sorted(connection.execute(text("SELECT octet_length(content) FROM blobs")).scalars())


def test_run_persists_inputs_reports_and_binary_blobs_across_api_factories(client, services):
    project = create_project(
        client,
        "from pathlib import Path\nfrom helpers import report\nimport os\n"
        "out = Path(os.environ['OUTPUT_DIR'])\n"
        "(out / 'index.md').write_text(report(Path('data/input.bin').read_bytes()))\n"
        "(out / 'input.bin').write_bytes(Path('data/input.bin').read_bytes())\nprint(Path.cwd())\n",
        extra_files=[
            {"path": "helpers.py", "content": "def report(data):\n    return '# Report\\n' + data.hex()\n"},
            {"path": "data/input.bin", "content": "AP8=", "encoding": "base64"},
        ],
    )
    completed = run_project(client, project)
    completed = run_project(client, completed)
    assert completed["id"] == project["id"]
    assert client.get("/projects").json()["total"] == 1
    assert completed["result"]["exit_code"] == 0
    with TestClient(playground.create_api(services).app) as reopened:
        persisted = reopened.get(f"/projects/{project['id']}").json()
    assert persisted == completed
    assert persisted["document"]["files"] == project["document"]["files"]
    result = persisted["result"]
    artifacts = {file["path"]: base64.b64decode(file["content"]) for file in result["artifacts"]}
    assert artifacts == {"index.md": b"# Report\n00ff", "input.bin": b"\x00\xff"}
    assert not Path(result["logs"].strip().splitlines()[-1]).exists()
    assert blob_sizes(services).count(2) == 2  # Uploaded and generated binary payloads stay binary in Postgres.


def test_conflicting_edit_preserves_latest_source_and_previous_results(client):
    project = run_project(client, create_project(client, "print('first')"))
    document = project["document"]
    document["title"] = "Renamed experiment"
    saved = client.put(f"/projects/{project['id']}", json={"revision": 1, "document": document})
    assert saved.status_code == 200
    assert saved.json()["revision"] == 2
    assert saved.json()["result_revision"] == 1
    stale = client.put(f"/projects/{project['id']}", json={"revision": 1, "document": {**document, "title": "Stale"}})
    assert stale.status_code == 409
    assert client.post(f"/projects/{project['id']}/run", json={"revision": 1}).status_code == 409
    assert client.delete(f"/projects/{project['id']}?revision=1").status_code == 409
    assert client.get(f"/projects/{project['id']}").json() == saved.json()


def test_replacing_source_results_and_deleting_project_reclaims_blobs(client, services):
    content = "from pathlib import Path\nPath('output/old.bin').write_bytes(b'x' * 100)\n"
    project = run_project(client, create_project(client, content))
    assert blob_sizes(services) == sorted([len(content.encode()), 100])
    document = {**project["document"], "files": [{"path": "main.py", "content": "print('fresh')"}]}
    saved = client.put(f"/projects/{project['id']}", json={"revision": 1, "document": document}).json()
    completed = run_project(client, saved)
    assert completed["result"]["artifacts"] == []
    assert blob_sizes(services) == [len(b"print('fresh')")]
    assert client.delete(f"/projects/{project['id']}?revision=2").status_code == 204
    assert client.get(f"/projects/{project['id']}").status_code == 404
    assert blob_sizes(services) == []
    with services.engine().connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM runs")).scalar_one() == 0


def test_failed_program_preserves_partial_outputs_and_traceback(client):
    project = create_project(
        client,
        "from pathlib import Path\n"
        "Path('output/index.md').write_text('partial')\n"
        "raise RuntimeError('failed computation')\n",
    )
    result = run_project(client, project)["result"]
    assert result["exit_code"] != 0
    assert not result["timed_out"]
    assert "RuntimeError: failed computation" in result["logs"]
    assert base64.b64decode(result["artifacts"][0]["content"]) == b"partial"


def test_timeout_reaps_process_and_keeps_partial_report(client):
    project = create_project(
        client, "from pathlib import Path\nPath('output/index.md').write_text('started')\nwhile True: pass\n"
    )
    result = run_project(client, project, timeout=2)["result"]
    assert result["timed_out"]
    assert result["exit_code"] != 0
    assert base64.b64decode(result["artifacts"][0]["content"]) == b"started"


def test_input_limit_rejects_save_without_changing_source_or_blobs(client, services):
    project = create_project(client, "print('retained')")
    sizes = blob_sizes(services)
    document = project["document"]
    document["files"].append(
        {"path": "oversized.bin", "content": base64.b64encode(b"x" * (5 * 1024 * 1024)).decode(), "encoding": "base64"}
    )
    response = client.put(f"/projects/{project['id']}", json={"revision": 1, "document": document})
    assert response.status_code == 413
    restored = client.get(f"/projects/{project['id']}").json()
    assert restored["document"]["files"] == [{"path": "main.py", "content": "print('retained')", "encoding": "utf-8"}]
    assert restored["revision"] == 1
    assert blob_sizes(services) == sizes


def test_output_and_log_limits_persist_only_bounded_results(client, services):
    project = create_project(
        client,
        "from pathlib import Path\n"
        "Path('output/large.bin').write_bytes(b'x' * (10 * 1024 * 1024 + 1))\n"
        "Path('output/index.md').write_text('# Small report')\n"
        "print('x' * 300000)\n",
    )
    result = run_project(client, project)["result"]
    assert result["exit_code"] == 0
    assert len(result["logs"].encode()) <= 256 * 1024
    assert [file["path"] for file in result["artifacts"]] == ["index.md"]
    assert len(result["warnings"]) == 2
    assert max(blob_sizes(services)) < 1024


def test_project_index_search_sort_and_pagination(client):
    for title in ["Zulu", "alpha", "Beta"]:
        create_project(client, "print('hello')", title=title)
    ordered = client.get("/projects", params={"sort": "title", "direction": "asc"}).json()
    assert [item["title"] for item in ordered["items"]] == ["alpha", "Beta", "Zulu"]
    assert all(item["status"] == "never" and item["file_count"] == 1 for item in ordered["items"])
    searched = client.get("/projects", params={"query": "ALP"}).json()
    assert [item["title"] for item in searched["items"]] == ["alpha"]
    tail = client.get("/projects", params={"sort": "title", "direction": "desc", "offset": 2}).json()
    assert tail["total"] == 3
    assert [item["title"] for item in tail["items"]] == ["alpha"]
    assert client.get("/projects", params={"query": "%"}).json()["items"] == []


def test_directory_zip_import_preserves_inputs_requirements_and_runs(client, tmp_path):
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\n"
        "Path('output/index.md').write_text('# Uploaded project\\n' + Path('data.bin').read_bytes().hex())\n"
    )
    (tmp_path / "requirements.txt").write_text("# Standard library only\n")
    (tmp_path / "data.bin").write_bytes(b"\x00\xff")
    (tmp_path / ".env").write_text("do not upload")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "old.md").write_text("stale")
    payload = directory_archive(tmp_path)
    response = client.post(
        "/projects/import",
        params={"title": "Uploaded project"},
        content=payload,
        headers={"Content-Type": "application/zip"},
    )
    assert response.status_code == 201, response.text
    project = response.json()
    assert {file["path"] for file in project["document"]["files"]} == {"main.py", "requirements.txt", "data.bin"}
    assert project["document"]["packages"] == "# Standard library only\n"
    completed = run_project(client, project)
    assert completed["result"]["exit_code"] == 0
    assert base64.b64decode(completed["result"]["artifacts"][0]["content"]) == b"# Uploaded project\n00ff"
    override = client.post(
        "/projects/import", params={"title": "Override", "packages": ["numpy>=2", "seaborn"]}, content=payload
    ).json()
    assert override["document"]["packages"] == "numpy>=2\nseaborn"


def test_compressed_zip_over_input_limit_is_rejected_without_saving(client):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("main.py", 'print("hi")')
        archive.writestr("data.bin", b"x" * (5 * 1024 * 1024))
    response = client.post("/projects/import", params={"title": "Oversized"}, content=payload.getvalue())
    assert response.status_code == 413
    assert client.get("/projects").json()["total"] == 0


def test_default_rigging_runtime_reads_input_and_inherits_storage_credentials(client, monkeypatch):
    monkeypatch.setenv("CW_KEY_ID", "test-key")
    monkeypatch.setenv("CW_KEY_SECRET", "test-secret")
    project = create_project(
        client,
        "from pathlib import Path\n"
        "import os\n"
        "from rigging.filesystem.buckets import filesystem_for\n"
        "fs, path = filesystem_for('input.txt')\n"
        "assert os.environ['CW_KEY_ID'] == 'test-key' and os.environ['CW_KEY_SECRET'] == 'test-secret'\n"
        "with fs.open(path) as source:\n"
        "    Path('output/index.md').write_bytes(source.read())\n",
        extra_files=[{"path": "input.txt", "content": "# Runtime package available"}],
    )
    result = run_project(client, project)["result"]
    assert result["exit_code"] == 0
    assert base64.b64decode(result["artifacts"][0]["content"]) == b"# Runtime package available"


def test_cli_discovers_separate_applet_origin_and_uploads_project(client, tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('uploaded through the canonical origin')")
    registry = "http://127.0.0.1:8000"
    app_url = f"http://127.0.0.1:9000/a/{APPLET_ID}/"

    def remote_request(_session, method, url, **kwargs):
        method = method.upper()
        response = requests.Response()
        response.headers["Content-Type"] = "application/json"
        if method == "GET" and url == f"{registry}/api/marina/applets/{APPLET_ID}":
            response.status_code = 200
            response._content = json.dumps({"authenticated_url": app_url}).encode()
            return response
        assert method == "POST" and url == app_url + "api/projects/import"
        uploaded = client.post("/projects/import", params=kwargs["params"], content=kwargs["data"])
        response.status_code = uploaded.status_code
        response._content = uploaded.content
        return response

    monkeypatch.setattr(requests.Session, "request", remote_request)
    result = CliRunner().invoke(cli, ["upload", str(tmp_path), "--title", "CLI project", "--url", registry])
    assert result.exit_code == 0, result.output
    published = json.loads(result.output)
    assert published["url"] == app_url + "#/projects/" + published["id"]
    saved = client.get("/projects/" + published["id"]).json()
    assert saved["document"]["files"][0]["content"] == "print('uploaded through the canonical origin')"
