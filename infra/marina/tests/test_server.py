# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import pytest
from marina.manifest import discover_apps, load_manifest
from marina.server import MarinaConfig, create_app, serve_app_file
from starlette.testclient import TestClient

TASKTROVE_MANIFEST = """
title = "TaskTrove"
description = "Browse tasks."
connect_src = ["https://huggingface.co"]
build_command = "true"
"""


def write_app(apps_dir: Path, name: str, manifest: str = TASKTROVE_MANIFEST, built: bool = True) -> Path:
    root = apps_dir / name
    root.mkdir(parents=True)
    (root / "app.toml").write_text(manifest)
    if built:
        (root / "dist" / "static").mkdir(parents=True)
        (root / "dist" / "index.html").write_text("<title>TaskTrove</title>")
        (root / "dist" / "static" / "app.js").write_text("console.log(1)")
        with gzip.open(root / "dist" / "labels.json.gz", "wt") as f:
            f.write('{"a": 1}')
    return root


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    write_app(tmp_path, "tasktrove")
    write_app(tmp_path, "unbuilt", built=False)
    return TestClient(create_app(MarinaConfig(apps_dir=tmp_path, iap_audience=None)), client=("127.0.0.1", 40000))


def test_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    root = write_app(tmp_path, "bad", manifest=TASKTROVE_MANIFEST + 'hostname = "x"\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_manifest(root)


def test_discovery_skips_underscore_dirs(tmp_path: Path) -> None:
    write_app(tmp_path, "tasktrove")
    (tmp_path / "_shared").mkdir()
    assert [app.name for app in discover_apps(tmp_path)] == ["tasktrove"]


def test_app_directory_and_identity(client: TestClient) -> None:
    apps = client.get("/api/marina/apps").json()["apps"]
    assert [app["path"] for app in apps] == ["/tasktrove/", "/unbuilt/"]
    me = client.get("/api/marina/me").json()
    assert me == {"user": "anonymous", "role": "admin"}


def test_static_file_and_spa_fallback(client: TestClient) -> None:
    file = client.get("/tasktrove/static/app.js")
    assert file.status_code == 200 and file.text == "console.log(1)"
    route = client.get("/tasktrove/s/some-source")
    assert route.status_code == 200 and "<title>TaskTrove</title>" in route.text
    assert "https://huggingface.co" in route.headers["content-security-policy"]
    assert client.get("/tasktrove", follow_redirects=False).headers["location"] == "/tasktrove/"


def test_path_traversal_falls_back_to_index(tmp_path: Path) -> None:
    app = load_manifest(write_app(tmp_path, "tasktrove"))
    response = serve_app_file(app, "../app.toml")
    assert Path(response.path) == app.dist / "index.html"


def test_precompressed_file_served_with_encoding(client: TestClient) -> None:
    response = client.get("/tasktrove/labels.json")
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"a": 1}


def test_unbuilt_app_reports_503(client: TestClient) -> None:
    assert client.get("/unbuilt/").status_code == 503


def test_non_loopback_without_iap_is_denied(tmp_path: Path) -> None:
    write_app(tmp_path, "tasktrove")
    app = create_app(MarinaConfig(apps_dir=tmp_path, iap_audience=None))
    remote = TestClient(app, client=("10.0.0.7", 1234))
    assert remote.get("/api/marina/me").status_code == 401
    assert remote.get("/healthz").status_code == 200
