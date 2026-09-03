# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path

import pytest
from marina.manifest import discover_apps, load_manifest
from marina.server import CANONICAL_ORIGIN_ENV, MarinaConfig, create_app, serve_app_file
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


def config_for(tmp_path: Path) -> MarinaConfig:
    data_root = tmp_path / "data"
    (data_root / "tasktrove").mkdir(parents=True, exist_ok=True)
    return MarinaConfig(apps_dir=tmp_path / "apps", data_root=str(data_root), iap_audience=None)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    write_app(tmp_path / "apps", "tasktrove")
    write_app(tmp_path / "apps", "unbuilt", built=False)
    config = config_for(tmp_path)
    (tmp_path / "data" / "tasktrove" / "sources.json").write_text('[{"source": "a"}]')
    with gzip.open(tmp_path / "data" / "tasktrove" / "labels.json.gz", "wt") as f:
        f.write("[1, 2]")
    return TestClient(create_app(config), client=("127.0.0.1", 40000))


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


def test_data_files_come_from_the_data_root(client: TestClient) -> None:
    plain = client.get("/tasktrove/data/sources.json")
    assert plain.status_code == 200 and plain.json() == [{"source": "a"}]
    assert plain.headers["cache-control"] == "private, max-age=300"
    compressed = client.get("/tasktrove/data/labels.json")
    assert compressed.headers["content-encoding"] == "gzip" and compressed.json() == [1, 2]
    assert client.get("/tasktrove/data/missing.json").status_code == 404
    assert client.get("/tasktrove/data/..%2Fother%2Fx").status_code == 404


def test_non_loopback_without_iap_is_denied(tmp_path: Path) -> None:
    write_app(tmp_path / "apps", "tasktrove")
    app = create_app(config_for(tmp_path))
    remote = TestClient(app, client=("10.0.0.7", 1234))
    assert remote.get("/api/marina/me").status_code == 401
    assert remote.get("/healthz").status_code == 200


def aliased_client(tmp_path: Path) -> TestClient:
    write_app(tmp_path / "apps", "tasktrove")
    config = replace(
        config_for(tmp_path),
        host_apps={"old.example": "tasktrove"},
        canonical_origin="https://marina.example",
    )
    return TestClient(create_app(config), client=("127.0.0.1", 40000))


def test_aliased_host_redirects_into_its_app_on_the_canonical_origin(tmp_path: Path) -> None:
    client = aliased_client(tmp_path)
    response = client.get("/wiki/59?x=1", headers={"host": "old.example"}, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://marina.example/tasktrove/wiki/59?x=1"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/", headers={"host": "other.example"}).status_code == 200


def test_aliased_host_does_not_prefix_a_path_already_inside_its_app(tmp_path: Path) -> None:
    # A link written or cached against the alias's own prefix must not collect a second copy.
    client = aliased_client(tmp_path)
    for path in ("/tasktrove", "/tasktrove/", "/tasktrove/wiki/59"):
        response = client.get(path, headers={"host": "old.example"}, follow_redirects=False)
        assert response.headers["location"] == f"https://marina.example{path}"


def test_aliased_host_keeps_another_apps_name_inside_its_own_app(tmp_path: Path) -> None:
    # The alias belongs to one app, so a path is that app's even when it reads like another
    # app's name: an app could otherwise shadow a route the alias's own app serves.
    client = aliased_client(tmp_path)
    response = client.get("/notes/", headers={"host": "old.example"}, follow_redirects=False)
    assert response.headers["location"] == "https://marina.example/tasktrove/notes/"


def test_an_api_call_to_an_aliased_host_is_told_where_the_api_moved(tmp_path: Path) -> None:
    # A redirect would be followed without the Authorization header, and the retry would come
    # back from IAP as a sign-in page with status 200 -- a success the caller cannot parse.
    client = aliased_client(tmp_path)
    response = client.post("/api/feedback", headers={"host": "old.example"}, follow_redirects=False)
    assert response.status_code == 421
    assert response.json() == {"error": "moved", "url": "https://marina.example/tasktrove/api/feedback"}


def test_a_page_whose_path_merely_starts_with_an_app_still_redirects(tmp_path: Path) -> None:
    client = aliased_client(tmp_path)
    response = client.get("/wiki/api/59", headers={"host": "old.example"}, follow_redirects=False)
    assert response.status_code == 307


def test_aliased_hosts_need_a_canonical_origin(tmp_path: Path) -> None:
    write_app(tmp_path / "apps", "tasktrove")
    config = replace(config_for(tmp_path), host_apps={"old.example": "tasktrove"})
    with pytest.raises(ValueError, match=CANONICAL_ORIGIN_ENV):
        create_app(config)
