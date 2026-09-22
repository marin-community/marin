# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

from marina.apps import migration, services_for
from marina.db import UrlDatabase
from marina.manifest import discover_apps
from marina.server import MarinaConfig, create_app
from starlette.testclient import TestClient

NOTES_APP = Path(__file__).parent / "notes_app"


def test_python_app_api_is_mounted_with_identity_and_own_schema(tmp_path: Path, database_url: str) -> None:
    apps_dir = tmp_path / "apps"
    for name in ("notes", "other"):
        shutil.copytree(NOTES_APP, apps_dir / name)
    database = UrlDatabase(url=database_url)
    for app in discover_apps(apps_dir):
        migration(app)(services_for(app, str(tmp_path), database).engine())
    config = MarinaConfig(apps_dir=apps_dir, data_root=str(tmp_path), iap_audience=None, database=database)
    client = TestClient(create_app(config), client=("127.0.0.1", 40000))

    assert client.get("/notes/api/whoami").json() == {"user": "anonymous", "data": f"{tmp_path}/notes"}
    assert client.post("/notes/api/notes", json={"body": "first"}).status_code == 200
    assert client.get("/notes/api/notes").json() == ["first"]
    assert client.get("/other/api/notes").json() == []
    assert client.get("/notes/").status_code == 200

    remote = TestClient(create_app(config), client=("10.0.0.7", 1234))
    assert remote.get("/notes/api/notes").status_code == 401
