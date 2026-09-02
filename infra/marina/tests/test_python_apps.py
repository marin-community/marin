# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import sqlalchemy
from marina.db import UrlDatabase, engine_for
from marina.server import MarinaConfig, create_app
from starlette.testclient import TestClient

NOTES_APP = """
from fastapi import FastAPI
from rigging.server_auth import get_verified_identity
from sqlalchemy import text


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.get("/whoami")
    def whoami():
        return {"user": get_verified_identity().user_id, "data": services.data_url}

    @api.post("/notes")
    def add(body: dict):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO notes (body) VALUES (:body)"), {"body": body["body"]})
        return {"ok": True}

    @api.get("/notes")
    def list_notes():
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(text("SELECT body FROM notes ORDER BY id"))]

    return api


def migrate(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS notes (id serial PRIMARY KEY, body text NOT NULL)"))
"""


def write_python_app(apps_dir: Path, name: str) -> Path:
    root = apps_dir / name
    root.mkdir(parents=True)
    (root / "app.toml").write_text(f'title = "{name}"\ndescription = "notes"\n')
    (root / "__init__.py").write_text("")
    (root / "app.py").write_text(NOTES_APP)
    (root / "dist").mkdir()
    (root / "dist" / "index.html").write_text("<title>notes</title>")
    return root


def test_python_app_api_is_mounted_with_identity_and_own_schema(tmp_path: Path, database_url: str) -> None:
    write_python_app(tmp_path / "apps", "notes")
    write_python_app(tmp_path / "apps", "other")
    database = UrlDatabase(url=database_url)
    for name in ("notes", "other"):
        engine = engine_for(database, name)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("DROP TABLE IF EXISTS notes"))
            conn.execute(sqlalchemy.text("CREATE TABLE notes (id serial PRIMARY KEY, body text NOT NULL)"))
        engine.dispose()
    config = MarinaConfig(apps_dir=tmp_path / "apps", data_root=str(tmp_path), iap_audience=None, database=database)
    client = TestClient(create_app(config), client=("127.0.0.1", 40000))

    who = client.get("/notes/api/whoami").json()
    assert who == {"user": "anonymous", "data": f"{tmp_path}/notes"}
    assert client.post("/notes/api/notes", json={"body": "first"}).status_code == 200
    assert client.get("/notes/api/notes").json() == ["first"]
    assert client.get("/other/api/notes").json() == []
    assert client.get("/notes/").status_code == 200

    remote = TestClient(create_app(config), client=("10.0.0.7", 1234))
    assert remote.get("/notes/api/notes").status_code == 401
