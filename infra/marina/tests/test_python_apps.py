# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from marina.applets import AppletStore, package_applet, read_applet_package
from marina.database_setup import applet_provisioning_statements, ensure_applet_provisioning
from marina.db import UrlDatabase, engine_for
from marina.preview import install_preview_sources
from marina.server import MarinaConfig, create_app
from marina.testing import isolated_database
from sqlalchemy import create_engine, make_url, text
from starlette.testclient import TestClient

NOTES_APP = Path(__file__).parent / "notes_app"


def test_python_app_api_is_mounted_with_identity_and_own_schema(tmp_path: Path, database_url: str) -> None:
    with isolated_database(database_url) as local_url:
        apps_dir = tmp_path / "apps"
        for name in ("notes", "other"):
            target = apps_dir / name
            shutil.copytree(NOTES_APP, target)
            (target / "applet.toml").write_text(
                f'name = "{name}"\npython_entrypoint = "app:create_api"\n' + (target / "applet.toml").read_text()
            )
        database = UrlDatabase(url=local_url)
        install_preview_sources(apps_dir, database)
        config = MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)
        client = TestClient(create_app(config), client=("127.0.0.1", 40000))

        assert client.get("/notes/api/whoami").json() == {"user": "anonymous", "data": f"{tmp_path}/notes"}
        assert client.post("/notes/api/notes", json={"body": "first"}).status_code == 200
        assert client.get("/notes/api/notes").json() == ["first"]
        assert client.get("/other/api/notes").json() == []
        assert client.get("/notes/").status_code == 200

        remote = TestClient(create_app(config), client=("10.0.0.7", 1234))
        assert remote.get("/notes/api/notes").status_code == 401


@pytest.fixture
def adoption_database(database_url: str) -> Iterator[UrlDatabase]:
    role = f"marina_test_{uuid4().hex}"
    admin = create_engine(database_url)
    try:
        with admin.begin() as connection:
            connection.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD 'marina-test'"))
        with isolated_database(database_url) as local_url:
            local_admin = create_engine(local_url)
            try:
                with local_admin.begin() as connection:
                    name = make_url(local_url).database
                    connection.execute(text(f'GRANT ALL ON DATABASE "{name}" TO {role}'))
                    for statement in applet_provisioning_statements(role):
                        connection.execute(text(statement))
                yield UrlDatabase(
                    make_url(local_url).set(username=role, password="marina-test").render_as_string(hide_password=False)
                )
            finally:
                local_admin.dispose()
    finally:
        with admin.begin() as connection:
            connection.execute(text(f"DROP ROLE {role}"))
        admin.dispose()


def test_named_release_adopts_existing_data_and_rolls_back_failed_migration(
    tmp_path: Path, adoption_database: UrlDatabase
) -> None:

    database = adoption_database
    legacy = engine_for(database, "notes")
    with legacy.begin() as connection:
        connection.execute(text("CREATE TABLE notes (id serial PRIMARY KEY, body text NOT NULL)"))
        connection.execute(text("INSERT INTO notes (body) VALUES ('existing')"))
    legacy.dispose()
    source = tmp_path / "notes"
    shutil.copytree(NOTES_APP, source)
    (source / "applet.toml").write_text(
        'name = "notes"\npython_entrypoint = "app:create_api"\n' + (source / "applet.toml").read_text()
    )
    store = AppletStore(database)
    ensure_applet_provisioning(store.engine)
    store.migrate()
    applet_id = UUID(int=101)
    store.publish(read_applet_package(package_applet(source)), "owner", applet_id, 0, frozenset({"owner"}))
    config = MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)
    with TestClient(create_app(config), client=("127.0.0.1", 40000)) as client:
        assert client.get("/notes/api/notes").json() == ["existing"]
        assert client.post("/notes/api/notes", json={"body": "new"}).status_code == 200
        (source / "app.py").write_text(
            (source / "app.py")
            .read_text()
            .replace(
                "def migrate(connection: Connection) -> None:",
                "def migrate(connection: Connection) -> None:\n"
                '    connection.execute(text("DELETE FROM notes"))\n'
                '    raise RuntimeError("migration failed")',
            )
        )
        with pytest.raises(RuntimeError, match="migration failed"):
            store.publish(read_applet_package(package_applet(source)), "owner", applet_id, 1)
        assert store.current_version(applet_id) == 1
        assert client.get("/notes/api/notes").json() == ["existing", "new"]
    store.engine.dispose()
