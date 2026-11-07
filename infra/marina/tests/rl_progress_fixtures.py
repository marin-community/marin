# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from marina.applets import AppletStore, package_applet, read_applet_package
from marina.database_setup import ensure_applet_provisioning
from marina.db import UrlDatabase
from marina.server import MarinaConfig, create_app
from marina.testing import isolated_database
from starlette.testclient import TestClient


def rl_progress_client(tmp_path, database_url, monkeypatch):
    from experiments.post_training import publish_rl_progress as publication  # noqa: PLC0415

    with isolated_database(database_url) as local_url:
        database = UrlDatabase(local_url)
        store = AppletStore(database)
        ensure_applet_provisioning(store.engine)
        store.migrate()
        source = Path(__file__).parents[1] / "apps/rl_progress"
        store.publish(read_applet_package(package_applet(source)), "test", UUID(publication.APPLET_ID))
        client = TestClient(
            create_app(MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)),
            client=("127.0.0.1", 40000),
        )
        query_path = f"/a/{publication.APPLET_ID}/query"

        def request(origin, method, path, *, json_body=None):
            response = client.request(method, path, json=json_body)
            response.raise_for_status()
            return response.json()

        monkeypatch.setattr(publication, "marina_request", request)
        request(
            "http://localhost",
            "POST",
            query_path,
            json_body={
                "sql": (
                    "CREATE TABLE rl_runs (run_id text, attempt_id text, record jsonb NOT NULL, "
                    "imported_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (run_id, attempt_id))"
                ),
                "parameters": {},
            },
        )
        try:
            yield client, request
        finally:
            client.close()
            store.engine.dispose()
