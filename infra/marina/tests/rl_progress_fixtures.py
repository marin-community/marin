# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path
from uuid import UUID

import pytest

APPLET_ID = tomllib.loads((Path(__file__).parents[1] / "apps/rl_progress/applet.toml").read_text())["applet_id"]


@pytest.fixture
def rl_progress_client(tmp_path, monkeypatch):
    from marina.applets import AppletStore, package_applet, read_applet_package  # noqa: PLC0415
    from marina.database_setup import ensure_applet_provisioning  # noqa: PLC0415
    from marina.db import UrlDatabase  # noqa: PLC0415
    from marina.server import MarinaConfig, create_app  # noqa: PLC0415
    from marina.testing import test_database  # noqa: PLC0415
    from starlette.testclient import TestClient  # noqa: PLC0415

    with test_database() as local_url:
        database = UrlDatabase(local_url)
        store = AppletStore(database)
        ensure_applet_provisioning(store.engine)
        store.migrate()
        source = Path(__file__).parents[1] / "apps/rl_progress"
        store.publish(read_applet_package(package_applet(source)), "test", UUID(APPLET_ID))
        client = TestClient(
            create_app(MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)),
            client=("127.0.0.1", 40000),
        )

        def request(origin, method, path, *, json_body=None, **kwargs):
            response = client.request(method, path, json=json_body)
            response.raise_for_status()
            return response.json()

        monkeypatch.setattr("marina.client.marina_request", request)
        try:
            yield client, request
        finally:
            client.close()
            store.engine.dispose()
