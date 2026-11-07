# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from rigging.grant_database import PostgresGrantDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool


@pytest.fixture
def openfga_http(tmp_path):
    """Canned remote decisions test the adapter; prove.py exercises the real engine."""
    state = {"allowed": True, "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(
                {"path": self.path, "payload": payload, "authorization": self.headers.get("Authorization")}
            )
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"allowed": state["allowed"]}).encode())

        def log_message(self, *_args):
            pass

    token_file = tmp_path / "openfga.token"
    token_file.write_text("synthetic-openfga-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield {
                "api_url": f"http://127.0.0.1:{server.server_port}",
                "token_file": token_file,
                "store_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "model_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                "state": state,
            }
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def postgres_database_url():
    """Use an explicitly supplied disposable PostgreSQL instance for replica tests."""
    admin_url = make_url(os.environ["MARIN_AUTH_TEST_POSTGRES_URL"])
    name = "auth_test_" + uuid.uuid4().hex
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
        yield admin_url.set(database=name).render_as_string(hide_password=False)
    finally:
        with engine.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        engine.dispose()


@pytest.fixture
def postgres_auth_database(postgres_database_url):
    database = PostgresGrantDatabase(postgres_database_url, ssl_context=None)
    try:
        yield database
    finally:
        database.close()
