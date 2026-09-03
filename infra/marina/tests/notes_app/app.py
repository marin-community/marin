# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A one-table app: who is calling, and a list of notes in the app's own schema."""

from fastapi import FastAPI
from marina.apps import Services
from rigging.server_auth import get_verified_identity
from sqlalchemy import Engine, text


def create_api(services: Services) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/whoami")
    def whoami() -> dict[str, str]:
        return {"user": get_verified_identity().user_id, "data": services.data_url}

    @api.post("/notes")
    def add(body: dict[str, str]) -> dict[str, bool]:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO notes (body) VALUES (:body)"), {"body": body["body"]})
        return {"ok": True}

    @api.get("/notes")
    def list_notes() -> list[str]:
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(text("SELECT body FROM notes ORDER BY id"))]

    return api


def migrate(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS notes"))
        conn.execute(text("CREATE TABLE notes (id serial PRIMARY KEY, body text NOT NULL)"))
