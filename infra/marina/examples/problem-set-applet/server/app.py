# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small dynamic backend used by the Marina applet example."""

from fastapi import FastAPI
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection


def migrate(connection: Connection) -> None:
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS problems ("
            "id INTEGER PRIMARY KEY, prompt TEXT NOT NULL, answer INTEGER NOT NULL)"
        )
    )
    connection.execute(
        text(
            "INSERT INTO problems (id, prompt, answer) VALUES "
            "(1, '12 * 7', 84), (2, '144 ÷ 12', 12), (3, '19 + 28', 47) "
            "ON CONFLICT (id) DO NOTHING"
        )
    )


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/problems")
    def problems() -> list[dict[str, object]]:
        with engine.connect() as connection:
            return [
                dict(row) for row in connection.execute(text("SELECT id, prompt FROM problems ORDER BY id")).mappings()
            ]

    @api.get("/revision")
    def revision() -> dict[str, int]:
        return {"version": services.version}

    return api
