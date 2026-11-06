# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read saved measurements through Marina's role-scoped engine."""

from typing import Any

from fastapi import FastAPI, HTTPException
from marina.apps import RegisteredApi, Services, registered_api
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .schema import SCHEMA_SQL


def migrate(connection: Connection) -> None:
    for statement in SCHEMA_SQL.split(";"):
        if statement.strip():
            connection.execute(text(statement))


def create_api(services: Services) -> RegisteredApi:
    api = FastAPI()
    engine = services.engine()

    @api.get("/runs")
    def runs() -> list[dict[str, Any]]:
        with engine.connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT i.run FROM knotmeter_current c JOIN knotmeter_imports i ON i.id = c.import_id "
                        "ORDER BY i.run->>'date' DESC, i.run->>'model', (i.run->>'concurrency')::integer"
                    )
                )
            ]

    @api.get("/runs/{run_key}")
    def run(run_key: str) -> dict[str, Any]:
        with engine.connect() as connection:
            payload = connection.execute(
                text(
                    "SELECT i.run FROM knotmeter_current c "
                    "JOIN knotmeter_imports i ON i.id = c.import_id WHERE c.run_key = :key"
                ),
                {"key": run_key},
            ).scalar_one_or_none()
        if payload is None:
            raise HTTPException(404, "Run has not been imported")
        return payload

    @api.get("/runs/{run_key}/requests")
    def requests(run_key: str) -> dict[str, Any]:
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT i.run->>'origin_ns' AS origin_ns, "
                        "jsonb_agg(r.payload ORDER BY r.ordinal) AS requests "
                        "FROM knotmeter_current c JOIN knotmeter_imports i ON i.id = c.import_id "
                        "JOIN knotmeter_requests r ON r.import_id = i.id "
                        "WHERE c.run_key = :key GROUP BY i.id"
                    ),
                    {"key": run_key},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise HTTPException(404, "Run has not been imported")
        return dict(row)

    @api.get("/corpus/{conversation_id}")
    def corpus(conversation_id: str, dataset: str, revision: str) -> dict[str, Any]:
        with engine.connect() as connection:
            structure = connection.execute(
                text(
                    "SELECT structure FROM knotmeter_corpus "
                    "WHERE dataset = :dataset AND revision = :revision AND conversation_id = :cid"
                ),
                {"dataset": dataset, "revision": revision, "cid": conversation_id},
            ).scalar_one_or_none()
        if structure is None:
            raise HTTPException(404, "This corpus conversation has not been loaded")
        return {"dataset": dataset, "revision": revision, "conversation_id": conversation_id, "structure": structure}

    return registered_api(api)
