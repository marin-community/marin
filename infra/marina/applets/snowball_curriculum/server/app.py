# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browse a separately loaded, immutable Snowball training selection."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection


def migrate(connection: Connection) -> None:
    statements = [
        "CREATE TABLE IF NOT EXISTS pool_loads (load_id TEXT PRIMARY KEY, metadata TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS pool_current (name TEXT PRIMARY KEY, load_id TEXT)",
        "INSERT INTO pool_current (name) VALUES ('latest') ON CONFLICT DO NOTHING",
        """CREATE TABLE IF NOT EXISTS pool_tasks (
            load_id TEXT NOT NULL, contract_id TEXT NOT NULL, task_id TEXT NOT NULL,
            source_id TEXT NOT NULL, family TEXT NOT NULL, difficulty TEXT NOT NULL,
            preview TEXT NOT NULL, search_text TEXT NOT NULL, detail TEXT NOT NULL,
            PRIMARY KEY (load_id, contract_id))""",
        "CREATE INDEX IF NOT EXISTS pool_filter ON pool_tasks (load_id, source_id, family, difficulty)",
        "CREATE INDEX IF NOT EXISTS pool_search ON pool_tasks USING GIN (lower(search_text) gin_trgm_ops)",
    ]
    for statement in statements:
        connection.execute(text(statement))


def current_pool(connection: Connection) -> dict[str, Any]:
    row = (
        connection.execute(
            text("SELECT l.* FROM pool_current c JOIN pool_loads l USING (load_id) WHERE c.name = 'latest'")
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(503, "The training pool has not been loaded")
    return dict(row)


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/overview")
    def overview() -> dict[str, Any]:
        with engine.connect() as connection:
            pool = current_pool(connection)
            groups = {}
            for column in ("source_id", "family", "difficulty"):
                groups[column] = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            f"SELECT {column} AS value, COUNT(*) AS count FROM pool_tasks "
                            f"WHERE load_id = :load_id GROUP BY {column} ORDER BY count DESC, {column}"
                        ),
                        {"load_id": pool["load_id"]},
                    ).mappings()
                ]
            return {"load_id": pool["load_id"], "metadata": json.loads(pool["metadata"]), "groups": groups}

    @api.get("/tasks")
    def tasks(
        source: str = "",
        family: str = "",
        difficulty: str = "",
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(40, ge=1, le=100),
    ) -> dict[str, Any]:
        with engine.connect() as connection:
            pool = current_pool(connection)
            parameters: dict[str, Any] = {"load_id": pool["load_id"], "limit": limit, "offset": offset}
            clauses = ["load_id = :load_id"]
            for column, value in (("source_id", source), ("family", family), ("difficulty", difficulty)):
                if value:
                    clauses.append(f"{column} = :{column}")
                    parameters[column] = value
            if q:
                clauses.append("LOWER(search_text) LIKE :query ESCAPE '\\'")
                parameters["query"] = "%" + q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where = " AND ".join(clauses)
            total = connection.execute(text(f"SELECT COUNT(*) FROM pool_tasks WHERE {where}"), parameters).scalar_one()
            rows = connection.execute(
                text(
                    "SELECT contract_id, task_id, source_id, family, difficulty, preview FROM pool_tasks "
                    f"WHERE {where} ORDER BY source_id, task_id, contract_id LIMIT :limit OFFSET :offset"
                ),
                parameters,
            ).mappings()
            return {"total": total, "offset": offset, "limit": limit, "tasks": [dict(row) for row in rows]}

    @api.get("/task/{contract_id}")
    def task(contract_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            pool = current_pool(connection)
            detail = connection.execute(
                text("SELECT detail FROM pool_tasks WHERE load_id = :load_id AND contract_id = :contract_id"),
                {"load_id": pool["load_id"], "contract_id": contract_id},
            ).scalar_one_or_none()
            if detail is None:
                raise HTTPException(404, "Task not found")
            return json.loads(detail)

    return api
