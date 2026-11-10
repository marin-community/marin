# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve RL progress records stored in Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from marin.rl.rl_progress import RunSummary
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Connection


def migrate(connection: Connection) -> None:
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS rl_runs (
            run_id text NOT NULL,
            attempt_id text NOT NULL,
            record jsonb NOT NULL,
            imported_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (run_id, attempt_id)
        )
    """
        )
    )


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.post("/runs")
    def publish(record: dict[str, Any]) -> dict:
        try:
            summary = RunSummary.model_validate(record)
            encoded = json.dumps(record, allow_nan=False)
        except ValidationError as error:
            raise HTTPException(
                422, detail=error.errors(include_input=False, include_context=False, include_url=False)
            ) from error
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error
        with engine.begin() as connection:
            return connection.execute(
                text(
                    """INSERT INTO rl_runs (run_id, attempt_id, record)
                    VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
                    ON CONFLICT (run_id, attempt_id) DO UPDATE SET record = jsonb_set(
                        EXCLUDED.record, '{issue}',
                        COALESCE(NULLIF(EXCLUDED.record -> 'issue', 'null'::jsonb),
                                 rl_runs.record -> 'issue', 'null'::jsonb))
                    RETURNING record"""
                ),
                {"run_id": summary.run_id, "attempt_id": summary.attempt_id, "record": encoded},
            ).scalar_one()

    @api.get("/runs")
    def runs():
        records = []
        skipped = []
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT run_id, attempt_id, record FROM rl_runs ORDER BY run_id, attempt_id")
            ).mappings()
            for row in rows:
                try:
                    summary = RunSummary.model_validate(row["record"])
                    if (summary.run_id, summary.attempt_id) != (row["run_id"], row["attempt_id"]):
                        raise ValueError("Stored key disagrees with the record identity")
                    json.dumps(row["record"], allow_nan=False)
                except (ValidationError, ValueError) as error:
                    reason = (
                        "; ".join(
                            f"{'.'.join(map(str, item['loc']))}: {item['type']}"
                            for item in error.errors(include_input=False)
                        )
                        if isinstance(error, ValidationError)
                        else str(error)
                    )
                    skipped.append({"run_id": row["run_id"], "attempt_id": row["attempt_id"], "error": reason})
                    continue
                records.append(row["record"])
        return {"runs": records, "skipped_records": skipped}

    return api
