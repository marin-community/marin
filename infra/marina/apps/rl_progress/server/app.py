# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve RL progress records stored in Postgres."""

import json

from fastapi import FastAPI
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
