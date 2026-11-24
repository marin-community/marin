# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve RL progress records stored in Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from finestore.rl.rl_progress import AttemptSummary
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ProgrammingError

_RETAINED_FIELDS = ("issue", "gate", "gate_result", "model", "evaldash_run_ids", "grader_report")
_RETAINED_VALUES = ", ".join(
    f"'{key}', COALESCE(NULLIF(EXCLUDED.record -> '{key}', 'null'::jsonb), rl_runs.record -> '{key}', 'null'::jsonb)"
    for key in _RETAINED_FIELDS
)
_UPSERT = text(
    f"""INSERT INTO rl_runs (run_id, attempt_id, record)
    VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
    ON CONFLICT (run_id, attempt_id) DO UPDATE
      SET record = EXCLUDED.record || jsonb_build_object({_RETAINED_VALUES})
    RETURNING record"""
)


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
            encoded = json.dumps(record, allow_nan=False)
            summary = AttemptSummary.model_validate_json(encoded, strict=True)
        except ValidationError as error:
            raise HTTPException(
                422, detail=error.errors(include_input=False, include_context=False, include_url=False)
            ) from error
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error
        try:
            with engine.begin() as connection:
                incoming = summary.model_dump(mode="json")
                stored = connection.execute(
                    _UPSERT,
                    {"run_id": summary.run_id, "attempt_id": summary.attempt_id, "record": summary.model_dump_json()},
                ).scalar_one()
                repaired = dict(stored)
                for field in _RETAINED_FIELDS:
                    if incoming[field] is None:
                        try:
                            json.dumps(repaired[field], allow_nan=False)
                        except ValueError:
                            repaired[field] = None
                try:
                    summary = AttemptSummary.model_validate_json(json.dumps(repaired, allow_nan=False), strict=True)
                except ValidationError as error:
                    invalid = {item["loc"][0] for item in error.errors() if item["loc"]}
                    for field in _RETAINED_FIELDS:
                        if field in invalid and incoming[field] is None:
                            repaired[field] = None
                    summary = AttemptSummary.model_validate_json(json.dumps(repaired, allow_nan=False), strict=True)
                normalized = summary.model_dump(mode="json")
                if normalized != stored:
                    connection.execute(
                        text(
                            "UPDATE rl_runs SET record = CAST(:record AS jsonb) "
                            "WHERE run_id = :run_id AND attempt_id = :attempt_id"
                        ),
                        {
                            "run_id": summary.run_id,
                            "attempt_id": summary.attempt_id,
                            "record": summary.model_dump_json(),
                        },
                    )
                return normalized
        except ProgrammingError as error:
            details = error.orig.args[0]
            if not isinstance(details, dict) or details.get("C") not in {"22P02", "22P05"}:
                raise
            raise HTTPException(422, detail="Record contains text PostgreSQL cannot store as JSON.") from error

    @api.get("/runs")
    def runs():
        records = []
        skipped = []
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT run_id, attempt_id, CASE WHEN jsonb_typeof(record) != 'object' "
                    "OR record -> 'grader_report' = 'null'::jsonb "
                    "THEN record ELSE record - 'grader_report' END AS record, "
                    "record -> 'grader_report' IS NOT NULL AND record -> 'grader_report' != 'null'::jsonb "
                    "AS has_grader_report FROM rl_runs ORDER BY run_id, attempt_id"
                )
            ).mappings()
            for row in rows:
                try:
                    record = row["record"]
                    if isinstance(record, dict) and record.get("issue") == "opted_out":
                        record = {**record, "issue": None}
                    encoded = json.dumps(record, allow_nan=False)
                    summary = AttemptSummary.model_validate_json(encoded, strict=True)
                    if (summary.run_id, summary.attempt_id) != (row["run_id"], row["attempt_id"]):
                        raise ValueError("Stored key disagrees with the record identity")
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
                if row["has_grader_report"]:
                    record["has_grader_report"] = True
                records.append(record)
        return {"runs": records, "skipped_records": skipped}

    @api.get("/graders")
    def graders(run_id: str, attempt_id: str):
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT record ->> 'label' AS label, record -> 'grader_report' AS report "
                        "FROM rl_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"
                    ),
                    {"run_id": run_id, "attempt_id": attempt_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise HTTPException(404, detail="Unknown native attempt")
        return dict(row)

    return api
