# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve RL progress records stored in Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from finestore.rl.rl_progress import AttemptSummary, completed_controls
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ProgrammingError

_RETAINED_FIELDS = (
    "issue",
    "gate",
    "gate_result",
    "model",
    "grader_report",
    "is_control",
    "evaluation_data_identity",
    "evaluation_task_set",
    "evaluation_protocol_identity",
    "native_run_id",
    "base_commit",
    "base_preset_version",
    "launch_config",
)
_UPSERT = text(
    """INSERT INTO rl_runs (run_id, attempt_id, record)
    VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
    ON CONFLICT (run_id, attempt_id) DO UPDATE
      SET record = CASE
        WHEN EXCLUDED.record ->> 'status' = 'running'
          AND rl_runs.record ->> 'status' IN ('succeeded', 'failed') THEN rl_runs.record
        ELSE EXCLUDED.record || (
            SELECT jsonb_object_agg(key, COALESCE(NULLIF(EXCLUDED.record -> key, 'null'::jsonb),
                                                  rl_runs.record -> key, 'null'::jsonb))
            FROM unnest(CAST(:retained_fields AS text[])) AS retained(key)
        )
      END
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
    connection.execute(
        text(
            """CREATE TABLE IF NOT EXISTS rl_legacy_runs (
            run_id text NOT NULL,
            attempt_id text NOT NULL,
            record jsonb NOT NULL,
            imported_at timestamptz NOT NULL
        )"""
        )
    )
    connection.execute(
        text(
            """WITH legacy AS (
            DELETE FROM rl_runs
            WHERE NOT record ? 'format_version' AND record ?& ARRAY['evaluation', 'baseline', 'final']
            RETURNING run_id, attempt_id, record, imported_at
        ) INSERT INTO rl_legacy_runs SELECT * FROM legacy"""
        )
    )
    connection.execute(
        text(
            """UPDATE rl_runs SET record = record || jsonb_build_object('format_version', 1)
            WHERE jsonb_typeof(record) = 'object' AND NOT record ? 'format_version'"""
        )
    )


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.post("/runs")
    def publish(record: dict[str, Any]) -> dict:
        try:
            encoded = json.dumps(record, allow_nan=False)
            summary = AttemptSummary.model_validate_json(encoded, extra="forbid")
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
                    {
                        "run_id": summary.run_id,
                        "attempt_id": summary.attempt_id,
                        "record": summary.model_dump_json(),
                        "retained_fields": list(_RETAINED_FIELDS),
                    },
                ).scalar_one()
                repaired = dict(stored)
                for field in _RETAINED_FIELDS:
                    if incoming.get(field) is None:
                        try:
                            json.dumps(repaired.get(field), allow_nan=False)
                        except ValueError:
                            repaired[field] = None
                try:
                    summary = AttemptSummary.model_validate_json(json.dumps(repaired, allow_nan=False), strict=True)
                except ValidationError as error:
                    invalid = {item["loc"][0] for item in error.errors() if item["loc"]}
                    for field in _RETAINED_FIELDS:
                        if field in invalid and incoming.get(field) is None:
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
        summaries = []
        skipped = []
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """SELECT run_id, attempt_id,
                    CASE WHEN jsonb_typeof(record) = 'object' THEN record - 'grader_report' - 'launch_config'
                         ELSE record END AS record,
                    COALESCE(record -> 'grader_report' <> 'null'::jsonb, false) AS has_grader_report
                    FROM rl_runs ORDER BY run_id, attempt_id"""
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
                record["has_grader_report"] = row["has_grader_report"]
                records.append(record)
                summaries.append(summary)
        controls = {(summary.run_id, summary.attempt_id) for summary in completed_controls(summaries)}
        for record in records:
            record["is_main_control"] = (record["run_id"], record["attempt_id"]) in controls
        return {"runs": records, "skipped_records": skipped}

    @api.get("/configuration")
    def configuration(run_id: str, attempt_id: str):
        with engine.connect() as connection:
            config = connection.execute(
                text(
                    "SELECT record -> 'launch_config' FROM rl_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"
                ),
                {"run_id": run_id, "attempt_id": attempt_id},
            ).scalar_one_or_none()
        if config is None:
            raise HTTPException(404, detail="No configuration recorded for this attempt")
        return config

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
