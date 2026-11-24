# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve RL progress records stored in Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from finestore.rl.rl_progress import AttemptSummary, ComparisonRole
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ProgrammingError

_HISTORY_FIELDS = (
    "native_run_id",
    "base_preset_revision",
    "control_preset",
    "evaluation_data_identity",
    "evaluation_protocol_identity",
)
_RETAINED_FIELDS = (
    "issue",
    "model",
    "grader_report",
    *(field for field in _HISTORY_FIELDS if field in AttemptSummary.model_fields),
)
_RETAINED_VALUES = ", ".join(
    f"'{key}', COALESCE(NULLIF(EXCLUDED.record -> '{key}', 'null'::jsonb), rl_runs.record -> '{key}', 'null'::jsonb)"
    for key in _RETAINED_FIELDS
)
_KEEP_HISTORY_GATE = """EXCLUDED.record ->> 'format_version' = '1'
            AND rl_runs.record ->> 'format_version' = '2'
            AND jsonb_typeof(rl_runs.record -> 'gate_result') = 'object'"""
_UPSERT = text(
    f"""INSERT INTO rl_runs (run_id, attempt_id, record)
    VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
    ON CONFLICT (run_id, attempt_id) DO UPDATE
      SET record = CASE
        WHEN EXCLUDED.record ->> 'status' = 'running'
          AND rl_runs.record ->> 'status' IN ('succeeded', 'failed') THEN rl_runs.record
        ELSE EXCLUDED.record || jsonb_build_object(
          {_RETAINED_VALUES},
          'gate', CASE WHEN {_KEEP_HISTORY_GATE} THEN rl_runs.record -> 'gate'
            ELSE COALESCE(NULLIF(EXCLUDED.record -> 'gate', 'null'::jsonb), rl_runs.record -> 'gate')
          END,
          'gate_result', CASE WHEN {_KEEP_HISTORY_GATE} THEN rl_runs.record -> 'gate_result'
            ELSE COALESCE(NULLIF(EXCLUDED.record -> 'gate_result', 'null'::jsonb), rl_runs.record -> 'gate_result')
          END,
          'format_version', CASE WHEN '2' IN (EXCLUDED.record ->> 'format_version', rl_runs.record ->> 'format_version')
            THEN 2 ELSE 1 END,
          'evaldash_run_ids', (
            SELECT jsonb_agg(link ORDER BY first_seen)
            FROM (
              SELECT link, min(position) AS first_seen
              FROM jsonb_array_elements(
                CASE WHEN jsonb_typeof(rl_runs.record -> 'evaldash_run_ids') = 'array'
                  THEN rl_runs.record -> 'evaldash_run_ids' ELSE '[]'::jsonb END
                || COALESCE(NULLIF(EXCLUDED.record -> 'evaldash_run_ids', 'null'::jsonb), '[]'::jsonb)
              ) WITH ORDINALITY AS links(link, position)
              WHERE jsonb_typeof(link) = 'string' AND link <> '""'::jsonb
              GROUP BY link
            ) saved_links
          )
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
                retained_fields = (*_RETAINED_FIELDS, "gate", "gate_result")
                legacy_gate = incoming["format_version"] == 1 and repaired.get("format_version") == 2
                for field in retained_fields:
                    if incoming.get(field) is None or (legacy_gate and field == "gate_result"):
                        try:
                            json.dumps(repaired.get(field), allow_nan=False)
                        except ValueError:
                            repaired[field] = incoming.get(field)
                try:
                    summary = AttemptSummary.model_validate_json(json.dumps(repaired, allow_nan=False), strict=True)
                except ValidationError as error:
                    invalid = {item["loc"][0] for item in error.errors() if item["loc"]}
                    for field in retained_fields:
                        if field in invalid and (
                            incoming.get(field) is None or (legacy_gate and field == "gate_result")
                        ):
                            repaired[field] = incoming.get(field)
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
                if incoming["format_version"] == 1:
                    legacy = {key: value for key, value in normalized.items() if key not in _HISTORY_FIELDS}
                    legacy["format_version"] = 1
                    if legacy_gate and (gate := legacy.get("gate_result")) is not None:
                        legacy["gate_result"] = {
                            key: value
                            for key, value in gate.items()
                            if key not in {"control_commit", "control_revision"}
                        } | {
                            "comparisons": [row for row in gate["comparisons"] if row["role"] != ComparisonRole.SLICE],
                            "invariants": [fact for fact in gate["invariants"] if fact["replicate"] is not None],
                        }
                    return legacy
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
                    """SELECT run_id, attempt_id,
                    CASE WHEN jsonb_typeof(record) = 'object' THEN record - 'grader_report'
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
