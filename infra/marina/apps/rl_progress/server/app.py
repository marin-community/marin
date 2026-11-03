# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the existing Postgres run summaries without changing stored evidence."""

import hashlib
import json
from datetime import datetime

from fastapi import FastAPI
from sqlalchemy import text


def comparison_fields(run: dict) -> dict:
    """Retain the dashboard's content/protocol/timing comparison boundary."""
    evaluation = run["evaluation"]
    task_ids = evaluation.get("task_ids")
    if task_ids is not None:
        membership = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
        if len(set(task_ids)) != len(task_ids) or membership != evaluation.get("membership_sha256"):
            raise ValueError("Evaluation membership does not match its recorded identity")
        if evaluation.get("task_count") is not None and evaluation["task_count"] != len(task_ids):
            raise ValueError("Evaluation task count disagrees with membership")
    if evaluation.get("protocol") is not None:
        protocol = json.dumps(evaluation["protocol"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(protocol.encode()).hexdigest() != evaluation.get("protocol_sha256"):
            raise ValueError("Evaluation protocol does not match its recorded identity")
    expected = evaluation.get("task_count")
    if expected is None and evaluation.get("task_ids") is not None:
        expected = len(evaluation["task_ids"])
    key = None
    if evaluation.get("task_set_sha256") and evaluation.get("protocol_sha256") and run.get("timing_scope"):
        identity = {name: evaluation[name] for name in ("task_set_sha256", "protocol_sha256")}
        identity["timing_scope"] = run["timing_scope"]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def complete(result):
        return result is not None and expected is not None and result["attempted"] == result["scored"] == expected

    baseline_full = complete(run.get("baseline"))
    final_full = complete(run.get("final"))
    final = run.get("final")
    gain = None if final is None else (final["reward"] - run["baseline"]["reward"]) / (run["elapsed_seconds"] / 3600)
    pareto = key is not None and run["status"] == "succeeded" and final_full
    return {
        "comparison_key": key,
        "expected_membership_count": expected,
        "gain_per_hour": gain,
        "partial_coverage": not (baseline_full and final_full),
        "provisional": key is None or not (baseline_full and final_full),
        "trusted_gain_eligible": pareto and baseline_full,
        "pareto_eligible": pareto,
    }


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.get("/runs")
    def runs():
        records = []
        updated_at = None
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT record, imported_at FROM rl_runs ORDER BY run_id, attempt_id")
            ).mappings()
            for row in rows:
                run = dict(row["record"])
                # Match the prior API's UTC datetime serialization.
                for field in ("run_date", "started_at", "completed_at"):
                    if run.get(field):
                        run[field] = datetime.fromisoformat(run[field]).isoformat().replace("+00:00", "Z")
                records.append(run | comparison_fields(run))
                imported_at = row["imported_at"]
                updated_at = imported_at if updated_at is None else max(updated_at, imported_at)
        return {"runs": records, "updated_at": updated_at}

    return api
