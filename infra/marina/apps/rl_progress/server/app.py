# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the existing Postgres run summaries without changing stored evidence."""

import hashlib
import json
from datetime import datetime

from fastapi import FastAPI
from marin.rl.evaluation_identity import validate_evaluation_identity
from sqlalchemy import text


def comparison_fields(run: dict) -> dict:
    """Validate evaluation identity and compute coverage, gain and comparison eligibility."""
    evaluation = run["evaluation"]
    validate_evaluation_identity(
        task_ids=evaluation.get("task_ids"),
        task_count=evaluation.get("task_count"),
        membership_sha256=evaluation.get("membership_sha256"),
        protocol=evaluation.get("protocol"),
        protocol_sha256=evaluation.get("protocol_sha256"),
    )
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

    def eligible(result):
        return (
            result is not None
            and expected is not None
            and expected > 0
            and result["scored"] / expected >= run.get("min_scored_fraction", 1.0)
        )

    baseline = run.get("baseline")
    final = run.get("final")
    baseline_eligible, final_eligible = eligible(baseline), eligible(final)
    gain = (
        (final["reward"] - baseline["reward"]) / (run["elapsed_seconds"] / 3600)
        if baseline is not None and final is not None and run["elapsed_seconds"] > 0
        else None
    )
    pareto = key is not None and run["status"] == "succeeded" and final_eligible and run["elapsed_seconds"] > 0
    return {
        "comparison_key": key,
        "expected_membership_count": expected,
        "gain_per_hour": gain,
        "partial_coverage": not (baseline_full and final_full),
        "provisional": key is None or not (baseline_eligible and final_eligible),
        "trusted_gain_eligible": pareto and baseline_eligible and gain is not None,
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
                # Serialize UTC datetimes with a Z suffix.
                for field in ("run_date", "started_at", "completed_at"):
                    if run.get(field):
                        run[field] = datetime.fromisoformat(run[field]).isoformat().replace("+00:00", "Z")
                records.append(run | comparison_fields(run))
                imported_at = row["imported_at"]
                updated_at = imported_at if updated_at is None else max(updated_at, imported_at)
        return {"runs": records, "updated_at": updated_at}

    return api
