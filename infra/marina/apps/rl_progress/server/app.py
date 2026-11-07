# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the existing Postgres run summaries without changing stored evidence."""

import hashlib
import json
import math
from datetime import datetime

from fastapi import FastAPI
from marin.rl.rl_progress import EvaluationResult, RunSummary
from pydantic import ValidationError
from sqlalchemy import text


def comparison_fields(run: RunSummary) -> dict:
    """Compute coverage, gain and comparison eligibility from a validated attempt."""
    evaluation = run.evaluation
    expected = evaluation.task_count
    key = None
    if evaluation.task_set_sha256 and evaluation.protocol_sha256:
        identity = {
            "task_set_sha256": evaluation.task_set_sha256,
            "membership_sha256": evaluation.membership_sha256,
            "protocol_sha256": evaluation.protocol_sha256,
            "timing_scope": run.timing_scope,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def complete(result: EvaluationResult | None) -> bool:
        return result is not None and result.attempted == result.scored == expected

    def eligible(result: EvaluationResult | None) -> bool:
        return result is not None and result.scored / expected >= run.min_scored_fraction

    baseline_full, final_full = complete(run.baseline), complete(run.final)
    baseline_eligible, final_eligible = eligible(run.baseline), eligible(run.final)
    timed = run.elapsed_seconds is not None and run.elapsed_seconds > 0
    gain = (
        (run.final.reward - run.baseline.reward) / run.elapsed_seconds * 3600
        if run.elapsed_seconds and run.final is not None and run.baseline is not None
        else None
    )
    if gain is not None and not math.isfinite(gain):
        raise ValueError("Derived reward gain per hour must be finite")
    pareto = key is not None and run.status == "succeeded" and run.verdict != "invalid" and final_eligible and timed
    return {
        "comparison_key": key,
        "expected_membership_count": expected,
        "gain_per_hour": gain,
        "partial_coverage": not (baseline_full and final_full),
        "provisional": run.verdict == "invalid" or key is None or not (baseline_eligible and final_eligible and timed),
        "trusted_gain_eligible": pareto and baseline_eligible,
        "pareto_eligible": pareto,
    }


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.get("/runs")
    def runs():
        records = []
        skipped = []
        updated_at = None
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT run_id, attempt_id, record, imported_at FROM rl_runs ORDER BY run_id, attempt_id")
            ).mappings()
            for row in rows:
                try:
                    summary = RunSummary.model_validate(row["record"])
                    if (summary.run_id, summary.attempt_id) != (row["run_id"], row["attempt_id"]):
                        raise ValueError("Stored key disagrees with the record identity")
                    comparison = comparison_fields(summary)
                    json.dumps(dict(row["record"]) | comparison, allow_nan=False)
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
                run = dict(row["record"])
                for field in ("run_date", "started_at", "completed_at"):
                    if run.get(field):
                        run[field] = datetime.fromisoformat(run[field]).isoformat().replace("+00:00", "Z")
                records.append(run | comparison)
                imported_at = row["imported_at"]
                updated_at = imported_at if updated_at is None else max(updated_at, imported_at)
        return {"runs": records, "updated_at": updated_at, "skipped_records": skipped}

    return api
