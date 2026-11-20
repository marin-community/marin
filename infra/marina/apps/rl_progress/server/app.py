# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import Quick RL evidence and serve immutable run summaries from Postgres."""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from statistics import mean, stdev
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from rigging.telemetry.serialization import json_bytes
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .ingest import sync_attempts

SNOWBALL_MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SNOWBALL_INITIALIZER_ALIASES = {
    f"{SNOWBALL_MODEL}@fe666478454e79fe6027a0bab0388d9abd68f51a",
    "models/snowball-datakit-sft-fe666478454e79fe6027a0bab0388d9abd68f51a@2026.09.21:ead0f35f",
}
HISTORICAL_DEVELOPMENT_PANELS = {
    "Snowball fixed five-family development panel",
    "snowball-fixed-100-development",
    "snowball-fixed-development",
}
PRESETS = (
    "quick-rl",
    "speed-short",
    "data-short",
    "speed-medium",
    "speed-long",
    "learning-medium",
    "data-medium",
    "learning-long",
    "data-long",
    "train-long",
)
DEFAULT_PRESET = "learning-medium"
SNOWBALL_CONTROL_PREFIX = "checkpoints/snowball-rl/control/"
SNOWBALL_HELDOUT_REVISION = "2026.11.06.1"
SNOWBALL_HELDOUT_SET = f"rl-data/task-selection/snowball-test-panel@{SNOWBALL_HELDOUT_REVISION}:88ee12a7"


class ControlRewardPoint(TypedDict):
    run_id: str
    attempt_id: str
    label: str
    reward: float


class ControlRewardSummary(TypedDict):
    preset: str
    preset_revision: str | None
    hardware: dict | None
    eval_set: str
    score_basis: str | None
    points: list[ControlRewardPoint]
    mean: float
    sample_sd: float | None


def run_metadata(run: dict) -> dict[str, str | None]:
    """Resolve display labels and use the recorded series for historical presets."""
    initializer = run.get("initializer_identity")
    base_model = run.get("base_model") or (
        SNOWBALL_MODEL if initializer in SNOWBALL_INITIALIZER_ALIASES else initializer
    )
    model_label = "Grug-67B-A2B Datakit SFT" if base_model == SNOWBALL_MODEL else base_model or "Unknown base model"
    evaluation = run["evaluation"]
    name = evaluation.get("name")
    version = evaluation.get("version")
    if name in HISTORICAL_DEVELOPMENT_PANELS:
        eval_label = "Snowball development · 100 tasks"
    elif name == "panel-910":
        eval_label = "Snowball development · 910 tasks"
    elif name == "test_panel":
        count = evaluation.get("task_count")
        eval_label = f"Snowball held-out test · {count:,} tasks" if count is not None else "Snowball held-out test"
    elif name == "quick-rl-validation":
        selection = version.partition("@")[2].partition(":")[0] if version else ""
        eval_label = f"Quick RL validation · {selection}" if selection else "Quick RL validation"
    else:
        eval_label = name or "Unknown eval set"
    series = run.get("series")
    preset = run.get("preset") or (series if isinstance(series, str) and series in PRESETS else "unspecified")
    revision = run.get("preset_revision")
    if preset == "learning-medium" and version == SNOWBALL_HELDOUT_SET:
        revision = revision or SNOWBALL_HELDOUT_REVISION
    return {"preset": preset, "preset_revision": revision, "model_label": model_label, "eval_set_label": eval_label}


def migrate(connection: Connection) -> None:
    """Create the run store and the separate historical read path."""
    connection.execute(
        text(
            """CREATE TABLE IF NOT EXISTS rl_runs (
                run_id text NOT NULL,
                attempt_id text NOT NULL,
                record jsonb NOT NULL,
                imported_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (run_id, attempt_id)
            )"""
        )
    )
    connection.execute(text("CREATE TABLE IF NOT EXISTS rl_archived_runs (LIKE rl_runs INCLUDING ALL)"))
    connection.execute(
        text(
            """CREATE OR REPLACE VIEW rl_dashboard_runs AS
                SELECT run_id, attempt_id, record, imported_at FROM rl_runs
                UNION ALL
                SELECT archived.run_id, archived.attempt_id, archived.record, archived.imported_at
                FROM rl_archived_runs AS archived
                WHERE NOT EXISTS (
                    SELECT 1 FROM rl_runs AS live
                    WHERE live.run_id = archived.run_id AND live.attempt_id = archived.attempt_id
                )"""
        )
    )


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
        protocol = json_bytes(evaluation["protocol"])
        if hashlib.sha256(protocol).hexdigest() != evaluation.get("protocol_sha256"):
            raise ValueError("Evaluation protocol does not match its recorded identity")
    expected = evaluation.get("task_count")
    if expected is None and evaluation.get("task_ids") is not None:
        expected = len(evaluation["task_ids"])
    key = None
    if evaluation.get("task_set_sha256") and evaluation.get("protocol_sha256") and run.get("timing_scope"):
        identity: dict[str, object] = {name: evaluation[name] for name in ("task_set_sha256", "protocol_sha256")}
        identity["timing_scope"] = run["timing_scope"]
        if run.get("hardware"):
            identity["hardware"] = run["hardware"]
        if evaluation.get("score_basis"):
            identity["score_basis"] = evaluation["score_basis"]
        key = hashlib.sha256(json_bytes(identity)).hexdigest()

    def complete(result):
        return result is not None and expected is not None and result["attempted"] == result["scored"] == expected

    baseline_full = complete(run.get("baseline"))
    final_full = complete(run.get("final"))
    final = run.get("final")
    baseline = run.get("baseline")
    elapsed = run.get("elapsed_seconds")
    has_timing = elapsed is not None and elapsed > 0
    gain = (final["reward"] - baseline["reward"]) / (elapsed / 3600) if final and baseline and has_timing else None
    pareto = key is not None and run["status"] == "succeeded" and final_full and has_timing
    return {
        "comparison_key": key,
        "expected_membership_count": expected,
        "final_coverage_complete": final_full,
        "gain_per_hour": gain,
        "partial_coverage": not (baseline_full and final_full),
        "provisional": key is None or not has_timing or not (baseline_full and final_full),
        "trusted_gain_eligible": pareto and baseline_full,
        "pareto_eligible": pareto,
    }


def control_reward_summaries(records: list[dict]) -> list[ControlRewardSummary]:
    """Summarize independently named control repeats with matching run identities."""
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for run in sorted(
        records,
        key=lambda run: datetime.fromisoformat(run["completed_at"]).timestamp() if run.get("completed_at") else 0,
    ):
        is_control = run.get("is_control", run["run_id"].startswith(SNOWBALL_CONTROL_PREFIX))
        archive_cohort = run.get("control_reward_cohort")
        score_basis = run.get("evaluation", {}).get("score_basis")
        archive_control = archive_cohort and run.get("status") == "succeeded" and run.get("final_coverage_complete")
        if not is_control or not (run["pareto_eligible"] or archive_control):
            continue
        reward = run["final"]["reward"]
        if not math.isfinite(reward):
            continue
        key = (
            run["preset"],
            run.get("preset_revision"),
            archive_cohort or run["comparison_key"],
            score_basis,
            run.get("initializer_identity"),
            run.get("data_identity"),
            run.get("source_revision"),
            run.get("optimizer_steps"),
            json_bytes(run.get("engine")),
        )
        groups[key][run["run_id"]] = run

    summaries: list[ControlRewardSummary] = []
    for group in groups.values():
        members = list(group.values())
        sample = [
            ControlRewardPoint(
                run_id=run["run_id"], attempt_id=run["attempt_id"], label=run["label"], reward=run["final"]["reward"]
            )
            for run in members
        ]
        metadata = members[0]
        rewards = [point["reward"] for point in sample]
        summaries.append(
            ControlRewardSummary(
                preset=metadata["preset"],
                preset_revision=metadata.get("preset_revision"),
                hardware=metadata.get("hardware"),
                eval_set=metadata["eval_set_label"],
                score_basis=metadata.get("evaluation", {}).get("score_basis"),
                points=sample,
                mean=mean(rewards),
                sample_sd=stdev(rewards) if len(rewards) > 1 else None,
            )
        )
    return summaries


def create_api(services):
    api = FastAPI()
    engine = services.engine()

    @api.get("/runs")
    def runs():
        records = []
        updated_at = None
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT record, imported_at FROM rl_dashboard_runs ORDER BY run_id, attempt_id")
            ).mappings()
            for row in rows:
                run = dict(row["record"])
                run.pop("grader_report", None)
                # Match the prior API's UTC datetime serialization.
                for field in ("run_date", "started_at", "completed_at"):
                    if run.get(field):
                        run[field] = datetime.fromisoformat(run[field]).isoformat().replace("+00:00", "Z")
                records.append(run | comparison_fields(run) | run_metadata(run))
                imported_at = row["imported_at"]
                updated_at = imported_at if updated_at is None else max(updated_at, imported_at)
        return {
            "runs": records,
            "presets": PRESETS,
            "default_preset": DEFAULT_PRESET,
            "control_rewards": control_reward_summaries(records),
            "updated_at": updated_at,
        }

    @api.post("/sync")
    def sync():
        result = sync_attempts(engine)
        if result.errors:
            raise HTTPException(502, detail=asdict(result))
        return asdict(result)

    @api.get("/graders")
    def graders(run_id: str, attempt_id: str):
        with engine.connect() as connection:
            record = connection.execute(
                text("SELECT record FROM rl_dashboard_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"),
                {"run_id": run_id, "attempt_id": attempt_id},
            ).scalar_one_or_none()
        if record is None:
            raise HTTPException(404, "Run attempt not found")
        return {"label": record["label"], "report": record.get("grader_report")}

    return api
