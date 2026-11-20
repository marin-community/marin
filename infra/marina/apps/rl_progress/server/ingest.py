# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import terminal Quick RL attempts from CoreWeave into the dashboard."""

import hashlib
import json
import logging
import math
import re
from contextlib import closing
from dataclasses import dataclass
from typing import Any, TypedDict

from iris.cluster.types import TERMINAL_JOB_STATES
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from marina.discovery import resolve_internal_ip
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rigging.telemetry.serialization import json_bytes
from rigging.timing import Timestamp
from sqlalchemy import text
from sqlalchemy.engine import Engine

INBOX = "s3://marin-us-east-02a/marin/rl-data/quick-loop-dashboard/v1"
RUN_ROOT = re.compile(r"s3://marin-us-east-02a/marin/users/[\w.-]+/checkpoints/snowball-inventory/[\w.-]+")
EVIDENCE_LIMIT = 2 * 1024 * 1024
COORDINATOR_WALL_TIME = "coordinator_submission_to_terminal"
logger = logging.getLogger(__name__)


def _digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_object(path: StoragePath) -> dict | None:
    """Read one bounded Quick RL evidence object, or None if it is absent."""
    try:
        with path.open("rb") as source:
            raw = source.read(EVIDENCE_LIMIT + 1)
    except FileNotFoundError:
        return None
    if len(raw) > EVIDENCE_LIMIT:
        raise ValueError(f"Oversize Quick RL evidence: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a Quick RL evidence object: {path}")
    return value


def run_root(terminal_uri: str) -> StoragePath:
    """Restrict registrations to the Quick RL artifact directory."""
    terminal = StoragePath(terminal_uri)
    root = terminal.parent
    if terminal.name != "terminal.json" or not RUN_ROOT.fullmatch(str(root)) or root.name in {".", ".."}:
        raise ValueError("Expected a CoreWeave Quick RL terminal manifest")
    return root


class EvaluationScore(TypedDict):
    reward: float
    attempted: int
    scored: int


@dataclass(frozen=True)
class EvaluationMeasurement:
    score: EvaluationScore | None = None
    identity: dict | None = None
    duration: float | None = None


def _evaluation_measurement(report: dict, step: int) -> EvaluationMeasurement:
    key = str(step)
    metrics = report.get("evaluation_metadata", {}).get(key, {})
    reward = metrics.get("reward")
    attempted = metrics.get("attempted")
    scored = metrics.get("scored")
    score = None
    if (
        isinstance(reward, int | float)
        and isinstance(attempted, int | float)
        and isinstance(scored, int | float)
        and all(math.isfinite(value) for value in (reward, attempted, scored))
        and 0 <= scored <= attempted
        and int(scored) == scored
        and int(attempted) == attempted
    ):
        score = EvaluationScore(reward=reward, attempted=int(attempted), scored=int(scored))
    duration = report.get("evaluation_duration_seconds", {}).get(key)
    if not isinstance(duration, int | float) or not math.isfinite(duration) or duration < 0:
        duration = None
    return EvaluationMeasurement(score, metrics.get("identity"), duration)


@dataclass
class SyncCounts:
    imported: int = 0
    pending: int = 0
    errors: int = 0


def _registration_filename(run_id: str, attempt_id: str) -> str:
    digest = _digest([run_id, attempt_id])
    return f"{digest}.json"


def terminal_record(entry: dict, job: job_pb2.JobStatus) -> dict:
    """Build one display record against the coordinator's actual terminal timestamps."""
    root = run_root(entry["terminal_manifest_uri"])
    attempt_id = entry["attempt_id"]
    if not re.fullmatch(r"a[0-9a-f]{11}", attempt_id):
        raise ValueError("Expected a native Quick RL attempt ID")
    terminal = read_object(root / "terminal.json") or {}
    native = terminal.get("result") or {}
    has_current_manifest = native.get("run_id") == entry["run_id"] and native.get("attempt_id") == attempt_id
    succeeded = job.state == job_pb2.JOB_STATE_SUCCEEDED and has_current_manifest and native["state"] == "succeeded"
    report_dir = root / "exports" / "grader_reports" / attempt_id
    grader_uri = report_dir / "grader_report.json"
    grader_report = read_object(grader_uri)
    baseline = _evaluation_measurement(grader_report or {}, 0)
    final = _evaluation_measurement(grader_report or {}, int(entry["expected_step"]))
    submitted = job.submitted_at.epoch_ms
    finished = job.finished_at.epoch_ms
    if not (0 < submitted <= finished):
        raise ValueError("Iris omitted the coordinator's terminal timestamps")
    wall = (finished - submitted) / 1000
    validation = (
        baseline.duration + final.duration if baseline.duration is not None and final.duration is not None else None
    )
    adjusted = wall - validation if validation is not None and wall > validation else None
    trainer_timing = read_object(report_dir / "trainer_timing.json") or {}
    trainer_elapsed = trainer_timing.get("elapsed_excluding_eval_seconds")
    if (
        isinstance(trainer_elapsed, int | float)
        and not isinstance(trainer_elapsed, bool)
        and math.isfinite(trainer_elapsed)
        and trainer_elapsed > 0
    ):
        elapsed = trainer_elapsed
        timing_scope = "trainer_elapsed_excluding_eval"
    elif adjusted is not None:
        elapsed = adjusted
        timing_scope = "coordinator_wall_minus_validation"
    else:
        elapsed = wall
        timing_scope = COORDINATOR_WALL_TIME
    common = (
        baseline.identity
        if timing_scope != COORDINATOR_WALL_TIME
        and baseline.identity == final.identity
        and entry["source_revision"] != "unknown"
        else None
    )
    return {
        "run_id": entry["run_id"],
        "attempt_id": attempt_id,
        "label": entry["run_id"].rsplit("/", 1)[-1],
        "coordinator_job_id": entry["coordinator_job_id"],
        "iris_job_id": native.get("iris_job_id") if has_current_manifest else None,
        "run_date": Timestamp(submitted).as_formatted_date(),
        "date_basis": "submission_acknowledged",
        "started_at": Timestamp(job.started_at.epoch_ms or submitted).as_formatted_date(),
        "completed_at": Timestamp(finished).as_formatted_date(),
        "elapsed_seconds": elapsed,
        "timing_scope": timing_scope,
        "coordinator_wall_seconds": wall,
        "validation_seconds": {"baseline": baseline.duration, "final": final.duration},
        "status": "succeeded" if succeeded else "failed",
        "optimizer_steps": entry["expected_step"] if succeeded and final.score is not None else None,
        "source_revision": entry["source_revision"],
        "data_identity": entry["data_identity"],
        "initializer_identity": entry["initializer_identity"],
        "preset": entry.get("preset") or "quick-rl",
        "preset_revision": entry.get("preset_revision"),
        "base_model": entry.get("base_model"),
        "hardware": entry.get("hardware"),
        "evaluation": {
            "name": "quick-rl-validation",
            "version": entry["validation_identity"],
            "task_count": baseline.score["attempted"] if baseline.score is not None else None,
            "task_set_sha256": _digest([entry["validation_identity"], common["membership_sha256"]]) if common else None,
            "protocol_sha256": (
                _digest(
                    [
                        entry["source_revision"],
                        entry["initializer_identity"],
                        {key: value for key, value in common.items() if key != "membership_sha256"},
                    ]
                )
                if common
                else None
            ),
        },
        "baseline": baseline.score,
        "final": final.score if baseline.score is not None else None,
        "artifact_uri": entry["terminal_manifest_uri"],
        "grader_report_uri": str(grader_uri) if grader_report is not None else None,
        "grader_report": grader_report,
        "evidence": [
            *(["The coordinator did not produce a successful terminal attempt."] if not succeeded else []),
            *(
                ["Trainer and validation timings are incomplete. Elapsed time includes validation."]
                if timing_scope == COORDINATOR_WALL_TIME
                else []
            ),
        ],
    }


def _pending_registrations(engine: Engine, counts: SyncCounts) -> list[dict]:
    filesystem, prefix = filesystem_for(INBOX)
    filesystem.invalidate_cache(prefix)
    paths = (StoragePath(INBOX) / "*.json").glob()
    with engine.connect() as connection:
        imported = {
            _registration_filename(row.run_id, row.attempt_id)
            for row in connection.execute(text("SELECT run_id, attempt_id FROM rl_runs"))
        }
    pending = []
    for path in paths:
        if path.name in imported:
            continue
        try:
            entry = read_object(path)
            if entry is None:
                continue
            if path.name != _registration_filename(entry["run_id"], entry["attempt_id"]):
                raise ValueError("Registration filename disagrees with its attempt identity")
            pending.append(entry)
        except (KeyError, OSError, ValueError):
            logger.exception("Could not read Quick RL registration %s", path)
            counts.errors += 1
    return pending


def _insert_record(engine: Engine, record: dict) -> int:
    with engine.begin() as connection:
        saved = connection.execute(
            text(
                """INSERT INTO rl_runs (run_id, attempt_id, record)
                VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
                ON CONFLICT (run_id, attempt_id) DO NOTHING"""
            ),
            {
                "run_id": record["run_id"],
                "attempt_id": record["attempt_id"],
                "record": json_bytes(record).decode(),
            },
        )
        return saved.rowcount


def sync_attempts(engine: Engine) -> SyncCounts:
    """Reconcile registered attempts once Iris records a terminal state."""
    result = SyncCounts()
    pending = _pending_registrations(engine, result)
    if not pending:
        return result
    address = resolve_internal_ip(
        "hai-gcp-models", "us-central1-a", "labels.iris-marin-controller=true AND status=RUNNING", timeout=5
    )
    client = ControllerServiceClientSync(address=f"http://{address}:10000", timeout_ms=5000)
    with closing(client):
        for entry in pending:
            try:
                request = controller_pb2.Controller.GetJobStatusRequest(job_id=entry["coordinator_job_id"])
                job = client.get_job_status(request).job
                if job.state not in TERMINAL_JOB_STATES:
                    result.pending += 1
                    continue
                result.imported += _insert_record(engine, terminal_record(entry, job))
            except Exception:
                logger.exception("Could not reconcile Quick RL attempt %s", entry["attempt_id"])
                result.errors += 1
    return result
