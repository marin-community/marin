# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import terminal Quick RL attempts from CoreWeave into the dashboard."""

import hashlib
import json
import logging
import math
import re
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

from iris.cluster.types import TERMINAL_JOB_STATES
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from marina.discovery import resolve_internal_ip
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from sqlalchemy import text
from sqlalchemy.engine import Engine

INBOX = "s3://marin-us-east-02a/marin/rl-data/quick-loop-dashboard/v1"
RUN_ROOT = re.compile(r"s3://marin-us-east-02a/marin/users/[\w.-]+/checkpoints/snowball-inventory/[\w.-]+")
EVIDENCE_LIMIT = 2 * 1024 * 1024
logger = logging.getLogger(__name__)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_object(path: StoragePath) -> dict | None:
    """Read a small JSON object, including a trainer's one-line aggregate."""
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
    root, marker, suffix = terminal_uri.rpartition("/terminal.json")
    if not marker or suffix or not RUN_ROOT.fullmatch(root) or root.rsplit("/", 1)[-1] in {".", ".."}:
        raise ValueError("Expected a CoreWeave Quick RL terminal manifest")
    return StoragePath(root)


def _eval(root: StoragePath, step: int) -> tuple[dict | None, dict | None, float | None]:
    directory = root / "exports" / "dumped_evals" / f"global_step_{step}_evals"
    metrics = read_object(directory / "aggregated_results.jsonl") or {}
    identity = read_object(directory / "dashboard_identity.json")
    timing = read_object(directory / "dashboard_duration.json") or {}
    reward = metrics.get("eval/all/avg_verifier_score")
    attempted = metrics.get("eval/all/num_attempted")
    scored = metrics.get("eval/all/num_scored")
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
        score = {"reward": reward, "attempted": int(attempted), "scored": int(scored)}
    duration = timing.get("duration_seconds")
    if not isinstance(duration, int | float) or not math.isfinite(duration) or duration < 0:
        duration = None
    return score, identity, duration


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


def terminal_record(entry: dict, job: job_pb2.JobStatus) -> dict:
    """Build one display record against the coordinator's actual terminal timestamps."""
    root = run_root(entry["terminal_manifest_uri"])
    terminal = read_object(root / "terminal.json") or {}
    native = terminal.get("result") or {}
    has_current_manifest = native.get("run_id") == entry["run_id"] and native.get("attempt_id") == entry["attempt_id"]
    succeeded = job.state == job_pb2.JOB_STATE_SUCCEEDED and has_current_manifest and native["state"] == "succeeded"
    baseline, baseline_identity, baseline_time = _eval(root, 0) if has_current_manifest else (None, None, None)
    final, final_identity, final_time = (
        _eval(root, int(entry["expected_step"])) if has_current_manifest else (None, None, None)
    )
    submitted = job.submitted_at.epoch_ms
    finished = job.finished_at.epoch_ms
    if not (0 < submitted <= finished):
        raise ValueError("Iris omitted the coordinator's terminal timestamps")
    wall = (finished - submitted) / 1000
    timed = baseline_time is not None and final_time is not None and wall > baseline_time + final_time
    comparable = (
        timed
        and baseline_identity is not None
        and baseline_identity == final_identity
        and entry["source_revision"] != "unknown"
    )
    common = baseline_identity if comparable else None
    grader_uri = root / "exports" / "grader_report.json"
    grader_report = read_object(grader_uri) if has_current_manifest else None
    return {
        "run_id": entry["run_id"],
        "attempt_id": entry["attempt_id"],
        "label": entry["run_id"].rsplit("/", 1)[-1],
        "coordinator_job_id": entry["coordinator_job_id"],
        "iris_job_id": native.get("iris_job_id") if has_current_manifest else None,
        "run_date": _iso(submitted),
        "date_basis": "submission_acknowledged",
        "started_at": _iso(job.started_at.epoch_ms or submitted),
        "completed_at": _iso(finished),
        "elapsed_seconds": wall - baseline_time - final_time if timed else wall,
        "timing_scope": "coordinator_wall_minus_validation" if timed else "coordinator_submission_to_terminal",
        "coordinator_wall_seconds": wall,
        "validation_seconds": {"baseline": baseline_time, "final": final_time},
        "status": "succeeded" if succeeded else "failed",
        "optimizer_steps": entry["expected_step"] if succeeded else None,
        "source_revision": entry["source_revision"],
        "data_identity": entry["data_identity"],
        "initializer_identity": entry["initializer_identity"],
        "evaluation": {
            "name": "quick-rl-validation",
            "version": entry["validation_identity"],
            "task_count": baseline["attempted"] if baseline is not None else None,
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
        "baseline": baseline,
        "final": final if baseline is not None else None,
        "artifact_uri": entry["terminal_manifest_uri"],
        "grader_report_uri": str(grader_uri) if grader_report is not None else None,
        "grader_report": grader_report,
        "evidence": [
            *(["The coordinator did not produce a successful terminal attempt."] if not succeeded else []),
            *(["Validation timings are incomplete. Elapsed time includes validation."] if not timed else []),
        ],
    }


def sync_attempts(engine: Engine) -> dict[str, int]:
    """Reconcile registered attempts once Iris records a terminal state."""
    filesystem, prefix = filesystem_for(INBOX)
    filesystem.invalidate_cache(prefix)
    paths = (StoragePath(INBOX) / "*.json").glob()
    result = {"imported": 0, "pending": 0, "errors": 0}
    with engine.connect() as connection:
        imported = {
            hashlib.sha256(json.dumps([row.run_id, row.attempt_id]).encode()).hexdigest() + ".json"
            for row in connection.execute(text("SELECT run_id, attempt_id FROM rl_runs"))
        }
    pending = []
    for path in paths:
        if path.name in imported:
            continue
        try:
            entry = read_object(path)
            if entry is not None:
                expected = hashlib.sha256(json.dumps([entry["run_id"], entry["attempt_id"]]).encode()).hexdigest()
                if path.name != f"{expected}.json":
                    raise ValueError("Registration filename disagrees with its attempt identity")
                pending.append(entry)
        except (KeyError, OSError, ValueError):
            logger.exception("Could not read Quick RL registration %s", path)
            result["errors"] += 1
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
                    result["pending"] += 1
                    continue
                record = terminal_record(entry, job)
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
                            "record": json.dumps(record, allow_nan=False),
                        },
                    )
                    result["imported"] += saved.rowcount
            except Exception:
                logger.exception("Could not reconcile Quick RL attempt %s", entry["attempt_id"])
                result["errors"] += 1
    return result
