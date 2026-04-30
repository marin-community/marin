# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect zephyr perf metrics from a finished ferry run.

Inputs:
  --status   gs:// path written by the ferry's ``_write_status`` helper.
             Contract: ``{"status": "...", "marin_prefix": "gs://..."}``.
  --job-id   Iris job id of the ferry leg. Used to pull coordinator logs and
             worker-pool exit signals.

Output (JSON, written to ``--out`` and stdout):

    {
      "status": "succeeded" | "failed" | ...,
      "wall_seconds_total": 1872.4,
      "stage_wall_seconds": {"download": 12.0, "normalize": 845.0, ...},
      "ooms": 0,
      "failed_shards": 0,
      "peak_worker_memory_mb": 14202,
      "counters": {"documents_processed": 9268156, ...},
      "wandb_url": "https://wandb.ai/...",
      "iris_job_id": "...",
      "ferry_module": "experiments.ferries.datakit_ferry"
    }

Many fields are best-effort — collectors that can't pull a value emit ``null``
and add a note to the ``warnings`` list rather than crashing. The compare step
treats null fields as "no signal" instead of inventing a regression.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from typing import Any

from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)


STAGE_PROGRESS_RE = re.compile(r"\[(?P<stage>stage\d+-[A-Za-z0-9_→ ]+)\]\s+" r"(?P<done>\d+)/(?P<total>\d+)\s+complete")
WALL_TIME_RE = re.compile(
    r"Datakit \w+ ferry total wall time.*?(?P<seconds>\d+\.\d+)s",
    re.IGNORECASE,
)


def _read_status(status_url: str) -> dict[str, Any]:
    fs, _ = url_to_fs(status_url)
    if not fs.exists(status_url):
        return {"status": "unknown", "marin_prefix": None}
    with fs.open(status_url, "r") as f:
        return json.loads(f.read())


def _coord_logs(iris_config: str, job_id: str) -> str:
    """Pull the latest-attempt coordinator logs for ``job_id``.

    Best-effort: returns the empty string if the iris CLI fails (e.g. job
    expired). Caller should treat absence as "no signal".
    """
    try:
        return subprocess.check_output(
            [
                ".venv/bin/iris",
                f"--config={iris_config}",
                "rpc",
                "controller",
                "get-task-logs",
                "--id",
                job_id,
                "--max-total-lines",
                "20000",
                "--attempt-id",
                "-1",
                "--tail",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to pull coord logs for %s: %s", job_id, e)
        return ""


def _list_tasks(iris_config: str, job_id: str) -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                ".venv/bin/iris",
                f"--config={iris_config}",
                "rpc",
                "controller",
                "list-tasks",
                "--job-id",
                job_id,
                "--json",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else parsed.get("tasks", [])
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.warning("Failed to list tasks for %s: %s", job_id, e)
        return []


def _stage_wall_seconds(coord_logs: str) -> dict[str, float]:
    """Extract per-stage durations from coordinator progress lines.

    The coordinator prints a progress line every ~5s with the current stage and
    an in-progress count. We use the timestamps of the first and last line per
    stage as the stage's start/end.
    """
    timestamps: dict[str, list[float]] = {}
    for line in coord_logs.splitlines():
        m = STAGE_PROGRESS_RE.search(line)
        if not m:
            continue
        # Iris log lines start with an ISO-ish timestamp; pull the leading
        # timestamp if present, otherwise skip — wall-time is best-effort.
        ts_match = re.match(r"^([0-9TZ:.\-+]+)\s", line)
        if not ts_match:
            continue
        try:
            ts = _parse_iso_seconds(ts_match.group(1))
        except ValueError:
            continue
        timestamps.setdefault(m.group("stage"), []).append(ts)
    return {stage: round(max(values) - min(values), 1) for stage, values in timestamps.items() if len(values) >= 2}


def _parse_iso_seconds(ts: str) -> float:
    import datetime as _dt

    ts = ts.rstrip("Z")
    return _dt.datetime.fromisoformat(ts).replace(tzinfo=_dt.timezone.utc).timestamp()


def _ooms_and_failures(tasks: list[dict[str, Any]]) -> tuple[int, int]:
    ooms = 0
    fails = 0
    for t in tasks:
        state = (t.get("state") or "").upper()
        exit_reason = (t.get("exitReason") or t.get("exit_reason") or "").lower()
        if "oom" in exit_reason or t.get("oomKilled"):
            ooms += 1
        if state in {"FAILED", "KILLED"} and "oom" not in exit_reason:
            fails += 1
    return ooms, fails


def _peak_memory_mb(tasks: list[dict[str, Any]]) -> int | None:
    peaks = []
    for t in tasks:
        usage = t.get("resourceUsage") or t.get("resource_usage") or {}
        peak = usage.get("memoryPeakMb")
        if isinstance(peak, (int, float)):
            peaks.append(int(peak))
    return max(peaks) if peaks else None


def _total_wall(coord_logs: str) -> float | None:
    m = WALL_TIME_RE.search(coord_logs)
    return float(m.group("seconds")) if m else None


def _counters(iris_config: str, coord_actor: str | None) -> dict[str, int]:
    """Pull final counter snapshot via `iris actor call get_counters`.

    Skipped when ``coord_actor`` is None — the caller hasn't found the
    coordinator endpoint. Returns an empty dict on any failure.
    """
    if not coord_actor:
        return {}
    try:
        raw = subprocess.check_output(
            [
                ".venv/bin/iris",
                f"--config={iris_config}",
                "actor",
                "call",
                coord_actor,
                "get_counters",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parsed = json.loads(raw)
        return parsed.get("counters", parsed) if isinstance(parsed, dict) else {}
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def collect(
    *,
    status_url: str,
    job_id: str | None,
    iris_config: str,
    coord_actor: str | None,
    ferry_module: str | None,
    wandb_url: str | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    status_payload = _read_status(status_url)
    if status_payload.get("status") == "unknown":
        warnings.append(f"status file not found at {status_url}")

    coord_logs = _coord_logs(iris_config, job_id) if job_id else ""
    tasks = _list_tasks(iris_config, job_id) if job_id else []
    if job_id and not tasks:
        warnings.append(f"no tasks returned for {job_id}")

    stage_wall = _stage_wall_seconds(coord_logs)
    if not stage_wall and coord_logs:
        warnings.append("no stage transitions parsed from coord logs")

    ooms, failed_shards = _ooms_and_failures(tasks)
    return {
        "status": status_payload.get("status"),
        "marin_prefix": status_payload.get("marin_prefix"),
        "wall_seconds_total": _total_wall(coord_logs),
        "stage_wall_seconds": stage_wall,
        "ooms": ooms,
        "failed_shards": failed_shards,
        "peak_worker_memory_mb": _peak_memory_mb(tasks),
        "counters": _counters(iris_config, coord_actor),
        "wandb_url": wandb_url,
        "iris_job_id": job_id,
        "ferry_module": ferry_module,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, help="gs:// path to ferry_run_status.json")
    parser.add_argument("--job-id", help="Iris job id of the ferry leg.")
    parser.add_argument("--iris-config", default="lib/iris/examples/marin.yaml")
    parser.add_argument(
        "--coord-actor",
        help="Coordinator actor endpoint (from coord task logs); enables counter snapshot.",
    )
    parser.add_argument("--ferry-module", help="Ferry module name (passthrough into output).")
    parser.add_argument("--wandb-url")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = collect(
        status_url=args.status,
        job_id=args.job_id,
        iris_config=args.iris_config,
        coord_actor=args.coord_actor,
        ferry_module=args.ferry_module,
        wandb_url=args.wandb_url,
    )

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
