# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect zephyr perf metrics from a finished ferry run.

Inputs:
  --status   gs:// path written by the ferry's ``_write_status`` helper.
             Contract: ``{"status": "...", "marin_prefix": "gs://..."}``.
  --job-id   Iris job id of the ferry leg. Used to pull task summary
             (per-task wall time, peak memory, exit code) and the tail of
             task-0 logs (per-step durations from ``marin.execution.step_runner``).

Output (JSON, written to ``--out`` and stdout):

    {
      "status": "succeeded" | "failed" | ...,
      "wall_seconds_total": 3054.6,
      "stage_wall_seconds": {"download": 116.1, "normalize": 1012.6, ...},
      "ooms": 0,
      "failed_shards": 0,
      "peak_worker_memory_mb": 476,
      "wandb_url": "https://wandb.ai/...",
      "iris_job_id": "...",
      "ferry_module": "experiments.ferries.datakit_ferry"
    }

The `stage_wall_seconds` map is keyed by the marin step name (`download`,
`normalize`, `minhash`, `fuzzy_dups`, `consolidate`, `tokenize` for the
datakit ferry). Names come straight from the step runner's
``Step <prefix>/<step>_<hash> succeeded in <H:MM:SS.ffff>`` log lines, so the
keys track whatever pipeline the ferry runs without further plumbing.

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


# `Step <prefix>/<step>_<hash> succeeded in <H:MM:SS.ffff>` from
# ``marin.execution.step_runner``. The hash suffix isolates the friendly step
# name (e.g. ``minhash``).
STEP_DURATION_RE = re.compile(
    r"Step\s+\S+?/(?P<step>[a-z][a-z_]*?)_[0-9a-f]+\s+succeeded in\s+" r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)"
)
# `rigging.timing Datakit ferry total wall time took H:MM:SS.ffff`
WALL_TIME_RE = re.compile(r"Datakit ferry total wall time took\s+" r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)")
# Fallback log line cap for `iris job logs`. Gate 1 control was ~73k lines /
# 50min; Gate 2 (Nemotron, ~24h) will be larger. The seven markers we care
# about (six step completions + one wall-time) are emitted near the *end* of
# the run, so `--tail` keeps them in window for any plausible volume.
LOG_LINE_CAP = 200_000


def _read_status(status_url: str) -> dict[str, Any]:
    fs, _ = url_to_fs(status_url)
    if not fs.exists(status_url):
        return {"status": "unknown", "marin_prefix": None}
    with fs.open(status_url, "r") as f:
        return json.loads(f.read())


def _task_logs(iris_config: str, job_id: str) -> str:
    """Pull the tail of task-0 logs for ``job_id``.

    Best-effort: returns the empty string if the iris CLI fails (e.g. job
    expired). The job_id path is task-scoped (``<job>/0``); preempted /
    multi-attempt jobs would need explicit attempt selection, which the
    current ``iris job logs`` CLI doesn't expose. Gate ferries are
    non-preemptible so this isn't a problem in practice.
    """
    task_path = f"{job_id}/0"
    try:
        return subprocess.check_output(
            [
                ".venv/bin/iris",
                f"--config={iris_config}",
                "job",
                "logs",
                task_path,
                "--tail",
                "--max-lines",
                str(LOG_LINE_CAP),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to pull task logs for %s: %s", task_path, e)
        return ""


def _job_summary(iris_config: str, job_id: str) -> dict[str, Any]:
    """Fetch ``iris job summary --json`` for the job.

    Returns the parsed dict on success or an empty dict on failure. The
    returned shape is the controller's job-summary contract::

        {"job_id": ..., "state": "succeeded", "tasks": [
            {"task_id": ..., "state": ..., "exit_code": int,
             "duration_ms": int, "memory_peak_mb": int, "error": str, ...},
            ...]}
    """
    try:
        raw = subprocess.check_output(
            [
                ".venv/bin/iris",
                f"--config={iris_config}",
                "job",
                "summary",
                "--json",
                job_id,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to fetch job summary for %s: %s", job_id, e)
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("job summary for %s is not JSON: %s", job_id, e)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _hms_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def _step_wall_seconds(task_logs: str) -> dict[str, float]:
    """Per-step wall time from `Step ... succeeded in H:MM:SS` log lines."""
    out: dict[str, float] = {}
    for line in task_logs.splitlines():
        m = STEP_DURATION_RE.search(line)
        if not m:
            continue
        out[m.group("step")] = round(_hms_to_seconds(m.group("h"), m.group("m"), m.group("s")), 1)
    return out


def _ooms_and_failures(tasks: list[dict[str, Any]]) -> tuple[int, int]:
    """Count OOMs and non-OOM failures from per-task summary entries.

    The current ``iris job summary`` task shape is flat: ``state``,
    ``exit_code``, ``error``. There's no first-class OOM signal, so we use
    Linux's exit-code 137 (SIGKILL after OOM) plus an ``error``-string
    fallback as a heuristic.
    """
    ooms = 0
    fails = 0
    for t in tasks:
        state = (t.get("state") or "").lower()
        exit_code = t.get("exit_code")
        error = (t.get("error") or "").lower()
        is_oom = exit_code == 137 or "oom" in error or "out of memory" in error
        if is_oom:
            ooms += 1
        elif state in {"failed", "killed"}:
            fails += 1
    return ooms, fails


def _peak_memory_mb(tasks: list[dict[str, Any]]) -> int | None:
    # The controller may report 0 once live metrics have decayed past the
    # post-completion retention window — treat 0 as "no signal" since a
    # container that ran real work necessarily peaked above zero.
    peaks = [
        int(t["memory_peak_mb"])
        for t in tasks
        if isinstance(t.get("memory_peak_mb"), (int, float)) and t["memory_peak_mb"] > 0
    ]
    return max(peaks) if peaks else None


def _total_wall_seconds(tasks: list[dict[str, Any]], task_logs: str) -> float | None:
    """Total ferry wall time. Prefer the authoritative task duration; fall
    back to the rigging.timing log line if the summary didn't include tasks."""
    if tasks:
        durations = [t["duration_ms"] / 1000.0 for t in tasks if isinstance(t.get("duration_ms"), (int, float))]
        if durations:
            return round(max(durations), 1)
    m = WALL_TIME_RE.search(task_logs)
    if m:
        return round(_hms_to_seconds(m.group("h"), m.group("m"), m.group("s")), 1)
    return None


EXPECTED_STEPS = ("download", "normalize", "minhash", "fuzzy_dups", "consolidate", "tokenize")


def collect(
    *,
    status_url: str | None,
    job_id: str | None,
    iris_config: str,
    ferry_module: str | None,
    wandb_url: str | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if status_url:
        status_payload = _read_status(status_url)
        if status_payload.get("status") == "unknown":
            warnings.append(f"status file not found at {status_url}")
    else:
        # Scheduled-baseline runs don't have a long-lived status JSON (TTL=1d
        # on FERRY_STATUS_PATH). Fall back to the iris job state in that case.
        status_payload = {"status": None, "marin_prefix": None}

    summary = _job_summary(iris_config, job_id) if job_id else {}
    tasks = summary.get("tasks") or []
    if job_id and not tasks:
        warnings.append(f"no tasks returned by `iris job summary` for {job_id}")

    task_logs = _task_logs(iris_config, job_id) if job_id else ""
    step_wall = _step_wall_seconds(task_logs)
    missing = [s for s in EXPECTED_STEPS if s not in step_wall]
    if task_logs and missing:
        warnings.append(
            f"missing step durations for {missing}; consider raising LOG_LINE_CAP "
            f"(currently {LOG_LINE_CAP}) — `Step ... succeeded in` lines may have "
            "been pushed past the tail window."
        )

    ooms, failed_shards = _ooms_and_failures(tasks)
    return {
        "status": status_payload.get("status"),
        "marin_prefix": status_payload.get("marin_prefix"),
        "wall_seconds_total": _total_wall_seconds(tasks, task_logs),
        "stage_wall_seconds": step_wall,
        "ooms": ooms,
        "failed_shards": failed_shards,
        "peak_worker_memory_mb": _peak_memory_mb(tasks),
        "wandb_url": wandb_url,
        "iris_job_id": job_id,
        "ferry_module": ferry_module,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        default=None,
        help=(
            "gs:// path to ferry_run_status.json (written by the ferry's "
            "`_write_status` helper). Optional — scheduled-baseline runs have "
            "TTL=1d on this path and may not have it; in that case the "
            "collector falls back to the iris job state."
        ),
    )
    parser.add_argument("--job-id", help="Iris job id of the ferry leg.")
    parser.add_argument("--iris-config", default="lib/iris/examples/marin.yaml")
    parser.add_argument("--ferry-module", help="Ferry module name (passthrough into output).")
    parser.add_argument("--wandb-url")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = collect(
        status_url=args.status,
        job_id=args.job_id,
        iris_config=args.iris_config,
        ferry_module=args.ferry_module,
        wandb_url=args.wandb_url,
    )

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
