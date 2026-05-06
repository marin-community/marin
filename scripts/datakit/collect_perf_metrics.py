#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect a structured perf report for a finished datakit ferry run.

Given an iris job id, this shells out to the iris CLI to extract:
- per-step wall times (regex over task-0 logs emitted by ``marin.execution.step_runner``)
- per-task peak memory and exit codes (``iris job summary --json``)
- job-level preemption / failure counts and per-state task counts
- a heuristic bucket classification of non-succeeded tasks

The report is written as JSON locally and (optionally) mirrored to a GCS prefix
under a ``report_<utc-ts>_<short-name>/`` directory so that runs can be compared
across time and architecture changes.

Used by the scheduled ``marin-canary-datakit-tier{1,2,3}`` workflows.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Step labels emitted by marin.execution.step_runner are
# ``<pipeline-prefix>/<step-name>_<content-hash>``. The hash is hex (typically
# 8+ chars). Step names can contain underscores (e.g. ``fuzzy_dups``), so we
# anchor on the trailing hex hash to peel it off.
STEP_DURATION_RE = re.compile(r"Step (?P<label>\S+?)_(?P<hash>[0-9a-f]{6,})\s+succeeded in (?P<elapsed>\S+)")

# Step runner emits this line when a step is cache-hit (StepAlreadyDone, the
# step lock indicates the output already exists). No duration is emitted, so we
# record cache-hit steps with seconds=0.0 and surface them via cached_steps.
STEP_CACHE_HIT_RE = re.compile(r"Step (?P<label>\S+?)_(?P<hash>[0-9a-f]{6,})\s+completed by another worker")

# Total-wall-time line from ``rigging.timing.log_time``. The ferries log a
# label ending in "ferry total wall time", e.g. ``"Datakit ferry total wall
# time"``. Used as a fallback when no per-task duration is exposed by iris.
WALL_TIME_RE = re.compile(r"ferry total wall time took (?P<elapsed>[^\n]+)")

# Tail size for ``iris job logs``. Step-completion lines are emitted near the
# end of the run; 200k lines comfortably covers Tier 3 (~6h) wall time.
LOG_LINE_CAP = 200_000

# Non-fatal warning if any of these step names is missing from the parsed
# durations. ``download`` is intentionally absent — not all ferries download
# (the nemotron ferry verifies a pre-staged dump), and even when present it
# may cache-hit and surface only via cached_steps.
EXPECTED_STEPS: tuple[str, ...] = (
    "normalize",
    "minhash",
    "fuzzy_dups",
    "consolidate",
    "tokenize",
)

# Buckets surfaced in ``infra_failures``. Order preserved so JSON output is
# stable across runs.
FAILURE_BUCKETS: tuple[str, ...] = (
    "preempted",
    "oom",
    "hardware_fault",
    "scheduling_timeout",
    "application_failure",
    "other",
)


@dataclass
class PerfReport:
    """In-memory model of the report. Serialised verbatim to JSON."""

    iris_job_id: str
    status: str | None = None
    marin_prefix: str | None = None
    wall_seconds_total: float | None = None
    stage_wall_seconds: dict[str, float] = field(default_factory=dict)
    cached_steps: list[str] = field(default_factory=list)
    ooms: int = 0
    failed_shards: int = 0
    peak_worker_memory_mb: int = 0
    preemption_count: int = 0
    failure_count: int = 0
    task_state_counts: dict[str, int] = field(default_factory=dict)
    # Number of jobs in the iris tree under this run (launcher + child jobs).
    # The three counts above are summed across all of them.
    tree_job_count: int = 0
    infra_failures: dict[str, int] = field(default_factory=lambda: {b: 0 for b in FAILURE_BUCKETS})
    workflow_run_id: str | None = None
    workflow_run_attempt: str | None = None
    workflow_name: str | None = None
    commit_sha: str | None = None
    collected_at_utc: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=False)


# --------------------------------------------------------------------------- #
# iris CLI helpers
# --------------------------------------------------------------------------- #


def _iris_command() -> list[str]:
    venv_iris = _REPO_ROOT / ".venv" / "bin" / "iris"
    if venv_iris.exists():
        return [str(venv_iris)]
    return ["uv", "run", "--package", "iris", "iris"]


def _run_iris(args: list[str], iris_config: Path) -> subprocess.CompletedProcess:
    cmd = [*_iris_command(), f"--config={iris_config}", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 and "No such file or directory: 'gcloud'" in result.stderr:
        raise click.ClickException(
            "iris CLI requires `gcloud` on PATH (controller tunnel uses gcloud SSH). "
            "Install Google Cloud SDK and retry."
        )
    return result


def fetch_job_summary(job_id: str, iris_config: Path) -> dict | None:
    """Return the parsed ``iris job summary --json <job>`` payload, or None."""
    result = _run_iris(["job", "summary", "--json", job_id], iris_config)
    if result.returncode != 0:
        logger.warning("iris job summary failed (exit %s): %s", result.returncode, result.stderr.strip())
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("iris job summary returned non-JSON: %s", exc)
        return None


def fetch_job_tree(job_id: str, iris_config: Path) -> list[dict] | None:
    """Return ``iris job list --json --prefix <job>`` — the parent + all descendants.

    Each entry includes job-level ``preemption_count`` / ``failure_count`` /
    ``task_state_counts``. We need the tree (not just the parent's summary)
    because the launcher task is the only thing under the parent itself; the
    actual fan-out workers live in child iris jobs (zephyr pipeline subjobs).
    """
    result = _run_iris(["job", "list", "--json", "--prefix", job_id], iris_config)
    if result.returncode != 0:
        logger.warning("iris job list --prefix failed (exit %s): %s", result.returncode, result.stderr.strip())
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("iris job list returned non-JSON: %s", exc)
        return None


def aggregate_job_tree(jobs: list[dict]) -> dict:
    """Sum preemption / failure / task-state counts across every job in the tree.

    Returns a dict with the same field names as ``iris job summary``:
    ``preemption_count``, ``failure_count``, ``task_state_counts``, plus
    ``job_count`` for sanity-checking. Used to override the parent-only
    counts that ``iris job summary <parent>`` returns, since those only
    describe the launcher task and miss the fan-out workers.
    """
    preemption_count = 0
    failure_count = 0
    task_state_counts: dict[str, int] = {}
    for j in jobs:
        preemption_count += int(j.get("preemption_count") or 0)
        failure_count += int(j.get("failure_count") or 0)
        for state, n in (j.get("task_state_counts") or {}).items():
            task_state_counts[state] = task_state_counts.get(state, 0) + int(n)
    return {
        "preemption_count": preemption_count,
        "failure_count": failure_count,
        "task_state_counts": task_state_counts,
        "job_count": len(jobs),
    }


# finelog can refuse very large tail requests with a connect/deadline timeout.
# Match those errors so we can retry with a smaller tail before giving up.
_LOG_TIMEOUT_PATTERNS = ("request timed out", "deadline exceeded", "statserror", "deadline_exceeded")


def _is_finelog_timeout(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(pat in s for pat in _LOG_TIMEOUT_PATTERNS)


def fetch_task0_logs(job_id: str, iris_config: Path, max_lines: int = LOG_LINE_CAP) -> str:
    """Return the tail of task-0 logs for the job, or empty string on persistent failure.

    finelog can time out on large tails. We retry with progressively smaller
    tails (200k → 50k → 10k) on timeout-class errors, since the step-completion
    lines we care about are emitted near the end of the run regardless of cap.
    """
    attempts: list[int] = []
    seen: set[int] = set()
    for n in (max_lines, max_lines // 4, max_lines // 20, 5000):
        if n > 0 and n not in seen:
            attempts.append(n)
            seen.add(n)

    last_stderr = ""
    backoff_seconds = 2.0
    for attempt_idx, lines in enumerate(attempts):
        result = _run_iris(
            ["job", "logs", f"{job_id}/0", "--tail", "--max-lines", str(lines)],
            iris_config,
        )
        if result.returncode == 0:
            if attempt_idx > 0:
                logger.info("iris job logs succeeded with --max-lines=%d after %d retries", lines, attempt_idx)
            return result.stdout

        last_stderr = result.stderr.strip()
        if not _is_finelog_timeout(last_stderr):
            # Not a transient timeout — no point retrying with a smaller tail.
            logger.warning("iris job logs failed (exit %s): %s", result.returncode, last_stderr)
            return ""

        if attempt_idx + 1 < len(attempts):
            next_lines = attempts[attempt_idx + 1]
            logger.warning(
                "iris job logs timed out at --max-lines=%d; retrying with %d after %.1fs",
                lines,
                next_lines,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    logger.warning("iris job logs failed after %d attempts; final error: %s", len(attempts), last_stderr)
    return ""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_timedelta_str(s: str) -> float:
    """Parse a ``str(timedelta)`` value into total seconds.

    Handles both forms produced by Python's timedelta repr:
        ``"0:13:25.412345"`` and ``"1 day, 0:13:25.412345"`` / ``"2 days, ..."``.
    """
    days = 0
    s = s.strip()
    if "day" in s:
        day_part, _, rest = s.partition(",")
        days = int(day_part.split()[0])
        s = rest.strip()
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unparseable timedelta string: {s!r}")
    hours, minutes, seconds = parts
    return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_stage_wall_seconds(logs: str) -> tuple[dict[str, float], list[str]]:
    """Extract ``({step_name: seconds}, [cached_step_names])`` from step_runner logs.

    Two log lines are consumed:
    - ``Step <label> succeeded in <elapsed>`` — step ran; record its duration.
    - ``Step <label> completed by another worker`` — cache hit (StepAlreadyDone);
      record duration as 0.0 and add to ``cached_steps``.

    If a step appears in both forms (rare; only happens across retries), the
    last occurrence wins — same logic step_runner itself uses.
    """
    durations: dict[str, float] = {}
    cached: dict[str, bool] = {}

    for match in STEP_DURATION_RE.finditer(logs):
        label = match.group("label")
        elapsed = match.group("elapsed")
        step_name = label.rsplit("/", 1)[-1]
        try:
            durations[step_name] = parse_timedelta_str(elapsed)
            cached[step_name] = False
        except ValueError:
            logger.warning("Could not parse step duration %r for %s", elapsed, step_name)

    for match in STEP_CACHE_HIT_RE.finditer(logs):
        label = match.group("label")
        step_name = label.rsplit("/", 1)[-1]
        # Only mark cached if no real run was seen for this step.
        if step_name not in durations:
            durations[step_name] = 0.0
            cached[step_name] = True

    cached_steps = sorted(name for name, was_cached in cached.items() if was_cached)
    return durations, cached_steps


def parse_total_wall_seconds_from_logs(logs: str) -> float | None:
    """Fallback: pull total wall time from the ferry's log_time line."""
    match = WALL_TIME_RE.search(logs)
    if not match:
        return None
    try:
        return parse_timedelta_str(match.group("elapsed"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


def classify_task_failure(state: str, exit_code: int | None, error: str | None) -> str | None:
    """Bucket a non-succeeded task into one of FAILURE_BUCKETS, or None.

    Heuristic — refined as we see real failure shapes from scheduled runs.
    Order matters: preempt and OOM win over the generic application_failure
    bucket so we don't lose specificity.
    """
    state_lc = (state or "").lower()
    if state_lc == "succeeded":
        return None
    error_lc = (error or "").lower()
    if state_lc == "preempted" or "preempt" in error_lc:
        return "preempted"
    if exit_code == 137 or "oom" in error_lc or "out of memory" in error_lc:
        return "oom"
    if "tpu" in error_lc or "hardware" in error_lc or "node_failure" in error_lc:
        return "hardware_fault"
    if "schedule" in error_lc or "timeout" in error_lc or state_lc == "unschedulable":
        return "scheduling_timeout"
    if state_lc in {"failed", "killed", "worker_failed"}:
        return "application_failure"
    return "other"


def classify_failures(tasks: list[dict]) -> tuple[dict[str, int], int, int]:
    """Classify tasks; return ``(infra_failures, ooms, failed_shards)``.

    ``failed_shards`` is the number of non-succeeded tasks whose bucket is
    ``application_failure``-like (i.e. neither OOM nor preempted) — kept for
    backward compatibility with the README's original two-field schema.
    """
    buckets: dict[str, int] = {b: 0 for b in FAILURE_BUCKETS}
    ooms = 0
    failed_shards = 0
    for t in tasks:
        bucket = classify_task_failure(
            state=t.get("state", ""),
            exit_code=t.get("exit_code"),
            error=t.get("error"),
        )
        if bucket is None:
            continue
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if bucket == "oom":
            ooms += 1
        elif bucket == "application_failure":
            failed_shards += 1
    return buckets, ooms, failed_shards


# --------------------------------------------------------------------------- #
# Status file
# --------------------------------------------------------------------------- #


def load_ferry_status(status_path: str | None) -> dict | None:
    """Best-effort read of the ferry's FERRY_STATUS_PATH JSON. Returns None on miss."""
    if not status_path:
        return None
    try:
        fs, path = url_to_fs(status_path)
        if not fs.exists(path):
            return None
        with fs.open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read ferry status %s: %s", status_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def build_report(
    *,
    job_id: str,
    summary: dict | None,
    job_tree: list[dict] | None,
    logs: str,
    status: dict | None,
    workflow_env: dict[str, str | None],
) -> PerfReport:
    """Assemble a PerfReport from iris summary + tree + task-0 logs + ferry status.

    The parent ``iris job summary`` describes only the launcher task; ``job_tree``
    is the result of ``iris job list --prefix <parent>`` which surfaces every
    descendant job. We use the tree (when available) for ``preemption_count``,
    ``failure_count``, and ``task_state_counts`` so they reflect the whole
    pipeline, not just the launcher.
    """
    report = PerfReport(
        iris_job_id=job_id,
        collected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        workflow_run_id=workflow_env.get("run_id"),
        workflow_run_attempt=workflow_env.get("run_attempt"),
        workflow_name=workflow_env.get("workflow"),
        commit_sha=workflow_env.get("commit_sha"),
    )

    if status:
        report.status = status.get("status")
        report.marin_prefix = status.get("marin_prefix")
    else:
        report.warnings.append("ferry_status_path: not readable; status/marin_prefix unset")

    if summary is None:
        report.warnings.append("iris job summary --json: failed; per-task fields unavailable")
    else:
        tasks = summary.get("tasks") or []
        if tasks:
            durations = [t.get("duration_ms") for t in tasks if t.get("duration_ms")]
            if durations:
                report.wall_seconds_total = max(durations) / 1000.0
            mems = [t.get("memory_peak_mb", 0) for t in tasks]
            report.peak_worker_memory_mb = max(mems) if mems else 0
            report.infra_failures, report.ooms, report.failed_shards = classify_failures(tasks)

    # Aggregate preemption / failure / task-state counts across the whole job
    # tree. Falls back to the parent-only summary fields when the tree is
    # unavailable, so a list-RPC failure doesn't zero these out.
    if job_tree is not None:
        agg = aggregate_job_tree(job_tree)
        report.preemption_count = agg["preemption_count"]
        report.failure_count = agg["failure_count"]
        report.task_state_counts = agg["task_state_counts"]
        report.tree_job_count = agg["job_count"]
    elif summary is not None:
        report.preemption_count = int(summary.get("preemption_count") or 0)
        report.failure_count = int(summary.get("failure_count") or 0)
        report.task_state_counts = dict(summary.get("task_state_counts") or {})
        report.warnings.append("iris job list --prefix: failed; counts reflect launcher task only")

    if report.task_state_counts.get("preempted"):
        report.warnings.append("task_state_counts.preempted > 0: stage durations may be split across attempts")

    report.stage_wall_seconds, report.cached_steps = parse_stage_wall_seconds(logs)
    if not report.stage_wall_seconds:
        report.warnings.append("no Step ... succeeded/cache-hit lines found in task-0 log tail")

    if report.wall_seconds_total is None:
        fallback = parse_total_wall_seconds_from_logs(logs)
        if fallback is not None:
            report.wall_seconds_total = fallback
        else:
            report.warnings.append("wall_seconds_total: could not derive from iris summary or logs")

    missing = [step for step in EXPECTED_STEPS if step not in report.stage_wall_seconds]
    if missing:
        report.warnings.append(f"missing expected steps in stage_wall_seconds: {missing}")

    return report


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _utc_timestamp_compact() -> str:
    """Return a filesystem-safe UTC timestamp like ``20260506T071523Z``."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_report_local(report: PerfReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json())


def upload_report_to_gcs(report: PerfReport, gcs_prefix: str, report_name: str, timestamp: str) -> str:
    """Write the JSON to ``<gcs_prefix>/report_<timestamp>_<report_name>/perf_report.json``.

    Returns the full destination URL.
    """
    if not gcs_prefix.startswith("gs://"):
        raise click.UsageError(f"--gcs-prefix must start with gs://, got {gcs_prefix!r}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", report_name)
    dest = f"{gcs_prefix.rstrip('/')}/report_{timestamp}_{safe_name}/perf_report.json"
    fs, path = url_to_fs(dest)
    with fs.open(path, "w") as fh:
        fh.write(report.to_json())
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--job-id", required=True, help="Iris job id of the ferry run.")
@click.option(
    "--iris-config",
    default="lib/iris/examples/marin.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to iris config file used for the iris CLI.",
)
@click.option(
    "--status",
    "status_path",
    default=None,
    help="Optional FERRY_STATUS_PATH gs:// URL written by the ferry's _write_status helper.",
)
@click.option(
    "--report-name",
    default=None,
    help="Short stable name embedded in the GCS path (required only when --gcs-prefix is set).",
)
@click.option(
    "--out",
    default=None,
    type=click.Path(path_type=Path),
    help="Local path to write the JSON report. When omitted, prints JSON to stdout.",
)
@click.option(
    "--gcs-prefix",
    default=None,
    help="Optional gs:// prefix; mirrors to <prefix>/report_<utc-ts>_<report-name>/perf_report.json.",
)
@click.option(
    "--gcs-output-env",
    default=None,
    help="If set, write the resulting GCS URL to this $GITHUB_OUTPUT key.",
)
def main(
    job_id: str,
    iris_config: Path,
    status_path: str | None,
    report_name: str | None,
    out: Path | None,
    gcs_prefix: str | None,
    gcs_output_env: str | None,
) -> None:
    """Collect a perf report for a finished datakit ferry run.

    With only --job-id, the report is printed as JSON to stdout. Pass --out
    to write a local file, and --gcs-prefix --report-name to mirror to GCS.
    """
    if gcs_prefix and not report_name:
        raise click.UsageError("--gcs-prefix requires --report-name")

    # All script logging goes to stderr; stdout stays clean for the JSON
    # output when --out is omitted.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    workflow_env = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "commit_sha": os.environ.get("GITHUB_SHA"),
    }

    summary = fetch_job_summary(job_id, iris_config)
    job_tree = fetch_job_tree(job_id, iris_config)
    logs = fetch_task0_logs(job_id, iris_config)
    status = load_ferry_status(status_path)

    report = build_report(
        job_id=job_id,
        summary=summary,
        job_tree=job_tree,
        logs=logs,
        status=status,
        workflow_env=workflow_env,
    )

    if out is not None:
        write_report_local(report, out)
        logger.info("Wrote perf report to %s", out)
    else:
        click.echo(report.to_json())

    if gcs_prefix:
        assert report_name is not None  # validated above
        ts = _utc_timestamp_compact()
        dest = upload_report_to_gcs(report, gcs_prefix, report_name, ts)
        logger.info("Mirrored perf report to %s", dest)
        gh_output = os.environ.get("GITHUB_OUTPUT")
        if gcs_output_env and gh_output:
            with open(gh_output, "a") as fh:
                fh.write(f"{gcs_output_env}={dest}\n")

    if report.warnings:
        for warn in report.warnings:
            logger.warning("warning: %s", warn)


if __name__ == "__main__":
    sys.exit(main())
