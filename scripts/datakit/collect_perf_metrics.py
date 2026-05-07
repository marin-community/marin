#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect a structured perf report for a finished datakit ferry run.

Given an iris job id, this shells out to the iris CLI to extract:
- per-step wall times derived from deterministic ``zephyr-<step>-*`` child-job
  names + ``started_at``/``finished_at`` on ``iris job list --prefix``
- aggregated preemption / failure / task-state counts across the whole tree
- per-task peak memory and a heuristic bucket classification of non-succeeded
  tasks, fetched via ``iris job summary --json`` for each leaf worker job

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
from dataclasses import dataclass, field
from pathlib import Path

import click
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Each ferry step that fans out work submits a child iris job with a
# deterministic name prefix. ``zephyr-fuzzy-dups-...-pN-aM`` etc. share a
# prefix, so multi-phase steps (CC iterations, levanter cache prep) sum
# naturally. ``zephyr-levanter-cache-{copy,probe}-*`` belong to the tokenize
# step. ``download`` is optional — the nemotron ferry verifies a pre-staged
# dump rather than downloading.
_STEP_PREFIXES: dict[str, str] = {
    "zephyr-download-hf-": "download",
    "zephyr-normalize-": "normalize",
    "zephyr-minhash-attrs-": "minhash",
    "zephyr-fuzzy-dups-": "fuzzy_dups",
    "zephyr-consolidate-filter-": "consolidate",
    "zephyr-tokenize-train-": "tokenize",
    "zephyr-levanter-cache-copy-": "tokenize",
    "zephyr-levanter-cache-probe-": "tokenize",
}

# Non-fatal warning if any of these step names is missing from the parsed
# durations. ``download`` is intentionally absent — see _STEP_PREFIXES above.
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


def fetch_leaf_summaries(job_tree: list[dict], iris_config: Path) -> list[dict]:
    """Fetch ``iris job summary --json`` for every leaf job in the tree.

    Per-task data (``memory_peak_mb``, ``error``, ``exit_code``) lives on each
    job's own task array, which ``iris job list --prefix`` does not return.
    Leaves are jobs with ``has_children == false`` — those are the worker
    pools where the actual fan-out work runs. Coordinator jobs are skipped:
    their tasks are dispatcher-only and don't carry useful memory or error
    signal.
    """
    summaries: list[dict] = []
    for job in job_tree:
        if job.get("has_children") is not False:
            continue
        job_id = job.get("job_id")
        if not job_id:
            continue
        s = fetch_job_summary(job_id, iris_config)
        if s is not None:
            summaries.append(s)
    return summaries


def aggregate_per_task_metrics(summaries: list[dict]) -> tuple[int, dict[str, int], int, int]:
    """Walk every task across all summaries and return cross-tree per-task metrics.

    Returns ``(peak_worker_memory_mb, infra_failures, ooms, failed_shards)``,
    aggregated across the launcher and every leaf worker job.
    """
    peak_memory = 0
    buckets: dict[str, int] = {b: 0 for b in FAILURE_BUCKETS}
    ooms = 0
    failed_shards = 0
    for summary in summaries:
        for task in summary.get("tasks") or []:
            mem = int(task.get("memory_peak_mb") or 0)
            if mem > peak_memory:
                peak_memory = mem
            bucket = classify_task_failure(
                state=task.get("state", ""),
                exit_code=task.get("exit_code"),
                error=task.get("error"),
            )
            if bucket is None:
                continue
            buckets[bucket] = buckets.get(bucket, 0) + 1
            if bucket == "oom":
                ooms += 1
            elif bucket == "application_failure":
                failed_shards += 1
    return peak_memory, buckets, ooms, failed_shards


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


# --------------------------------------------------------------------------- #
# Per-step wall times derived from the iris job tree
# --------------------------------------------------------------------------- #


def _job_depth(job_id: str) -> int:
    """Number of ``/`` separators — proxies for tree depth in the iris namespace."""
    return job_id.count("/")


def compute_stage_wall_seconds(
    jobs: list[dict],
    parent_id: str,
) -> tuple[dict[str, float], list[str]]:
    """Bucket direct-child iris jobs into ferry steps and sum their wall times.

    For each direct child of ``parent_id``, look up its name prefix in
    ``_STEP_PREFIXES`` and accumulate ``finished_at - started_at``. Multi-phase
    steps (``zephyr-fuzzy-dups-...-pN-aM``, ``zephyr-tokenize-train-pN-aM``)
    share a prefix, so their per-phase durations sum.

    We restrict to direct children because workers nested under coordinators
    would double-count their parent's wall time.

    Returns ``(stage_wall_seconds, cached_steps)``. Steps in ``EXPECTED_STEPS``
    that don't appear in the tree are reported with ``0.0`` and added to
    ``cached_steps`` — those steps always run unless the artifact already
    exists, so absence implies a cache hit.
    """
    parent_depth = _job_depth(parent_id)
    durations: dict[str, float] = {}

    for job in jobs:
        job_id = job.get("job_id") or ""
        if not job_id.startswith(parent_id):
            continue
        if _job_depth(job_id) != parent_depth + 1:
            continue
        name = job_id.rsplit("/", 1)[-1]
        for prefix, step in _STEP_PREFIXES.items():
            if not name.startswith(prefix):
                continue
            start_ms = int((job.get("started_at") or {}).get("epoch_ms") or 0)
            end_ms = int((job.get("finished_at") or {}).get("epoch_ms") or 0)
            if start_ms and end_ms and end_ms > start_ms:
                durations[step] = durations.get(step, 0.0) + (end_ms - start_ms) / 1000.0
            break

    cached_steps = sorted(s for s in EXPECTED_STEPS if s not in durations)
    for s in cached_steps:
        durations[s] = 0.0
    return durations, cached_steps


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


def classify_task_failure(state: str, exit_code: int | None, error: str | None) -> str | None:
    """Bucket a non-succeeded task into one of FAILURE_BUCKETS, or None.

    Heuristic — refined as we see real failure shapes from scheduled runs.
    Order matters: preempt and OOM win over the generic application_failure
    bucket so we don't lose specificity.

    ``state=killed`` returns None: across the marin pipelines, killed tasks
    are almost always cleanup kills after a coordinator finishes (the iris
    controller terminates remaining workers). Counting them as failures
    would inflate ``application_failure`` on every healthy run. The rare
    case (e.g. user-cancelled run) shows up via the parent's job state, not
    via per-task counts.
    """
    state_lc = (state or "").lower()
    if state_lc in {"succeeded", "killed"}:
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
    if state_lc in {"failed", "worker_failed"}:
        return "application_failure"
    return "other"


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
    leaf_summaries: list[dict],
    status: dict | None,
    workflow_env: dict[str, str | None],
) -> PerfReport:
    """Assemble a PerfReport from iris summary + tree + leaf summaries + status.

    Sources, in order of who-knows-what:
    - parent ``iris job summary``: launcher task duration → ``wall_seconds_total``.
    - ``iris job list --prefix``: per-step wall times (deterministic zephyr-*
      child names) and aggregated preemption / failure / task-state counts
      across the whole tree.
    - per-leaf ``iris job summary``: per-task ``memory_peak_mb`` and ``error``
      strings, which only live on the leaf workers, not on the parent.
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
        report.warnings.append("iris job summary --json: failed; wall_seconds_total unavailable")
    else:
        tasks = summary.get("tasks") or []
        durations = [t.get("duration_ms") for t in tasks if t.get("duration_ms")]
        if durations:
            report.wall_seconds_total = max(durations) / 1000.0

    # Per-task metrics across the launcher AND every leaf worker job.
    summaries_for_tasks = ([summary] if summary else []) + leaf_summaries
    if summaries_for_tasks:
        report.peak_worker_memory_mb, report.infra_failures, report.ooms, report.failed_shards = (
            aggregate_per_task_metrics(summaries_for_tasks)
        )
    if not leaf_summaries:
        report.warnings.append("no leaf summaries fetched; peak_worker_memory_mb/infra_failures reflect launcher only")

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

    if job_tree is not None:
        report.stage_wall_seconds, report.cached_steps = compute_stage_wall_seconds(job_tree, job_id)
        if all(report.stage_wall_seconds.get(s, 0.0) == 0.0 for s in EXPECTED_STEPS):
            report.warnings.append("all expected steps cache-hit; pipeline may not have done any work")
    else:
        report.warnings.append("iris job tree unavailable; stage_wall_seconds empty")

    if report.wall_seconds_total is None:
        report.warnings.append("wall_seconds_total: launcher duration_ms missing from iris summary")

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
    leaf_summaries = fetch_leaf_summaries(job_tree, iris_config) if job_tree else []
    status = load_ferry_status(status_path)

    report = build_report(
        job_id=job_id,
        summary=summary,
        job_tree=job_tree,
        leaf_summaries=leaf_summaries,
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
