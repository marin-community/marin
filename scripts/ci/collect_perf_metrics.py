#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect a structured perf report for a finished datakit ferry run.

Given an iris job id, this uses the Iris client to extract:
- per-step wall times derived from deterministic ``zephyr-<step>-*`` child-job
  names + ``started_at``/``finished_at`` on the job tree
- aggregated preemption / failure / task-state counts across the whole tree
- per-task peak memory and a heuristic bucket classification of non-succeeded
  tasks, fetched from each leaf worker job

The report is written as JSON locally and (optionally) mirrored to a GCS prefix
under a ``report_<utc-ts>_<short-name>/`` directory so that runs can be compared
across time and architecture changes.

Used by the scheduled ``marin-canary-datakit-tier{1,2,3}`` workflows.
"""

import datetime
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
from connectrpc.errors import ConnectError
from google.protobuf import json_format
from iris.cli.connect import connect_controller, rpc_client
from iris.cli.job import build_job_summary
from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2, query_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.filesystem import StoragePath

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

# SQL stores task states by numeric enum value.
_TASK_STATE_SUCCEEDED = job_pb2.TASK_STATE_SUCCEEDED

# Wrapped in an outer SELECT so the query starts with SELECT (required by ExecuteRawQuery).
_TASK_WALL_TIME_SQL = """\
SELECT task_wall_ms FROM (
    WITH RECURSIVE descendants(job_id) AS (
        SELECT job_id FROM jobs WHERE job_id = '{job_id}'
        UNION ALL
        SELECT j.job_id FROM jobs j
        JOIN descendants d ON j.parent_job_id = d.job_id
    ),
    leaves(job_id) AS (
        SELECT d.job_id FROM descendants d
        WHERE NOT EXISTS (SELECT 1 FROM jobs c WHERE c.parent_job_id = d.job_id)
    )
    SELECT SUM(ta.finished_at_ms - ta.started_at_ms) AS task_wall_ms
    FROM task_attempts ta
    JOIN tasks t ON t.task_id = ta.task_id
    JOIN leaves USING (job_id)
    WHERE ta.started_at_ms IS NOT NULL
      AND ta.finished_at_ms IS NOT NULL
      AND ta.finished_at_ms > ta.started_at_ms
      {state_filter}
)"""

# Per-direct-child breakdown: the recursive CTE carries child_job_id (the
# direct child of the root job each descendant belongs to) so we can group by it.
_TASK_WALL_TIME_BY_CHILD_SQL = """\
SELECT child_job_id, SUM(duration_ms) AS task_wall_ms FROM (
    WITH RECURSIVE descendants(job_id, child_job_id) AS (
        SELECT job_id, job_id AS child_job_id FROM jobs WHERE parent_job_id = '{job_id}'
        UNION ALL
        SELECT j.job_id, d.child_job_id FROM jobs j
        JOIN descendants d ON j.parent_job_id = d.job_id
    ),
    leaves(job_id, child_job_id) AS (
        SELECT d.job_id, d.child_job_id FROM descendants d
        WHERE NOT EXISTS (SELECT 1 FROM jobs c WHERE c.parent_job_id = d.job_id)
    )
    SELECT ta.finished_at_ms - ta.started_at_ms AS duration_ms, leaves.child_job_id
    FROM task_attempts ta
    JOIN tasks t ON t.task_id = ta.task_id
    JOIN leaves USING (job_id)
    WHERE ta.started_at_ms IS NOT NULL
      AND ta.finished_at_ms IS NOT NULL
      AND ta.finished_at_ms > ta.started_at_ms
      {state_filter}
)
GROUP BY child_job_id
ORDER BY child_job_id"""


@dataclass
class PerfReport:
    """In-memory model of the report. Serialised verbatim to JSON."""

    iris_job_id: str
    status: str | None = None
    marin_prefix: str | None = None
    wall_seconds_total: float | None = None
    stage_wall_seconds: dict[str, float] = field(default_factory=dict)
    sum_task_wall_seconds_total: float | None = None
    stage_sum_task_wall_seconds: dict[str, float | None] = field(default_factory=dict)
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
# Iris client / query helpers
# --------------------------------------------------------------------------- #


def _job_status_to_dict(job: job_pb2.JobStatus) -> dict:
    data = json_format.MessageToDict(job, preserving_proto_field_name=True)
    data["has_children"] = bool(job.has_children)
    return data


def fetch_job_summary(client: IrisClient, job_id: str) -> dict | None:
    """Return a job summary, or None when the Iris RPC fails."""
    try:
        job_name = JobName.from_wire(job_id)
        return build_job_summary(client.status(job_name), client.list_tasks(job_name))
    except ConnectError as exc:
        logger.warning("iris client job summary failed for %s: %s", job_id, exc)
        return None


def fetch_job_tree(client: IrisClient, job_id: str) -> list[dict] | None:
    """Return the parent + descendants, or None when the Iris RPC fails.

    Each entry includes job-level ``preemption_count`` / ``failure_count`` /
    ``task_state_counts``. We need the tree (not just the parent's summary)
    because the launcher task is the only thing under the parent itself; the
    actual fan-out workers live in child iris jobs (zephyr pipeline subjobs).
    """
    try:
        jobs = client.list_jobs(prefix=job_id)
        jobs.sort(key=lambda j: j.submitted_at.epoch_ms, reverse=True)
        return [_job_status_to_dict(job) for job in jobs]
    except ConnectError as exc:
        logger.warning("iris client job list failed for prefix %s: %s", job_id, exc)
        return None


def fetch_leaf_summaries(client: IrisClient, job_tree: list[dict]) -> list[dict]:
    """Fetch job summaries for every leaf job in the tree.

    Per-task data (``memory_peak_mb``, ``error``, ``exit_code``) lives on each
    job's own task array, which the job tree does not return.
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
        s = fetch_job_summary(client, job_id)
        if s is not None:
            summaries.append(s)
    return summaries


def _query_rows(controller: ControllerServiceClientSync, sql: str) -> list[dict[str, object]]:
    response = controller.execute_raw_query(query_pb2.RawQueryRequest(sql=sql))
    columns = [column.name for column in response.columns]
    rows = [json.loads(row) for row in response.rows]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def fetch_raw_query_task_wall_ms(
    controller: ControllerServiceClientSync, job_id: str, *, include_failed: bool = False
) -> int | None:
    """Sum per-attempt wall-clock durations across the subtree via ExecuteRawQuery."""
    state_filter = "" if include_failed else f"AND t.state = {_TASK_STATE_SUCCEEDED}"
    sql = _TASK_WALL_TIME_SQL.format(job_id=job_id.replace("'", "''"), state_filter=state_filter)
    try:
        rows = _query_rows(controller, sql)
        if not rows:
            return None
        val = rows[0]["task_wall_ms"]
        return int(val) if val is not None else 0
    except (ConnectError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("iris query task_wall_ms failed: %s", exc)
        return None


def fetch_raw_query_task_wall_ms_by_child(
    controller: ControllerServiceClientSync, job_id: str, *, include_failed: bool = False
) -> dict[str, int] | None:
    """Return per-direct-child task wall ms via ExecuteRawQuery, keyed by child job_id."""
    state_filter = "" if include_failed else f"AND t.state = {_TASK_STATE_SUCCEEDED}"
    sql = _TASK_WALL_TIME_BY_CHILD_SQL.format(job_id=job_id.replace("'", "''"), state_filter=state_filter)
    try:
        rows = _query_rows(controller, sql)
        return {str(row["child_job_id"]): int(row["task_wall_ms"]) for row in rows}
    except (ConnectError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("iris query by_child failed: %s", exc)
        return None


def bucket_by_step(by_child: dict[str, int], parent_id: str) -> dict[str, int | None]:
    """Bucket per-child task_wall_ms into step names using the same prefix logic as compute_stage_wall_seconds.

    All EXPECTED_STEPS are always present; steps with no matching child jobs have value None.
    """
    parent_depth = _job_depth(parent_id)
    by_step: dict[str, int | None] = {step: None for step in EXPECTED_STEPS}
    for child_job_id, task_wall_ms in by_child.items():
        if not child_job_id.startswith(parent_id):
            continue
        if _job_depth(child_job_id) != parent_depth + 1:
            continue
        name = child_job_id.rsplit("/", 1)[-1]
        for prefix, step in _STEP_PREFIXES.items():
            if name.startswith(prefix):
                by_step[step] = (by_step.get(step) or 0) + task_wall_ms
                break
    return by_step


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
        status_sp = StoragePath(status_path)
        if not status_sp.exists():
            return None
        return json.loads(status_sp.read_text())
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
        collected_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
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
        report.warnings.append("iris client job summary failed; wall_seconds_total unavailable")
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
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


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
    StoragePath(dest).write_text(report.to_json())
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--job-id", required=True, help="Iris job id of the ferry run.")
@click.option(
    "--iris-config",
    default="lib/iris/config/marin.yaml",
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
@click.option(
    "--task-wall-time/--no-task-wall-time",
    "fetch_task_wall_time",
    default=True,
    help="Fetch summed task wall time via ExecuteRawQuery and include it in the report.",
)
def main(
    job_id: str,
    iris_config: Path,
    status_path: str | None,
    report_name: str | None,
    out: Path | None,
    gcs_prefix: str | None,
    gcs_output_env: str | None,
    fetch_task_wall_time: bool,
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

    with connect_controller(config_file=iris_config) as endpoint:
        with (
            IrisClient.remote(endpoint.url, workspace=_REPO_ROOT, credentials=endpoint.credentials) as client,
            rpc_client(endpoint.url, endpoint.credentials) as controller,
        ):
            summary = fetch_job_summary(client, job_id)
            job_tree = fetch_job_tree(client, job_id)
            leaf_summaries = fetch_leaf_summaries(client, job_tree) if job_tree else []
            status = load_ferry_status(status_path)

            report = build_report(
                job_id=job_id,
                summary=summary,
                job_tree=job_tree,
                leaf_summaries=leaf_summaries,
                status=status,
                workflow_env=workflow_env,
            )

            if fetch_task_wall_time:
                task_wall_ms = fetch_raw_query_task_wall_ms(controller, job_id)
                if task_wall_ms is None:
                    report.warnings.append("iris query task_wall_ms: failed; sum_task_wall_seconds_total unset")
                else:
                    report.sum_task_wall_seconds_total = task_wall_ms / 1000.0
                by_child = fetch_raw_query_task_wall_ms_by_child(controller, job_id)
                if by_child is None:
                    report.warnings.append("iris query by_child: failed; stage_sum_task_wall_seconds empty")
                else:
                    report.stage_sum_task_wall_seconds = {
                        step: ms / 1000.0 if ms is not None else None
                        for step, ms in bucket_by_step(by_child, job_id).items()
                    }

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
