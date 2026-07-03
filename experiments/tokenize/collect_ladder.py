# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the tokenizer bake-off ladder from Iris without ever streaming a live or failed job.

Ad-hoc ``iris job logs | grep`` is slow (it pulls the whole log) and hangs on running or
failed jobs. This queries job STATE first (one ``iris job list``), then pulls only the metric
tail of each *succeeded* run and reduces it with
:func:`experiments.tokenize.collect_metrics.collect_run`. Running / failed / missing jobs are
reported and skipped, never awaited, and a per-subprocess timeout keeps one stuck job from
wedging the whole collection. Reused names that carry stale failed attempts (e.g. a failed
``...-s1500`` superseded by a succeeded ``...-s1500-r2``) are handled: the failed one is
skipped, the succeeded one contributes its point.

Two ways to name the runs:

    # discover every top-level grug-bakeoff-* job and infer its arm from the name
    uv run python -m experiments.tokenize.collect_ladder --prefix grug-bakeoff- --out ladder.json

    # or list explicit arm=job-path points (like collect_metrics' --point)
    uv run python -m experiments.tokenize.collect_ladder \
        --point marin-128k=/power/grug-bakeoff-marin-128k-s1500 --out ladder.json

Assemble with :func:`experiments.tokenize.collect_metrics.build_ladder` into the
``{arms: {name: [[flops, bpb], ...]}}`` file that
:mod:`experiments.tokenize.bakeoff_analysis` consumes.

Cluster access: export ``KUBECONFIG`` for the target cluster before running; ``GH_TOKEN`` is
stripped from the iris subprocess because it confuses the controller's auth.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass

from experiments.tokenize.collect_metrics import RunPoint, build_ladder, collect_run

DEFAULT_CLUSTER = "cw-rno2a"
DEFAULT_NAME_PREFIX = "grug-bakeoff-"
DEFAULT_MAX_LOG_LINES = 3000
DEFAULT_TIMEOUT = 120.0
SUCCEEDED_STATE = "JOB_STATE_SUCCEEDED"
_STATE_PREFIX = "JOB_STATE_"
IRIS = ("uv", "run", "iris")

# arm = the run name with the DEFAULT_NAME_PREFIX removed and the -s<steps> compute point
# (plus any -r<n> retry suffix) stripped: marin-128k-ngram-s3500-r2 -> marin-128k-ngram. A name
# without an -s<steps> segment (e.g. a smoke run) is not a ladder cell and infers no arm.
_ARM_RE = re.compile(r"^(?P<arm>.+?)-s\d+(?:-r\d+)?$")


@dataclass(frozen=True)
class LadderJob:
    """A ladder run to collect: its arm, wire-form job path, and last-known controller state."""

    arm: str
    job_id: str
    state: str


@dataclass(frozen=True)
class CollectResult:
    """Outcome for one run: either a ladder ``point`` or a ``skip_reason`` (exactly one is set)."""

    job: LadderJob
    point: RunPoint | None
    skip_reason: str | None


def iris_env() -> dict[str, str]:
    """The caller's environment for iris subprocesses, minus GH_TOKEN (which confuses auth)."""
    env = dict(os.environ)
    env.pop("GH_TOKEN", None)
    return env


def _run_iris(args: list[str], cluster: str, env: dict[str, str], timeout: float) -> str:
    """Run one ``iris --cluster=<cluster> <args...>`` and return its stdout.

    Raises ``subprocess.TimeoutExpired`` if it exceeds ``timeout`` and ``CalledProcessError`` on
    a non-zero exit; the caller decides whether that is fatal or a per-job skip.
    """
    cmd = [*IRIS, f"--cluster={cluster}", *args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def fetch_job_states(cluster: str, env: dict[str, str], list_prefix: str, timeout: float) -> dict[str, str]:
    """Map wire-form job_id -> state for every job under the anchored ``list_prefix``."""
    out = _run_iris(["job", "list", "--prefix", list_prefix, "--json"], cluster, env, timeout)
    return {job["job_id"]: job.get("state", "") for job in json.loads(out)}


def infer_arm(run_name: str, name_prefix: str) -> str | None:
    """Infer the bake-off arm from a top-level run name, or None if it is not a ladder cell."""
    if not run_name.startswith(name_prefix):
        return None
    match = _ARM_RE.match(run_name[len(name_prefix) :])
    return match.group("arm") if match else None


def discover_jobs(states: dict[str, str], name_prefix: str) -> list[LadderJob]:
    """Top-level jobs whose name is a ``<prefix><arm>-s<steps>[-r<n>]`` ladder cell.

    A job is top-level when its immediate parent path is not itself a job (so per-task children
    like ``grug-train-bakeoff-*`` and ``tokenize-*`` are dropped, as are non-ladder names).
    """
    ids = set(states)
    jobs = []
    for job_id, state in states.items():
        if job_id.rsplit("/", 1)[0] in ids:  # has a parent job -> a descendant, not a ladder cell
            continue
        arm = infer_arm(job_id.rsplit("/", 1)[-1], name_prefix)
        if arm is not None:
            jobs.append(LadderJob(arm=arm, job_id=job_id, state=state))
    return jobs


def resolve_explicit(points: list[str], states: dict[str, str]) -> list[LadderJob]:
    """Turn ``arm=job-path`` specs into LadderJobs, attaching each path's state (or "" if absent)."""
    jobs = []
    for spec in points:
        arm, job_id = spec.split("=", 1)
        jobs.append(LadderJob(arm=arm, job_id=job_id, state=states.get(job_id, "")))
    return jobs


def collect_job(job: LadderJob, cluster: str, env: dict[str, str], max_log_lines: int, timeout: float) -> CollectResult:
    """Pull a succeeded run's metric tail and reduce it to a point; skip anything else."""
    if job.state != SUCCEEDED_STATE:
        return CollectResult(job=job, point=None, skip_reason=_state_reason(job.state))
    try:
        out = _run_iris(["job", "logs", job.job_id, "--max-lines", str(max_log_lines), "--tail"], cluster, env, timeout)
    except subprocess.TimeoutExpired:
        return CollectResult(job=job, point=None, skip_reason=f"log fetch timed out after {timeout:g}s")
    except subprocess.CalledProcessError as exc:
        return CollectResult(job=job, point=None, skip_reason=f"log fetch failed (exit {exc.returncode})")

    try:
        point = collect_run(out.splitlines(), job.arm)
    except ValueError as exc:
        return CollectResult(job=job, point=None, skip_reason=str(exc))
    if point.bpb is None:
        return CollectResult(job=job, point=None, skip_reason="no eval/bpb in logs")
    return CollectResult(job=job, point=point, skip_reason=None)


def _state_reason(state: str) -> str:
    if not state:
        return "not found on controller"
    short = state[len(_STATE_PREFIX) :] if state.startswith(_STATE_PREFIX) else state
    return f"state {short}"


def _short_state(state: str) -> str:
    if not state:
        return "MISSING"
    return state[len(_STATE_PREFIX) :] if state.startswith(_STATE_PREFIX) else state


def print_table(cluster: str, results: list[CollectResult]) -> None:
    """Print one aligned row per run: arm, state, and either its (flops, bpb) or why it was skipped."""
    rows = []
    for result in results:
        if result.point is not None:
            outcome = f"flops={result.point.total_train_flops:.3e} bpb={result.point.bpb:.4f}"
        else:
            outcome = f"skip: {result.skip_reason}"
        rows.append((result.job.arm, _short_state(result.job.state), outcome, result.job.job_id))

    arm_w = max((len(r[0]) for r in rows), default=3)
    state_w = max((len(r[1]) for r in rows), default=5)
    outcome_w = max((len(r[2]) for r in rows), default=6)
    print(f"\nbake-off ladder collection on {cluster}: {len(rows)} runs")
    for arm, state, outcome, job_id in sorted(rows):
        print(f"  {arm:<{arm_w}}  {state:<{state_w}}  {outcome:<{outcome_w}}  {job_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER)
    ap.add_argument("--out", required=True, help="path to write the ladder JSON")
    ap.add_argument("--prefix", default=DEFAULT_NAME_PREFIX, help="run-name prefix for discovery mode")
    ap.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="ARM=JOB_PATH",
        help="explicit run as arm=wire-job-path (repeatable); disables discovery",
    )
    ap.add_argument(
        "--list-prefix",
        default="/",
        help="anchored wire-form prefix for the job list; narrow it (e.g. /power/grug-bakeoff-) on large clusters",
    )
    ap.add_argument("--max-log-lines", type=int, default=DEFAULT_MAX_LOG_LINES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-subprocess timeout in seconds")
    args = ap.parse_args()

    env = iris_env()
    states = fetch_job_states(args.cluster, env, args.list_prefix, args.timeout)
    jobs = resolve_explicit(args.point, states) if args.point else discover_jobs(states, args.prefix)
    if not jobs:
        raise SystemExit(f"no runs matched (prefix={args.prefix!r}, list-prefix={args.list_prefix!r}) on {args.cluster}")

    results = [collect_job(job, args.cluster, env, args.max_log_lines, args.timeout) for job in jobs]
    print_table(args.cluster, results)

    ladder = build_ladder(result.point for result in results if result.point is not None)
    with open(args.out, "w") as f:
        json.dump(ladder, f, indent=2)
    summary = ", ".join(f"{arm} ({len(pts)} pts)" for arm, pts in ladder["arms"].items())
    print(f"\nwrote {args.out}: {summary or '(no points collected)'}")


if __name__ == "__main__":
    main()
