#!/usr/bin/env python3
"""Monitor Ray jobs and optionally resubmit on failure.

This is designed for long-running Ray/TPU sweeps where the top-level driver job may
fail due to preemption or transient cluster issues. It can be run in a separate
terminal to periodically print a compact status summary, and (optionally) execute
a user-provided resubmit command when a job is no longer RUNNING.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass

from fray.cluster.ray.dashboard import DashboardConfig, ray_dashboard


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    submission_id: str | None
    status: str | None
    message: str | None


def _run_json(cmd: list[str], timeout_s: int) -> list[dict]:
    try:
        out = subprocess.check_output(cmd, text=True, timeout=timeout_s, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = (e.output or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg}") from e
    # Ray CLI sometimes prints extra log lines (e.g. NumExpr) before the JSON payload.
    # Salvage by extracting the outermost JSON list.
    payload = out.strip()
    if not payload.startswith("["):
        start = payload.find("[")
        end = payload.rfind("]")
        if start != -1 and end != -1 and end > start:
            payload = payload[start : end + 1]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        head = out[:500].strip()
        raise RuntimeError(f"Failed to parse JSON from: {' '.join(cmd)}\n{head}") from e
    if not isinstance(data, list):
        raise RuntimeError(f"Expected JSON list from: {' '.join(cmd)}")
    return data


def _get_job(job_id: str, timeout_s: int) -> JobSummary | None:
    jobs = _run_json(
        ["ray", "list", "jobs", "--detail", "--format=json", "--limit=10000", "--filter", f"job_id={job_id}"],
        timeout_s,
    )
    if not jobs:
        return None
    job = jobs[0]
    return JobSummary(
        job_id=job.get("job_id") or job_id,
        submission_id=job.get("submission_id"),
        status=job.get("status"),
        message=job.get("message"),
    )


def _task_state_counts(job_id: str, timeout_s: int, limit: int) -> Counter[str]:
    tasks = _run_json(
        [
            "ray",
            "list",
            "tasks",
            "--format=json",
            "--detail",
            f"--limit={limit}",
            "--filter",
            f"job_id={job_id}",
        ],
        timeout_s,
    )
    return Counter(t.get("state") or "UNKNOWN" for t in tasks)


def _render_counts(counts: Counter[str]) -> str:
    if not counts:
        return "tasks=0"
    parts = [f"{k.lower()}={v}" for k, v in counts.most_common()]
    return "tasks " + " ".join(parts)


def _maybe_resubmit(job_id: str, summary: JobSummary | None, resubmit_cmd: str | None) -> None:
    if not resubmit_cmd:
        return
    status = summary.status if summary else None
    if status == "RUNNING":
        return
    env = {
        "RAY_JOB_ID": job_id,
        "RAY_SUBMISSION_ID": (summary.submission_id or "") if summary else "",
        "RAY_JOB_STATUS": (status or "") if summary else "",
    }
    print(f"resubmit: job_id={job_id} status={status!s}", flush=True)
    subprocess.run(resubmit_cmd, shell=True, check=True, env={**os.environ, **env})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Cluster config YAML (e.g. infra/marin-us-central1.yaml)")
    ap.add_argument("--job-id", action="append", required=True, help="Ray job_id to monitor (repeatable)")
    ap.add_argument("--interval-sec", type=int, default=3600, help="Poll interval in seconds (default: 3600)")
    ap.add_argument("--watch", action="store_true", help="Watch indefinitely (default: once)")
    ap.add_argument("--timeout-sec", type=int, default=60, help="Timeout for each ray CLI call")
    ap.add_argument("--task-limit", type=int, default=500, help="Max tasks to fetch per job for state summary")
    ap.add_argument(
        "--resubmit-cmd",
        default=None,
        help=(
            "Shell command to run if a job is not RUNNING. "
            "Env vars set: RAY_JOB_ID, RAY_SUBMISSION_ID, RAY_JOB_STATUS."
        ),
    )
    args = ap.parse_args()

    cfg = DashboardConfig.from_cluster(args.config)
    with ray_dashboard(cfg):
        while True:
            for job_id in args.job_id:
                try:
                    summary = _get_job(job_id, timeout_s=args.timeout_sec)
                    if summary is None:
                        print(f"{job_id}\tMISSING", flush=True)
                        continue
                    counts = _task_state_counts(job_id, timeout_s=args.timeout_sec, limit=args.task_limit)
                    print(
                        f"{job_id}\t{summary.status}\t{summary.submission_id or ''}\t{_render_counts(counts)}",
                        flush=True,
                    )
                    _maybe_resubmit(job_id, summary, args.resubmit_cmd)
                except Exception as e:
                    print(f"{job_id}\tERROR\t{e}", flush=True)

            if not args.watch:
                return
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
