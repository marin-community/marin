#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor the A/B exact-dedup benchmark jobs (4 variants).

Usage:
    uv run python experiments/dedup/monitor_bench.py          # one-shot status
    uv run python experiments/dedup/monitor_bench.py --loop    # poll every 60s
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

IRIS_CMD = ["uv", "run", "iris", "--config", "lib/iris/examples/marin-dev.yaml"]

# Job IDs from bench_external_merge_exact.py run on 2026-04-01
JOBS = {
    "main-10pct": "/power/iris-run-nemotron_1split_exact-20260402-005308",
    "branch-10pct": "/power/iris-run-nemotron_1split_exact-20260402-005322",
    "main-full": "/power/iris-run-nemotron_1split_exact-20260402-003956",
    "branch-full": "/power/iris-run-nemotron_1split_exact-20260402-004004",
}


def run_quiet(cmd: list[str], timeout: int = 60) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in r.stdout.splitlines() if not re.match(r"^I\d{8}", l))


def get_job_state(job_id: str) -> str:
    """Get the top-level job state from iris job list."""
    raw = run_quiet([*IRIS_CMD, "job", "list"], timeout=90)
    for line in raw.splitlines():
        if job_id in line:
            parts = line.split()
            for p in parts:
                if p in ("running", "succeeded", "failed", "killed", "pending", "queued"):
                    return p
    return "unknown"


def discover_coord_actor(job_id: str) -> str | None:
    """Find the coordinator actor endpoint from child job names."""
    raw = run_quiet([*IRIS_CMD, "job", "list"], timeout=90)
    # Look for the coord child: .../zephyr-*-p0-a0/zephyr-*-p0-coord-0
    for line in raw.splitlines():
        if job_id in line and "coord-0" in line:
            # First token is the job ID / task path
            path = line.split()[0]
            return path
    # Try alternative: the coordinator is a child of the zephyr pipeline job
    for line in raw.splitlines():
        if job_id in line and "-p0-a0" in line and "workers" not in line and "coord" not in line:
            # This is the pipeline child — construct coord path
            pipeline_id = line.split()[0]
            # Coordinator actor is named like the pipeline but with -coord-0 appended
            base = pipeline_id.rsplit("/", 1)[-1]  # e.g. zephyr-exact-para-dedup-XXX-p0-a0
            coord_name = base.replace("-a0", "-coord-0")
            return f"{pipeline_id}/{coord_name}"
    return None


def actor_call(endpoint: str, method: str, timeout: int = 60) -> str:
    return run_quiet([*IRIS_CMD, "actor", "call", endpoint, method], timeout=timeout)


def parse_status(raw: str) -> dict:
    m = re.search(
        r"stage='([^']*)'.*?completed=(\d+).*?total=(\d+).*?retries=(\d+)"
        r".*?in_flight=(\d+).*?queue_depth=(\d+).*?done=(\w+).*?fatal_error=(\w+)",
        raw,
    )
    if not m:
        return {"error": raw[:200]}
    return {
        "stage": m.group(1),
        "completed": int(m.group(2)),
        "total": int(m.group(3)),
        "retries": int(m.group(4)),
        "in_flight": int(m.group(5)),
        "queued": int(m.group(6)),
        "done": m.group(7),
        "fatal": m.group(8),
        "busy": raw.count("'state': 'busy'"),
        "idle": raw.count("'state': 'idle'"),
        "dead": raw.count("'state': 'dead'"),
        "ready": raw.count("'state': 'ready'"),
    }


def parse_counters(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw[:200]}


def fmt_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def print_status():
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n{'=' * 72}")
    print(f"  Benchmark Monitor — {now}")
    print(f"  Jobs: {len(JOBS)} (main vs branch x 10%/full)")
    print(f"{'=' * 72}\n")

    all_done = True
    for label, job_id in JOBS.items():
        state = get_job_state(job_id)
        print(f"  [{label}] {job_id}")
        print(f"    State: {state}")

        if state not in ("running",):
            if state not in ("succeeded",):
                all_done = False
            else:
                print("    ✓ Completed")
            print()
            continue

        all_done = False
        coord = discover_coord_actor(job_id)
        if not coord:
            print("    (coordinator not yet discoverable)")
            print()
            continue

        try:
            status = parse_status(actor_call(coord, "get_status", timeout=90))
        except Exception as e:
            print(f"    (status query failed: {e})")
            print()
            continue

        if "error" in status:
            print(f"    Status: {status['error']}")
            print()
            continue

        pct = status["completed"] / status["total"] * 100 if status["total"] else 0
        print(f"    Stage:    {status['stage']}")
        print(f"    Progress: {status['completed']}/{status['total']} ({pct:.0f}%)")
        print(f"    In-flight: {status['in_flight']}  Queued: {status['queued']}  Retries: {status['retries']}")
        print(f"    Workers:  {status['busy']} busy, {status['idle']} idle, {status['dead']} dead")

        try:
            counters = parse_counters(actor_call(coord, "get_counters"))
            if "error" not in counters:
                counter_str = ", ".join(f"{k}={fmt_count(v)}" for k, v in counters.items())
                print(f"    Counters: {counter_str}")
        except Exception:
            pass

        print()

    if all_done:
        print("  *** ALL JOBS FINISHED ***\n")
    return all_done


def main():
    loop = "--loop" in sys.argv
    if loop:
        while True:
            done = print_status()
            if done:
                break
            print("  (next check in 10m, Ctrl-C to stop)\n")
            time.sleep(600)
    else:
        print_status()


if __name__ == "__main__":
    main()
