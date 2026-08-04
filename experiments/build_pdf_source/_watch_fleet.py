# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- a terminal dashboard for the fleet extract run.

DELETE when the production run is recorded. Nothing imports this; it runs on a laptop against the
cluster's log API, not on the cluster.

Everything renders from the FLEET-STATS JSON lines the driver emits every minute (registered pods,
broker depth, completed total, pool-job states), fetched with a server-side substring filter so the
sender fleet's log volume cannot drown them out of a tail window. Throughput and ETA come from
differentiating ``completed_total`` across polls.

    uv run python -m experiments.build_pdf_source._watch_fleet [entry-job-id]

Without an argument the most recently submitted job under the entry-job prefix is watched.
"""

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass

from iris.cli.connect import open_iris_client
from iris.client import IrisClient
from iris.cluster.types import JobName
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_CLUSTER = "marin"
# Anchored wire-form prefix for discovery: `list_jobs` matching is prefix-based, and on a busy
# cluster an unfiltered recency listing is all other users' work.
_JOB_PREFIX = "/muchanem/pdf-extract-backfill"
_POLL_SECONDS = 30.0
# Converters per pool pod; mirrors extract_fleet's operating point for the utilization figure.
_PROCESSES_PER_INSTANCE = 4
# The router's OCR-route count in the 10% sample -- the backfill run's denominator. Approximate on
# purpose: the authoritative total is whatever the classify step routed, and the progress bar only
# needs to be honest to a fraction of a percent.
_EXPECTED_DOCUMENTS = 101_332

_STATS_MARKER = "FLEET-STATS "


@dataclass(frozen=True)
class Snapshot:
    stats: dict | None
    last_log_line: str


def _discover_job_id(client: IrisClient) -> str:
    jobs = client.list_jobs(prefix=_JOB_PREFIX, limit=10)
    if not jobs:
        raise SystemExit(f"No job under {_JOB_PREFIX} found; pass the entry job id explicitly")
    # The listing is full of descendants (step driver, pool jobs, sender workers), whose ids extend
    # the entry job's. Truncate back to the entry job's two path segments; log fetches on the entry
    # job cover the whole tree.
    segments = jobs[0].job_id.split("/")
    return "/".join(segments[:3])


def _poll(client: IrisClient, job_name: JobName) -> Snapshot:
    stats_entries = client.fetch_task_logs(job_name, substring=_STATS_MARKER.strip(), tail=True, max_lines=3)
    stats = None
    for entry in reversed(stats_entries):
        if _STATS_MARKER in entry.data:
            stats = json.loads(entry.data.split(_STATS_MARKER, 1)[1])
            break
    tail_entries = client.fetch_task_logs(job_name, tail=True, max_lines=1)
    last_line = tail_entries[-1].data.strip()[-160:] if tail_entries else ""
    return Snapshot(stats=stats, last_log_line=last_line)


def _rate_per_minute(history: deque[tuple[float, int]]) -> float | None:
    if len(history) < 2:
        return None
    (t0, c0), (t1, c1) = history[0], history[-1]
    if t1 <= t0:
        return None
    return 60.0 * (c1 - c0) / (t1 - t0)


def _render(job_id: str, snapshot: Snapshot, history: deque[tuple[float, int]]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()

    stats = snapshot.stats
    if stats is None:
        table.add_row("fleet", "waiting for the first FLEET-STATS line (fleet starting up)")
    else:
        pods = stats["pods_registered"]
        converters = stats["converters"]
        converting = stats["converting"]
        jobs = ", ".join(f"{state} {count}" for state, count in sorted(stats["pool_jobs"].items()))
        utilization = f"{100 * converting / converters:.0f}%" if converters else "-"
        table.add_row("workers", f"{pods} pods registered ({converters} converters); jobs: {jobs}")
        table.add_row("in flight", f"{converting} converting ({utilization} util), {stats['queued']} queued at broker")

        done = stats["completed_total"]
        pct = 100 * done / _EXPECTED_DOCUMENTS
        progress_line = f"{done:,} / ~{_EXPECTED_DOCUMENTS:,} documents ({pct:.1f}%)"
        rate = _rate_per_minute(history)
        if rate and rate > 0:
            remaining_minutes = (_EXPECTED_DOCUMENTS - done) / rate
            progress_line += f"  |  {rate:,.0f} docs/min, ~{remaining_minutes / 60:.1f}h left"
        table.add_row("progress", progress_line)
        table.add_row("stats age", f"{int(time.time() - stats['time'])}s")

    table.add_row("last log", Text(snapshot.last_log_line, style="dim"))
    # Text, not str: job ids start with "/", so "[{job_id}]" would parse as a closing markup tag.
    return Panel(table, title=Text(f"pdf-extract-fleet  [{job_id}]"), border_style="green")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", nargs="?", help="entry job id; discovered from the job listing when omitted")
    args = parser.parse_args()

    history: deque[tuple[float, int]] = deque(maxlen=60)
    console = Console()
    with open_iris_client(cluster_name=_CLUSTER, workspace=None) as client:
        job_id = args.job_id or _discover_job_id(client)
        job_name = JobName.from_wire(job_id)
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    snapshot = _poll(client, job_name)
                except Exception as error:
                    live.update(Panel(Text(f"poll failed: {error}"), border_style="red"))
                    time.sleep(_POLL_SECONDS)
                    continue
                if snapshot.stats is not None:
                    history.append((time.time(), snapshot.stats["completed_total"]))
                live.update(_render(job_id, snapshot, history))
                time.sleep(_POLL_SECONDS)


if __name__ == "__main__":
    main()
