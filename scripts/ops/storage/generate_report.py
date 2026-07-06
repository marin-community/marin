#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end storage report: scan all marin-* buckets, aggregate, and publish.

Driven from one command on the operator's laptop (or a CI runner). The
orchestrator opens a tunnel to the named Iris cluster and submits each compute
stage as an Iris job using the Python client (no subprocess into the iris CLI),
then publishes the result locally.

  1. Scan stage    — Iris coordinator + N worker replicas walk every GCS
                     prefix and write consolidated parquet segments to
                     STAGING_DIR (delegates to ``run_distributed`` in
                     ``scan_gcs.py``).
  2. Dedup stage   — Iris coordinator job runs a Zephyr ``group_by`` to
                     collapse the raw parquets into one row per (bucket, name)
                     under STAGING_DIR/deduped. Pipeline construction lives
                     in this file (see ``_dedup_stage``).
  3. Report stage  — Iris coordinator job reads the deduped parquets, builds a
                     DuckDB rollup + week-over-week diff (see ``render_report``)
                     and writes ``report.md`` back into STAGING_DIR.
  4. Publish       — Local: fetch ``report.md``, optionally push a gist
                     (``--gist public|secret|none``) and/or post a summary with
                     the biggest increases/decreases to Discord (``--discord``).

Prereqs (local / CI runner):
    - ``gh`` authenticated as the gist owner (for ``--gist``)
    - ``gcloud`` with GCS read access to fetch ``report.md`` back
    - The named cluster's controller is reachable (same tunnel machinery as
      ``iris --cluster=<name> ...``)
    - For ``--discord``, the channel webhook resolvable by ``scripts/ops/discord.py``

Usage:
    ./scripts/ops/storage/generate_report.py
    ./scripts/ops/storage/generate_report.py --workers 64
    ./scripts/ops/storage/generate_report.py --skip-scan          # reuse existing parquets
    ./scripts/ops/storage/generate_report.py --skip-scan --skip-dedup --skip-report  # just re-publish
    # Weekly automation (see .github/workflows/ops-storage-report.yaml):
    ./scripts/ops/storage/generate_report.py --gist secret --discord internal-discuss
"""

import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import fsspec
from fray import ResourceConfig
from iris.cli.connect import open_iris_client
from iris.client import IrisClient
from iris.cluster.constraints import Constraint, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from zephyr import Dataset, ZephyrContext

from scripts.ops.storage.constants import MARIN_BUCKETS
from scripts.ops.storage.render_report import (
    DEFAULT_CHANGE_THRESHOLD_BYTES,
    compute_changes,
    find_latest_snapshot,
    generate_report,
    load_parquet_db,
    read_snapshot,
    render_changes_section,
    snapshot_dir_summary,
    snapshot_path,
    write_snapshot,
)
from scripts.ops.storage.scan_gcs import run_distributed

DEFAULT_STAGING_DIR = "gs://marin-us-central2/tmp/storage-scan"
# Stable location for week-over-week snapshots, independent of the (often
# date-stamped or truncated) staging dir so history accumulates across runs.
DEFAULT_HISTORY_DIR = "gs://marin-us-central2/storage-report-history"
REPO_ROOT = Path(__file__).resolve().parents[3]

# Discord summary: rows to show per Increases/Decreases section (full detail is
# in the gist) and the Overview rows lifted from report.md for the headline.
MAX_CHANGES_IN_MESSAGE = 5
_OVERVIEW_LABELS = ("Total Objects", "Total Size", "Est. Monthly Cost")


# ---------------------------------------------------------------------------
# Stage bodies — top-level so Entrypoint.from_callable can cloudpickle them.
# These run on the Iris coordinator, not on the operator's laptop.
# ---------------------------------------------------------------------------


def _scan_stage(staging_dir: str, workers: int) -> None:
    """Iris-side entrypoint for stage 1: distributed scan into parquet segments."""

    run_distributed(
        buckets=MARIN_BUCKETS,
        num_workers=workers,
        project=None,
        staging_dir=staging_dir,
    )


def _dedup_bucket_name_key(row: dict) -> tuple[str, str]:
    """Group key for the dedup pipeline.

    Top-level (not a lambda) so cloudpickle round-trips cleanly when shipping
    the Zephyr pipeline to the coordinator job.
    """
    return (row["bucket"], row["name"])


def _dedup_stage(input_glob: str, output_dir: str, num_shards: int, worker_cpu: int, worker_ram: str) -> None:
    """Iris-side entrypoint for stage 2: Zephyr ``group_by`` dedup.

    Belt-and-suspenders for the scan: every parquet row is the metadata for
    one GCS object, but the upstream scan can in principle emit the same
    object more than once (RPC retries, overlapping prefix scans, etc). This
    collapses those into one row per (bucket, name) and lives inside the
    coordinator job so the pipeline is constructed where it executes.
    """

    output_pattern = f"{output_dir.rstrip('/')}/objects-{{shard:05d}}.parquet"

    pipeline = (
        Dataset.from_files(input_glob)
        .load_parquet()
        .deduplicate(
            key=_dedup_bucket_name_key,
            num_output_shards=num_shards,
        )
        .write_parquet(output_pattern)
    )

    ctx = ZephyrContext(
        name="storage-dedup",
        resources=ResourceConfig(cpu=worker_cpu, ram=worker_ram),
    )
    ctx.execute(pipeline)


def _report_stage(deduped_dir: str, report_path: str, history_dir: str, today: str) -> None:
    """Iris-side entrypoint for stage 3: build the markdown report.

    Also archives a compact per-prefix snapshot to ``history_dir`` and, when a
    prior snapshot exists, inserts a week-over-week changes section diffing
    against it. ``today`` (YYYY-MM-DD) names this run's snapshot and excludes
    it from the "most recent prior" lookup.
    """

    conn = load_parquet_db(deduped_dir)
    current = snapshot_dir_summary(conn)

    prior = find_latest_snapshot(history_dir, before_date=today)
    if prior is None:
        changes_section = render_changes_section([], previous_date=None, threshold_bytes=DEFAULT_CHANGE_THRESHOLD_BYTES)
    else:
        prev_path, prev_date = prior
        changes = compute_changes(current, read_snapshot(prev_path), threshold_bytes=DEFAULT_CHANGE_THRESHOLD_BYTES)
        changes_section = render_changes_section(
            changes, previous_date=prev_date, threshold_bytes=DEFAULT_CHANGE_THRESHOLD_BYTES
        )

    report = generate_report(conn, changes_section=changes_section)
    with fsspec.open(report_path, "w") as f:
        f.write(report)

    write_snapshot(current, snapshot_path(history_dir, today))
    print(f"Report written to {report_path}; snapshot archived for {today}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local helpers (run on the operator's laptop)
# ---------------------------------------------------------------------------


def _push_gist(content: str, description: str, filename: str, *, public: bool) -> str:
    """Create a gist via ``gh`` and return the URL.

    ``public=False`` creates a secret gist (still URL-accessible, but not
    listed or indexed) — the right default for automated/internal runs.
    """
    cmd = ["gh", "gist", "create"]
    if public:
        cmd.append("--public")
    cmd += ["--filename", filename, "--desc", description, "-"]  # "-" reads body from stdin
    result = subprocess.run(
        cmd,
        input=content,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _fetch_report(report_path: str) -> str:
    """Read report.md with gcloud — avoids the Python TLS/cert maze for gs://."""
    return subprocess.run(["gcloud", "storage", "cat", report_path], check=True, text=True, capture_output=True).stdout


def _post_to_discord(channel: str, message: str) -> None:
    subprocess.run(
        ["uv", "run", "python", str(REPO_ROOT / "scripts" / "ops" / "discord.py"), "-c", channel],
        input=message,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )


def _extract_overview(report_md: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for label in _OVERVIEW_LABELS:
        match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|", report_md)
        if match:
            out[label] = match.group(1).strip()
    return out


def _extract_changes(report_md: str) -> tuple[str | None, list[str], list[str]]:
    """Return (since_date, increase_lines, decrease_lines) from the changes section.

    Rows are classified by the sign of the Δ Size cell. ``since_date`` is None
    on a baseline run; each list preserves the report's (magnitude-sorted) order.
    """
    section = re.search(r"## Week-over-Week Changes\n(.*?)(?:\n## |\Z)", report_md, re.S)
    if not section:
        return None, [], []
    body = section.group(1)
    since_match = re.search(r"since (\d{4}-\d{2}-\d{2})", body)
    since = since_match.group(1) if since_match else None

    increases: list[str] = []
    decreases: list[str] = []
    for line in body.splitlines():
        if not line.startswith("|") or "---" in line or "Δ Size" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        bucket, prefix, delta, _now, _was, status = cells
        rendered = f"`{bucket}/{prefix}` {delta} ({status})"
        (increases if delta.startswith("+") else decreases).append(rendered)
    return since, increases, decreases


def _change_block(title: str, lines: list[str]) -> str:
    if not lines:
        return f"**{title}:** _none above threshold_"
    shown = lines[:MAX_CHANGES_IN_MESSAGE]
    block = f"**{title}:**\n" + "\n".join(f"- {line}" for line in shown)
    if len(lines) > MAX_CHANGES_IN_MESSAGE:
        block += f"\n- _(+{len(lines) - MAX_CHANGES_IN_MESSAGE} more in the report)_"
    return block


def _compose_discord_message(report_md: str, *, date: str, gist_url: str | None) -> str:
    """Two-section summary (increases first — growth is the alarming signal)."""
    overview = _extract_overview(report_md)
    since, increases, decreases = _extract_changes(report_md)

    headline = " · ".join(overview.values()) if overview else "(totals unavailable)"
    if since is None:
        change_section = "_Baseline run — week-over-week diffs start next run._"
    elif not increases and not decreases:
        change_section = f"_No prefix changes above threshold since {since}._"
    else:
        change_section = (
            f"_Changes since {since}:_\n\n"
            + _change_block("Biggest increases", increases)
            + "\n\n"
            + _change_block("Biggest decreases", decreases)
        )

    report_line = f"- report: {gist_url}\n" if gist_url else ""
    return f"**Weekly storage report** (UTC {date})\n- totals: {headline}\n{report_line}\n{change_section}"


def _submit_callable(
    client: IrisClient,
    *,
    name: str,
    fn,
    args: tuple,
    cpu: float,
    memory: str,
    disk: str,
    constraints: list[Constraint] | None = None,
) -> None:
    """Submit a Python callable as an Iris job and stream logs until completion."""
    job = client.submit(
        entrypoint=Entrypoint.from_callable(fn, *args),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        environment=EnvironmentSpec(env_vars={}),
        constraints=constraints,
    )
    print(f"Submitted {name}: {job.job_id}", file=sys.stderr)
    job.wait(stream_logs=True, timeout=float("inf"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--cluster", default="marin", show_default=True, help="Iris cluster name to submit jobs to.")
@click.option(
    "--staging-dir",
    default=DEFAULT_STAGING_DIR,
    show_default=True,
    help="GCS path used for parquet segments + report.md.",
)
@click.option("--workers", default=128, show_default=True, type=int, help="Number of Iris worker replicas for the scan.")
@click.option(
    "--dedup-shards", default=64, show_default=True, type=int, help="Number of output shards for the dedup stage."
)
@click.option("--skip-scan", is_flag=True, help="Reuse parquet segments already at --staging-dir.")
@click.option("--skip-dedup", is_flag=True, help="Reuse deduped parquets already at --staging-dir/deduped.")
@click.option(
    "--skip-report", is_flag=True, help="Reuse an existing report.md at --staging-dir (skip Iris aggregation)."
)
@click.option(
    "--history-dir",
    default=DEFAULT_HISTORY_DIR,
    show_default=True,
    help="Stable GCS dir of dated snapshots for the week-over-week diff.",
)
@click.option(
    "--gist",
    "gist_visibility",
    type=click.Choice(["public", "secret", "none"]),
    default="public",
    show_default=True,
    help="Stage 4 publish: 'public'/'secret' gist via gh, or 'none' to skip "
    "(e.g. when an outer digest owns publishing).",
)
@click.option(
    "--run-id",
    default=None,
    help="Suffix for Iris job names so re-runs don't collide with prior jobs "
    "of the same name (defaults to today's UTC date).",
)
@click.option(
    "--discord",
    "discord_channel",
    default=None,
    help="Discord channel to post a summary to (e.g. 'internal-discuss'). Omit to skip.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Compose and print what would be published; skip gist creation and the Discord post.",
)
def main(
    cluster: str,
    staging_dir: str,
    workers: int,
    dedup_shards: int,
    skip_scan: bool,
    skip_dedup: bool,
    skip_report: bool,
    history_dir: str,
    gist_visibility: str,
    run_id: str | None,
    discord_channel: str | None,
    dry_run: bool,
) -> None:
    staging_dir = staging_dir.rstrip("/")
    history_dir = history_dir.rstrip("/")
    deduped_dir = f"{staging_dir}/deduped"
    report_path = f"{staging_dir}/report.md"
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    run_id = run_id or today

    with open_iris_client(cluster_name=cluster, workspace=REPO_ROOT) as client:
        if not skip_scan:
            print("=== Stage 1: distributed scan on Iris ===", file=sys.stderr)
            # Pin the scan coordinator to non-preemptible (on-demand) capacity so a
            # spot reclaim can't kill it mid-scan and reset the staging dir. The
            # coordinator only buffers ~2M objects (~1-1.5 GiB resident), so it fits
            # the on-demand n2-highmem-2 pool (2 vCPU / 16 GiB) with room to spare.
            # It is too large (cpu>1 or mem>4 GiB) to trip Iris's auto-non-preemptible
            # executor heuristic, so we request the constraint explicitly. Workers stay
            # preemptible: they are stateless and re-pull tasks, and the on-demand pool
            # is far too small to host all of them.
            _submit_callable(
                client,
                name=f"storage-scan-{run_id}",
                fn=_scan_stage,
                args=(staging_dir, workers),
                cpu=1,
                memory="12GB",
                disk="30GB",
                constraints=[preemptible_constraint(False)],
            )

        if not skip_dedup:
            print("=== Stage 2: Zephyr dedup on Iris ===", file=sys.stderr)
            _submit_callable(
                client,
                name=f"storage-dedup-{run_id}",
                fn=_dedup_stage,
                args=(f"{staging_dir}/objects_*.parquet", deduped_dir, dedup_shards, 2, "8g"),
                cpu=1,
                memory="4GB",
                disk="30GB",
            )

        if not skip_report:
            print("=== Stage 3: report aggregation on Iris ===", file=sys.stderr)
            _submit_callable(
                client,
                name=f"storage-report-{run_id}",
                fn=_report_stage,
                args=(deduped_dir, report_path, history_dir, today),
                cpu=4,
                # The week-over-week snapshot adds a full-cardinality
                # GROUP BY (bucket, dir_prefix) over ~20M dir_summary rows;
                # DuckDB's soft memory limit doesn't hold for high-cardinality
                # string aggregation, so give the (unconstrained, big-node)
                # report stage real headroom.
                memory="64GB",
                # ~10 GB deduped download + DuckDB spill headroom.
                disk="100GB",
            )

    if gist_visibility == "none" and not discord_channel:
        print(f"=== Done. Report at {report_path} (nothing published) ===", file=sys.stderr)
        print(report_path)
        return

    print(f"=== Stage 4: publish (gist={gist_visibility}, discord={discord_channel or 'off'}) ===", file=sys.stderr)
    content = _fetch_report(report_path)

    desc = f"Marin storage report — {today}"
    message = _compose_discord_message(content, date=today, gist_url="<gist URL>") if discord_channel else None

    if dry_run:
        print("=== Dry run — not creating a gist or posting to Discord ===", file=sys.stderr)
        if message is not None:
            print(message)
        return

    gist_url = None
    if gist_visibility != "none":
        gist_url = _push_gist(content, desc, "marin-storage-report.md", public=gist_visibility == "public")
        print(gist_url)

    if discord_channel:
        _post_to_discord(discord_channel, _compose_discord_message(content, date=today, gist_url=gist_url))
        print(f"Posted summary to #{discord_channel}", file=sys.stderr)


if __name__ == "__main__":
    main()
