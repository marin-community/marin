#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weekly storage report digest.

Runs the storage report pipeline (scan -> dedup -> aggregate) on the marin
Iris cluster via ``build_report.py``, publishes the resulting ``report.md`` as
a *secret* gist, and posts a short summary (headline totals + biggest
week-over-week changes) to Discord linking to the gist.

Modeled on ``scripts/ops/egress_report.py``. Intended to run from CI on a
weekly schedule (see ``.github/workflows/ops-storage-report.yaml``), but
runnable locally for testing with ``--dry-run`` / ``--skip-pipeline``.

NOTE: This runs in a public GitHub Actions log. The report content (bucket
names, prefixes, sizes) is only safe inside the secret gist and the Discord
message — do not echo it at INFO/WARN.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STAGING_DIR = "gs://marin-us-central2/tmp/storage-report"
DEFAULT_HISTORY_DIR = "gs://marin-us-central2/storage-report-history"
MAX_CHANGES_IN_MESSAGE = 10

# Overview rows we lift from report.md for the Discord headline.
_OVERVIEW_LABELS = ("Total Objects", "Total Size", "Est. Monthly Cost")


def _run_pipeline(*, cluster: str, staging_dir: str, history_dir: str, workers: int) -> str:
    """Run build_report (no gist) and return the gs:// path of report.md."""
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.ops.storage.build_report",
            "--cluster",
            cluster,
            "--staging-dir",
            staging_dir,
            "--history-dir",
            history_dir,
            "--workers",
            str(workers),
            "--gist",
            "none",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return f"{staging_dir.rstrip('/')}/report.md"


def _fetch_report(report_path: str, dest: Path) -> str:
    """Download report.md with gcloud (avoids the Python TLS/cert maze)."""
    text = subprocess.run(
        ["gcloud", "storage", "cat", report_path],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    dest.write_text(text)
    return text


def _create_secret_gist(report_file: Path, description: str) -> str:
    result = subprocess.run(
        ["gh", "gist", "create", "--desc", description, str(report_file)],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().splitlines()[-1]


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


def _extract_changes(report_md: str) -> tuple[str | None, list[str]]:
    """Return (since_date, rendered change lines) from the changes section.

    ``since_date`` is None on a baseline run (no prior snapshot).
    """
    section = re.search(r"## Week-over-Week Changes\n(.*?)(?:\n## |\Z)", report_md, re.S)
    if not section:
        return None, []
    body = section.group(1)
    since_match = re.search(r"since (\d{4}-\d{2}-\d{2})", body)
    since = since_match.group(1) if since_match else None

    lines: list[str] = []
    for line in body.splitlines():
        if not line.startswith("|") or "---" in line or "Δ Size" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        bucket, prefix, delta, _now, _was, status = cells
        lines.append(f"`{bucket}/{prefix}` {delta} ({status})")
    return since, lines


def _compose_message(
    *, date: str, overview: dict[str, str], since: str | None, changes: list[str], gist_url: str
) -> str:
    headline = " · ".join(f"{v}" for v in overview.values()) if overview else "(totals unavailable)"

    if since is None:
        change_section = "_Baseline run — week-over-week diffs start next run._"
    elif not changes:
        change_section = f"_No prefix changes above threshold since {since}._"
    else:
        shown = changes[:MAX_CHANGES_IN_MESSAGE]
        change_section = f"**Biggest changes since {since}:**\n" + "\n".join(f"- {c}" for c in shown)
        if len(changes) > MAX_CHANGES_IN_MESSAGE:
            change_section += f"\n- _(+{len(changes) - MAX_CHANGES_IN_MESSAGE} more in the report)_"

    return f"**Weekly storage report** (UTC {date})\n- totals: {headline}\n- report: {gist_url}\n\n{change_section}"


@click.command(help=__doc__)
@click.option("--cluster", default="marin", show_default=True, help="Iris cluster to run the pipeline on.")
@click.option("--staging-dir", default=DEFAULT_STAGING_DIR, show_default=True)
@click.option("--history-dir", default=DEFAULT_HISTORY_DIR, show_default=True)
@click.option("--workers", default=128, show_default=True, type=int, help="Scan worker replicas.")
@click.option("--channel", default="internal-discuss", help="Discord channel name.")
@click.option(
    "--skip-pipeline",
    is_flag=True,
    help="Reuse an existing report.md at --staging-dir (skip the cluster run); for testing the publish path.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Skip gist creation and Discord post; print the message that would have been sent.",
)
def main(
    cluster: str,
    staging_dir: str,
    history_dir: str,
    workers: int,
    channel: str,
    skip_pipeline: bool,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if skip_pipeline:
        report_path = f"{staging_dir.rstrip('/')}/report.md"
        logging.info("Skipping pipeline; reusing %s", report_path)
    else:
        report_path = _run_pipeline(cluster=cluster, staging_dir=staging_dir, history_dir=history_dir, workers=workers)

    workdir = Path(tempfile.mkdtemp(prefix="storage-digest-"))
    report_file = workdir / "marin-storage-report.md"
    report_md = _fetch_report(report_path, report_file)

    overview = _extract_overview(report_md)
    since, changes = _extract_changes(report_md)

    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    description = f"Marin storage report — {date}"

    if dry_run:
        message = _compose_message(
            date=date, overview=overview, since=since, changes=changes, gist_url="<dry-run gist URL>"
        )
        logging.info("Dry run — would create secret gist %r and post to #%s.", description, channel)
        print(message)
        return

    gist_url = _create_secret_gist(report_file, description)
    message = _compose_message(date=date, overview=overview, since=since, changes=changes, gist_url=gist_url)
    _post_to_discord(channel, message)
    logging.info("Posted storage report for %s to #%s (%s).", date, channel, gist_url)


if __name__ == "__main__":
    main()
