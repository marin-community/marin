#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daily cross-region egress digest.

Runs ``scripts/ops/cross_region.py`` over the last 24h of Iris logs, publishes
the resulting report + appendix as a secret gist, and posts a short summary
to Discord linking to the gist.

Always posts (writes "No offenders today." when nobody crosses the per-user
threshold).

NOTE: This script runs in a public GitHub Actions log. Do not log message
bodies or summary contents at INFO/WARN — the per-user counts are only safe
inside the Discord message and the gist.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import click

LARGE_TIER_USER_THRESHOLD = 10
MAX_OFFENDERS_IN_MESSAGE = 20

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cross_region(hours: float, outdir: Path) -> dict:
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(REPO_ROOT / "scripts" / "ops" / "cross_region.py"),
            "--hours",
            str(hours),
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads((outdir / "cross_region_ops_summary.json").read_text())


def _create_gist(outdir: Path, description: str) -> str:
    """Create a secret gist with the report files; return its URL."""
    result = subprocess.run(
        [
            "gh",
            "gist",
            "create",
            "--desc",
            description,
            str(outdir / "cross_region_ops_report.md"),
            str(outdir / "cross_region_ops_appendix.csv"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().splitlines()[-1]


def _post_to_discord(channel: str, message: str) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(REPO_ROOT / "scripts" / "ops" / "discord.py"),
            "-c",
            channel,
        ],
        input=message,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )


def _compose_message(
    *,
    hours: float,
    date: str,
    total: int,
    offenders: list[tuple[str, int]],
    report_url: str,
) -> str:
    if offenders:
        shown = offenders[:MAX_OFFENDERS_IN_MESSAGE]
        offender_section = "**Potential large-egress contributors:**\n" + "\n".join(
            f"- `@{u}` — {n} cross-region large-tier mentions" for u, n in shown
        )
        if len(offenders) > MAX_OFFENDERS_IN_MESSAGE:
            extra = len(offenders) - MAX_OFFENDERS_IN_MESSAGE
            offender_section += f"\n- _(+{extra} more in the report)_"
    else:
        offender_section = "_No offenders today._"

    return (
        f"**Daily cross-region egress report** ({hours:.0f}h window, UTC {date})\n"
        f"- total cross-region mentions: **{total}**\n"
        f"- report: {report_url}\n\n"
        f"{offender_section}"
    )


@click.command(help=__doc__)
@click.option("--hours", type=float, default=24.0, help="Analysis window in hours. Defaults to 24.")
@click.option("--channel", default="internal-discuss", help="Discord channel name.")
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Skip gist creation and Discord post; print the message that would have been sent.",
)
def main(hours: float, channel: str, dry_run: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    outdir = Path(tempfile.mkdtemp(prefix="egress-report-"))
    logging.info("Workdir: %s", outdir)

    summary = _run_cross_region(hours, outdir)

    total = summary["cross_region_path_mentions"]
    large_users: dict[str, int] = summary.get("cross_region_large_users") or {}
    offenders = [(u, n) for u, n in large_users.items() if n >= LARGE_TIER_USER_THRESHOLD and u != "<unknown>"]

    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    description = f"Marin cross-region egress {date} ({hours:.0f}h window)"

    if dry_run:
        report_url = "<dry-run gist URL>"
    else:
        report_url = _create_gist(outdir, description)

    message = _compose_message(
        hours=hours,
        date=date,
        total=total,
        offenders=offenders,
        report_url=report_url,
    )

    if dry_run:
        logging.info("Dry run — would create gist %r and post to #%s.", description, channel)
        print(message)
        return

    _post_to_discord(channel, message)
    logging.info("Posted egress report for %s to #%s.", date, channel)


if __name__ == "__main__":
    main()
