#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-region egress digest.

Runs ``scripts/ops/cross_region.py`` over the last 24h of Iris logs as an
Iris CPU job in us-central2, next to the finelog log segments, so the scan
pays no internet egress and the runner never reads the logs. The job writes
only the summary, report, and appendix to a us-central2 staging prefix; this
script fetches them, publishes the report + appendix as a secret gist, and
posts a short summary to Discord linking to the gist.

Always posts (writes "No offenders today." when nobody crosses the per-user
threshold).

NOTE: This script runs in a public GitHub Actions log. Do not log message
bodies or summary contents at INFO/WARN — the per-user counts are only safe
inside the Discord message and the gist.
"""

import datetime as dt
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from scripts.ops.cross_region import (
    APPENDIX_CSV,
    DEFAULT_LOOKBACK_HOURS,
    REPORT_MD,
    SUMMARY_JSON,
    parse_window,
    run_analysis,
    write_reports,
)

LARGE_TIER_USER_THRESHOLD = 10
MAX_OFFENDERS_IN_MESSAGE = 20

REPO_ROOT = Path(__file__).resolve().parents[2]
# The finelog log segments live in us-central2; the job runs there and stages
# its outputs there so every large read is same-region.
LOG_REGION = "us-central2"
DEFAULT_STAGING_DIR = "gs://marin-us-central2/tmp/egress-report"
JOB_TIMEOUT = Duration.from_seconds(30 * 60)
REPORT_FILES = (SUMMARY_JSON, REPORT_MD, APPENDIX_CSV)
# The scan's DuckDB threads and memory limit follow this request (cross_region.SCAN_MEMORY_FRACTION).
JOB_RESOURCES = ResourceSpec(cpu=16, memory="16GB", disk="5GB")


def _report_stage(config: str, hours: float, end_time: str, output_dir: str) -> None:
    """Iris-side entrypoint: scan the logs and upload the report files to ``output_dir``.

    ``config`` is relative to the repository root, which is the job's working directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    window = parse_window(end_time, hours)
    with tempfile.TemporaryDirectory(prefix="egress-report-") as workdir:
        local = Path(workdir)
        summary = run_analysis(config, window, DEFAULT_LOOKBACK_HOURS, None, local)
        write_reports(summary, local / SUMMARY_JSON, local / APPENDIX_CSV, local / REPORT_MD)
        for name in REPORT_FILES:
            (StoragePath(output_dir) / name).upload_from(str(local / name))


def _run_cross_region_on_iris(config: str, hours: float, end_time: str, output_dir: str, run_id: str) -> None:
    with open_iris_client(config_file=REPO_ROOT / config, workspace=REPO_ROOT) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_report_stage, config, hours, end_time, output_dir),
            name=f"egress-report-{run_id}",
            # Disk holds the ~100 MiB controller checkpoint.
            resources=JOB_RESOURCES,
            environment=EnvironmentSpec(env_vars={}),
            constraints=[region_constraint([LOG_REGION])],
            timeout=JOB_TIMEOUT,
        )
        print(f"Submitted egress report: {job.job_id}", file=sys.stderr)
        job.wait(stream_logs=True, timeout=float("inf"))


def _fetch_outputs(output_dir: str, outdir: Path) -> dict:
    """Copy the job's report files locally with gcloud and return the parsed summary."""
    for name in REPORT_FILES:
        subprocess.run(["gcloud", "storage", "cp", str(StoragePath(output_dir) / name), str(outdir / name)], check=True)
    return json.loads((outdir / SUMMARY_JSON).read_text())


def _create_gist(outdir: Path, description: str) -> str:
    """Create a secret gist with the report files; return its URL."""
    result = subprocess.run(
        [
            "gh",
            "gist",
            "create",
            "--desc",
            description,
            str(outdir / REPORT_MD),
            str(outdir / APPENDIX_CSV),
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
    "--config",
    default="lib/iris/config/marin.yaml",
    show_default=True,
    help="Iris cluster config, relative to the repository root, for both the job and the logs it scans.",
)
@click.option(
    "--staging-dir",
    default=DEFAULT_STAGING_DIR,
    show_default=True,
    help=f"{LOG_REGION} GCS prefix the Iris job writes its report files under.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Skip gist creation and Discord post; print the message that would have been sent.",
)
def main(hours: float, channel: str, config: str, staging_dir: str, dry_run: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    outdir = Path(tempfile.mkdtemp(prefix="egress-report-"))
    logging.info("Workdir: %s", outdir)

    now = dt.datetime.now(dt.UTC)
    run_id = now.strftime("%Y%m%d-%H%M%S")
    output_dir = str(StoragePath(staging_dir) / run_id)
    _run_cross_region_on_iris(config, hours, now.isoformat(), output_dir, run_id)
    summary = _fetch_outputs(output_dir, outdir)

    total = summary["cross_region_path_mentions"]
    large_users: dict[str, int] = summary.get("cross_region_large_users") or {}
    offenders = [(u, n) for u, n in large_users.items() if n >= LARGE_TIER_USER_THRESHOLD and u != "<unknown>"]

    date = now.strftime("%Y-%m-%d")
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
