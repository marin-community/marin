#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["click", "fsspec", "gcsfs"]
# ///
"""End-to-end storage report: scan all marin-* buckets, aggregate, push public gist.

Four stages, all triggered from one command on the operator's laptop:

  1. `distributed_scan.py` runs on Iris (coordinator + N worker replicas)
     and writes consolidated parquet segments to STAGING_DIR on GCS.
  2. `dedup_objects.py` runs a Zephyr group_by on Iris to collapse the raw
     parquets into one row per (bucket, name) under STAGING_DIR/deduped.
  3. `report.py` runs on Iris over the deduped parquets, builds a DuckDB
     rollup, and writes `report.md` back into STAGING_DIR.
  4. The operator's local `gh` auth uploads `report.md` as a public gist
     and prints the URL.

Prereqs (local):
    - `gh` authenticated as the user who should own the gist
    - `uv` on PATH so `iris` resolves via `uv run iris`
    - GCS read access (default ADC) to read `report.md` back

Usage:
    ./scripts/ops/storage/storage_report.py
    ./scripts/ops/storage/storage_report.py --workers 64
    ./scripts/ops/storage/storage_report.py --skip-scan          # reuse existing parquets
    ./scripts/ops/storage/storage_report.py --skip-scan --skip-report  # just re-push the gist
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import click
import fsspec

DEFAULT_STAGING_DIR = "gs://marin-us-central2/tmp/storage-scan"


def _run_iris_job(*, cpu: int, memory: str, disk: str, cmd: list[str]) -> None:
    """Submit a synchronous Iris job; raise on non-zero exit."""
    iris_cmd = [
        "uv",
        "run",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--cpu",
        str(cpu),
        "--memory",
        memory,
        "--disk",
        disk,
        "--enable-extra-resources",
        "--",
        *cmd,
    ]
    print(f"$ {' '.join(iris_cmd)}", file=sys.stderr)
    subprocess.run(iris_cmd, check=True)


def _push_public_gist(content: str, description: str, filename: str) -> str:
    """Create a public gist via `gh` and return the URL."""
    result = subprocess.run(
        [
            "gh",
            "gist",
            "create",
            "--public",
            "--filename",
            filename,
            "--desc",
            description,
            "-",  # read body from stdin
        ],
        input=content,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@click.command()
@click.option(
    "--staging-dir",
    default=DEFAULT_STAGING_DIR,
    show_default=True,
    help="GCS path used for parquet segments + report.md.",
)
@click.option("--workers", default=128, show_default=True, type=int, help="Number of Iris worker replicas for the scan.")
@click.option("--skip-scan", is_flag=True, help="Reuse parquet segments already at --staging-dir.")
@click.option("--skip-dedup", is_flag=True, help="Reuse deduped parquets already at --staging-dir/deduped.")
@click.option(
    "--skip-report", is_flag=True, help="Reuse an existing report.md at --staging-dir (skip Iris aggregation)."
)
def main(staging_dir: str, workers: int, skip_scan: bool, skip_dedup: bool, skip_report: bool) -> None:
    staging_dir = staging_dir.rstrip("/")
    deduped_dir = f"{staging_dir}/deduped"
    report_path = f"{staging_dir}/report.md"

    if not skip_scan:
        print("=== Stage 1: distributed scan on Iris ===", file=sys.stderr)
        _run_iris_job(
            cpu=2,
            memory="30GB",
            disk="30GB",
            cmd=[
                "uv",
                "run",
                "python",
                "scripts/ops/storage/distributed_scan.py",
                "--staging-dir",
                staging_dir,
                "--workers",
                str(workers),
            ],
        )

    if not skip_dedup:
        print("=== Stage 2: Zephyr dedup on Iris ===", file=sys.stderr)
        _run_iris_job(
            cpu=1,
            memory="4GB",
            disk="30GB",
            cmd=[
                "uv",
                "run",
                "python",
                "scripts/ops/storage/dedup_objects.py",
                "--input-glob",
                f"{staging_dir}/objects_*.parquet",
                "--output-dir",
                deduped_dir,
            ],
        )

    if not skip_report:
        print("=== Stage 3: report aggregation on Iris ===", file=sys.stderr)
        _run_iris_job(
            cpu=4,
            memory="16GB",
            disk="30GB",
            cmd=[
                "uv",
                "run",
                "python",
                "scripts/ops/storage/report.py",
                deduped_dir,
                "-o",
                report_path,
            ],
        )

    print(f"=== Stage 4: fetch {report_path} and push gist ===", file=sys.stderr)
    with fsspec.open(report_path, "r") as f:
        content = f.read()

    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    desc = f"Marin storage report — {ts}"
    url = _push_public_gist(content, desc, "marin-storage-report.md")
    print(url)


if __name__ == "__main__":
    main()
