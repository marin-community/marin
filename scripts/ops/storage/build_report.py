#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end storage report: scan all marin-* buckets, aggregate, push public gist.

Four stages, all driven from one command on the operator's laptop. The
orchestrator opens a tunnel to the named Iris cluster and submits each stage
as an Iris job using the Python client (no subprocess into the iris CLI).

  1. Scan stage    — Iris coordinator + N worker replicas walk every GCS
                     prefix and write consolidated parquet segments to
                     STAGING_DIR (delegates to ``run_distributed`` in
                     ``distributed_scan.py``).
  2. Dedup stage   — Iris coordinator job runs a Zephyr ``group_by`` to
                     collapse the raw parquets into one row per (bucket, name)
                     under STAGING_DIR/deduped. Pipeline construction lives
                     in this file (see ``_dedup_stage``).
  3. Report stage  — Iris coordinator job reads the deduped parquets, builds
                     a DuckDB rollup, and writes ``report.md`` back into
                     STAGING_DIR.
  4. Gist          — Local: fetch ``report.md`` from STAGING_DIR and push as
                     a public gist via the operator's ``gh`` auth.

Prereqs (local):
    - ``gh`` authenticated as the user who should own the gist
    - GCS read access (default ADC) to read ``report.md`` back
    - The named cluster's controller is reachable (uses the same tunnel
      machinery as ``iris --cluster=<name> ...``)

Usage:
    ./scripts/ops/storage/build_report.py
    ./scripts/ops/storage/build_report.py --workers 64
    ./scripts/ops/storage/build_report.py --skip-scan          # reuse existing parquets
    ./scripts/ops/storage/build_report.py --skip-scan --skip-report  # just re-push the gist
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import fsspec
from fray import ResourceConfig
from iris.cli.main import IRIS_CLUSTER_CONFIG_DIRS, create_client_token_provider, resolve_cluster_name
from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.config_discovery import resolve_cluster_config
from zephyr import Dataset, ZephyrContext

from scripts.ops.storage.constants import MARIN_BUCKETS
from scripts.ops.storage.distributed_scan import run_distributed
from scripts.ops.storage.report import generate_report, load_parquet_db

DEFAULT_STAGING_DIR = "gs://marin-us-central2/tmp/storage-scan"
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _report_stage(deduped_dir: str, report_path: str) -> None:
    """Iris-side entrypoint for stage 3: build the markdown report."""

    conn = load_parquet_db(deduped_dir)
    report = generate_report(conn)
    with fsspec.open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local helpers (run on the operator's laptop)
# ---------------------------------------------------------------------------


def _push_public_gist(content: str, description: str, filename: str) -> str:
    """Create a public gist via ``gh`` and return the URL."""
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


def _open_iris_client(cluster: str) -> tuple[IrisClient, object]:
    """Resolve the named cluster, open a controller tunnel, and return a client.

    Returns ``(client, tunnel_ctx)``; callers must close the tunnel context
    when finished (it backs the controller URL the client talks to).
    """
    config_path = resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    iris_config = IrisConfig.load(config_path)

    token_provider = None
    cluster_name = resolve_cluster_name(iris_config.proto, None, cluster)
    if iris_config.proto.HasField("auth"):
        token_provider = create_client_token_provider(iris_config.proto.auth, cluster_name=cluster_name)

    bundle = iris_config.provider_bundle()
    controller_address = iris_config.controller_address() or bundle.controller.discover_controller(
        iris_config.proto.controller
    )

    tunnel_cm = bundle.controller.tunnel(address=controller_address)
    tunnel_url = tunnel_cm.__enter__()
    client = IrisClient.remote(tunnel_url, workspace=REPO_ROOT, token_provider=token_provider)
    return client, tunnel_cm


def _submit_callable(
    client: IrisClient,
    *,
    name: str,
    fn,
    args: tuple,
    cpu: float,
    memory: str,
    disk: str,
) -> None:
    """Submit a Python callable as an Iris job and stream logs until completion."""
    job = client.submit(
        entrypoint=Entrypoint.from_callable(fn, *args),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        environment=EnvironmentSpec(env_vars={}),
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
def main(
    cluster: str,
    staging_dir: str,
    workers: int,
    dedup_shards: int,
    skip_scan: bool,
    skip_dedup: bool,
    skip_report: bool,
) -> None:
    staging_dir = staging_dir.rstrip("/")
    deduped_dir = f"{staging_dir}/deduped"
    report_path = f"{staging_dir}/report.md"

    client, tunnel_cm = _open_iris_client(cluster)
    try:
        if not skip_scan:
            print("=== Stage 1: distributed scan on Iris ===", file=sys.stderr)
            _submit_callable(
                client,
                name="storage-scan",
                fn=_scan_stage,
                args=(staging_dir, workers),
                cpu=2,
                memory="30GB",
                disk="30GB",
            )

        if not skip_dedup:
            print("=== Stage 2: Zephyr dedup on Iris ===", file=sys.stderr)
            _submit_callable(
                client,
                name="storage-dedup",
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
                name="storage-report",
                fn=_report_stage,
                args=(deduped_dir, report_path),
                cpu=4,
                memory="16GB",
                disk="30GB",
            )
    finally:
        tunnel_cm.__exit__(None, None, None)

    print(f"=== Stage 4: fetch {report_path} and push gist ===", file=sys.stderr)
    with fsspec.open(report_path, "r") as f:
        content = f.read()

    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    desc = f"Marin storage report — {ts}"
    url = _push_public_gist(content, desc, "marin-storage-report.md")
    print(url)


if __name__ == "__main__":
    main()
