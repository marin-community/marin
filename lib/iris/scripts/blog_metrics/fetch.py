# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline step 1: pull raw finelog data to local disk.

Two sources, both cached under ``<data_dir>/raw`` so re-running ``extract`` /
``charts`` never re-fetches:

* ``iris.worker`` + ``iris.task`` parquet — mirrored wholesale with ``gsutil
  rsync`` (a few GB each). These structured namespaces carry everything the
  core charts need (device variant -> chips/FLOPS, running task counts, task
  ids -> users).
* ``/system/controller`` audit lines — *optional*, off by default. The ``log``
  namespace is ~33 GB and clustered by ``key``; filtering to the controller key
  still streams the big ``data`` column, so this is the expensive path. Only
  needed for submission/demand series and milestone cross-checks.

This module shells out to ``gsutil`` and the ``finelog`` CLI rather than
reimplementing GCS/DuckDB plumbing.
"""

from __future__ import annotations

import logging
import os
import subprocess

import config
import wandb_history

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def mirror_stats_namespaces(paths: config.Paths, *, force: bool = False) -> None:
    """Mirror the iris.worker / iris.task parquet archives to ``raw/``.

    Uses ``gsutil -m rsync`` so a re-run only transfers changed/new segments.
    ``force`` adds ``-d`` to delete local segments no longer present remotely.
    """
    for namespace, dest in (
        (config.WORKER_NAMESPACE, paths.worker_parquet),
        (config.TASK_NAMESPACE, paths.task_parquet),
    ):
        os.makedirs(dest, exist_ok=True)
        src = f"{config.REMOTE_LOG_DIR}/{namespace}"
        cmd = ["gsutil", "-m", "rsync"]
        if force:
            cmd.append("-d")
        cmd += [src, dest]
        logger.info("mirroring %s -> %s", src, dest)
        _run(cmd)


def extract_controller_audit(paths: config.Paths) -> None:
    """Extract ``/system/controller`` audit lines to a local parquet file.

    Runs a key-pruned DuckDB query against the GCS-archived ``log`` namespace
    via ``finelog gcs-query`` and keeps only the structured ``event=`` audit
    lines (job submissions, worker lifecycle, scheduling passes). The result is
    tiny; the cost is the remote scan of the ``data`` column.
    """
    os.makedirs(paths.raw_dir, exist_ok=True)
    out = paths.controller_audit_parquet
    # COPY ... TO writes parquet directly from the DuckDB result. Keep only
    # audit lines (``event=``) to drop the high-volume provider/slice chatter.
    sql = (
        f"COPY (SELECT epoch_ms, data FROM log "
        f"WHERE key = '{config.CONTROLLER_LOG_KEY}' AND data LIKE '%event=%') "
        f"TO '{out}' (FORMAT parquet)"
    )
    cmd = [
        "uv",
        "run",
        "finelog",
        "gcs-query",
        config.FINELOG_DEPLOYMENT,
        sql,
        "--namespace",
        "log",
    ]
    logger.info("extracting controller audit lines -> %s (slow remote scan)", out)
    # finelog lives in its own workspace package; run from there.
    finelog_dir = os.path.abspath(os.path.join(config.PACKAGE_DIR, "..", "..", "..", "finelog"))
    logger.info("$ (cwd=%s) %s", finelog_dir, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=finelog_dir)


def fetch_wandb(paths: config.Paths, *, force: bool = False) -> None:
    """Pull W&B run history to the local cache, unless it already exists.

    The full pull walks tens of thousands of runs (minutes), so it is skipped
    when the cache is present; pass ``force`` to refresh.
    """
    if os.path.exists(paths.wandb_runs_parquet) and not force:
        logger.info("W&B cache present (%s); skipping pull (use --force to refresh)", paths.wandb_runs_parquet)
        return
    wandb_history.fetch_runs(paths)


def run(
    paths: config.Paths,
    *,
    with_controller: bool = False,
    with_wandb: bool = True,
    force: bool = False,
) -> None:
    """Fetch all raw inputs for the pipeline."""
    mirror_stats_namespaces(paths, force=force)
    if with_wandb:
        fetch_wandb(paths, force=force)
    if with_controller:
        extract_controller_audit(paths)
    else:
        logger.info("skipping controller audit extraction (pass --with-controller to enable)")
