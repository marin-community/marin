# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the SWE-rebench ConTree trace dataset to the HuggingFace Hub.

Reads the parquet shards produced by
``experiments/swe_rebench_trace/contree_pipeline.py`` (default
``gs://marin-us-central1/raw/swe-rebench-contree-traces``), drops the sentinel
rows (early-exit records carry an empty ``text``), projects each surviving row
to a lean plain-text schema, and uploads the result to ``--repo-id`` as a
HuggingFace dataset.

Shards are filtered and staged a batch at a time, then committed with one
``create_commit`` per batch — so peak local disk stays bounded and the repo
history doesn't accrue one commit per shard.

Token resolution order:
  1. ``--token`` CLI arg
  2. ``HF_TOKEN`` env var (set on iris workers)

Run on iris in the data's region so the GCS reads stay local::

    uv run iris --cluster marin job run --region us-central1 --memory 8GB --disk 32GB \\
        -- python scripts/datakit/upload_contree_traces_to_hf.py
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

import gcsfs
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_PATH = "gs://marin-us-central1/raw/swe-rebench-contree-traces"
DEFAULT_REPO_ID = "marin-community/swe-rebench-v2-CodeWorldModeling"
# Columns kept in the published dataset: the plain-text trace plus the
# SWE-rebench identifiers needed to trace a row back to its instance.
PUBLISHED_COLUMNS = ["instance_id", "test_id", "affected", "text"]
# Shards staged + committed together. Bounds peak disk to ~batch x shard size.
DEFAULT_BATCH_SIZE = 500

DATASET_CARD = """\
---
license: cc-by-4.0
language:
  - en
size_categories:
  - n>1T
task_categories:
  - text-generation
pretty_name: SWE-rebench ConTree Traces
tags:
  - code
  - execution-traces
  - swe-rebench
  - marin
---

# SWE-rebench ConTree Traces

Line-by-line Python execution traces for the test suites of
[`nebius/SWE-rebench-V2`](https://huggingface.co/datasets/nebius/SWE-rebench-V2),
captured by running each instance's tests under a tracer inside Nebius ConTree
sandboxes.

For each instance the pipeline traces the PR-affected tests before and after
the fix patch, plus the repository's full test suite (broad phase). Each row is
one ``(instance_id, test_id)`` trace.

## Schema

| column | type | notes |
| --- | --- | --- |
| `instance_id` | string | SWE-rebench-V2 row id |
| `test_id` | string | pytest node id |
| `affected` | bool | True = test touched by the PR patch; False = broad-phase test |
| `text` | string | plain-text annotated source + execution trace |

`affected=True` rows render as ``<test source>`` / ``# --- pre-patch trace ---``
/ ``# --- patch ---`` / ``# --- post-patch trace ---``; `affected=False` rows
render as ``<test source>`` / ``# --- trace ---``.

Sentinel rows from instances that failed before any trace was captured are
excluded.

## Provenance

- **Source**: [`nebius/SWE-rebench-V2`](https://huggingface.co/datasets/nebius/SWE-rebench-V2)
- **Generator**: `experiments/swe_rebench_trace/contree_pipeline.py` in
  [marin-community/marin](https://github.com/marin-community/marin).
"""


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("No HF token: pass --token or set HF_TOKEN.")
    return token


def _filter_shard(local_in: Path, local_out: Path) -> int:
    """Read a raw shard, drop empty-``text`` sentinels, project, rewrite.

    Returns the number of surviving rows (0 if the shard is all sentinels).
    """
    table = pq.read_table(local_in, columns=PUBLISHED_COLUMNS)
    text = table.column("text")
    keep = pc.and_(pc.is_valid(text), pc.not_equal(pc.binary_length(text), 0))
    table = table.filter(keep)
    if table.num_rows == 0:
        return 0
    pq.write_table(table, local_out, compression="zstd")
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--token", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N shards (debugging).")
    parser.add_argument("--dry-run", action="store_true", help="Filter and count but do not create/upload.")
    args = parser.parse_args()

    configure_logging()
    # Dry runs only read + filter, so they don't need an HF token.
    api = None if args.dry_run else HfApi(token=_resolve_token(args.token))

    fs = gcsfs.GCSFileSystem()
    src = args.source_path.removeprefix("gs://").rstrip("/")
    shards = sorted(fs.glob(f"{src}/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {args.source_path}")
    if args.limit is not None:
        shards = shards[: args.limit]
    logger.info("Found %d shards under %s", len(shards), args.source_path)

    if not args.dry_run:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(DATASET_CARD.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    scratch = Path(tempfile.mkdtemp(prefix="contree-hf-stage-"))
    kept_rows = 0
    kept_shards = 0
    pending: list[tuple[str, Path]] = []  # (path_in_repo, local_path)

    def flush(batch_index: int) -> None:
        if args.dry_run or not pending:
            return
        ops = [CommitOperationAdd(path_in_repo=pir, path_or_fileobj=str(lp)) for pir, lp in pending]
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Upload trace shards (batch {batch_index}, {len(ops)} files)",
        )
        for _, lp in pending:
            lp.unlink(missing_ok=True)
        pending.clear()

    try:
        batch_index = 0
        for i, remote in enumerate(shards):
            name = Path(remote).name
            raw_local = scratch / f"raw-{name}"
            with fs.open(remote, "rb", block_size=8 * 1024 * 1024) as src_f, raw_local.open("wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f, length=8 * 1024 * 1024)

            out_local = scratch / name
            n = _filter_shard(raw_local, out_local)
            raw_local.unlink(missing_ok=True)
            if n == 0:
                continue  # all-sentinel shard — nothing useful to publish

            kept_rows += n
            kept_shards += 1
            pending.append((f"data/{name}", out_local))

            if len(pending) >= args.batch_size:
                logger.info("committing batch %d (%d shards, %d useful rows so far)", batch_index, len(pending), kept_rows)
                flush(batch_index)
                batch_index += 1

        flush(batch_index)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    logger.info(
        "%s: %d useful rows across %d shards (%d shards were all-sentinel)",
        "would upload" if args.dry_run else "uploaded",
        kept_rows,
        kept_shards,
        len(shards) - kept_shards,
    )


if __name__ == "__main__":
    main()
