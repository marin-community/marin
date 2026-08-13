# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rescore already-scored shards and compare to the stored values.

Settles whether today's checkout still produces the scores that are already in the
dataset. The scorer reads its model from object storage but its *code* ships with the
job bundle, so an edit to the model module changes what a rerun computes even though the
checkpoint is untouched.

Runs the production scorer -- ``score_corpus.py score``, as a subprocess, the same way
``node`` mode fans out -- against a manifest whose ``output_path`` has been redirected to
a scratch prefix, then joins the fresh scores to the stored ones on ``id``. Nothing is
written to the real output prefix, so a divergence costs a scratch file rather than a
corrupted shard.

Bit equality is not the bar: the same graph on a different device schedules its
reductions differently. The tolerance is a measured GPU-vs-CPU parity figure.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --gpu 4 --extra gpu --cpu 32 --memory 128g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.parity_rescore_check --shards 4
"""

import argparse
import json
import logging
import subprocess
import sys
from io import BytesIO

import fsspec
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    DEFAULT_FOLDED_DIR,
    DEFAULT_MANIFEST,
    MODULE,
    read_manifest,
)

logger = logging.getLogger(__name__)

SCRATCH = "s3://marin-us-east-02a/marin/user/muchanem/quality_scores_run/_parity_check"
# Measured GPU-vs-CPU parity for this model. Not bit equality: the same graph on a
# different device orders its reductions differently.
TOLERANCE = 5.19e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--max-embed-rows", type=int, default=200_000, help="skip shards larger than this")
    args = ap.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")

    rows = [r for r in read_manifest(args.manifest).to_dicts() if 0 < (r.get("embed_rows") or 0) <= args.max_embed_rows]
    if not rows:
        raise ValueError("no manifest rows small enough to rescore")
    rows.sort(key=lambda r: (r["source"], r["shard_index"]))
    picked = rows[:: max(1, len(rows) // args.shards)][: args.shards]
    scratch = args.scratch.rstrip("/")
    redirected = [
        {**r, "output_path": f"{scratch}/shards/{r['source'].replace('/', '__')}__{r['shard_index']}.parquet"}
        for r in picked
    ]

    manifest_dir = f"{scratch}/manifest"
    table = pa.table({name: [r[name] for r in redirected] for name in redirected[0]})
    with open_url(f"{manifest_dir}/manifest.parquet", "wb") as handle:
        pq.write_table(table, handle)
    logger.info("staged %d shard tasks at %s", len(redirected), manifest_dir)

    # The production scorer, driven exactly as `node` mode drives it.
    argv = [
        sys.executable,
        "-u",
        "-m",
        MODULE,
        "score",
        "--manifest",
        manifest_dir,
        "--model-dir",
        args.model_dir,
        "--worker",
        "0",
        "--num-workers",
        "1",
    ]
    logger.info("running %s", " ".join(argv))
    subprocess.run(argv, check=True)

    deltas = []
    per_shard = []
    for row in redirected:
        fresh = pl.read_parquet(BytesIO(fs.cat(row["output_path"].removeprefix("s3://"))), columns=["id", "score"])
        stored_path = next(
            r["output_path"] for r in picked if r["shard_index"] == row["shard_index"] and r["source"] == row["source"]
        )
        stored = pl.read_parquet(BytesIO(fs.cat(stored_path.removeprefix("s3://"))), columns=["id", "score"])
        # Deduplicate before joining. A content-derived id repeats when a shard carries
        # the same text twice, and an unguarded inner join then fans out to more rows
        # than either side holds -- pairing a document against the *other* copy's score
        # and reporting a delta that is an artifact of the join, not of the scorer.
        fresh = fresh.unique(subset="id", keep="first")
        stored = stored.unique(subset="id", keep="first")
        joined = fresh.join(stored, on="id", how="inner", suffix="_stored")
        delta = np.abs(joined["score"].to_numpy() - joined["score_stored"].to_numpy())
        deltas.append(delta)
        per_shard.append(
            {
                "source": row["source"],
                "shard_index": row["shard_index"],
                "fresh_rows": fresh.height,
                "stored_rows": stored.height,
                "matched_rows": joined.height,
                "max_abs_delta": float(delta.max()) if len(delta) else None,
                "mean_abs_delta": float(delta.mean()) if len(delta) else None,
                "exact_matches": int((delta == 0).sum()),
            }
        )
        logger.info("PARITY shard %s", json.dumps(per_shard[-1]))

    combined = np.concatenate(deltas) if deltas else np.zeros(0)
    result = {
        "shards": len(per_shard),
        "rows_compared": len(combined),
        "max_abs_delta": float(combined.max()) if len(combined) else None,
        "mean_abs_delta": float(combined.mean()) if len(combined) else None,
        "exact_match_fraction": float((combined == 0).mean()) if len(combined) else None,
        "tolerance": TOLERANCE,
        "within_tolerance": bool(len(combined) and combined.max() <= TOLERANCE),
        "per_shard": per_shard,
    }
    logger.info("PARITY result %s", json.dumps(result, indent=2))
    if not result["within_tolerance"]:
        raise RuntimeError(f"rescore differs by {result['max_abs_delta']} > {TOLERANCE}")


if __name__ == "__main__":
    main()
