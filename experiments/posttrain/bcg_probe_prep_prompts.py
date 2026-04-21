#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the BCG probe's 50 tension-corner prompts in MARIN eval format and
upload to each target region's GCS bucket.

Marin's `load_eval_prompts()` in MARIN format reads sharded JSONL-gzipped
records from a directory. We produce a single-shard file because 50 rows is
tiny. Each row has {behavior_id, system_prompt, user_message, rubric,
config_id} plus extra metadata we carry along (tension_point_idx,
pair_id, tension_name) for debugging.

Usage:
    source .env && uv run python experiments/posttrain/bcg_probe_prep_prompts.py \\
        --input experiments/posttrain/stage3_output/bcg_sample_50.jsonl \\
        --remote-path alignment/bcg_probe_50_prompts \\
        --region us-east1 --region us-east5 --region eu-west4
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
from pathlib import Path

from rigging.filesystem import url_to_fs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bcg_probe_prep_prompts")


REGION_TO_BUCKET = {
    "us-central1": "marin-us-central1",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "eu-west4": "marin-eu-west4",
    "europe-west4": "marin-eu-west4",  # alias
}


def build_records(input_path: Path) -> list[dict]:
    """Read BCG sample JSONL and convert each tension point to a MARIN prompt record."""
    out: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tp = row["tension_point"]
            # behavior_id + config_id must together yield a unique prompt_id.
            out.append(
                {
                    "behavior_id": f"bcg::{row['pair_id']}",
                    "config_id": f"tp{row['tension_point_idx']:03d}",
                    "system_prompt": "",
                    "user_message": tp["example_prompt"],
                    # rubric is handled at scoring time with paired rubrics; this
                    # field is retained only for the Marin prompt record schema.
                    "rubric": "",
                    # Extra metadata for debugging downstream.
                    "bcg_pair_id": row["pair_id"],
                    "bcg_tension_point_idx": row["tension_point_idx"],
                    "bcg_tension_name": tp.get("tension_name", ""),
                }
            )
    return out


def write_sharded_jsonl_gz(records: list[dict], local_path: Path) -> None:
    """Write a single .jsonl.gz shard to local disk."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(local_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info("wrote %s (%d records)", local_path, len(records))


def upload_to_region(local_path: Path, region: str, remote_relative_path: str) -> str:
    """Upload the shard to gs://{bucket}/{remote_relative_path}/shard_00000.jsonl.gz."""
    if region not in REGION_TO_BUCKET:
        raise ValueError(f"Unknown region: {region}")
    bucket = REGION_TO_BUCKET[region]
    remote_dir = f"gs://{bucket}/{remote_relative_path.strip('/')}"
    remote_shard = f"{remote_dir}/shard_00000.jsonl.gz"
    fs, _ = url_to_fs(remote_shard)
    fs.makedirs(remote_dir, exist_ok=True)
    with open(local_path, "rb") as src, fs.open(remote_shard, "wb") as dst:
        dst.write(src.read())
    logger.info("uploaded -> %s", remote_shard)
    return remote_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--remote-path", required=True, help="Path relative to each region bucket, e.g. alignment/bcg_probe_50_prompts")
    ap.add_argument("--region", action="append", required=True, help="Region(s) to upload to. Repeat the flag.")
    args = ap.parse_args()

    records = build_records(args.input)
    logger.info("built %d MARIN prompt records from %s", len(records), args.input)

    tmp_shard = Path("/tmp/bcg_probe_prompts/shard_00000.jsonl.gz")
    write_sharded_jsonl_gz(records, tmp_shard)

    for region in args.region:
        upload_to_region(tmp_shard, region, args.remote_path)

    logger.info("done: uploaded to %d regions", len(args.region))
    return 0


if __name__ == "__main__":
    sys.exit(main())
