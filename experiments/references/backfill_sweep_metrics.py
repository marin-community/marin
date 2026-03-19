# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill tracker_metrics.jsonl from W&B for completed sweep trials."""

import json
import logging
import re

import fsspec
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = "gs://marin-us-central1"
SWEEP_PREFIX = "ref-sweep-moe-iter03-adamh-loop"
WANDB_ENTITY_PROJECT = "held/marin"


def main():
    api = wandb.Api()
    fs, _, _ = fsspec.get_fs_token_paths(f"{BUCKET}/checkpoints/")
    paths = fs.ls(f"{BUCKET}/checkpoints/")
    sweep_paths = sorted(p for p in paths if SWEEP_PREFIX in p)
    logger.info(f"Found {len(sweep_paths)} sweep trial paths")

    backfilled = 0
    skipped = 0
    failed = 0

    for gcs_path in sweep_paths:
        if not gcs_path.startswith("gs://"):
            gcs_path = f"gs://{gcs_path}"
        dirname = gcs_path.rstrip("/").split("/")[-1]
        gcs_run_id = re.sub(r"-[a-f0-9]{6}$", "", dirname)
        metrics_file = f"{gcs_path}/tracker_metrics.jsonl"

        # GCS uses suggestion_index (0-indexed), W&B uses Vizier trial_id (1-indexed).
        # Map trialN -> trial{N+1} for W&B lookup.
        wandb_run_id = re.sub(r"trial(\d+)$", lambda m: f"trial{int(m.group(1)) + 1}", gcs_run_id)

        try:
            run = api.run(f"{WANDB_ENTITY_PROJECT}/{wandb_run_id}")
            record = {
                "config": dict(run.config),
                "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
            }
            fs.makedirs(gcs_path, exist_ok=True)
            with fs.open(metrics_file, "w") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            logger.info(f"OK {gcs_run_id} <- W&B {wandb_run_id}: backfilled ({run.state})")
            backfilled += 1
        except Exception as e:
            logger.error(f"FAIL {gcs_run_id} (tried W&B {wandb_run_id}): {e}")
            failed += 1

    logger.info(f"Done: {backfilled} backfilled, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
