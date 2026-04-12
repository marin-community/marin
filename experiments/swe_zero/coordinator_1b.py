# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Coordinator for the SWE-ZERO 1B-token distributed pipeline.

Runs as a lightweight CPU job on Iris and spawns N child TPU jobs (one per
shard) via the Iris Python client API. This bypasses a cluster-level issue
where top-level `iris job run --tpu` can't find available TPU chips even
though workers are Ready — child jobs submitted via the client API go
through a different scheduling path that works.

Each child runs `run_swe_zero_multilang.py --all-prs --shard-index N
--total-shards M` with auto-resume from the shard's GCS checkpoint.

Usage (submit the coordinator as a CPU job):
    iris job run --job-name swe-zero-1b \
      --cpu 2 --memory 2GB --disk 5GB --priority batch \
      --extra vllm --extra tpu \
      --env-vars VLLM_TPU_SKIP_PRECOMPILE 1 \
      --env-vars VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 \
      --env-vars VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 \
      --env-vars HF_TOKEN $HF_TOKEN \
      -- python experiments/swe_zero/coordinator_1b.py \
           --total-shards 50 --tpu v6e-4 --child-priority batch
"""

from __future__ import annotations

import argparse
import logging
import time

from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, tpu_device
from iris.rpc import job_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

PRIORITY_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def main():
    parser = argparse.ArgumentParser(description="Coordinator for SWE-ZERO 1B pipeline")
    parser.add_argument("--total-shards", type=int, default=50)
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument("--tpu", default="v6e-4")
    parser.add_argument("--child-priority", default="batch", choices=["production", "interactive", "batch"])
    parser.add_argument("--child-max-retries", type=int, default=10)
    parser.add_argument("--output-root", default="gs://marin-us-central2/experiments/swe_zero_1b")
    parser.add_argument("--poll-interval", type=int, default=120, help="Seconds between status checks")
    args = parser.parse_args()

    client = IrisClient.connect()
    logger.info(
        "Coordinator: %d shards, tpu=%s, priority=%s, output=%s",
        args.total_shards,
        args.tpu,
        args.child_priority,
        args.output_root,
    )

    # Build the child job template.
    child_resources = ResourceSpec(
        cpu=16,
        memory="24GB",
        disk="60GB",
        device=tpu_device(args.tpu),
    )
    child_priority = PRIORITY_MAP[args.child_priority]

    # Submit all shards.
    children = {}
    for shard in range(args.total_shards):
        name = f"shard-{shard:03d}"
        cmd = [
            "python",
            "experiments/swe_zero/run_swe_zero_multilang.py",
            "--local",
            "--all-prs",
            "--shard-index",
            str(shard),
            "--total-shards",
            str(args.total_shards),
            "--n-rollouts",
            str(args.n_rollouts),
            "--tensor-parallel-size",
            "4",
            "--max-num-seqs",
            "256",
            "--max-model-len",
            "32768",
            "--max-total-tokens",
            "32768",
            "--concurrency",
            "64",
            "--seed",
            "7",
            "--output_dir",
            args.output_root,
        ]
        try:
            job = client.submit(
                entrypoint=Entrypoint(command=cmd),
                name=name,
                resources=child_resources,
                max_retries_failure=args.child_max_retries,
                max_retries_preemption=args.child_max_retries,
                priority_band=child_priority,
            )
            children[shard] = job
            logger.info("[shard %03d] submitted: %s", shard, job.job_id)
        except Exception as e:
            logger.error("[shard %03d] submit failed: %s", shard, e)

    logger.info("=== Submitted %d / %d shards ===", len(children), args.total_shards)

    # Poll until all children are done.
    while True:
        time.sleep(args.poll_interval)
        states = {"running": 0, "succeeded": 0, "failed": 0, "pending": 0, "other": 0}
        for _shard, job in children.items():
            try:
                status = job.status()
                state_name = status.state.name.lower() if hasattr(status.state, "name") else str(status.state)
                if "succeed" in state_name or "complet" in state_name:
                    states["succeeded"] += 1
                elif "fail" in state_name or "kill" in state_name:
                    states["failed"] += 1
                elif "run" in state_name or "assign" in state_name or "build" in state_name:
                    states["running"] += 1
                elif "pend" in state_name:
                    states["pending"] += 1
                else:
                    states["other"] += 1
            except Exception:
                states["other"] += 1

        logger.info(
            "Progress: succeeded=%d running=%d pending=%d failed=%d other=%d / %d total",
            states["succeeded"],
            states["running"],
            states["pending"],
            states["failed"],
            states["other"],
            len(children),
        )

        if states["succeeded"] + states["failed"] >= len(children):
            logger.info("=== All shards finished. succeeded=%d failed=%d ===", states["succeeded"], states["failed"])
            break

    logger.info("Coordinator done.")


if __name__ == "__main__":
    main()
