# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive swarm launcher for SWE-ZERO 140B synthesis — single-region (us-east5).

Forked from experiments/baseline_collection/launch_adaptive.py (Michael Ryan).
Key hardening: REQUIRED region constraint pinning workers to us-east5 only.
Each child runs run_swe_zero_swarm.py which loops claiming unclaimed 126-shard
slots via the existing ShardLease primitive (atomic GCS if_generation_match=0).

Usage::

    uv run iris --cluster marin job run \
        --memory 4GB --cpu 2 --priority production --no-wait \
        --job-name swe-zero-swarm-launcher \
        -- python experiments/swe_zero/launch_adaptive_swe_zero.py \
        --max-count 64 --initial-batch 8 --chunk-size 4
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# --- HARD INVARIANT: single-region. ------------------------------------------
# Cross-region transfer cost a $1.5K/day incident earlier; do NOT widen this.
PINNED_REGION = "us-east5"
OUTPUT_ROOT = "gs://marin-us-east5/experiments/swe_zero_100b"
SCRIPT = "experiments/swe_zero/run_swe_zero_swarm.py"
TPU_TYPE = "v6e-4"
TENSOR_PARALLEL = 4

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def _hf_token_from_env() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise RuntimeError(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) not set in launcher env. "
            "The inner worker needs it to download the rollout model from HuggingFace."
        )
    return tok


def submit_chunk(
    client: IrisClient,
    seed_start: int,
    chunk_size: int,
    child_priority_band: int,
    hf_token: str,
) -> list[str]:
    """Submit a chunk of swarm worker jobs. Returns list of submitted job ids."""
    submitted = []
    for i in range(chunk_size):
        seed = seed_start + i
        name = f"swe-zero-swarm-{TPU_TYPE}-{seed:04d}"
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command(
                    "python",
                    SCRIPT,
                    "--output-root",
                    OUTPUT_ROOT,
                    "--worker-seed",
                    str(seed),
                    "--tensor-parallel",
                    str(TENSOR_PARALLEL),
                ),
                name=name,
                resources=ResourceSpec(
                    cpu=32,
                    memory="128GB",
                    disk="50GB",
                    device=tpu_device(TPU_TYPE),
                ),
                environment=EnvironmentSpec(
                    extras=["vllm", "tpu"],
                    # Mirrors monitor_140b_pipeline.py — these env vars are
                    # required for vLLM TPU rollouts at max_model_len=32768.
                    env_vars={
                        "VLLM_TPU_SKIP_PRECOMPILE": "1",
                        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
                        "HF_TOKEN": hf_token,
                    },
                ),
                # HARD region constraint — REQUIRED, single value. If you change
                # this you risk cross-region GCS reads/writes which are extremely
                # expensive on this project. Don't.
                constraints=[
                    preemptible_constraint(True),
                    Constraint(
                        key=WellKnownAttribute.REGION,
                        op=ConstraintOp.IN,
                        values=(PINNED_REGION,),
                        # mode=0 / CONSTRAINT_MODE_REQUIRED — autoscaler may not
                        # exclude this, unlike Michael's PREFERRED variant.
                        mode=0,
                    ),
                ],
                max_retries_preemption=100,
                max_retries_failure=3,
                priority_band=child_priority_band,
            )
            logger.info("  submitted %s -> %s", name, job.job_id)
            submitted.append(str(job.job_id))
        except Exception as e:
            logger.error("  failed to submit %s: %s", name, e)
    return submitted


def _count_child_states(client: IrisClient, job_ids: list[str]) -> tuple[int, int, int]:
    from iris.cluster.types import JobName

    running = pending = failed = 0
    for job_id_str in job_ids:
        try:
            status = client.status(JobName.from_wire(job_id_str))
            if status.state == job_pb2.JOB_STATE_RUNNING:
                running += 1
            elif status.state == job_pb2.JOB_STATE_PENDING:
                pending += 1
            elif status.state in (job_pb2.JOB_STATE_FAILED, job_pb2.JOB_STATE_KILLED):
                failed += 1
            else:
                pending += 1
        except Exception:
            pending += 1
    return running, pending, failed


def run_adaptive(
    client: IrisClient,
    max_count: int,
    initial_batch: int,
    chunk_size: int,
    check_interval: int,
    patience: int,
    child_priority_band: int,
    hf_token: str,
):
    logger.info(
        "swe-zero swarm launcher: tpu=%s region=%s max=%d initial=%d chunk=%d",
        TPU_TYPE,
        PINNED_REGION,
        max_count,
        initial_batch,
        chunk_size,
    )

    seed = 0
    all_ids: list[str] = []
    initial = min(initial_batch, max_count)
    logger.info("=== initial batch: %d ===", initial)
    all_ids.extend(submit_chunk(client, seed, initial, child_priority_band, hf_token))
    seed += initial

    backoff = 1
    stall = 0
    while len(all_ids) < max_count:
        time.sleep(check_interval)
        running, pending, failed = _count_child_states(client, all_ids)
        logger.info("states: %d running, %d pending, %d failed (of %d)", running, pending, failed, len(all_ids))

        if pending > 0:
            stall += 1
            if stall >= patience:
                cooldown = min(check_interval * backoff, 1800)
                backoff = min(backoff * 2, 6)
                logger.info("stalled %dx — backing off %ds", stall, cooldown)
                time.sleep(cooldown)
                stall = 0
            continue

        if running == 0:
            stall += 1
            if stall >= patience:
                cooldown = min(check_interval * backoff, 1800)
                backoff = min(backoff * 2, 6)
                logger.info("all dead — backoff %ds, then probe", cooldown)
                time.sleep(cooldown)
                stall = 0
                if len(all_ids) < max_count:
                    probe = min(2, max_count - len(all_ids))
                    all_ids.extend(submit_chunk(client, seed, probe, child_priority_band, hf_token))
                    seed += probe
            continue

        stall = 0
        backoff = 1
        next_chunk = min(chunk_size, max_count - len(all_ids))
        logger.info("=== scaling up: +%d (total=%d/%d) ===", next_chunk, len(all_ids) + next_chunk, max_count)
        all_ids.extend(submit_chunk(client, seed, next_chunk, child_priority_band, hf_token))
        seed += next_chunk

    logger.info("=== done: %d/%d submitted; staying alive to keep them ===", len(all_ids), max_count)
    while True:
        time.sleep(3600)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-count", type=int, default=64)
    parser.add_argument("--initial-batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--check-interval", type=int, default=300)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--child-priority",
        choices=list(PRIORITY_BAND_MAP),
        default="batch",
    )
    args = parser.parse_args()

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — run via `iris job run`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    hf_token = _hf_token_from_env()

    run_adaptive(
        client,
        max_count=args.max_count,
        initial_batch=args.initial_batch,
        chunk_size=args.chunk_size,
        check_interval=args.check_interval,
        patience=args.patience,
        child_priority_band=PRIORITY_BAND_MAP[args.child_priority],
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
