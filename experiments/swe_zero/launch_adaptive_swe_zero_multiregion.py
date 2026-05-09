# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-region adaptive swarm launcher for SWE-ZERO synthesis.

Forked from `launch_adaptive_swe_zero.py`. Spans multiple GCP regions while
preserving the cost-safety invariant of the single-region launcher: each
spawned worker is pinned via ``Constraint(REGION, EQ, region)`` mode=REQUIRED,
and writes only to that region's ``gs://marin-{region}/`` bucket. Static
``shard_idx % N == region_idx`` partitioning eliminates cross-region
coordination too.

Region matrix is hard-coded (`REGIONS_CONFIG`) — every entry must have:
  1. an ``marin-{region}`` GCS bucket in that region
  2. an iris-supported (region, tpu) pair (verified via probe)
  3. a vLLM-compatible TP value matching the chip count

Usage::

    uv run iris --cluster marin job run \\
        --memory 4GB --cpu 2 --priority interactive --no-wait \\
        --enable-extra-resources \\
        --job-name swe-zero-multiswarm-launcher \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/swe_zero/launch_adaptive_swe_zero_multiregion.py \\
        --max-count-per-region 16 --initial-batch-per-region 4
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass

from iris.client.client import IrisClient
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# --- HARD INVARIANT: every region in this list MUST own a `marin-{region}` ---
# bucket in that region. Any region whose bucket lives elsewhere would
# silently incur cross-region writes when a worker writes a rollout. Verified
# 2026-05-05.
SCRIPT = "experiments/swe_zero/run_swe_zero_swarm_multiregion.py"


@dataclass(frozen=True)
class RegionConfig:
    region: str
    tpu: str
    tensor_parallel: int
    # When set, the worker runs in "imported mode": it claims work from
    # `claim_region`'s partition (cross-region claim/state reads), generates
    # rollouts in its own bucket under imported/, and on iter completion
    # copies the resume file back to claim_region (one-time cross-region
    # write of ~250-400 MB). Lets idle compute in fully-done regions absorb
    # work from saturated regions while keeping the source region's bucket
    # canonical for downstream aggregation.
    claim_region: str | None = None

    @property
    def label(self) -> str:
        """Unique key per (region, tpu, claim_region) entry — used to avoid
        dict collisions when the same compute region appears multiple times
        (one native, several importing different source regions)."""
        cr = self.claim_region or self.region
        if cr == self.region:
            return f"{self.region}-{self.tpu}"
        return f"{self.region}-{self.tpu}-imp-{cr}"


REGIONS_CONFIG = [
    # tensor_parallel + slice constraints (verified 2026-05-09):
    # 1. TP must equal the number of locally-addressable TPU chips. v5p-N is
    #    N/2 chips (each v5p chip = 2 TensorCores); v6e-N and v5litepod-N
    #    are N chips total. Using TP=8 on v5p-8 (4 chips) made vLLM's mesh
    #    fail with "Number of devices 4 must be >= product of mesh_shape".
    # 2. TP must divide the model's attention head count. Mini-coder-1.7b
    #    has 16 heads → TP ∈ {1,2,4,8,16}.
    # 3. **Slice must be single-host.** Multi-host TPU slices (v6e-8+, v5p-16+,
    #    v5e-8+) need a Ray-coordinated multi-process vLLM deploy; our
    #    launcher's single-process `vllm serve` blows up the first time
    #    `tpu_inference.utils.hbm_usage_bytes` calls `.memory_stats()` on a
    #    non-addressable (remote-host) device — JaxRuntimeError "MemoryStats
    #    only supported for addressable PjRt devices". Until we set up
    #    multi-host vLLM, **only single-host slices are usable.**
    #
    # Single-host slices for each family (chips/host = 4 across v6e/v5p/v5e):
    #   - v6e-4         → 4 chips, 1 host, TP=4  ✓
    #   - v5p-8         → 4 chips, 1 host, TP=4  ✓
    #   - v5litepod-4   → 4 chips, 1 host, TP=4  ✓
    # Everything bigger (v6e-16, v5p-16, v5p-32, v5litepod-16, etc.) is
    # multi-host and silently fails — dropped 2026-05-09 22:10 UTC.
    #
    RegionConfig("us-east5", "v6e-4", 4),
    RegionConfig("us-east5", "v5p-8", 4),
    RegionConfig("us-east1", "v6e-4", 4),
    RegionConfig("us-west4", "v5litepod-4", 4),
    RegionConfig("us-central1", "v5p-8", 4),
    # ─── IMPORTED-MODE absorbing entries ──────────────────────────────────
    # us-east5 absorbs us-central1 (P3): use single-host slices only.
    RegionConfig("us-east5", "v6e-4", 4, claim_region="us-central1"),
    RegionConfig("us-east5", "v5p-8", 4, claim_region="us-central1"),
    # us-west4 absorbs us-east1 (P1).
    RegionConfig("us-west4", "v5litepod-4", 4, claim_region="us-east1"),
]

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
            "Inner worker needs it to download the rollout model from HuggingFace."
        )
    return tok


def submit_chunk(
    client: IrisClient,
    region_config: RegionConfig,
    seed_start: int,
    chunk_size: int,
    child_priority_band: int,
    hf_token: str,
) -> list[str]:
    """Submit a chunk of swarm workers in one specific region."""
    submitted = []
    for i in range(chunk_size):
        seed = seed_start + i
        # Embed region (and source-region for imported entries) in name so
        # iris job list is browsable.
        if region_config.claim_region and region_config.claim_region != region_config.region:
            name = (
                f"swe-zero-multiswarm-{region_config.region}-{region_config.tpu}"
                f"-imp{region_config.claim_region}-{seed:04d}"
            )
        else:
            name = f"swe-zero-multiswarm-{region_config.region}-{region_config.tpu}-{seed:04d}"
        cmd_args = [
            SCRIPT,
            "--worker-seed",
            str(seed),
            "--tensor-parallel",
            str(region_config.tensor_parallel),
        ]
        if region_config.claim_region and region_config.claim_region != region_config.region:
            cmd_args.extend(["--claim-region", region_config.claim_region])
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", *cmd_args),
                name=name,
                resources=ResourceSpec(
                    cpu=32,
                    memory="128GB",
                    disk="50GB",
                    device=tpu_device(region_config.tpu),
                ),
                environment=EnvironmentSpec(
                    extras=["vllm", "tpu"],
                    env_vars={
                        "VLLM_TPU_SKIP_PRECOMPILE": "1",
                        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
                        "HF_TOKEN": hf_token,
                    },
                ),
                # HARD region pin — same pattern as the single-region launcher.
                # If you change this you risk cross-region GCS reads/writes
                # which are extremely expensive on this project. Don't.
                # `region_constraint` is mode=REQUIRED by default.
                constraints=[
                    preemptible_constraint(True),
                    region_constraint([region_config.region]),
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
    max_count_per_region: int,
    initial_batch_per_region: int,
    chunk_size: int,
    check_interval: int,
    patience: int,
    child_priority_band: int,
    hf_token: str,
):
    """Adaptive scaler that runs N independent ramps, one per region.

    Each region's ramp behaves like the single-region adaptive launcher: ramp
    up `chunk_size` workers per cycle when the existing fleet is making
    progress, back off on stalls, until reaching `max_count_per_region`. The
    region ramps don't coordinate with each other — capacity in one region
    doesn't affect another.
    """
    logger.info(
        "MULTI-REGION SWE-ZERO LAUNCHER:\n" "  regions: %s\n" "  max_count_per_region=%d initial_batch=%d chunk_size=%d",
        ", ".join(f"{rc.region}({rc.tpu}, TP={rc.tensor_parallel})" for rc in REGIONS_CONFIG),
        max_count_per_region,
        initial_batch_per_region,
        chunk_size,
    )

    # Per-(region, tpu) state — keyed by `rc.label` so two RegionConfigs with
    # the same region (e.g. us-east5 v6e-4 + v5p-8) don't collide in the dicts.
    seeds = {rc.label: rc_idx * 10000 for rc_idx, rc in enumerate(REGIONS_CONFIG)}
    job_ids: dict[str, list[str]] = {rc.label: [] for rc in REGIONS_CONFIG}
    backoffs: dict[str, int] = {rc.label: 1 for rc in REGIONS_CONFIG}
    stalls: dict[str, int] = {rc.label: 0 for rc in REGIONS_CONFIG}

    # Initial seed batch per RegionConfig (parallel).
    initial = min(initial_batch_per_region, max_count_per_region)
    for rc in REGIONS_CONFIG:
        logger.info("=== initial batch in %s: %d ===", rc.label, initial)
        ids = submit_chunk(client, rc, seeds[rc.label], initial, child_priority_band, hf_token)
        job_ids[rc.label].extend(ids)
        seeds[rc.label] += initial

    # Adaptive loop: every `check_interval`, scan each RegionConfig independently
    # and decide whether to scale, back off, or probe.
    while any(len(job_ids[rc.label]) < max_count_per_region for rc in REGIONS_CONFIG):
        time.sleep(check_interval)
        for rc in REGIONS_CONFIG:
            key = rc.label
            if len(job_ids[key]) >= max_count_per_region:
                continue
            running, pending, failed = _count_child_states(client, job_ids[key])
            logger.info(
                "[%s] %d running, %d pending, %d failed (of %d)",
                key,
                running,
                pending,
                failed,
                len(job_ids[key]),
            )

            if pending > 0:
                stalls[key] += 1
                if stalls[key] >= patience:
                    cooldown = min(check_interval * backoffs[key], 1800)
                    backoffs[key] = min(backoffs[key] * 2, 6)
                    logger.info("[%s] stalled — backoff %ds", key, cooldown)
                    stalls[key] = 0
                continue

            if running == 0:
                stalls[key] += 1
                if stalls[key] >= patience:
                    cooldown = min(check_interval * backoffs[key], 1800)
                    backoffs[key] = min(backoffs[key] * 2, 6)
                    logger.info("[%s] all dead — backoff %ds, then probe", key, cooldown)
                    stalls[key] = 0
                    if len(job_ids[key]) < max_count_per_region:
                        probe = min(2, max_count_per_region - len(job_ids[key]))
                        ids = submit_chunk(client, rc, seeds[key], probe, child_priority_band, hf_token)
                        job_ids[key].extend(ids)
                        seeds[key] += probe
                continue

            stalls[key] = 0
            backoffs[key] = 1
            next_chunk = min(chunk_size, max_count_per_region - len(job_ids[key]))
            logger.info(
                "[%s] === scaling up: +%d (config total=%d/%d) ===",
                key,
                next_chunk,
                len(job_ids[key]) + next_chunk,
                max_count_per_region,
            )
            ids = submit_chunk(client, rc, seeds[key], next_chunk, child_priority_band, hf_token)
            job_ids[key].extend(ids)
            seeds[key] += next_chunk

    total = sum(len(v) for v in job_ids.values())
    target = max_count_per_region * len(REGIONS_CONFIG)
    logger.info("=== done: %d/%d submitted across %d regions ===", total, target, len(REGIONS_CONFIG))
    while True:
        time.sleep(3600)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-count-per-region", type=int, default=16)
    parser.add_argument("--initial-batch-per-region", type=int, default=4)
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
        max_count_per_region=args.max_count_per_region,
        initial_batch_per_region=args.initial_batch_per_region,
        chunk_size=args.chunk_size,
        check_interval=args.check_interval,
        patience=args.patience,
        child_priority_band=PRIORITY_BAND_MAP[args.child_priority],
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
