# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Launcher for the distributed SWE-ZERO 1B-token pipeline (marin-community/marin#4666).

Fans out N parallel shards of `run_swe_zero_multilang.py --all-prs --shard-index ...`
across the Iris cluster at ``--priority batch`` so it doesn't displace interactive
work. Each shard:

* Materializes a deterministic round-robin slice of all 32k+ SWE-rebench V2 PRs
  (each shard sees ~equal counts and a roughly proportional language mix).
* Runs ``--n-rollouts`` rollouts per PR with the standard 32K-context recipe.
* Auto-resumes from its own output_dir/rollouts.json if a previous attempt left
  a partial save (idempotent under preemption).

Usage:
    uv run python experiments/swe_zero/launch_1b_pipeline.py \\
        --total-shards 50 --n-rollouts 3 \\
        --output_root gs://marin-us-central2/experiments/swe_zero_1b
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


IRIS_CONFIG = "/home/kevin/marin-iris-tpu-cli/lib/iris/examples/marin.yaml"
IRIS_BIN = "/home/kevin/marin-iris-tpu-cli/.venv/bin/iris"


def _build_iris_cmd(
    *,
    shard_index: int,
    total_shards: int,
    n_rollouts: int,
    output_root: str,
    job_name_prefix: str,
    tpu: str,
    priority: str,
    max_retries: int,
    cpu: int,
    memory: str,
    disk: str,
    extras: list[str],
    env_vars: list[tuple[str, str]],
    regions: list[str],
) -> list[str]:
    output_dir = f"{output_root.rstrip('/')}/shard_{shard_index:03d}_of_{total_shards:03d}"
    cmd: list[str] = [
        IRIS_BIN,
        "--config",
        IRIS_CONFIG,
        "job",
        "run",
        "--job-name",
        f"{job_name_prefix}-s{shard_index:03d}",
        "--tpu",
        tpu,
        "--enable-extra-resources",
        "--priority",
        priority,
        "--cpu",
        str(cpu),
        "--memory",
        memory,
        "--disk",
        disk,
        "--max-retries",
        str(max_retries),
        "--no-wait",
    ]
    for region in regions:
        cmd.extend(["--region", region])
    for e in extras:
        cmd.extend(["--extra", e])
    for k, v in env_vars:
        cmd.extend(["--env-vars", k, v])
    cmd.append("--")
    cmd.extend(
        [
            "python",
            "experiments/swe_zero/run_swe_zero_multilang.py",
            "--local",
            "--model",
            "ricdomolm/mini-coder-1.7b",
            "--all-prs",
            "--shard-index",
            str(shard_index),
            "--total-shards",
            str(total_shards),
            "--n-rollouts",
            str(n_rollouts),
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
            output_dir,
        ]
    )
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch SWE-ZERO 1B-token sharded pipeline on Iris")
    parser.add_argument("--total-shards", type=int, default=50)
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument(
        "--output_root",
        default="gs://marin-us-central2/experiments/swe_zero_1b",
        help="Each shard writes to <output_root>/shard_NNN_of_MMM/",
    )
    parser.add_argument("--job-name-prefix", default="swe-zero-1b")
    parser.add_argument(
        "--tpu",
        default="v6e-4",
        help="TPU type. v6e-4 has the most capacity right now (100+ ready in europe-west4-a).",
    )
    parser.add_argument(
        "--priority",
        default="batch",
        choices=["production", "interactive", "batch"],
        help="Iris priority band. Use 'batch' to yield to interactive jobs.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", default="24GB")
    parser.add_argument("--disk", default="60GB")
    parser.add_argument(
        "--shard-range",
        default=None,
        help="Optional 'START:END' to launch only a sub-range of shards (useful for retries).",
    )
    parser.add_argument(
        "--regions",
        default="us-east5,us-east1",
        help="Comma-separated Iris region constraint. Default avoids europe-west4 "
        "where some workers have a broken vllm-tpu venv (CUDA torch instead of CPU).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    if args.shard_range:
        start, end = (int(x) for x in args.shard_range.split(":"))
    else:
        start, end = 0, args.total_shards

    extras = ["vllm", "tpu"]  # tpu pin needed until marin-community/marin#4663 lands
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN env var is required (the worker needs it to download the model)")
        sys.exit(1)
    env_vars = [
        ("VLLM_TPU_SKIP_PRECOMPILE", "1"),
        ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
        ("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1"),
        ("HF_TOKEN", hf_token),
    ]

    logger.info(
        "Launching shards %d..%d of %d at priority=%s, tpu=%s, max-retries=%d",
        start,
        end - 1,
        args.total_shards,
        args.priority,
        args.tpu,
        args.max_retries,
    )

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    submitted = []
    failed = []
    for shard in range(start, end):
        cmd = _build_iris_cmd(
            shard_index=shard,
            total_shards=args.total_shards,
            n_rollouts=args.n_rollouts,
            output_root=args.output_root,
            job_name_prefix=args.job_name_prefix,
            tpu=args.tpu,
            priority=args.priority,
            max_retries=args.max_retries,
            cpu=args.cpu,
            memory=args.memory,
            disk=args.disk,
            extras=extras,
            env_vars=env_vars,
            regions=regions,
        )
        if args.dry_run:
            print(" ".join(shlex.quote(c) for c in cmd))
            submitted.append(f"{args.job_name_prefix}-s{shard:03d}")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            tail = (r.stdout + r.stderr).strip().splitlines()
            job_line = next((ln for ln in tail if ln.startswith("/kevin/")), None)
            if r.returncode == 0 and job_line:
                submitted.append(job_line.strip())
                logger.info("[shard %03d] submitted: %s", shard, job_line.strip())
            else:
                failed.append((shard, r.returncode, "\n".join(tail[-5:])))
                logger.warning("[shard %03d] FAILED rc=%d: %s", shard, r.returncode, tail[-3:])
        except subprocess.TimeoutExpired:
            failed.append((shard, -1, "iris submit timeout"))
            logger.warning("[shard %03d] iris submit timed out", shard)

    logger.info("=== Launch summary ===")
    logger.info("  submitted: %d", len(submitted))
    logger.info("  failed:    %d", len(failed))
    if failed:
        for shard, rc, msg in failed[:5]:
            logger.warning("    shard %03d (rc=%d): %s", shard, rc, msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
