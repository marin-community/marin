# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Throughput benchmark for the SWE-ZERO async rollout generator.

Starts vLLM once, then runs Step-4-style 10-rollout batches at several
client concurrency levels back-to-back against the same server. Reports
wall time and rollouts/second per config so we can pick the sweet spot.

Usage (on a TPU worker, after MARIN_VLLM_MODE=native):
    python experiments/swe_zero/benchmark_concurrency.py \
        --model ricdomolm/mini-coder-1.7b \
        --max-num-seqs 32 \
        --concurrencies 1,4,8,16,32 \
        --output_dir gs://marin-us-central2/experiments/swe_zero_mvp/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _save_json(data, path: str) -> None:
    if path.startswith("gs://"):
        import fsspec

        with fsspec.open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="vLLM concurrency benchmark for SWE-ZERO")
    parser.add_argument("--model", default="ricdomolm/mini-coder-1.7b")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--concurrencies",
        default="1,4,8,16,32",
        help="Comma-separated list of client concurrency levels to benchmark",
    )
    parser.add_argument("--n-rollouts-per-trial", type=int, default=10)
    parser.add_argument("--output_dir", default="/tmp/swe_zero_benchmark")
    args = parser.parse_args()

    concurrencies = [int(s) for s in args.concurrencies.split(",")]
    model_path = args.model_path or args.model

    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment
    from marin.utils import remove_tpu_lockfile_on_exit

    from experiments.swe_zero.data_loader import SWERebenchV2Loader
    from experiments.swe_zero.rollout_generator import RolloutBatch, run_rollouts_concurrently

    os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")

    model_config = ModelConfig(
        name=args.model,
        path=model_path,
        engine_kwargs={"max_model_len": args.max_model_len},
    )
    extra_args = ["--max-num-seqs", str(args.max_num_seqs)]

    # Pick a fixed PR set so each config sees identical work.
    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=7)
    pr = loader.sample_prs(repos[0], n=1, seed=7)[0]
    logger.info("Benchmark PR: %s", pr.instance_id)
    batches = [RolloutBatch(pr=pr, n_rollouts=args.n_rollouts_per_trial)]

    results: list[dict] = []
    with remove_tpu_lockfile_on_exit(), VllmEnvironment(model_config, extra_args=extra_args) as env:
        logger.info("vLLM server ready at %s", env.server_url)

        # Warmup pass: 1 rollout to absorb JIT compile cost so the first
        # measured config is not penalized.
        logger.info("Warmup: 1 rollout to absorb JIT compile cost")
        warmup_t0 = time.monotonic()
        _ = run_rollouts_concurrently(
            api_base=env.server_url,
            api_key="EMPTY",
            model=args.model,
            batches=[RolloutBatch(pr=pr, n_rollouts=1)],
            concurrency=1,
        )
        logger.info("Warmup done in %.1fs", time.monotonic() - warmup_t0)

        for c in concurrencies:
            logger.info("=== Benchmarking concurrency=%d (max_num_seqs=%d) ===", c, args.max_num_seqs)
            t0 = time.monotonic()
            rollouts = run_rollouts_concurrently(
                api_base=env.server_url,
                api_key="EMPTY",
                model=args.model,
                batches=batches,
                concurrency=c,
            )
            elapsed = time.monotonic() - t0
            n_finished = sum(1 for r in rollouts if r.finished)
            n_errored = sum(1 for r in rollouts if r.error)
            mean_steps = sum(len(r.steps) for r in rollouts) / len(rollouts)
            mean_prompt_tokens = sum(r.total_prompt_tokens for r in rollouts) / len(rollouts)
            mean_completion_tokens = sum(r.total_completion_tokens for r in rollouts) / len(rollouts)
            total_completion_tokens = sum(r.total_completion_tokens for r in rollouts)
            row = {
                "concurrency": c,
                "max_num_seqs": args.max_num_seqs,
                "n_rollouts": len(rollouts),
                "n_finished": n_finished,
                "n_errored": n_errored,
                "wall_time_s": round(elapsed, 2),
                "rollouts_per_s": round(len(rollouts) / elapsed, 3),
                "completion_tokens_per_s": round(total_completion_tokens / elapsed, 1),
                "mean_steps_per_rollout": round(mean_steps, 1),
                "mean_prompt_tokens_per_rollout": round(mean_prompt_tokens, 0),
                "mean_completion_tokens_per_rollout": round(mean_completion_tokens, 0),
            }
            results.append(row)
            logger.info(
                "concurrency=%-3d: %s rollouts in %.1fs = %.3f rollouts/s (%.0f completion tok/s)",
                c,
                len(rollouts),
                elapsed,
                row["rollouts_per_s"],
                row["completion_tokens_per_s"],
            )

    output_path = os.path.join(args.output_dir, "concurrency_sweep.json")
    _save_json(results, output_path)
    logger.info("Saved results to %s", output_path)
    logger.info("=== Benchmark complete ===")
    for r in results:
        logger.info(
            "  c=%-3d wall=%.1fs throughput=%.3f rollouts/s tok/s=%.0f finished=%d/%d",
            r["concurrency"],
            r["wall_time_s"],
            r["rollouts_per_s"],
            r["completion_tokens_per_s"],
            r["n_finished"],
            r["n_rollouts"],
        )


if __name__ == "__main__":
    main()
