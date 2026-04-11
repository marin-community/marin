# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Throughput comparison: SWE-ZERO rollouts at different context-length caps.

Boots vLLM once on the same hardware, then runs N rollouts at each
``max_total_tokens`` value back-to-back. Reports rollouts/s, completion
tokens/s, mean turns, and submission rate per cell so we can see how the
client-side context cap affects data-synthesis throughput.

vLLM is started with ``--max-model-len`` set to the largest cap so the same
server can handle every cell.

Usage on a TPU worker (e.g. v6e-4 with TP=4):
    python experiments/swe_zero/benchmark_context_length.py \
        --model ricdomolm/mini-coder-1.7b \
        --tensor-parallel-size 4 \
        --max-num-seqs 256 \
        --concurrency 64 \
        --n-rollouts 10 \
        --context-lengths 8192,16384 \
        --output_dir gs://marin-us-central2/experiments/swe_zero_mvp/bench/ctx_length
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
    parser = argparse.ArgumentParser(description="Context-length throughput comparison for SWE-ZERO")
    parser.add_argument("--model", default="ricdomolm/mini-coder-1.7b")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument(
        "--context-lengths",
        default="8192,16384",
        help="Comma-separated client-side max_total_tokens values to compare. "
        "vLLM is started with the largest value as --max-model-len.",
    )
    parser.add_argument("--output_dir", default="/tmp/swe_zero_ctx_length")
    args = parser.parse_args()

    context_lengths = [int(s) for s in args.context_lengths.split(",")]
    vllm_max_model_len = max(context_lengths)
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
        engine_kwargs={"max_model_len": vllm_max_model_len},
    )
    extra_args = [
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--enforce-eager",
    ]
    logger.info(
        "vLLM config: max_model_len=%d, max_num_seqs=%d, TP=%d, client_concurrency=%d, n_rollouts=%d",
        vllm_max_model_len,
        args.max_num_seqs,
        args.tensor_parallel_size,
        args.concurrency,
        args.n_rollouts,
    )
    logger.info("Context lengths to compare (max_total_tokens): %s", context_lengths)

    # Pin to the same fixed PR the throughput benchmark uses, for continuity.
    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=7)
    pr = loader.sample_prs(repos[0], n=1, seed=7)[0]
    logger.info("Benchmark PR: %s (repo=%s)", pr.instance_id, pr.repo)

    results: list[dict] = []
    with remove_tpu_lockfile_on_exit(), VllmEnvironment(model_config, extra_args=extra_args) as env:
        logger.info("vLLM server ready at %s", env.server_url)

        # Warmup with 1 rollout at the smallest context length to absorb JIT
        # compile cost — same hardware sees the longer-ctx runs second.
        warmup_ctx = min(context_lengths)
        logger.info("Warmup: 1 rollout @ max_total_tokens=%d", warmup_ctx)
        warmup_t0 = time.monotonic()
        _ = run_rollouts_concurrently(
            api_base=env.server_url,
            api_key="EMPTY",
            model=args.model,
            batches=[RolloutBatch(pr=pr, n_rollouts=1)],
            concurrency=1,
            temperature=1.0,
            max_total_tokens=warmup_ctx,
        )
        logger.info("Warmup done in %.1fs", time.monotonic() - warmup_t0)

        for ctx in context_lengths:
            logger.info("=== Cell max_total_tokens=%d (n=%d) ===", ctx, args.n_rollouts)
            t0 = time.monotonic()
            rollouts = run_rollouts_concurrently(
                api_base=env.server_url,
                api_key="EMPTY",
                model=args.model,
                batches=[RolloutBatch(pr=pr, n_rollouts=args.n_rollouts)],
                concurrency=args.concurrency,
                temperature=1.0,
                max_total_tokens=ctx,
            )
            elapsed = time.monotonic() - t0

            n = len(rollouts)
            n_finished = sum(1 for r in rollouts if r.finished)
            n_errored = sum(1 for r in rollouts if r.error)
            mean_turns = sum(sum(1 for s in r.steps if s.role == "assistant") for r in rollouts) / n
            mean_steps = sum(len(r.steps) for r in rollouts) / n
            mean_prompt_tok = sum(r.total_prompt_tokens for r in rollouts) / n
            mean_completion_tok = sum(r.total_completion_tokens for r in rollouts) / n
            total_completion_tok = sum(r.total_completion_tokens for r in rollouts)
            total_prompt_tok = sum(r.total_prompt_tokens for r in rollouts)

            row = {
                "max_total_tokens": ctx,
                "n_rollouts": n,
                "n_finished": n_finished,
                "n_errored": n_errored,
                "wall_time_sec": round(elapsed, 2),
                "rollouts_per_sec": round(n / elapsed, 4),
                "completion_tokens_per_sec": round(total_completion_tok / elapsed, 1),
                "prompt_tokens_per_sec": round(total_prompt_tok / elapsed, 1),
                "mean_turns_per_rollout": round(mean_turns, 1),
                "mean_steps_per_rollout": round(mean_steps, 1),
                "mean_prompt_tokens_per_rollout": round(mean_prompt_tok, 0),
                "mean_completion_tokens_per_rollout": round(mean_completion_tok, 0),
            }
            results.append(row)
            logger.info(
                "ctx=%-5d  wall=%.1fs  rps=%.4f  tok/s=%.0f  finished=%d/%d  mean_turns=%.1f",
                ctx,
                elapsed,
                row["rollouts_per_sec"],
                row["completion_tokens_per_sec"],
                n_finished,
                n,
                mean_turns,
            )

    output_path = os.path.join(args.output_dir, "ctx_length_sweep.json")
    _save_json(results, output_path)
    logger.info("Saved results to %s", output_path)

    logger.info("=== Context-length comparison ===")
    for r in results:
        logger.info(
            "  ctx=%-5d  rps=%.4f  tok/s=%.0f  mean_turns=%.1f  finished=%d/%d  wall=%.1fs",
            r["max_total_tokens"],
            r["rollouts_per_sec"],
            r["completion_tokens_per_sec"],
            r["mean_turns_per_rollout"],
            r["n_finished"],
            r["n_rollouts"],
            r["wall_time_sec"],
        )


if __name__ == "__main__":
    main()
