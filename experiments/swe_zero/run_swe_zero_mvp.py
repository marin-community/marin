# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO MVP: Generate execution-free agentic rollouts from SWE-rebench V2.

Launches as a Fray job on the Iris cluster, following the same pattern as
harbor_evaluator.py: a VllmEnvironment starts vLLM on a TPU worker, then
the rollout generation runs against it.

Usage:
    # Submit to Iris cluster (Steps 5+6 combined):
    uv run python experiments/swe_zero/run_swe_zero_mvp.py \
        --model ricdomolm/mini-coder-1.7b \
        --step 5 \
        --output_dir gs://marin-us-central2/experiments/swe_zero_mvp

    # Direct usage with an already-running vLLM server:
    uv run python experiments/swe_zero/run_swe_zero_mvp.py \
        --api_base http://localhost:8000/v1 \
        --model ricdomolm/mini-coder-1.7b \
        --step 4 \
        --output_dir /tmp/swe_zero_mvp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def save_json(data, output_path: str) -> None:
    if output_path.startswith("gs://"):
        import fsspec

        with fsspec.open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def _save_diversity(report, step_dir: str) -> None:
    payload: dict = {
        "n_rollouts": report.n_rollouts,
        "n_unique": report.n_unique,
        "mean_pairwise_jaccard": report.mean_pairwise_jaccard,
        "median_pairwise_jaccard": report.median_pairwise_jaccard,
        "min_pairwise_jaccard": report.min_pairwise_jaccard,
        "max_pairwise_jaccard": report.max_pairwise_jaccard,
        "std_pairwise_jaccard": report.std_pairwise_jaccard,
    }
    if report.blocked is not None:
        b = report.blocked
        payload["blocked"] = {
            "total_observations": b.total_observations,
            "total_blocked": b.total_blocked,
            "blocked_rate": (b.total_blocked / b.total_observations) if b.total_observations else 0.0,
            "rollouts_with_blocked": b.rollouts_with_blocked,
            "max_blocked_per_rollout": b.max_blocked_per_rollout,
            "by_reason": b.by_reason,
        }
    save_json(payload, os.path.join(step_dir, "diversity_report.json"))


def _run_steps(api_base: str, model: str, step: int, output_dir: str, concurrency: int) -> None:
    """Run rollout generation against a running vLLM server."""
    from openai import OpenAI

    from experiments.swe_zero.data_loader import SWERebenchV2Loader
    from experiments.swe_zero.diversity import measure_diversity
    from experiments.swe_zero.rollout_generator import (
        Rollout,
        RolloutBatch,
        generate_rollout,
        run_rollouts_concurrently,
    )

    loader = SWERebenchV2Loader(language_filter="python")

    if step == 3:
        # Step 3 is a single rollout — sequential is fine.
        client = OpenAI(base_url=api_base, api_key="EMPTY")
        logger.info("=== STEP 3: 1 rollout x 1 PR x 1 repo ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=1, seed=7)
        logger.info("Repo: %s, PR: %s", repos[0], prs[0].instance_id)
        rollout = generate_rollout(client=client, model=model, pr=prs[0], temperature=0.7)
        logger.info("Rollout: %d steps, finished=%s", len(rollout.steps), rollout.finished)
        save_json([rollout.to_dict()], os.path.join(output_dir, "step3", "rollouts.json"))
        return

    # Steps 4/5/6 all share the same shape: build a list of (PR, n_rollouts)
    # batches, run them all concurrently with one shared async client, save
    # incrementally via the progress callback, and measure diversity at the
    # end.
    if step == 4:
        logger.info("=== STEP 4: 10 rollouts x 1 PR ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=1, seed=7)
        batches = [RolloutBatch(pr=prs[0], n_rollouts=10)]
        step_dir = os.path.join(output_dir, "step4")
    elif step == 5:
        logger.info("=== STEP 5: 10 rollouts x 10 PRs x 1 repo ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=10, seed=7)
        logger.info("Repo: %s, %d PRs", repos[0], len(prs))
        batches = [RolloutBatch(pr=pr, n_rollouts=10) for pr in prs]
        step_dir = os.path.join(output_dir, "step5")
    elif step == 6:
        logger.info("=== STEP 6: 10 rollouts x 10 PRs x 10 repos ===")
        repos = loader.sample_repos(n=10, min_prs=10, seed=7)
        logger.info("Selected %d repos", len(repos))
        batches = []
        for repo in repos:
            for pr in loader.sample_prs(repo, n=10, seed=7):
                batches.append(RolloutBatch(pr=pr, n_rollouts=10))
        step_dir = os.path.join(output_dir, "step6")
    else:
        raise ValueError(f"Unknown step: {step}")

    total = sum(b.n_rollouts for b in batches)
    logger.info("Running %d total rollouts with concurrency=%d", total, concurrency)

    # Incremental save every 50 rollouts so we don't lose data on preemption.
    completed_rollouts: list[Rollout] = []
    save_interval = max(10, min(50, total // 20))

    def _on_rollout_done(done: int, total_n: int, rollout: Rollout) -> None:
        completed_rollouts.append(rollout)
        if done % save_interval == 0 or done == total_n:
            save_json(
                [r.to_dict() for r in completed_rollouts],
                os.path.join(step_dir, "rollouts.json"),
            )
            logger.info("Checkpoint: saved %d/%d rollouts", done, total_n)

    rollouts = run_rollouts_concurrently(
        api_base=api_base,
        api_key="EMPTY",
        model=model,
        batches=batches,
        concurrency=concurrency,
        temperature=1.0,
        progress_callback=_on_rollout_done,
    )

    # Final save (the callback already saves periodically, but ensure the
    # final state lands even if `total` doesn't divide save_interval evenly).
    save_json([r.to_dict() for r in rollouts], os.path.join(step_dir, "rollouts.json"))

    report = measure_diversity(rollouts)
    logger.info(report.summary())
    _save_diversity(report, step_dir)


def _run_with_vllm(
    model_name: str,
    model_path: str,
    step: int,
    output_dir: str,
    concurrency: int,
    max_num_seqs: int,
    max_model_len: int,
) -> None:
    """Start a VllmEnvironment and run rollout generation against it.

    ``max_num_seqs`` caps how many sequences vLLM keeps in flight at once.
    For a 1.7B model on v6e-1 with seq_len=8192, ~16-32 fits comfortably in
    HBM; bumping it past the client's concurrency setting is wasted work.
    """
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment
    from marin.utils import remove_tpu_lockfile_on_exit

    os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")

    model_config = ModelConfig(
        name=model_name,
        path=model_path,
        engine_kwargs={"max_model_len": max_model_len},
    )

    extra_args = [
        "--max-num-seqs",
        str(max_num_seqs),
        # Skip torch.compile / CUDA graphs. On TPU this avoids the slow
        # precompile step that otherwise dominates startup time, at the
        # cost of some steady-state inference throughput.
        "--enforce-eager",
    ]
    logger.info(
        "vLLM config: max_model_len=%d, max_num_seqs=%d, client_concurrency=%d, enforce_eager=True",
        max_model_len,
        max_num_seqs,
        concurrency,
    )

    with remove_tpu_lockfile_on_exit(), VllmEnvironment(model_config, extra_args=extra_args) as env:
        logger.info("vLLM server ready at %s", env.server_url)
        _run_steps(env.server_url, model_name, step, output_dir, concurrency)


def _print_ray_run_command(model_name: str, step: int, output_dir: str, tpu_type: str) -> None:
    """Print the ray_run.py command to submit this job to the Iris cluster."""
    cmd = (
        f"uv run lib/marin/src/marin/run/ray_run.py \\\n"
        f"  --extra vllm \\\n"
        f"  --env_vars VLLM_TPU_SKIP_PRECOMPILE 1 \\\n"
        f"  --env_vars VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 \\\n"
        f"  --env_vars VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 \\\n"
        f"  --env_vars MARIN_VLLM_MODE docker \\\n"
        f"  --env_vars HF_TOKEN $HF_TOKEN \\\n"
        f"  --tpu_type {tpu_type} \\\n"
        f"  -- python experiments/swe_zero/run_swe_zero_mvp.py \\\n"
        f"    --local \\\n"
        f"    --model {model_name} \\\n"
        f"    --step {step} \\\n"
        f"    --output_dir {output_dir}"
    )
    print(f"\n{'=' * 60}")
    print("Submit to Iris cluster with:")
    print(f"{'=' * 60}")
    print(cmd)
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="SWE-ZERO MVP rollout generation")
    parser.add_argument("--api_base", default=None, help="vLLM API base URL (if server already running)")
    parser.add_argument("--model", default="ricdomolm/mini-coder-1.7b", help="Model name")
    parser.add_argument("--model_path", default=None, help="Model path (defaults to --model)")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--step", type=int, required=True, choices=[3, 4, 5, 6], help="Which step to run")
    parser.add_argument("--output_dir", default="/tmp/swe_zero_mvp", help="Output directory (local or gs://)")
    parser.add_argument("--tpu_type", default="v6e-8", help="TPU type for cluster submission")
    parser.add_argument("--local", action="store_true", help="Run locally with VllmEnvironment (no Fray)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of rollouts in flight at once for steps 4/5/6 (vLLM batches them)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="vLLM max_num_seqs (server-side batch ceiling). Should be >= --concurrency.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max_model_len (per-sequence context window in tokens).",
    )
    args = parser.parse_args()

    model_path = args.model_path or args.model

    if args.api_base:
        # Direct mode: server already running
        _run_steps(args.api_base, args.model, args.step, args.output_dir, args.concurrency)
    elif args.local:
        # Local/on-worker mode: start VllmEnvironment in this process
        _run_with_vllm(
            args.model,
            model_path,
            args.step,
            args.output_dir,
            args.concurrency,
            args.max_num_seqs,
            args.max_model_len,
        )
    else:
        # Print the ray_run.py command for cluster submission
        _print_ray_run_command(args.model, args.step, args.output_dir, args.tpu_type)


if __name__ == "__main__":
    main()
