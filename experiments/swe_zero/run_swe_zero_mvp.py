# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO MVP: Generate execution-free agentic rollouts from SWE-rebench V2.

Uses Marin's VllmEnvironment to serve Gemma 4 E2B on TPU with
VLLM_TPU_SKIP_PRECOMPILE=1 to avoid slow XLA precompilation.

Usage (on Iris/Ray cluster with native vllm-tpu):
    uv run lib/marin/src/marin/run/ray_run.py \
        --extra vllm \
        --env_vars VLLM_TPU_SKIP_PRECOMPILE 1 \
        --env_vars MARIN_VLLM_MODE native \
        --env_vars HF_TOKEN $HF_TOKEN \
        -- python experiments/swe_zero/run_swe_zero_mvp.py \
            --model google/gemma-4-E2B-it \
            --step 4 \
            --output_dir gs://marin-us-central2/experiments/swe_zero_mvp

Direct usage (with vLLM server already running):
    python experiments/swe_zero/run_swe_zero_mvp.py \
        --api_base http://localhost:8000/v1 \
        --model google/gemma-4-E2B-it \
        --step 4 \
        --output_dir /tmp/swe_zero_mvp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from experiments.swe_zero.data_loader import SWERebenchV2Loader
from experiments.swe_zero.diversity import DiversityReport, measure_diversity
from experiments.swe_zero.rollout_generator import Rollout, generate_rollout, generate_rollouts_for_pr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def save_rollouts(rollouts: list[Rollout], output_path: str) -> None:
    """Save rollouts to a JSON file (supports local paths and GCS via fsspec)."""
    data = [r.to_dict() for r in rollouts]
    if output_path.startswith("gs://"):
        import fsspec

        with fsspec.open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    logger.info("Saved %d rollouts to %s", len(rollouts), output_path)


def save_report(report: DiversityReport, output_path: str) -> None:
    """Save diversity report."""
    data = {
        "n_rollouts": report.n_rollouts,
        "n_unique": report.n_unique,
        "mean_pairwise_jaccard": report.mean_pairwise_jaccard,
        "median_pairwise_jaccard": report.median_pairwise_jaccard,
        "min_pairwise_jaccard": report.min_pairwise_jaccard,
        "max_pairwise_jaccard": report.max_pairwise_jaccard,
        "std_pairwise_jaccard": report.std_pairwise_jaccard,
    }
    if output_path.startswith("gs://"):
        import fsspec

        with fsspec.open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    logger.info("Saved diversity report to %s", output_path)


def run_step3(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 3: 1 rollout from 1 PR from 1 repo."""
    logger.info("=== STEP 3: 1 rollout x 1 PR x 1 repo ===")

    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=7)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=1, seed=7)
    pr = prs[0]

    logger.info("Selected repo: %s", repo)
    logger.info("Selected PR: %s", pr.instance_id)

    rollout = generate_rollout(client=client, model=model, pr=pr, temperature=0.7)

    logger.info("Rollout: %d steps, finished=%s, %.1fs", len(rollout.steps), rollout.finished, rollout.duration_sec)
    save_rollouts([rollout], os.path.join(output_dir, "step3", "rollouts.json"))


def run_step4(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 4: 10 rollouts from 1 PR, measure diversity."""
    logger.info("=== STEP 4: 10 rollouts x 1 PR ===")

    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=7)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=1, seed=7)
    pr = prs[0]

    logger.info("Selected repo: %s, PR: %s", repo, pr.instance_id)

    rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)

    step_dir = os.path.join(output_dir, "step4")
    save_rollouts(rollouts, os.path.join(step_dir, "rollouts.json"))

    report = measure_diversity(rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def run_step5(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 5: 10 rollouts x 10 PRs x 1 repo = 100 rollouts."""
    logger.info("=== STEP 5: 10 rollouts x 10 PRs x 1 repo ===")

    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=7)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=10, seed=7)

    logger.info("Selected repo: %s, %d PRs", repo, len(prs))

    all_rollouts: list[Rollout] = []
    step_dir = os.path.join(output_dir, "step5")
    for i, pr in enumerate(prs):
        logger.info("PR %d/%d: %s", i + 1, len(prs), pr.instance_id)
        rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)
        all_rollouts.extend(rollouts)
        save_rollouts(all_rollouts, os.path.join(step_dir, "rollouts.json"))
        if len(all_rollouts) >= 2:
            report = measure_diversity(all_rollouts)
            logger.info(
                "After %d rollouts: unique=%d, mean_jaccard=%.4f",
                len(all_rollouts),
                report.n_unique,
                report.mean_pairwise_jaccard,
            )

    report = measure_diversity(all_rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def run_step6(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 6: 10 rollouts x 10 PRs x 10 repos = 1000 rollouts."""
    logger.info("=== STEP 6: 10 rollouts x 10 PRs x 10 repos ===")

    loader = SWERebenchV2Loader(language_filter="python")
    repos = loader.sample_repos(n=10, min_prs=10, seed=7)

    logger.info("Selected %d repos: %s", len(repos), repos)

    all_rollouts: list[Rollout] = []
    step_dir = os.path.join(output_dir, "step6")
    for ri, repo in enumerate(repos):
        logger.info("Repo %d/%d: %s", ri + 1, len(repos), repo)
        prs = loader.sample_prs(repo, n=10, seed=7)
        for pi, pr in enumerate(prs):
            logger.info("  PR %d/%d: %s", pi + 1, len(prs), pr.instance_id)
            rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)
            all_rollouts.extend(rollouts)
            if len(all_rollouts) % 100 == 0:
                save_rollouts(all_rollouts, os.path.join(step_dir, "rollouts.json"))
                logger.info("  Saved %d rollouts so far", len(all_rollouts))

    save_rollouts(all_rollouts, os.path.join(step_dir, "rollouts.json"))
    report = measure_diversity(all_rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def main():
    parser = argparse.ArgumentParser(description="SWE-ZERO MVP rollout generation")
    parser.add_argument("--api_base", default=None, help="vLLM API base URL (if server already running)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Model to use")
    parser.add_argument("--model_path", default=None, help="Model path (for VllmEnvironment auto-start)")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--step", type=int, required=True, choices=[3, 4, 5, 6], help="Which step to run")
    parser.add_argument("--output_dir", default="/tmp/swe_zero_mvp", help="Output directory")
    args = parser.parse_args()

    if args.api_base:
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        logger.info("Using existing server at %s", args.api_base)
    else:
        from marin.evaluation.evaluators.evaluator import ModelConfig
        from marin.inference.vllm_server import VllmEnvironment

        os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")

        model_path = args.model_path or args.model
        model_config = ModelConfig(
            name=args.model,
            path=model_path,
            engine_kwargs={"max_model_len": 8192},
        )

        logger.info("Starting VllmEnvironment for %s", args.model)
        env = VllmEnvironment(model_config)
        env.__enter__()
        client = OpenAI(base_url=env.server_url, api_key="EMPTY")
        logger.info("vLLM server ready at %s", env.server_url)

    step_fn = {3: run_step3, 4: run_step4, 5: run_step5, 6: run_step6}
    step_fn[args.step](client, args.model, args.output_dir)


if __name__ == "__main__":
    main()
