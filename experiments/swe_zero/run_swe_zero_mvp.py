# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO MVP: Generate execution-free agentic rollouts from SWE-rebench V2.

This script implements Steps 3-6 of the SWE-ZERO MVP plan:
  Step 3: 1 rollout x 1 PR x 1 repo
  Step 4: 10 rollouts x 1 PR (measure diversity)
  Step 5: 10 rollouts x 10 PRs x 1 repo (measure diversity)
  Step 6: 10 rollouts x 10 PRs x 10 repos (measure diversity)

Usage:
    # Step 3: Single rollout test
    python experiments/swe_zero/run_swe_zero_mvp.py \
        --api_base http://localhost:8000/v1 \
        --model google/gemma-4-E2B-it \
        --step 3 \
        --output_dir /tmp/swe_zero_mvp

    # Step 6: Full 1000-rollout run
    python experiments/swe_zero/run_swe_zero_mvp.py \
        --api_base http://localhost:8000/v1 \
        --model google/gemma-4-E2B-it \
        --step 6 \
        --output_dir gs://marin-us-central2/swe_zero_mvp
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


def analyze_rollout(rollout: Rollout) -> None:
    """Print a detailed analysis of a single rollout."""
    logger.info("=" * 60)
    logger.info("Rollout Analysis for %s", rollout.instance_id)
    logger.info("  Repo: %s", rollout.repo)
    logger.info("  Steps: %d", len(rollout.steps))
    logger.info("  Finished: %s", rollout.finished)
    logger.info("  Duration: %.1fs", rollout.duration_sec)
    logger.info("  Prompt tokens: %d", rollout.total_prompt_tokens)
    logger.info("  Completion tokens: %d", rollout.total_completion_tokens)
    if rollout.error:
        logger.info("  Error: %s", rollout.error)

    # Analyze tool calls
    tool_calls = []
    for step in rollout.steps:
        if step.tool_calls:
            for tc in step.tool_calls:
                fn = tc.get("function", {})
                tool_calls.append(fn.get("name", "unknown"))

    logger.info("  Tool call sequence: %s", " → ".join(tool_calls))

    # Count tool types
    from collections import Counter

    counts = Counter(tool_calls)
    logger.info("  Tool call breakdown:")
    for tool, count in counts.most_common():
        logger.info("    %s: %d", tool, count)
    logger.info("=" * 60)


def run_step3(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 3: 1 rollout from 1 PR from 1 repo."""
    logger.info("=== STEP 3: 1 rollout x 1 PR x 1 repo ===")

    loader = SWERebenchV2Loader(language_filter="Python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=42)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=1, seed=42)
    pr = prs[0]

    logger.info("Selected repo: %s", repo)
    logger.info("Selected PR: %s", pr.instance_id)
    logger.info("Problem: %s", pr.problem_statement[:200])

    rollout = generate_rollout(client=client, model=model, pr=pr, temperature=0.7)

    analyze_rollout(rollout)
    save_rollouts([rollout], os.path.join(output_dir, "step3", "rollouts.json"))


def run_step4(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 4: 10 rollouts from 1 PR, measure diversity."""
    logger.info("=== STEP 4: 10 rollouts x 1 PR ===")

    loader = SWERebenchV2Loader(language_filter="Python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=42)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=1, seed=42)
    pr = prs[0]

    logger.info("Selected repo: %s", repo)
    logger.info("Selected PR: %s", pr.instance_id)

    rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)

    step_dir = os.path.join(output_dir, "step4")
    save_rollouts(rollouts, os.path.join(step_dir, "rollouts.json"))

    report = measure_diversity(rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def run_step5(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 5: 10 rollouts x 10 PRs x 1 repo = 100 rollouts."""
    logger.info("=== STEP 5: 10 rollouts x 10 PRs x 1 repo ===")

    loader = SWERebenchV2Loader(language_filter="Python")
    repos = loader.sample_repos(n=1, min_prs=10, seed=42)
    repo = repos[0]
    prs = loader.sample_prs(repo, n=10, seed=42)

    logger.info("Selected repo: %s", repo)
    logger.info("Selected %d PRs", len(prs))

    all_rollouts: list[Rollout] = []
    for i, pr in enumerate(prs):
        logger.info("PR %d/%d: %s", i + 1, len(prs), pr.instance_id)
        rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)
        all_rollouts.extend(rollouts)

    step_dir = os.path.join(output_dir, "step5")
    save_rollouts(all_rollouts, os.path.join(step_dir, "rollouts.json"))

    report = measure_diversity(all_rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def run_step6(client: OpenAI, model: str, output_dir: str) -> None:
    """Step 6: 10 rollouts x 10 PRs x 10 repos = 1000 rollouts."""
    logger.info("=== STEP 6: 10 rollouts x 10 PRs x 10 repos ===")

    loader = SWERebenchV2Loader(language_filter="Python")
    repos = loader.sample_repos(n=10, min_prs=10, seed=42)

    logger.info("Selected %d repos: %s", len(repos), repos)

    all_rollouts: list[Rollout] = []
    for ri, repo in enumerate(repos):
        logger.info("Repo %d/%d: %s", ri + 1, len(repos), repo)
        prs = loader.sample_prs(repo, n=10, seed=42)
        logger.info("  Selected %d PRs", len(prs))

        for pi, pr in enumerate(prs):
            logger.info("  PR %d/%d: %s", pi + 1, len(prs), pr.instance_id)
            rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)
            all_rollouts.extend(rollouts)

    step_dir = os.path.join(output_dir, "step6")
    save_rollouts(all_rollouts, os.path.join(step_dir, "rollouts.json"))

    report = measure_diversity(all_rollouts)
    logger.info(report.summary())
    save_report(report, os.path.join(step_dir, "diversity_report.json"))


def main():
    parser = argparse.ArgumentParser(description="SWE-ZERO MVP rollout generation")
    parser.add_argument("--api_base", required=True, help="vLLM API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Model to use")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--step", type=int, required=True, choices=[3, 4, 5, 6], help="Which step to run")
    parser.add_argument("--output_dir", default="/tmp/swe_zero_mvp", help="Output directory")
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    logger.info("Using model %s at %s", args.model, args.api_base)

    if args.step == 3:
        run_step3(client, args.model, args.output_dir)
    elif args.step == 4:
        run_step4(client, args.model, args.output_dir)
    elif args.step == 5:
        run_step5(client, args.model, args.output_dir)
    elif args.step == 6:
        run_step6(client, args.model, args.output_dir)


if __name__ == "__main__":
    main()
