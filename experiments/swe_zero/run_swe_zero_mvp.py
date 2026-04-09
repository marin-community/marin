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
        --model google/gemma-4-E2B-it \
        --step 5 \
        --output_dir gs://marin-us-central2/experiments/swe_zero_mvp

    # Direct usage with an already-running vLLM server:
    uv run python experiments/swe_zero/run_swe_zero_mvp.py \
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
    save_json(
        {
            "n_rollouts": report.n_rollouts,
            "n_unique": report.n_unique,
            "mean_pairwise_jaccard": report.mean_pairwise_jaccard,
            "median_pairwise_jaccard": report.median_pairwise_jaccard,
        },
        os.path.join(step_dir, "diversity_report.json"),
    )


def _run_steps(api_base: str, model: str, step: int, output_dir: str) -> None:
    """Run rollout generation against a running vLLM server."""
    from openai import OpenAI

    from experiments.swe_zero.data_loader import SWERebenchV2Loader
    from experiments.swe_zero.diversity import measure_diversity
    from experiments.swe_zero.rollout_generator import Rollout, generate_rollout, generate_rollouts_for_pr

    client = OpenAI(base_url=api_base, api_key="EMPTY")

    loader = SWERebenchV2Loader(language_filter="python")

    if step == 3:
        logger.info("=== STEP 3: 1 rollout x 1 PR x 1 repo ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=1, seed=7)
        logger.info("Repo: %s, PR: %s", repos[0], prs[0].instance_id)
        rollout = generate_rollout(client=client, model=model, pr=prs[0], temperature=0.7)
        logger.info("Rollout: %d steps, finished=%s", len(rollout.steps), rollout.finished)
        save_json([rollout.to_dict()], os.path.join(output_dir, "step3", "rollouts.json"))

    elif step == 4:
        logger.info("=== STEP 4: 10 rollouts x 1 PR ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=1, seed=7)
        rollouts = generate_rollouts_for_pr(client=client, model=model, pr=prs[0], n_rollouts=10, temperature=1.0)
        step_dir = os.path.join(output_dir, "step4")
        save_json([r.to_dict() for r in rollouts], os.path.join(step_dir, "rollouts.json"))
        report = measure_diversity(rollouts)
        logger.info(report.summary())
        _save_diversity(report, step_dir)

    elif step == 5:
        logger.info("=== STEP 5: 10 rollouts x 10 PRs x 1 repo ===")
        repos = loader.sample_repos(n=1, min_prs=10, seed=7)
        prs = loader.sample_prs(repos[0], n=10, seed=7)
        logger.info("Repo: %s, %d PRs", repos[0], len(prs))
        all_rollouts: list[Rollout] = []
        step_dir = os.path.join(output_dir, "step5")
        for i, pr in enumerate(prs):
            logger.info("PR %d/%d: %s", i + 1, len(prs), pr.instance_id)
            rollouts = generate_rollouts_for_pr(client=client, model=model, pr=pr, n_rollouts=10, temperature=1.0)
            all_rollouts.extend(rollouts)
            save_json([r.to_dict() for r in all_rollouts], os.path.join(step_dir, "rollouts.json"))
        report = measure_diversity(all_rollouts)
        logger.info(report.summary())
        _save_diversity(report, step_dir)

    elif step == 6:
        logger.info("=== STEP 6: 10 rollouts x 10 PRs x 10 repos ===")
        repos = loader.sample_repos(n=10, min_prs=10, seed=7)
        logger.info("Selected %d repos", len(repos))
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
                    save_json([r.to_dict() for r in all_rollouts], os.path.join(step_dir, "rollouts.json"))
                    logger.info("  Saved %d rollouts", len(all_rollouts))
        save_json([r.to_dict() for r in all_rollouts], os.path.join(step_dir, "rollouts.json"))
        report = measure_diversity(all_rollouts)
        logger.info(report.summary())
        _save_diversity(report, step_dir)


def _run_with_vllm(model_name: str, model_path: str, step: int, output_dir: str) -> None:
    """Start a VllmEnvironment and run rollout generation against it."""
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment
    from marin.utils import remove_tpu_lockfile_on_exit

    os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")

    model_config = ModelConfig(
        name=model_name,
        path=model_path,
        engine_kwargs={"max_model_len": 8192},
    )

    with remove_tpu_lockfile_on_exit(), VllmEnvironment(model_config) as env:
        logger.info("vLLM server ready at %s", env.server_url)
        _run_steps(env.server_url, model_name, step, output_dir)


def _launch_on_cluster(model_name: str, model_path: str, step: int, output_dir: str, tpu_type: str) -> None:
    """Submit as a Fray job to the Iris cluster (same pattern as harbor_evaluator)."""
    from fray.v1.cluster import (
        Entrypoint,
        EnvironmentConfig,
        JobRequest,
        ResourceConfig,
        TpuConfig,
        current_cluster,
    )
    from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, resolve_vllm_mode

    mode_str = resolve_vllm_mode(None)
    pip_packages = VLLM_NATIVE_PIP_PACKAGES if mode_str == "native" else ()

    env_vars = {
        "VLLM_TPU_SKIP_PRECOMPILE": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    }
    for key in ("HF_TOKEN", "WANDB_API_KEY", "MARIN_VLLM_MODE", "MARIN_PREFIX"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value

    def _run() -> None:
        from marin.utils import remove_tpu_lockfile_on_exit

        with remove_tpu_lockfile_on_exit():
            _run_with_vllm(model_name, model_path, step, output_dir)

    job_request = JobRequest(
        name=f"swe-zero-step{step}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=ResourceConfig(device=TpuConfig(variant=tpu_type)),
        environment=EnvironmentConfig.create(
            extras=["vllm"],
            pip_packages=list(pip_packages),
            env_vars=env_vars,
        ),
        max_retries_failure=0,
        max_retries_preemption=10,
    )

    cluster = current_cluster()
    logger.info("Submitting swe-zero-step%d to cluster (%s)", step, tpu_type)
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


def main():
    parser = argparse.ArgumentParser(description="SWE-ZERO MVP rollout generation")
    parser.add_argument("--api_base", default=None, help="vLLM API base URL (if server already running)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Model name")
    parser.add_argument("--model_path", default=None, help="Model path (defaults to --model)")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--step", type=int, required=True, choices=[3, 4, 5, 6], help="Which step to run")
    parser.add_argument("--output_dir", default="/tmp/swe_zero_mvp", help="Output directory (local or gs://)")
    parser.add_argument("--tpu_type", default="v6e-8", help="TPU type for cluster submission")
    parser.add_argument("--local", action="store_true", help="Run locally with VllmEnvironment (no Fray)")
    args = parser.parse_args()

    model_path = args.model_path or args.model

    if args.api_base:
        # Direct mode: server already running
        _run_steps(args.api_base, args.model, args.step, args.output_dir)
    elif args.local:
        # Local mode: start VllmEnvironment in this process
        _run_with_vllm(args.model, model_path, args.step, args.output_dir)
    else:
        # Cluster mode: submit as Fray job
        _launch_on_cluster(args.model, model_path, args.step, args.output_dir, args.tpu_type)


if __name__ == "__main__":
    main()
