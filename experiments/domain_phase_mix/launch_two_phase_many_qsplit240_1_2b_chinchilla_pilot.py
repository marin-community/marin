# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a reduced qsplit240 baseline pilot at 1.2B."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace

from marin.execution.executor import ExecutorMainConfig, executor_main

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import (
    REGMIX_1_2B_CHINCHILLA_BUDGET,
    get_num_train_steps,
    regmix_1_2b_muonh_base,
    regmix_1_2b_proxy,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    BASELINES3_PANEL,
    DEFAULT_TARGET_BUDGET,
    DEFAULT_TARGET_BUDGET_MULTIPLIER,
    build_qsplit240_replay_launch_artifacts,
    build_qsplit240_replay_run_specs,
    create_qsplit240_replay_experiment,
    normalize_tpu_regions,
    replay_description,
    skip_eval_harness_for_training_step,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_1_2b_chinchilla"
MODEL_FAMILY = "regmix_1_2b_proxy"
EXPERIMENT_BUDGET = REGMIX_1_2B_CHINCHILLA_BUDGET
TARGET_BUDGET = DEFAULT_TARGET_BUDGET
TARGET_BUDGET_MULTIPLIER = DEFAULT_TARGET_BUDGET_MULTIPLIER
BATCH_SIZE = 256
SEQ_LEN = 2048
NUM_TRAIN_STEPS = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_TPU_TYPE = "v5p-64"
DEFAULT_TPU_REGIONS = ("us-east5",)
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_PANEL = BASELINES3_PANEL
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_RESUME_LATEST_CHECKPOINTS = True


def build_run_specs(*, panel: str = DEFAULT_PANEL):
    """Build the replay manifest for the selected qsplit240 pilot panel."""
    return build_qsplit240_replay_run_specs(
        cohort="original_swarm_1_2b_chinchilla_pilot",
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        panel=panel,
    )


def create_experiment(
    *,
    name: str,
    tpu_type: str = DEFAULT_TPU_TYPE,
    tpu_regions: tuple[str, ...] = DEFAULT_TPU_REGIONS,
    tpu_zone: str | None = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
):
    """Create the 1.2B qsplit240 pilot experiment on the requested region."""
    return create_qsplit240_replay_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str | None,
    panel: str,
    eval_datasets_cache_path: str,
    resume_latest_checkpoints: bool,
):
    """Resolve the 1.2B pilot launch graph without submitting it."""
    return build_qsplit240_replay_launch_artifacts(
        name_prefix=name_prefix,
        cohort="original_swarm_1_2b_chinchilla_pilot",
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        panel=panel,
        shard_count=1,
        shard_index=0,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        eval_datasets_cache_path=eval_datasets_cache_path,
        resume_latest_checkpoints=resume_latest_checkpoints,
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the qsplit240 reduced 1.2B pilot.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-regions", default=",".join(DEFAULT_TPU_REGIONS))
    parser.add_argument("--tpu-region")
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    parser.add_argument(
        "--perplexity-only",
        "--skip-eval-harness",
        dest="skip_eval_harness",
        action="store_true",
        help="Set LEVANTER_SKIP_EVAL_HARNESS=1 while keeping validation/perplexity and checkpointing.",
    )
    parser.add_argument(
        "--resume-latest-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME_LATEST_CHECKPOINTS,
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    tpu_regions = normalize_tpu_regions(args.tpu_region or args.tpu_regions)

    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=args.tpu_zone,
        panel=args.panel,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        resume_latest_checkpoints=args.resume_latest_checkpoints,
    )
    if args.skip_eval_harness:
        artifacts = replace(
            artifacts,
            training_steps=[
                skip_eval_harness_for_training_step(training_step) for training_step in artifacts.training_steps
            ],
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built qsplit240 1.2B pilot graph in CI with %d runs under panel %s; skipping executor launch.",
            len(artifacts.run_specs),
            args.panel,
        )
        return

    logger.info(
        "Launching %d qsplit240 1.2B pilot runs on %s across %s%s with max_concurrent=%d.",
        len(artifacts.run_specs),
        args.tpu_type,
        ",".join(tpu_regions),
        f" (zone={args.tpu_zone})" if args.tpu_zone else "",
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=replay_description(
            execution_name_prefix=artifacts.execution_name_prefix,
            label="baselines3 qsplit240 pilot at 1.2B / 24B",
        ),
    )


if __name__ == "__main__":
    main()
