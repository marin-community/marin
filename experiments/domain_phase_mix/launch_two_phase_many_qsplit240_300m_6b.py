# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the original qsplit240 swarm at 300M / 6B with expanded task evals."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorMainConfig, executor_main

import experiments.domain_phase_mix.qsplit240_replay as qsplit240_replay_common
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import (
    get_num_train_steps,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    ALL_PANEL,
    build_qsplit240_replay_launch_artifacts,
    build_qsplit240_replay_run_specs,
    create_qsplit240_replay_experiment,
    replay_description,
    resolve_qsplit240_eval_cache_path_for_current_region,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import BATCH_SIZE, SEQ_LEN
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_SL_VERB_5_SHOT

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
MODEL_FAMILY = "regmix_300m_proxy"
EXPERIMENT_BUDGET = 6_000_000_000
TARGET_BUDGET = qsplit240_replay_common.DEFAULT_TARGET_BUDGET
TARGET_BUDGET_MULTIPLIER = qsplit240_replay_common.DEFAULT_TARGET_BUDGET_MULTIPLIER
NUM_TRAIN_STEPS = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_PANEL = ALL_PANEL
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
EVAL_DATASETS_CACHE_DEP_ENV_VAR = qsplit240_replay_common.EVAL_DATASETS_CACHE_DEP_ENV_VAR
add_eval_cache_dependency_to_training_step = qsplit240_replay_common.add_eval_cache_dependency_to_training_step
select_run_specs_for_shard = qsplit240_replay_common.select_run_specs_for_shard
shard_execution_name_prefix = qsplit240_replay_common.shard_execution_name_prefix
QSPLIT240_300M_EVAL_TASKS = (
    MMLU_5_SHOT,
    MMLU_SL_VERB_5_SHOT,
    EvalTaskConfig("arc_easy", 10),
    EvalTaskConfig("piqa", 10),
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
)


def build_run_specs(*, panel: str = DEFAULT_PANEL):
    """Build the replay manifest for the selected qsplit240 panel."""
    return build_qsplit240_replay_run_specs(
        cohort="original_swarm_300m",
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
    tpu_type: str,
    tpu_region: str = DEFAULT_TPU_REGION,
    tpu_zone: str = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
):
    """Create the 300M qsplit240 experiment with the expanded task suite."""
    return create_qsplit240_replay_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type=tpu_type,
        tpu_regions=(tpu_region,),
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the original qsplit240 swarm at 300M / 6B.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping qsplit240 300M / 6B swarm launch in CI environment")
        return

    eval_datasets_cache_path = resolve_qsplit240_eval_cache_path_for_current_region(args.eval_datasets_cache_path)
    artifacts = build_qsplit240_replay_launch_artifacts(
        name_prefix=args.name_prefix,
        cohort="original_swarm_300m",
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type=args.tpu_type,
        tpu_regions=(args.tpu_region,),
        tpu_zone=args.tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        panel=args.panel,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )

    logger.info(
        "Launching shard %d/%d with %d qsplit240 300M / 6B runs on %s in %s/%s with max_concurrent=%d.",
        args.shard_index + 1,
        args.shard_count,
        len(artifacts.run_specs),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=replay_description(
            execution_name_prefix=artifacts.execution_name_prefix,
            label="original qsplit240 swarm replay at 300M / 6B",
        ),
    )


if __name__ == "__main__":
    main()
