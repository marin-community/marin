# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a representative 12-run qsplit240 pilot at 1.2B on central1."""

from __future__ import annotations

import argparse
import logging
import os
import sys

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
    REPRESENTATIVE12_PANEL,
    build_qsplit240_replay_launch_artifacts,
    build_qsplit240_replay_run_specs,
    create_qsplit240_replay_experiment,
    replay_description,
    resolve_qsplit240_eval_cache_path,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_1_2b_chinchilla"
MODEL_FAMILY = "regmix_1_2b_proxy"
EXPERIMENT_BUDGET = REGMIX_1_2B_CHINCHILLA_BUDGET
BATCH_SIZE = 256
SEQ_LEN = 2048
NUM_TRAIN_STEPS = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_TPU_TYPE = "v5p-64"
DEFAULT_TPU_REGION = "us-central1"
DEFAULT_TPU_ZONE = "us-central1-a"
DEFAULT_PANEL = REPRESENTATIVE12_PANEL
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"


def build_run_specs(*, panel: str = DEFAULT_PANEL):
    """Build the replay manifest for the selected qsplit240 pilot panel."""
    return build_qsplit240_replay_run_specs(
        cohort="original_swarm_1_2b_chinchilla_pilot",
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        num_train_steps=NUM_TRAIN_STEPS,
        panel=panel,
    )


def create_experiment(
    *,
    name: str,
    tpu_type: str = DEFAULT_TPU_TYPE,
    tpu_region: str = DEFAULT_TPU_REGION,
    tpu_zone: str = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
):
    """Create the 1.2B qsplit240 pilot experiment on the requested region."""
    return create_qsplit240_replay_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    panel: str,
    eval_datasets_cache_path: str,
):
    """Resolve the 1.2B pilot launch graph without submitting it."""
    return build_qsplit240_replay_launch_artifacts(
        name_prefix=name_prefix,
        cohort="original_swarm_1_2b_chinchilla_pilot",
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        panel=panel,
        shard_count=1,
        shard_index=0,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the qsplit240 representative 12-run 1.2B pilot.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        panel=args.panel,
        eval_datasets_cache_path=resolve_qsplit240_eval_cache_path(args.eval_datasets_cache_path),
    )
    if os.getenv("CI") is not None:
        logger.info(
            "Built qsplit240 1.2B pilot graph in CI with %d runs under panel %s; skipping executor launch.",
            len(artifacts.run_specs),
            args.panel,
        )
        return

    logger.info(
        "Launching %d qsplit240 1.2B pilot runs on %s in %s/%s with max_concurrent=%d.",
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
            label="representative12 qsplit240 pilot at 1.2B / 24B",
        ),
    )


if __name__ == "__main__":
    main()
