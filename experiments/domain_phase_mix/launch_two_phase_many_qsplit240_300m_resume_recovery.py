# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume a single original qsplit240 300M run via the shared replay helper."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from marin.execution.executor import ExecutorMainConfig, executor_main

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    BATCH_SIZE,
    EVAL_DATASETS_CACHE_PATH,
    EXPERIMENT_BUDGET,
    MODEL_FAMILY,
    NAME,
    NUM_TRAIN_STEPS,
    QSPLIT240_300M_EVAL_TASKS,
    SEQ_LEN,
    TARGET_BUDGET,
    TARGET_BUDGET_MULTIPLIER,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    ALL_PANEL,
    build_qsplit240_replay_launch_artifacts,
    replay_description,
)

logger = logging.getLogger(__name__)

DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_SHARD_COUNT = 240
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_MARIN_PREFIX = "gs://marin-us-east5"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Resume one original qsplit240 300M run from the latest checkpoint.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--panel", default=ALL_PANEL)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    parser.add_argument("--marin-prefix", default=DEFAULT_MARIN_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    sys.argv = [sys.argv[0], *remaining]

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
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        resume_latest_checkpoints=True,
    )

    if not artifacts.run_specs:
        raise ValueError(
            f"No qsplit240 300M run resolved for shard {args.shard_index}/{args.shard_count} under panel {args.panel!r}."
        )

    run_names = [spec.run_name for spec in artifacts.run_specs]
    logger.info(
        "Prepared resumed recovery for shard %d/%d with runs=%s on %s in %s/%s.",
        args.shard_index + 1,
        args.shard_count,
        ",".join(run_names),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
    )

    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=replay_description(
            execution_name_prefix=artifacts.execution_name_prefix,
            label="original qsplit240 swarm replay at 300M / 6B resumed recovery",
        ),
    )


if __name__ == "__main__":
    main()
