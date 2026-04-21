# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relaunch one strong-tier scaling-study cell without relaunching the full parent."""

from __future__ import annotations

import argparse

from marin.execution.executor import ExecutorMainConfig, executor_main

from experiments.domain_phase_mix.launch_two_phase_many_strong_tier_scaling_study import (
    EVAL_DATASETS_CACHE_PATH,
    QSPLIT240_300M_EVAL_TASKS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline import (
    build_launch_artifacts as build_stratified_launch_artifacts,
)
from experiments.domain_phase_mix.qsplit240_replay import build_qsplit240_replay_launch_artifacts
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyPath,
    build_strong_tier_cells,
    resolve_scale_spec,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument(
        "--path",
        required=True,
        choices=[ScalingStudyPath.QSPLIT_REPRESENTATIVE12.value, ScalingStudyPath.STRATIFIED.value],
    )
    parser.add_argument(
        "--resume-latest-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = ScalingStudyPath(args.path)
    cells = build_strong_tier_cells()
    cell = next((cell for cell in cells if cell.name_prefix == args.name_prefix and cell.path == path), None)
    if cell is None:
        raise ValueError(f"No strong-tier cell found for path={args.path!r} name_prefix={args.name_prefix!r}")

    spec = resolve_scale_spec(cell.scale)
    if cell.path == ScalingStudyPath.QSPLIT_REPRESENTATIVE12:
        artifacts = build_qsplit240_replay_launch_artifacts(
            name_prefix=cell.name_prefix,
            cohort=cell.cohort,
            model_family=cell.model_family,
            experiment_budget=cell.experiment_budget,
            target_budget=cell.target_budget,
            target_budget_multiplier=cell.target_budget_multiplier,
            num_train_steps=cell.num_train_steps,
            batch_size=cell.batch_size,
            seq_len=cell.seq_len,
            model_config=spec.model_config,
            optimizer_config=spec.optimizer_config,
            tpu_type=cell.tpu_type,
            tpu_regions=cell.tpu_regions,
            tpu_zone=cell.tpu_zone,
            eval_tasks=QSPLIT240_300M_EVAL_TASKS,
            panel=cell.panel or "",
            shard_count=1,
            shard_index=0,
            wandb_entity=WANDB_ENTITY,
            wandb_project=WANDB_PROJECT,
            eval_datasets_cache_path=EVAL_DATASETS_CACHE_PATH,
            resume_latest_checkpoints=args.resume_latest_checkpoints,
        )
    else:
        artifacts = build_stratified_launch_artifacts(
            scale=cell.scale,
            name_prefix=cell.name_prefix,
            experiment_budget=cell.experiment_budget,
            target_budget=cell.target_budget,
            target_budget_multiplier=cell.target_budget_multiplier,
            tpu_type=cell.tpu_type,
            tpu_regions=cell.tpu_regions,
            tpu_zone=cell.tpu_zone,
            eval_datasets_cache_path=EVAL_DATASETS_CACHE_PATH,
            resume_latest_checkpoints=args.resume_latest_checkpoints,
            cohort=cell.cohort,
        )

    print(
        f"path={cell.path.value} scale={cell.scale.value} name_prefix={cell.name_prefix} "
        f"tpu_type={cell.tpu_type} target_budget_multiplier={cell.target_budget_multiplier}"
    )
    if args.dry_run:
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=1),
        steps=artifacts.steps,
        description=f"Relaunch strong-tier scaling-study cell {cell.name_prefix}",
    )


if __name__ == "__main__":
    main()
