# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the strong-tier mixture-scaling study with explicit simulated-epoch semantics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    EVAL_DATASETS_CACHE_PATH,
    QSPLIT240_300M_EVAL_TASKS,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline import (
    build_launch_artifacts as build_stratified_launch_artifacts,
)
from experiments.domain_phase_mix.qsplit240_replay import build_qsplit240_replay_launch_artifacts
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyCell,
    ScalingStudyPath,
    build_strong_tier_cells,
    count_new_submission_runs,
    external_holdout_references,
    new_submission_cells,
    resolve_scale_spec,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_strong_tier_scaling_study"
STRONG_TIER_MANIFEST_FILE = "strong_tier_scaling_study_manifest.json"
DEFAULT_MAX_CONCURRENT = count_new_submission_runs()
DEFAULT_RESUME_LATEST_CHECKPOINTS = True


@dataclass(frozen=True)
class SaveStrongTierManifestConfig:
    """Config for writing the strong-tier study manifest."""

    output_path: str
    study_name: str
    cells_json: str
    external_references_json: str


def save_strong_tier_manifest(config: SaveStrongTierManifestConfig) -> None:
    """Persist the study matrix, including reused and holdout cells."""
    cells = json.loads(config.cells_json)
    payload = {
        "study_name": config.study_name,
        "n_cells": len(cells),
        "n_new_cells": sum(1 for cell in cells if cell["status"] == "new"),
        "n_reused_cells": sum(1 for cell in cells if cell["status"] == "reused"),
        "n_holdout_cells": sum(1 for cell in cells if cell["status"] == "holdout_only"),
        "n_new_runs": sum(cell["run_count"] for cell in cells if cell["status"] == "new"),
        "cells": cells,
        "external_references": json.loads(config.external_references_json),
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, STRONG_TIER_MANIFEST_FILE), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_strong_tier_manifest_step(*, name_prefix: str, cells: list[ScalingStudyCell]) -> ExecutorStep:
    """Create the top-level manifest writer step for the strong-tier study."""
    return ExecutorStep(
        name=f"{name_prefix}/study_manifest",
        description="Save the strong-tier scaling-study matrix and reuse decisions",
        fn=save_strong_tier_manifest,
        config=SaveStrongTierManifestConfig(
            output_path=this_output_path(),
            study_name=name_prefix,
            cells_json=json.dumps([cell.to_manifest_dict() for cell in cells], sort_keys=True),
            external_references_json=json.dumps(external_holdout_references(), sort_keys=True),
        ),
    )


def build_launch_steps(*, resume_latest_checkpoints: bool) -> tuple[list[ScalingStudyCell], list[object]]:
    """Build the full executor step graph for all new strong-tier submissions."""
    cells = build_strong_tier_cells()
    steps: list[object] = [create_strong_tier_manifest_step(name_prefix=NAME, cells=cells)]
    for cell in new_submission_cells(cells):
        spec = resolve_scale_spec(cell.scale)
        if cell.path == ScalingStudyPath.QSPLIT_REPRESENTATIVE12:
            qsplit_artifacts = build_qsplit240_replay_launch_artifacts(
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
                resume_latest_checkpoints=resume_latest_checkpoints,
            )
            steps.extend(qsplit_artifacts.steps)
            continue

        if cell.path == ScalingStudyPath.STRATIFIED:
            stratified_artifacts = build_stratified_launch_artifacts(
                scale=cell.scale,
                name_prefix=cell.name_prefix,
                experiment_budget=cell.experiment_budget,
                target_budget=cell.target_budget,
                target_budget_multiplier=cell.target_budget_multiplier,
                tpu_type=cell.tpu_type,
                tpu_regions=cell.tpu_regions,
                tpu_zone=cell.tpu_zone,
                eval_datasets_cache_path=EVAL_DATASETS_CACHE_PATH,
                resume_latest_checkpoints=resume_latest_checkpoints,
                cohort=cell.cohort,
            )
            steps.extend(stratified_artifacts.steps)
            continue

        raise ValueError(f"Unsupported new submission path {cell.path!r}")

    return cells, steps


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--resume-latest-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME_LATEST_CHECKPOINTS,
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    cells, steps = build_launch_steps(resume_latest_checkpoints=args.resume_latest_checkpoints)
    if os.getenv("CI") is not None:
        logger.info(
            "Built strong-tier scaling-study graph in CI with %d total cells, %d new runs, and %d executor steps; "
            "skipping launch.",
            len(cells),
            count_new_submission_runs(cells),
            len(steps),
        )
        return

    logger.info(
        "Launching strong-tier scaling study with %d total cells, %d new runs, and max_concurrent=%d.",
        len(cells),
        count_new_submission_runs(cells),
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            f"{NAME}: strong-tier mixture-scaling study over representative12 qsplit cells and stratified anchors "
            "with explicit target-budget multipliers"
        ),
    )


if __name__ == "__main__":
    main()
