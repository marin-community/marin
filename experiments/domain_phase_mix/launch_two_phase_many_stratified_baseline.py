# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a standalone stratified baseline on one explicit scaling-study cell."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, replace

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.qsplit240_replay import (
    DEFAULT_TARGET_BUDGET_MULTIPLIER,
    Qsplit240ReplayRunSpec,
    checkpoint_initialization_path,
    create_run_manifest_step,
    normalize_tpu_regions,
    resolve_latest_checkpoint_path,
    resolve_qsplit240_eval_cache_path_for_regions,
    skip_eval_harness_for_training_step,
)
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyScale as StratifiedScale,
    resolve_scale_spec,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    STRATIFIED_RUN_NAME,
    STRATIFIED_RUN_ID,
    create_stratified_weight_config,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_RESUME_LATEST_CHECKPOINTS = True


@dataclass(frozen=True)
class StratifiedLaunchArtifacts:
    """Resolved manifest and training steps for one stratified launch."""

    run_spec: Qsplit240ReplayRunSpec
    run_manifest_step: ExecutorStep
    training_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        return [self.run_manifest_step, self.training_step]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=StratifiedScale, choices=list(StratifiedScale), required=True)
    parser.add_argument("--name-prefix")
    parser.add_argument("--tpu-type")
    parser.add_argument("--tpu-regions")
    parser.add_argument("--tpu-region")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--experiment-budget", type=int)
    parser.add_argument("--target-budget", type=int)
    parser.add_argument("--target-budget-multiplier", type=float, default=DEFAULT_TARGET_BUDGET_MULTIPLIER)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    parser.add_argument(
        "--perplexity-only",
        "--skip-eval-harness",
        dest="skip_eval_harness",
        action="store_true",
        help="Set LEVANTER_SKIP_EVAL_HARNESS=1 while keeping normal validation/perplexity evaluation.",
    )
    parser.add_argument(
        "--resume-latest-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME_LATEST_CHECKPOINTS,
    )
    return parser.parse_known_args()


def build_run_spec(
    *,
    scale: StratifiedScale,
    experiment_budget: int,
    target_budget: int,
    target_budget_multiplier: float,
    cohort: str | None = None,
) -> Qsplit240ReplayRunSpec:
    """Build a manifest entry for one stratified baseline launch."""
    spec = resolve_scale_spec(scale)
    weight_config = create_stratified_weight_config()
    return Qsplit240ReplayRunSpec(
        run_id=STRATIFIED_RUN_ID,
        run_name=STRATIFIED_RUN_NAME,
        cohort=cohort or f"baseline_stratified_{scale.value}",
        model_family=spec.model_family,
        trainer_seed=None,
        data_seed=STRATIFIED_RUN_ID,
        simulated_epoch_subset_seed=None,
        candidate_run_id=STRATIFIED_RUN_ID,
        candidate_run_name=STRATIFIED_RUN_NAME,
        candidate_source_experiment=spec.stratified_name_prefix,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        target_budget_multiplier=target_budget_multiplier,
        num_train_steps=experiment_budget // (spec.batch_size * spec.seq_len),
        phase_weights=weight_config.phase_weights,
    )


def build_launch_artifacts(
    *,
    scale: StratifiedScale,
    name_prefix: str,
    experiment_budget: int,
    target_budget: int,
    target_budget_multiplier: float,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str | None,
    eval_datasets_cache_path: str,
    resume_latest_checkpoints: bool,
    cohort: str | None = None,
) -> StratifiedLaunchArtifacts:
    """Resolve the stratified manifest and training step without launching."""
    spec = resolve_scale_spec(scale)
    run_spec = build_run_spec(
        scale=scale,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        target_budget_multiplier=target_budget_multiplier,
        cohort=cohort,
    )

    train_kwargs: dict[str, object] = {}
    if resume_latest_checkpoints:
        latest_checkpoint_path = resolve_latest_checkpoint_path(
            experiment_name_prefix=name_prefix,
            run_name=STRATIFIED_RUN_NAME,
            checkpoint_regions=tpu_regions,
        )
        if latest_checkpoint_path is not None:
            train_kwargs["initialize_from_checkpoint_path"] = checkpoint_initialization_path(latest_checkpoint_path)
            train_kwargs["reset_data_loader_on_init"] = False

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        model_config=spec.model_config,
        optimizer_config=spec.optimizer_config,
        resources=ResourceConfig.with_tpu(tpu_type, regions=list(tpu_regions), zone=tpu_zone),
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=resolve_qsplit240_eval_cache_path_for_regions(
            tpu_regions,
            eval_datasets_cache_path,
        ),
        runtime_cache_region=tpu_regions if len(tpu_regions) > 1 else tpu_regions[0],
    )
    if experiment.num_train_steps != run_spec.num_train_steps:
        raise ValueError(
            "Stratified launch step mismatch: "
            f"run_spec.num_train_steps={run_spec.num_train_steps}, "
            f"experiment.num_train_steps={experiment.num_train_steps}"
        )
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=name_prefix,
        experiment_name=name_prefix,
        model_family=spec.model_family,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        target_budget_multiplier=target_budget_multiplier,
        num_train_steps=run_spec.num_train_steps,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        run_specs=[run_spec],
    )
    training_step = experiment.create_training_step(
        weight_config=create_stratified_weight_config(),
        name_prefix=name_prefix,
        run_name=STRATIFIED_RUN_NAME,
        **train_kwargs,
    )
    return StratifiedLaunchArtifacts(
        run_spec=run_spec,
        run_manifest_step=run_manifest_step,
        training_step=training_step,
    )


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    spec = resolve_scale_spec(args.scale)
    name_prefix = args.name_prefix or spec.stratified_name_prefix
    tpu_type = args.tpu_type or spec.tpu_type
    tpu_regions = normalize_tpu_regions(args.tpu_region or args.tpu_regions or spec.tpu_regions)
    tpu_zone = args.tpu_zone or spec.tpu_zone
    experiment_budget = args.experiment_budget or spec.experiment_budget_for_multiplier(args.target_budget_multiplier)
    target_budget = args.target_budget or spec.target_budget_for_multiplier(args.target_budget_multiplier)
    artifacts = build_launch_artifacts(
        scale=args.scale,
        name_prefix=name_prefix,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        target_budget_multiplier=args.target_budget_multiplier,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        resume_latest_checkpoints=args.resume_latest_checkpoints,
    )
    if args.skip_eval_harness:
        artifacts = replace(artifacts, training_step=skip_eval_harness_for_training_step(artifacts.training_step))
    if os.getenv("CI") is not None:
        logger.info(
            "Built stratified baseline graph in CI for scale %s with target_budget=%d and multiplier=%.3f; "
            "skipping executor launch.",
            spec.scale,
            target_budget,
            args.target_budget_multiplier,
        )
        return

    logger.info(
        "Launching stratified baseline on %s with budget=%d, target_budget=%d, multiplier=%.3f, "
        "tpu=%s, regions=%s, zone=%s",
        spec.scale,
        experiment_budget,
        target_budget,
        args.target_budget_multiplier,
        tpu_type,
        ",".join(tpu_regions),
        tpu_zone,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=1),
        steps=artifacts.steps,
        description=f"{name_prefix}: {STRATIFIED_RUN_NAME} ({spec.scale})",
    )


if __name__ == "__main__":
    main()
