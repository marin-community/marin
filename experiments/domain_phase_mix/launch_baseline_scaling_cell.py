# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one missing baseline-scaling trajectory cell.

This launcher is intentionally narrow: it fills cells for the central
`paper_plots/baseline_scaling_trajectories.py` workflow without relaunching an
entire scaling-study panel. It keeps validation/perplexity evaluation and final
checkpointing, and can skip lm-eval through `LEVANTER_SKIP_EVAL_HARNESS=1`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
import logging
import os
import sys

from marin.execution.executor import ExecutorMainConfig, executor_main

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    BASELINES3_PANEL,
    Qsplit240ReplayLaunchArtifacts,
    Qsplit240ReplayRunSpec,
    add_eval_cache_dependency_to_training_step,
    checkpoint_initialization_path,
    build_qsplit240_replay_run_specs,
    create_cache_eval_datasets_step,
    create_qsplit240_replay_experiment,
    create_run_manifest_step,
    normalize_tpu_regions,
    resolve_latest_checkpoint_path,
    resolve_qsplit240_eval_cache_path_for_regions,
    skip_eval_harness_for_training_step,
)
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyScale,
    resolve_scale_spec,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import (
    RUN_NAME as OLMIX_UNCHEATABLE_RUN_NAME,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    STRATIFIED_RUN_ID,
    STRATIFIED_RUN_NAME,
    create_stratified_weight_config,
)

logger = logging.getLogger(__name__)

BASELINE_SCALING_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling"
DEFAULT_TARGET_BUDGET_MULTIPLIER = 1.0
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_RESUME_LATEST_CHECKPOINTS = True


class BaselineScalingMethod(StrEnum):
    """Single-cell methods supported by the baseline-scaling plot workflow."""

    GRP_NO_L2 = "grp_no_l2"
    OLMIX = "olmix"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class BaselineScalingCellArtifacts:
    """Resolved launch artifacts for one baseline-scaling cell."""

    run_spec: Qsplit240ReplayRunSpec
    launch_artifacts: Qsplit240ReplayLaunchArtifacts

    @property
    def steps(self) -> list[object]:
        return self.launch_artifacts.steps


def baseline_scaling_source_experiment(method: BaselineScalingMethod, scale: ScalingStudyScale) -> str:
    """Return the canonical source experiment prefix for one baseline-scaling cell."""
    return f"{BASELINE_SCALING_NAME_PREFIX}_{method.value}_{scale.value}"


def _grp_no_l2_run_name(scale: ScalingStudyScale) -> str:
    summary = genericfamily_penalty_raw_optimum_summary("power_family_penalty_no_l2")
    return f"{summary.run_name}_{scale.value}"


def build_baseline_scaling_run_spec(
    *,
    method: BaselineScalingMethod,
    scale: ScalingStudyScale,
    target_budget_multiplier: float = DEFAULT_TARGET_BUDGET_MULTIPLIER,
) -> Qsplit240ReplayRunSpec:
    """Build the single-run manifest spec for one baseline-scaling cell."""
    scale_spec = resolve_scale_spec(scale)
    experiment_budget = scale_spec.experiment_budget_for_multiplier(target_budget_multiplier)
    target_budget = scale_spec.target_budget_for_multiplier(target_budget_multiplier)
    num_train_steps = scale_spec.num_train_steps_for_multiplier(target_budget_multiplier)

    if method == BaselineScalingMethod.OLMIX:
        run_specs = build_qsplit240_replay_run_specs(
            cohort=f"baseline_scaling_{method.value}_{scale.value}",
            model_family=scale_spec.model_family,
            experiment_budget=experiment_budget,
            target_budget=target_budget,
            target_budget_multiplier=target_budget_multiplier,
            num_train_steps=num_train_steps,
            panel=BASELINES3_PANEL,
        )
        matches = [run_spec for run_spec in run_specs if run_spec.run_name == OLMIX_UNCHEATABLE_RUN_NAME]
        if len(matches) != 1:
            raise ValueError(f"Expected one Olmix run spec for {scale.value}, found {len(matches)}")
        return matches[0]

    if method == BaselineScalingMethod.GRP_NO_L2:
        summary = genericfamily_penalty_raw_optimum_summary("power_family_penalty_no_l2")
        return Qsplit240ReplayRunSpec(
            run_id=summary.run_id,
            run_name=_grp_no_l2_run_name(scale),
            cohort=f"baseline_scaling_{method.value}_{scale.value}",
            model_family=scale_spec.model_family,
            trainer_seed=None,
            data_seed=0,
            simulated_epoch_subset_seed=None,
            candidate_run_id=summary.run_id,
            candidate_run_name=summary.run_name,
            candidate_source_experiment=GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
            experiment_budget=experiment_budget,
            target_budget=target_budget,
            target_budget_multiplier=target_budget_multiplier,
            num_train_steps=num_train_steps,
            phase_weights=summary.phase_weights,
        )

    if method == BaselineScalingMethod.UNIFORM:
        weight_config = create_stratified_weight_config()
        return Qsplit240ReplayRunSpec(
            run_id=STRATIFIED_RUN_ID,
            run_name=STRATIFIED_RUN_NAME,
            cohort=f"baseline_scaling_{method.value}_{scale.value}",
            model_family=scale_spec.model_family,
            trainer_seed=None,
            data_seed=STRATIFIED_RUN_ID,
            simulated_epoch_subset_seed=None,
            candidate_run_id=STRATIFIED_RUN_ID,
            candidate_run_name=STRATIFIED_RUN_NAME,
            candidate_source_experiment=baseline_scaling_source_experiment(method, scale),
            experiment_budget=experiment_budget,
            target_budget=target_budget,
            target_budget_multiplier=target_budget_multiplier,
            num_train_steps=num_train_steps,
            phase_weights=weight_config.phase_weights,
        )

    raise ValueError(f"Unsupported baseline-scaling method: {method!r}")


def build_launch_artifacts(
    *,
    method: BaselineScalingMethod,
    scale: ScalingStudyScale,
    name_prefix: str,
    target_budget_multiplier: float,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str | None,
    eval_datasets_cache_path: str,
    resume_latest_checkpoints: bool,
    skip_eval_harness: bool,
) -> BaselineScalingCellArtifacts:
    """Resolve a single-cell launch graph without submitting it."""
    scale_spec = resolve_scale_spec(scale)
    run_spec = build_baseline_scaling_run_spec(
        method=method,
        scale=scale,
        target_budget_multiplier=target_budget_multiplier,
    )
    experiment = create_qsplit240_replay_experiment(
        name=name_prefix,
        experiment_budget=run_spec.experiment_budget,
        target_budget=run_spec.target_budget,
        batch_size=scale_spec.batch_size,
        seq_len=scale_spec.seq_len,
        model_config=scale_spec.model_config,
        optimizer_config=scale_spec.optimizer_config,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(tpu_regions, eval_datasets_cache_path)
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=name_prefix,
        experiment_name=name_prefix,
        model_family=scale_spec.model_family,
        experiment_budget=run_spec.experiment_budget,
        target_budget=run_spec.target_budget,
        target_budget_multiplier=target_budget_multiplier,
        num_train_steps=run_spec.num_train_steps,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        run_specs=[run_spec],
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=resolved_eval_cache_path,
        name_prefix=name_prefix,
    )

    train_kwargs: dict[str, object] = {}
    if resume_latest_checkpoints:
        latest_checkpoint_path = resolve_latest_checkpoint_path(
            experiment_name_prefix=name_prefix,
            run_name=run_spec.run_name,
            checkpoint_regions=tpu_regions,
        )
        if latest_checkpoint_path is not None:
            train_kwargs["initialize_from_checkpoint_path"] = checkpoint_initialization_path(latest_checkpoint_path)
            train_kwargs["reset_data_loader_on_init"] = False

    training_step = experiment.create_training_step(
        weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
        name_prefix=name_prefix,
        run_name=run_spec.run_name,
        data_seed=run_spec.data_seed,
        **train_kwargs,
    )
    training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
    if skip_eval_harness:
        training_step = skip_eval_harness_for_training_step(training_step)

    results_step = create_manifest_results_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=[training_step],
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )
    launch_artifacts = Qsplit240ReplayLaunchArtifacts(
        run_specs=[run_spec],
        execution_name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=[training_step],
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )
    return BaselineScalingCellArtifacts(run_spec=run_spec, launch_artifacts=launch_artifacts)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=BaselineScalingMethod, choices=list(BaselineScalingMethod), required=True)
    parser.add_argument("--scale", type=ScalingStudyScale, choices=list(ScalingStudyScale), required=True)
    parser.add_argument("--name-prefix")
    parser.add_argument("--target-budget-multiplier", type=float, default=DEFAULT_TARGET_BUDGET_MULTIPLIER)
    parser.add_argument("--tpu-type")
    parser.add_argument("--tpu-region")
    parser.add_argument("--tpu-regions")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--max-concurrent", type=int, default=1)
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
    scale_spec = resolve_scale_spec(args.scale)
    name_prefix = args.name_prefix or baseline_scaling_source_experiment(args.method, args.scale)
    tpu_regions = normalize_tpu_regions(args.tpu_region or args.tpu_regions or scale_spec.tpu_regions)
    tpu_type = args.tpu_type or scale_spec.tpu_type
    tpu_zone = args.tpu_zone or scale_spec.tpu_zone

    artifacts = build_launch_artifacts(
        method=args.method,
        scale=args.scale,
        name_prefix=name_prefix,
        target_budget_multiplier=args.target_budget_multiplier,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        resume_latest_checkpoints=args.resume_latest_checkpoints,
        skip_eval_harness=args.skip_eval_harness,
    )
    if os.getenv("CI") is not None:
        logger.info(
            "Built baseline-scaling %s/%s graph for run %s; skipping executor launch.",
            args.method.value,
            args.scale.value,
            artifacts.run_spec.run_name,
        )
        return

    logger.info(
        "Launching baseline-scaling cell method=%s scale=%s run=%s tpu=%s regions=%s zone=%s",
        args.method.value,
        args.scale.value,
        artifacts.run_spec.run_name,
        tpu_type,
        ",".join(tpu_regions),
        tpu_zone,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=f"{name_prefix}: baseline-scaling {args.method.value} {args.scale.value}",
    )


if __name__ == "__main__":
    main()
