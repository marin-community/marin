# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 300M / 6B validations for GRP penalty raw-optimum candidates."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

import experiments.domain_phase_mix.qsplit240_replay as qsplit240_replay
from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_penalty_raw_optima_baselines as raw_optima,
)
from experiments.domain_phase_mix.determinism_analysis import (
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    EVAL_DATASETS_CACHE_PATH,
    EXPERIMENT_BUDGET,
    MODEL_FAMILY,
    NUM_TRAIN_STEPS,
    QSPLIT240_300M_EVAL_TASKS,
    TARGET_BUDGET,
    TARGET_BUDGET_MULTIPLIER,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.qsplit240_replay import Qsplit240ReplayRunSpec
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import BATCH_SIZE, SEQ_LEN

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_300m_6b"
DEFAULT_VARIANTS = "power_family_penalty"
DEFAULT_TPU_REGIONS = ",".join(qsplit240_replay.DEFAULT_REGION_AGNOSTIC_TPU_REGIONS)
FIT_SUMMARY_JSON = "genericfamily_penalty_raw_optima_300m_fit_summaries.json"
FIT_SUMMARY_CSV = "genericfamily_penalty_raw_optima_300m_fit_summaries.csv"


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting the source raw-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write raw-optimum source summaries used for the 300M replay."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as handle:
        handle.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as handle:
        handle.write(config.fit_summaries_csv)


def _validation_run_name(run_name: str) -> str:
    return f"{run_name}_300m_6b"


def build_run_specs(*, variants: tuple[str, ...] | None = None) -> list[Qsplit240ReplayRunSpec]:
    """Build the 300M replay manifest for the requested raw-optimum variants."""
    summaries = raw_optima.genericfamily_penalty_raw_optimum_summaries(variants)
    return [
        Qsplit240ReplayRunSpec(
            run_id=summary.run_id,
            run_name=_validation_run_name(summary.run_name),
            cohort="grp_penalty_raw_optimum_300m_6b",
            model_family=MODEL_FAMILY,
            trainer_seed=None,
            data_seed=0,
            simulated_epoch_subset_seed=None,
            candidate_run_id=summary.run_id,
            candidate_run_name=summary.run_name,
            candidate_source_experiment=raw_optima.GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
            experiment_budget=EXPERIMENT_BUDGET,
            target_budget=TARGET_BUDGET,
            target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
            num_train_steps=NUM_TRAIN_STEPS,
            phase_weights=summary.phase_weights,
        )
        for summary in summaries
    ]


def create_experiment(
    *,
    name: str,
    tpu_type: str,
    tpu_regions: str | tuple[str, ...] = DEFAULT_TPU_REGIONS,
    tpu_zone: str | None = None,
    eval_datasets_cache_path: str | None = None,
):
    """Create the 300M replay experiment for raw-optimum validation."""
    return qsplit240_replay.create_qsplit240_replay_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type=tpu_type,
        tpu_regions=qsplit240_replay.normalize_tpu_regions(tpu_regions),
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGIONS)
    parser.add_argument("--tpu-zone", default=None)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping 300M / 6B GRP penalty raw-optimum validation launch in CI environment")
        return

    variants = raw_optima.parse_penalty_raw_optimum_variants(args.variants)
    tpu_regions = qsplit240_replay.normalize_tpu_regions(args.tpu_region)
    run_specs = build_run_specs(variants=variants)
    summaries = raw_optima.genericfamily_penalty_raw_optimum_summaries(variants)
    execution_name_prefix = args.name_prefix
    experiment = create_experiment(
        name=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
    )

    fit_summary_step = ExecutorStep(
        name=f"{execution_name_prefix}/fit_summaries",
        description="Save source GRP raw-optimum fit summaries for 300M replay",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=raw_optima.genericfamily_penalty_raw_optimum_summaries_json(variants),
            fit_summaries_csv=raw_optima.genericfamily_penalty_raw_optimum_summaries_frame(variants).to_csv(index=False),
        ),
    )
    run_manifest_step = qsplit240_replay.create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=args.name_prefix,
        model_family=MODEL_FAMILY,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        run_specs=run_specs,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=qsplit240_replay.resolve_qsplit240_eval_cache_path_for_regions(
            tpu_regions,
            args.eval_datasets_cache_path,
        ),
        name_prefix=execution_name_prefix,
    )

    training_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=args.name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
        )
        training_steps.append(
            qsplit240_replay.add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        )

    results_step = create_manifest_results_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )

    logger.info(
        "Launching %d 300M / 6B GRP penalty raw-optimum validations on %s across %s%s with max_concurrent=%d",
        len(run_specs),
        args.tpu_type,
        ",".join(tpu_regions),
        f" zone={args.tpu_zone}" if args.tpu_zone else "",
        args.max_concurrent,
    )
    for summary, run_spec in zip(summaries, run_specs, strict=True):
        logger.info(
            "  variant=%s source_run=%s replay_run=%s predicted=%.6f nearest_tv=%.6f",
            summary.variant_name,
            summary.run_name,
            run_spec.run_name,
            summary.raw_predicted_optimum_value,
            summary.nearest_observed_tv_distance,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[
            fit_summary_step,
            run_manifest_step,
            cache_eval_datasets_step,
            *training_steps,
            results_step,
            fit_dataset_step,
        ],
        description=f"{args.name_prefix}: 300M / 6B GRP penalty raw-optimum validations",
    )


if __name__ == "__main__":
    main()
