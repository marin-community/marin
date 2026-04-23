# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 1.2B / 24B validations for GRP penalty raw-optimum candidates."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import fsspec
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

import experiments.domain_phase_mix.qsplit240_replay as qsplit240_replay
from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_penalty_raw_optima_baselines as raw_optima,
)
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_1_2b_chinchilla_pilot import (
    BATCH_SIZE,
    DEFAULT_TPU_REGIONS,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    EVAL_DATASETS_CACHE_PATH,
    EXPERIMENT_BUDGET,
    MODEL_FAMILY,
    NUM_TRAIN_STEPS,
    SEQ_LEN,
    TARGET_BUDGET,
    TARGET_BUDGET_MULTIPLIER,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_1_2b_muonh_base, regmix_1_2b_proxy
from experiments.domain_phase_mix.qsplit240_replay import Qsplit240ReplayRunSpec

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_1_2b_24b"
DEFAULT_VARIANTS = "power_family_penalty_no_l2"
FIT_SUMMARY_JSON = "genericfamily_penalty_raw_optima_1_2b_fit_summaries.json"
FIT_SUMMARY_CSV = "genericfamily_penalty_raw_optima_1_2b_fit_summaries.csv"


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting the source raw-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write raw-optimum source summaries used for the 1.2B replay."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as handle:
        handle.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as handle:
        handle.write(config.fit_summaries_csv)


def _validation_run_name(run_name: str) -> str:
    return f"{run_name}_1_2b_24b"


def build_run_specs(*, variants: tuple[str, ...] | None = None) -> list[Qsplit240ReplayRunSpec]:
    """Build the 1.2B replay manifest for the requested raw-optimum variants."""
    resolved_variants = variants or raw_optima.parse_penalty_raw_optimum_variants(DEFAULT_VARIANTS)
    summaries = raw_optima.genericfamily_penalty_raw_optimum_summaries(resolved_variants)
    return [
        Qsplit240ReplayRunSpec(
            run_id=summary.run_id,
            run_name=_validation_run_name(summary.run_name),
            cohort="grp_penalty_raw_optimum_1_2b_24b",
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
    tpu_zone: str | None = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
):
    """Create the 1.2B replay experiment for raw-optimum validation."""
    return qsplit240_replay.create_qsplit240_replay_experiment(
        name=name,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
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
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=",".join(DEFAULT_TPU_REGIONS))
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping 1.2B / 24B GRP penalty raw-optimum validation launch in CI environment")
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
        description="Save source GRP raw-optimum fit summaries for 1.2B replay",
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
    training_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=args.name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
        )
        training_steps.append(training_step)

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
        "Launching %d 1.2B / 24B GRP penalty raw-optimum validations on %s across %s%s with max_concurrent=%d",
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
            *training_steps,
            results_step,
            fit_dataset_step,
        ],
        description=f"{args.name_prefix}: 1.2B / 24B GRP penalty raw-optimum validations",
    )


if __name__ == "__main__":
    main()
