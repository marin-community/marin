# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun OLMoBaseEval easy-overlap metrics for selected 60M baseline validations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from functools import cache

from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_olmo_base_easy_overlap_rerun import (
    CollectEvalResultsConfig,
    OriginalSwarmEvalSpec,
    collect_eval_results,
    create_run_manifest_step,
    select_run_specs_for_shard,
    shard_execution_name_prefix,
)
from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import RESULTS_JSON
from experiments.domain_phase_mix.qsplit240_replay import (
    BASELINES3_PANEL,
    load_panel_observed_runs,
    resolve_latest_checkpoint_root,
    resolve_qsplit240_eval_cache_path_for_regions,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    STRATIFIED_RUN_ID,
    STRATIFIED_RUN_NAME,
    create_stratified_weight_config,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import (
    RUN_ID as OLMIX_UNCHEATABLE_RUN_ID,
    RUN_NAME as OLMIX_UNCHEATABLE_RUN_NAME,
    SOURCE_EXPERIMENT as OLMIX_UNCHEATABLE_SOURCE_EXPERIMENT,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_CACHE_PATH,
    OLMO_BASE_EASY_OVERLAP_TASKS,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_selected_baselines_olmo_base_easy_overlap_rerun"
COHORT = "selected_baselines_olmo_base_easy_overlap_rerun"
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
CHECKPOINT_REGIONS = ("us-east5", "us-central1")
EVAL_DATASETS_CACHE_PATH = OLMO_BASE_EASY_OVERLAP_CACHE_PATH
STRATIFIED_60M_1P2B_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b"


@cache
def _olmix_uncheatable_run_spec() -> OriginalSwarmEvalSpec:
    for run in load_panel_observed_runs(BASELINES3_PANEL):
        if run.run_name == OLMIX_UNCHEATABLE_RUN_NAME:
            return OriginalSwarmEvalSpec(
                run_id=OLMIX_UNCHEATABLE_RUN_ID,
                run_name=run.run_name,
                cohort=COHORT,
                source_experiment=OLMIX_UNCHEATABLE_SOURCE_EXPERIMENT,
                checkpoint_root=None,
                phase_weights=run.phase_weights,
            )
    raise ValueError(f"Failed to locate {OLMIX_UNCHEATABLE_RUN_NAME!r} in {BASELINES3_PANEL!r} panel")


def build_run_specs() -> list[OriginalSwarmEvalSpec]:
    """Build OLMoBaseEval-overlap rerun specs for the selected 60M baselines."""
    power_family_penalty = genericfamily_penalty_raw_optimum_summary("power_family_penalty")
    return [
        OriginalSwarmEvalSpec(
            run_id=power_family_penalty.run_id,
            run_name=power_family_penalty.run_name,
            cohort=COHORT,
            source_experiment=GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
            checkpoint_root=None,
            phase_weights=power_family_penalty.phase_weights,
        ),
        OriginalSwarmEvalSpec(
            run_id=STRATIFIED_RUN_ID,
            run_name=STRATIFIED_RUN_NAME,
            cohort=COHORT,
            source_experiment=STRATIFIED_60M_1P2B_SOURCE_EXPERIMENT,
            checkpoint_root=None,
            phase_weights=create_stratified_weight_config().phase_weights,
        ),
        _olmix_uncheatable_run_spec(),
    ]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def resolve_checkpoint_roots(
    run_specs: list[OriginalSwarmEvalSpec],
    *,
    checkpoint_regions: tuple[str, ...] = CHECKPOINT_REGIONS,
) -> list[OriginalSwarmEvalSpec]:
    """Resolve the newest checkpoint root for each selected baseline."""
    resolved_specs: list[OriginalSwarmEvalSpec] = []
    for spec in run_specs:
        checkpoint_root = resolve_latest_checkpoint_root(
            experiment_name_prefix=spec.source_experiment,
            run_name=spec.run_name,
            checkpoint_regions=checkpoint_regions,
        )
        if checkpoint_root is None:
            raise ValueError(
                "No checkpoint root found for "
                f"{spec.run_name!r} under {spec.source_experiment!r} in regions {checkpoint_regions!r}"
            )
        resolved_specs.append(replace(spec, checkpoint_root=checkpoint_root))
    return resolved_specs


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping selected-baseline OLMoBaseEval-overlap rerun launch in CI environment")
        return

    run_specs = select_run_specs_for_shard(
        resolve_checkpoint_roots(build_run_specs(), checkpoint_regions=(args.tpu_region,)),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    eval_datasets_cache_path = resolve_qsplit240_eval_cache_path_for_regions(
        (args.tpu_region,),
        args.eval_datasets_cache_path,
    )
    execution_name_prefix = shard_execution_name_prefix(
        name_prefix=args.name_prefix,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=args.name_prefix,
        run_specs=run_specs,
    )
    cache_eval_step = create_cache_eval_datasets_step(
        eval_tasks=OLMO_BASE_EASY_OVERLAP_TASKS,
        gcs_path=eval_datasets_cache_path,
        name_prefix=execution_name_prefix,
    )

    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone)
    cache_dependency = output_path_of(cache_eval_step, ".eval_datasets_manifest.json")
    eval_steps: list[ExecutorStep] = []
    results_by_run: dict[str, InputName] = {}
    for spec in run_specs:
        if spec.checkpoint_root is None:
            raise ValueError(f"Checkpoint root was not resolved for {spec.run_name}")
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=spec.run_name,
            model_path=spec.checkpoint_root,
            evals=list(OLMO_BASE_EASY_OVERLAP_TASKS),
            resource_config=resource_config,
            discover_latest_checkpoint=True,
            eval_datasets_cache_path=eval_datasets_cache_path,
            eval_datasets_cache_dependency=cache_dependency,
        )
        eval_steps.append(eval_step)
        results_by_run[spec.run_name] = output_path_of(eval_step, RESULTS_JSON)

    collect_step = ExecutorStep(
        name=f"{execution_name_prefix}/collect_results",
        description=f"Collect selected-baseline OLMoBaseEval-overlap results for {len(run_specs)} runs",
        fn=collect_eval_results,
        config=CollectEvalResultsConfig(
            output_path=this_output_path(),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            results_by_run=results_by_run,
        ),
    )

    logger.info(
        "Launching shard %d/%d of the selected-baseline OLMoBaseEval-overlap rerun for %d runs on %s in %s/%s "
        "with max_concurrent=%d",
        args.shard_index + 1,
        args.shard_count,
        len(run_specs),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, cache_eval_step, *eval_steps, collect_step],
        description=f"{execution_name_prefix}: selected-baseline OLMoBaseEval-overlap rerun",
    )


if __name__ == "__main__":
    main()
