# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run OLMoBaseEval easy-overlap for the original Olmix BPB baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace

from fray.cluster import ResourceConfig
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
)
from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import RESULTS_JSON, resolve_unique_checkpoint_root
from experiments.domain_phase_mix.qsplit240_replay import resolve_qsplit240_eval_cache_path_for_regions
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_PHASE_WEIGHTS,
    OLMIX_LOGLINEAR_RUN_NAME,
    OLMIX_LOGLINEAR_SOURCE_EXPERIMENT,
    create_olmix_loglinear_import_source,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_CACHE_PATH,
    OLMO_BASE_EASY_OVERLAP_TASKS,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_bpb_olmo_base_easy_overlap_rerun"
COHORT = "olmix_bpb_olmo_base_easy_overlap_rerun"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 1
LOCAL_RUN_ID = create_olmix_loglinear_import_source().local_run_id


def build_run_specs() -> list[OriginalSwarmEvalSpec]:
    """Build the single-run OLMoBaseEval overlap spec for the Olmix BPB baseline."""
    return [
        OriginalSwarmEvalSpec(
            run_id=LOCAL_RUN_ID,
            run_name=OLMIX_LOGLINEAR_RUN_NAME,
            cohort=COHORT,
            source_experiment=OLMIX_LOGLINEAR_SOURCE_EXPERIMENT,
            checkpoint_root=None,
            phase_weights=OLMIX_LOGLINEAR_PHASE_WEIGHTS,
        )
    ]


def resolve_checkpoint_roots(run_specs: list[OriginalSwarmEvalSpec]) -> list[OriginalSwarmEvalSpec]:
    """Resolve checkpoint roots for all eval specs."""
    return [
        replace(
            spec,
            checkpoint_root=resolve_unique_checkpoint_root(
                source_experiment=spec.source_experiment,
                run_name=spec.run_name,
            ),
        )
        for spec in run_specs
    ]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=OLMO_BASE_EASY_OVERLAP_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping Olmix BPB OLMoBaseEval-overlap rerun launch in CI environment")
        return

    run_specs = resolve_checkpoint_roots(build_run_specs())
    eval_datasets_cache_path = resolve_qsplit240_eval_cache_path_for_regions(
        (args.tpu_region,),
        args.eval_datasets_cache_path,
    )
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=args.name_prefix,
        experiment_name=args.name_prefix,
        run_specs=run_specs,
    )
    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone)
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
        )
        eval_steps.append(eval_step)
        results_by_run[spec.run_name] = output_path_of(eval_step, RESULTS_JSON)

    collect_step = ExecutorStep(
        name=f"{args.name_prefix}/collect_results",
        description="Collect Olmix BPB OLMoBaseEval-overlap results",
        fn=collect_eval_results,
        config=CollectEvalResultsConfig(
            output_path=this_output_path(),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            results_by_run=results_by_run,
        ),
    )

    logger.info(
        "Launching Olmix BPB OLMoBaseEval-overlap rerun on %s in %s/%s with max_concurrent=%d",
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, *eval_steps, collect_step],
        description=f"{args.name_prefix}: Olmix BPB OLMoBaseEval-overlap rerun",
    )


if __name__ == "__main__":
    main()
