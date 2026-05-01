# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 60M benchmark-aggregate no-L2 GRP raw-optimum validations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import sys

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.qsplit240_replay import skip_eval_harness_for_training_step
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_benchmark_aggregate_raw_optima import (
    BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS,
    BENCHMARK_AGGREGATE_RAW_OPTIMA_SOURCE_EXPERIMENT,
    benchmark_aggregate_raw_optimum_summaries,
    benchmark_aggregate_raw_optimum_summaries_csv,
    benchmark_aggregate_raw_optimum_summaries_json,
    parse_benchmark_aggregate_raw_optimum_slugs,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness, evaluate_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_CACHE_PATH, OLMO_BASE_EASY_OVERLAP_TASKS
from experiments.evals.task_configs import GSM8K_5_SHOT, HUMANEVAL_10_SHOT

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = BENCHMARK_AGGREGATE_RAW_OPTIMA_SOURCE_EXPERIMENT
DEFAULT_SLUGS = ",".join(BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS)
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 4
FIT_SUMMARY_JSON = "benchmark_aggregate_grp_no_l2_raw_optima_fit_summaries.json"
FIT_SUMMARY_CSV = "benchmark_aggregate_grp_no_l2_raw_optima_fit_summaries.csv"
GENERATION_ENGINE_KWARGS = {"max_num_batched_tokens": 1024}


@dataclass(frozen=True)
class SaveFitSummariesConfig:
    """Config for persisting 60M benchmark-aggregate no-L2 GRP raw-optimum summaries."""

    output_path: str
    fit_summaries_json: str
    fit_summaries_csv: str


def save_fit_summaries(config: SaveFitSummariesConfig) -> None:
    """Write 60M benchmark-aggregate no-L2 GRP raw-optimum summaries."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_JSON), "w") as handle:
        handle.write(config.fit_summaries_json)
    with fsspec.open(os.path.join(config.output_path, FIT_SUMMARY_CSV), "w") as handle:
        handle.write(config.fit_summaries_csv)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--slugs", default=DEFAULT_SLUGS)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run the default Levanter eval harness during training. Default is perplexity/checkpoint only.",
    )
    parser.add_argument(
        "--skip-downstream-evals",
        action="store_true",
        help="Only train/checkpoint the optima; do not append OLMoBase, GSM8K, and HumanEval eval steps.",
    )
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--allow-local", action="store_true", help="Allow a non-dry-run local executor launch.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print launch plan without submitting.")
    return parser.parse_known_args()


def _has_iris_context() -> bool:
    try:
        from iris.client.client import get_iris_ctx
    except ImportError:
        return False
    return get_iris_ctx() is not None


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    slugs = parse_benchmark_aggregate_raw_optimum_slugs(args.slugs)
    summaries = benchmark_aggregate_raw_optimum_summaries(slugs)

    if args.dry_run:
        print(f"name_prefix={args.name_prefix}")
        print(f"tpu={args.tpu_type} region={args.tpu_region} zone={args.tpu_zone}")
        print(f"max_concurrent={args.max_concurrent}")
        print(f"include_eval_harness={args.include_eval_harness}")
        downstream_evals = "disabled" if args.skip_downstream_evals else "OLMoBase easy-overlap + GSM8K + HumanEval"
        print(f"downstream_evals={downstream_evals}")
        for summary in summaries:
            print(
                f"slug={summary.slug} run_id={summary.run_id} run_name={summary.run_name} "
                f"cv_rmse={summary.fit_cv_rmse:.6f} cv_spearman={summary.fit_cv_spearman:.6f} "
                f"support={summary.phase0_support_gt_1e4}/{summary.phase1_support_gt_1e4} weights={summary.weights_csv}"
            )
        return

    if os.getenv("CI") is not None:
        logger.info("Skipping benchmark-aggregate no-L2 GRP raw-optimum launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone),
        eval_harness_tasks=None if args.include_eval_harness else (),
    )
    fit_summary_step = ExecutorStep(
        name=f"{args.name_prefix}/fit_summaries",
        description="Save 60M benchmark-aggregate no-L2 GRP raw-optimum summaries",
        fn=save_fit_summaries,
        config=SaveFitSummariesConfig(
            output_path=this_output_path(),
            fit_summaries_json=benchmark_aggregate_raw_optimum_summaries_json(slugs),
            fit_summaries_csv=benchmark_aggregate_raw_optimum_summaries_csv(slugs),
        ),
    )
    train_steps: list[ExecutorStep] = []
    downstream_eval_steps: list[ExecutorStep] = []
    cache_eval_step: ExecutorStep | None = None
    cache_dependency = None
    downstream_eval_resource_config = ResourceConfig.with_tpu(
        args.tpu_type,
        regions=[args.tpu_region],
        zone=args.tpu_zone,
    )
    if not args.skip_downstream_evals:
        cache_eval_step = create_cache_eval_datasets_step(
            eval_tasks=OLMO_BASE_EASY_OVERLAP_TASKS,
            gcs_path=OLMO_BASE_EASY_OVERLAP_CACHE_PATH,
            name_prefix=args.name_prefix,
        )
        cache_dependency = output_path_of(cache_eval_step, ".eval_datasets_manifest.json")

    for summary in summaries:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights),
            name_prefix=args.name_prefix,
            run_name=summary.run_name,
        )
        if not args.include_eval_harness:
            training_step = skip_eval_harness_for_training_step(training_step)
        train_steps.append(training_step)
        if args.skip_downstream_evals:
            continue

        checkpoint_root = output_path_of(training_step)
        downstream_eval_steps.append(
            evaluate_levanter_lm_evaluation_harness(
                model_name=f"{summary.run_name}_olmo_base_easy_overlap",
                model_path=checkpoint_root,
                evals=list(OLMO_BASE_EASY_OVERLAP_TASKS),
                resource_config=downstream_eval_resource_config,
                max_eval_instances=args.max_eval_instances,
                discover_latest_checkpoint=True,
                eval_datasets_cache_path=OLMO_BASE_EASY_OVERLAP_CACHE_PATH,
                eval_datasets_cache_dependency=cache_dependency,
            )
        )
        downstream_eval_steps.append(
            evaluate_lm_evaluation_harness(
                model_name=f"{summary.run_name}_gsm8k_humaneval",
                model_path=checkpoint_root,
                evals=[GSM8K_5_SHOT, HUMANEVAL_10_SHOT],
                max_eval_instances=args.max_eval_instances,
                engine_kwargs=GENERATION_ENGINE_KWARGS,
                resource_config=downstream_eval_resource_config,
                discover_latest_checkpoint=True,
                wandb_tags=["benchmark-aggregate-raw-optimum", "gsm8k-humaneval"],
            )
        )

    logger.info(
        "Launching %d benchmark-aggregate no-L2 GRP raw-optimum validations and %d downstream evals "
        "in %s (%s/%s) with max_concurrent=%d",
        len(train_steps),
        len(downstream_eval_steps),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    for summary in summaries:
        logger.info(
            "  slug=%s run=%s cv_rmse=%.6f cv_spearman=%.6f support=%d/%d",
            summary.slug,
            summary.run_name,
            summary.fit_cv_rmse,
            summary.fit_cv_spearman,
            summary.phase0_support_gt_1e4,
            summary.phase1_support_gt_1e4,
        )

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[
            fit_summary_step,
            *(tuple() if cache_eval_step is None else (cache_eval_step,)),
            *train_steps,
            *downstream_eval_steps,
        ],
        description=f"{args.name_prefix}: 60M benchmark-aggregate no-L2 GRP raw-optimum validations",
    )


if __name__ == "__main__":
    main()
