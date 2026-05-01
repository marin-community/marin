# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a compute-scaling repeat study for the exact run_00097 schedule."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    COMPUTE_SCALING_NOISE_SUMMARY_CSV,
    COMPUTE_SCALING_NOISE_SUMMARY_JSON,
    RUN_MANIFEST_FILE,
    TRAJECTORY_RAW_PARQUET,
    create_compute_scaling_noise_report_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    NAME as BASELINE_SEED_STUDY_NAME,
    OBJECTIVE_METRIC,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_run_specs as build_run_00097_seed_specs,
)
from experiments.domain_phase_mix.proxy_sweep import get_num_train_steps, olmo3_30m_proxy, regmix_60m_proxy
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    SEQ_LEN,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_two_phase_many_phase_weights

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_compute_scaling"
SOURCE_RUN_NAME = "run_00097"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
SEED_SWEEP_START = 10_000
N_SEED_SWEEP_RUNS = 10
REGMIX60M_6B_LADDER = "regmix60m_6b"
OLMO3_30M_3B_LADDER = "olmo3_30m_3b"
REGMIX60M_6B_BUDGET = 6_000_000_000
OLMO3_30M_3B_BUDGET = 3_000_000_000
REGMIX60M_6B_MODEL_FAMILY = "regmix_60m_proxy"
OLMO3_30M_3B_MODEL_FAMILY = "olmo3_30m_proxy"
PRIMARY_MMLU_METRICS = (
    "lm_eval/mmlu_5shot/bpb",
    "lm_eval/mmlu_5shot/choice_logprob",
    "lm_eval/mmlu_5shot/choice_logprob_norm",
    "lm_eval/mmlu_5shot/choice_prob_norm",
)
SECONDARY_METRICS = (
    "eval/paloma/c4_en/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/macro_bpb",
)
RUN_00097_PHASE_WEIGHTS = load_two_phase_many_phase_weights(SOURCE_RUN_NAME)


@dataclass(frozen=True)
class ComputeScalingRunSpec:
    """Manifest entry for one run in the compute-scaling study."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep"]
    ladder: Literal["regmix60m_6b", "olmo3_30m_3b"]
    model_family: str
    trainer_seed: int
    data_seed: int | None
    source_run_name: str
    experiment_budget: int
    num_train_steps: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the compute-scaling study manifest."""

    output_path: str
    experiment_name: str
    objective_metric: str
    run_specs_json: str


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the study manifest for downstream collection and analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "objective_metric": config.objective_metric,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_baseline_seed_manifest_json() -> str:
    run_specs = [asdict(spec) for spec in build_run_00097_seed_specs() if spec.cohort == "seed_sweep"]
    payload = {
        "experiment_name": BASELINE_SEED_STUDY_NAME,
        "objective_metric": OBJECTIVE_METRIC,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    return json.dumps(payload, sort_keys=True)


def build_run_specs() -> list[ComputeScalingRunSpec]:
    """Build the 20-run compute-scaling study manifest."""
    run_specs: list[ComputeScalingRunSpec] = []
    next_run_id = 0

    ladder_specs = (
        (REGMIX60M_6B_LADDER, REGMIX60M_6B_MODEL_FAMILY, REGMIX60M_6B_BUDGET),
        (OLMO3_30M_3B_LADDER, OLMO3_30M_3B_MODEL_FAMILY, OLMO3_30M_3B_BUDGET),
    )
    for ladder, model_family, experiment_budget in ladder_specs:
        num_train_steps = get_num_train_steps(experiment_budget, BATCH_SIZE, SEQ_LEN)
        for offset in range(N_SEED_SWEEP_RUNS):
            trainer_seed = SEED_SWEEP_START + offset
            run_specs.append(
                ComputeScalingRunSpec(
                    run_id=next_run_id,
                    run_name=f"{ladder}_trainer_seed_{trainer_seed}",
                    cohort="seed_sweep",
                    ladder=ladder,
                    model_family=model_family,
                    trainer_seed=trainer_seed,
                    data_seed=None,
                    source_run_name=SOURCE_RUN_NAME,
                    experiment_budget=experiment_budget,
                    num_train_steps=num_train_steps,
                    phase_weights=RUN_00097_PHASE_WEIGHTS,
                )
            )
            next_run_id += 1

    return run_specs


def create_run_manifest_step(*, name_prefix: str, run_specs: list[ComputeScalingRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for this study."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save run_00097 compute-scaling manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            objective_metric=OBJECTIVE_METRIC,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the run_00097 compute-scaling repeat study.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--source-run-name", default=SOURCE_RUN_NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping run_00097 compute-scaling study launch in CI environment")
        return
    if args.source_run_name != SOURCE_RUN_NAME:
        raise ValueError(f"Only {SOURCE_RUN_NAME!r} is currently supported, got {args.source_run_name!r}")

    resources = ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone)
    experiments_by_ladder = {
        REGMIX60M_6B_LADDER: create_two_phase_dolma3_dolmino_top_level_experiment(
            name=args.name_prefix,
            experiment_budget=REGMIX60M_6B_BUDGET,
            model_config=regmix_60m_proxy,
            resources=resources,
        ),
        OLMO3_30M_3B_LADDER: create_two_phase_dolma3_dolmino_top_level_experiment(
            name=args.name_prefix,
            experiment_budget=OLMO3_30M_3B_BUDGET,
            model_config=olmo3_30m_proxy,
            resources=resources,
        ),
    }
    run_specs = build_run_specs()
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        experiment = experiments_by_ladder[spec.ladder]
        training_steps.append(
            experiment.create_training_step(
                weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
                name_prefix=args.name_prefix,
                run_name=spec.run_name,
                trainer_seed=spec.trainer_seed,
                data_seed=spec.data_seed,
            )
        )

    all_metrics = tuple(dict.fromkeys((*PRIMARY_MMLU_METRICS, *SECONDARY_METRICS)))
    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        extra_metrics=all_metrics,
        depends_on=training_steps,
    )
    report_step = create_compute_scaling_noise_report_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        baseline_manifest_json=_build_baseline_seed_manifest_json(),
        primary_metrics=PRIMARY_MMLU_METRICS,
        secondary_metrics=SECONDARY_METRICS,
    )

    logger.info(
        "Launching %d run_00097 compute-scaling repeats on %s. Final report will include %s, %s, and %s.",
        len(run_specs),
        args.tpu_type,
        COMPUTE_SCALING_NOISE_SUMMARY_JSON,
        COMPUTE_SCALING_NOISE_SUMMARY_CSV,
        TRAJECTORY_RAW_PARQUET,
    )
    all_steps = [run_manifest_step, *training_steps, results_step, report_step]
    executor_main(
        ExecutorMainConfig(max_concurrent=len(training_steps)),
        steps=all_steps,
        description=f"{args.name_prefix}: run_00097 compute-scaling repeat study",
    )


if __name__ == "__main__":
    main()
