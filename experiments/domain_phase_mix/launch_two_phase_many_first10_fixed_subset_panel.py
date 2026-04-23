# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a fixed-subset observed-mix panel using the first 10 completed swarm runs."""

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
    RUN_MANIFEST_FILE,
    create_manifest_results_step,
    create_panel_vs_noise_report_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    NAME as FIXED_SUBSET_BASELINE_NAME,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    SIMULATED_EPOCH_SUBSET_SEED,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_run_specs as build_fixed_subset_run_specs,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_two_phase_many_phase_weights

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_first10_fixed_subset_panel"
SOURCE_RUN_NAMES = tuple(f"run_{run_id:05d}" for run_id in range(2, 12))
PANEL_TRAINER_SEED = 0


@dataclass(frozen=True)
class ObservedPanelRunSpec:
    """Manifest entry for one observed fixed-subset panel run."""

    run_id: int
    run_name: str
    cohort: Literal["observed_panel"]
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: int
    source_run_name: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the observed-panel manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


def build_run_specs() -> list[ObservedPanelRunSpec]:
    """Build the first-10 observed fixed-subset panel manifest."""
    run_specs: list[ObservedPanelRunSpec] = []
    for run_id, source_run_name in enumerate(SOURCE_RUN_NAMES):
        run_specs.append(
            ObservedPanelRunSpec(
                run_id=run_id,
                run_name=f"panel_{source_run_name}",
                cohort="observed_panel",
                trainer_seed=PANEL_TRAINER_SEED,
                data_seed=None,
                simulated_epoch_subset_seed=SIMULATED_EPOCH_SUBSET_SEED,
                source_run_name=source_run_name,
                phase_weights=load_two_phase_many_phase_weights(source_run_name),
            )
        )
    return run_specs


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the observed fixed-subset panel manifest."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[ObservedPanelRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for this panel."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save first-10 fixed-subset panel manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _build_fixed_subset_baseline_manifest_json() -> str:
    run_specs = [asdict(spec) for spec in build_fixed_subset_run_specs() if spec.cohort == "seed_sweep"]
    payload = {
        "experiment_name": FIXED_SUBSET_BASELINE_NAME,
        "n_runs": len(run_specs),
        "runs": run_specs,
    }
    return json.dumps(payload, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the first-10 fixed-subset observed-run panel.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping first-10 fixed-subset panel launch in CI environment")
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=args.name_prefix,
        resources=ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"]),
    )
    run_specs = build_run_specs()
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        training_steps.append(
            experiment.create_training_step(
                weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
                name_prefix=args.name_prefix,
                run_name=spec.run_name,
                trainer_seed=spec.trainer_seed,
                data_seed=spec.data_seed,
                simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
            )
        )

    all_metrics = tuple(dict.fromkeys((*PRIMARY_METRICS, *SECONDARY_METRICS)))
    results_step = create_manifest_results_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        extra_metrics=all_metrics,
        depends_on=training_steps,
    )
    panel_report_step = create_panel_vs_noise_report_step(
        name_prefix=args.name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        baseline_manifest_json=_build_fixed_subset_baseline_manifest_json(),
        primary_metrics=PRIMARY_METRICS,
        secondary_metrics=SECONDARY_METRICS,
    )

    logger.info(
        "Launching %d first-10 fixed-subset panel runs on %s with trainer_seed=%d and simulated_epoch_subset_seed=%d.",
        len(run_specs),
        args.tpu_type,
        PANEL_TRAINER_SEED,
        SIMULATED_EPOCH_SUBSET_SEED,
    )
    all_steps = [run_manifest_step, *training_steps, results_step, panel_report_step]
    executor_main(
        ExecutorMainConfig(max_concurrent=len(training_steps)),
        steps=all_steps,
        description=f"{args.name_prefix}: first-10 fixed-subset observed-mix panel",
    )


if __name__ == "__main__":
    main()
