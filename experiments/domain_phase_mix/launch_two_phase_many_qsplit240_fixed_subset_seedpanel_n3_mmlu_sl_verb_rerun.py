# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun MMLU SL-Verb evals for the completed qsplit240 fixed-subset seedpanel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.determinism_analysis import RESULTS_CSV, RUN_MANIFEST_FILE
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3 import (
    NAME as SOURCE_EXPERIMENT,
    build_run_specs as build_seedpanel_run_specs,
)
from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import (
    RESULTS_JSON,
    flatten_eval_results,
    phase_weights_to_columns,
    resolve_unique_checkpoint_root,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.task_configs import MMLU_SL_VERB_5_SHOT

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_rerun"
DEFAULT_MAX_CONCURRENT = 24


@dataclass(frozen=True)
class FixedSubsetSeedpanelEvalSpec:
    """Manifest entry for one fixed-subset seedpanel MMLU SL-Verb rerun."""

    run_id: int
    run_name: str
    cohort: Literal["replicated_swarm"]
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: int
    candidate_run_id: int
    candidate_run_name: str
    candidate_source_experiment: str
    source_experiment: str
    checkpoint_root: str | None
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the rerun manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


@dataclass(frozen=True)
class CollectEvalResultsConfig:
    """Config for flattening eval-harness outputs from the fixed-subset seedpanel rerun."""

    output_path: str
    run_specs_json: str
    results_by_run: dict[str, InputName]


def build_run_specs() -> list[FixedSubsetSeedpanelEvalSpec]:
    """Build rerun specs for the completed qsplit240 fixed-subset seedpanel."""
    return [
        FixedSubsetSeedpanelEvalSpec(
            run_id=spec.run_id,
            run_name=spec.run_name,
            cohort="replicated_swarm",
            trainer_seed=spec.trainer_seed,
            data_seed=spec.data_seed,
            simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
            candidate_run_id=spec.candidate_run_id,
            candidate_run_name=spec.candidate_run_name,
            candidate_source_experiment=spec.candidate_source_experiment,
            source_experiment=SOURCE_EXPERIMENT,
            checkpoint_root=None,
            phase_weights=spec.phase_weights,
        )
        for spec in build_seedpanel_run_specs()
    ]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Rerun MMLU SL-Verb for the completed qsplit240 fixed-subset seedpanel."
    )
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    return parser.parse_known_args()


def _resolve_checkpoint_root(spec: FixedSubsetSeedpanelEvalSpec) -> FixedSubsetSeedpanelEvalSpec:
    return replace(
        spec,
        checkpoint_root=resolve_unique_checkpoint_root(
            source_experiment=spec.source_experiment,
            run_name=spec.run_name,
        ),
    )


def resolve_checkpoint_roots(run_specs: list[FixedSubsetSeedpanelEvalSpec]) -> list[FixedSubsetSeedpanelEvalSpec]:
    """Resolve the finished checkpoint root for every fixed-subset seedpanel run."""
    return [_resolve_checkpoint_root(spec) for spec in run_specs]


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the rerun manifest for downstream analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "task_alias": MMLU_SL_VERB_5_SHOT.task_alias,
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[FixedSubsetSeedpanelEvalSpec]) -> ExecutorStep:
    """Create the manifest writer step."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save fixed-subset seedpanel MMLU SL-Verb rerun manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def collect_eval_results(config: CollectEvalResultsConfig) -> None:
    """Write flattened MMLU SL-Verb results for the fixed-subset seedpanel rerun."""
    run_specs = [FixedSubsetSeedpanelEvalSpec(**spec) for spec in json.loads(config.run_specs_json)]
    rows: list[dict[str, Any]] = []

    for spec in run_specs:
        results_path = config.results_by_run[spec.run_name]
        with fsspec.open(results_path, "r") as f:
            payload = json.load(f)

        rows.append(
            {
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "cohort": spec.cohort,
                "trainer_seed": spec.trainer_seed,
                "data_seed": spec.data_seed,
                "simulated_epoch_subset_seed": spec.simulated_epoch_subset_seed,
                "candidate_run_id": spec.candidate_run_id,
                "candidate_run_name": spec.candidate_run_name,
                "candidate_source_experiment": spec.candidate_source_experiment,
                "source_experiment": spec.source_experiment,
                "checkpoint_root": spec.checkpoint_root,
                **phase_weights_to_columns(spec.phase_weights),
                **flatten_eval_results(payload),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["run_id", "trainer_seed"]).reset_index(drop=True)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        results_df.to_csv(f, index=False)


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping fixed-subset seedpanel MMLU SL-Verb rerun launch in CI environment")
        return

    run_specs = resolve_checkpoint_roots(build_run_specs())
    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)

    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"])
    eval_steps: list[ExecutorStep] = []
    results_by_run: dict[str, InputName] = {}
    for spec in run_specs:
        if spec.checkpoint_root is None:
            raise ValueError(f"Checkpoint root was not resolved for {spec.run_name}")
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=spec.run_name,
            model_path=spec.checkpoint_root,
            evals=[MMLU_SL_VERB_5_SHOT],
            resource_config=resource_config,
            discover_latest_checkpoint=True,
        )
        eval_steps.append(eval_step)
        results_by_run[spec.run_name] = output_path_of(eval_step, RESULTS_JSON)

    collect_step = ExecutorStep(
        name=f"{args.name_prefix}/collect_results",
        description=f"Collect fixed-subset seedpanel MMLU SL-Verb results for {len(run_specs)} runs",
        fn=collect_eval_results,
        config=CollectEvalResultsConfig(
            output_path=this_output_path(),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            results_by_run=results_by_run,
        ),
    )

    logger.info(
        "Launching fixed-subset seedpanel MMLU SL-Verb rerun for %d runs on %s with max_concurrent=%d. "
        "Outputs will include %s and %s.",
        len(run_specs),
        args.tpu_type,
        args.max_concurrent,
        RUN_MANIFEST_FILE,
        RESULTS_CSV,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, *eval_steps, collect_step],
        description=f"{args.name_prefix}: fixed-subset seedpanel MMLU SL-Verb rerun",
    )


if __name__ == "__main__":
    main()
