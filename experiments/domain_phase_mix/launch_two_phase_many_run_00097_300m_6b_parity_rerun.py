# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun the missing 300M parity evals for the completed fixed-subset noise baseline."""

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
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.determinism_analysis import RESULTS_CSV, RUN_MANIFEST_FILE
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study import (
    NAME as SOURCE_EXPERIMENT,
    build_run_specs as build_source_run_specs,
)
from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import RESULTS_JSON, phase_weights_to_columns
from experiments.domain_phase_mix.parity_eval_rerun_common import (
    PARITY_300M_EVAL_TASKS,
    flatten_parity_eval_results,
    resolve_completed_checkpoint_root,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_CACHE_PATH

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_parity_rerun"
DEFAULT_MAX_CONCURRENT = 10
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"


@dataclass(frozen=True)
class Run00097300mParityEvalSpec:
    """Manifest entry for one 300M fixed-subset parity rerun."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep"]
    trainer_seed: int
    data_seed: int | None
    simulated_epoch_subset_seed: int
    source_run_name: str
    source_experiment: str
    checkpoint_root: str | None
    num_train_steps: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the rerun manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


@dataclass(frozen=True)
class CollectEvalResultsConfig:
    """Config for flattening eval-harness outputs."""

    output_path: str
    run_specs_json: str
    results_by_run: dict[str, InputName]


def build_run_specs() -> list[Run00097300mParityEvalSpec]:
    """Build parity rerun specs for the 300M fixed-subset seed-sweep runs."""
    return [
        Run00097300mParityEvalSpec(
            run_id=spec.run_id,
            run_name=spec.run_name,
            cohort=spec.cohort,
            trainer_seed=spec.trainer_seed,
            data_seed=spec.data_seed,
            simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
            source_run_name=spec.source_run_name,
            source_experiment=SOURCE_EXPERIMENT,
            checkpoint_root=None,
            num_train_steps=spec.num_train_steps,
            phase_weights=spec.phase_weights,
        )
        for spec in build_source_run_specs()
        if spec.cohort == "seed_sweep"
    ]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Rerun the missing 300M parity evals for the fixed-subset noise baseline."
    )
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path", default=OLMO_BASE_EASY_OVERLAP_CACHE_PATH)
    return parser.parse_known_args()


def _eval_model_name(*, name_prefix: str, run_name: str) -> str:
    """Return a namespaced eval model name to avoid output-path collisions."""
    safe_prefix = name_prefix.replace("/", "__")
    return f"{safe_prefix}__{run_name}"


def _resolve_checkpoint_root(
    spec: Run00097300mParityEvalSpec,
    *,
    checkpoint_regions: tuple[str, ...],
) -> Run00097300mParityEvalSpec | None:
    checkpoint_root = resolve_completed_checkpoint_root(
        source_experiment=spec.source_experiment,
        run_name=spec.run_name,
        num_train_steps=spec.num_train_steps,
        checkpoint_regions=checkpoint_regions,
    )
    if checkpoint_root is None:
        return None
    return replace(spec, checkpoint_root=checkpoint_root)


def resolve_checkpoint_roots(
    run_specs: list[Run00097300mParityEvalSpec],
    *,
    checkpoint_regions: tuple[str, ...],
) -> list[Run00097300mParityEvalSpec]:
    """Resolve only fully completed checkpoint roots for the 300M fixed-subset parity rerun."""
    resolved: list[Run00097300mParityEvalSpec] = []
    skipped = 0
    for spec in run_specs:
        resolved_spec = _resolve_checkpoint_root(spec, checkpoint_regions=checkpoint_regions)
        if resolved_spec is None:
            skipped += 1
            continue
        resolved.append(resolved_spec)

    logger.info(
        "Resolved %d completed 300M fixed-subset parity rerun checkpoints and skipped %d unfinished runs.",
        len(resolved),
        skipped,
    )
    return resolved


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the rerun manifest for downstream analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "task_aliases": [task.task_alias for task in PARITY_300M_EVAL_TASKS],
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[Run00097300mParityEvalSpec]) -> ExecutorStep:
    """Create the manifest writer step."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save run_00097 300M parity rerun manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def collect_eval_results(config: CollectEvalResultsConfig) -> None:
    """Write flattened parity-eval results for the 300M fixed-subset baseline."""
    run_specs = [Run00097300mParityEvalSpec(**spec) for spec in json.loads(config.run_specs_json)]
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
                "source_run_name": spec.source_run_name,
                "source_experiment": spec.source_experiment,
                "checkpoint_root": spec.checkpoint_root,
                "num_train_steps": spec.num_train_steps,
                **phase_weights_to_columns(spec.phase_weights),
                **flatten_parity_eval_results(payload),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        results_df.to_csv(f, index=False)


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping run_00097 300M parity rerun launch in CI environment")
        return

    run_specs = resolve_checkpoint_roots(build_run_specs(), checkpoint_regions=(args.tpu_region,))
    if not run_specs:
        raise ValueError("No completed 300M fixed-subset runs are available for this parity rerun")

    run_manifest_step = create_run_manifest_step(name_prefix=args.name_prefix, run_specs=run_specs)
    cache_eval_step = create_cache_eval_datasets_step(
        eval_tasks=PARITY_300M_EVAL_TASKS,
        gcs_path=args.eval_datasets_cache_path,
        name_prefix=args.name_prefix,
    )

    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=[args.tpu_region], zone=args.tpu_zone)
    cache_dependency = output_path_of(cache_eval_step, ".eval_datasets_manifest.json")
    eval_steps: list[ExecutorStep] = []
    results_by_run: dict[str, InputName] = {}
    for spec in run_specs:
        if spec.checkpoint_root is None:
            raise ValueError(f"Checkpoint root was not resolved for {spec.run_name}")
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=_eval_model_name(name_prefix=args.name_prefix, run_name=spec.run_name),
            model_path=spec.checkpoint_root,
            evals=list(PARITY_300M_EVAL_TASKS),
            resource_config=resource_config,
            discover_latest_checkpoint=True,
            eval_datasets_cache_path=args.eval_datasets_cache_path,
            eval_datasets_cache_dependency=cache_dependency,
        )
        eval_steps.append(eval_step)
        results_by_run[spec.run_name] = output_path_of(eval_step, RESULTS_JSON)

    collect_step = ExecutorStep(
        name=f"{args.name_prefix}/collect_results",
        description=f"Collect run_00097 300M parity results for {len(run_specs)} runs",
        fn=collect_eval_results,
        config=CollectEvalResultsConfig(
            output_path=this_output_path(),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            results_by_run=results_by_run,
        ),
    )

    logger.info(
        "Launching run_00097 300M parity rerun for %d completed fixed-subset runs on %s in %s/%s with "
        "max_concurrent=%d.",
        len(run_specs),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, cache_eval_step, *eval_steps, collect_step],
        description=f"{args.name_prefix}: run_00097 300M parity rerun",
    )


if __name__ == "__main__":
    main()
