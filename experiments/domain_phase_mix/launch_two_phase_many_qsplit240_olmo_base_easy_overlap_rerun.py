# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun the OLMoBaseEval easy-suite overlap on the original two-phase-many swarm."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import (
    RESULTS_JSON,
    flatten_eval_results,
    phase_weights_to_columns,
    resolve_unique_checkpoint_root,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_CACHE_PATH,
    OLMO_BASE_EASY_OVERLAP_TASKS,
    add_olmo_base_easy_overlap_metrics,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_olmo_base_easy_overlap_rerun"
DEFAULT_MAX_CONCURRENT = 12
EVAL_DATASETS_CACHE_PATH = OLMO_BASE_EASY_OVERLAP_CACHE_PATH


@dataclass(frozen=True)
class OriginalSwarmEvalSpec:
    """Manifest entry for one original-swarm OLMoBaseEval-overlap rerun."""

    run_id: int
    run_name: str
    cohort: Literal["original_swarm_olmo_base_easy_overlap_rerun"]
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
    """Config for flattening eval-harness outputs from the original swarm rerun."""

    output_path: str
    run_specs_json: str
    results_by_run: dict[str, InputName]


def build_run_specs() -> list[OriginalSwarmEvalSpec]:
    """Build rerun specs for the original qsplit240 swarm plus core baselines."""
    observed_runs = load_original_qsplit240_with_core_baselines()
    return [
        OriginalSwarmEvalSpec(
            run_id=observed.run_id,
            run_name=observed.run_name,
            cohort="original_swarm_olmo_base_easy_overlap_rerun",
            source_experiment=observed.source_experiment,
            checkpoint_root=None,
            phase_weights=observed.phase_weights,
        )
        for observed in observed_runs
    ]


def shard_label(*, shard_index: int, shard_count: int) -> str:
    """Return a stable user-facing shard label."""
    return f"shard_{shard_index + 1:02d}of{shard_count:02d}"


def select_run_specs_for_shard(
    run_specs: list[OriginalSwarmEvalSpec], *, shard_index: int, shard_count: int
) -> list[OriginalSwarmEvalSpec]:
    """Select one contiguous shard from the full overlap rerun manifest."""
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")

    shard_size = math.ceil(len(run_specs) / shard_count)
    start = shard_index * shard_size
    end = min(start + shard_size, len(run_specs))
    return run_specs[start:end]


def shard_execution_name_prefix(*, name_prefix: str, shard_index: int, shard_count: int) -> str:
    """Return the executor-side name prefix for shard-local bookkeeping steps."""
    if shard_count == 1:
        return name_prefix
    return f"{name_prefix}/{shard_label(shard_index=shard_index, shard_count=shard_count)}"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Rerun the OLMoBaseEval overlap suite for the original swarm.")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def _resolve_checkpoint_root(spec: OriginalSwarmEvalSpec) -> OriginalSwarmEvalSpec:
    return replace(
        spec,
        checkpoint_root=resolve_unique_checkpoint_root(
            source_experiment=spec.source_experiment,
            run_name=spec.run_name,
        ),
    )


def resolve_checkpoint_roots(run_specs: list[OriginalSwarmEvalSpec]) -> list[OriginalSwarmEvalSpec]:
    """Resolve the finished checkpoint root for every original-swarm run."""
    return [_resolve_checkpoint_root(spec) for spec in run_specs]


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist the rerun manifest for downstream analysis."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "task_aliases": [task.task_alias for task in OLMO_BASE_EASY_OVERLAP_TASKS],
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(
    *,
    step_name_prefix: str,
    experiment_name: str,
    run_specs: list[OriginalSwarmEvalSpec],
) -> ExecutorStep:
    """Create the manifest writer step."""
    return ExecutorStep(
        name=f"{step_name_prefix}/run_manifest",
        description=f"Save original-swarm OLMoBaseEval-overlap rerun manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def collect_eval_results(config: CollectEvalResultsConfig) -> None:
    """Write flattened OLMoBaseEval-overlap results for the original swarm rerun."""
    run_specs = [OriginalSwarmEvalSpec(**spec) for spec in json.loads(config.run_specs_json)]
    rows: list[dict[str, Any]] = []

    for spec in run_specs:
        results_path = config.results_by_run[spec.run_name]
        with fsspec.open(results_path, "r") as f:
            payload = json.load(f)

        flat_metrics = flatten_eval_results(payload)
        flat_metrics.update(add_olmo_base_easy_overlap_metrics(flat_metrics))
        rows.append(
            {
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "cohort": spec.cohort,
                "source_experiment": spec.source_experiment,
                "checkpoint_root": spec.checkpoint_root,
                **phase_weights_to_columns(spec.phase_weights),
                **flat_metrics,
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
        logger.info("Skipping original-swarm OLMoBaseEval-overlap rerun launch in CI environment")
        return

    run_specs = select_run_specs_for_shard(
        resolve_checkpoint_roots(build_run_specs()),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
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
        gcs_path=args.eval_datasets_cache_path,
        name_prefix=execution_name_prefix,
    )

    resource_config = ResourceConfig.with_tpu(args.tpu_type, regions=["us-east5"], zone="us-east5-a")
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
            eval_datasets_cache_path=args.eval_datasets_cache_path,
            eval_datasets_cache_dependency=cache_dependency,
        )
        eval_steps.append(eval_step)
        results_by_run[spec.run_name] = output_path_of(eval_step, RESULTS_JSON)

    collect_step = ExecutorStep(
        name=f"{execution_name_prefix}/collect_results",
        description=f"Collect original-swarm OLMoBaseEval-overlap results for {len(run_specs)} runs",
        fn=collect_eval_results,
        config=CollectEvalResultsConfig(
            output_path=this_output_path(),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            results_by_run=results_by_run,
        ),
    )

    logger.info(
        "Launching shard %d/%d of the original-swarm OLMoBaseEval-overlap rerun for %d runs on %s with "
        "max_concurrent=%d. Eval step outputs stay under %s; shard bookkeeping outputs go under %s. "
        "Outputs will include %s and %s.",
        args.shard_index + 1,
        args.shard_count,
        len(run_specs),
        args.tpu_type,
        args.max_concurrent,
        args.name_prefix,
        execution_name_prefix,
        RUN_MANIFEST_FILE,
        RESULTS_CSV,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[run_manifest_step, cache_eval_step, *eval_steps, collect_step],
        description=f"{execution_name_prefix}: original-swarm OLMoBaseEval-overlap rerun",
    )


if __name__ == "__main__":
    main()
