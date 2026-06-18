# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "torch"]
# ///
"""Launch a 300M/6B qsplit240 single-phase exposure-average ablation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, output_path_of, this_output_path
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_ZONE,
    EXPERIMENT_BUDGET,
    MODEL_FAMILY,
    NUM_TRAIN_STEPS,
    QSPLIT240_300M_EVAL_TASKS,
    TARGET_BUDGET,
    TARGET_BUDGET_MULTIPLIER,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import NAME as SOURCE_TWO_PHASE_EXPERIMENT
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    build_run_specs as build_source_qsplit240_300m_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.qsplit240_replay import (
    BASELINES3_PANEL,
    EVAL_DATASETS_CACHE_DEP_ENV_VAR,
    REPRESENTATIVE12_PANEL,
    SKIP_EVAL_HARNESS_ENV_VAR,
    create_qsplit240_replay_experiment,
    replay_description,
    resolve_qsplit240_eval_cache_path_for_regions,
    select_run_specs_for_shard,
    shard_execution_name_prefix,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)

NAME = "calvin/dm/spavg_q240_300m"
COHORT = "single_phase_exposure_average_qsplit240_300m"
SCALE = "300m_6b"
SCALE_DISPLAY_LABEL = "100M/6B"
SINGLE_PHASE_STRATEGY = "exposure_average_80_20"
PHASE_FRACTIONS = {"phase_0": 0.8, "phase_1": 0.2}
DEFAULT_PANEL = REPRESENTATIVE12_PANEL
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "single_phase_exposure_average_qsplit240_300m_6b"
)
LOCAL_MANIFEST_CSV = "single_phase_exposure_average_qsplit240_300m_manifest.csv"
LOCAL_SUMMARY_JSON = "single_phase_exposure_average_qsplit240_300m_summary.json"
LOCAL_RUN_SPECS_JSON = "single_phase_exposure_average_qsplit240_300m_run_specs.json"


@dataclass(frozen=True)
class SinglePhaseQsplit240RunSpec:
    """Manifest entry for one 300M qsplit240 single-phase exposure-average run."""

    run_id: int
    run_name: str
    cohort: str
    model_family: str
    trainer_seed: int | None
    data_seed: int
    simulated_epoch_subset_seed: int | None
    source_run_id: int
    source_run_name: str
    source_two_phase_experiment: str
    candidate_run_id: int
    candidate_run_name: str
    candidate_source_experiment: str
    single_phase_strategy: str
    source_panel: str
    phase_tv: float
    scale: str
    scale_display_label: str
    experiment_budget: int
    realized_experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    target_final_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveSinglePhaseQsplit240ManifestConfig:
    """Config for writing the single-phase qsplit240 run manifest."""

    output_path: str
    experiment_name: str
    model_family: str
    experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    eval_task_aliases: tuple[str, ...]
    run_specs_json: str


@dataclass(frozen=True)
class SinglePhaseQsplit240LaunchArtifacts:
    """Resolved executor graph for one single-phase qsplit240 launch."""

    run_specs: list[SinglePhaseQsplit240RunSpec]
    execution_name_prefix: str
    run_manifest_step: ExecutorStep
    cache_eval_datasets_step: ExecutorStep | None
    training_steps: list[ExecutorStep]
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        cache_steps: list[ExecutorStep] = []
        if self.cache_eval_datasets_step is not None:
            cache_steps.append(self.cache_eval_datasets_step)
        return [self.run_manifest_step, *cache_steps, *self.training_steps, self.results_step, self.fit_dataset_step]


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _phase_vector(phase_weights: dict[str, dict[str, float]], phase_name: str) -> np.ndarray:
    values = np.asarray([float(phase_weights[phase_name][domain_name]) for domain_name in DOMAIN_NAMES])
    if np.any(values < 0):
        raise ValueError(f"{phase_name} has negative weights")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{phase_name} has non-positive weight sum {total}")
    return values / total


def _phase_weights_from_vector(weights: np.ndarray) -> dict[str, dict[str, float]]:
    domain_weights = {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, weights, strict=True)}
    return {"phase_0": dict(domain_weights), "phase_1": dict(domain_weights)}


def _single_phase_vector(source_phase_weights: dict[str, dict[str, float]]) -> tuple[np.ndarray, float]:
    if PHASE_BOUNDARIES != [0.8]:
        raise ValueError(f"Expected two-phase 0.8/0.2 schedule, got PHASE_BOUNDARIES={PHASE_BOUNDARIES!r}")
    phase_0 = _phase_vector(source_phase_weights, "phase_0")
    phase_1 = _phase_vector(source_phase_weights, "phase_1")
    single = PHASE_FRACTIONS["phase_0"] * phase_0 + PHASE_FRACTIONS["phase_1"] * phase_1
    single = single / single.sum()
    phase_tv = float(0.5 * np.abs(phase_1 - phase_0).sum())
    return single, phase_tv


def _ablation_run_name(source_run_name: str) -> str:
    return f"singleavg_{source_run_name}"


def build_run_specs(
    *,
    panel: str = DEFAULT_PANEL,
    limit: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[SinglePhaseQsplit240RunSpec]:
    """Build single-phase qsplit240 run specs for the selected panel/shard."""
    source_specs = build_source_qsplit240_300m_run_specs(panel=panel)
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        source_specs = source_specs[:limit]
    source_specs = select_run_specs_for_shard(source_specs, shard_index=shard_index, shard_count=shard_count)

    run_specs: list[SinglePhaseQsplit240RunSpec] = []
    for source_spec in source_specs:
        single_vector, phase_tv = _single_phase_vector(source_spec.phase_weights)
        run_specs.append(
            SinglePhaseQsplit240RunSpec(
                run_id=source_spec.run_id,
                run_name=_ablation_run_name(source_spec.run_name),
                cohort=COHORT,
                model_family=MODEL_FAMILY,
                trainer_seed=None,
                data_seed=source_spec.run_id,
                simulated_epoch_subset_seed=None,
                source_run_id=source_spec.run_id,
                source_run_name=source_spec.run_name,
                source_two_phase_experiment=SOURCE_TWO_PHASE_EXPERIMENT,
                candidate_run_id=source_spec.candidate_run_id,
                candidate_run_name=source_spec.candidate_run_name,
                candidate_source_experiment=source_spec.candidate_source_experiment,
                single_phase_strategy=SINGLE_PHASE_STRATEGY,
                source_panel=panel,
                phase_tv=phase_tv,
                scale=SCALE,
                scale_display_label=SCALE_DISPLAY_LABEL,
                experiment_budget=EXPERIMENT_BUDGET,
                realized_experiment_budget=NUM_TRAIN_STEPS * BATCH_SIZE * SEQ_LEN,
                target_budget=TARGET_BUDGET,
                target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
                num_train_steps=NUM_TRAIN_STEPS,
                target_final_checkpoint_step=NUM_TRAIN_STEPS - 1,
                phase_weights=_phase_weights_from_vector(single_vector),
            )
        )
    validate_run_specs(run_specs)
    return run_specs


def validate_run_specs(run_specs: list[SinglePhaseQsplit240RunSpec]) -> None:
    """Validate manifest invariants before launch."""
    if not run_specs:
        raise ValueError("Single-phase qsplit240 launch requires at least one run spec")
    run_names = [spec.run_name for spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate single-phase qsplit240 run names")
    for spec in run_specs:
        if set(spec.phase_weights) != set(PHASE_NAMES):
            raise ValueError(f"{spec.run_name} phase names do not match expected {PHASE_NAMES}")
        phase_0 = spec.phase_weights["phase_0"]
        phase_1 = spec.phase_weights["phase_1"]
        if set(phase_0) != set(DOMAIN_NAMES) or set(phase_1) != set(DOMAIN_NAMES):
            raise ValueError(f"{spec.run_name} domain names do not match expected top-level domains")
        for phase_name, weights in spec.phase_weights.items():
            values = np.asarray(list(weights.values()), dtype=float)
            if np.any(values < 0):
                raise ValueError(f"{spec.run_name} has negative {phase_name} weights")
            if not np.isclose(values.sum(), 1.0, atol=1e-12):
                raise ValueError(f"{spec.run_name} {phase_name} weights sum to {values.sum()}, not 1")
        max_abs_delta = max(abs(phase_0[domain] - phase_1[domain]) for domain in DOMAIN_NAMES)
        if max_abs_delta > 1e-15:
            raise ValueError(f"{spec.run_name} is not single-phase: max phase delta {max_abs_delta}")


def _flat_manifest_row(spec: SinglePhaseQsplit240RunSpec) -> dict[str, Any]:
    row = asdict(spec)
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[_phase_column(phase_name, domain_name)] = value
    return row


def write_local_manifest(run_specs: list[SinglePhaseQsplit240RunSpec], output_dir: Path) -> None:
    """Write local audit artifacts for launch review and handoff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flat_manifest_row(spec) for spec in run_specs]
    pd.DataFrame.from_records(rows).to_csv(output_dir / LOCAL_MANIFEST_CSV, index=False)
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "experiment_name": NAME,
                "source_two_phase_experiment": SOURCE_TWO_PHASE_EXPERIMENT,
                "single_phase_strategy": SINGLE_PHASE_STRATEGY,
                "phase_fractions": PHASE_FRACTIONS,
                "runs": [asdict(spec) for spec in run_specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "experiment_name": NAME,
        "source_two_phase_experiment": SOURCE_TWO_PHASE_EXPERIMENT,
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "row_count": len(run_specs),
        "source_panel": sorted({spec.source_panel for spec in run_specs}),
        "model_family": MODEL_FAMILY,
        "scale": SCALE,
        "experiment_budget": EXPERIMENT_BUDGET,
        "realized_experiment_budget": NUM_TRAIN_STEPS * BATCH_SIZE * SEQ_LEN,
        "target_budget": TARGET_BUDGET,
        "target_budget_multiplier": TARGET_BUDGET_MULTIPLIER,
        "num_train_steps": NUM_TRAIN_STEPS,
        "target_final_checkpoint_step": NUM_TRAIN_STEPS - 1,
        "run_names": [spec.run_name for spec in run_specs],
    }
    (output_dir / LOCAL_SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def save_run_manifest(config: SaveSinglePhaseQsplit240ManifestConfig) -> None:
    """Persist the executor-side run manifest for downstream collection."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "model_family": config.model_family,
        "experiment_budget": config.experiment_budget,
        "target_budget": config.target_budget,
        "target_budget_multiplier": config.target_budget_multiplier,
        "num_train_steps": config.num_train_steps,
        "eval_tasks": list(config.eval_task_aliases),
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "n_runs": len(run_specs),
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
    eval_task_aliases: tuple[str, ...],
    run_specs: list[SinglePhaseQsplit240RunSpec],
) -> ExecutorStep:
    """Create the manifest writer step for the single-phase ablation."""
    return ExecutorStep(
        name=f"{step_name_prefix}/run_manifest",
        description=f"Save single-phase qsplit240 manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveSinglePhaseQsplit240ManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            model_family=MODEL_FAMILY,
            experiment_budget=EXPERIMENT_BUDGET,
            target_budget=TARGET_BUDGET,
            target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
            num_train_steps=NUM_TRAIN_STEPS,
            eval_task_aliases=eval_task_aliases,
            run_specs_json=json.dumps([asdict(run_spec) for run_spec in run_specs], sort_keys=True),
        ),
    )


def _add_eval_cache_dependency_for_region(
    training_step: ExecutorStep,
    cache_step: ExecutorStep,
    *,
    tpu_region: str,
) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")

    original_output_path = training_step.override_output_path
    if original_output_path is None:
        prefix = marin_prefix_for_region(tpu_region)
        executor = Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
        executor.compute_version(training_step, is_pseudo_dep=False)
        original_output_path = executor.output_paths[training_step]

    env_vars = dict(config.env_vars or {})
    env_vars[EVAL_DATASETS_CACHE_DEP_ENV_VAR] = output_path_of(cache_step, ".eval_datasets_manifest.json")
    return replace(training_step, config=replace(config, env_vars=env_vars), override_output_path=original_output_path)


def _configure_training_step(
    training_step: ExecutorStep,
    *,
    tpu_region: str,
    include_eval_harness: bool,
) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    if not include_eval_harness:
        env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, env_vars=env_vars))


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    panel: str,
    limit: int | None,
    shard_count: int,
    shard_index: int,
    include_eval_harness: bool,
    eval_datasets_cache_path: str | None,
) -> SinglePhaseQsplit240LaunchArtifacts:
    """Resolve the launch graph without submitting it."""
    if tpu_region != DEFAULT_TPU_REGION or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This launcher is intentionally pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; "
            f"got {tpu_region}/{tpu_zone}"
        )
    run_specs = build_run_specs(panel=panel, limit=limit, shard_count=shard_count, shard_index=shard_index)
    execution_name_prefix = shard_execution_name_prefix(
        name_prefix=name_prefix,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    experiment = create_qsplit240_replay_experiment(
        name=name_prefix,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type=tpu_type,
        tpu_regions=(tpu_region,),
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    eval_task_aliases = tuple(task.task_alias for task in QSPLIT240_300M_EVAL_TASKS) if include_eval_harness else ()
    run_manifest_step = create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=name_prefix,
        eval_task_aliases=eval_task_aliases,
        run_specs=run_specs,
    )

    cache_eval_datasets_step: ExecutorStep | None = None
    if include_eval_harness:
        resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions((tpu_region,), eval_datasets_cache_path)
        cache_eval_datasets_step = create_cache_eval_datasets_step(
            eval_tasks=QSPLIT240_300M_EVAL_TASKS,
            gcs_path=resolved_eval_cache_path,
            name_prefix=execution_name_prefix,
        )

    training_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
        )
        training_step = _configure_training_step(
            training_step,
            tpu_region=tpu_region,
            include_eval_harness=include_eval_harness,
        )
        if cache_eval_datasets_step is not None:
            training_step = _add_eval_cache_dependency_for_region(
                training_step,
                cache_eval_datasets_step,
                tpu_region=tpu_region,
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
    artifacts = SinglePhaseQsplit240LaunchArtifacts(
        run_specs=run_specs,
        execution_name_prefix=execution_name_prefix,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )
    validate_launch_artifacts(artifacts, tpu_region=tpu_region, include_eval_harness=include_eval_harness)
    return artifacts


def validate_launch_artifacts(
    artifacts: SinglePhaseQsplit240LaunchArtifacts,
    *,
    tpu_region: str,
    include_eval_harness: bool,
) -> None:
    """Validate graph invariants that matter for launch safety."""
    validate_run_specs(artifacts.run_specs)
    if include_eval_harness and artifacts.cache_eval_datasets_step is None:
        raise ValueError("Eval-harness launch is missing cache_eval_datasets_step")
    if not include_eval_harness and artifacts.cache_eval_datasets_step is not None:
        raise ValueError("Perplexity-only launch unexpectedly includes cache_eval_datasets_step")
    if len(artifacts.training_steps) != len(artifacts.run_specs):
        raise ValueError("Training step count does not match run spec count")

    expected_prefix = marin_prefix_for_region(tpu_region)
    seen_step_names: set[str] = set()
    for training_step in artifacts.training_steps:
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != expected_prefix:
            raise ValueError(f"{training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        if training_step.name in seen_step_names:
            raise ValueError(f"Duplicate training step name {training_step.name!r}")
        seen_step_names.add(training_step.name)
        has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
        has_cache_dep = EVAL_DATASETS_CACHE_DEP_ENV_VAR in env_vars
        if include_eval_harness:
            if has_skip:
                raise ValueError(f"{training_step.name} unexpectedly skips eval harness")
            if not has_cache_dep:
                raise ValueError(f"{training_step.name} is missing eval cache dependency")
        else:
            if not has_skip:
                raise ValueError(f"{training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")
            if has_cache_dep:
                raise ValueError(f"{training_step.name} unexpectedly has eval cache dependency")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--panel", default=DEFAULT_PANEL, choices=(REPRESENTATIVE12_PANEL, BASELINES3_PANEL, "all"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-datasets-cache-path")
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run the lm-eval harness during training. Default is in-loop eval/checkpoint only.",
    )
    parser.add_argument("--local-artifact-dir", default=str(DEFAULT_LOCAL_ARTIFACT_DIR))
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        panel=args.panel,
        limit=args.limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        include_eval_harness=args.include_eval_harness,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
    )
    write_local_manifest(artifacts.run_specs, Path(args.local_artifact_dir))

    if args.dry_run or os.getenv("CI") is not None:
        print(
            json.dumps(
                {
                    "name_prefix": args.name_prefix,
                    "execution_name_prefix": artifacts.execution_name_prefix,
                    "run_count": len(artifacts.run_specs),
                    "panel": args.panel,
                    "limit": args.limit,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "include_eval_harness": args.include_eval_harness,
                    "local_artifact_dir": str(Path(args.local_artifact_dir)),
                    "first_run_names": [run_spec.run_name for run_spec in artifacts.run_specs[:8]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    logger.info(
        "Launching %d single-phase qsplit240 300M / 6B runs on %s in %s/%s with max_concurrent=%d.",
        len(artifacts.run_specs),
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=replay_description(
            execution_name_prefix=artifacts.execution_name_prefix,
            label="single-phase exposure-average qsplit240 ablation at 300M / 6B",
        ),
    )


if __name__ == "__main__":
    main()
