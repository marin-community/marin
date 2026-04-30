# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "torch"]
# ///
"""Launch a 60M/1.2B single-phase exposure-average ablation for the fit swarm."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    EXPERIMENT_BUDGET,
    NUM_TRAIN_STEPS,
    PHASE_NAMES,
    REALIZED_EXPERIMENT_BUDGET,
    SEQ_LEN,
    TARGET_BUDGET,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_single_phase_exposure_average_60m_1p2b"
MODEL_FAMILY = "regmix_60m_proxy"
COHORT = "single_phase_exposure_average"
SINGLE_PHASE_STRATEGY = "exposure_average_80_20"
SOURCE_FIT_SWARM_CSV = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "metric_registry"
    / "fit_datasets"
    / "eval_uncheatable_eval_bpb__60m_1p2b__signal__fit_swarm_60m_default.csv"
)
SOURCE_100M_TRANSFER_CSV = (
    Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "qsplit240_300m_6b_completed_vs_60m.csv"
)
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "single_phase_exposure_average_60m_1p2b"
)
LOCAL_MANIFEST_CSV = "single_phase_exposure_average_manifest.csv"
LOCAL_SUMMARY_JSON = "single_phase_exposure_average_summary.json"
LOCAL_RUN_SPECS_JSON = "single_phase_exposure_average_run_specs.json"
PRECOMPUTED_RUN_SPECS_JSON = DEFAULT_LOCAL_ARTIFACT_DIR / LOCAL_RUN_SPECS_JSON
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
OBJECTIVE_VALUE_COLUMN = "objective_metric"
TARGET_BUDGET_MULTIPLIER = 1.0
PHASE_FRACTIONS = {"phase_0": 0.8, "phase_1": 0.2}
EXPECTED_SOURCE_ROWS = 242
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 256
SKIP_EVAL_HARNESS_ENV_VAR = "LEVANTER_SKIP_EVAL_HARNESS"
TOP_RANK_COUNT = 32
HIGH_PHASE_TV_COUNT = 48
DIVERSITY_COUNT = 96


@dataclass(frozen=True)
class SinglePhaseAverageRunSpec:
    """Manifest entry for one single-phase exposure-average ablation run."""

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
    priority_rank: int
    priority_tier: str
    phase_tv: float
    source_60m_bpb: float
    source_60m_rank: int
    source_100m_bpb: float | None
    source_100m_rank: int | None
    rank_shift: int | None
    experiment_budget: int
    realized_experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    target_final_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveSinglePhaseAverageManifestConfig:
    """Config for writing the single-phase ablation manifest."""

    output_path: str
    experiment_name: str
    run_specs_json: str


@dataclass(frozen=True)
class SinglePhaseAverageLaunchArtifacts:
    """Resolved steps for the single-phase exposure-average ablation."""

    run_specs: list[SinglePhaseAverageRunSpec]
    run_manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        return [self.run_manifest_step, *self.training_steps, self.results_step, self.fit_dataset_step]


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _required_phase_columns() -> list[str]:
    return [_phase_column(phase_name, domain_name) for phase_name in PHASE_NAMES for domain_name in DOMAIN_NAMES]


def _normalized_domain_vector(row: pd.Series, phase_name: str) -> np.ndarray:
    values = np.asarray([float(row[_phase_column(phase_name, domain_name)]) for domain_name in DOMAIN_NAMES])
    if np.any(values < 0):
        raise ValueError(f"{row['run_name']} has negative weights in {phase_name}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{row['run_name']} has non-positive weight sum in {phase_name}: {total}")
    return values / total


def _phase_weights_from_vector(weights: np.ndarray) -> dict[str, dict[str, float]]:
    domain_weights = {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, weights, strict=True)}
    return {"phase_0": dict(domain_weights), "phase_1": dict(domain_weights)}


def _load_source_frame() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_FIT_SWARM_CSV)
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ROWS} fit-swarm rows, found {len(frame)} in {SOURCE_FIT_SWARM_CSV}")
    missing_columns = sorted(set(_required_phase_columns()) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{SOURCE_FIT_SWARM_CSV} missing phase-weight columns: {missing_columns[:5]}")
    required = {"run_id", "run_name", "source_experiment", OBJECTIVE_VALUE_COLUMN}
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        raise ValueError(f"{SOURCE_FIT_SWARM_CSV} missing required columns: {missing_required}")
    if frame["run_name"].duplicated().any():
        duplicates = sorted(frame.loc[frame["run_name"].duplicated(), "run_name"].astype(str).unique())
        raise ValueError(f"Duplicate source run names in {SOURCE_FIT_SWARM_CSV}: {duplicates}")

    frame = frame.copy()
    frame["source_60m_bpb"] = pd.to_numeric(frame[OBJECTIVE_VALUE_COLUMN], errors="raise")
    frame["source_60m_rank"] = frame["source_60m_bpb"].rank(method="min", ascending=True).astype(int)
    frame["source_run_id"] = frame["run_id"].astype(int)
    frame["source_run_name"] = frame["run_name"].astype(str)
    frame["source_100m_bpb"] = np.nan
    frame["source_100m_rank"] = np.nan
    frame["rank_shift"] = np.nan

    if SOURCE_100M_TRANSFER_CSV.exists():
        transfer = pd.read_csv(SOURCE_100M_TRANSFER_CSV)
        transfer = transfer.drop_duplicates(subset=["run_name"], keep="last").set_index("run_name")
        for column in ("bpb_300m_6b", "rank_300m_6b", "rank_shift"):
            if column not in transfer.columns:
                raise ValueError(f"{SOURCE_100M_TRANSFER_CSV} missing expected column {column!r}")
        mapped = frame["source_run_name"].map(transfer["bpb_300m_6b"])
        frame["source_100m_bpb"] = pd.to_numeric(mapped, errors="coerce")
        frame["source_100m_rank"] = pd.to_numeric(
            frame["source_run_name"].map(transfer["rank_300m_6b"]), errors="coerce"
        )
        frame["rank_shift"] = pd.to_numeric(frame["source_run_name"].map(transfer["rank_shift"]), errors="coerce")

    single_vectors: list[np.ndarray] = []
    phase_tvs: list[float] = []
    for _, row in frame.iterrows():
        phase_0 = _normalized_domain_vector(row, "phase_0")
        phase_1 = _normalized_domain_vector(row, "phase_1")
        single = PHASE_FRACTIONS["phase_0"] * phase_0 + PHASE_FRACTIONS["phase_1"] * phase_1
        single = single / single.sum()
        single_vectors.append(single)
        phase_tvs.append(float(0.5 * np.abs(phase_1 - phase_0).sum()))

    frame["_single_vector"] = single_vectors
    frame["phase_tv"] = phase_tvs
    return frame


def _append_ordered(order: list[int], tiers: dict[int, str], indices: list[int], tier: str) -> None:
    seen = set(order)
    for index in indices:
        if index in seen:
            continue
        order.append(index)
        tiers[index] = tier
        seen.add(index)


def _top_indices(frame: pd.DataFrame, column: str, count: int, *, ascending: bool = True) -> list[int]:
    subset = frame.loc[pd.to_numeric(frame[column], errors="coerce").notna()].copy()
    if subset.empty:
        return []
    return list(subset.sort_values([column, "source_run_name"], ascending=[ascending, True]).head(count).index)


def _greedy_diversity_indices(
    frame: pd.DataFrame,
    *,
    remaining: set[int],
    selected: list[int],
    count: int,
) -> list[int]:
    if not remaining or count <= 0:
        return []
    vectors = {int(index): np.asarray(vector, dtype=float) for index, vector in frame["_single_vector"].items()}
    selected_vectors = [vectors[index] for index in selected if index in vectors]
    chosen: list[int] = []
    candidates = set(remaining)
    while candidates and len(chosen) < count:
        best_key: tuple[float, float, str] | None = None
        best_index: int | None = None
        for index in candidates:
            vector = vectors[index]
            comparison_vectors = selected_vectors + [vectors[chosen_index] for chosen_index in chosen]
            if comparison_vectors:
                min_distance = min(float(0.5 * np.abs(vector - other).sum()) for other in comparison_vectors)
            else:
                min_distance = 0.0
            source_rank = float(frame.at[index, "source_60m_rank"])
            # Maximize diversity, then prefer better 60M source rank, then stable run name.
            key = (min_distance, -source_rank, str(frame.at[index, "source_run_name"]))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
        candidates.remove(best_index)
    return chosen


def _priority_order(frame: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    order: list[int] = []
    tiers: dict[int, str] = {}
    _append_ordered(order, tiers, _top_indices(frame, "source_60m_rank", TOP_RANK_COUNT), "top_60m")
    _append_ordered(order, tiers, _top_indices(frame, "source_100m_rank", TOP_RANK_COUNT), "top_100m")
    frame = frame.copy()
    frame["abs_rank_shift"] = pd.to_numeric(frame["rank_shift"], errors="coerce").abs()
    _append_ordered(order, tiers, _top_indices(frame, "abs_rank_shift", TOP_RANK_COUNT, ascending=False), "rank_shift")

    remaining = set(frame.index) - set(order)
    high_phase_tv = _top_indices(frame.loc[list(remaining)], "phase_tv", HIGH_PHASE_TV_COUNT, ascending=False)
    _append_ordered(order, tiers, high_phase_tv, "high_phase_tv")

    remaining = set(frame.index) - set(order)
    diversity = _greedy_diversity_indices(frame, remaining=remaining, selected=order, count=DIVERSITY_COUNT)
    _append_ordered(order, tiers, diversity, "diversity")

    remaining_rows = frame.loc[sorted(set(frame.index) - set(order))]
    remaining_indices = list(remaining_rows.sort_values(["source_60m_rank", "source_run_name"]).index)
    _append_ordered(order, tiers, remaining_indices, "remaining")
    if len(order) != len(frame):
        raise ValueError(f"Priority ordering covered {len(order)} rows, expected {len(frame)}")
    return order, tiers


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _maybe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _ablation_run_name(source_run_name: str) -> str:
    return f"singleavg_{source_run_name}"


def build_run_specs() -> list[SinglePhaseAverageRunSpec]:
    """Build the full 242-row single-phase exposure-average manifest."""
    if not SOURCE_FIT_SWARM_CSV.exists() and PRECOMPUTED_RUN_SPECS_JSON.exists():
        payload = json.loads(PRECOMPUTED_RUN_SPECS_JSON.read_text())
        specs = [SinglePhaseAverageRunSpec(**row) for row in payload["runs"]]
        validate_run_specs(specs)
        return specs

    frame = _load_source_frame()
    order, tiers = _priority_order(frame)
    specs: list[SinglePhaseAverageRunSpec] = []
    for priority_rank, source_index in enumerate(order):
        row = frame.loc[source_index]
        source_run_name = str(row["source_run_name"])
        source_run_id = int(row["source_run_id"])
        single_vector = np.asarray(row["_single_vector"], dtype=float)
        specs.append(
            SinglePhaseAverageRunSpec(
                run_id=priority_rank,
                run_name=_ablation_run_name(source_run_name),
                cohort=COHORT,
                model_family=MODEL_FAMILY,
                trainer_seed=None,
                data_seed=source_run_id,
                simulated_epoch_subset_seed=None,
                source_run_id=source_run_id,
                source_run_name=source_run_name,
                source_two_phase_experiment=str(row["source_experiment"]),
                candidate_run_id=source_run_id,
                candidate_run_name=source_run_name,
                candidate_source_experiment=str(row["source_experiment"]),
                single_phase_strategy=SINGLE_PHASE_STRATEGY,
                priority_rank=priority_rank,
                priority_tier=tiers[source_index],
                phase_tv=float(row["phase_tv"]),
                source_60m_bpb=float(row["source_60m_bpb"]),
                source_60m_rank=int(row["source_60m_rank"]),
                source_100m_bpb=_maybe_float(row["source_100m_bpb"]),
                source_100m_rank=_maybe_int(row["source_100m_rank"]),
                rank_shift=_maybe_int(row["rank_shift"]),
                experiment_budget=EXPERIMENT_BUDGET,
                realized_experiment_budget=REALIZED_EXPERIMENT_BUDGET,
                target_budget=TARGET_BUDGET,
                target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
                num_train_steps=NUM_TRAIN_STEPS,
                target_final_checkpoint_step=NUM_TRAIN_STEPS - 1,
                phase_weights=_phase_weights_from_vector(single_vector),
            )
        )
    validate_run_specs(specs)
    return specs


def validate_run_specs(run_specs: list[SinglePhaseAverageRunSpec]) -> None:
    """Validate manifest invariants before launch."""
    if len(run_specs) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ROWS} run specs, got {len(run_specs)}")
    run_names = [spec.run_name for spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate single-phase ablation run names")
    priority_ranks = [spec.priority_rank for spec in run_specs]
    if priority_ranks != list(range(len(run_specs))):
        raise ValueError("Priority ranks must be contiguous and match manifest order")
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


def _flat_manifest_row(spec: SinglePhaseAverageRunSpec) -> dict[str, Any]:
    row = asdict(spec)
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[_phase_column(phase_name, domain_name)] = value
    return row


def write_local_manifest(run_specs: list[SinglePhaseAverageRunSpec], output_dir: Path) -> None:
    """Write local manifest and summary artifacts for audit and launch handoff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flat_manifest_row(spec) for spec in run_specs]
    pd.DataFrame.from_records(rows).to_csv(output_dir / LOCAL_MANIFEST_CSV, index=False)
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "experiment_name": NAME,
                "single_phase_strategy": SINGLE_PHASE_STRATEGY,
                "runs": [asdict(spec) for spec in run_specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tier_counts = pd.Series([spec.priority_tier for spec in run_specs]).value_counts().sort_index().to_dict()
    summary = {
        "experiment_name": NAME,
        "source_fit_swarm_csv": str(SOURCE_FIT_SWARM_CSV),
        "source_100m_transfer_csv": str(SOURCE_100M_TRANSFER_CSV),
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "row_count": len(run_specs),
        "model_family": MODEL_FAMILY,
        "experiment_budget": EXPERIMENT_BUDGET,
        "realized_experiment_budget": REALIZED_EXPERIMENT_BUDGET,
        "target_budget": TARGET_BUDGET,
        "target_budget_multiplier": TARGET_BUDGET_MULTIPLIER,
        "num_train_steps": NUM_TRAIN_STEPS,
        "target_final_checkpoint_step": NUM_TRAIN_STEPS - 1,
        "priority_tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "first_32_run_names": [spec.run_name for spec in run_specs[:32]],
    }
    (output_dir / LOCAL_SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def save_run_manifest(config: SaveSinglePhaseAverageManifestConfig) -> None:
    """Persist the ablation manifest for downstream collection and fitting."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": len(run_specs),
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "runs": run_specs,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_specs: list[SinglePhaseAverageRunSpec]) -> ExecutorStep:
    """Create the manifest writer step for the single-phase ablation."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save single-phase exposure-average ablation manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveSinglePhaseAverageManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _configure_training_env_for_step(
    training_step: ExecutorStep,
    *,
    tpu_region: str,
    include_eval_harness: bool,
    child_job_name: str,
) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    if not include_eval_harness:
        env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, env_vars=env_vars, job_name=child_job_name))


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    include_eval_harness: bool,
) -> SinglePhaseAverageLaunchArtifacts:
    """Resolve the single-phase exposure-average launch graph without submitting it."""
    if tpu_region != DEFAULT_TPU_REGION or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This launcher is intentionally pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; "
            f"got {tpu_region}/{tpu_zone}"
        )
    run_specs = build_run_specs()
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_60m_proxy,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        runtime_cache_region=tpu_region,
    )
    run_manifest_step = create_run_manifest_step(name_prefix=name_prefix, run_specs=run_specs)
    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=name_prefix,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
            simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
        )
        training_step = _configure_training_env_for_step(
            training_step,
            tpu_region=tpu_region,
            include_eval_harness=include_eval_harness,
            child_job_name=f"train_lm_{spec.run_name}",
        )
        training_steps.append(training_step)

    results_step = create_manifest_results_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )
    return SinglePhaseAverageLaunchArtifacts(
        run_specs=run_specs,
        run_manifest_step=run_manifest_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )


def _has_iris_context() -> bool:
    try:
        from iris.client.client import get_iris_ctx
    except ImportError:
        return False
    return get_iris_ctx() is not None


def _executor_prefix(executor_prefix: str | None, default_tpu_region: str) -> str | None:
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_tpu_region), executor_prefix)


def _validate_training_graph(artifacts: SinglePhaseAverageLaunchArtifacts, *, include_eval_harness: bool) -> None:
    for training_step in artifacts.training_steps:
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        actual_num_train_steps = int(config.train_config.trainer.num_train_steps)
        if actual_num_train_steps != NUM_TRAIN_STEPS:
            raise ValueError(f"{training_step.name} has num_train_steps={actual_num_train_steps}")
        env_vars = dict(config.env_vars or {})
        has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(DEFAULT_TPU_REGION):
            raise ValueError(f"{training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        if config.job_name == "train_lm":
            raise ValueError(f"{training_step.name} has non-unique child job name {config.job_name!r}")
        if include_eval_harness and has_skip:
            raise ValueError(f"{training_step.name} unexpectedly skips eval harness")
        if not include_eval_harness and not has_skip:
            raise ValueError(f"{training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--local-artifact-dir", default=str(DEFAULT_LOCAL_ARTIFACT_DIR))
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run Levanter lm-eval harness during training. Default is perplexity/checkpoint only.",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ.setdefault("MARIN_PREFIX", marin_prefix_for_region(args.tpu_region))
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        include_eval_harness=args.include_eval_harness,
    )
    _validate_training_graph(artifacts, include_eval_harness=args.include_eval_harness)
    write_local_manifest(artifacts.run_specs, Path(args.local_artifact_dir))
    logger.info("Wrote local manifest to %s/%s", args.local_artifact_dir, LOCAL_MANIFEST_CSV)
    logger.info("Prepared %d single-phase exposure-average runs.", len(artifacts.run_specs))
    logger.info("First 16 runs by priority: %s", ", ".join(spec.run_name for spec in artifacts.run_specs[:16]))
    logger.info(
        "Launch config: tpu=%s region=%s zone=%s max_concurrent=%d eval_harness=%s",
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
        "enabled" if args.include_eval_harness else "skipped",
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.name_prefix}: 242-row 60M/1.2B single-phase exposure-average ablation. "
            f"Outputs include {RUN_MANIFEST_FILE}, {RESULTS_CSV}, {FIT_DATASET_CSV}, and "
            f"{FIT_DATASET_SUMMARY_JSON}."
        ),
    )


if __name__ == "__main__":
    main()
