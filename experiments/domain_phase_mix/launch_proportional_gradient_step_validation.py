# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 8 proportional local-gradient validation runs.

This launcher validates whether one finite-difference gradient step around the
proportional mixture improves target-scale BPB. It uses four 60M-derived
candidates at 60M/1.2B and four 100M-derived candidates at historical
300m_6b (displayed as corrected 100M/6B). All candidates are phase-constant
(`phase_0 == phase_1`) so the experiment isolates mixture movement rather than
phase scheduling.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fsspec
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    EVAL_DATASETS_CACHE_DEP_ENV_VAR,
    SKIP_EVAL_HARNESS_ENV_VAR,
    add_eval_cache_dependency_to_training_step,
    create_cache_eval_datasets_step,
    create_qsplit240_replay_experiment,
    create_run_manifest_step,
    normalize_tpu_regions,
    resolve_qsplit240_eval_cache_path_for_regions,
    skip_eval_harness_for_training_step,
)
from experiments.domain_phase_mix.scaling_study_recipes import ScalingStudyScale, resolve_scale_spec
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES, PHASE_NAMES
from experiments.domain_phase_mix.two_phase_many_observed_runs import ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT

logger = logging.getLogger(__name__)

BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_gradient_step_validation"
COHORT = "proportional_gradient_step_validation_domain_only"
BASE_RUN_NAME = "baseline_proportional"
BASE_RUN_ID = 0
RUN_ID_BASE = 790_000
TARGET_BUDGET_MULTIPLIER = 1.0
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_gradient_step_validation_20260510"
)
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_perturbation_scale_transfer_20260507"
    / "gradient_step_candidates_domain_only"
)
DEFAULT_CANDIDATE_WEIGHTS_URI = str(DEFAULT_CANDIDATE_DIR / "candidate_weights.csv")
DEFAULT_CANDIDATE_SUMMARY_URI = str(DEFAULT_CANDIDATE_DIR / "candidate_summary_with_grp_no_l2.csv")
LOCAL_CANDIDATE_SOURCE_SUMMARY_CSV = "candidate_source_summary.csv"
LOCAL_TRAINING_MANIFEST_CSV = "training_manifest.csv"
LOCAL_RUN_SPECS_JSON = "run_specs.json"
LOCAL_SUMMARY_JSON = "summary.json"
SCALES = (ScalingStudyScale.REGMIX_60M_1P2B, ScalingStudyScale.REGMIX_300M_6B)
DISPLAY_LABELS = {
    ScalingStudyScale.REGMIX_60M_1P2B: "60M/1.2B",
    ScalingStudyScale.REGMIX_300M_6B: "100M/6B",
}
EXPECTED_FINAL_CHECKPOINT_STEPS = {
    ScalingStudyScale.REGMIX_60M_1P2B: 4576,
    ScalingStudyScale.REGMIX_300M_6B: 22887,
}
REQUIRED_CANDIDATES: tuple[tuple[str, str, ScalingStudyScale], ...] = (
    ("60m_observed_good_all_unscaled", "pgrad_60m_good_all_unscaled", ScalingStudyScale.REGMIX_60M_1P2B),
    ("60m_observed_good_all_tv0.050", "pgrad_60m_good_all_tv005", ScalingStudyScale.REGMIX_60M_1P2B),
    ("60m_balanced_all_unscaled", "pgrad_60m_balanced_all_unscaled", ScalingStudyScale.REGMIX_60M_1P2B),
    ("60m_balanced_all_tv0.050", "pgrad_60m_balanced_all_tv005", ScalingStudyScale.REGMIX_60M_1P2B),
    ("100m_observed_good_all_unscaled", "pgrad_100m_good_all_unscaled", ScalingStudyScale.REGMIX_300M_6B),
    ("100m_observed_good_all_tv0.050", "pgrad_100m_good_all_tv005", ScalingStudyScale.REGMIX_300M_6B),
    ("100m_balanced_all_unscaled", "pgrad_100m_balanced_all_unscaled", ScalingStudyScale.REGMIX_300M_6B),
    ("100m_balanced_all_tv0.050", "pgrad_100m_balanced_all_tv005", ScalingStudyScale.REGMIX_300M_6B),
)


@dataclass(frozen=True)
class CandidateSpec:
    """One frozen gradient-step candidate before training-scale launch."""

    candidate_index: int
    candidate_id: str
    run_id: int
    run_name: str
    gradient_source_scale: str
    gradient_construction: str
    radius_policy: str
    target_tv: float | None
    actual_tv: float
    entropy: float | None
    support_gt_0p001: int | None
    max_domain: str | None
    max_weight: float | None
    min_weight: float | None
    top_weights: str | None
    bottom_weights: str | None
    notes: str | None
    predicted_60m_bpb_effect: float
    predicted_100m_bpb_effect: float
    predicted_scale_interaction_bpb: float
    grp_no_l2_predicted_bpb: float | None
    grp_no_l2_proportional_predicted_bpb: float | None
    grp_no_l2_proportional_actual_bpb: float | None
    grp_no_l2_predicted_delta_vs_proportional: float | None
    grp_no_l2_rank: int | None
    training_scale: str
    scale_display_label: str
    phase_mode: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class GradientValidationRunSpec:
    """Manifest entry for one validation training run."""

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
    gradient_candidate_id: str
    gradient_source_scale: str
    gradient_construction: str
    radius_policy: str
    target_tv: float | None
    actual_tv: float
    entropy: float | None
    support_gt_0p001: int | None
    max_domain: str | None
    max_weight: float | None
    min_weight: float | None
    top_weights: str | None
    bottom_weights: str | None
    notes: str | None
    predicted_60m_bpb_effect: float
    predicted_100m_bpb_effect: float
    predicted_scale_interaction_bpb: float
    grp_no_l2_predicted_bpb: float | None
    grp_no_l2_proportional_predicted_bpb: float | None
    grp_no_l2_proportional_actual_bpb: float | None
    grp_no_l2_predicted_delta_vs_proportional: float | None
    grp_no_l2_rank: int | None
    phase_mode: str
    scale: str
    scale_display_label: str
    experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    target_final_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ScaleLaunchArtifacts:
    """Resolved launch graph for one scale."""

    scale: ScalingStudyScale
    name_prefix: str
    run_specs: list[GradientValidationRunSpec]
    run_manifest_step: ExecutorStep
    cache_eval_datasets_step: ExecutorStep
    training_steps: list[ExecutorStep]
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        return [
            self.run_manifest_step,
            self.cache_eval_datasets_step,
            *self.training_steps,
            self.results_step,
            self.fit_dataset_step,
        ]


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved full launch graph."""

    candidate_specs: list[CandidateSpec]
    scale_artifacts: list[ScaleLaunchArtifacts]
    candidate_weights_uri: str
    candidate_summary_uri: str

    @property
    def training_steps(self) -> list[ExecutorStep]:
        return [step for artifact in self.scale_artifacts for step in artifact.training_steps]

    @property
    def steps(self) -> list[object]:
        return [step for artifact in self.scale_artifacts for step in artifact.steps]


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _read_csv_rows(uri: str) -> list[dict[str, str]]:
    with fsspec.open(uri, "rt") as f:
        return list(csv.DictReader(f))


def _required_float(row: dict[str, str], key: str, *, row_name: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"{row_name} missing required float column {key!r}")
    return float(value)


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_str(row: dict[str, str], key: str) -> str | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return value


def _radius_policy(candidate_id: str) -> str:
    if candidate_id.endswith("_unscaled"):
        return "unscaled"
    if candidate_id.endswith("_tv0.050"):
        return "tv0.050"
    raise ValueError(f"Cannot infer radius policy for {candidate_id!r}")


def _expected_source_scale(scale: ScalingStudyScale) -> str:
    if scale == ScalingStudyScale.REGMIX_60M_1P2B:
        return "60m"
    if scale == ScalingStudyScale.REGMIX_300M_6B:
        return "100m"
    raise ValueError(f"Unsupported scale {scale}")


def _validate_domain_weights(weights: dict[str, float], *, label: str) -> None:
    if set(weights) != set(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES) - set(weights))
        extra = sorted(set(weights) - set(DOMAIN_NAMES))
        raise ValueError(f"{label} domain mismatch: missing={missing}, extra={extra}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} weights sum to {total}, expected 1.0")
    negative = {domain_name: value for domain_name, value in weights.items() if value < -1e-15}
    if negative:
        raise ValueError(f"{label} has negative weights: {negative}")


def _validate_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names do not match {PHASE_NAMES}")
    for phase_name, weights in phase_weights.items():
        _validate_domain_weights(weights, label=f"{run_name}/{phase_name}")
    phase_0 = phase_weights["phase_0"]
    phase_1 = phase_weights["phase_1"]
    max_phase_delta = max(abs(phase_0[domain_name] - phase_1[domain_name]) for domain_name in DOMAIN_NAMES)
    if max_phase_delta > 1e-15:
        raise ValueError(f"{run_name} is not phase-constant: max phase delta {max_phase_delta}")


def _phase_weights_from_row(row: dict[str, str], *, candidate_id: str) -> dict[str, dict[str, float]]:
    phase_weights: dict[str, dict[str, float]] = {}
    for phase_name in PHASE_NAMES:
        phase_weights[phase_name] = {}
        for domain_name in DOMAIN_NAMES:
            phase_weights[phase_name][domain_name] = _required_float(
                row,
                _phase_column(phase_name, domain_name),
                row_name=candidate_id,
            )
    _validate_phase_weights(phase_weights, run_name=candidate_id)
    return phase_weights


def _load_candidate_specs(
    *,
    candidate_weights_uri: str,
    candidate_summary_uri: str,
) -> list[CandidateSpec]:
    weight_rows = {row["candidate_id"]: row for row in _read_csv_rows(candidate_weights_uri)}
    summary_rows = {row["candidate_id"]: row for row in _read_csv_rows(candidate_summary_uri)}
    specs: list[CandidateSpec] = []
    for candidate_index, (candidate_id, run_name, training_scale) in enumerate(REQUIRED_CANDIDATES):
        if candidate_id not in weight_rows:
            raise ValueError(f"Missing {candidate_id} in {candidate_weights_uri}")
        if candidate_id not in summary_rows:
            raise ValueError(f"Missing {candidate_id} in {candidate_summary_uri}")
        weight_row = weight_rows[candidate_id]
        summary_row = summary_rows[candidate_id]
        gradient_source_scale = summary_row["source_scale"]
        expected_source_scale = _expected_source_scale(training_scale)
        if gradient_source_scale != expected_source_scale:
            raise ValueError(
                f"{candidate_id} has source_scale={gradient_source_scale!r}, expected {expected_source_scale!r}"
            )
        if weight_row.get("source_scale") != gradient_source_scale:
            raise ValueError(f"{candidate_id} source_scale disagrees between weights and summary")
        phase_weights = _phase_weights_from_row(weight_row, candidate_id=candidate_id)
        phase_mode = weight_row.get("phase_mode")
        if phase_mode != "both_phases":
            raise ValueError(f"{candidate_id} has phase_mode={phase_mode!r}")
        specs.append(
            CandidateSpec(
                candidate_index=candidate_index,
                candidate_id=candidate_id,
                run_id=RUN_ID_BASE + candidate_index,
                run_name=run_name,
                gradient_source_scale=gradient_source_scale,
                gradient_construction=summary_row["construction"],
                radius_policy=_radius_policy(candidate_id),
                target_tv=_optional_float(summary_row, "target_tv"),
                actual_tv=_required_float(summary_row, "actual_tv", row_name=candidate_id),
                entropy=_optional_float(summary_row, "entropy"),
                support_gt_0p001=_optional_int(summary_row, "support_gt_0p001"),
                max_domain=_optional_str(summary_row, "max_domain"),
                max_weight=_optional_float(summary_row, "max_weight"),
                min_weight=_optional_float(summary_row, "min_weight"),
                top_weights=_optional_str(summary_row, "top_weights"),
                bottom_weights=_optional_str(summary_row, "bottom_weights"),
                notes=_optional_str(summary_row, "notes"),
                predicted_60m_bpb_effect=_required_float(
                    summary_row,
                    "predicted_60m_bpb_effect",
                    row_name=candidate_id,
                ),
                predicted_100m_bpb_effect=_required_float(
                    summary_row,
                    "predicted_100m_bpb_effect",
                    row_name=candidate_id,
                ),
                predicted_scale_interaction_bpb=_required_float(
                    summary_row,
                    "predicted_scale_interaction_bpb",
                    row_name=candidate_id,
                ),
                grp_no_l2_predicted_bpb=_optional_float(summary_row, "grp_no_l2_predicted_bpb"),
                grp_no_l2_proportional_predicted_bpb=_optional_float(
                    summary_row,
                    "grp_no_l2_proportional_predicted_bpb",
                ),
                grp_no_l2_proportional_actual_bpb=_optional_float(
                    summary_row,
                    "grp_no_l2_proportional_actual_bpb",
                ),
                grp_no_l2_predicted_delta_vs_proportional=_optional_float(
                    summary_row,
                    "grp_no_l2_predicted_delta_vs_proportional",
                ),
                grp_no_l2_rank=_optional_int(summary_row, "grp_no_l2_rank"),
                training_scale=training_scale.value,
                scale_display_label=DISPLAY_LABELS[training_scale],
                phase_mode=phase_mode,
                phase_weights=phase_weights,
            )
        )
    validate_candidate_specs(specs)
    return specs


def validate_candidate_specs(specs: list[CandidateSpec]) -> None:
    """Validate candidate-level invariants before creating training steps."""
    if len(specs) != 8:
        raise ValueError(f"Expected 8 candidate specs, got {len(specs)}")
    if [spec.run_id for spec in specs] != list(range(RUN_ID_BASE, RUN_ID_BASE + 8)):
        raise ValueError("Run IDs are not contiguous")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise ValueError("Duplicate run names")
    if len({spec.candidate_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate candidate IDs")
    required_ids = [candidate_id for candidate_id, _, _ in REQUIRED_CANDIDATES]
    if [spec.candidate_id for spec in specs] != required_ids:
        raise ValueError("Candidate ordering does not match REQUIRED_CANDIDATES")
    scale_counts = {scale.value: sum(1 for spec in specs if spec.training_scale == scale.value) for scale in SCALES}
    if scale_counts != {
        ScalingStudyScale.REGMIX_60M_1P2B.value: 4,
        ScalingStudyScale.REGMIX_300M_6B.value: 4,
    }:
        raise ValueError(f"Unexpected scale counts: {scale_counts}")
    for spec in specs:
        _validate_phase_weights(spec.phase_weights, run_name=spec.candidate_id)
        if spec.gradient_source_scale not in {"60m", "100m"}:
            raise ValueError(f"{spec.candidate_id} has unexpected source scale {spec.gradient_source_scale}")
        if spec.radius_policy == "tv0.050" and not math.isclose(spec.actual_tv, 0.05, abs_tol=1e-12):
            raise ValueError(f"{spec.candidate_id} actual_tv={spec.actual_tv}, expected 0.05")


def _run_spec_for_candidate(candidate: CandidateSpec) -> GradientValidationRunSpec:
    scale = ScalingStudyScale(candidate.training_scale)
    scale_spec = resolve_scale_spec(scale)
    num_train_steps = scale_spec.num_train_steps_for_multiplier(TARGET_BUDGET_MULTIPLIER)
    expected_final_step = EXPECTED_FINAL_CHECKPOINT_STEPS[scale]
    if num_train_steps - 1 != expected_final_step:
        raise ValueError(f"{scale.value} final checkpoint step={num_train_steps - 1}, expected {expected_final_step}")
    return GradientValidationRunSpec(
        run_id=candidate.run_id,
        run_name=candidate.run_name,
        cohort=COHORT,
        model_family=scale_spec.model_family,
        trainer_seed=None,
        data_seed=candidate.run_id,
        simulated_epoch_subset_seed=None,
        source_run_id=BASE_RUN_ID,
        source_run_name=BASE_RUN_NAME,
        source_two_phase_experiment=ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        candidate_run_id=candidate.run_id,
        candidate_run_name=candidate.run_name,
        candidate_source_experiment=BASE_NAME_PREFIX,
        gradient_candidate_id=candidate.candidate_id,
        gradient_source_scale=candidate.gradient_source_scale,
        gradient_construction=candidate.gradient_construction,
        radius_policy=candidate.radius_policy,
        target_tv=candidate.target_tv,
        actual_tv=candidate.actual_tv,
        entropy=candidate.entropy,
        support_gt_0p001=candidate.support_gt_0p001,
        max_domain=candidate.max_domain,
        max_weight=candidate.max_weight,
        min_weight=candidate.min_weight,
        top_weights=candidate.top_weights,
        bottom_weights=candidate.bottom_weights,
        notes=candidate.notes,
        predicted_60m_bpb_effect=candidate.predicted_60m_bpb_effect,
        predicted_100m_bpb_effect=candidate.predicted_100m_bpb_effect,
        predicted_scale_interaction_bpb=candidate.predicted_scale_interaction_bpb,
        grp_no_l2_predicted_bpb=candidate.grp_no_l2_predicted_bpb,
        grp_no_l2_proportional_predicted_bpb=candidate.grp_no_l2_proportional_predicted_bpb,
        grp_no_l2_proportional_actual_bpb=candidate.grp_no_l2_proportional_actual_bpb,
        grp_no_l2_predicted_delta_vs_proportional=candidate.grp_no_l2_predicted_delta_vs_proportional,
        grp_no_l2_rank=candidate.grp_no_l2_rank,
        phase_mode=candidate.phase_mode,
        scale=scale.value,
        scale_display_label=DISPLAY_LABELS[scale],
        experiment_budget=scale_spec.experiment_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget=scale_spec.target_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=num_train_steps,
        target_final_checkpoint_step=num_train_steps - 1,
        phase_weights=candidate.phase_weights,
    )


def _run_manifest_step(
    *,
    execution_name_prefix: str,
    experiment_name: str,
    run_specs: list[GradientValidationRunSpec],
) -> ExecutorStep:
    scale = ScalingStudyScale(run_specs[0].scale)
    scale_spec = resolve_scale_spec(scale)
    return create_run_manifest_step(
        step_name_prefix=execution_name_prefix,
        experiment_name=experiment_name,
        model_family=scale_spec.model_family,
        experiment_budget=run_specs[0].experiment_budget,
        target_budget=run_specs[0].target_budget,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=run_specs[0].num_train_steps,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        run_specs=run_specs,  # type: ignore[arg-type]
    )


def _configure_training_step(
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


def _scale_name_prefix(base_name_prefix: str, scale: ScalingStudyScale) -> str:
    return f"{base_name_prefix}_{scale.value}"


def _build_scale_launch_artifacts(
    *,
    base_name_prefix: str,
    scale: ScalingStudyScale,
    all_run_specs: list[GradientValidationRunSpec],
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
) -> ScaleLaunchArtifacts:
    if tpu_regions != (DEFAULT_TPU_REGION,) or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This experiment is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; got {tpu_regions}/{tpu_zone}"
        )
    scale_spec = resolve_scale_spec(scale)
    name_prefix = _scale_name_prefix(base_name_prefix, scale)
    run_specs = [run_spec for run_spec in all_run_specs if run_spec.scale == scale.value]
    if len(run_specs) != 4:
        raise ValueError(f"{scale.value} has {len(run_specs)} run specs, expected 4")
    experiment = create_qsplit240_replay_experiment(
        name=name_prefix,
        experiment_budget=run_specs[0].experiment_budget,
        target_budget=run_specs[0].target_budget,
        batch_size=scale_spec.batch_size,
        seq_len=scale_spec.seq_len,
        model_config=scale_spec.model_config,
        optimizer_config=scale_spec.optimizer_config,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(
        tpu_regions,
        eval_datasets_cache_path,
    )
    run_manifest_step = _run_manifest_step(
        execution_name_prefix=name_prefix,
        experiment_name=name_prefix,
        run_specs=run_specs,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=resolved_eval_cache_path,
        name_prefix=name_prefix,
    )
    training_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
            simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        )
        training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        training_step = _configure_training_step(
            training_step,
            tpu_region=tpu_regions[0],
            include_eval_harness=include_eval_harness,
            child_job_name=f"train_lm_{scale.value}_{run_spec.run_name}",
        )
        if not include_eval_harness:
            training_step = skip_eval_harness_for_training_step(training_step)
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
    return ScaleLaunchArtifacts(
        scale=scale,
        name_prefix=name_prefix,
        run_specs=run_specs,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )


def build_launch_artifacts(
    *,
    base_name_prefix: str,
    candidate_weights_uri: str,
    candidate_summary_uri: str,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
) -> LaunchArtifacts:
    """Resolve the full 8-run launch graph without submitting."""
    candidate_specs = _load_candidate_specs(
        candidate_weights_uri=candidate_weights_uri,
        candidate_summary_uri=candidate_summary_uri,
    )
    run_specs = [_run_spec_for_candidate(candidate) for candidate in candidate_specs]
    scale_artifacts = [
        _build_scale_launch_artifacts(
            base_name_prefix=base_name_prefix,
            scale=scale,
            all_run_specs=run_specs,
            tpu_type=tpu_type,
            tpu_regions=tpu_regions,
            tpu_zone=tpu_zone,
            eval_datasets_cache_path=eval_datasets_cache_path,
            include_eval_harness=include_eval_harness,
        )
        for scale in SCALES
    ]
    artifacts = LaunchArtifacts(
        candidate_specs=candidate_specs,
        scale_artifacts=scale_artifacts,
        candidate_weights_uri=candidate_weights_uri,
        candidate_summary_uri=candidate_summary_uri,
    )
    validate_launch_artifacts(artifacts, include_eval_harness=include_eval_harness)
    return artifacts


def validate_launch_artifacts(
    artifacts: LaunchArtifacts,
    *,
    include_eval_harness: bool,
) -> None:
    """Validate graph invariants before launch."""
    validate_candidate_specs(artifacts.candidate_specs)
    if len(artifacts.scale_artifacts) != 2:
        raise ValueError(f"Expected 2 scale artifacts, got {len(artifacts.scale_artifacts)}")
    if len(artifacts.training_steps) != 8:
        raise ValueError(f"Expected 8 training steps, got {len(artifacts.training_steps)}")
    all_training_names = [step.name for step in artifacts.training_steps]
    if len(set(all_training_names)) != len(all_training_names):
        raise ValueError("Duplicate training step names")
    all_output_roots = [str(step.override_output_path or step.name) for step in artifacts.training_steps]
    if len(set(all_output_roots)) != len(all_output_roots):
        raise ValueError("Duplicate training output paths")
    for artifact in artifacts.scale_artifacts:
        expected_final_step = EXPECTED_FINAL_CHECKPOINT_STEPS[artifact.scale]
        if len(artifact.run_specs) != 4:
            raise ValueError(f"{artifact.scale.value} has {len(artifact.run_specs)} specs, expected 4")
        run_names = [run_spec.run_name for run_spec in artifact.run_specs]
        if len(set(run_names)) != len(run_names):
            raise ValueError(f"{artifact.scale.value} has duplicate run names")
        for run_spec in artifact.run_specs:
            _validate_phase_weights(run_spec.phase_weights, run_name=f"{artifact.scale.value}/{run_spec.run_name}")
            expected_source_scale = _expected_source_scale(artifact.scale)
            if run_spec.gradient_source_scale != expected_source_scale:
                raise ValueError(f"{run_spec.run_name} has wrong gradient source scale")
            if run_spec.num_train_steps - 1 != expected_final_step:
                raise ValueError(f"{run_spec.run_name} has num_train_steps={run_spec.num_train_steps}")
            if run_spec.target_final_checkpoint_step != expected_final_step:
                raise ValueError(
                    f"{run_spec.run_name} has target_final_checkpoint_step={run_spec.target_final_checkpoint_step}"
                )
    for training_step in artifacts.training_steps:
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(DEFAULT_TPU_REGION):
            raise ValueError(f"{training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        if env_vars.get(EVAL_DATASETS_CACHE_DEP_ENV_VAR) is None:
            raise ValueError(f"{training_step.name} missing eval cache dependency env var")
        has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
        if include_eval_harness and has_skip:
            raise ValueError(f"{training_step.name} unexpectedly skips eval harness")
        if not include_eval_harness and not has_skip:
            raise ValueError(f"{training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")
        if config.job_name == "train_lm":
            raise ValueError(f"{training_step.name} has non-unique child job name {config.job_name!r}")
        actual_num_train_steps = int(config.train_config.trainer.num_train_steps)
        matching_specs = [
            run_spec
            for artifact in artifacts.scale_artifacts
            for run_spec in artifact.run_specs
            if training_step.name.endswith(f"{artifact.name_prefix}/{run_spec.run_name}")
        ]
        if len(matching_specs) != 1:
            raise ValueError(f"Could not resolve matching spec for {training_step.name}")
        if actual_num_train_steps != matching_specs[0].num_train_steps:
            raise ValueError(f"{training_step.name} has num_train_steps={actual_num_train_steps}")
        hf_save_steps = config.train_config.hf_save_steps
        if hf_save_steps is None:
            raise ValueError(f"{training_step.name} has HF export disabled")
        if int(hf_save_steps) != matching_specs[0].num_train_steps:
            raise ValueError(f"{training_step.name} does not export HF checkpoint at final step")


def _flat_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[_phase_column(phase_name, domain_name)] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_local_manifests(artifacts: LaunchArtifacts, output_dir: Path) -> None:
    """Write local audit artifacts for the launch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [_flat_manifest_row(asdict(spec)) for spec in artifacts.candidate_specs]
    training_rows = [
        _flat_manifest_row(asdict(run_spec)) for artifact in artifacts.scale_artifacts for run_spec in artifact.run_specs
    ]
    _write_csv(output_dir / LOCAL_CANDIDATE_SOURCE_SUMMARY_CSV, candidate_rows)
    _write_csv(output_dir / LOCAL_TRAINING_MANIFEST_CSV, training_rows)
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "base_name_prefix": BASE_NAME_PREFIX,
                "cohort": COHORT,
                "candidate_weights_uri": artifacts.candidate_weights_uri,
                "candidate_summary_uri": artifacts.candidate_summary_uri,
                "candidates": [asdict(spec) for spec in artifacts.candidate_specs],
                "scales": {
                    artifact.scale.value: {
                        "name_prefix": artifact.name_prefix,
                        "run_specs": [asdict(run_spec) for run_spec in artifact.run_specs],
                    }
                    for artifact in artifacts.scale_artifacts
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "base_name_prefix": BASE_NAME_PREFIX,
        "cohort": COHORT,
        "base_run_name": BASE_RUN_NAME,
        "base_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        "candidate_weights_uri": artifacts.candidate_weights_uri,
        "candidate_summary_uri": artifacts.candidate_summary_uri,
        "candidate_count": len(artifacts.candidate_specs),
        "training_run_count": len(training_rows),
        "scale_prefixes": {artifact.scale.value: artifact.name_prefix for artifact in artifacts.scale_artifacts},
        "scale_display_labels": {scale.value: label for scale, label in DISPLAY_LABELS.items()},
        "radius_policy_counts": {
            radius_policy: sum(1 for spec in artifacts.candidate_specs if spec.radius_policy == radius_policy)
            for radius_policy in sorted({spec.radius_policy for spec in artifacts.candidate_specs})
        },
        "construction_counts": {
            construction: sum(1 for spec in artifacts.candidate_specs if spec.gradient_construction == construction)
            for construction in sorted({spec.gradient_construction for spec in artifacts.candidate_specs})
        },
        "outputs": {
            "candidate_source_summary_csv": LOCAL_CANDIDATE_SOURCE_SUMMARY_CSV,
            "training_manifest_csv": LOCAL_TRAINING_MANIFEST_CSV,
            "run_specs_json": LOCAL_RUN_SPECS_JSON,
        },
    }
    (output_dir / LOCAL_SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name-prefix", default=BASE_NAME_PREFIX)
    parser.add_argument("--candidate-weights-uri", default=DEFAULT_CANDIDATE_WEIGHTS_URI)
    parser.add_argument("--candidate-summary-uri", default=DEFAULT_CANDIDATE_SUMMARY_URI)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-regions")
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-datasets-cache-path", default=DEFAULT_EVAL_DATASETS_CACHE_PATH)
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
    tpu_regions = normalize_tpu_regions(args.tpu_regions or args.tpu_region)
    os.environ.setdefault("MARIN_PREFIX", marin_prefix_for_region(DEFAULT_TPU_REGION))
    if not args.dry_run:
        for arg_name in ("candidate_weights_uri", "candidate_summary_uri"):
            value = getattr(args, arg_name)
            if not str(value).startswith("gs://"):
                raise ValueError(f"Live launch requires --{arg_name.replace('_', '-')} to be a gs:// URI")
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    artifacts = build_launch_artifacts(
        base_name_prefix=args.base_name_prefix,
        candidate_weights_uri=args.candidate_weights_uri,
        candidate_summary_uri=args.candidate_summary_uri,
        tpu_type=args.tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        include_eval_harness=args.include_eval_harness,
    )
    write_local_manifests(artifacts, Path(args.local_artifact_dir))
    logger.info("Wrote local manifests to %s", args.local_artifact_dir)
    logger.info(
        "Prepared %d candidates and %d training steps.", len(artifacts.candidate_specs), len(artifacts.training_steps)
    )
    logger.info(
        "Launch config: tpu=%s regions=%s zone=%s max_concurrent=%d eval_harness=%s",
        args.tpu_type,
        ",".join(tpu_regions),
        args.tpu_zone,
        args.max_concurrent,
        "enabled" if args.include_eval_harness else "skipped",
    )
    for artifact in artifacts.scale_artifacts:
        logger.info("%s prefix: %s (%d runs)", artifact.scale.value, artifact.name_prefix, len(artifact.run_specs))
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_prefix = _executor_prefix(args.executor_prefix, DEFAULT_TPU_REGION)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.base_name_prefix}: proportional gradient-step validation. Runs 4 local-gradient "
            f"candidates at 60M/1.2B and 4 at 100M/6B. Outputs include {RUN_MANIFEST_FILE}, "
            f"{RESULTS_CSV}, {FIT_DATASET_CSV}, and {FIT_DATASET_SUMMARY_JSON} per scale."
        ),
    )


if __name__ == "__main__":
    main()
