# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 300M proportional domain-deletion and log-tilt diagnostics.

This experiment probes two related questions around ``baseline_proportional``:

- which proportional domains are coverage-critical when deleted entirely, and
- which selected KL-local log-tilt directions have measurable directional value.

The design is intentionally single-scale and phase-constant. It is not a
repeated-seed gradient estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

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
from experiments.domain_phase_mix.launch_proportional_perturbation_scale_transfer import (
    BASE_RUN_ID,
    BASE_RUN_NAME,
    _base_proportional_weights,
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

BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m"
COHORT = "proportional_controllability_300m"
FAMILY = "proportional_controllability_300m_6b"
RUN_ID_BASE = 800_000
TARGET_BUDGET_MULTIPLIER = 1.0
LOG_TILT_ALPHA = 0.10
SCALE = ScalingStudyScale.REGMIX_300M_6B
DISPLAY_LABEL = "100M/6B"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/proportional-controllability-300m"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_controllability_300m_20260520"
)
LOCAL_INTERVENTION_MANIFEST_CSV = "intervention_manifest.csv"
LOCAL_TRAINING_MANIFEST_CSV = "training_manifest.csv"
LOCAL_RUN_SPECS_JSON = "run_specs.json"
LOCAL_SUMMARY_JSON = "summary.json"


class InterventionType(StrEnum):
    """300M proportional controllability intervention families."""

    DOMAIN_DELETION = "domain_deletion"
    LOG_TILT = "central_log_tilt"


@dataclass(frozen=True)
class ControllabilityInterventionSpec:
    """One proportional controllability intervention before scale expansion."""

    intervention_index: int
    intervention_id: str
    run_id: int
    run_name: str
    intervention_type: str
    target_domain: str | None
    direction_id: str | None
    direction_type: str | None
    tilt_sign: str | None
    alpha: float | None
    base_mass: float
    tv_distance: float
    renormalizer: str
    phase_mode: str
    base_run_name: str
    base_run_id: int
    base_source_experiment: str
    target_mass_before: float | None
    target_mass_after: float | None
    direction_positive_mass: float | None
    direction_negative_mass: float | None
    direction_l2p_norm: float | None
    direction_l2p_mean: float | None
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ControllabilityRunSpec:
    """Manifest entry for one 300M proportional controllability training run."""

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
    intervention_index: int
    intervention_id: str
    intervention_type: str
    target_domain: str | None
    direction_id: str | None
    direction_type: str | None
    tilt_sign: str | None
    alpha: float | None
    base_mass: float
    tv_distance: float
    renormalizer: str
    phase_mode: str
    target_mass_before: float | None
    target_mass_after: float | None
    direction_positive_mass: float | None
    direction_negative_mass: float | None
    direction_l2p_norm: float | None
    direction_l2p_mean: float | None
    scale: str
    scale_display_label: str
    experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    target_final_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved launch graph for the 300M controllability panel."""

    interventions: list[ControllabilityInterventionSpec]
    name_prefix: str
    run_specs: list[ControllabilityRunSpec]
    run_manifest_step: ExecutorStep
    cache_eval_datasets_step: ExecutorStep | None
    training_steps: list[ExecutorStep]
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        steps: list[object] = [self.run_manifest_step]
        if self.cache_eval_datasets_step is not None:
            steps.append(self.cache_eval_datasets_step)
        steps.extend(self.training_steps)
        steps.extend([self.results_step, self.fit_dataset_step])
        return steps


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError(f"Cannot create slug from {value!r}")
    return normalized


def _domain_direction_id(domain_name: str) -> str:
    return f"domain_{_slug(domain_name)}"


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _validate_domain_weights(weights: dict[str, float], *, label: str) -> None:
    if set(weights) != set(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES) - set(weights))
        extra = sorted(set(weights) - set(DOMAIN_NAMES))
        raise ValueError(f"{label} domain mismatch: missing={missing[:5]}, extra={extra[:5]}")
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"{label} has negative weights")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} weights sum to {total}, not 1")


def _validate_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names do not match {PHASE_NAMES}")
    for phase_name, weights in phase_weights.items():
        _validate_domain_weights(weights, label=f"{run_name}/{phase_name}")
    phase_0 = phase_weights["phase_0"]
    phase_1 = phase_weights["phase_1"]
    max_delta = max(abs(phase_0[domain_name] - phase_1[domain_name]) for domain_name in DOMAIN_NAMES)
    if max_delta > 1e-15:
        raise ValueError(f"{run_name} is not phase-constant: max delta {max_delta}")


def _constant_phase_weights(weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase_name: dict(weights) for phase_name in PHASE_NAMES}


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a[domain_name] - b[domain_name]) for domain_name in DOMAIN_NAMES)


def _raw_direction(direction_id: str) -> tuple[str, dict[str, float]]:
    target_domains = [domain_name for domain_name in DOMAIN_NAMES if _domain_direction_id(domain_name) == direction_id]
    if len(target_domains) == 1:
        target_domain = target_domains[0]
        return "domain", {domain_name: 1.0 if domain_name == target_domain else 0.0 for domain_name in DOMAIN_NAMES}
    raise ValueError(f"Unsupported log-tilt direction {direction_id!r}")


def _center_normalize_direction(
    base_weights: dict[str, float],
    raw: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    mean = sum(base_weights[domain_name] * raw[domain_name] for domain_name in DOMAIN_NAMES)
    centered = {domain_name: raw[domain_name] - mean for domain_name in DOMAIN_NAMES}
    norm = math.sqrt(sum(base_weights[domain_name] * centered[domain_name] ** 2 for domain_name in DOMAIN_NAMES))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero log-tilt direction")
    normalized = {domain_name: centered[domain_name] / norm for domain_name in DOMAIN_NAMES}
    normalized_mean = sum(base_weights[domain_name] * normalized[domain_name] for domain_name in DOMAIN_NAMES)
    normalized_norm = math.sqrt(
        sum(base_weights[domain_name] * normalized[domain_name] ** 2 for domain_name in DOMAIN_NAMES)
    )
    return normalized, normalized_mean, normalized_norm


def _log_tilt_weights(base_weights: dict[str, float], direction: dict[str, float], *, sign: int) -> dict[str, float]:
    if sign not in {-1, 1}:
        raise ValueError(f"Unsupported tilt sign {sign}")
    unnormalized = {
        domain_name: base_weights[domain_name] * math.exp(sign * LOG_TILT_ALPHA * direction[domain_name])
        for domain_name in DOMAIN_NAMES
    }
    total = sum(unnormalized.values())
    weights = {domain_name: value / total for domain_name, value in unnormalized.items()}
    _validate_domain_weights(weights, label=f"log tilt sign={sign}")
    return weights


def _domain_deletion_spec(
    *,
    base_weights: dict[str, float],
    domain_name: str,
) -> ControllabilityInterventionSpec:
    base_mass = base_weights[domain_name]
    if not 0 < base_mass < 1:
        raise ValueError(f"Cannot delete {domain_name} with base mass {base_mass}")
    weights = {
        candidate_domain: 0.0 if candidate_domain == domain_name else base_weights[candidate_domain] / (1 - base_mass)
        for candidate_domain in DOMAIN_NAMES
    }
    _validate_domain_weights(weights, label=f"delete {domain_name}")
    intervention_id = f"delete_{_slug(domain_name)}"
    return ControllabilityInterventionSpec(
        intervention_index=-1,
        intervention_id=intervention_id,
        run_id=-1,
        run_name=f"pctrl_del_{_slug(domain_name)}",
        intervention_type=InterventionType.DOMAIN_DELETION.value,
        target_domain=domain_name,
        direction_id=None,
        direction_type=None,
        tilt_sign=None,
        alpha=None,
        base_mass=base_mass,
        tv_distance=_tv_distance(base_weights, weights),
        renormalizer="delete_domain_redistribute_to_all_other_domains_proportionally",
        phase_mode="both_phases",
        base_run_name=BASE_RUN_NAME,
        base_run_id=BASE_RUN_ID,
        base_source_experiment=ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        target_mass_before=base_mass,
        target_mass_after=0.0,
        direction_positive_mass=None,
        direction_negative_mass=None,
        direction_l2p_norm=None,
        direction_l2p_mean=None,
        phase_weights=_constant_phase_weights(weights),
    )


def _positive_negative_mass(base_weights: dict[str, float], raw: dict[str, float]) -> tuple[float, float]:
    positive_mass = sum(base_weights[domain_name] for domain_name, value in raw.items() if value > 0)
    negative_mass = sum(base_weights[domain_name] for domain_name, value in raw.items() if value < 0)
    return positive_mass, negative_mass


def _log_tilt_specs(
    *,
    base_weights: dict[str, float],
    direction_id: str,
) -> list[ControllabilityInterventionSpec]:
    direction_type, raw = _raw_direction(direction_id)
    direction, direction_mean, direction_norm = _center_normalize_direction(base_weights, raw)
    positive_mass, negative_mass = _positive_negative_mass(base_weights, raw)
    target_domain = next((domain_name for domain_name, value in raw.items() if value > 0), None)
    specs = []
    for sign_name, sign in (("plus", 1), ("minus", -1)):
        weights = _log_tilt_weights(base_weights, direction, sign=sign)
        intervention_id = f"tilt_{direction_id}_{sign_name}"
        specs.append(
            ControllabilityInterventionSpec(
                intervention_index=-1,
                intervention_id=intervention_id,
                run_id=-1,
                run_name=f"pctrl_tilt_{_slug(direction_id)}_{sign_name}",
                intervention_type=InterventionType.LOG_TILT.value,
                target_domain=target_domain,
                direction_id=direction_id,
                direction_type=direction_type,
                tilt_sign=sign_name,
                alpha=LOG_TILT_ALPHA,
                base_mass=positive_mass,
                tv_distance=_tv_distance(base_weights, weights),
                renormalizer="central_log_tilt_relative_to_proportional",
                phase_mode="both_phases",
                base_run_name=BASE_RUN_NAME,
                base_run_id=BASE_RUN_ID,
                base_source_experiment=ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                target_mass_before=None if target_domain is None else base_weights[target_domain],
                target_mass_after=None if target_domain is None else weights[target_domain],
                direction_positive_mass=positive_mass,
                direction_negative_mass=negative_mass,
                direction_l2p_norm=direction_norm,
                direction_l2p_mean=direction_mean,
                phase_weights=_constant_phase_weights(weights),
            )
        )
    return specs


def build_interventions() -> list[ControllabilityInterventionSpec]:
    """Build the deterministic 117-row 300M controllability manifest."""
    base_weights = _base_proportional_weights()
    raw_specs = [
        *[_domain_deletion_spec(base_weights=base_weights, domain_name=domain_name) for domain_name in DOMAIN_NAMES],
        *[
            spec
            for direction_id in (_domain_direction_id(domain_name) for domain_name in DOMAIN_NAMES)
            for spec in _log_tilt_specs(base_weights=base_weights, direction_id=direction_id)
        ],
    ]
    specs = [replace(spec, intervention_index=index, run_id=RUN_ID_BASE + index) for index, spec in enumerate(raw_specs)]
    validate_interventions(specs)
    return specs


def validate_interventions(specs: list[ControllabilityInterventionSpec]) -> None:
    """Validate intervention-level invariants before launch graph construction."""
    if len(specs) != 117:
        raise ValueError(f"Expected 117 interventions, got {len(specs)}")
    type_counts = {
        intervention_type.value: sum(1 for spec in specs if spec.intervention_type == intervention_type.value)
        for intervention_type in InterventionType
    }
    expected_counts = {
        InterventionType.DOMAIN_DELETION.value: 39,
        InterventionType.LOG_TILT.value: 78,
    }
    if type_counts != expected_counts:
        raise ValueError(f"Unexpected intervention counts: {type_counts}")
    run_names = [spec.run_name for spec in specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate intervention run names")
    run_ids = [spec.run_id for spec in specs]
    if run_ids != list(range(RUN_ID_BASE, RUN_ID_BASE + len(specs))):
        raise ValueError("Intervention run IDs are not contiguous")

    base_weights = _base_proportional_weights()
    log_tilt_counts: dict[str, set[str]] = {}
    for index, spec in enumerate(specs):
        if spec.intervention_index != index:
            raise ValueError(f"{spec.run_name} has intervention_index={spec.intervention_index}, expected {index}")
        _validate_phase_weights(spec.phase_weights, run_name=spec.run_name)
        phase_weights = spec.phase_weights["phase_0"]
        if spec.intervention_type == InterventionType.DOMAIN_DELETION.value:
            if spec.target_domain is None:
                raise ValueError(f"{spec.run_name} is missing target_domain")
            if not math.isclose(phase_weights[spec.target_domain], 0.0, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{spec.run_name} target domain was not deleted")
            for domain_name in DOMAIN_NAMES:
                if domain_name == spec.target_domain:
                    continue
                expected = base_weights[domain_name] / (1 - spec.base_mass)
                if not math.isclose(phase_weights[domain_name], expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{spec.run_name} donor {domain_name} was not proportional")
            if not math.isclose(spec.tv_distance, spec.base_mass, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{spec.run_name} TV {spec.tv_distance} != deleted mass {spec.base_mass}")
        elif spec.intervention_type == InterventionType.LOG_TILT.value:
            if spec.direction_id is None or spec.tilt_sign is None or spec.alpha != LOG_TILT_ALPHA:
                raise ValueError(f"{spec.run_name} has incomplete log-tilt metadata")
            if spec.direction_l2p_mean is None or abs(spec.direction_l2p_mean) > 1e-12:
                raise ValueError(f"{spec.run_name} direction is not centered")
            if spec.direction_l2p_norm is None or not math.isclose(
                spec.direction_l2p_norm,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{spec.run_name} direction is not unit L2(p)")
            log_tilt_counts.setdefault(spec.direction_id, set()).add(spec.tilt_sign)
        else:
            raise ValueError(f"Unsupported intervention type {spec.intervention_type!r}")

    expected_directions = {_domain_direction_id(domain_name) for domain_name in DOMAIN_NAMES}
    if set(log_tilt_counts) != expected_directions:
        raise ValueError(f"Unexpected log-tilt directions: {sorted(log_tilt_counts)}")
    missing_pairs = {direction: signs for direction, signs in log_tilt_counts.items() if signs != {"plus", "minus"}}
    if missing_pairs:
        raise ValueError(f"Missing log-tilt plus/minus pairs: {missing_pairs}")


def _run_spec(intervention: ControllabilityInterventionSpec) -> ControllabilityRunSpec:
    scale_spec = resolve_scale_spec(SCALE)
    num_train_steps = scale_spec.num_train_steps_for_multiplier(TARGET_BUDGET_MULTIPLIER)
    return ControllabilityRunSpec(
        run_id=intervention.run_id,
        run_name=intervention.run_name,
        cohort=COHORT,
        model_family=scale_spec.model_family,
        trainer_seed=None,
        data_seed=intervention.run_id,
        simulated_epoch_subset_seed=None,
        source_run_id=BASE_RUN_ID,
        source_run_name=BASE_RUN_NAME,
        source_two_phase_experiment=ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        candidate_run_id=intervention.run_id,
        candidate_run_name=intervention.run_name,
        candidate_source_experiment=BASE_NAME_PREFIX,
        intervention_index=intervention.intervention_index,
        intervention_id=intervention.intervention_id,
        intervention_type=intervention.intervention_type,
        target_domain=intervention.target_domain,
        direction_id=intervention.direction_id,
        direction_type=intervention.direction_type,
        tilt_sign=intervention.tilt_sign,
        alpha=intervention.alpha,
        base_mass=intervention.base_mass,
        tv_distance=intervention.tv_distance,
        renormalizer=intervention.renormalizer,
        phase_mode=intervention.phase_mode,
        target_mass_before=intervention.target_mass_before,
        target_mass_after=intervention.target_mass_after,
        direction_positive_mass=intervention.direction_positive_mass,
        direction_negative_mass=intervention.direction_negative_mass,
        direction_l2p_norm=intervention.direction_l2p_norm,
        direction_l2p_mean=intervention.direction_l2p_mean,
        scale=SCALE.value,
        scale_display_label=DISPLAY_LABEL,
        experiment_budget=scale_spec.experiment_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget=scale_spec.target_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=num_train_steps,
        target_final_checkpoint_step=num_train_steps - 1,
        phase_weights=intervention.phase_weights,
    )


def build_run_specs() -> list[ControllabilityRunSpec]:
    """Build the 117 300M run specs."""
    return [_run_spec(intervention) for intervention in build_interventions()]


def _select_interventions(
    interventions: list[ControllabilityInterventionSpec],
    only_run_names: tuple[str, ...],
) -> list[ControllabilityInterventionSpec]:
    if not only_run_names:
        return interventions
    requested_run_names = tuple(dict.fromkeys(only_run_names))
    known_run_names = {intervention.run_name for intervention in interventions}
    unknown_run_names = sorted(set(requested_run_names) - known_run_names)
    if unknown_run_names:
        raise ValueError(f"Unknown proportional-controllability run names: {unknown_run_names}")
    requested = set(requested_run_names)
    return [intervention for intervention in interventions if intervention.run_name in requested]


def _run_manifest_step(
    *,
    execution_name_prefix: str,
    experiment_name: str,
    run_specs: list[ControllabilityRunSpec],
) -> ExecutorStep:
    scale_spec = resolve_scale_spec(SCALE)
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
    base_name_prefix: str,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
    only_run_names: tuple[str, ...] = (),
) -> LaunchArtifacts:
    """Resolve the launch graph without submitting it."""
    if tpu_regions != (DEFAULT_TPU_REGION,) or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This experiment is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; got {tpu_regions}/{tpu_zone}"
        )
    if "us-central" in eval_datasets_cache_path:
        raise ValueError(f"Eval cache path must be east5-local, got {eval_datasets_cache_path}")
    scale_spec = resolve_scale_spec(SCALE)
    interventions = _select_interventions(build_interventions(), only_run_names)
    run_specs = [_run_spec(intervention) for intervention in interventions]
    if not run_specs:
        raise ValueError("No proportional-controllability runs selected")
    experiment = create_qsplit240_replay_experiment(
        name=base_name_prefix,
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
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(tpu_regions, eval_datasets_cache_path)
    run_manifest_step = _run_manifest_step(
        execution_name_prefix=base_name_prefix,
        experiment_name=base_name_prefix,
        run_specs=run_specs,
    )
    cache_eval_datasets_step = (
        create_cache_eval_datasets_step(
            eval_tasks=QSPLIT240_300M_EVAL_TASKS,
            gcs_path=resolved_eval_cache_path,
            name_prefix=base_name_prefix,
        )
        if include_eval_harness
        else None
    )
    training_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=base_name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
            simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        )
        if cache_eval_datasets_step is not None:
            training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        training_step = _configure_training_step(
            training_step,
            tpu_region=tpu_regions[0],
            include_eval_harness=include_eval_harness,
        )
        if not include_eval_harness:
            training_step = skip_eval_harness_for_training_step(training_step)
        training_steps.append(training_step)

    results_step = create_manifest_results_step(
        name_prefix=base_name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=base_name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )
    artifacts = LaunchArtifacts(
        interventions=interventions,
        name_prefix=base_name_prefix,
        run_specs=run_specs,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )
    validate_launch_artifacts(artifacts, include_eval_harness=include_eval_harness)
    return artifacts


def validate_launch_artifacts(artifacts: LaunchArtifacts, *, include_eval_harness: bool) -> None:
    """Validate graph invariants before launch."""
    expected_count = len(artifacts.run_specs)
    if expected_count <= 0:
        raise ValueError("Expected at least one run spec")
    if len(artifacts.interventions) != expected_count or len(artifacts.training_steps) != expected_count:
        raise ValueError(
            "Expected matching intervention, run spec, and training step counts; "
            f"got {len(artifacts.interventions)}, {len(artifacts.run_specs)}, {len(artifacts.training_steps)}"
        )
    if include_eval_harness and artifacts.cache_eval_datasets_step is None:
        raise ValueError("Eval harness is enabled but eval-dataset cache step is missing")
    if not include_eval_harness and artifacts.cache_eval_datasets_step is not None:
        raise ValueError("Eval harness is skipped but eval-dataset cache step is present")
    expected_steps = resolve_scale_spec(SCALE).num_train_steps_for_multiplier(TARGET_BUDGET_MULTIPLIER)
    run_names = [run_spec.run_name for run_spec in artifacts.run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate run names")
    output_roots = [str(step.override_output_path or step.name) for step in artifacts.training_steps]
    if len(set(output_roots)) != len(output_roots):
        raise ValueError("Duplicate training output paths")

    for run_spec in artifacts.run_specs:
        _validate_phase_weights(run_spec.phase_weights, run_name=run_spec.run_name)
        if run_spec.scale != SCALE.value:
            raise ValueError(f"{run_spec.run_name} has scale={run_spec.scale}")
        if run_spec.num_train_steps != expected_steps:
            raise ValueError(f"{run_spec.run_name} has num_train_steps={run_spec.num_train_steps}")
        if run_spec.target_final_checkpoint_step != expected_steps - 1:
            raise ValueError(f"{run_spec.run_name} final step={run_spec.target_final_checkpoint_step}")

    for training_step in artifacts.training_steps:
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(DEFAULT_TPU_REGION):
            raise ValueError(f"{training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        eval_cache_dep = str(env_vars.get(EVAL_DATASETS_CACHE_DEP_ENV_VAR, ""))
        if "us-central" in eval_cache_dep:
            raise ValueError(f"{training_step.name} has central-region eval cache dependency")
        if not include_eval_harness and eval_cache_dep:
            raise ValueError(f"{training_step.name} has eval cache dependency despite skipped eval harness")
        has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
        if include_eval_harness and has_skip:
            raise ValueError(f"{training_step.name} unexpectedly skips eval harness")
        if not include_eval_harness and not has_skip:
            raise ValueError(f"{training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")
        if int(config.train_config.trainer.num_train_steps) != expected_steps:
            raise ValueError(f"{training_step.name} trainer has wrong num_train_steps")
        hf_save_steps = config.train_config.hf_save_steps
        if hf_save_steps is None or int(hf_save_steps) != expected_steps:
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_local_manifests(artifacts: LaunchArtifacts, output_dir: Path) -> None:
    """Write local audit artifacts for the launch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    intervention_rows = [_flat_manifest_row(asdict(spec)) for spec in artifacts.interventions]
    training_rows = [_flat_manifest_row(asdict(run_spec)) for run_spec in artifacts.run_specs]
    _write_csv(output_dir / LOCAL_INTERVENTION_MANIFEST_CSV, intervention_rows)
    _write_csv(output_dir / LOCAL_TRAINING_MANIFEST_CSV, training_rows)
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "base_name_prefix": BASE_NAME_PREFIX,
                "cohort": COHORT,
                "family": FAMILY,
                "scale": SCALE.value,
                "interventions": [asdict(spec) for spec in artifacts.interventions],
                "run_specs": [asdict(run_spec) for run_spec in artifacts.run_specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "base_name_prefix": BASE_NAME_PREFIX,
        "cohort": COHORT,
        "family": FAMILY,
        "scale": SCALE.value,
        "scale_display_label": DISPLAY_LABEL,
        "base_run_name": BASE_RUN_NAME,
        "base_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        "log_tilt_alpha": LOG_TILT_ALPHA,
        "log_tilt_directions": [_domain_direction_id(domain_name) for domain_name in DOMAIN_NAMES],
        "intervention_count": len(artifacts.interventions),
        "training_run_count": len(training_rows),
        "type_counts": {
            intervention_type.value: sum(
                1 for spec in artifacts.interventions if spec.intervention_type == intervention_type.value
            )
            for intervention_type in InterventionType
        },
        "outputs": {
            "intervention_manifest_csv": LOCAL_INTERVENTION_MANIFEST_CSV,
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
    raise ValueError(
        "--executor-prefix is disabled for this training launcher. A custom executor prefix rewrites the shared "
        "raw/tokenized data-prep output paths, causing expensive re-download/re-tokenization instead of reusing "
        f"{marin_prefix_for_region(default_tpu_region)} caches. Resubmit without --executor-prefix."
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name-prefix", default=BASE_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-regions")
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--executor-prefix",
        help=(
            "Disabled for this launcher: setting it would rewrite shared raw/tokenized cache paths and trigger "
            "data re-materialization."
        ),
    )
    parser.add_argument("--eval-datasets-cache-path", default=DEFAULT_EVAL_DATASETS_CACHE_PATH)
    parser.add_argument("--local-artifact-dir", default=str(DEFAULT_LOCAL_ARTIFACT_DIR))
    parser.add_argument(
        "--only-run-name",
        action="append",
        default=[],
        help="Restrict launch to one run name. Can be passed multiple times for targeted retries.",
    )
    parser.add_argument(
        "--only-run-name-file",
        help="Restrict launch to run names listed one per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run Levanter lm-eval harness during training. Default is perplexity/checkpoint only.",
    )
    return parser.parse_known_args()


def _read_only_run_names(args: argparse.Namespace) -> tuple[str, ...]:
    run_names = list(args.only_run_name or [])
    if args.only_run_name_file:
        with Path(args.only_run_name_file).open() as handle:
            run_names.extend(stripped for line in handle if (stripped := line.strip()) and not stripped.startswith("#"))
    return tuple(dict.fromkeys(run_names))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    tpu_regions = normalize_tpu_regions(args.tpu_regions or args.tpu_region)
    only_run_names = _read_only_run_names(args)
    os.environ.setdefault("MARIN_PREFIX", marin_prefix_for_region(DEFAULT_TPU_REGION))
    _executor_prefix(args.executor_prefix, DEFAULT_TPU_REGION)
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    artifacts = build_launch_artifacts(
        base_name_prefix=args.base_name_prefix,
        tpu_type=args.tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        include_eval_harness=args.include_eval_harness,
        only_run_names=only_run_names,
    )
    write_local_manifests(artifacts, Path(args.local_artifact_dir))
    logger.info("Wrote local manifests to %s", args.local_artifact_dir)
    intervention_count = len(artifacts.interventions)
    training_step_count = len(artifacts.training_steps)
    logger.info("Prepared %d interventions and %d training steps.", intervention_count, training_step_count)
    if only_run_names:
        logger.info("Subset retry includes %d requested run names.", len(only_run_names))
    logger.info(
        "Launch config: tpu=%s regions=%s zone=%s max_concurrent=%d eval_harness=%s",
        args.tpu_type,
        ",".join(tpu_regions),
        args.tpu_zone,
        args.max_concurrent,
        "enabled" if args.include_eval_harness else "skipped",
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_main(
        ExecutorMainConfig(prefix=None, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.base_name_prefix}: 300M proportional domain-deletion and log-tilt diagnostics. "
            f"Runs {training_step_count} selected endpoint(s). Outputs include "
            f"{RUN_MANIFEST_FILE}, {RESULTS_CSV}, {FIT_DATASET_CSV}, and {FIT_DATASET_SUMMARY_JSON}."
        ),
    )


if __name__ == "__main__":
    main()
