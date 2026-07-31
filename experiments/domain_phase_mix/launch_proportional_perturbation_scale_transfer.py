# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch proportional perturbations at 60M/1.2B and 100M/6B.

This experiment probes whether local exposure increases around the proportional
mixture change the BPB drop from 60M/1.2B to 100M/6B. It uses a deliberately
small, interpretable Stage 1 design:

- one +5pp bump for each top-level domain,
- one +5pp bump for each GRP generic family, and
- one low-to-high quality swap for each CC high/low topic pair.

The launcher builds both scales in one executor graph while keeping each scale
under a distinct source experiment prefix.
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
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DOMAIN_NAMES,
    PHASE_NAMES,
    create_initial_fixed_weight_configs,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT

logger = logging.getLogger(__name__)

BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_perturbation_scale_transfer"
COHORT = "proportional_perturbation_scale_transfer_stage1"
BASE_RUN_NAME = "baseline_proportional"
BASE_RUN_ID = 0
INTERVENTION_RUN_ID_BASE = 780_000
DOMAIN_BUMP_EPSILON = 0.05
FAMILY_BUMP_EPSILON = 0.05
QUALITY_SWAP_FRACTION = 0.5
TARGET_BUDGET_MULTIPLIER = 1.0
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_perturbation_scale_transfer_20260507"
)
LOCAL_INTERVENTION_MANIFEST_CSV = "intervention_manifest.csv"
LOCAL_TRAINING_MANIFEST_CSV = "training_manifest.csv"
LOCAL_RUN_SPECS_JSON = "run_specs.json"
LOCAL_SUMMARY_JSON = "summary.json"
SCALES = (ScalingStudyScale.REGMIX_60M_1P2B, ScalingStudyScale.REGMIX_300M_6B)
DISPLAY_LABELS = {
    ScalingStudyScale.REGMIX_60M_1P2B: "60M/1.2B",
    ScalingStudyScale.REGMIX_300M_6B: "100M/6B",
}
GENERIC_FAMILY_NAMES = ("broad_text", "tech_code", "reasoning")


class InterventionType(StrEnum):
    """Stage 1 perturbation families."""

    DOMAIN_BUMP = "domain_bump"
    FAMILY_BUMP = "family_bump"
    QUALITY_SWAP = "quality_swap"


@dataclass(frozen=True)
class InterventionSpec:
    """One proportional-mixture perturbation before scale expansion."""

    intervention_index: int
    intervention_id: str
    run_id: int
    run_name: str
    intervention_type: str
    target_unit: str
    target_domain: str | None
    target_family: str | None
    quality_high_domain: str | None
    quality_low_domain: str | None
    bump_epsilon: float | None
    quality_swap_fraction: float | None
    quality_swap_mass: float | None
    renormalizer: str
    donor_pool: str
    phase_mode: str
    base_run_name: str
    base_run_id: int
    base_source_experiment: str
    tv_distance: float
    target_mass_before: float
    target_mass_after: float
    donor_mass_before: float
    donor_mass_after: float
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PerturbationRunSpec:
    """Manifest entry for one intervention at one scale."""

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
    target_unit: str
    target_domain: str | None
    target_family: str | None
    quality_high_domain: str | None
    quality_low_domain: str | None
    bump_epsilon: float | None
    quality_swap_fraction: float | None
    quality_swap_mass: float | None
    renormalizer: str
    donor_pool: str
    phase_mode: str
    tv_distance: float
    target_mass_before: float
    target_mass_after: float
    donor_mass_before: float
    donor_mass_after: float
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
    run_specs: list[PerturbationRunSpec]
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
class PairedLaunchArtifacts:
    """Resolved paired-scale launch graph."""

    interventions: list[InterventionSpec]
    scale_artifacts: list[ScaleLaunchArtifacts]

    @property
    def steps(self) -> list[object]:
        steps: list[object] = []
        for artifact in self.scale_artifacts:
            steps.extend(artifact.steps)
        return steps

    @property
    def training_steps(self) -> list[ExecutorStep]:
        steps: list[ExecutorStep] = []
        for artifact in self.scale_artifacts:
            steps.extend(artifact.training_steps)
        return steps


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError(f"Cannot create slug from {value!r}")
    return normalized


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _base_proportional_weights() -> dict[str, float]:
    baseline_configs = {config.run_name: config.weight_config for config in create_initial_fixed_weight_configs()}
    base_config = baseline_configs[BASE_RUN_NAME]
    phase_0 = base_config.phase_weights["phase_0"]
    phase_1 = base_config.phase_weights["phase_1"]
    _validate_domain_weights(phase_0, label=f"{BASE_RUN_NAME}/phase_0")
    _validate_domain_weights(phase_1, label=f"{BASE_RUN_NAME}/phase_1")
    for domain_name in DOMAIN_NAMES:
        if not math.isclose(phase_0[domain_name], phase_1[domain_name], rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{BASE_RUN_NAME} is not phase-constant for {domain_name}")
    return dict(phase_0)


def _validate_domain_weights(weights: dict[str, float], *, label: str) -> None:
    if set(weights) != set(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES) - set(weights))
        extra = sorted(set(weights) - set(DOMAIN_NAMES))
        raise ValueError(f"{label} domain mismatch: missing={missing[:5]}, extra={extra[:5]}")
    values = list(weights.values())
    if any(value < 0 for value in values):
        raise ValueError(f"{label} has negative weights")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} weights sum to {total}, not 1")


def _constant_phase_weights(weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase_name: dict(weights) for phase_name in PHASE_NAMES}


def _generic_family_name(domain_name: str) -> str:
    """Return the GRP generic family used by the current no-L2 family model."""
    is_broad = (
        domain_name.startswith("dolma3_cc/")
        or domain_name
        in {
            "dolma3_wikipedia",
            "dolmino_common_crawl_hq",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_stem_heavy_crawl",
        }
        or domain_name.endswith("synth_qa")
    )
    is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or (
        domain_name in {"dolma3_arxiv", "dolma3_finemath_3plus"}
    )
    is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
    matches = [
        name
        for name, matched in (
            ("broad_text", is_broad),
            ("tech_code", is_tech),
            ("reasoning", is_reasoning),
        )
        if matched
    ]
    if len(matches) != 1:
        raise ValueError(f"{domain_name} matched {matches}, expected exactly one generic family")
    return matches[0]


def _family_domains(family_name: str) -> tuple[str, ...]:
    if family_name not in GENERIC_FAMILY_NAMES:
        raise ValueError(f"Unsupported family {family_name!r}")
    domains = tuple(domain_name for domain_name in DOMAIN_NAMES if _generic_family_name(domain_name) == family_name)
    if not domains:
        raise ValueError(f"No domains found for family {family_name}")
    return domains


def _quality_pairs() -> tuple[tuple[str, str, str], ...]:
    pairs: list[tuple[str, str, str]] = []
    domains = set(DOMAIN_NAMES)
    for domain_name in DOMAIN_NAMES:
        if not domain_name.startswith("dolma3_cc/") or not domain_name.endswith("_high"):
            continue
        topic = domain_name.removeprefix("dolma3_cc/").removesuffix("_high")
        low_name = f"dolma3_cc/{topic}_low"
        if low_name not in domains:
            raise ValueError(f"Missing low-quality pair for {domain_name}")
        pairs.append((topic, domain_name, low_name))
    if len(pairs) != 13:
        raise ValueError(f"Expected 13 CC high/low pairs, found {len(pairs)}")
    return tuple(pairs)


def _bump_unit(
    base_weights: dict[str, float],
    target_domains: tuple[str, ...],
    epsilon: float,
) -> tuple[dict[str, float], float, float, float, float]:
    target_set = set(target_domains)
    target_mass = sum(base_weights[domain_name] for domain_name in target_domains)
    donor_domains = tuple(domain_name for domain_name in DOMAIN_NAMES if domain_name not in target_set)
    donor_mass = sum(base_weights[domain_name] for domain_name in donor_domains)
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not 0 < target_mass < 1:
        raise ValueError(f"Invalid target mass {target_mass} for {target_domains}")
    if donor_mass <= epsilon:
        raise ValueError(f"epsilon {epsilon} exceeds donor mass {donor_mass}")

    weights = dict(base_weights)
    target_scale = (target_mass + epsilon) / target_mass
    donor_scale = (donor_mass - epsilon) / donor_mass
    for domain_name in target_domains:
        weights[domain_name] = base_weights[domain_name] * target_scale
    for domain_name in donor_domains:
        weights[domain_name] = base_weights[domain_name] * donor_scale
    _validate_domain_weights(weights, label=f"bump {target_domains}")
    return weights, target_mass, target_mass + epsilon, donor_mass, donor_mass - epsilon


def _quality_swap(
    base_weights: dict[str, float],
    *,
    high_domain: str,
    low_domain: str,
) -> tuple[dict[str, float], float, float, float, float, float]:
    low_mass = base_weights[low_domain]
    high_mass = base_weights[high_domain]
    move = QUALITY_SWAP_FRACTION * low_mass
    if move <= 0 or move >= low_mass:
        raise ValueError(f"Invalid quality-swap mass {move} for {low_domain} with mass {low_mass}")
    weights = dict(base_weights)
    weights[high_domain] = high_mass + move
    weights[low_domain] = low_mass - move
    _validate_domain_weights(weights, label=f"quality swap {high_domain}/{low_domain}")
    return weights, high_mass, high_mass + move, low_mass, low_mass - move, move


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a[domain_name] - b[domain_name]) for domain_name in DOMAIN_NAMES)


def _domain_bump_specs(base_weights: dict[str, float]) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = []
    for domain_name in DOMAIN_NAMES:
        weights, before, after, donor_before, donor_after = _bump_unit(
            base_weights,
            (domain_name,),
            DOMAIN_BUMP_EPSILON,
        )
        intervention_id = f"domain_{_slug(domain_name)}"
        specs.append(
            _intervention_spec(
                base_weights=base_weights,
                weights=weights,
                intervention_type=InterventionType.DOMAIN_BUMP,
                target_unit=domain_name,
                target_domain=domain_name,
                target_family=None,
                quality_high_domain=None,
                quality_low_domain=None,
                bump_epsilon=DOMAIN_BUMP_EPSILON,
                quality_swap_fraction=None,
                quality_swap_mass=None,
                renormalizer="add_absolute_mass_remove_non_target_proportionally",
                donor_pool="all_non_target_domains",
                target_mass_before=before,
                target_mass_after=after,
                donor_mass_before=donor_before,
                donor_mass_after=donor_after,
                intervention_id=intervention_id,
            )
        )
    return specs


def _family_bump_specs(base_weights: dict[str, float]) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = []
    for family_name in GENERIC_FAMILY_NAMES:
        target_domains = _family_domains(family_name)
        weights, before, after, donor_before, donor_after = _bump_unit(
            base_weights,
            target_domains,
            FAMILY_BUMP_EPSILON,
        )
        intervention_id = f"family_{family_name}"
        specs.append(
            _intervention_spec(
                base_weights=base_weights,
                weights=weights,
                intervention_type=InterventionType.FAMILY_BUMP,
                target_unit=family_name,
                target_domain=None,
                target_family=family_name,
                quality_high_domain=None,
                quality_low_domain=None,
                bump_epsilon=FAMILY_BUMP_EPSILON,
                quality_swap_fraction=None,
                quality_swap_mass=None,
                renormalizer="add_absolute_family_mass_preserve_within_family_remove_non_family_proportionally",
                donor_pool="all_non_family_domains",
                target_mass_before=before,
                target_mass_after=after,
                donor_mass_before=donor_before,
                donor_mass_after=donor_after,
                intervention_id=intervention_id,
            )
        )
    return specs


def _quality_swap_specs(base_weights: dict[str, float]) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = []
    for topic, high_domain, low_domain in _quality_pairs():
        weights, high_before, high_after, low_before, low_after, move = _quality_swap(
            base_weights,
            high_domain=high_domain,
            low_domain=low_domain,
        )
        intervention_id = f"qswap_{_slug(topic)}"
        specs.append(
            _intervention_spec(
                base_weights=base_weights,
                weights=weights,
                intervention_type=InterventionType.QUALITY_SWAP,
                target_unit=topic,
                target_domain=None,
                target_family=None,
                quality_high_domain=high_domain,
                quality_low_domain=low_domain,
                bump_epsilon=None,
                quality_swap_fraction=QUALITY_SWAP_FRACTION,
                quality_swap_mass=move,
                renormalizer="move_fraction_low_quality_to_high_quality_preserve_topic_total",
                donor_pool=low_domain,
                target_mass_before=high_before,
                target_mass_after=high_after,
                donor_mass_before=low_before,
                donor_mass_after=low_after,
                intervention_id=intervention_id,
            )
        )
    return specs


def _intervention_spec(
    *,
    base_weights: dict[str, float],
    weights: dict[str, float],
    intervention_type: InterventionType,
    target_unit: str,
    target_domain: str | None,
    target_family: str | None,
    quality_high_domain: str | None,
    quality_low_domain: str | None,
    bump_epsilon: float | None,
    quality_swap_fraction: float | None,
    quality_swap_mass: float | None,
    renormalizer: str,
    donor_pool: str,
    target_mass_before: float,
    target_mass_after: float,
    donor_mass_before: float,
    donor_mass_after: float,
    intervention_id: str,
) -> InterventionSpec:
    run_name = f"ppert_{intervention_id}"
    return InterventionSpec(
        intervention_index=-1,
        intervention_id=intervention_id,
        run_id=-1,
        run_name=run_name,
        intervention_type=intervention_type.value,
        target_unit=target_unit,
        target_domain=target_domain,
        target_family=target_family,
        quality_high_domain=quality_high_domain,
        quality_low_domain=quality_low_domain,
        bump_epsilon=bump_epsilon,
        quality_swap_fraction=quality_swap_fraction,
        quality_swap_mass=quality_swap_mass,
        renormalizer=renormalizer,
        donor_pool=donor_pool,
        phase_mode="both_phases",
        base_run_name=BASE_RUN_NAME,
        base_run_id=BASE_RUN_ID,
        base_source_experiment=ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
        tv_distance=_tv_distance(base_weights, weights),
        target_mass_before=target_mass_before,
        target_mass_after=target_mass_after,
        donor_mass_before=donor_mass_before,
        donor_mass_after=donor_mass_after,
        phase_weights=_constant_phase_weights(weights),
    )


def build_interventions() -> list[InterventionSpec]:
    """Build the deterministic 55-row Stage 1 intervention manifest."""
    base_weights = _base_proportional_weights()
    raw_specs = [
        *_domain_bump_specs(base_weights),
        *_family_bump_specs(base_weights),
        *_quality_swap_specs(base_weights),
    ]
    specs: list[InterventionSpec] = []
    for index, spec in enumerate(raw_specs):
        specs.append(
            replace(
                spec,
                intervention_index=index,
                run_id=INTERVENTION_RUN_ID_BASE + index,
            )
        )
    validate_interventions(specs)
    return specs


def validate_interventions(specs: list[InterventionSpec]) -> None:
    """Validate intervention-level invariants before scale expansion."""
    if len(specs) != 55:
        raise ValueError(f"Expected 55 interventions, got {len(specs)}")
    type_counts = {
        intervention_type: sum(1 for spec in specs if spec.intervention_type == intervention_type.value)
        for intervention_type in InterventionType
    }
    expected_counts = {
        InterventionType.DOMAIN_BUMP.value: 39,
        InterventionType.FAMILY_BUMP.value: 3,
        InterventionType.QUALITY_SWAP.value: 13,
    }
    if type_counts != expected_counts:
        raise ValueError(f"Unexpected intervention counts: {type_counts}")
    run_names = [spec.run_name for spec in specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate intervention run names")
    run_ids = [spec.run_id for spec in specs]
    if run_ids != list(range(INTERVENTION_RUN_ID_BASE, INTERVENTION_RUN_ID_BASE + len(specs))):
        raise ValueError("Intervention run IDs are not contiguous")
    for index, spec in enumerate(specs):
        if spec.intervention_index != index:
            raise ValueError(f"{spec.run_name} has intervention_index={spec.intervention_index}, expected {index}")
        if spec.phase_mode != "both_phases":
            raise ValueError(f"{spec.run_name} has unsupported phase_mode={spec.phase_mode}")
        _validate_phase_weights(spec.phase_weights, run_name=spec.run_name)
        phase_0 = spec.phase_weights["phase_0"]
        phase_1 = spec.phase_weights["phase_1"]
        max_phase_delta = max(abs(phase_0[domain_name] - phase_1[domain_name]) for domain_name in DOMAIN_NAMES)
        if max_phase_delta > 1e-15:
            raise ValueError(f"{spec.run_name} is not phase-constant: max delta {max_phase_delta}")
        if spec.intervention_type in {InterventionType.DOMAIN_BUMP.value, InterventionType.FAMILY_BUMP.value}:
            if not math.isclose(
                spec.target_mass_after - spec.target_mass_before,
                DOMAIN_BUMP_EPSILON,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{spec.run_name} target mass did not increase by 5pp")
            if not math.isclose(
                spec.donor_mass_before - spec.donor_mass_after,
                DOMAIN_BUMP_EPSILON,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{spec.run_name} donor mass did not decrease by 5pp")
        if spec.intervention_type == InterventionType.QUALITY_SWAP.value:
            if spec.quality_swap_mass is None:
                raise ValueError(f"{spec.run_name} missing quality_swap_mass")
            if not math.isclose(
                spec.target_mass_after - spec.target_mass_before,
                spec.quality_swap_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{spec.run_name} high-quality split did not gain moved mass")
            if not math.isclose(
                spec.donor_mass_before - spec.donor_mass_after,
                spec.quality_swap_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{spec.run_name} low-quality split did not lose moved mass")


def _validate_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names do not match {PHASE_NAMES}")
    for phase_name, weights in phase_weights.items():
        _validate_domain_weights(weights, label=f"{run_name}/{phase_name}")


def _scale_name_prefix(base_name_prefix: str, scale: ScalingStudyScale) -> str:
    return f"{base_name_prefix}_{scale.value}"


def _run_spec_for_scale(intervention: InterventionSpec, scale: ScalingStudyScale) -> PerturbationRunSpec:
    scale_spec = resolve_scale_spec(scale)
    num_train_steps = scale_spec.num_train_steps_for_multiplier(TARGET_BUDGET_MULTIPLIER)
    return PerturbationRunSpec(
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
        target_unit=intervention.target_unit,
        target_domain=intervention.target_domain,
        target_family=intervention.target_family,
        quality_high_domain=intervention.quality_high_domain,
        quality_low_domain=intervention.quality_low_domain,
        bump_epsilon=intervention.bump_epsilon,
        quality_swap_fraction=intervention.quality_swap_fraction,
        quality_swap_mass=intervention.quality_swap_mass,
        renormalizer=intervention.renormalizer,
        donor_pool=intervention.donor_pool,
        phase_mode=intervention.phase_mode,
        tv_distance=intervention.tv_distance,
        target_mass_before=intervention.target_mass_before,
        target_mass_after=intervention.target_mass_after,
        donor_mass_before=intervention.donor_mass_before,
        donor_mass_after=intervention.donor_mass_after,
        scale=scale.value,
        scale_display_label=DISPLAY_LABELS[scale],
        experiment_budget=scale_spec.experiment_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget=scale_spec.target_budget_for_multiplier(TARGET_BUDGET_MULTIPLIER),
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=num_train_steps,
        target_final_checkpoint_step=num_train_steps - 1,
        phase_weights=intervention.phase_weights,
    )


def build_run_specs_for_scale(scale: ScalingStudyScale) -> list[PerturbationRunSpec]:
    """Build the 55 perturbed run specs for one scale."""
    return [_run_spec_for_scale(intervention, scale) for intervention in build_interventions()]


def _run_specs_as_qsplit_specs(run_specs: list[PerturbationRunSpec]) -> list[Any]:
    return [asdict(run_spec) for run_spec in run_specs]


def _run_manifest_step(
    *,
    execution_name_prefix: str,
    experiment_name: str,
    run_specs: list[PerturbationRunSpec],
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


def build_scale_launch_artifacts(
    *,
    base_name_prefix: str,
    scale: ScalingStudyScale,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
) -> ScaleLaunchArtifacts:
    """Resolve the launch graph for one scale without submitting."""
    if tpu_regions != (DEFAULT_TPU_REGION,) or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This experiment is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; got {tpu_regions}/{tpu_zone}"
        )
    scale_spec = resolve_scale_spec(scale)
    name_prefix = _scale_name_prefix(base_name_prefix, scale)
    run_specs = build_run_specs_for_scale(scale)
    execution_name_prefix = name_prefix
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
        execution_name_prefix=execution_name_prefix,
        experiment_name=name_prefix,
        run_specs=run_specs,
    )
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


def build_paired_launch_artifacts(
    *,
    base_name_prefix: str,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
) -> PairedLaunchArtifacts:
    """Resolve both scale graphs without submitting."""
    interventions = build_interventions()
    scale_artifacts = [
        build_scale_launch_artifacts(
            base_name_prefix=base_name_prefix,
            scale=scale,
            tpu_type=tpu_type,
            tpu_regions=tpu_regions,
            tpu_zone=tpu_zone,
            eval_datasets_cache_path=eval_datasets_cache_path,
            include_eval_harness=include_eval_harness,
        )
        for scale in SCALES
    ]
    artifacts = PairedLaunchArtifacts(interventions=interventions, scale_artifacts=scale_artifacts)
    validate_paired_launch_artifacts(artifacts, include_eval_harness=include_eval_harness)
    return artifacts


def validate_paired_launch_artifacts(
    artifacts: PairedLaunchArtifacts,
    *,
    include_eval_harness: bool,
) -> None:
    """Validate paired-scale graph invariants before launch."""
    if len(artifacts.interventions) != 55:
        raise ValueError(f"Expected 55 interventions, got {len(artifacts.interventions)}")
    if len(artifacts.scale_artifacts) != 2:
        raise ValueError(f"Expected 2 scale artifacts, got {len(artifacts.scale_artifacts)}")
    all_training_names = [step.name for step in artifacts.training_steps]
    if len(set(all_training_names)) != len(all_training_names):
        raise ValueError("Duplicate training step names")
    all_output_roots = [str(step.override_output_path or step.name) for step in artifacts.training_steps]
    if len(set(all_output_roots)) != len(all_output_roots):
        raise ValueError("Duplicate training output paths")
    for artifact in artifacts.scale_artifacts:
        expected_steps = resolve_scale_spec(artifact.scale).num_train_steps_for_multiplier(TARGET_BUDGET_MULTIPLIER)
        if len(artifact.run_specs) != 55:
            raise ValueError(f"{artifact.scale.value} has {len(artifact.run_specs)} specs, expected 55")
        run_names = [run_spec.run_name for run_spec in artifact.run_specs]
        if len(set(run_names)) != len(run_names):
            raise ValueError(f"{artifact.scale.value} has duplicate run names")
        for run_spec in artifact.run_specs:
            _validate_phase_weights(run_spec.phase_weights, run_name=f"{artifact.scale.value}/{run_spec.run_name}")
            if run_spec.num_train_steps != expected_steps:
                raise ValueError(f"{run_spec.run_name} has num_train_steps={run_spec.num_train_steps}")
            if run_spec.target_final_checkpoint_step != expected_steps - 1:
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


def write_local_manifests(artifacts: PairedLaunchArtifacts, output_dir: Path) -> None:
    """Write local audit artifacts for the paired launch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    intervention_rows = [_flat_manifest_row(asdict(spec)) for spec in artifacts.interventions]
    training_rows = [
        _flat_manifest_row(asdict(run_spec)) for artifact in artifacts.scale_artifacts for run_spec in artifact.run_specs
    ]
    _write_csv(output_dir / LOCAL_INTERVENTION_MANIFEST_CSV, intervention_rows)
    _write_csv(output_dir / LOCAL_TRAINING_MANIFEST_CSV, training_rows)
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "base_name_prefix": BASE_NAME_PREFIX,
                "cohort": COHORT,
                "interventions": [asdict(spec) for spec in artifacts.interventions],
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
        "domain_bump_epsilon": DOMAIN_BUMP_EPSILON,
        "family_bump_epsilon": FAMILY_BUMP_EPSILON,
        "quality_swap_fraction": QUALITY_SWAP_FRACTION,
        "intervention_count": len(artifacts.interventions),
        "training_run_count": len(training_rows),
        "scale_prefixes": {artifact.scale.value: artifact.name_prefix for artifact in artifacts.scale_artifacts},
        "scale_display_labels": {scale.value: label for scale, label in DISPLAY_LABELS.items()},
        "type_counts": {
            intervention_type.value: sum(
                1 for spec in artifacts.interventions if spec.intervention_type == intervention_type.value
            )
            for intervention_type in InterventionType
        },
        "first_10_interventions": [spec.intervention_id for spec in artifacts.interventions[:10]],
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
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_tpu_region), executor_prefix)


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
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    artifacts = build_paired_launch_artifacts(
        base_name_prefix=args.base_name_prefix,
        tpu_type=args.tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        include_eval_harness=args.include_eval_harness,
    )
    write_local_manifests(artifacts, Path(args.local_artifact_dir))
    logger.info("Wrote local manifests to %s", args.local_artifact_dir)
    logger.info(
        "Prepared %d interventions and %d training steps.",
        len(artifacts.interventions),
        len(artifacts.training_steps),
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
            f"{args.base_name_prefix}: proportional perturbation scale-transfer Stage 1. "
            f"Runs 55 perturbed mixtures at 60M/1.2B and 100M/6B. Outputs include "
            f"{RUN_MANIFEST_FILE}, {RESULTS_CSV}, {FIT_DATASET_CSV}, and {FIT_DATASET_SUMMARY_JSON} per scale."
        ),
    )


if __name__ == "__main__":
    main()
