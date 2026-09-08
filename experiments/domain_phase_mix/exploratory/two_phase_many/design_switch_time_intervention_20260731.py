# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["scipy"]
# ///
"""Preregister the WSD80 switch-time temporal-state intervention.

The design triangulates history dependence with three complementary arms:

1. fixed aggregate and fixed contrast with antithetic orderings;
2. fixed aggregate and fixed phase-1 mixture;
3. fixed aggregate and fixed phase-0 mixture.

The optimizer schedule never changes. This script writes design artifacts only;
it neither builds nor submits training jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import brentq
from scipy.stats import nct
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as wsd80  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "switch_time_intervention_design_20260731"
LAUNCHER_PATH = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd_80_20_switch_time_intervention.py"
BASE_LAUNCHER_PATH = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py"
EVALUATOR_PATH = SCRIPT_DIR / "evaluate_switch_time_intervention_20260731.py"
REPEAT_NOISE_SOURCE_PATH = (
    SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_global_optimum_fiber_observations.csv"
)
FIBER_COUNTEREXAMPLE_REPORT_PATH = (
    SCRIPT_DIR / "reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/report.md"
)

EVAL_INTERVAL_STEPS = 40
EXPERIMENT_BUDGET = 2_000_000_000
TRANSITION_MAX_STEP = 6400
SWITCH_STEPS = (3200, 4480, 5760, 6096, 6400, 6720, 7040, 7360)
OBSERVED_OPTIMUM_SWITCH_STEPS = (5760, 6096, 6720, 7040)
FIXED_LATE_SWITCH_STEPS = SWITCH_STEPS
FIXED_EARLY_SWITCH_STEPS = (3200, 4480, 5760, 6096, 6400, 6720)
TIED_BASIN_FIXED_EARLY_SWITCH_STEPS = (3200, 4480, 5760, 6096)
REPLICATED_MAIN_SWITCH_STEPS = (3200, 4480, 6096, 7040)
ASYMMETRIC_SEED_VALUES = (20260731, 20260732, 20260733)
SPINE_SEED_VALUES = (20260731, 20260732, 20260733, 20260734, 20260735, 20260736)

STARCODER_SOURCE_TOKENS = 216_567_300_822
CODE_WEIGHT_FLOOR = 0.05
CODE_WEIGHT_CEILING = 0.75
FLOAT_TOLERANCE = 1e-12

ALPHA = 0.05
POWER = 0.80
EQUIVALENCE_BPB = 0.005
POSITIVE_CONTROL_MIN_GAIN_BPB = 0.002
MIN_STATIC_SUBSPACE_RESIDUAL = 0.30
MIN_SEALED_SEPARATED_SWITCH_FOLDS = 4
MIN_SWITCH_MEMORY_FRACTION = 0.10
MIN_MEMORY_SWITCH_FOLDS = 3
MIN_RELAXATION_STEPS = EVAL_INTERVAL_STEPS
MAX_SPINE_LOLO_RMSE_BPB = 0.005
SYNTHETIC_NOISE_REGIMES = {
    "independent_steps": (0.0, 0.0),
    "mixed_autocorrelation": (0.50, 0.50),
    "persistent_autocorrelation": (0.75, 0.90),
}

ACQUISITION_RATE_GRID = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24)
FORGETTING_RATIO_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
REPETITION_POWERS = (1, 2)
REPETITION_ONSET_EPOCHS = 1.0

PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
CODE_TRANSFER_TARGETS = (
    "eval/uncheatable_eval/github_python-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
)
NEGATIVE_CONTROL_TARGETS = (
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/falcon-refinedweb-llama3/bpb",
)


@dataclass(frozen=True)
class Anchor:
    """One fixed token-average StarCoder mixture."""

    anchor_id: str
    description: str
    aggregate_code_weight: float
    main_contrast: float
    prior_role: str
    intervention_anchor: bool


@dataclass(frozen=True)
class Coordinate:
    """One unique switch-time policy coordinate."""

    coordinate_id: str
    anchor_id: str
    design_arm: str
    role: str
    pair_id: str
    cv_switch_group: str
    switch_step: int
    switch_fraction: float
    signed_contrast: float
    phase_0_code_weight: float
    phase_1_code_weight: float
    aggregate_code_weight: float
    fixed_level_kind: str
    fixed_level_value: float | str
    coordinate_sha256: str


@dataclass(frozen=True)
class Observation:
    """One seeded training observation for a unique coordinate."""

    observation_id: str
    coordinate_id: str
    anchor_id: str
    design_arm: str
    role: str
    pair_id: str
    cv_switch_group: str
    replicate_index: int
    run_seed: int
    simulated_epoch_subset_seed: int
    switch_step: int
    switch_fraction: float
    lr_decay_start_step: int
    signed_contrast: float
    phase_0_code_weight: float
    phase_1_code_weight: float
    phase_0_broad_weight: float
    phase_1_broad_weight: float
    aggregate_code_weight: float
    aggregate_error: float
    contrast_error: float
    fixed_level_kind: str
    fixed_level_value: float | str
    phase_0_steps: int
    phase_1_steps: int
    phase_0_mixture_blocks: int
    phase_0_code_tokens: float
    phase_1_code_tokens: float
    code_subset_tokens: float
    phase_0_code_epochs: float
    phase_1_code_epochs: float
    total_code_epochs: float
    normalized_token_mass_at_switch: float
    normalized_lr_mass_at_switch: float
    token_minus_lr_clock: float
    token_weighted_code_dose: float
    lr_weighted_code_dose: float
    coordinate_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_paired_sd_bpb() -> tuple[float, dict[str, Any]]:
    """Estimate same-seed paired-difference noise from the sealed WSD80 repeat fiber."""
    with REPEAT_NOISE_SOURCE_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    repeated = [row for row in rows if int(row["data_seed"]) != 20260711]
    tied_by_seed = {int(row["data_seed"]): float(row["wsd80_bpb"]) for row in repeated if int(row["fiber_index"]) == 6}
    deltas_by_coordinate: dict[int, list[float]] = {}
    for row in repeated:
        fiber_index = int(row["fiber_index"])
        if fiber_index == 6:
            continue
        seed = int(row["data_seed"])
        if seed not in tied_by_seed:
            continue
        deltas_by_coordinate.setdefault(fiber_index, []).append(float(row["wsd80_bpb"]) - tied_by_seed[seed])
    coordinate_sds = [statistics.stdev(values) for values in deltas_by_coordinate.values() if len(values) >= 2]
    if len(coordinate_sds) < 3:
        raise ValueError("Historical repeat fiber does not contain enough paired coordinates")
    pooled_rms_sd = math.sqrt(sum(value**2 for value in coordinate_sds) / len(coordinate_sds))
    return pooled_rms_sd, {
        "source_path": str(REPEAT_NOISE_SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_sha256": sha256_path(REPEAT_NOISE_SOURCE_PATH),
        "reference_tied_fiber_index": 6,
        "paired_coordinates": len(coordinate_sds),
        "seeds_per_coordinate": sorted({len(values) for values in deltas_by_coordinate.values()}),
        "coordinate_paired_sds_bpb": coordinate_sds,
        "pooled_rms_paired_sd_bpb": pooled_rms_sd,
    }


def csv_string_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the exact string representation observed by csv.DictReader."""
    return [{key: str(value) for key, value in row.items()} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def realized_weights(aggregate: float, signed_contrast: float, switch_fraction: float) -> tuple[float, float]:
    """Return phase weights at fixed aggregate and phase-1-minus-phase-0 contrast."""
    phase_0 = aggregate - (1.0 - switch_fraction) * signed_contrast
    phase_1 = aggregate + switch_fraction * signed_contrast
    return phase_0, phase_1


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int, decay_start_step: int) -> float:
    """Reproduce the normalized canonical WSD warmup-stable-cosine schedule."""
    if not 0 <= step < total_steps:
        raise ValueError(f"Step outside training range: {step}")
    if step < warmup_steps:
        return step / warmup_steps
    if step < decay_start_step:
        return 1.0
    decay_steps = total_steps - decay_start_step
    decay_progress = (step - decay_start_step) / decay_steps
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def normalized_lr_mass_by_step(*, total_steps: int, warmup_steps: int, decay_start_step: int) -> tuple[float, ...]:
    """Return cumulative normalized LR mass before every training step."""
    multipliers = [
        learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_start_step=decay_start_step,
        )
        for step in range(total_steps)
    ]
    total_mass = sum(multipliers)
    cumulative = [0.0]
    for multiplier in multipliers:
        cumulative.append(cumulative[-1] + multiplier / total_mass)
    return tuple(cumulative)


def build_anchors(total_steps: int, lr_boundary_step: int) -> tuple[Anchor, ...]:
    canonical_fraction = lr_boundary_step / total_steps
    observed_global_aggregate = canonical_fraction * 0.1 + (1.0 - canonical_fraction) * 0.5
    return (
        Anchor(
            anchor_id="off_optimum_code_anchor",
            description="Realized aggregate of the original 1B Programming-Languages optimum (p0=0.1, p1=0.5)",
            aggregate_code_weight=observed_global_aggregate,
            main_contrast=0.13,
            prior_role="off-basin control where a large aggregate-conditioned phase effect is plausible",
            intervention_anchor=True,
        ),
        Anchor(
            anchor_id="lower_spine_anchor",
            description="Tied-only support between the off-basin and tied-optimal plateau",
            aggregate_code_weight=0.25,
            main_contrast=0.0,
            prior_role="aggregate-spine derivative identification only",
            intervention_anchor=False,
        ),
        Anchor(
            anchor_id="tied_basin_lower_anchor",
            description="2B measured-grid tied optimum with a replicated gain at d=+0.20",
            aggregate_code_weight=0.35,
            main_contrast=0.20,
            prior_role="positive control inside the tied-optimal uncertainty region",
            intervention_anchor=True,
        ),
        Anchor(
            anchor_id="tied_basin_upper_anchor",
            description="Adjacent 2B tied-basin comparator where d=+0.20 was unresolved",
            aggregate_code_weight=0.40,
            main_contrast=0.20,
            prior_role="aggregate-conditioned contrast to the lower tied-basin anchor",
            intervention_anchor=True,
        ),
        Anchor(
            anchor_id="high_code_anchor",
            description="Aggregate above the 2B tied-optimal plateau",
            aggregate_code_weight=0.50,
            main_contrast=0.25,
            prior_role="high-code control on the opposite side of the tied basin",
            intervention_anchor=True,
        ),
    )


def make_coordinate(
    *,
    coordinate_index: int,
    anchor: Anchor,
    design_arm: str,
    role: str,
    pair_id: str,
    switch_step: int,
    phase_0_code_weight: float,
    phase_1_code_weight: float,
    fixed_level_kind: str,
    fixed_level_value: float | str,
    total_steps: int,
) -> Coordinate:
    switch_fraction = switch_step / total_steps
    aggregate = switch_fraction * phase_0_code_weight + (1.0 - switch_fraction) * phase_1_code_weight
    signed_contrast = phase_1_code_weight - phase_0_code_weight
    identity = {
        "anchor_id": anchor.anchor_id,
        "design_arm": design_arm,
        "role": role,
        "switch_step": switch_step,
        "phase_0_code_weight": phase_0_code_weight,
        "phase_1_code_weight": phase_1_code_weight,
        "aggregate_code_weight": aggregate,
        "fixed_level_kind": fixed_level_kind,
        "fixed_level_value": fixed_level_value,
    }
    return Coordinate(
        coordinate_id=f"sti_{coordinate_index:03d}",
        anchor_id=anchor.anchor_id,
        design_arm=design_arm,
        role=role,
        pair_id=pair_id,
        cv_switch_group="none" if role.endswith("tied_control") else f"switch_{switch_step}",
        switch_step=switch_step,
        switch_fraction=switch_fraction,
        signed_contrast=signed_contrast,
        phase_0_code_weight=phase_0_code_weight,
        phase_1_code_weight=phase_1_code_weight,
        aggregate_code_weight=aggregate,
        fixed_level_kind=fixed_level_kind,
        fixed_level_value=fixed_level_value,
        coordinate_sha256=sha256_json(identity),
    )


def build_coordinates(anchors: tuple[Anchor, ...], total_steps: int, lr_boundary_step: int) -> list[Coordinate]:
    coordinates: list[Coordinate] = []

    def add(
        anchor: Anchor,
        *,
        design_arm: str,
        role: str,
        pair_id: str,
        switch_step: int,
        phase_0_code_weight: float,
        phase_1_code_weight: float,
        fixed_level_kind: str = "none",
        fixed_level_value: float | str = "",
    ) -> None:
        coordinates.append(
            make_coordinate(
                coordinate_index=len(coordinates),
                anchor=anchor,
                design_arm=design_arm,
                role=role,
                pair_id=pair_id,
                switch_step=switch_step,
                phase_0_code_weight=phase_0_code_weight,
                phase_1_code_weight=phase_1_code_weight,
                fixed_level_kind=fixed_level_kind,
                fixed_level_value=fixed_level_value,
                total_steps=total_steps,
            )
        )

    intervention_anchors = tuple(anchor for anchor in anchors if anchor.intervention_anchor)

    for anchor in intervention_anchors:
        for switch_step in SWITCH_STEPS:
            switch_fraction = switch_step / total_steps
            pair_id = f"{anchor.anchor_id}_main_s{switch_step}"
            for sign, role in ((1.0, "plus"), (-1.0, "minus")):
                phase_0, phase_1 = realized_weights(
                    anchor.aggregate_code_weight,
                    sign * anchor.main_contrast,
                    switch_fraction,
                )
                add(
                    anchor,
                    design_arm="fixed_aggregate_contrast",
                    role=f"main_antithetic_{role}",
                    pair_id=pair_id,
                    switch_step=switch_step,
                    phase_0_code_weight=phase_0,
                    phase_1_code_weight=phase_1,
                    fixed_level_kind="contrast",
                    fixed_level_value=sign * anchor.main_contrast,
                )
    off_optimum_anchor = intervention_anchors[0]
    for switch_step in OBSERVED_OPTIMUM_SWITCH_STEPS:
        switch_fraction = switch_step / total_steps
        phase_0, phase_1 = realized_weights(
            off_optimum_anchor.aggregate_code_weight,
            0.40,
            switch_fraction,
        )
        add(
            off_optimum_anchor,
            design_arm="legacy_surface_optimum_contrast",
            role="one_sided_off_basin_control",
            pair_id=f"{off_optimum_anchor.anchor_id}_observed_s{switch_step}",
            switch_step=switch_step,
            phase_0_code_weight=phase_0,
            phase_1_code_weight=phase_1,
            fixed_level_kind="contrast",
            fixed_level_value=0.40,
        )
    fixed_late_weights = {
        "off_optimum_code_anchor": 0.25,
        "tied_basin_lower_anchor": 0.55,
        "tied_basin_upper_anchor": 0.60,
        "high_code_anchor": 0.70,
    }
    for anchor in intervention_anchors:
        fixed_late_weight = fixed_late_weights[anchor.anchor_id]
        for switch_step in FIXED_LATE_SWITCH_STEPS:
            switch_fraction = switch_step / total_steps
            phase_0 = (anchor.aggregate_code_weight - (1.0 - switch_fraction) * fixed_late_weight) / switch_fraction
            pair_id = f"{anchor.anchor_id}_fixed_p1_s{switch_step}"
            add(
                anchor,
                design_arm="fixed_late_mixture",
                role="fixed_phase_1",
                pair_id=pair_id,
                switch_step=switch_step,
                phase_0_code_weight=phase_0,
                phase_1_code_weight=fixed_late_weight,
                fixed_level_kind="phase_1_code_weight",
                fixed_level_value=fixed_late_weight,
            )
    tied_basin_lower_anchor = next(
        anchor for anchor in intervention_anchors if anchor.anchor_id == "tied_basin_lower_anchor"
    )
    fixed_early_arms = (
        (off_optimum_anchor, 0.12, FIXED_EARLY_SWITCH_STEPS),
        (tied_basin_lower_anchor, 0.25, TIED_BASIN_FIXED_EARLY_SWITCH_STEPS),
    )
    for anchor, fixed_early_weight, switch_steps in fixed_early_arms:
        for switch_step in switch_steps:
            switch_fraction = switch_step / total_steps
            phase_1 = (anchor.aggregate_code_weight - switch_fraction * fixed_early_weight) / (1.0 - switch_fraction)
            pair_id = f"{anchor.anchor_id}_fixed_p0_s{switch_step}"
            add(
                anchor,
                design_arm="fixed_early_mixture",
                role="fixed_phase_0",
                pair_id=pair_id,
                switch_step=switch_step,
                phase_0_code_weight=fixed_early_weight,
                phase_1_code_weight=phase_1,
                fixed_level_kind="phase_0_code_weight",
                fixed_level_value=fixed_early_weight,
            )
    for anchor in anchors:
        add(
            anchor,
            design_arm="aggregate_spine_tied",
            role="spine_tied_control",
            pair_id=f"{anchor.anchor_id}_spine",
            switch_step=lr_boundary_step,
            phase_0_code_weight=anchor.aggregate_code_weight,
            phase_1_code_weight=anchor.aggregate_code_weight,
        )

    return coordinates


def coordinate_replicates(item: Coordinate) -> int:
    """Return the preregistered number of seeds for one design coordinate."""
    if item.role == "spine_tied_control":
        return len(SPINE_SEED_VALUES)
    if item.design_arm in {"fixed_late_mixture", "fixed_early_mixture"}:
        return len(ASYMMETRIC_SEED_VALUES)
    if "_main_" in item.pair_id and item.switch_step in REPLICATED_MAIN_SWITCH_STEPS:
        return len(ASYMMETRIC_SEED_VALUES)
    if "_observed_" in item.pair_id and item.switch_step == 6096:
        return len(ASYMMETRIC_SEED_VALUES)
    return 1


def build_observations(
    coordinates: list[Coordinate],
    *,
    anchors: tuple[Anchor, ...],
    total_steps: int,
    lr_boundary_step: int,
    warmup_steps: int,
    materialized_tokens: int,
    target_budget: int,
) -> list[Observation]:
    observations: list[Observation] = []
    anchor_aggregates = {anchor.anchor_id: anchor.aggregate_code_weight for anchor in anchors}
    lr_mass = normalized_lr_mass_by_step(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        decay_start_step=lr_boundary_step,
    )
    tokens_per_step = wsd80.BATCH_SIZE * wsd80.SEQ_LEN
    code_subset_tokens = STARCODER_SOURCE_TOKENS * materialized_tokens / target_budget
    for item in coordinates:
        seeds = SPINE_SEED_VALUES if item.role == "spine_tied_control" else ASYMMETRIC_SEED_VALUES
        for replicate_index, run_seed in enumerate(seeds[: coordinate_replicates(item)]):
            phase_0_steps = item.switch_step
            phase_1_steps = total_steps - item.switch_step
            phase_0_code_tokens = phase_0_steps * tokens_per_step * item.phase_0_code_weight
            phase_1_code_tokens = phase_1_steps * tokens_per_step * item.phase_1_code_weight
            phase_0_epochs = phase_0_code_tokens / code_subset_tokens
            phase_1_epochs = phase_1_code_tokens / code_subset_tokens
            normalized_lr_mass = lr_mass[item.switch_step]
            lr_weighted_dose = (
                normalized_lr_mass * item.phase_0_code_weight + (1.0 - normalized_lr_mass) * item.phase_1_code_weight
            )
            aggregate = (
                item.switch_fraction * item.phase_0_code_weight + (1.0 - item.switch_fraction) * item.phase_1_code_weight
            )
            observations.append(
                Observation(
                    observation_id=f"sti_obs_{len(observations):03d}",
                    coordinate_id=item.coordinate_id,
                    anchor_id=item.anchor_id,
                    design_arm=item.design_arm,
                    role=item.role,
                    pair_id=item.pair_id,
                    cv_switch_group=item.cv_switch_group,
                    replicate_index=replicate_index,
                    run_seed=run_seed,
                    simulated_epoch_subset_seed=run_seed,
                    switch_step=item.switch_step,
                    switch_fraction=item.switch_fraction,
                    lr_decay_start_step=lr_boundary_step,
                    signed_contrast=item.signed_contrast,
                    phase_0_code_weight=item.phase_0_code_weight,
                    phase_1_code_weight=item.phase_1_code_weight,
                    phase_0_broad_weight=1.0 - item.phase_0_code_weight,
                    phase_1_broad_weight=1.0 - item.phase_1_code_weight,
                    aggregate_code_weight=item.aggregate_code_weight,
                    aggregate_error=aggregate - anchor_aggregates[item.anchor_id],
                    contrast_error=(
                        (item.phase_1_code_weight - item.phase_0_code_weight) - float(item.fixed_level_value)
                        if item.fixed_level_kind == "contrast"
                        else 0.0
                    ),
                    fixed_level_kind=item.fixed_level_kind,
                    fixed_level_value=item.fixed_level_value,
                    phase_0_steps=phase_0_steps,
                    phase_1_steps=phase_1_steps,
                    phase_0_mixture_blocks=phase_0_steps * wsd80.BATCH_SIZE // wsd80.MIXTURE_BLOCK_SIZE,
                    phase_0_code_tokens=phase_0_code_tokens,
                    phase_1_code_tokens=phase_1_code_tokens,
                    code_subset_tokens=code_subset_tokens,
                    phase_0_code_epochs=phase_0_epochs,
                    phase_1_code_epochs=phase_1_epochs,
                    total_code_epochs=phase_0_epochs + phase_1_epochs,
                    normalized_token_mass_at_switch=item.switch_fraction,
                    normalized_lr_mass_at_switch=normalized_lr_mass,
                    token_minus_lr_clock=item.switch_fraction - normalized_lr_mass,
                    token_weighted_code_dose=aggregate,
                    lr_weighted_code_dose=lr_weighted_dose,
                    coordinate_sha256=item.coordinate_sha256,
                )
            )
    return observations


def build_eval_schedule(total_steps: int) -> list[dict[str, Any]]:
    final_checkpoint_step = total_steps - 1
    transition_steps = list(range(EVAL_INTERVAL_STEPS, TRANSITION_MAX_STEP + 1, EVAL_INTERVAL_STEPS))
    if not transition_steps or transition_steps[-1] != TRANSITION_MAX_STEP:
        raise ValueError(f"Unexpected transition evaluation schedule: {transition_steps[-3:]}")
    sealed_steps = list(range(TRANSITION_MAX_STEP + EVAL_INTERVAL_STEPS, total_steps, EVAL_INTERVAL_STEPS))
    eval_steps = [*transition_steps, *sealed_steps, final_checkpoint_step]
    rows: list[dict[str, Any]] = []
    for switch_step in SWITCH_STEPS:
        for eval_step in eval_steps:
            relation = "pre_switch"
            if eval_step >= switch_step and eval_step - EVAL_INTERVAL_STEPS < switch_step:
                relation = "first_post_switch_evaluation"
            elif eval_step > switch_step:
                relation = "post_switch"
            rows.append(
                {
                    "switch_step": switch_step,
                    "switch_fraction": switch_step / total_steps,
                    "eval_step": eval_step,
                    "relation_to_switch": relation,
                    "steps_from_switch": eval_step - switch_step,
                    "endpoint_role": (
                        "transition_identification" if eval_step <= TRANSITION_MAX_STEP else "sealed_decay_falsification"
                    ),
                }
            )
    return rows


def minimum_detectable_effect(sd: float, repeats: int) -> float:
    """Return the two-sided paired-t MDE at the frozen alpha and power."""
    if sd <= 0.0:
        raise ValueError("Paired-difference SD must be positive")
    if repeats < 2:
        raise ValueError("Paired-t power requires at least two repeats")
    degrees_of_freedom = repeats - 1
    critical_value = float(student_t.ppf(1.0 - ALPHA / 2.0, degrees_of_freedom))

    def achieved_power(effect: float) -> float:
        noncentrality = effect * math.sqrt(repeats) / sd
        return float(
            nct.cdf(-critical_value, degrees_of_freedom, noncentrality)
            + nct.sf(critical_value, degrees_of_freedom, noncentrality)
        )

    upper = sd
    while achieved_power(upper) < POWER:
        upper *= 2.0
    return float(brentq(lambda effect: achieved_power(effect) - POWER, 0.0, upper))


def mechanism_definitions() -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": "token_dose_null",
            "state": "none",
            "transition": "D_tok(s)=integral q(u) du",
            "response": "Delta L=c[D_tok(two-phase)-D_tok(tied)] with zero intercept",
            "free_transition_parameters": 0,
            "expected_signature": "zero endpoint effect at fixed token-average aggregate",
            "falsification": "nonzero leave-switch-out trajectory or endpoint structure",
        },
        {
            "mechanism_id": "lr_mass_dose_null",
            "state": "none",
            "transition": "D_lr(s)=integral q(u) dM(u), where M is normalized canonical LR mass",
            "response": "Delta L=c[D_lr(two-phase)-D_lr(tied)] with zero intercept",
            "free_transition_parameters": 0,
            "expected_signature": "dose follows the frozen LR kernel without post-switch lag",
            "falsification": "fixed-p1 history dependence or lagged relaxation",
        },
        {
            "mechanism_id": "terminal_level_null",
            "state": "current mixture q(s) only",
            "transition": "instantaneous jump from p0 to p1 at the data switch",
            "response": "Delta L=c[q_current(two-phase)-a] with zero intercept",
            "free_transition_parameters": 0,
            "expected_signature": "same post-switch level for every fixed-p1 coordinate",
            "falsification": "fixed-p1 trajectories retain switch-history dependence",
        },
        {
            "mechanism_id": "phase_local_repetition",
            "state": "phase-local materialized code epochs",
            "transition": "R_h=sum_phase max(E_phase-1,0)^h, h in {1,2}",
            "response": "Delta L=c[R_h(two-phase)-R_h(tied)] with zero intercept",
            "free_transition_parameters": 1,
            "expected_signature": "predominantly even asymmetry harm without lag",
            "falsification": "signed history dependence remains after exact phase-epoch accounting",
        },
        {
            "mechanism_id": "token_clock_acquisition_forgetting",
            "state": "bounded retained code response x in [0,1]",
            "transition": "dx/du=k_a*r_code(u)*(1-x)-k_f*(1-q(u))*x",
            "response": (
                "d_dyn=x(two-phase)-x(tied); compare the aggregate-potential null "
                "[A'_t(a)d_dyn,d_dyn^2], signed aggregate-linear "
                "[d_dyn,(a-0.35)d_dyn,d_dyn^2], and signed aggregate-invariant "
                "[d_dyn,d_dyn^2] zero-intercept heads; signed heads leave the even response unconstrained"
            ),
            "free_transition_parameters": 2,
            "expected_signature": (
                "post-switch relaxation collapses in token progress and ordering value changes across aggregate"
            ),
            "falsification": "LR-mass clock, static shock, or memoryless null wins blocked post-switch CV",
        },
        {
            "mechanism_id": "lr_clock_acquisition_forgetting",
            "state": "bounded retained code response x in [0,1]",
            "transition": "dx/dM=k_a*r_code(M)*(1-x)-k_f*(1-q(M))*x",
            "response": (
                "d_dyn=x(two-phase)-x(tied); compare the aggregate-potential null "
                "[A'_t(a)d_dyn,d_dyn^2], signed aggregate-linear "
                "[d_dyn,(a-0.35)d_dyn,d_dyn^2], and signed aggregate-invariant "
                "[d_dyn,d_dyn^2] zero-intercept heads; signed heads leave the even response unconstrained"
            ),
            "free_transition_parameters": 2,
            "expected_signature": "state movement contracts nonlinearly through cosine decay",
            "falsification": "token clock, static shock, or LR-dose null wins blocked post-switch CV",
        },
        {
            "mechanism_id": "static_switch_control_null",
            "state": "instantaneous persistent impulse proxy d_static=lr(switch)*(p1-p0) after the switch",
            "transition": "d_static=0 before the switch; d_static=lr(switch)*(p1-p0) after the switch",
            "response": (
                "compare aggregate-potential, signed aggregate-linear, and signed aggregate-invariant "
                "zero-intercept heads on d_static"
            ),
            "free_transition_parameters": 0,
            "expected_signature": "immediate non-decaying response with optionally aggregate-varying utility",
            "falsification": "a bounded relaxing state improves blocked post-switch trajectories beyond uncertainty",
        },
    ]


def quantitative_gates(mde: float) -> list[dict[str, Any]]:
    return [
        {
            "gate_id": "positive_control_reproduction",
            "scope": "2B a=0.35,d=+0.20 at the canonical WSD boundary versus its same-seed tied control",
            "metric": PRIMARY_TARGET,
            "threshold": f"mean two-phase-minus-tied <= -{POSITIVE_CONTROL_MIN_GAIN_BPB:.3f} BPB",
            "uncertainty": (
                f"three fresh paired seeds; exact paired-t MDE={mde:.6f} BPB; prior five-seed evidence is not "
                "pooled into this diagnostic"
            ),
            "role": (
                "descriptive signal-regime check motivated by the independently replicated 2B fiber "
                "counterexample; it does not license endpoint conclusions"
            ),
        },
        {
            "gate_id": "tied_basin_signed_response",
            "scope": "blocked predictions and observed gains throughout the tied-optimal uncertainty region",
            "metric": "signed gain, calibration, and uncertainty by aggregate and switch",
            "threshold": (
                "no response sign is assumed; aggregate-potential, signed aggregate-linear, and signed "
                "aggregate-invariant heads must compete under identical blocked folds"
            ),
            "uncertainty": "coordinate/seed cluster bootstrap with tied-spine uncertainty propagated",
            "role": (
                "diagnostic and model-form falsification; the replicated 2B gain prohibits a hard fiber-null sign gate"
            ),
        },
        {
            "gate_id": "response_form_selection",
            "scope": "all asymmetric transition rows on the primary target",
            "metric": "leave-switch-out and leave-anchor-out RMSE, bias, amplitude ratio, and optimism",
            "threshold": (
                "compare all three forms descriptively under identical blocked folds; direct comparisons do not "
                "license endpoint access because the response form was selected on those outcomes"
            ),
            "uncertainty": "cluster bootstrap by shared tied-control anchor and seed",
            "role": "prevents the fiber hypothesis from being imposed through coefficient bounds",
        },
        {
            "gate_id": "nested_selection_stability",
            "scope": "all leave-switch-out and leave-anchor-out outer folds",
            "metric": "selected dynamic mechanism status and signed response mode; family and clock are diagnostic",
            "threshold": (
                "every outer fold selects a dynamic mechanism with the same response mode as the full-data "
                "selection; clock and nonlinear rates may vary"
            ),
            "uncertainty": "fold-by-fold stability table; nonlinear rates may vary and are reported",
            "role": (
                "ensures sealed blocked predictions test the licensed temporal mechanism rather than static fallbacks"
            ),
        },
        {
            "gate_id": "dynamic_static_feature_separation",
            "scope": "outcome-independent candidate features in every transition and sealed switch fold",
            "metric": (
                "relative residual after projecting normalized dynamic features onto the joint memoryless-null span"
            ),
            "threshold": (
                f"transition folds: minimum eligible residual >= {MIN_STATIC_SUBSPACE_RESIDUAL:.2f}; sealed design: "
                f"global residual >= {MIN_STATIC_SUBSPACE_RESIDUAL:.2f} and at least "
                f"{MIN_SEALED_SEPARATED_SWITCH_FOLDS} switch folds individually clear that floor"
            ),
            "uncertainty": "none; this is a deterministic design-matrix identifiability check",
            "role": (
                "required before a selected dynamic state can license endpoint access; every sealed fold is reported, "
                "but folds where the dynamic trajectory has mechanistically converged to a memoryless limit need not "
                "remain individually separable. Sealed-design separation is computed before outcomes are read"
            ),
        },
        {
            "gate_id": "aggregate_spine_identification",
            "scope": f"primary-target tied controls through step {TRANSITION_MAX_STEP}",
            "metric": "leave-one-aggregate-level-out RMSE; seed-bootstrap optimum stability is diagnostic only",
            "threshold": f"LOLO RMSE <= {MAX_SPINE_LOLO_RMSE_BPB:.3f} BPB",
            "uncertainty": "six independently trained seeds at each of five tied aggregate levels",
            "role": (
                "required only if the aggregate-potential null is selected; direct signed heads do not inherit "
                "fiber optimality through a tied-spine derivative"
            ),
        },
        {
            "gate_id": "trajectory_structure",
            "scope": f"post-switch intermediate evaluations through step {TRANSITION_MAX_STEP}",
            "metric": "limb-balanced leave-switch-out and leave-anchor-out RMSE on pre- and post-switch rows",
            "threshold": "candidate improves at least 5% over zero, token-dose, LR-dose, and terminal-level nulls",
            "uncertainty": "paired cluster-bootstrap candidate-minus-null RMSE upper bound < 0",
            "role": "structural selection",
        },
        {
            "gate_id": "history_identification",
            "scope": "four-anchor fixed-phase-1 and two-anchor fixed-phase-0 arms; pre- and post-switch rows",
            "metric": "coordinate-equal-weight leave-switch-out RMSE",
            "threshold": "candidate improves at least 5% over both terminal-level and static-shock nulls",
            "uncertainty": "paired cluster-bootstrap by singleton coordinate; RMSE upper bound < 0",
            "role": "required to claim temporal memory",
        },
        {
            "gate_id": "clock_and_rate_identification",
            "scope": "informative asymmetric switch folds retaining phase-0 initial-condition memory",
            "metric": "paired-bootstrap token-clock minus LR-clock blocked RMSE and rate bootstrap distribution",
            "threshold": (
                "dynamic state beats the static-shock null; exact clock/rate claims require bootstrap separation"
            ),
            "uncertainty": "report uncertainty and abstain on clock/rate identity when intervals overlap",
            "role": "mechanistic identification; no vote-count gate",
        },
        {
            "gate_id": "timescale_identification",
            "scope": "dynamic-state candidates before endpoint access",
            "metric": "retained phase-0 initial-condition memory and post-switch relaxation time",
            "threshold": (
                f"at least {MIN_MEMORY_SWITCH_FOLDS} switch folds retain >= "
                f"{MIN_SWITCH_MEMORY_FRACTION:.2f} initial-condition memory and relaxation lasts >= "
                f"{MIN_RELAXATION_STEPS} steps"
            ),
            "uncertainty": "deterministic state audit plus seed-bootstrap relaxation interval",
            "role": "prevents saturated rate-grid boundaries from licensing an apparent memory mechanism",
        },
        {
            "gate_id": "sealed_final_transfer",
            "scope": f"post-switch rows of the cosine-decay trajectory after step {TRANSITION_MAX_STEP}",
            "metric": "sealed post-switch trajectory and endpoint delta RMSE",
            "threshold": "at least 10% improvement over zero and every memoryless null on the primary target",
            "uncertainty": "leave-switch-out predictions; paired bootstrap by antithetic pair or singleton",
            "role": "frozen long-horizon decay transfer; no sealed point may alter form, rates, or response mode",
        },
        {
            "gate_id": "code_target_transfer",
            "scope": "GitHub Python and C++ targets",
            "metric": "endpoint delta RMSE and sign",
            "threshold": "no target worsens its strongest memoryless null by more than 5%; signs reported separately",
            "uncertainty": "paired cluster bootstrap by coordinate",
            "role": "transfer, not independent multiplicity",
        },
        {
            "gate_id": "broad_text_response_bounds",
            "scope": "C4 English and Falcon RefinedWeb",
            "metric": "largest predicted phase gain",
            "threshold": f"no false gain greater than {EQUIVALENCE_BPB:.3f} BPB",
            "uncertainty": "blocked-fold predictions only",
            "role": "diagnostic transfer bound; broad-text phase responses are not assumed to be zero",
        },
    ]


def validate_design(
    *,
    anchors: tuple[Anchor, ...],
    coordinates: list[Coordinate],
    observations: list[Observation],
    eval_schedule: list[dict[str, Any]],
    total_steps: int,
    lr_boundary_step: int,
    mixture_alignment: int,
) -> dict[str, Any]:
    failures: list[str] = []
    for item in observations:
        code_weights = (item.phase_0_code_weight, item.phase_1_code_weight)
        if min(code_weights) < CODE_WEIGHT_FLOOR - FLOAT_TOLERANCE:
            failures.append(f"StarCoder weight below floor: {item.observation_id}")
        if max(code_weights) > CODE_WEIGHT_CEILING + FLOAT_TOLERANCE:
            failures.append(f"StarCoder weight above ceiling: {item.observation_id}")
        if abs(item.aggregate_error) > FLOAT_TOLERANCE:
            failures.append(f"aggregate mismatch: {item.observation_id} {item.aggregate_error}")
        if abs(item.contrast_error) > FLOAT_TOLERANCE:
            failures.append(f"contrast mismatch: {item.observation_id} {item.contrast_error}")
        if item.switch_step % mixture_alignment:
            failures.append(f"mixture misalignment: {item.observation_id}")
        if item.phase_0_steps * wsd80.BATCH_SIZE % wsd80.MIXTURE_BLOCK_SIZE:
            failures.append(f"partial phase-0 mixture block: {item.observation_id}")
        if item.lr_decay_start_step != lr_boundary_step:
            failures.append(f"LR boundary changed: {item.observation_id}")

    antithetic_groups: dict[tuple[str, int], set[int]] = {}
    for item in coordinates:
        if item.design_arm != "fixed_aggregate_contrast":
            continue
        key = (item.anchor_id, item.switch_step)
        antithetic_groups.setdefault(key, set()).add(int(math.copysign(1, item.signed_contrast)))
    incomplete_pairs = [key for key, signs in antithetic_groups.items() if signs != {-1, 1}]
    if incomplete_pairs:
        failures.append(f"incomplete antithetic pairs: {incomplete_pairs}")

    off_optimum_anchor = anchors[0]
    canonical = [
        item
        for item in coordinates
        if item.design_arm == "legacy_surface_optimum_contrast" and item.switch_step == lr_boundary_step
    ]
    if len(canonical) != 1:
        failures.append("canonical observed-optimum coordinate missing or duplicated")
    else:
        item = canonical[0]
        if abs(item.phase_0_code_weight - 0.1) > FLOAT_TOLERANCE:
            failures.append(f"canonical p0 mismatch: {item.phase_0_code_weight}")
        if abs(item.phase_1_code_weight - 0.5) > FLOAT_TOLERANCE:
            failures.append(f"canonical p1 mismatch: {item.phase_1_code_weight}")
        if abs(item.aggregate_code_weight - off_optimum_anchor.aggregate_code_weight) > FLOAT_TOLERANCE:
            failures.append("canonical aggregate mismatch")

    fixed_late = [item for item in coordinates if item.design_arm == "fixed_late_mixture"]
    if len(fixed_late) != 4 * len(FIXED_LATE_SWITCH_STEPS):
        failures.append("fixed-late arm size mismatch")
    expected_fixed_late = {
        "off_optimum_code_anchor": 0.25,
        "tied_basin_lower_anchor": 0.55,
        "tied_basin_upper_anchor": 0.60,
        "high_code_anchor": 0.70,
    }
    if any(abs(item.phase_1_code_weight - expected_fixed_late[item.anchor_id]) > FLOAT_TOLERANCE for item in fixed_late):
        failures.append("fixed-late arm does not hold its anchor-specific phase 1 level fixed")

    fixed_early = [item for item in coordinates if item.design_arm == "fixed_early_mixture"]
    expected_fixed_early_count = len(FIXED_EARLY_SWITCH_STEPS) + len(TIED_BASIN_FIXED_EARLY_SWITCH_STEPS)
    if len(fixed_early) != expected_fixed_early_count:
        failures.append("fixed-early arm size mismatch")
    expected_fixed_early = {
        "off_optimum_code_anchor": 0.12,
        "tied_basin_lower_anchor": 0.25,
    }
    if any(
        abs(item.phase_0_code_weight - expected_fixed_early[item.anchor_id]) > FLOAT_TOLERANCE for item in fixed_early
    ):
        failures.append("fixed-early arm does not hold its anchor-specific phase 0 level fixed")

    if len({item.coordinate_id for item in coordinates}) != len(coordinates):
        failures.append("duplicate coordinate ID")
    if len({item.observation_id for item in observations}) != len(observations):
        failures.append("duplicate observation ID")

    spine_coordinates = [item for item in coordinates if item.role == "spine_tied_control"]
    if len(spine_coordinates) != len(anchors):
        failures.append("each aggregate anchor must have exactly one tied-spine coordinate")
    if {item.anchor_id for item in spine_coordinates} != {anchor.anchor_id for anchor in anchors}:
        failures.append("tied-spine coordinates do not cover every aggregate anchor")

    replicate_counts = Counter(item.coordinate_id for item in observations)
    expected_replicated = {item.coordinate_id for item in coordinates if coordinate_replicates(item) > 1}
    observed_replicated = {coordinate_id for coordinate_id, count in replicate_counts.items() if count > 1}
    if expected_replicated != observed_replicated:
        failures.append("replicate block differs from the preregistered coordinates")
    if any(count not in (1, 3, 6) for count in replicate_counts.values()):
        failures.append("unexpected replicate count")
    if any(replicate_counts[item.coordinate_id] != len(SPINE_SEED_VALUES) for item in spine_coordinates):
        failures.append("every tied-spine coordinate must use the six-seed block")

    fold_counts = Counter(item.switch_step for item in coordinates if not item.role.endswith("tied_control"))
    if set(fold_counts) != set(SWITCH_STEPS):
        failures.append("switch folds do not cover the frozen ladder")
    if min(fold_counts.values()) < 12 or max(fold_counts.values()) > 15:
        failures.append(f"unbalanced coordinate folds: {dict(fold_counts)}")
    if max(fold_counts.values()) / sum(fold_counts.values()) > 0.20:
        failures.append("one switch fold exceeds 20% of asymmetric coordinates")

    post_switch_counts = Counter()
    for row in eval_schedule:
        if row["endpoint_role"] == "transition_identification" and row["relation_to_switch"] in {
            "first_post_switch_evaluation",
            "post_switch",
        }:
            post_switch_counts[int(row["switch_step"])] += 1
    relaxation_switches = tuple(step for step in SWITCH_STEPS if step <= TRANSITION_MAX_STEP - MIN_RELAXATION_STEPS)
    if any(post_switch_counts[step] < 2 for step in relaxation_switches):
        failures.append(f"insufficient post-switch evaluations: {dict(post_switch_counts)}")
    for switch_step in SWITCH_STEPS:
        post_switch_counts.setdefault(switch_step, 0)

    expected_coordinates = 115
    expected_observations = 290
    if len(coordinates) != expected_coordinates:
        failures.append(f"expected {expected_coordinates} coordinates, got {len(coordinates)}")
    if len(observations) != expected_observations:
        failures.append(f"expected {expected_observations} observations, got {len(observations)}")

    if failures:
        raise ValueError("Switch-time design failed:\n- " + "\n- ".join(failures))

    paired_sd, noise_source = historical_paired_sd_bpb()
    clock_departures = {
        item.switch_step: item.token_minus_lr_clock
        for item in observations
        if item.replicate_index == 0 and not item.role.endswith("tied_control")
    }
    return {
        "passed": True,
        "total_steps": total_steps,
        "final_checkpoint_step": total_steps - 1,
        "lr_boundary_step": lr_boundary_step,
        "mixture_alignment_steps": mixture_alignment,
        "eval_interval_steps": EVAL_INTERVAL_STEPS,
        "anchors": len(anchors),
        "unique_coordinates": len(coordinates),
        "observations": len(observations),
        "antithetic_pairs": len(antithetic_groups),
        "replicated_coordinates": len(observed_replicated),
        "coordinate_fold_counts": dict(sorted(fold_counts.items())),
        "post_switch_intermediate_eval_counts": dict(sorted(post_switch_counts.items())),
        "clock_departures": dict(sorted(clock_departures.items())),
        "fixed_late_coordinates": len(fixed_late),
        "fixed_early_coordinates": len(fixed_early),
        "spine_tied_coordinates": sum(item.role == "spine_tied_control" for item in coordinates),
        "spine_tied_observations": sum(item.role == "spine_tied_control" for item in observations),
        "distinct_tied_training_streams": len(
            {(item.anchor_id, item.run_seed) for item in observations if item.role == "spine_tied_control"}
        ),
        "minimum_starcoder_weight": min(
            min(item.phase_0_code_weight, item.phase_1_code_weight) for item in observations
        ),
        "maximum_starcoder_weight": max(
            max(item.phase_0_code_weight, item.phase_1_code_weight) for item in observations
        ),
        "minimum_broad_weight": min(min(item.phase_0_broad_weight, item.phase_1_broad_weight) for item in observations),
        "maximum_broad_weight": max(max(item.phase_0_broad_weight, item.phase_1_broad_weight) for item in observations),
        "maximum_absolute_aggregate_error": max(abs(item.aggregate_error) for item in observations),
        "maximum_absolute_contrast_error": max(abs(item.contrast_error) for item in observations),
        "historical_paired_sd_bpb": paired_sd,
        "historical_noise_source": noise_source,
        "three_seed_mde_bpb": minimum_detectable_effect(paired_sd, len(ASYMMETRIC_SEED_VALUES)),
    }


def training_configuration(schedule: dict[str, int | float]) -> dict[str, Any]:
    optimizer = wsd80._optimizer(EXPERIMENT_BUDGET)
    return {
        "architecture": {
            "family": "Llama",
            "hidden_dim": 768,
            "intermediate_dim": 1536,
            "num_layers": 10,
            "num_heads": 8,
            "num_kv_heads": 8,
            "tied_embeddings": True,
            "approximate_trainable_parameters": "157.5M including tied 128256-token embedding; about 59M transformer",
        },
        "tokenizer": "Llama 3.1 tokenizer",
        "sequence_length": wsd80.SEQ_LEN,
        "batch_size": wsd80.BATCH_SIZE,
        "experiment_budget_tokens": EXPERIMENT_BUDGET,
        "materialized_tokens": int(schedule["materialized_tokens"]),
        "target_budget_tokens": wsd80.TARGET_BUDGET,
        "simulated_epoch_subset_seed_policy": "equals run_seed for joint data-order and subset replication",
        "starcoder_source_tokens": STARCODER_SOURCE_TOKENS,
        "training_data": "Nemotron-CC broad pool plus Dolma StarCoder rare bucket",
        "optimizer": {
            "name": "MuonH",
            "learning_rate": optimizer.learning_rate,
            "adam_learning_rate": optimizer.adam_lr,
            "momentum": optimizer.momentum,
            "beta1": optimizer.beta1,
            "beta2": optimizer.beta2,
            "epsilon": optimizer.epsilon,
            "muon_epsilon": optimizer.muon_epsilon,
            "max_gradient_norm": optimizer.max_grad_norm,
            "warmup_steps": optimizer.warmup,
            "stable_until_step": int(schedule["boundary_step"]),
            "decay_steps": optimizer.decay,
            "schedule": optimizer.lr_schedule,
            "minimum_lr_ratio": optimizer.min_lr_ratio,
        },
    }


def protocol_payload(
    *,
    anchors: tuple[Anchor, ...],
    coordinates: list[Coordinate],
    observations: list[Observation],
    eval_schedule: list[dict[str, Any]],
    checks: dict[str, Any],
    schedule: dict[str, int | float],
    manifest_hash: str,
    eval_schedule_hash: str,
) -> dict[str, Any]:
    paired_sd = float(checks["historical_paired_sd_bpb"])
    mde = minimum_detectable_effect(paired_sd, len(ASYMMETRIC_SEED_VALUES))
    sealed_eval_steps = [
        row["eval_step"]
        for row in eval_schedule
        if row["switch_step"] == SWITCH_STEPS[0] and row["endpoint_role"] == "sealed_decay_falsification"
    ]
    return {
        "candidate_id": "WSD80-SUR-076",
        "status": "design_only_not_submitted",
        "scientific_question": (
            "Does an independently identifiable temporal state explain phase-order effects after token-average "
            "aggregate, contrast, terminal mixture, repetition dose, and optimizer clock are separately controlled?"
        ),
        "parameterization": {
            "aggregate": "a=tau*p0+(1-tau)*p1",
            "contrast": "delta=p1-p0",
            "fixed_aggregate_contrast": "p0=a-(1-tau)*delta; p1=a+tau*delta",
            "fixed_phase_1": "p0=[a-(1-tau)*p1_star]/tau",
            "fixed_phase_0": "p1=[a-tau*p0_star]/(1-tau)",
            "tau": "realized phase-0 token fraction=switch_step/total_steps",
        },
        "training_configuration": training_configuration(schedule),
        "schedule": {
            **schedule,
            "final_checkpoint_step": int(schedule["total_steps"]) - 1,
            "data_switch_steps": list(SWITCH_STEPS),
            "eval_interval_steps": EVAL_INTERVAL_STEPS,
            "transition_fit_steps": f"40,80,...,{TRANSITION_MAX_STEP}",
            "sealed_decay_steps": sealed_eval_steps,
            "sealed_final_step": int(schedule["total_steps"]) - 1,
            "sealed_gap_steps": int(schedule["total_steps"]) - 1 - TRANSITION_MAX_STEP,
        },
        "anchors": [asdict(anchor) for anchor in anchors],
        "panel": {
            "unique_coordinates": len(coordinates),
            "observations": len(observations),
            "asymmetric_seed_values": list(ASYMMETRIC_SEED_VALUES),
            "spine_seed_values": list(SPINE_SEED_VALUES),
            "simulated_epoch_subset_seed_policy": "equals_run_seed",
            "manifest_sha256": manifest_hash,
            "evaluation_schedule_sha256": eval_schedule_hash,
            "starcoder_weight_floor": CODE_WEIGHT_FLOOR,
            "starcoder_weight_ceiling": CODE_WEIGHT_CEILING,
        },
        "targets": {
            "primary": PRIMARY_TARGET,
            "code_transfer": list(CODE_TRANSFER_TARGETS),
            "broad_text_negative_controls": list(NEGATIVE_CONTROL_TARGETS),
            "multiplicity_note": (
                "The three code targets are correlated and are reported separately, not as three tests."
            ),
            "selection_policy": (
                "Select mechanism, clock, and nonlinear rates only on the primary Programming-Languages target. "
                "Freeze that state before examining transfer-target predictions. Transfer targets may fit only "
                "target-specific amplitudes in the already selected response mode on pre- and post-switch rows "
                "through step "
                f"{TRANSITION_MAX_STEP}."
            ),
        },
        "candidate_family": {
            "mechanisms": mechanism_definitions(),
            "acquisition_rate_grid": list(ACQUISITION_RATE_GRID),
            "forgetting_ratio_grid": list(FORGETTING_RATIO_GRID),
            "repetition_onset_epochs": REPETITION_ONSET_EPOCHS,
            "repetition_powers": list(REPETITION_POWERS),
            "response_head": (
                "Every dynamic and static state competes with three zero-intercept heads: "
                "(1) an aggregate-potential null [A'_t(a)d,d^2] with nonnegative coefficients, "
                "(2) signed aggregate-linear [d,(a-0.35)d,d^2], and "
                "(3) signed aggregate-invariant [d,d^2]. The latter two permit phase gains inside a tied-optimal "
                "basin and leave the even response unconstrained. "
                "State-specific amplitudes are never shared across non-commensurate coordinates"
            ),
            "aggregate_spine": (
                "target- and step-specific unconstrained local quadratic fitted only to five six-seed tied aggregate "
                "levels; "
                "cross-fitted derivatives exclude the prediction row's seed. Asymmetric outcomes cannot rewrite "
                f"the spine. Sealed post-{TRANSITION_MAX_STEP} predictions reuse the frozen "
                f"step-{TRANSITION_MAX_STEP} spine derivative. Same-anchor, same-seed tied outcomes define observed "
                "deltas at both transition and sealed stages; the spine supplies only the aggregate derivative and "
                "cannot be rewritten by asymmetric outcomes"
            ),
            "counterfactual": "subtract the same state evaluated under the tied anchor at every time point",
            "state_units": (
                "token clock and cumulative LR-mass clock are dimensionless on [0,1]; code input is measured in "
                "materialized StarCoder epochs per unit clock; acquisition rate is inverse materialized epochs; "
                "forgetting ratio is dimensionless; A'_t is BPB per aggregate-code share. The dynamic state "
                "displacement d_dyn=x(two-phase)-x(tied) and static impulse proxy "
                "d_static=lr(switch)*(p1-p0) are separately normalized dimensionless coordinates with separately "
                "fitted response heads; their amplitudes are not shared or numerically comparable"
            ),
        },
        "identification": {
            "structural_selection": (
                f"pre- and post-switch smooth-target evaluations through step {TRANSITION_MAX_STEP} only"
            ),
            "primary_target_only_selection": (
                "mechanism, clock, and nonlinear rates are selected only from the primary target; transfer targets "
                "reuse the frozen state, response mode, and tied-only aggregate spine"
            ),
            "outer_cv": (
                "report both leave-one-asymmetric-switch and leave-one-anchor-out predictions. Raw asymmetric outcomes "
                "are differenced against same-anchor, same-seed tied controls; the cross-fitted tied-only spine "
                "supplies only aggregate derivatives. Pre- and post-switch limbs receive equal total weight"
            ),
            "fold_coordinate_counts": checks["coordinate_fold_counts"],
            "bootstrap_unit": (
                "anchor and run seed, preserving every asymmetric coordinate that shares one tied control; time "
                "points are not IID rows"
            ),
            "history_test": (
                "four-anchor replicated fixed-phase-1 and two-anchor fixed-phase-0 arms against the joint span of all "
                "memoryless comparators"
            ),
            "feature_identifiability": (
                "before asymmetric outcomes can license the endpoint, every eligible transition switch fold must "
                f"retain at least {MIN_STATIC_SUBSPACE_RESIDUAL:.2f} relative feature energy after projection onto the "
                "joint memoryless span. On the outcome-free sealed design, the global residual and at least "
                f"{MIN_SEALED_SEPARATED_SWITCH_FOLDS} individual folds must clear the same floor; all folds remain "
                "reported so expected convergence to a memoryless limit is visible"
            ),
            "aggregate_spine_identifiability": (
                "five aggregate levels with six independent seeds are evaluated by leave-one-level-out error before "
                "asymmetric outcomes are read; seed-bootstrap optimum stability is reported only as a diagnostic. "
                "This gate is licensing only for the aggregate-potential null response"
            ),
            "clock_test": "paired-bootstrap token clock against normalized cumulative canonical learning-rate mass",
            "noise_estimation": (
                "historical paired-difference SD is derived from the sealed optimum-fiber repeat artifact; the new "
                "panel estimates asymmetric and tied-spine uncertainty from independently trained seed blocks"
            ),
            "final_unseal": (
                f"materialize transition rows only through step {TRANSITION_MAX_STEP}; freeze selected mechanism, "
                "response mode, "
                "parameters, analysis-code "
                "hash, transition-prediction hash, and leave-switch-out target heads in data_use_ledger.csv before "
                "materializing the complete sealed cosine-decay trajectory; sealed gates use only predictions from "
                "the corresponding held-out-switch and held-out-anchor models"
            ),
            "fit_prohibitions": [
                "no final-endpoint structural or hyperparameter selection",
                "no target-specific switch indicator",
                "no unconstrained endpoint calibration or intercept",
                "no per-coordinate latent state",
                "no support-distance correction",
                "no post-outcome rate-grid extension",
            ],
        },
        "power": {
            "historical_same_configuration_paired_sd_bpb": paired_sd,
            "source": checks["historical_noise_source"],
            "paired_repeats": len(ASYMMETRIC_SEED_VALUES),
            "alpha_two_sided": ALPHA,
            "power": POWER,
            "paired_t_mde_bpb": mde,
            "equivalence_and_false_gain_bound_bpb": EQUIVALENCE_BPB,
            "synthetic_noise_sensitivity": {
                name: {
                    "run_intercept_variance_fraction": values[0],
                    "step_ar1_correlation": values[1],
                }
                for name, values in SYNTHETIC_NOISE_REGIMES.items()
            },
        },
        "gates": quantitative_gates(mde),
        "checks": checks,
        "sources": {
            "design_script_sha256": sha256_path(Path(__file__)),
            "launcher_sha256": sha256_path(LAUNCHER_PATH),
            "canonical_wsd80_launcher_sha256": sha256_path(BASE_LAUNCHER_PATH),
            "evaluator_sha256": sha256_path(EVALUATOR_PATH),
            "fiber_counterexample_report_sha256": sha256_path(FIBER_COUNTEREXAMPLE_REPORT_PATH),
        },
        "prohibitions": {
            "training_submission": "requires explicit user approval after local and independent review",
            "new_panel_outcomes_inspected": False,
            "sealed_paths": "no path containing targeted_pairwise may be accessed",
            "fiber_hypothesis_status": (
                "hard constraint prohibited by the replicated 2B counterexample; phase-weighted dose remains a null "
                "comparator only"
            ),
        },
    }


def markdown_table(rows: Iterable[Iterable[Any]]) -> str:
    materialized = [[str(value) for value in row] for row in rows]
    header = materialized[0]
    body = materialized[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(":--" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    protocol_sha256: str,
    anchors: tuple[Anchor, ...],
    observations: list[Observation],
    checks: dict[str, Any],
    schedule: dict[str, int | float],
) -> None:
    anchor_rows: list[list[Any]] = [["Anchor", "Aggregate code", "Main |delta|", "Role"]]
    for anchor in anchors:
        anchor_rows.append(
            [
                anchor.anchor_id,
                f"{anchor.aggregate_code_weight:.6f}",
                f"{anchor.main_contrast:.3f}",
                anchor.prior_role,
            ]
        )

    switch_rows: list[list[Any]] = [
        ["Switch", "Token fraction", "LR-mass departure", "Post-switch transition evals", "Coordinates"]
    ]
    for switch_step in SWITCH_STEPS:
        switch_rows.append(
            [
                switch_step,
                f"{switch_step / int(schedule['total_steps']):.6f}",
                f"{checks['clock_departures'].get(switch_step, float('nan')):+.6f}",
                checks["post_switch_intermediate_eval_counts"][switch_step],
                checks["coordinate_fold_counts"][switch_step],
            ]
        )

    report = f"""# Switch-Time Temporal-State Intervention

**Status: design only; not submitted.**

- Candidate ID: `WSD80-SUR-076`
- Protocol: `{protocol_sha256}`
- Unique coordinates: `{checks['unique_coordinates']}`
- Training observations: `{checks['observations']}`
- Antithetic pairs: `{checks['antithetic_pairs']}`
- Fixed-late / fixed-early coordinates: `{checks['fixed_late_coordinates']}` / `{checks['fixed_early_coordinates']}`
- Canonical LR decay starts at step `{schedule['boundary_step']}` of `{schedule['total_steps']}`.
- Final sealed falsification endpoint: step `{checks['final_checkpoint_step']}`.

## Identification, Not Another Endpoint Fit

For phase-0 token fraction `tau`, aggregate `a`, and contrast `delta=p1-p0`,

```text
p0 = a - (1 - tau) delta
p1 = a + tau delta
```

The main arm holds `a` and `delta` fixed. Four anchor-specific fixed-`p1` arms
break the switch-time/terminal-mixture collinearity, while two diagnostic
fixed-`p0` arms at the off-basin and lower tied-basin anchors separate early
history from late exposure. Five tied aggregate levels are trained independently
with six seeds each and define a cross-fitted aggregate spine; duplicated
pseudo-controls are not treated as independent runs. The optimizer remains the
canonical WSD80 warmup-stable-cosine schedule.

## Anchors

{markdown_table(anchor_rows)}

The first anchor uses the exact realized aggregate of `(p0,p1)=(0.1,0.5)` at
the canonical switch. Main contrasts are deliberately smaller than the observed
`delta=0.40` so every StarCoder phase weight stays in
`[{CODE_WEIGHT_FLOOR}, {CODE_WEIGHT_CEILING}]`; the broad-domain weight is its complement.

## Switch Ladder

{markdown_table(switch_rows)}

All switches align with the 2,048-example mixture block. The canonical WSD
boundary at step 6,096 lies between 40-step evaluations; its first post-switch
evaluation is step 6,120. Tied controls have no switch fold. Asymmetric fold
sizes are balanced at twelve to fourteen coordinates.

## Replication and Power

- Every fixed-late and fixed-early coordinate uses all three asymmetric seeds.
- All five tied-spine anchors use six independently trained seeds.
- Every anchor's antithetic coordinates are repeated at steps
  `{', '.join(str(step) for step in REPLICATED_MAIN_SWITCH_STEPS)}`.
- The legacy `(p0,p1)=(0.1,0.5)` off-basin coordinate is repeated at the canonical switch.
- Each replicate sets the simulated-epoch subset seed equal to its run seed; same-seed tied and asymmetric
  policies therefore share both data order and the sampled StarCoder subset.
- Historical paired SD is `{checks['historical_paired_sd_bpb']:.6f}` BPB; the three-seed
  exact paired-t MDE is `{checks['three_seed_mde_bpb']:.6f}` BPB. The positive-control
  reproduction is descriptive because this MDE exceeds the frozen 0.002-BPB point threshold.

## Frozen Comparisons

The mechanism family contains two explicit dose nulls, a terminal-level null,
phase-local repetition with a fixed one-epoch onset, a static switch-control
null, and bounded acquisition/forgetting under token and LR-mass clocks. A local
aggregate spine is fitted from tied policies only. Dynamic and static states each
compete under an aggregate-potential null, a signed aggregate-linear head, and a
signed aggregate-invariant zero-intercept head. This explicitly permits phase gains
within a tied-optimal uncertainty region, as required by the replicated 2B
counterexample. Structural selection uses pre- and post-switch rows through step
{TRANSITION_MAX_STEP}, balanced by trajectory limb. Bootstrap clusters are
anchor/seed blocks, preserving all asymmetric runs that share one observed tied control.

The sealed trajectory after step {TRANSITION_MAX_STEP} may be materialized
only after the selected mechanism, response mode, transition
parameters, analysis-code hash, and transition predictions are appended to the
data-use ledger. A pass licenses a nested 300M test; it does not promote a shared
surrogate by itself.

## Local Checks

```json
{json.dumps(checks, indent=2, sort_keys=True)}
```

## Artifacts

- `protocol.json`
- `manifest.csv`
- `coordinates.csv`
- `evaluation_schedule.csv`
- `mechanism_definitions.csv`
- `acceptance_gates.csv`
- `design_checks.json`
- `data_use_ledger.csv`
"""
    path.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LAUNCHER_PATH.exists():
        raise FileNotFoundError(f"Reviewed launcher missing: {LAUNCHER_PATH}")

    schedule = wsd80._schedule_summary(EXPERIMENT_BUDGET)
    total_steps = int(schedule["total_steps"])
    lr_boundary_step = int(schedule["boundary_step"])
    warmup_steps = int(schedule["warmup_steps"])
    materialized_tokens = int(schedule["materialized_tokens"])
    mixture_alignment = wsd80.MIXTURE_BLOCK_SIZE // math.gcd(wsd80.BATCH_SIZE, wsd80.MIXTURE_BLOCK_SIZE)

    anchors = build_anchors(total_steps, lr_boundary_step)
    coordinates = build_coordinates(anchors, total_steps, lr_boundary_step)
    observations = build_observations(
        coordinates,
        anchors=anchors,
        total_steps=total_steps,
        lr_boundary_step=lr_boundary_step,
        warmup_steps=warmup_steps,
        materialized_tokens=materialized_tokens,
        target_budget=wsd80.TARGET_BUDGET,
    )
    eval_schedule = build_eval_schedule(total_steps)
    checks = validate_design(
        anchors=anchors,
        coordinates=coordinates,
        observations=observations,
        eval_schedule=eval_schedule,
        total_steps=total_steps,
        lr_boundary_step=lr_boundary_step,
        mixture_alignment=mixture_alignment,
    )

    observation_rows = [asdict(item) for item in observations]
    coordinate_rows = [asdict(item) for item in coordinates]
    mechanism_rows = mechanism_definitions()
    gate_rows = quantitative_gates(checks["three_seed_mde_bpb"])

    write_csv(output_dir / "manifest.csv", observation_rows)
    write_csv(output_dir / "coordinates.csv", coordinate_rows)
    write_csv(output_dir / "evaluation_schedule.csv", eval_schedule)
    write_csv(output_dir / "mechanism_definitions.csv", mechanism_rows)
    write_csv(output_dir / "acceptance_gates.csv", gate_rows)
    (output_dir / "design_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")

    manifest_hash = sha256_json(csv_string_rows(observation_rows))
    eval_schedule_hash = sha256_json(csv_string_rows(eval_schedule))
    payload = protocol_payload(
        anchors=anchors,
        coordinates=coordinates,
        observations=observations,
        eval_schedule=eval_schedule,
        checks=checks,
        schedule=schedule,
        manifest_hash=manifest_hash,
        eval_schedule_hash=eval_schedule_hash,
    )
    protocol_sha256 = sha256_json(payload)
    persisted_protocol = {**payload, "protocol_sha256": protocol_sha256}
    (output_dir / "protocol.json").write_text(json.dumps(persisted_protocol, indent=2, sort_keys=True) + "\n")
    write_report(
        output_dir / "report.md",
        protocol_sha256=protocol_sha256,
        anchors=anchors,
        observations=observations,
        checks=checks,
        schedule=schedule,
    )
    write_csv(
        output_dir / "data_use_ledger.csv",
        [
            {
                "candidate_id": "WSD80-SUR-076",
                "stage": "corrected_design",
                "protocol_sha256": protocol_sha256,
                "outcomes_inspected": "existing exposed WSD80 and 300M evidence only",
                "new_panel_outcomes_inspected": False,
                "decision": "design_only_not_submitted",
                "next_action": "local dry run and independent review; explicit user approval required before launch",
            }
        ],
    )
    print(json.dumps({"output_dir": str(output_dir), "protocol_sha256": protocol_sha256, **checks}, indent=2))


if __name__ == "__main__":
    main()
