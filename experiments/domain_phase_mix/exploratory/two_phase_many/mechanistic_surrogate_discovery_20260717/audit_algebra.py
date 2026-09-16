# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Audit limiting cases and invariants before statistical model screening."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.mechanistic_models import (
    ModelConfig,
    Panel,
    bounded_coverage_state,
    build_design,
    family_ces_deficit,
    foundation_gated_exposure,
    literal_replay,
    recency_exposure,
    replay_hazard_state,
    sequential_error_mass,
    simulated_epochs,
    two_level_prior_floor,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/algebraic_audit.json"


def mini_panel() -> Panel:
    domains = ("broad", "rare")
    singleton = (np.asarray([0]), np.asarray([1]))
    return Panel(
        name="algebraic_audit",
        target="synthetic",
        weights=np.zeros((0, 2, 2), dtype=float),
        observed=np.zeros(0, dtype=float),
        phase_epoch_factors=np.asarray([4.0, 1.0]),
        phase_fractions=np.asarray([0.8, 0.2]),
        domains=domains,
        proportional=np.asarray([0.75, 0.25]),
        family_names=domains,
        family_members=singleton,
        group_names=domains,
        group_members=singleton,
        group_family_indices=np.asarray([0, 1]),
    )


def analytic_constant_policy(
    panel: Panel,
    policy: np.ndarray,
    acquisition: float,
    forgetting: float,
    competition: float,
) -> np.ndarray:
    total_factor = float(panel.phase_epoch_factors.sum())
    outside = 1.0 - policy
    effective_exposure = total_factor * policy / (1.0 + competition * outside)
    hazard = forgetting * outside
    gain = np.divide(
        -np.expm1(-hazard),
        hazard,
        out=np.ones_like(hazard),
        where=hazard > 1e-12,
    )
    evidence = acquisition * effective_exposure * gain
    return np.exp(-evidence)


def main() -> None:
    panel = mini_panel()
    policy = np.asarray([0.6, 0.4])
    tied = np.asarray([[policy, policy]])
    results: dict[str, float | bool] = {}
    for competition in (0.0, 1.5):
        sequential = sequential_error_mass(
            panel,
            tied,
            acquisition=1.2,
            forgetting=0.7,
            competition=competition,
        )[0]
        analytic = analytic_constant_policy(panel, policy, 1.2, 0.7, competition)
        error = float(np.max(np.abs(sequential - analytic)))
        results[f"phase_tied_semigroup_error_competition_{competition:g}"] = error
        if error > 1e-12:
            raise AssertionError(f"Phase subdivision changed constant-policy state: {error}")

    no_forgetting = sequential_error_mass(
        panel,
        tied,
        acquisition=0.8,
        forgetting=0.0,
        competition=0.0,
    )[0]
    expected = np.exp(-0.8 * panel.phase_epoch_factors.sum() * policy)
    no_forgetting_error = float(np.max(np.abs(no_forgetting - expected)))
    results["no_forgetting_exponential_evidence_error"] = no_forgetting_error
    if no_forgetting_error > 1e-12:
        raise AssertionError("No-forgetting limit is not cumulative exponential evidence")

    coverage = bounded_coverage_state(panel, tied, forgetting=0.0)[0]
    expected_coverage = -np.expm1(-panel.phase_epoch_factors.sum() * policy)
    coverage_error = float(np.max(np.abs(coverage - expected_coverage)))
    results["bounded_coverage_no_forgetting_semigroup_error"] = coverage_error
    results["bounded_coverage_in_unit_interval"] = bool(np.all((coverage >= 0.0) & (coverage <= 1.0)))
    if coverage_error > 1e-12 or not results["bounded_coverage_in_unit_interval"]:
        raise AssertionError("Bounded coverage failed its no-forgetting limit")

    more_rare_policy = np.asarray([[[0.55, 0.45], [0.55, 0.45]]])
    more_coverage = bounded_coverage_state(panel, more_rare_policy, forgetting=0.0)[0, 1]
    results["more_exposure_increases_bounded_coverage"] = bool(more_coverage > coverage[1])
    if not results["more_exposure_increases_bounded_coverage"]:
        raise AssertionError("Bounded coverage did not increase with exposure")

    replay_no_hazard = replay_hazard_state(panel, tied, hazard_rate=0.0)[0]
    replay_no_hazard_error = float(np.max(np.abs(replay_no_hazard - expected_coverage)))
    results["replay_hazard_zero_ablation_error"] = replay_no_hazard_error
    if replay_no_hazard_error > 1e-12:
        raise AssertionError("Zero replay hazard must recover unique coverage")

    alternate_boundary = Panel(
        **{
            **panel.__dict__,
            "phase_epoch_factors": np.asarray([2.5, 2.5]),
            "phase_fractions": np.asarray([0.5, 0.5]),
        }
    )
    replay_original = replay_hazard_state(panel, tied, hazard_rate=2.0)[0]
    replay_alternate = replay_hazard_state(alternate_boundary, tied, hazard_rate=2.0)[0]
    boundary_error = float(np.max(np.abs(replay_original - replay_alternate)))
    results["replay_hazard_tied_boundary_error"] = boundary_error
    results["replay_hazard_state_nonnegative"] = bool(np.all(replay_original >= 0.0))
    if boundary_error > 2e-6 or not results["replay_hazard_state_nonnegative"]:
        raise AssertionError("Replay-hazard state depends materially on an artificial tied boundary")

    phase0, phase1, total = simulated_epochs(panel, tied)
    replay = literal_replay(total)
    results["simulated_epoch_additivity_error"] = float(np.max(np.abs(total - phase0 - phase1)))
    results["literal_replay_nonnegative"] = bool(np.all(replay >= 0.0))
    if not results["literal_replay_nonnegative"]:
        raise AssertionError("Literal replay must be nonnegative")

    more_rare = np.asarray([[[0.55, 0.45], [0.55, 0.45]]])
    base_mass = sequential_error_mass(panel, tied, 1.0, 0.0, 0.0)[0, 1]
    more_mass = sequential_error_mass(panel, more_rare, 1.0, 0.0, 0.0)[0, 1]
    results["more_exposure_reduces_unresolved_mass"] = bool(more_mass < base_mass)
    if not results["more_exposure_reduces_unresolved_mass"]:
        raise AssertionError("Unresolved error did not decrease with exposure")

    physical = total[0]
    ungated = foundation_gated_exposure(panel, tied, acquisition=2.0, boost=0.0)[0]
    ungated_error = float(np.max(np.abs(physical - ungated)))
    results["foundation_boost_zero_ablation_error"] = ungated_error
    if ungated_error > 1e-12:
        raise AssertionError("Zero foundation boost must recover physical exposure")

    acquisition = 2.0
    boost = 1.5
    broad_weight = policy[0]
    hazard = acquisition * broad_weight
    average_foundation_state = 1.0 - (-np.expm1(-hazard) / hazard)
    expected_rare = physical[1] * (1.0 + boost * average_foundation_state)
    gated = foundation_gated_exposure(panel, tied, acquisition=acquisition, boost=boost)[0]
    tied_error = float(abs(gated[1] - expected_rare))
    results["foundation_phase_tied_integral_error"] = tied_error
    if tied_error > 1e-12:
        raise AssertionError("Foundation-gated tied policy does not match the exact integral")

    late_rare = np.asarray([[[0.7, 0.3], [0.3, 0.7]]])
    tied_same_aggregate = np.asarray([[[0.62, 0.38], [0.62, 0.38]]])
    late_rare_exposure = foundation_gated_exposure(panel, late_rare, acquisition=acquisition, boost=boost)[0, 1]
    tied_rare_exposure = foundation_gated_exposure(panel, tied_same_aggregate, acquisition=acquisition, boost=boost)[
        0, 1
    ]
    results["late_specialist_gains_from_foundation"] = bool(late_rare_exposure > tied_rare_exposure)
    if not results["late_specialist_gains_from_foundation"]:
        raise AssertionError("Directional foundation transfer did not favor later specialist data")

    for recency in (0.0, 2.0, 8.0):
        recency_tied = recency_exposure(panel, tied, recency)[0]
        recency_error = float(np.max(np.abs(recency_tied - physical)))
        results[f"recency_phase_tied_exposure_error_{recency:g}"] = recency_error
        if recency_error > 1e-12:
            raise AssertionError("Normalized recency changed phase-tied exposure")

    shared_floor = 0.3
    alpha = 0.25
    physical_config = ModelConfig(
        "physical_scaling_deficit",
        (("floor", shared_floor), ("alpha", alpha)),
    )
    two_level_config = ModelConfig(
        "two_level_prior_deficit",
        (
            ("foundation_floor", shared_floor),
            ("specialist_floor", shared_floor),
            ("alpha", alpha),
        ),
    )
    physical_design = build_design(panel, tied, physical_config)
    two_level_design = build_design(panel, tied, two_level_config)
    nested_error = float(np.max(np.abs(physical_design.values - two_level_design.values)))
    results["two_level_common_prior_ablation_error"] = nested_error
    if nested_error > 1e-12:
        raise AssertionError("Equal two-level priors must recover the common-prior deficit")

    floor = two_level_prior_floor(panel, foundation_floor=1.0, specialist_floor=0.1)
    zero_deficit = np.power(floor, -alpha) - np.power(1.0 + floor, -alpha)
    results["smaller_specialist_prior_increases_zero_exposure_cost"] = bool(zero_deficit[1] > zero_deficit[0])
    if not results["smaller_specialist_prior_increases_zero_exposure_cost"]:
        raise AssertionError("Specialist prior does not steepen the starvation cost")

    exposure_ratio = np.linspace(0.0, 10.0, 1001)
    deficit_curve = np.power(exposure_ratio + 0.1, -alpha) - math.pow(1.1, -alpha)
    results["prior_deficit_monotone_decreasing"] = bool(np.all(np.diff(deficit_curve) < 0.0))
    results["prior_deficit_finite_at_zero"] = bool(np.isfinite(deficit_curve[0]))
    results["prior_deficit_bounded_below"] = bool(deficit_curve[-1] > -math.pow(1.1, -alpha))
    if not all(
        results[name]
        for name in (
            "prior_deficit_monotone_decreasing",
            "prior_deficit_finite_at_zero",
            "prior_deficit_bounded_below",
        )
    ):
        raise AssertionError("Equivalent-prior deficit failed its limiting-case audit")

    unit_ratio = np.ones((1, len(panel.group_names)), dtype=float)
    ces_reference = family_ces_deficit(
        unit_ratio,
        panel,
        substitution_order=4.0,
        floor=0.1,
        alpha=0.25,
    )
    results["ces_proportional_reference_error"] = float(np.max(np.abs(ces_reference)))
    if results["ces_proportional_reference_error"] > 1e-12:
        raise AssertionError("CES deficit must vanish at the proportional reference")

    starved_ratio = unit_ratio.copy()
    starved_ratio[0, 1] = 0.1
    ces_starved = family_ces_deficit(
        starved_ratio,
        panel,
        substitution_order=4.0,
        floor=0.1,
        alpha=0.25,
    )
    results["ces_starvation_increases_deficit"] = bool(ces_starved[0, 1] > ces_reference[0, 1])
    if not results["ces_starvation_increases_deficit"]:
        raise AssertionError("CES production does not price a starved capability")

    common_ratio = np.full_like(unit_ratio, 0.5)
    ces_low_order = family_ces_deficit(common_ratio, panel, 0.25, 0.1, 0.25)
    ces_high_order = family_ces_deficit(common_ratio, panel, 16.0, 0.1, 0.25)
    results["ces_equal_input_order_invariance_error"] = float(np.max(np.abs(ces_low_order - ces_high_order)))
    if results["ces_equal_input_order_invariance_error"] > 1e-12:
        raise AssertionError("CES order changed a family with equal inputs")

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
