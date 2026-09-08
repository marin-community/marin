# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///
"""Survey outcome-free support for non-projective phase functionals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_saturating_phase_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    saturating_phase_control_20260730 as saturating,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_functional_design_survey_20260730"
RANDOM_DIRECTIONS_PER_ROW = 512
RANDOM_SEED = 20260730
MINIMUM_SIGN_FRACTION = 0.25
MAXIMUM_CONTROL_CORRELATION = 0.5
MINIMUM_RESIDUAL_NORM_FRACTION = 0.2
MAXIMUM_CONDITION_NUMBER = 1e4
ABSOLUTE_ZERO_TOLERANCE = 1e-12
RELATIVE_ZERO_TOLERANCE = 1e-9
PRIMARY_FUNCTIONAL = "quadratic"
FUNCTIONALS = ("quadratic", "entropy", "overload")


def centered_gradient(
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
) -> np.ndarray:
    """Return the tangent projection of the fitted aggregate gradient."""

    gradient = saturating.tangent_gradient(weights, aggregate)
    return gradient - gradient.mean(axis=1, keepdims=True)


def proportional_weights_300m(
    dataset: benchmark.benchmark.Dataset,
) -> np.ndarray:
    """Return the observed physically tied proportional policy."""

    selected = dataset.frame["run_name"].eq("baseline_proportional").to_numpy()
    tied = replay_control.tied_rows(dataset.weights)
    rows = np.flatnonzero(selected & tied)
    if len(rows) != 1:
        raise ValueError(f"expected one tied proportional row, found {len(rows)}")
    return dataset.weights[rows]


def proportional_weights_wsd(panel: benchmark.wsd80.Panel) -> np.ndarray:
    """Return the observed natural-proportional WSD80 policy."""

    reasons = panel.frame["forced_reasons"].fillna("").astype(str)
    rows = np.flatnonzero(reasons.str.contains("natural_proportional").to_numpy())
    if len(rows) != 1:
        raise ValueError(f"expected one natural-proportional row, found {len(rows)}")
    return panel.weights[rows]


def standardized_columns(columns: np.ndarray) -> np.ndarray:
    """Center and scale nonconstant columns."""

    centered = columns - columns.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=0)
    active = scale > ABSOLUTE_ZERO_TOLERANCE
    if not np.any(active):
        return np.empty((len(columns), 0))
    return centered[:, active] / scale[active]


def residual_norm_fraction(candidate: np.ndarray, base: np.ndarray) -> float:
    """Fraction of standardized candidate norm outside the base column span."""

    candidate_standardized = standardized_columns(candidate[:, None])
    if candidate_standardized.shape[1] == 0:
        return 0.0
    base_standardized = standardized_columns(base)
    if base_standardized.shape[1] == 0:
        return 1.0
    coefficients, *_ = np.linalg.lstsq(
        base_standardized,
        candidate_standardized[:, 0],
        rcond=None,
    )
    residual = candidate_standardized[:, 0] - base_standardized @ coefficients
    return float(np.linalg.norm(residual))


def design_condition_number(columns: np.ndarray) -> float:
    """Condition number after centering, scaling, and removing constants."""

    standardized = standardized_columns(columns)
    if standardized.shape[1] < 2:
        return 1.0
    singular = np.linalg.svd(standardized, compute_uv=False)
    if singular[-1] <= ABSOLUTE_ZERO_TOLERANCE:
        return float("inf")
    return float(singular[0] / singular[-1])


def phase_functional(
    weights: np.ndarray,
    exposure_per_share: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return the signed late-minus-early bucket-summed functional."""

    phase0 = weights[:, 0, :]
    phase1 = weights[:, 1, :]
    if name == "quadratic":
        value0 = np.sum(exposure_per_share * phase0**2, axis=1)
        value1 = np.sum(exposure_per_share * phase1**2, axis=1)
    elif name == "entropy":
        entropy0 = np.zeros_like(phase0)
        entropy1 = np.zeros_like(phase1)
        positive0 = phase0 > 0.0
        positive1 = phase1 > 0.0
        entropy0[positive0] = phase0[positive0] * np.log(phase0[positive0])
        entropy1[positive1] = phase1[positive1] * np.log(phase1[positive1])
        value0 = np.sum(exposure_per_share * entropy0, axis=1)
        value1 = np.sum(exposure_per_share * entropy1, axis=1)
    elif name == "overload":
        value0 = np.sum(np.maximum(exposure_per_share * phase0 - 1.0, 0.0) ** 2, axis=1)
        value1 = np.sum(np.maximum(exposure_per_share * phase1 - 1.0, 0.0) ** 2, axis=1)
    else:
        raise ValueError(f"unknown phase functional: {name}")
    return value1 - value0


def geometric_alignment_null(
    gradient: np.ndarray,
    mass: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample Gaussian tangent directions scaled to each row's transported mass."""

    rng = np.random.default_rng(seed)
    full_values = []
    trimmed_values = []
    for row_gradient, row_mass in zip(gradient, mass, strict=True):
        directions = rng.normal(size=(draws, gradient.shape[1]))
        directions -= directions.mean(axis=1, keepdims=True)
        l1 = 0.5 * np.sum(np.abs(directions), axis=1)
        directions *= row_mass / np.maximum(l1, ABSOLUTE_ZERO_TOLERANCE)[:, None]
        control = np.abs(directions @ row_gradient)
        full_authority = np.ptp(row_gradient)
        trimmed_authority = np.quantile(row_gradient, 0.95) - np.quantile(row_gradient, 0.05)
        full_values.append(control / max(row_mass * full_authority, ABSOLUTE_ZERO_TOLERANCE))
        trimmed_values.append(control / max(row_mass * trimmed_authority, ABSOLUTE_ZERO_TOLERANCE))
    return np.concatenate(full_values), np.concatenate(trimmed_values)


def panel_survey(
    panel_name: str,
    weights: np.ndarray,
    geometry: replay_control.Geometry,
    aggregate: replay_control.AggregateFitted,
    proportional: np.ndarray,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | int | str]]:
    """Survey all fixed functionals and the scalar-projection null."""

    tied = replay_control.tied_rows(weights)
    asymmetric = ~tied
    delta = weights[:, 1, :] - weights[:, 0, :]
    mass = 0.5 * np.sum(np.abs(delta), axis=1)
    gradient = centered_gradient(weights, aggregate)
    control = np.sum(gradient * delta, axis=1)
    authority = np.ptp(gradient, axis=1)
    trimmed_authority = np.quantile(gradient, 0.95, axis=1) - np.quantile(gradient, 0.05, axis=1)
    full_alignment = np.zeros(len(weights))
    trimmed_alignment = np.zeros(len(weights))
    np.divide(
        np.abs(control),
        mass * authority,
        out=full_alignment,
        where=mass * authority > ABSOLUTE_ZERO_TOLERANCE,
    )
    np.divide(
        np.abs(control),
        mass * trimmed_authority,
        out=trimmed_alignment,
        where=mass * trimmed_authority > ABSOLUTE_ZERO_TOLERANCE,
    )

    null_full, null_trimmed = geometric_alignment_null(
        gradient[asymmetric],
        mass[asymmetric],
        RANDOM_DIRECTIONS_PER_ROW,
        RANDOM_SEED,
    )
    reference_gradient = centered_gradient(proportional, aggregate)[0]
    reference_norm = float(np.linalg.norm(reference_gradient))
    if reference_norm <= ABSOLUTE_ZERO_TOLERANCE:
        raise ValueError(f"{panel_name} proportional gradient has zero tangent norm")
    kappa = np.linalg.norm(gradient, axis=1) / reference_norm
    phase_information = replay_control.phase_information_cost(weights, geometry)
    replay_jensen = replay_control.replay_jensen_cost(weights, geometry)
    base = np.column_stack([control, phase_information, replay_jensen])
    exposure_per_share = geometry.c0 + geometry.c1

    rows = []
    for functional in FUNCTIONALS:
        raw = phase_functional(weights, exposure_per_share, functional)
        candidate = kappa * raw
        asym_candidate = candidate[asymmetric]
        positive = float(np.mean(asym_candidate > ABSOLUTE_ZERO_TOLERANCE))
        negative = float(np.mean(asym_candidate < -ABSOLUTE_ZERO_TOLERANCE))
        scale = max(float(np.max(np.abs(asym_candidate))), ABSOLUTE_ZERO_TOLERANCE)
        tied_ratio = float(np.max(np.abs(candidate[tied])) / scale)
        correlation = (
            float(np.corrcoef(candidate[asymmetric], control[asymmetric])[0, 1])
            if np.std(candidate[asymmetric]) > ABSOLUTE_ZERO_TOLERANCE
            and np.std(control[asymmetric]) > ABSOLUTE_ZERO_TOLERANCE
            else 0.0
        )
        independent_fraction = residual_norm_fraction(candidate[asymmetric], base[asymmetric])
        condition = design_condition_number(np.column_stack([base[asymmetric], candidate[asymmetric]]))
        passes = (
            tied_ratio <= max(RELATIVE_ZERO_TOLERANCE, ABSOLUTE_ZERO_TOLERANCE / scale)
            and min(positive, negative) >= MINIMUM_SIGN_FRACTION
            and abs(correlation) < MAXIMUM_CONTROL_CORRELATION
            and independent_fraction >= MINIMUM_RESIDUAL_NORM_FRACTION
            and condition < MAXIMUM_CONDITION_NUMBER
        )
        rows.append(
            {
                "panel": panel_name,
                "functional": functional,
                "n_tied": int(tied.sum()),
                "n_asymmetric": int(asymmetric.sum()),
                "positive_fraction": positive,
                "negative_fraction": negative,
                "maximum_tied_relative_magnitude": tied_ratio,
                "control_correlation": correlation,
                "residual_norm_fraction": independent_fraction,
                "combined_condition_number": condition,
                "passes_design_gate": passes,
            }
        )

    alignment = {
        "panel": panel_name,
        "dimensions": weights.shape[2],
        "observed_full_alignment_p50": float(np.quantile(full_alignment[asymmetric], 0.50)),
        "observed_full_alignment_p95": float(np.quantile(full_alignment[asymmetric], 0.95)),
        "null_full_alignment_p50": float(np.quantile(null_full, 0.50)),
        "null_full_alignment_p95": float(np.quantile(null_full, 0.95)),
        "observed_trimmed_alignment_p50": float(np.quantile(trimmed_alignment[asymmetric], 0.50)),
        "observed_trimmed_alignment_p95": float(np.quantile(trimmed_alignment[asymmetric], 0.95)),
        "null_trimmed_alignment_p50": float(np.quantile(null_trimmed, 0.50)),
        "null_trimmed_alignment_p95": float(np.quantile(null_trimmed, 0.95)),
    }
    return rows, alignment


def render_report(design: pd.DataFrame, alignment: pd.DataFrame) -> str:
    """Render the outcome-free decision."""

    design_table = design.to_markdown(index=False, floatfmt=".6f")
    alignment_table = alignment.to_markdown(index=False, floatfmt=".6f")
    primary = design.loc[design["functional"].eq(PRIMARY_FUNCTIONAL)]
    passed = bool(primary["passes_design_gate"].all())
    decision = (
        "The quadratic phase-ordered replay functional clears every frozen "
        "design gate and may proceed to the separately frozen outcome audit."
        if passed
        else "The quadratic phase-ordered replay functional fails at least one "
        "frozen design gate and is blocked before outcome fitting. Diagnostic "
        "alternatives are not promoted automatically."
    )
    return f"""# Non-projective phase-functional design survey

This survey reads no asymmetric evaluation outcomes. It compares scalar
gradient alignment with a Gaussian tangent-space null and checks whether fixed,
bucket-summed phase functionals are nondegenerate and independent of existing
phase columns.

## Gradient alignment

WSD80's full-range alignment is algebraically one because its tangent space is
one-dimensional; it is not empirical evidence of unusually strong alignment.
The Gaussian null is geometric and need not correspond to feasible policies.

{alignment_table}

## Functional support

{design_table}

## Decision

{decision}
"""


def main() -> None:
    """Run the frozen survey and write durable artifacts."""

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    design_rows = []
    alignment_rows = []

    panel, aggregate = benchmark.fit_full_aggregate_wsd()
    rows, alignment = panel_survey(
        "wsd80",
        panel.weights,
        aggregate.geometry,
        aggregate,
        proportional_weights_wsd(panel),
    )
    design_rows.extend(rows)
    alignment_rows.append(alignment)

    for target in benchmark.benchmark.TARGETS:
        dataset, aggregate = benchmark.fit_full_aggregate_300m(target)
        rows, alignment = panel_survey(
            f"300m_{target}",
            dataset.weights,
            aggregate.geometry,
            aggregate,
            proportional_weights_300m(dataset),
        )
        design_rows.extend(rows)
        alignment_rows.append(alignment)

    design = pd.DataFrame(design_rows)
    alignment = pd.DataFrame(alignment_rows)
    design.to_csv(output_dir / "functional_support.csv", index=False)
    alignment.to_csv(output_dir / "alignment_null.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "primary_passed": bool(
                    design.loc[
                        design["functional"].eq(PRIMARY_FUNCTIONAL),
                        "passes_design_gate",
                    ].all()
                ),
                "functional_support": design_rows,
                "alignment_null": alignment_rows,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "report.md").write_text(render_report(design, alignment))


if __name__ == "__main__":
    main()
