# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///
"""Run the frozen, outcome-free phase-functional independence audit."""

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
    survey_phase_functional_design_20260730 as survey,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_functional_independence_audit_20260730"
PRIMARY_FUNCTIONAL = "quadratic"
MINIMUM_SIGN_FRACTION = 0.25
MINIMUM_RESIDUAL_NORM_FRACTION = 0.2
MAXIMUM_CONDITION_NUMBER = 1e4
ABSOLUTE_ZERO_TOLERANCE = 1e-12
RELATIVE_ZERO_TOLERANCE = 1e-9
GRADIENT_NORM_FLOOR_RATIO = 1e-9


def normalized_gradient_control(
    gradient: np.ndarray,
    delta: np.ndarray,
    reference_norm: float,
) -> np.ndarray:
    """Project contrast onto the local unit tangent gradient."""

    if reference_norm <= ABSOLUTE_ZERO_TOLERANCE:
        raise ValueError("reference tangent-gradient norm must be positive")
    gradient_norm = np.linalg.norm(gradient, axis=1)
    control = np.sum(gradient * delta, axis=1)
    normalized = np.zeros(len(control))
    active = gradient_norm > GRADIENT_NORM_FLOOR_RATIO * reference_norm
    np.divide(control, gradient_norm, out=normalized, where=active)
    return normalized


def basis_diagnostics(columns: np.ndarray, names: tuple[str, ...]) -> dict[str, float | int | str]:
    """Summarize correlation and effective rank of fixed functional columns."""

    standardized = survey.standardized_columns(columns)
    if standardized.shape[1] != len(names):
        raise ValueError("all fixed functional columns must be nonconstant")
    singular = np.linalg.svd(standardized, compute_uv=False)
    energy = singular**2
    probabilities = energy / energy.sum()
    stable_rank = float(energy.sum() / energy.max())
    participation_rank = float(1.0 / np.sum(probabilities**2))
    correlation = np.corrcoef(columns, rowvar=False)
    row: dict[str, float | int | str] = {
        "basis_dimensions": len(names),
        "first_component_explained_variance": float(probabilities[0]),
        "stable_rank": stable_rank,
        "participation_rank": participation_rank,
    }
    for index, value in enumerate(singular):
        row[f"singular_value_{index + 1}"] = float(value)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            row[f"correlation_{names[left]}__{names[right]}"] = float(correlation[left, right])
    return row


def panel_audit(
    panel_name: str,
    weights: np.ndarray,
    geometry: replay_control.Geometry,
    aggregate: replay_control.AggregateFitted,
    proportional: np.ndarray,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | int | str]]:
    """Audit one panel without reading asymmetric evaluation outcomes."""

    tied = replay_control.tied_rows(weights)
    asymmetric = ~tied
    delta = weights[:, 1, :] - weights[:, 0, :]
    gradient = survey.centered_gradient(weights, aggregate)
    reference_gradient = survey.centered_gradient(proportional, aggregate)[0]
    reference_norm = float(np.linalg.norm(reference_gradient))
    normalized_control = normalized_gradient_control(gradient, delta, reference_norm)
    raw_control = np.sum(gradient * delta, axis=1)
    phase_information = replay_control.phase_information_cost(weights, geometry)
    replay_jensen = replay_control.replay_jensen_cost(weights, geometry)
    deconfounded_base = np.column_stack([normalized_control, phase_information, replay_jensen])
    deployed_base = np.column_stack([raw_control, phase_information, replay_jensen])
    gradient_norm = np.linalg.norm(gradient, axis=1)
    kappa = gradient_norm / reference_norm
    exposure_per_share = geometry.c0 + geometry.c1

    raw_functionals = np.column_stack(
        [survey.phase_functional(weights, exposure_per_share, functional) for functional in survey.FUNCTIONALS]
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for index, functional in enumerate(survey.FUNCTIONALS):
        raw = raw_functionals[:, index]
        deployed = kappa * raw
        asym_raw = raw[asymmetric]
        positive = float(np.mean(asym_raw > ABSOLUTE_ZERO_TOLERANCE))
        negative = float(np.mean(asym_raw < -ABSOLUTE_ZERO_TOLERANCE))
        scale = max(float(np.max(np.abs(asym_raw))), ABSOLUTE_ZERO_TOLERANCE)
        tied_ratio = float(np.max(np.abs(raw[tied])) / scale)
        correlation = (
            float(np.corrcoef(asym_raw, normalized_control[asymmetric])[0, 1])
            if np.std(asym_raw) > ABSOLUTE_ZERO_TOLERANCE
            and np.std(normalized_control[asymmetric]) > ABSOLUTE_ZERO_TOLERANCE
            else 0.0
        )
        residual_fraction = survey.residual_norm_fraction(
            asym_raw,
            deconfounded_base[asymmetric],
        )
        deployed_condition = survey.design_condition_number(
            np.column_stack([deployed_base[asymmetric], deployed[asymmetric]])
        )
        passes = (
            tied_ratio <= max(RELATIVE_ZERO_TOLERANCE, ABSOLUTE_ZERO_TOLERANCE / scale)
            and min(positive, negative) >= MINIMUM_SIGN_FRACTION
            and residual_fraction >= MINIMUM_RESIDUAL_NORM_FRACTION
            and deployed_condition < MAXIMUM_CONDITION_NUMBER
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
                "deconfounded_control_correlation": correlation,
                "deconfounded_residual_norm_fraction": residual_fraction,
                "deployed_condition_number": deployed_condition,
                "passes_corrected_design_gate": passes,
            }
        )

    basis = {
        "panel": panel_name,
        **basis_diagnostics(raw_functionals[asymmetric], survey.FUNCTIONALS),
    }
    return rows, basis


def render_report(design: pd.DataFrame, basis: pd.DataFrame) -> str:
    """Render a standalone report and frozen decision."""

    primary = design.loc[design["functional"].eq(PRIMARY_FUNCTIONAL)]
    primary_passed = bool(primary["passes_corrected_design_gate"].all())
    if primary_passed:
        decision = (
            "The preregistered quadratic functional clears the corrected "
            "outcome-free gate. Its outcome protocol must be frozen before any "
            "asymmetric BPB is read. Diagnostic functionals remain ineligible."
        )
    else:
        decision = (
            "The preregistered quadratic functional fails the corrected gate "
            "on at least one mandatory panel. Block the bucket-summed "
            "functional route without outcome fitting."
        )
    return f"""# Corrected phase-functional independence audit

This audit reads no asymmetric evaluation outcomes. It removes the shared
aggregate-gradient magnitude from the independence calculation and makes the
previously stated residual-norm criterion binding.

## Corrected functional support

{design.to_markdown(index=False, floatfmt=".6f")}

## Fixed-basis identifiability

{basis.to_markdown(index=False, floatfmt=".6f")}

The basis diagnostics determine whether the three fixed stories are
observationally distinguishable. They do not promote entropy or overload and
do not override the preregistered primary-functional decision.

## Decision

{decision}
"""


def main() -> None:
    """Run the frozen audit and write durable artifacts."""

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    design_rows: list[dict[str, float | int | str | bool]] = []
    basis_rows: list[dict[str, float | int | str]] = []

    panel, aggregate = benchmark.fit_full_aggregate_wsd()
    rows, basis = panel_audit(
        "wsd80",
        panel.weights,
        aggregate.geometry,
        aggregate,
        survey.proportional_weights_wsd(panel),
    )
    design_rows.extend(rows)
    basis_rows.append(basis)

    for target in benchmark.benchmark.TARGETS:
        dataset, aggregate = benchmark.fit_full_aggregate_300m(target)
        rows, basis = panel_audit(
            f"300m_{target}",
            dataset.weights,
            aggregate.geometry,
            aggregate,
            survey.proportional_weights_300m(dataset),
        )
        design_rows.extend(rows)
        basis_rows.append(basis)

    design = pd.DataFrame(design_rows)
    basis = pd.DataFrame(basis_rows)
    primary = design.loc[design["functional"].eq(PRIMARY_FUNCTIONAL)]
    summary = {
        "primary_passed": bool(primary["passes_corrected_design_gate"].all()),
        "primary_panels": primary.to_dict(orient="records"),
        "basis": basis.to_dict(orient="records"),
    }
    design.to_csv(DEFAULT_OUTPUT_DIR / "corrected_functional_support.csv", index=False)
    basis.to_csv(DEFAULT_OUTPUT_DIR / "basis_diagnostics.csv", index=False)
    (DEFAULT_OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DEFAULT_OUTPUT_DIR / "report.md").write_text(render_report(design, basis))


if __name__ == "__main__":
    main()
