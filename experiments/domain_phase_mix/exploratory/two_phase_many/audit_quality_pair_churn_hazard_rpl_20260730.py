# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Outcome-free audit of quality-pair churn inside RPL's retained state."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import audit_semantic_family_mediated_retention_20260730 as audit_helpers  # noqa: E402
import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    quality_pair_churn_hazard_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "quality_pair_churn_hazard_rpl_20260730"
HAZARD_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(values, levels), strict=True)
    }


def collinearity_diagnostics(
    weights: np.ndarray,
    geometry: rpl.Geometry,
    churn: np.ndarray,
    asymmetric: np.ndarray,
) -> dict[str, object]:
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    total_churn = churn.sum(axis=1)
    total_variation = 0.5 * np.abs(phase_1 - phase_0).sum(axis=1)
    global_hellinger = 1.0 - np.sqrt(phase_0 * phase_1).sum(axis=1)
    concentration = rpl.concentration_gap(weights, geometry)
    features = np.column_stack(
        [
            np.ones(asymmetric.sum()),
            concentration[asymmetric],
            total_variation[asymmetric],
            global_hellinger[asymmetric],
        ]
    )
    target = total_churn[asymmetric]
    predicted = features @ np.linalg.lstsq(features, target, rcond=None)[0]
    residual = target - predicted
    r_squared = 1.0 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)
    correlations = np.corrcoef(
        np.column_stack(
            [
                target,
                concentration[asymmetric],
                total_variation[asymmetric],
                global_hellinger[asymmetric],
            ]
        ),
        rowvar=False,
    )[0, 1:]
    return {
        "r_squared_on_global_geometry": float(r_squared),
        "correlation_concentration": float(correlations[0]),
        "correlation_total_variation": float(correlations[1]),
        "correlation_global_hellinger": float(correlations[2]),
    }


def audit_300m() -> tuple[dict[str, object], pd.DataFrame]:
    dataset = benchmark.load_300m("uncheatable")
    geometry = benchmark.geometry_300m(dataset)
    churn_families = candidate.quality_pair_families(dataset.domain_names)
    churn = candidate.conditional_family_churn(
        dataset.weights,
        churn_families,
    )
    bucket_churn = candidate.bucket_churn(dataset.weights, churn_families)
    tied = replay_control.tied_rows(dataset.weights)
    asymmetric = ~tied
    total_churn = churn.sum(axis=1)
    counts = np.unique(churn_families, return_counts=True)[1]
    pair_families = counts > 1
    active_pairs = (churn[asymmetric][:, pair_families] > 1e-12).sum(axis=1)
    diagnostics: dict[str, object] = {
        "rows": len(dataset.weights),
        "tied_rows": int(tied.sum()),
        "asymmetric_rows": int(asymmetric.sum()),
        "families": len(counts),
        "quality_pairs": int(pair_families.sum()),
        "singletons": int((counts == 1).sum()),
        "active_asymmetric_fraction": float(np.mean(total_churn[asymmetric] > 1e-12)),
        "active_pairs_per_row_quantiles": quantiles(active_pairs),
        "total_churn_quantiles": quantiles(total_churn[asymmetric]),
        "per_pair_churn_quantiles": quantiles(churn[asymmetric][:, pair_families].ravel()),
        **collinearity_diagnostics(
            dataset.weights,
            geometry,
            churn,
            asymmetric,
        ),
    }

    contrast = dataset.weights[:, 1, :] - dataset.weights[:, 0, :]
    asymmetric_cells = asymmetric[:, None] & ((np.abs(contrast) > 1e-12) | (bucket_churn > 1e-12))
    benefit_columns = len(np.unique(geometry.families)) + len(geometry.excess_domains)
    rows = []
    max_hazard_zero_error = 0.0
    max_tied_error = 0.0
    for shape_id, shape in enumerate(audit_helpers.selected_shapes()):
        baseline = rpl.design_matrix(dataset.weights, geometry, shape)
        neighboring_designs = {
            (retention, late_multiplier): rpl.design_matrix(
                dataset.weights,
                geometry,
                replace(
                    shape,
                    retention=retention,
                    late_multiplier=late_multiplier,
                ),
            )
            for retention in rpl.RETENTIONS
            for late_multiplier in rpl.LATE_MULTIPLIERS
        }
        directional_rms = float(np.sqrt(np.mean((shape.retention * contrast[asymmetric_cells]) ** 2)))
        for hazard in HAZARD_GRID:
            design = candidate.design_matrix(
                dataset.weights,
                geometry,
                shape,
                hazard,
                churn_families,
            )
            if hazard == 0.0:
                max_hazard_zero_error = max(
                    max_hazard_zero_error,
                    float(np.max(np.abs(design - baseline))),
                )
            max_tied_error = max(
                max_tied_error,
                float(np.max(np.abs(design[tied] - baseline[tied]))),
            )
            neighbor_residuals = {
                key: audit_helpers.normalized_projection_residual(
                    design[:, :benefit_columns],
                    neighbor,
                )
                for key, neighbor in neighboring_designs.items()
            }
            nearest = min(
                neighbor_residuals,
                key=neighbor_residuals.__getitem__,
            )
            hazard_values = hazard * bucket_churn[asymmetric_cells]
            hazard_rms = float(np.sqrt(np.mean(hazard_values**2)))
            rank, condition = audit_helpers.standardized_condition(design)
            rows.append(
                {
                    "shape_id": shape_id,
                    "benefit_exponent": shape.benefit_exponent,
                    "benefit_offset": shape.benefit_offset,
                    "damage_exponent": shape.damage_exponent,
                    "retention": shape.retention,
                    "late_multiplier": shape.late_multiplier,
                    "ordering_channel": int(shape.ordering_channel),
                    "churn_hazard": hazard,
                    "hazard_rms": hazard_rms,
                    "directional_rms": directional_rms,
                    "hazard_to_directional_rms": hazard_rms / directional_rms if directional_rms > 0.0 else 0.0,
                    "benefit_projection_residual": audit_helpers.normalized_projection_residual(
                        design[:, :benefit_columns],
                        baseline,
                    ),
                    "nearest_rpl_projection_residual": neighbor_residuals[nearest],
                    "nearest_rpl_retention": nearest[0],
                    "nearest_rpl_late_multiplier": nearest[1],
                    "standardized_rank": rank,
                    "standardized_condition": condition,
                }
            )
    diagnostics["max_hazard_zero_design_error"] = max_hazard_zero_error
    diagnostics["max_tied_design_error"] = max_tied_error
    return diagnostics, pd.DataFrame(rows)


def audit_wsd80() -> dict[str, object]:
    panel = wsd80.load_surface()
    geometry = rpl.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    churn_families = np.arange(panel.weights.shape[-1])
    max_error = 0.0
    comparisons = 0
    for shape in rpl.shape_grid():
        baseline = rpl.design_matrix(panel.weights, geometry, shape)
        for hazard in HAZARD_GRID:
            design = candidate.design_matrix(
                panel.weights,
                geometry,
                shape,
                hazard,
                churn_families,
            )
            max_error = max(
                max_error,
                float(np.max(np.abs(design - baseline))),
            )
            comparisons += 1
    return {
        "rows": len(panel.y),
        "shapes": len(rpl.shape_grid()),
        "hazard_values": len(HAZARD_GRID),
        "comparisons": comparisons,
        "max_design_error": max_error,
    }


def write_report(
    output_dir: Path,
    diagnostics_300m: dict[str, object],
    design_rows: pd.DataFrame,
    diagnostics_wsd80: dict[str, object],
    passed: bool,
) -> None:
    active = design_rows.loc[design_rows["churn_hazard"].gt(0.0)]
    report = f"""# Quality-pair churn hazard RPL: outcome-free audit

## Partition and support

- quality pairs: {diagnostics_300m["quality_pairs"]}
- singleton controls: {diagnostics_300m["singletons"]}
- asymmetric rows: {diagnostics_300m["asymmetric_rows"]}
- rows with nonzero quality-pair churn:
  {diagnostics_300m["active_asymmetric_fraction"]:.1%}
- active pairs per row, median:
  {diagnostics_300m["active_pairs_per_row_quantiles"]["q50"]:.1f}
- total conditional churn, median:
  {diagnostics_300m["total_churn_quantiles"]["q50"]:.6f}
- per-pair churn, median:
  {diagnostics_300m["per_pair_churn_quantiles"]["q50"]:.6f}

## Independence from global asymmetry geometry

- R-squared on concentration, TV, and global Hellinger:
  {diagnostics_300m["r_squared_on_global_geometry"]:.3f}
- correlation with concentration:
  {diagnostics_300m["correlation_concentration"]:.3f}
- correlation with TV:
  {diagnostics_300m["correlation_total_variation"]:.3f}
- correlation with global Hellinger:
  {diagnostics_300m["correlation_global_hellinger"]:.3f}

## Exact invariants

- hazard=0 maximum 300M design error:
  `{diagnostics_300m["max_hazard_zero_design_error"]:.3e}`
- tied-row maximum 300M design error:
  `{diagnostics_300m["max_tied_design_error"]:.3e}`
- WSD80 maximum design error over {diagnostics_wsd80["comparisons"]} comparisons:
  `{diagnostics_wsd80["max_design_error"]:.3e}`

## Design activity

- nearest-RPL projection residual range, hazard > 0:
  {active["nearest_rpl_projection_residual"].min():.6f} to
  {active["nearest_rpl_projection_residual"].max():.6f}
- hazard/directional RMS range:
  {active["hazard_to_directional_rms"].min():.3f} to
  {active["hazard_to_directional_rms"].max():.3f}
- standardized condition range:
  {active["standardized_condition"].min():.1f} to
  {active["standardized_condition"].max():.1f}

## Decision

**{"PROCEED TO A FROZEN 300M OUTCOME GATE" if passed else "BLOCK WITHOUT OUTCOME FITTING"}**.

Passing establishes only that the quality-pair transition is active,
distinguishable from global phase geometry, and numerically regular.
"""
    (output_dir / "outcome_free_audit.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_300m, design_rows = audit_300m()
    diagnostics_wsd80 = audit_wsd80()
    active = design_rows.loc[design_rows["churn_hazard"].gt(0.0)]
    passed = bool(
        diagnostics_300m["active_asymmetric_fraction"] >= 0.25
        and diagnostics_300m["total_churn_quantiles"]["q50"] >= 0.10
        and active["nearest_rpl_projection_residual"].max() >= 0.05
        and diagnostics_300m["r_squared_on_global_geometry"] <= 0.95
        and active["standardized_condition"].max() <= 1e4
        and diagnostics_300m["max_hazard_zero_design_error"] == 0.0
        and diagnostics_300m["max_tied_design_error"] == 0.0
        and diagnostics_wsd80["max_design_error"] == 0.0
    )
    summary = {
        "300m": diagnostics_300m,
        "wsd80": diagnostics_wsd80,
        "gate": {
            "active_fraction_at_least_0p25": bool(diagnostics_300m["active_asymmetric_fraction"] >= 0.25),
            "median_total_churn_at_least_0p10": bool(diagnostics_300m["total_churn_quantiles"]["q50"] >= 0.10),
            "max_projection_residual_at_least_0p05": bool(active["nearest_rpl_projection_residual"].max() >= 0.05),
            "global_geometry_r_squared_at_most_0p95": bool(diagnostics_300m["r_squared_on_global_geometry"] <= 0.95),
            "condition_at_most_1e4": bool(active["standardized_condition"].max() <= 1e4),
            "exact_invariants": bool(
                diagnostics_300m["max_hazard_zero_design_error"] == 0.0
                and diagnostics_300m["max_tied_design_error"] == 0.0
                and diagnostics_wsd80["max_design_error"] == 0.0
            ),
            "passed": passed,
        },
    }
    design_rows.to_csv(args.output_dir / "design_audit.csv", index=False)
    (args.output_dir / "outcome_free_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        args.output_dir,
        diagnostics_300m,
        design_rows,
        diagnostics_wsd80,
        passed,
    )
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
