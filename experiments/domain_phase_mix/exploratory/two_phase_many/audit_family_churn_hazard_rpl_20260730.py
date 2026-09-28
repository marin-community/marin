# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Outcome-free audit of the family-churn hazard RPL transition."""

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

import audit_semantic_family_mediated_retention_20260730 as prior_audit  # noqa: E402
import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    family_churn_hazard_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "family_churn_hazard_rpl_20260730"
HAZARD_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(values, levels), strict=True)
    }


def churn_diagnostics(
    weights: np.ndarray,
    geometry: rpl.Geometry,
) -> dict[str, object]:
    by_family = candidate.family_churn(weights, geometry)
    total = by_family.sum(axis=1)
    tied = replay_control.tied_rows(weights)
    asymmetric = ~tied
    rows = []
    for index, family in enumerate(np.unique(geometry.families)):
        members = geometry.families == family
        values = by_family[asymmetric, index]
        rows.append(
            {
                "family": int(family),
                "members": int(members.sum()),
                "active_fraction": float(np.mean(values > 1e-12)),
                "churn_quantiles": quantiles(values),
            }
        )
    return {
        "rows": len(weights),
        "tied_rows": int(tied.sum()),
        "asymmetric_rows": int(asymmetric.sum()),
        "active_asymmetric_fraction": float(np.mean(total[asymmetric] > 1e-12)),
        "total_churn_quantiles": quantiles(total[asymmetric]),
        "families": rows,
    }


def audit_300m() -> tuple[dict[str, object], pd.DataFrame]:
    dataset = benchmark.load_300m("uncheatable")
    geometry = benchmark.geometry_300m(dataset)
    diagnostics = churn_diagnostics(dataset.weights, geometry)
    shapes = prior_audit.selected_shapes()
    tied = replay_control.tied_rows(dataset.weights)
    contrast = dataset.weights[:, 1, :] - dataset.weights[:, 0, :]
    churn = candidate.bucket_churn(dataset.weights, geometry)
    asymmetric_cells = (~tied)[:, None] & ((np.abs(contrast) > 1e-12) | (churn > 1e-12))
    benefit_columns = len(np.unique(geometry.families)) + len(geometry.excess_domains)

    rows = []
    max_hazard_zero_error = 0.0
    max_tied_error = 0.0
    for shape_id, shape in enumerate(shapes):
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
        directional = np.abs(shape.retention * contrast[asymmetric_cells])
        directional_rms = float(np.sqrt(np.mean(directional**2)))
        for hazard in HAZARD_GRID:
            design = candidate.design_matrix(
                dataset.weights,
                geometry,
                shape,
                hazard,
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
            hazard_values = hazard * churn[asymmetric_cells]
            hazard_rms = float(np.sqrt(np.mean(hazard_values**2)))
            rank, condition = prior_audit.standardized_condition(design)
            neighboring_residuals = {
                key: prior_audit.normalized_projection_residual(
                    design[:, :benefit_columns],
                    neighbor,
                )
                for key, neighbor in neighboring_designs.items()
            }
            nearest_neighbor = min(
                neighboring_residuals,
                key=neighboring_residuals.__getitem__,
            )
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
                    "hazard_max": float(np.max(hazard_values)),
                    "benefit_projection_residual": prior_audit.normalized_projection_residual(
                        design[:, :benefit_columns],
                        baseline,
                    ),
                    "nearest_rpl_projection_residual": neighboring_residuals[nearest_neighbor],
                    "nearest_rpl_retention": nearest_neighbor[0],
                    "nearest_rpl_late_multiplier": nearest_neighbor[1],
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
    family_table = "".join(
        (
            f'| {row["family"]} | {row["members"]} | '
            f'{row["active_fraction"]:.1%} | '
            f'{row["churn_quantiles"]["q50"]:.6f} | '
            f'{row["churn_quantiles"]["q90"]:.6f} |\n'
        )
        for row in diagnostics_300m["families"]
    )
    report = f"""# Family-churn hazard RPL: outcome-free audit

## Exact invariants

- hazard=0 maximum 300M design error:
  `{diagnostics_300m["max_hazard_zero_design_error"]:.3e}`
- tied-row maximum 300M design error:
  `{diagnostics_300m["max_tied_design_error"]:.3e}`
- WSD80 maximum design error over {diagnostics_wsd80["comparisons"]} comparisons:
  `{diagnostics_wsd80["max_design_error"]:.3e}`

## 300M churn support

- asymmetric rows: {diagnostics_300m["asymmetric_rows"]}
- active asymmetric fraction:
  {diagnostics_300m["active_asymmetric_fraction"]:.1%}
- total churn median:
  {diagnostics_300m["total_churn_quantiles"]["q50"]:.6f}
- total churn 10th--90th percentile:
  {diagnostics_300m["total_churn_quantiles"]["q10"]:.6f} to
  {diagnostics_300m["total_churn_quantiles"]["q90"]:.6f}

| Family | Buckets | Active | Median churn | 90th percentile |
|---:|---:|---:|---:|---:|
{family_table}
## Design activity

- changed-benefit projection residual range, hazard > 0:
  {active["benefit_projection_residual"].min():.6f} to
  {active["benefit_projection_residual"].max():.6f}
- residual range after projection against the nearest RPL
  retention/late-multiplier design:
  {active["nearest_rpl_projection_residual"].min():.6f} to
  {active["nearest_rpl_projection_residual"].max():.6f}
- hazard/directional RMS range:
  {active["hazard_to_directional_rms"].min():.3f} to
  {active["hazard_to_directional_rms"].max():.3f}
- maximum un-clipped hazard:
  {active["hazard_max"].max():.3f} against gate clip {rpl.GATE_CLIP:.1f}
- standardized condition range:
  {active["standardized_condition"].min():.1f} to
  {active["standardized_condition"].max():.1f}

## Decision

**{"PROCEED TO A FROZEN 300M OUTCOME GATE" if passed else "BLOCK WITHOUT OUTCOME FITTING"}**.

Passing establishes only that the family-churn transition is active,
nonredundant, and numerically regular. It does not establish predictive value.
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
        and diagnostics_300m["total_churn_quantiles"]["q50"] >= 0.05
        and active["nearest_rpl_projection_residual"].max() >= 0.05
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
            "median_total_churn_at_least_0p05": bool(diagnostics_300m["total_churn_quantiles"]["q50"] >= 0.05),
            "max_projection_residual_at_least_0p05": bool(active["nearest_rpl_projection_residual"].max() >= 0.05),
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
