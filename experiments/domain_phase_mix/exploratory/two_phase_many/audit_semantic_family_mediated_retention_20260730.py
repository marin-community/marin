# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Outcome-free audit of semantic-family mediated RPL retention."""

from __future__ import annotations

import argparse
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

import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402
import benchmark_centered_hierarchical_rpl_20260730 as centered_benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    semantic_family_mediated_retention_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "semantic_family_mediated_retention_20260730"
Q_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def selected_shapes() -> tuple[rpl.Shape, ...]:
    shapes = []
    for target in ("uncheatable", "table9"):
        frame = pd.read_csv(centered_benchmark.RPL_PARAMETERS[target])
        for row in frame.loc[frame["seed"].eq(0)].to_dict("records"):
            shape = rpl.Shape(
                benefit_exponent=float(row["benefit_exponent"]),
                benefit_offset=float(row["benefit_offset"]),
                damage_exponent=float(row["damage_exponent"]),
                damage_threshold=0.0,
                retention=float(row["retention"]),
                late_multiplier=float(row["late_multiplier"]),
                ordering_channel=bool(int(float(row["ordering_channel"]))),
            )
            if shape not in shapes:
                shapes.append(shape)
    return tuple(shapes)


def normalized_projection_residual(
    candidate_design: np.ndarray,
    baseline_design: np.ndarray,
) -> float:
    baseline = np.column_stack([np.ones(len(baseline_design)), baseline_design])
    fitted = (
        baseline
        @ np.linalg.lstsq(
            baseline,
            candidate_design,
            rcond=None,
        )[0]
    )
    centered = candidate_design - candidate_design.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(centered)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(candidate_design - fitted) / denominator)


def standardized_condition(design: np.ndarray) -> tuple[int, float]:
    centered = design - design.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=0)
    active = scale > 1e-12
    standardized = centered[:, active] / scale[active]
    singular = np.linalg.svd(standardized, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    condition = float(singular[0] / singular[rank - 1])
    return rank, condition


def state_diagnostics(
    weights: np.ndarray,
    geometry: rpl.Geometry,
) -> dict[str, object]:
    aggregate, contrast = candidate.aggregate_and_contrast(weights, geometry)
    mediated = candidate.mediated_contrast(weights, geometry, 1.0)
    tied = replay_control.tied_rows(weights)
    relative = np.linalg.norm(mediated - contrast, axis=1) / np.maximum(
        np.linalg.norm(contrast, axis=1),
        1e-12,
    )
    active = relative > 1e-10
    contrast_norm = np.linalg.norm(contrast, axis=1)
    mediated_norm = np.linalg.norm(mediated, axis=1)
    norm_ratio = mediated_norm / np.maximum(contrast_norm, 1e-12)
    cosine = np.sum(mediated * contrast, axis=1) / np.maximum(
        mediated_norm * contrast_norm,
        1e-12,
    )
    projection = np.sum(mediated * contrast, axis=1) / np.maximum(
        contrast_norm**2,
        1e-12,
    )
    orthogonal_ratio = np.linalg.norm(
        mediated - projection[:, None] * contrast,
        axis=1,
    ) / np.maximum(contrast_norm, 1e-12)

    family_rows = []
    conservation_error = 0.0
    for family in np.unique(geometry.families):
        members = geometry.families == family
        family_mass = aggregate[:, members].sum(axis=1)
        shares = np.divide(
            aggregate[:, members],
            family_mass[:, None],
            out=np.zeros_like(aggregate[:, members]),
            where=family_mass[:, None] > 0.0,
        )
        uniform = np.full(members.sum(), 1.0 / members.sum())
        tv = 0.5 * np.abs(shares - uniform).sum(axis=1)
        effective = np.divide(
            1.0,
            np.sum(shares**2, axis=1),
            out=np.zeros(len(shares)),
            where=np.sum(shares**2, axis=1) > 0.0,
        )
        family_contrast_norm = np.linalg.norm(contrast[:, members], axis=1)
        family_mediated_norm = np.linalg.norm(mediated[:, members], axis=1)
        family_active = family_contrast_norm > 1e-12
        family_norm_ratio = family_mediated_norm[family_active] / family_contrast_norm[family_active]
        signed_sum_ratio = np.abs(contrast[family_active][:, members].sum(axis=1)) / family_contrast_norm[family_active]
        conservation_error = max(
            conservation_error,
            float(np.max(np.abs(mediated[:, members].sum(axis=1) - contrast[:, members].sum(axis=1)))),
        )
        family_rows.append(
            {
                "family": int(family),
                "members": int(members.sum()),
                "median_share_tv": float(np.median(tv)),
                "max_share_tv": float(np.max(tv)),
                "median_effective_members": float(np.median(effective)),
                "median_state_norm_ratio": float(np.median(family_norm_ratio)),
                "median_signed_sum_to_l2": float(np.median(signed_sum_ratio)),
            }
        )

    asymmetric = ~tied
    return {
        "rows": len(weights),
        "tied_rows": int(tied.sum()),
        "asymmetric_rows": int(asymmetric.sum()),
        "active_asymmetric_rows": int((active & asymmetric).sum()),
        "active_asymmetric_fraction": float(np.mean(active[asymmetric])),
        "median_relative_state_change": float(np.median(relative[asymmetric])),
        "max_relative_state_change": float(np.max(relative[asymmetric])),
        "median_state_cosine": float(np.median(cosine[asymmetric])),
        "median_state_norm_ratio": float(np.median(norm_ratio[asymmetric])),
        "max_state_norm_ratio": float(np.max(norm_ratio[asymmetric])),
        "median_orthogonal_state_ratio": float(np.median(orthogonal_ratio[asymmetric])),
        "family_conservation_error": conservation_error,
        "families": family_rows,
    }


def audit_300m() -> tuple[dict[str, object], pd.DataFrame]:
    dataset = benchmark.load_300m("uncheatable")
    geometry = benchmark.geometry_300m(dataset)
    shapes = selected_shapes()
    diagnostics = state_diagnostics(dataset.weights, geometry)
    rows = []
    max_q0_error = 0.0
    max_tied_error = 0.0
    tied = replay_control.tied_rows(dataset.weights)
    benefit_columns = len(np.unique(geometry.families)) + len(geometry.excess_domains)
    for shape_id, shape in enumerate(shapes):
        baseline = rpl.design_matrix(dataset.weights, geometry, shape)
        for q in Q_GRID:
            design = candidate.design_matrix(dataset.weights, geometry, shape, q)
            if q == 0.0:
                max_q0_error = max(max_q0_error, float(np.max(np.abs(design - baseline))))
            max_tied_error = max(
                max_tied_error,
                float(np.max(np.abs(design[tied] - baseline[tied]))),
            )
            rank, condition = standardized_condition(design)
            rows.append(
                {
                    "shape_id": shape_id,
                    "benefit_exponent": shape.benefit_exponent,
                    "benefit_offset": shape.benefit_offset,
                    "damage_exponent": shape.damage_exponent,
                    "retention": shape.retention,
                    "late_multiplier": shape.late_multiplier,
                    "ordering_channel": int(shape.ordering_channel),
                    "q": q,
                    "benefit_projection_residual": normalized_projection_residual(
                        design[:, :benefit_columns],
                        baseline,
                    ),
                    "whole_design_projection_residual": normalized_projection_residual(
                        design,
                        baseline,
                    ),
                    "standardized_rank": rank,
                    "standardized_condition": condition,
                }
            )
    diagnostics["max_q0_design_error"] = max_q0_error
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
        for q in Q_GRID:
            design = candidate.design_matrix(panel.weights, geometry, shape, q)
            max_error = max(max_error, float(np.max(np.abs(design - baseline))))
            comparisons += 1
    return {
        "rows": len(panel.y),
        "shapes": len(rpl.shape_grid()),
        "q_values": len(Q_GRID),
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
    q1 = design_rows.loc[design_rows["q"].eq(1.0)]
    family_rows = "".join(
        (
            f'| {row["family"]} | {row["members"]} | '
            f'{row["median_state_norm_ratio"]:.3f} | {row["median_signed_sum_to_l2"]:.3f} |\n'
        )
        for row in diagnostics_300m["families"]
    )
    report = f"""# Semantic-family mediated retention: outcome-free audit

## Exact invariants

- q=0 maximum 300M design error: `{diagnostics_300m["max_q0_design_error"]:.3e}`
- tied-row maximum 300M design error: `{diagnostics_300m["max_tied_design_error"]:.3e}`
- family-total conservation error: `{diagnostics_300m["family_conservation_error"]:.3e}`
- WSD80 maximum design error over {diagnostics_wsd80["comparisons"]} shape/q comparisons:
  `{diagnostics_wsd80["max_design_error"]:.3e}`

## 300M mechanism activity

- asymmetric rows: {diagnostics_300m["asymmetric_rows"]}
- active asymmetric rows: {diagnostics_300m["active_asymmetric_rows"]}
  ({diagnostics_300m["active_asymmetric_fraction"]:.1%})
- median relative state change at q=1:
  {diagnostics_300m["median_relative_state_change"]:.3f}
- median cosine between mediated and original contrast:
  {diagnostics_300m["median_state_cosine"]:.3f}
- median mediated/original state-norm ratio:
  {diagnostics_300m["median_state_norm_ratio"]:.3f}
- median orthogonal mediated-state magnitude relative to original contrast:
  {diagnostics_300m["median_orthogonal_state_ratio"]:.3f}
- q=1 changed-benefit projection residual range outside the full RPL design:
  {q1["benefit_projection_residual"].min():.6f} to {q1["benefit_projection_residual"].max():.6f}
- q=1 whole-design projection residual range (reported for comparison):
  {q1["whole_design_projection_residual"].min():.6f} to
  {q1["whole_design_projection_residual"].max():.6f}
- q=1 standardized condition range:
  {q1["standardized_condition"].min():.1f} to {q1["standardized_condition"].max():.1f}

## Decision

**{"PROCEED TO A FROZEN SHAPE-PINNED q LADDER" if passed else "BLOCK WITHOUT OUTCOME FITTING"}**.

This audit establishes only whether the new cross-bucket state is active and
nonredundant. It does not establish predictive value.

## Family cancellation

| Family | Buckets | Median state-norm ratio | Median signed-sum/L2 ratio |
|---:|---:|---:|---:|
{family_rows}
"""
    (output_dir / "outcome_free_audit.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_300m, design_rows = audit_300m()
    diagnostics_wsd80 = audit_wsd80()
    q1 = design_rows.loc[design_rows["q"].eq(1.0)]
    passed = bool(
        diagnostics_300m["active_asymmetric_fraction"] >= 0.25
        and diagnostics_300m["median_relative_state_change"] >= 0.10
        and diagnostics_300m["median_state_norm_ratio"] >= 0.50
        and q1["benefit_projection_residual"].max() >= 0.05
        and diagnostics_300m["max_q0_design_error"] == 0.0
        and diagnostics_300m["max_tied_design_error"] == 0.0
        and diagnostics_wsd80["max_design_error"] == 0.0
    )
    summary = {
        "300m": diagnostics_300m,
        "wsd80": diagnostics_wsd80,
        "gate": {
            "active_fraction_at_least_0p25": bool(diagnostics_300m["active_asymmetric_fraction"] >= 0.25),
            "median_relative_state_change_at_least_0p10": bool(diagnostics_300m["median_relative_state_change"] >= 0.10),
            "median_state_norm_ratio_at_least_0p50": bool(diagnostics_300m["median_state_norm_ratio"] >= 0.50),
            "max_projection_residual_at_least_0p05": bool(q1["benefit_projection_residual"].max() >= 0.05),
            "exact_invariants": bool(
                diagnostics_300m["max_q0_design_error"] == 0.0
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
