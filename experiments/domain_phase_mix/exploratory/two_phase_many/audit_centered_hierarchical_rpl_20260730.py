# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Run the frozen outcome-free gate for centered hierarchical RPL."""

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

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    centered_hierarchical_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "centered_hierarchical_rpl_physical_20260730"
RPL_ROOT = SCRIPT_DIR / "reference_outputs" / "rpl_repaired_baseline_screen_20260730"
PARAMETER_PATHS = {
    "uncheatable": RPL_ROOT / "diagnostic_300m_uncheatable" / "parameters_300m.csv",
    "table9": RPL_ROOT / "diagnostic_300m_table9" / "parameters_300m.csv",
}
WSD_PARAMETER_PATH = RPL_ROOT / "parameters_wsd80.csv"
SPAN_TOLERANCE = 1e-10
EQUALITY_TOLERANCE = 1e-10


def shape_from_row(row: pd.Series) -> rpl.Shape:
    return rpl.Shape(
        benefit_exponent=float(row["benefit_exponent"]),
        benefit_offset=float(row["benefit_offset"]),
        damage_exponent=float(row["damage_exponent"]),
        damage_threshold=0.0,
        retention=float(row["retention"]),
        late_multiplier=float(row["late_multiplier"]),
        ordering_channel=bool(row["ordering_channel"]),
    )


def relative_projection_residual(source: np.ndarray, target: np.ndarray) -> float:
    projected = source @ np.linalg.lstsq(source, target, rcond=None)[0]
    return float(np.linalg.norm(projected - target) / max(np.linalg.norm(target), 1e-15))


def augmented_diagnostics(
    design: np.ndarray,
    ridge: float,
    operator: np.ndarray,
    geometry: rpl.Geometry,
) -> dict[str, float | int]:
    scale = np.maximum(np.abs(design).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    data = np.column_stack([np.ones(len(design)), design / scale])
    normalized_operator = candidate.penalty_in_normalized_coordinates(operator, scale, geometry)
    penalty = np.zeros((operator.shape[0] + 1, operator.shape[1] + 1))
    penalty[1:, 1:] = np.sqrt(ridge) * normalized_operator
    augmented = np.vstack([data, penalty])
    singular = np.linalg.svd(augmented, compute_uv=False)
    tolerance = np.finfo(float).eps * max(augmented.shape) * singular[0]
    positive = singular[singular > tolerance]
    rank = len(positive)
    return {
        "rows": augmented.shape[0],
        "columns": augmented.shape[1],
        "rank": rank,
        "nullity": augmented.shape[1] - rank,
        "condition_positive": float(positive[0] / positive[-1]),
        "smallest_positive_singular": float(positive[-1]),
    }


def audit_300m() -> tuple[list[dict[str, float | int | str]], dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | str]] = []
    max_span_residual = 0.0
    hessian_gate = True
    direct_rank_gate = True
    for target, parameter_path in PARAMETER_PATHS.items():
        dataset = benchmark.load_300m(target)
        geometry = benchmark.geometry_300m(dataset)
        parameters = pd.read_csv(parameter_path)
        for parameter in parameters.itertuples(index=False):
            shape = shape_from_row(pd.Series(parameter._asdict()))
            ridge = float(parameter.ridge)
            benefit, damage = candidate.response_blocks(dataset.weights, geometry, shape)
            span_residuals = []
            for direct in (benefit, damage):
                hierarchical = rpl._hierarchical_block(direct, geometry)
                span_residuals.extend(
                    [
                        relative_projection_residual(hierarchical, direct),
                        relative_projection_residual(direct, hierarchical),
                    ]
                )
                rank_with_intercept = np.linalg.matrix_rank(
                    np.column_stack([np.ones(len(direct)), direct]),
                    tol=1e-10,
                )
                direct_rank_gate &= rank_with_intercept >= direct.shape[1]
            max_span_residual = max(max_span_residual, *span_residuals)

            design = candidate.design_matrix(dataset.weights, geometry, shape)
            diagnostics = augmented_diagnostics(
                design,
                ridge,
                candidate.penalty_operator(geometry, shape),
                geometry,
            )
            expected_nullity = 2 if shape.ordering_channel else 1
            hessian_gate &= diagnostics["nullity"] == expected_nullity
            rows.append(
                {
                    "target": target,
                    "fold": int(parameter.fold),
                    "ridge": ridge,
                    "ordering_channel": int(shape.ordering_channel),
                    "benefit_rank": int(np.linalg.matrix_rank(benefit, tol=1e-10)),
                    "damage_rank": int(np.linalg.matrix_rank(damage, tol=1e-10)),
                    "max_span_residual": max(span_residuals),
                    "expected_structural_nullity": expected_nullity,
                    **diagnostics,
                }
            )
    summary = {
        "max_span_residual": max_span_residual,
        "span_gate": bool(max_span_residual <= SPAN_TOLERANCE),
        "direct_rank_gate": bool(direct_rank_gate),
        "penalized_hessian_gate": bool(hessian_gate),
    }
    return rows, summary


def audit_laplacian() -> dict[str, float | int | bool]:
    dataset = benchmark.load_300m("uncheatable")
    geometry = benchmark.geometry_300m(dataset)
    operator = candidate.family_centering_operator(geometry)
    family_count = len(np.unique(geometry.families))
    rank = np.linalg.matrix_rank(operator, tol=1e-10)
    max_constant_residual = 0.0
    for family in np.unique(geometry.families):
        constant = (geometry.families == family).astype(float)
        max_constant_residual = max(max_constant_residual, float(np.max(np.abs(operator @ constant))))
    return {
        "domains": len(geometry.c0),
        "families": family_count,
        "rank": int(rank),
        "nullity": int(len(geometry.c0) - rank),
        "expected_nullity": family_count,
        "max_family_constant_residual": max_constant_residual,
        "gate": bool(len(geometry.c0) - rank == family_count and max_constant_residual <= 1e-12),
    }


def audit_wsd() -> dict[str, float | int | bool]:
    panel = wsd80.load_surface()
    geometry = rpl.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.PHASE_0_FRACTION,
    )
    max_design_delta = 0.0
    max_penalty_delta = 0.0
    for shape in rpl.shape_grid():
        design = candidate.design_matrix(panel.weights, geometry, shape)
        scale = np.maximum(np.abs(design).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
        normalized_operator = candidate.penalty_in_normalized_coordinates(
            candidate.penalty_operator(geometry, shape),
            scale,
            geometry,
        )
        max_design_delta = max(
            max_design_delta,
            float(np.max(np.abs(design - rpl.design_matrix(panel.weights, geometry, shape)))),
        )
        max_penalty_delta = max(
            max_penalty_delta,
            float(np.max(np.abs(normalized_operator - np.diag(rpl.penalty_multipliers(geometry, shape))))),
        )

    max_prediction_delta = 0.0
    for parameter in pd.read_csv(WSD_PARAMETER_PATH).itertuples(index=False):
        shape = shape_from_row(pd.Series(parameter._asdict()))
        ridge = float(parameter.ridge)
        design = rpl.design_matrix(panel.weights, geometry, shape)
        base_intercept, base_coefficients = rpl.solve_head(
            design,
            panel.y,
            ridge,
            rpl.penalty_multipliers(geometry, shape),
        )
        centered_intercept, centered_coefficients = candidate.solve_head(
            design,
            panel.y,
            ridge,
            candidate.penalty_operator(geometry, shape),
            geometry,
        )
        base_prediction = base_intercept + design @ base_coefficients
        centered_prediction = centered_intercept + design @ centered_coefficients
        max_prediction_delta = max(
            max_prediction_delta,
            float(np.max(np.abs(centered_prediction - base_prediction))),
        )
    return {
        "shape_count": len(rpl.shape_grid()),
        "selected_fit_count": len(pd.read_csv(WSD_PARAMETER_PATH)),
        "max_design_delta": max_design_delta,
        "max_penalty_delta": max_penalty_delta,
        "max_fixed_fit_prediction_delta": max_prediction_delta,
        "gate": bool(max(max_design_delta, max_penalty_delta, max_prediction_delta) <= EQUALITY_TOLERANCE),
    }


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    report = f"""# Physical-amplitude centered hierarchical RPL outcome-free audit

This rerun corrects the coordinate system used by the preregistered
within-family partial-pooling prior. No model term or hyperparameter grid
changed after the invalid normalized-coordinate outcome was observed. No 300M
target value was used to choose the correction. Previously exposed WSD80
targets are used only to verify numerical identity with RPL at the same frozen
shapes and ridges.

## Gates

| Check | Result |
| --- | ---: |
| Direct/hierarchical response-span residual | {summary["300m"]["max_span_residual"]:.3e} |
| Direct-block rank | {"PASS" if summary["300m"]["direct_rank_gate"] else "FAIL"} |
| Penalized Hessian structural nullity | {"PASS" if summary["300m"]["penalized_hessian_gate"] else "FAIL"} |
| Laplacian family null space | {"PASS" if summary["laplacian"]["gate"] else "FAIL"} |
| WSD80 numerical reduction | {"PASS" if summary["wsd80"]["gate"] else "FAIL"} |

The only remaining null directions in the 300M penalized systems are the
deliberate unpenalized signed-column aliases: concentration, plus asymmetry when
the ordering channel is active.

## Decision

**{"PASS" if summary["passed"] else "BLOCK"}** the frozen 300M outcome comparison.
"""
    (output_dir / "outcome_free_audit.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hessian_rows, summary_300m = audit_300m()
    pd.DataFrame(hessian_rows).to_csv(args.output_dir / "outcome_free_hessian_audit.csv", index=False)
    summary: dict[str, object] = {
        "300m": summary_300m,
        "laplacian": audit_laplacian(),
        "wsd80": audit_wsd(),
    }
    summary["passed"] = bool(
        summary_300m["span_gate"]
        and summary_300m["direct_rank_gate"]
        and summary_300m["penalized_hessian_gate"]
        and summary["laplacian"]["gate"]
        and summary["wsd80"]["gate"]
    )
    (args.output_dir / "outcome_free_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
