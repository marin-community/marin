# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///

"""Outcome-free algebra and support audit for family-pooled RPL."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    family_pooled_retained_power_law_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "family_pooled_retained_power_law_20260730"
CONDITION_LIMIT = 1e4
INDEPENDENCE_MINIMUM = 0.20
SINGULAR_VALUE_RELATIVE_FLOOR = 1e-10


def standardized_nonzero_condition(design: np.ndarray) -> tuple[float, int]:
    """Condition number on the identified span after fitted-head scaling."""
    scale = np.maximum(np.abs(design).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    singular = np.linalg.svd(design / scale, full_matrices=False, compute_uv=False)
    nonzero = singular[singular > SINGULAR_VALUE_RELATIVE_FLOOR * singular[0]]
    return float(nonzero[0] / nonzero[-1]), len(nonzero)


def projection_residual(candidate_family: np.ndarray, base_benefit: np.ndarray) -> tuple[float, np.ndarray]:
    """Novelty of the candidate family block relative to the full base block."""
    base_scale = np.maximum(np.abs(base_benefit).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    candidate_scale = np.maximum(np.abs(candidate_family).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
    base = base_benefit / base_scale
    family = candidate_family / candidate_scale
    base -= base.mean(axis=0, keepdims=True)
    family -= family.mean(axis=0, keepdims=True)
    fitted = base @ np.linalg.lstsq(base, family, rcond=None)[0]
    residual = family - fitted
    block_ratio = float(np.linalg.norm(residual) / np.linalg.norm(family))
    family_ratio = np.linalg.norm(residual, axis=0) / np.linalg.norm(family, axis=0)
    return block_ratio, family_ratio


def corner_policies(domains: int) -> np.ndarray:
    indices = sorted({0, 1, domains // 2, domains - 2, domains - 1})
    rows = []
    for phase0 in indices:
        for phase1 in indices:
            weights = np.zeros((2, domains))
            weights[0, phase0] = 1.0
            weights[1, phase1] = 1.0
            rows.append(weights)
    return np.asarray(rows)


def shape_record(shape: rpl.Shape) -> dict[str, float | bool]:
    return asdict(shape)


def run(output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = rpl.shape_grid()

    surface = wsd80.load_surface()
    wsd_geometry = rpl.Geometry(
        c0=surface.c0,
        c1=surface.c1,
        phase_0_fraction=wsd80.PHASE_0_FRACTION,
    )
    wsd_equal = []
    for shape in shapes:
        base = rpl.design_matrix(surface.weights, wsd_geometry, shape)
        pooled = candidate.design_matrix(surface.weights, wsd_geometry, shape)
        wsd_equal.append(np.array_equal(base, pooled))

    dataset = benchmark.load_300m("uncheatable")
    geometry = rpl.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_0_fraction=float(np.median(dataset.c0 / (dataset.c0 + dataset.c1))),
        family_index=dataset.family_index,
    )
    family_count = len(np.unique(geometry.families))
    tied = np.isclose(dataset.weights[:, 0, :], dataset.weights[:, 1, :], atol=1e-12).all(axis=1)
    corners = corner_policies(dataset.weights.shape[2])

    condition_rows = []
    projection_rows = []
    seen_benefit_shapes: set[tuple[float, float, float, float]] = set()
    max_tied_ordering = 0.0
    all_columns_equal = True
    all_penalties_equal = True
    all_corners_finite = True
    all_average_equivalent = True

    for shape in shapes:
        base = rpl.design_matrix(dataset.weights, geometry, shape)
        pooled = candidate.design_matrix(dataset.weights, geometry, shape)
        all_columns_equal &= base.shape[1] == pooled.shape[1]
        all_penalties_equal &= np.array_equal(
            rpl.penalty_multipliers(geometry, shape),
            candidate.penalty_multipliers(geometry, shape),
        )
        condition, rank = standardized_nonzero_condition(pooled)
        condition_rows.append({**shape_record(shape), "condition": condition, "identified_rank": rank})
        all_corners_finite &= np.all(np.isfinite(candidate.design_matrix(corners, geometry, shape)))
        if shape.ordering_channel:
            max_tied_ordering = max(
                max_tied_ordering,
                float(np.abs(rpl.marginal_phase_block(dataset.weights[tied], geometry, shape)).max()),
            )

        benefit_key = (
            shape.benefit_exponent,
            shape.benefit_offset,
            shape.retention,
            shape.late_multiplier,
        )
        if benefit_key in seen_benefit_shapes:
            continue
        seen_benefit_shapes.add(benefit_key)

        retained_state = rpl.retained_share(
            dataset.weights,
            geometry,
            shape.retention,
            shape.late_multiplier,
        )
        base_bucket = (retained_state + shape.benefit_offset) ** (-shape.benefit_exponent)
        base_benefit = rpl._hierarchical_block(base_bucket, geometry)
        candidate_family = candidate.pooled_family_benefit(retained_state, geometry, shape)
        block_ratio, family_ratio = projection_residual(candidate_family, base_benefit)
        projection_rows.append(
            {
                "benefit_exponent": shape.benefit_exponent,
                "benefit_offset": shape.benefit_offset,
                "retention": shape.retention,
                "late_multiplier": shape.late_multiplier,
                "block_residual": block_ratio,
                **{f"family_{index}_residual": value for index, value in enumerate(family_ratio)},
            }
        )

        sum_form = candidate_family
        mean_form = candidate.mean_family_benefit(retained_state, geometry, shape)
        sum_normalized = sum_form / np.maximum(np.abs(sum_form).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
        mean_normalized = mean_form / np.maximum(np.abs(mean_form).max(axis=0), rpl.COLUMN_SCALE_FLOOR)
        all_average_equivalent &= np.allclose(sum_normalized, mean_normalized, rtol=1e-12, atol=1e-12)

    condition_frame = pd.DataFrame(condition_rows)
    projection_frame = pd.DataFrame(projection_rows)
    condition_frame.to_csv(output_dir / "condition_by_shape.csv", index=False)
    projection_frame.to_csv(output_dir / "projection_by_benefit_shape.csv", index=False)

    gates = {
        "wsd_all_shapes_exact": bool(all(wsd_equal)),
        "column_count_exact": bool(all_columns_equal),
        "penalty_multipliers_exact": bool(all_penalties_equal),
        "condition_below_limit": bool(condition_frame["condition"].max() < CONDITION_LIMIT),
        "projection_residual_above_minimum": bool(projection_frame["block_residual"].min() >= INDEPENDENCE_MINIMUM),
        "corner_values_finite": bool(all_corners_finite),
        "average_sum_equivalent": bool(all_average_equivalent),
        "tied_ordering_zero": bool(max_tied_ordering <= 1e-12),
    }
    pd.DataFrame([{"gate": name, "passed": passed} for name, passed in gates.items()]).to_csv(
        output_dir / "outcome_free_gates.csv",
        index=False,
    )

    report = [
        "# Family-pooled retained power law: outcome-free audit",
        "",
        "No BPB outcomes enter any statistic in this report.",
        "",
        "## Gate",
        "",
        "| Check | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items()],
        "",
        "## Diagnostics",
        "",
        f"- WSD80 rows: {len(surface.weights)}; exact shapes: {sum(wsd_equal)}/{len(shapes)}.",
        f"- 300M rows: {len(dataset.weights)}; tied rows: {int(tied.sum())}; families: {family_count}.",
        f"- Maximum nonzero standardized condition: {condition_frame['condition'].max():.6g}.",
        f"- Minimum block projection residual: {projection_frame['block_residual'].min():.6g}.",
        f"- Median block projection residual: {projection_frame['block_residual'].median():.6g}.",
        f"- Maximum tied ordering magnitude: {max_tied_ordering:.3e}.",
        "",
        f"Overall: {'PASS' if all(gates.values()) else 'BLOCK'}.",
    ]
    (output_dir / "outcome_free_audit.md").write_text("\n".join(report) + "\n")
    return all(gates.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not run(args.output_dir):
        raise SystemExit("family-pooled retained power law failed an outcome-free gate")


if __name__ == "__main__":
    main()
