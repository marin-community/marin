# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Shared bucket-value field, identified by pooling across targets and scales.

Why this axis
-------------
Fourteen earlier candidates closed four axes: how exposure enters the design, model
capacity in both directions, the response link, and the prior on the benefit field.
The recurring diagnosis was that per-bucket marginal value is real but is not
identifiable at 280 rows as 39 free coefficients. Shrinking the field toward
families (HSB) and constraining it to observables (SBF) both failed, from opposite
sides.

Neither attempt changed the amount of data used to identify it. This one does. A
bucket's data has an intrinsic usefulness that should affect both benchmarks and all
three model scales, with each panel weighting it differently. Writing

    beta_i^(panel) = scale_panel * v_i,     v >= 0

lets one shared 39-vector ``v`` be estimated from every panel at once. With six
panels of roughly 240 to 280 rows that is about 1400 observations for one vector
rather than 280, while each panel keeps its own intercept and scale.

Standardizing each panel's target before pooling makes ``scale_panel`` implicit, so
the shared field is linear in its coefficients and can be fitted by the same
nonnegative least squares used everywhere else. The assumption being tested is that
the *shape* of the bucket-value profile is common across targets and scales even
though its magnitude is not; the per-panel calibration slopes reported below are the
direct check on that.

Selection uses no heldout observation: the nonlinear shape is chosen by pooled
out-of-fold RMSE in standardized units across the same panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Fit,
    Model,
    Panel,
    evaluate,
    fit_head,
    grouped_splits,
    load_scale,
    metric_row,
    provenance,
    support_distance,
)
from swarm39_models_20260725 import bounded_saturation_shapes, build_bounded_saturation, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_shared_field_20260725"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
CV_SPLITS = 5
CV_SEED = 0
L2_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)


def usable_panels(panels: dict[str, Panel]) -> list[tuple[str, str, Panel]]:
    """Every (scale, target) panel that has finite outcomes."""
    out = []
    for scale, panel in panels.items():
        for target in TARGETS:
            if np.isfinite(panel.targets[target]).any():
                out.append((scale, target, panel.subset(np.isfinite(panel.targets[target]))))
    return out


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    centre = float(values.mean())
    spread = float(values.std(ddof=1))
    return (values - centre) / spread, centre, spread


def pooled_oof(blocks: list[tuple[str, str, Panel]], shape: dict, l2: float) -> float:
    """Out-of-fold RMSE in standardized units for a shared field at one shape.

    Folds are taken within each panel and the shared coefficients are refitted on the
    pooled training rows, so no panel ever sees its own held-out rows through the
    shared vector.
    """
    designs, responses, fold_ids = [], [], []
    for _, target, panel in blocks:
        designs.append(build_bounded_saturation(panel, shape).matrix)
        z, _, _ = standardize(panel.targets[target])
        responses.append(z)
        assignment = np.full(len(panel), -1)
        for index, (_, test) in enumerate(grouped_splits(panel, CV_SPLITS, CV_SEED)):
            assignment[test] = index
        fold_ids.append(assignment)
    design = np.vstack(designs)
    response = np.concatenate(responses)
    folds = np.concatenate(fold_ids)
    errors = []
    for index in range(CV_SPLITS):
        train = folds != index
        test = folds == index
        intercept, coefficients = fit_head(design[train], response[train], l2)
        errors.append(intercept + design[test] @ coefficients - response[test])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def fit_shared_field(blocks: list[tuple[str, str, Panel]]) -> tuple[dict, float, np.ndarray, float]:
    """Select the shape and ridge by pooled out-of-fold error, then fit the shared field."""
    best: tuple[float, dict, float] | None = None
    for shape in bounded_saturation_shapes():
        for l2 in L2_GRID:
            score = pooled_oof(blocks, shape, l2)
            if best is None or score < best[0]:
                best = (score, shape, l2)
    assert best is not None
    score, shape, l2 = best
    designs, responses = [], []
    for _, target, panel in blocks:
        designs.append(build_bounded_saturation(panel, shape).matrix)
        z, _, _ = standardize(panel.targets[target])
        responses.append(z)
    intercept, coefficients = fit_head(np.vstack(designs), np.concatenate(responses), l2)
    return shape, l2, coefficients, intercept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    loaded = {scale: load_scale(scale) for scale in SCALES}
    panels = {scale: pair[0] for scale, pair in loaded.items()}
    heldouts = {scale: pair[1] for scale, pair in loaded.items()}
    blocks = usable_panels(panels)
    print(f"pooling {len(blocks)} panels, {sum(len(b[2]) for b in blocks)} fit rows for one 39-vector")

    shape, l2, coefficients, intercept = fit_shared_field(blocks)
    print(f"selected shape {json.dumps({k: v for k, v in shape.items()}, sort_keys=True)} l2={l2}")

    # The shared field is in standardized units, so each panel is recalibrated by an
    # affine map fitted on its own fit rows only. That keeps the bucket profile shared
    # while letting magnitude differ by target and scale.
    model = Model("shared_value_field", build_bounded_saturation, bounded_saturation_shapes)
    rows, metric_frames = [], []
    for scale, target, panel in blocks:
        design = build_bounded_saturation(panel, shape).matrix
        raw = intercept + design @ coefficients
        observed = panel.targets[target]
        slope, offset = np.polyfit(raw, observed, 1)
        fit = Fit(
            model="shared_value_field",
            shape=shape,
            l2=l2,
            intercept=float(offset + slope * intercept),
            coefficients=slope * coefficients,
            names=tuple(f"shared:{b}" for b in panel.buckets),
            oof_rmse=float("nan"),
        )
        heldout = heldouts[scale]
        mask = np.isfinite(heldout.targets[target])
        distance = support_distance(panel, heldout.subset(mask))
        frame = evaluate(fit, model, panel, heldout, target, support=distance)
        metric_frames.append(frame)
        pooled = frame[frame.stratum_type == "pooled"].iloc[0]
        in_sample = metric_row(observed, offset + slope * raw)
        rows.append(
            {
                "scale": scale,
                "target": target,
                "fit_rows": len(panel),
                "recalibration_slope": float(slope),
                "in_sample_rmse": in_sample["rmse"],
                "heldout_rmse": pooled["rmse"],
                "heldout_spearman": pooled["spearman"],
                "heldout_regret_at_1": pooled["regret_at_1"],
                "heldout_optimism_over_0p05": pooled["optimism_over_0p05"],
            }
        )

    shared = pd.DataFrame(rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    shared.to_csv(output / "shared_field_metrics.csv", index=False)
    metrics.to_csv(output / "heldout_metrics.csv", index=False)
    pd.DataFrame(
        {"bucket": panels["delphi_3e18"].buckets, "shared_value": coefficients[: len(panels["delphi_3e18"].buckets)]}
    ).to_csv(output / "shared_value_vector.csv", index=False)

    baseline = observatory_baselines(panels["delphi_3e18"])
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "estimand": "one nonnegative 39-vector shared across targets and scales, panels recalibrated affinely",
                "pooled_panels": [[s, t] for s, t, _ in blocks],
                "pooled_fit_rows": sum(len(b[2]) for b in blocks),
                "shape_selection": "pooled out-of-fold RMSE in standardized units; no heldout used",
                "selected_shape": {k: v for k, v in shape.items()},
                "selected_l2": l2,
                "baseline_models_available": [m.name for m in baseline],
                "sealed_targeted_pairwise_panel_accessed": False,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    print("\n=== shared value field, per panel ===")
    print(shared.to_string(index=False))


if __name__ == "__main__":
    main()
