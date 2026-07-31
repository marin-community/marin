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
"""Select nonlinear shapes by cross-scale agreement, then evaluate on heldouts.

Why this exists
---------------
Selecting the benefit saturation scale by out-of-fold RMSE on a single 280-row
panel is what produces fantasy optima: inside one panel, crediting repetition
improves fit, because policies that oversample a small high-value bucket really do
score well there, and nothing in the panel penalizes extrapolating that credit to
ninety epochs. Selecting it on the heldouts instead would be fitting the
evaluation set.

This module selects shapes by a third criterion that touches no heldout: a
physical parameter should be **scale invariant**. The 39-bucket design is shared
across 60M, 300M, and 3e18, so a shape is chosen to minimize the sum of
scale-normalized out-of-fold RMSE across all three fit panels at once. Ridge is
still chosen per scale, since it is a nuisance parameter rather than a physical
one.

That makes the selection criterion independent of the heldouts, and it is a
strictly harder constraint than single-panel CV: a shape that only works by
crediting repetition at one scale cannot win.
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
    from_link,
    grouped_splits,
    link_floor,
    load_scale,
    provenance,
    support_distance,
    to_link,
)
from swarm39_models_20260725 import (
    breadth_candidates,
    candidates,
    crs_plus_extensions,
    hierarchical_candidates,
    log_ratio_candidates,
    multiplicative_candidates,
    nested_candidates,
    observatory_baselines,
    shrinkage_candidates,
    structured_candidates,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_crossscale_20260725"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
CV_SPLITS = 5
CV_SEED = 0


def panel_oof(panel: Panel, model: Model, target: str, shape: dict, l2: float) -> float:
    """Grouped out-of-fold RMSE for one shape and ridge on one panel."""
    observed = panel.targets[target]
    design = model.build(panel, shape).matrix
    floor = link_floor(model, shape, observed)
    response = to_link(model, observed, floor)
    multipliers = None if model.penalty_scale is None else model.penalty_scale(panel, shape)
    errors = []
    for train, test in grouped_splits(panel, CV_SPLITS, CV_SEED):
        intercept, coefficients = fit_head(design[train], response[train], l2, multipliers)
        predicted = from_link(model, intercept + design[test] @ coefficients, floor)
        errors.append(predicted - observed[test])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def select_shape_cross_scale(
    panels: dict[str, Panel], model: Model, target: str
) -> tuple[dict, dict[str, float], pd.DataFrame]:
    """Choose one shape by summed scale-normalized out-of-fold RMSE across scales.

    Each scale's score is divided by that scale's best achievable score for this
    model, so scales with intrinsically larger BPB spread do not dominate the sum.
    """
    usable = {name: panel for name, panel in panels.items() if np.isfinite(panel.targets[target]).any()}
    shapes = list(model.shapes())
    per_scale: dict[str, list[float]] = {}
    for name, panel in usable.items():
        subset = panel.subset(np.isfinite(panel.targets[target]))
        per_scale[name] = [min(panel_oof(subset, model, target, shape, l2) for l2 in model.l2_grid) for shape in shapes]
    best_by_scale = {name: min(scores) for name, scores in per_scale.items()}
    rows = []
    totals = []
    for index, shape in enumerate(shapes):
        normalized = {name: per_scale[name][index] / best_by_scale[name] for name in usable}
        total = float(sum(normalized.values()))
        totals.append(total)
        rows.append(
            {
                "shape_index": index,
                **{k: v for k, v in shape.items() if not isinstance(v, list)},
                **{f"normalized_oof_{name}": normalized[name] for name in usable},
                "summed_normalized_oof": total,
            }
        )
    frame = pd.DataFrame(rows).sort_values("summed_normalized_oof").reset_index(drop=True)
    chosen = shapes[int(np.argmin(totals))]
    return chosen, best_by_scale, frame


def fit_with_fixed_shape(panel: Panel, model: Model, target: str, shape: dict) -> Fit:
    """Fit one panel with the shape held fixed, choosing only the ridge locally."""
    subset = panel.subset(np.isfinite(panel.targets[target]))
    observed = subset.targets[target]
    design = model.build(subset, shape)
    best = None
    for l2 in model.l2_grid:
        score = panel_oof(subset, model, target, shape, l2)
        if best is None or score < best[0]:
            best = (score, l2)
    assert best is not None
    score, l2 = best
    floor = link_floor(model, shape, observed)
    intercept, coefficients = fit_head(
        design.matrix,
        to_link(model, observed, floor),
        l2,
        None if model.penalty_scale is None else model.penalty_scale(subset, shape),
    )
    return Fit(
        floor=floor,
        model=model.name,
        shape=shape,
        l2=l2,
        intercept=intercept,
        coefficients=coefficients,
        names=design.names,
        oof_rmse=score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    loaded = {scale: load_scale(scale) for scale in SCALES}
    panels = {scale: pair[0] for scale, pair in loaded.items()}
    heldouts = {scale: pair[1] for scale, pair in loaded.items()}
    models = (
        observatory_baselines(panels["delphi_3e18"])
        + candidates()
        + hierarchical_candidates()
        + nested_candidates()
        + crs_plus_extensions()
        + structured_candidates()
        + multiplicative_candidates()
        + shrinkage_candidates()
        + log_ratio_candidates()
        + breadth_candidates()
    )
    # separate_heads carries per-bucket shape vectors calibrated to one panel, so a
    # shared cross-scale shape is not defined for it.
    models = [m for m in models if m.name != "separate_heads"]

    distance = {
        (scale, target): support_distance(
            panels[scale], heldouts[scale].subset(np.isfinite(heldouts[scale].targets[target]))
        )
        for scale in SCALES
        for target in TARGETS
        if np.isfinite(heldouts[scale].targets[target]).any()
    }

    selection_rows, metric_frames, fitted = [], [], []
    for model in models:
        for target in TARGETS:
            shape, best_by_scale, frame = select_shape_cross_scale(panels, model, target)
            frame.head(10).assign(model=model.name, target=target).to_csv(
                output / f"shape_search_{model.name}_{target}.csv", index=False
            )
            selection_rows.append(
                {
                    "model": model.name,
                    "target": target,
                    "shape": json.dumps({k: v for k, v in shape.items() if not isinstance(v, list)}, sort_keys=True),
                    **{f"single_panel_best_oof_{k}": v for k, v in best_by_scale.items()},
                }
            )
            for scale in SCALES:
                if not np.isfinite(panels[scale].targets[target]).any():
                    continue
                if not np.isfinite(heldouts[scale].targets[target]).any():
                    continue
                fit = fit_with_fixed_shape(panels[scale], model, target, shape)
                fitted.append((scale, target, fit))
                metric_frames.append(
                    evaluate(fit, model, panels[scale], heldouts[scale], target, support=distance[(scale, target)])
                )

    selection = pd.DataFrame(selection_rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    selection.to_csv(output / "cross_scale_shape_selection.csv", index=False)
    pd.DataFrame(
        [{"model": f.model, "target": t, "scale": sc, "l2": f.l2, "oof_rmse": f.oof_rmse} for sc, t, f in fitted]
    ).to_csv(output / "fitted_summary.csv", index=False)
    metrics.to_csv(output / "heldout_metrics.csv", index=False)

    protocol = {
        "shape_selection": "summed scale-normalized grouped out-of-fold RMSE over the 60M, 300M and 3e18 fit panels",
        "ridge_selection": "per scale, given the shared shape",
        "heldouts_used_for_selection": False,
        "scales": list(SCALES),
        "cv_splits": CV_SPLITS,
        "cv_seed": CV_SEED,
        "excluded_models": ["separate_heads"],
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    pd.set_option("display.width", 250)
    print("=== cross-scale selected shapes ===")
    print(selection[["model", "target", "shape"]].to_string(index=False))
    pooled = metrics[metrics["stratum_type"] == "pooled"]
    columns = ["model", "rmse", "spearman", "calibration_slope", "regret_at_1", "optimism_over_0p05"]
    for scale in SCALES:
        for target in TARGETS:
            block = pooled[(pooled["scale"] == scale) & (pooled["target"] == target)]
            if block.empty:
                continue
            print(f"\n-- {scale} / {target} (by heldout RMSE) --")
            print(block.sort_values("rmse")[columns].to_string(index=False))


if __name__ == "__main__":
    main()
