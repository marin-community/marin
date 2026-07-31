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
"""Is the crs_plus extension result signal or noise?

The extension sweep gave family-pooled phase heads a low-predicted-tail win in 2 of
6 scale-by-target cells, and gave the geometry block a pooled-RMSE win in 6 of 6
while losing low-tail in 4 of 6. Both counts are small enough that they could be
sampling noise, so neither should be acted on without an interval.

Method
------
Models are fitted once per cell with shapes chosen by the heldout-free cross-scale
criterion, then the *evaluation set* is resampled. Because both models see exactly
the same resampled rows, the comparison is paired and fit noise cancels.

Two resampling units are reported:

* row bootstrap, which treats heldout observations as exchangeable;
* series cluster bootstrap, which resamples whole proposal series.

The archive is intervention-designed rather than IID, so rows inside a series are
correlated and the cluster interval is the honest one. The row interval is kept
only to show how much it understates the uncertainty.

Low-tail RMSE is recomputed inside every draw rather than carried over, since the
15 percent of rows a model ranks best changes as the sample changes. That is the
point of the metric: it measures accuracy over whatever the model would propose.

Scope
-----
This quantifies evaluation-set uncertainty. It does not cover shape-selection or
panel-resampling uncertainty, so it is a lower bound on total uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_swarm39_crossscale_20260725 import fit_with_fixed_shape, select_shape_cross_scale
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Fit,
    Model,
    Panel,
    load_scale,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_extension_bootstrap_20260725"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
DRAWS = 4000
SEED = 20260725
LOW_TAIL_FRACTION = 0.15
LOW_TAIL_MIN = 5


def low_tail_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE over the fraction of rows the model ranks best."""
    count = max(LOW_TAIL_MIN, math.ceil(LOW_TAIL_FRACTION * len(observed)))
    tail = np.argsort(predicted)[:count]
    return float(np.sqrt(np.mean((predicted[tail] - observed[tail]) ** 2)))


def pooled_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def bootstrap_indices(rng: np.random.Generator, series: np.ndarray, cluster: bool) -> np.ndarray:
    if not cluster:
        return rng.integers(0, len(series), len(series))
    groups = np.unique(series)
    picked = rng.integers(0, len(groups), len(groups))
    return np.concatenate([np.flatnonzero(series == groups[g]) for g in picked])


def compare(
    observed: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    series: np.ndarray,
    rng: np.random.Generator,
    cluster: bool,
) -> dict[str, float]:
    """Paired bootstrap of the candidate-minus-baseline metric difference."""
    low_tail, pooled = [], []
    for _ in range(DRAWS):
        pick = bootstrap_indices(rng, series, cluster)
        if len(np.unique(pick)) < LOW_TAIL_MIN * 2:
            continue
        y = observed[pick]
        low_tail.append(low_tail_rmse(y, candidate_prediction[pick]) - low_tail_rmse(y, baseline_prediction[pick]))
        pooled.append(pooled_rmse(y, candidate_prediction[pick]) - pooled_rmse(y, baseline_prediction[pick]))
    low_tail_array = np.asarray(low_tail)
    pooled_array = np.asarray(pooled)
    return {
        "point_low_tail_delta": (
            low_tail_rmse(observed, candidate_prediction) - low_tail_rmse(observed, baseline_prediction)
        ),
        "low_tail_ci_low": float(np.quantile(low_tail_array, 0.025)),
        "low_tail_ci_high": float(np.quantile(low_tail_array, 0.975)),
        "probability_low_tail_better": float(np.mean(low_tail_array < 0)),
        "point_pooled_delta": pooled_rmse(observed, candidate_prediction) - pooled_rmse(observed, baseline_prediction),
        "pooled_ci_low": float(np.quantile(pooled_array, 0.025)),
        "pooled_ci_high": float(np.quantile(pooled_array, 0.975)),
        "probability_pooled_better": float(np.mean(pooled_array < 0)),
        "draws": len(low_tail_array),
    }


def fitted_prediction(fit: Fit, model: Model, heldout: Panel) -> np.ndarray:
    return fit.predict(heldout, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    models = {m.name: m for m in nested_candidates() + crs_plus_extensions()}
    baseline = models["crs_plus"]
    candidates = ["crs_plus_heads", "crs_plus_geometry", "crs_plus_bucket_heads"]

    panels = {scale: load_scale(scale) for scale in SCALES}
    rows = []
    for target in TARGETS:
        usable_scales = [s for s in SCALES if np.isfinite(panels[s][0].targets[target]).any()]
        fit_panels = {s: panels[s][0] for s in usable_scales}
        shapes = {baseline.name: select_shape_cross_scale(fit_panels, baseline, target)[0]}
        for name in candidates:
            shapes[name] = select_shape_cross_scale(fit_panels, models[name], target)[0]
        for scale in usable_scales:
            fit_panel, heldout = panels[scale]
            mask = np.isfinite(heldout.targets[target])
            evaluation = heldout.subset(mask)
            observed = evaluation.targets[target]
            base_fit = fit_with_fixed_shape(fit_panel, baseline, target, shapes[baseline.name])
            base_prediction = fitted_prediction(base_fit, baseline, evaluation)
            for name in candidates:
                model = models[name]
                fit = fit_with_fixed_shape(fit_panel, model, target, shapes[name])
                prediction = fitted_prediction(fit, model, evaluation)
                for cluster in (False, True):
                    result = compare(observed, base_prediction, prediction, evaluation.series, rng, cluster)
                    rows.append(
                        {
                            "scale": scale,
                            "target": "uncheatable" if target == UNCHEATABLE else "table9",
                            "candidate": name,
                            "resampling": "series_cluster" if cluster else "row",
                            "n_heldout": len(observed),
                            "n_series": len(np.unique(evaluation.series)),
                            **result,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "extension_bootstrap.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "draws": DRAWS,
                "seed": SEED,
                "low_tail_fraction": LOW_TAIL_FRACTION,
                "paired": "both models scored on identical resampled rows; fit noise cancels",
                "resampling_units": ["row", "series_cluster"],
                "honest_interval": "series_cluster, because the archive is intervention-designed",
                "covers": "evaluation-set uncertainty only; not shape selection or fit-panel resampling",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 250)
    for resampling in ("row", "series_cluster"):
        block = frame[frame["resampling"] == resampling]
        print(f"\n=== {resampling} bootstrap: low-tail RMSE delta versus crs_plus (negative favours the candidate) ===")
        print(
            block[
                [
                    "scale",
                    "target",
                    "candidate",
                    "point_low_tail_delta",
                    "low_tail_ci_low",
                    "low_tail_ci_high",
                    "probability_low_tail_better",
                ]
            ].to_string(index=False)
        )
    print("\n=== geometry: pooled versus low-tail, series-cluster ===")
    geometry = frame[(frame["candidate"] == "crs_plus_geometry") & (frame["resampling"] == "series_cluster")]
    print(
        geometry[
            [
                "scale",
                "target",
                "point_pooled_delta",
                "probability_pooled_better",
                "point_low_tail_delta",
                "probability_low_tail_better",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
