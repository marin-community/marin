# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the band ensemble against the argmin incumbent on both objectives.

The hypothesis is that averaging over the configurations the selection criterion cannot
separate beats picking its argmin, because inside that band the criterion's rank correlation
with censored bias collapses or inverts while bias itself still varies by 2.8 run sigma.

Three arms, all fitted on the same 280-row panel and all panel-identified:

``argmin``          the incumbent, one configuration chosen by out-of-fold RMSE.
``band_uniform``    equal weight over the band.
``band_softmin``    exponential weight over the band.

Two properties make this a fair test. The band is built from out-of-fold error on the fitting
rows only, so nothing about the censored rows or the paired policies reaches the weights; and
in the censored arm the band is rebuilt on training rows alone, so the censored rows do not
influence which members are averaged.

The paired bootstrap shares draws across arms and reports per-draw differences. Band
membership is frozen per arm before resampling, so the intervals describe head instability
under resampling rather than instability in which shapes fall inside the band, and are
therefore a lower bound on total uncertainty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from band_ensemble_20260726 import build_band, out_of_fold_ensemble  # noqa: E402
from dual_objective_harness_20260726 import (  # noqa: E402
    CENSOR_FRACTIONS,
    build_benchmark,
    fit_metrics,
    fit_on,
    out_of_fold_predictions,
    select_by,
)
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "band_ensemble_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
CENSOR_FRACTION = CENSOR_FRACTIONS[0]
L2_GRID = (0.0,)
ARMS = ("argmin", "band_uniform", "band_softmin", "band_stacked")
BOOTSTRAP_DRAWS = 120
BOOTSTRAP_SEED = 20260726


def hpr_model() -> Model:
    return Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True), l2_grid=L2_GRID)


def censored_split(observed: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    return available & ~censored, censored


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def censored_scores(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    residual = prediction - truth
    return {
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
    }


def predict_arm(arm: str, panel, model: Model, target: str, rows: np.ndarray | None, target_panel, band) -> np.ndarray:
    if arm == "argmin":
        shape, l2, _ = select_by(panel, model, target, "rmse", rows=rows)
        return fit_on(panel, model, target, shape, l2, rows=rows).predict(target_panel)
    return band.predict(panel, rows, target_panel)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    model = hpr_model()
    point_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        observed = panel.targets[target]
        train, censored = censored_split(observed, CENSOR_FRACTION)

        # Bands for the censored arm see training rows only; bands for the fit and phase
        # arms see the whole panel, matching what each arm is allowed to know.
        censored_bands = {
            arm: build_band(panel, model, target, rows=train, weighting=arm.replace("band_", ""))
            for arm in ARMS
            if arm != "argmin"
        }
        full_bands = {
            arm: build_band(panel, model, target, weighting=arm.replace("band_", "")) for arm in ARMS if arm != "argmin"
        }
        phase_bands = {
            arm: build_band(benchmark.fit_60m, model, target, weighting=arm.replace("band_", ""))
            for arm in ARMS
            if arm != "argmin"
        }
        for arm, band in full_bands.items():
            print(f"{target} {arm}: band size {band.size}, best oof {band.best_oof_rmse:.5f}")

        for arm in ARMS:
            if arm == "argmin":
                shape, l2, _ = select_by(panel, model, target, "rmse")
                oof = out_of_fold_predictions(panel, model, target, shape, l2)
            else:
                oof = out_of_fold_ensemble(panel, full_bands[arm])
            fit = fit_metrics(observed, oof)

            censored_prediction = predict_arm(arm, panel, model, target, train, panel, censored_bands.get(arm))[censored]
            censored_result = censored_scores(censored_prediction, observed[censored])

            # Phase arm: fitted at 60M so the 300M pairs are out of sample.
            two = benchmark.paired_300m.two_phase_panel
            tied = benchmark.paired_300m.tied_panel
            if arm == "argmin":
                shape_60, l2_60, _ = select_by(benchmark.fit_60m, model, target, "rmse")
                fitted = fit_on(benchmark.fit_60m, model, target, shape_60, l2_60)
                delta = fitted.predict(two) - fitted.predict(tied)
            else:
                band = phase_bands[arm]
                delta = band.predict(benchmark.fit_60m, None, two) - band.predict(benchmark.fit_60m, None, tied)
            skill = phase_decision_skill(delta, benchmark.paired_300m.observed_delta[target])

            point_rows.append(
                {
                    "target": target,
                    "arm": arm,
                    "band_size": 1 if arm == "argmin" else full_bands[arm].size,
                    "oof_rmse": fit["rmse"],
                    "oof_spearman": fit["spearman"],
                    "oof_low_tail_rmse": fit["low_tail_rmse"],
                    **censored_result,
                    "phase_skill": skill["phase_skill_score"],
                    "phase_accuracy": skill["decision_accuracy"],
                }
            )
            print(f"  {target} {arm}: done")

        # Paired bootstrap on the censored arm, sharing draws across arms.
        frozen_argmin = select_by(panel, model, target, "rmse", rows=train)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        series = panel.series
        for draw in range(BOOTSTRAP_DRAWS):
            drawn: list[int] = []
            for name in np.unique(series[train]):
                pool = np.flatnonzero(train & (series == name))
                drawn.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
            resampled = np.zeros(len(series), dtype=bool)
            resampled[np.unique(np.asarray(drawn, dtype=int))] = True

            record: dict[str, Any] = {"target": target, "draw": draw}
            for arm in ARMS:
                if arm == "argmin":
                    shape, l2, _ = frozen_argmin[0], frozen_argmin[1], None
                    prediction = fit_on(panel, model, target, shape, l2, rows=resampled).predict(panel)[censored]
                else:
                    prediction = censored_bands[arm].predict(panel, resampled, panel)[censored]
                for key, value in censored_scores(prediction, observed[censored]).items():
                    record[f"{arm}:{key}"] = value
            draw_rows.append(record)
            if (draw + 1) % 40 == 0:
                print(f"  {target}: {draw + 1}/{BOOTSTRAP_DRAWS} draws")

    point_frame = pd.DataFrame(point_rows)
    draw_frame = pd.DataFrame(draw_rows)
    point_frame.to_csv(OUTPUT_DIR / "point_estimates.csv", index=False)
    draw_frame.to_csv(OUTPUT_DIR / "bootstrap_draws.csv", index=False)

    print("\n=== point estimates ===")
    print(point_frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    summary = []
    for target, group in draw_frame.groupby("target"):
        for arm in ("band_uniform", "band_softmin", "band_stacked"):
            for metric in ("cens_rmse", "cens_bias", "cens_spearman"):
                candidate = group[f"{arm}:{metric}"].to_numpy()
                incumbent = group[f"argmin:{metric}"].to_numpy()
                delta = candidate - incumbent
                if metric == "cens_spearman":
                    better = candidate > incumbent
                elif metric == "cens_bias":
                    better = np.abs(candidate) < np.abs(incumbent)
                else:
                    better = candidate < incumbent
                summary.append(
                    {
                        "target": target,
                        "arm": arm,
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": float(np.quantile(delta, 0.025)),
                        "ci95_high": float(np.quantile(delta, 0.975)),
                        "fraction_better": float(better.mean()),
                    }
                )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)
    print("\n=== paired bootstrap: ensemble minus argmin (120 shared draws) ===")
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "band_sigma": 1.0,
                "l2_grid": list(L2_GRID),
                "censor_fraction": CENSOR_FRACTION,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "note": "band membership frozen per arm before resampling, so intervals are a lower bound",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
