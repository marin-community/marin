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
"""How fast does a model's *proposal* converge as the fit panel grows?

The sub-280 learning curve measured the raw unconstrained optimum, which needs a
training run per point to score and which sits 10 to 45 standardized units
outside panel support at every size. This measures something observable instead.

Estimand: constrained proposal regret
-------------------------------------
Fix a deployment budget ``kappa`` on KL to the token-proportional policy. The
feasible set is the archive heldouts inside that ball. The model ranks them, and

    regret = observed(model's top-1 in F) - reference best in F

with the reference taken both as the outright minimum and as the 5th percentile,
since a minimum over roughly a thousand noisy rows is itself biased low
(run sigma is 0.00096 uncheatable, 0.0031 table9). Nothing here needs training:
every candidate in F has already been run.

The archive suits this better than the fit panel does. Its median KL to
proportional is 0.52 against the fit panel's 1.06, so it densely covers the
deployment-relevant region the 280-row panel undersamples.

Two modes, and the contrast is the point
----------------------------------------
``frozen`` selects the nonlinear shape once on the full panel and refits only the
linear head at each size. ``refit`` reselects the shape at every size, which is
what a practitioner actually does.

If ``frozen`` converges quickly and ``refit`` does not, then the slow convergence
in the sub-280 curve is shape and ridge selection, not coefficient estimation.
That is the prediction under test: the earlier curve showed the selected ridge
drifting weaker as n grew (9/10 subsamples picked l2=1.0 at n=48 against 4/10 at
n=232) while Table-9 max epochs rose from 10.3 to 19.9, so more data was making
the proposal more aggressive rather than more accurate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_swarm39_crossscale_20260725 import fit_with_fixed_shape
from dsp_exact_baseline_20260726 import (
    MODEL_NAME as DSP_NAME,
)
from dsp_exact_baseline_20260726 import (
    _as_fit,
    _fit_once,
    _packet,
    _params_from_shape,
    _weights,
    dsp_exact_model,
)
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Fit,
    Panel,
    fit_model,
    load_scale,
    provenance,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

sys.path.insert(0, str(Path(__file__).resolve().parent / "standalone_code"))

import dsp_exact as dsp

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "proposal_regret_convergence_20260726"
SCALE = "delphi_3e18"
TARGETS = (UNCHEATABLE, TABLE9)
PANEL_SIZES = (48, 80, 112, 144, 184, 232, 280)
SEEDS = (0, 1, 2, 3)
KAPPA_GRID = (0.25, 0.5, 1.0)
GRID_MODELS = ("compact_retained_state", "crs_plus")
LOW_TAIL_FRACTION = 0.15
LOW_TAIL_MIN = 5
RUN_SIGMA = {UNCHEATABLE: 0.000963, TABLE9: 0.003121}


def kl_to_proportional(aggregate: np.ndarray, prior: np.ndarray) -> np.ndarray:
    safe = np.clip(aggregate, 1e-12, None)
    return (safe * np.log(safe / prior)).sum(axis=1)


def low_tail_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    count = max(LOW_TAIL_MIN, math.ceil(LOW_TAIL_FRACTION * len(observed)))
    tail = np.argsort(predicted)[:count]
    return float(np.sqrt(np.mean((predicted[tail] - observed[tail]) ** 2)))


def fit_dsp_frozen(panel: Panel, target: str, shape: dict) -> Fit:
    """Refit only the DSP linear head, holding (rho, tau, gamma) fixed."""
    packet = _packet(panel, target)
    fitted = dsp.fit_linear_head(
        _weights(panel), panel.targets[target], packet, dsp.VARIANTS["effective_exposure"], _params_from_shape(shape)
    )
    return _as_fit(fitted, shape, panel, oof_rmse=float("nan"))


def proposal_row(
    predicted: np.ndarray,
    observed: np.ndarray,
    feasible: np.ndarray,
    kappa: float,
    target: str,
) -> dict:
    """Regret of the model's top-ranked feasible archive policy."""
    if feasible.sum() < 20:
        return {}
    p, o = predicted[feasible], observed[feasible]
    pick = int(np.argmin(p))
    best = float(np.min(o))
    reference_p5 = float(np.quantile(o, 0.05))
    return {
        "kappa": kappa,
        "n_feasible": int(feasible.sum()),
        "picked_observed_bpb": float(o[pick]),
        "feasible_best_observed_bpb": best,
        "feasible_p5_observed_bpb": reference_p5,
        "proposal_regret_vs_best": float(o[pick]) - best,
        "proposal_regret_vs_p5": float(o[pick]) - reference_p5,
        "proposal_regret_in_sigma": (float(o[pick]) - best) / RUN_SIGMA[target],
        "picked_percentile": float((o < o[pick]).mean()),
    }


def evaluate_fit(fit: Fit, model, archive: Panel, target: str, kl: np.ndarray) -> list[dict]:
    predicted = fit.predict(archive, model)
    observed = archive.targets[target]
    rows = []
    for kappa in KAPPA_GRID:
        row = proposal_row(predicted, observed, kl <= kappa, kappa, target)
        if row:
            rows.append(row)
    unconstrained = proposal_row(predicted, observed, np.ones(len(observed), dtype=bool), float("inf"), target)
    if unconstrained:
        rows.append(unconstrained)
    for row in rows:
        row["archive_spearman"] = float(pd.Series(predicted).corr(pd.Series(observed), method="spearman"))
        row["archive_low_tail_rmse"] = low_tail_rmse(observed, predicted)
        row["archive_pooled_rmse"] = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("frozen", "refit", "both"), default="both")
    parser.add_argument("--sizes", nargs="*", type=int, default=list(PANEL_SIZES))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fit_panel, heldout = load_scale(SCALE)
    catalogue = {m.name: m for m in observatory_baselines(fit_panel) + nested_candidates() + crs_plus_extensions()}
    modes = ("frozen", "refit") if args.mode == "both" else (args.mode,)

    records = []
    for target in TARGETS:
        archive = heldout.subset(np.isfinite(heldout.targets[target]))
        prior = fit_panel.proportional
        kl = kl_to_proportional(archive.aggregate, prior)
        full = fit_panel.subset(np.isfinite(fit_panel.targets[target]))

        # Full-panel reference shapes, used by frozen mode at every size.
        reference_shape = {name: fit_model(full, catalogue[name], target).shape for name in GRID_MODELS}
        _, dsp_reference_shape = _fit_once(full, target)

        for size in args.sizes:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + size)
                pick = rng.choice(len(full), size=min(size, len(full)), replace=False)
                mask = np.zeros(len(full), dtype=bool)
                mask[pick] = True
                subsample = full.subset(mask)
                for mode in modes:
                    for name in GRID_MODELS:
                        model = catalogue[name]
                        fit = (
                            fit_with_fixed_shape(subsample, model, target, reference_shape[name])
                            if mode == "frozen"
                            else fit_model(subsample, model, target)
                        )
                        for row in evaluate_fit(fit, model, archive, target, kl):
                            records.append(
                                {"model": name, "mode": mode, "n": size, "seed": seed, "target": target, **row}
                            )
                    if mode == "frozen":
                        dsp_fit = fit_dsp_frozen(subsample, target, dsp_reference_shape)
                    else:
                        fitted, shape = _fit_once(subsample, target)
                        dsp_fit = _as_fit(fitted, shape, subsample, oof_rmse=float("nan"))
                    for row in evaluate_fit(dsp_fit, dsp_exact_model(), archive, target, kl):
                        records.append(
                            {"model": DSP_NAME, "mode": mode, "n": size, "seed": seed, "target": target, **row}
                        )
                print(f"{target:18s} n={size:<4d} seed={seed} done", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(output / "proposal_regret_convergence.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "estimand": (
                    "observed BPB of the model's top-1 archive pick inside a KL ball, "
                    "minus the best (and 5th percentile) observed in that ball"
                ),
                "requires_training": False,
                "scale": SCALE,
                "archive_rows": len(heldout),
                "panel_sizes": list(args.sizes),
                "seeds": list(SEEDS),
                "kappa_grid": list(KAPPA_GRID),
                "modes": list(modes),
                "frozen_means": "nonlinear shape selected once on the full panel; only the linear head refits",
                "refit_means": "shape and ridge reselected at every panel size",
                "dsp_model": "standalone_code/dsp_exact.py effective_exposure, 158 parameters",
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    for target in TARGETS:
        block = frame[(frame["target"] == target) & (frame["kappa"] == 0.5)]
        if block.empty:
            continue
        print(f"\n=== {target}: proposal regret vs p5 at kappa=0.5, mean over seeds ===")
        print(
            block.pivot_table(index=["mode", "model"], columns="n", values="proposal_regret_vs_p5").round(5).to_string()
        )
        print(f"\n=== {target}: archive low-tail RMSE, mean over seeds ===")
        print(
            block.pivot_table(index=["mode", "model"], columns="n", values="archive_low_tail_rmse").round(5).to_string()
        )


if __name__ == "__main__":
    main()
