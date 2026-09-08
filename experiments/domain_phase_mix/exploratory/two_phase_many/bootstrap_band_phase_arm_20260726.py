# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the phase decision arm for the stacked band ensemble.

The stacked band ensemble improves all seven headline metrics on Table-9 against the argmin
incumbent, and the censored arm carries a 95 percent interval excluding zero. Phase decision
skill was reported only as a point estimate there, and it is the metric this project weights
most heavily, so it decides whether the candidate is real.

The phase arm is fitted at 60M and scored on the 238 exposure-matched 300M pairs, so the
evaluation policies are out of sample for the fit. Resampling is therefore over the 60M
fitting rows: each draw refits every band member on the resampled rows, recombines them with
the frozen stacked weights, and recomputes the two-phase-versus-tied decision value on the
same fixed 300M pairs. Draws are shared with the argmin arm so the reported statistic is the
per-draw difference.

Weights and band membership are frozen from the full 60M panel rather than refitted per draw.
That isolates head instability and makes the interval a lower bound on total uncertainty,
which is the same convention used elsewhere in this experiment.
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

from band_ensemble_20260726 import build_band  # noqa: E402
from dual_objective_harness_20260726 import build_benchmark, fit_on, select_by  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "band_ensemble_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
WEIGHTINGS = ("stacked",)
BOOTSTRAP_DRAWS = 120
BOOTSTRAP_SEED = 20260726


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_60m
    # The cross-validated ridge grid, matching how every Observatory baseline is scored. An
    # earlier version of this bootstrap used a zero-ridge grid, which tested a different
    # configuration from the candidate that beats the incumbent on the point estimates.
    model = Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True), l2_grid=(0.0, 0.01, 0.1, 1.0))
    two = benchmark.paired_300m.two_phase_panel
    tied = benchmark.paired_300m.tied_panel
    draws: list[dict[str, Any]] = []

    for target in TARGETS:
        observed = panel.targets[target]
        available = np.isfinite(observed)
        truth = benchmark.paired_300m.observed_delta[target]

        argmin_shape, argmin_l2, _ = select_by(panel, model, target, "rmse")
        bands = {name: build_band(panel, model, target, weighting=name) for name in WEIGHTINGS}
        for name, band in bands.items():
            active = sum(1 for member in band.members if member.weight > 1e-6)
            print(f"{target} {name}: {band.size} members, {active} with weight above 1e-6")

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        series = panel.series
        for draw in range(BOOTSTRAP_DRAWS):
            picked: list[int] = []
            for name in np.unique(series[available]):
                pool = np.flatnonzero(available & (series == name))
                picked.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
            resampled = np.zeros(len(series), dtype=bool)
            resampled[np.unique(np.asarray(picked, dtype=int))] = True

            record: dict[str, Any] = {"target": target, "draw": draw}
            fitted = fit_on(panel, model, target, argmin_shape, argmin_l2, rows=resampled)
            record["argmin"] = phase_decision_skill(fitted.predict(two) - fitted.predict(tied), truth)[
                "phase_skill_score"
            ]
            for name, band in bands.items():
                delta = band.predict(panel, resampled, two) - band.predict(panel, resampled, tied)
                record[f"band_{name}"] = phase_decision_skill(delta, truth)["phase_skill_score"]
            draws.append(record)
            if (draw + 1) % 40 == 0:
                print(f"  {target}: {draw + 1}/{BOOTSTRAP_DRAWS} draws")

    frame = pd.DataFrame(draws)
    frame.to_csv(OUTPUT_DIR / "phase_bootstrap_cvridge_draws.csv", index=False)

    summary = []
    for target, group in frame.groupby("target"):
        for name in WEIGHTINGS:
            candidate = group[f"band_{name}"].to_numpy()
            incumbent = group["argmin"].to_numpy()
            delta = candidate - incumbent
            summary.append(
                {
                    "target": target,
                    "arm": f"band_{name}",
                    "mean_phase_skill": float(candidate.mean()),
                    "argmin_phase_skill": float(incumbent.mean()),
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(np.quantile(delta, 0.025)),
                    "ci95_high": float(np.quantile(delta, 0.975)),
                    "fraction_better": float((candidate > incumbent).mean()),
                }
            )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "phase_bootstrap_cvridge_summary.csv", index=False)
    print("\n=== phase decision skill, paired bootstrap over 60M fitting rows (120 shared draws) ===")
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "phase_provenance_cvridge.json").write_text(
        json.dumps(
            {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "60M fitting row, stratified by design series",
                "evaluation": "238 exposure-matched 300M pairs, fixed across draws",
                "note": "band membership and stacked weights frozen from the full 60M panel",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
