# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compute-equivalent gain of MARINER's fixed mixtures over each baseline on the full-corpus scaling ladder.

Each baseline's measured losses (the plotted points of Figure 6 / Table 2) are fitted against compute: a power law
L = E + A (C / 1e18)^-alpha through four or more rungs, a line in log10 C otherwise. At each MARINER rung, the
baseline-equivalent compute is where the fitted curve reaches MARINER's measured loss; the multiplier is its ratio to
MARINER's compute and the saving is one minus its inverse. Multipliers beyond a baseline's measured rungs are flagged
as extrapolated and left out of the paper's table. Runs above 3e18 FLOPs are single, so no uncertainty is attached.

usage: uv run --offline --no-sync python compute_equivalence_20260915.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
PLOTTED_POINTS = REFERENCE / "frozen_scaling_update_20260913" / "figure" / "plotted_points.csv"
OUTPUT_DIR = REFERENCE / "compute_equivalence_20260915"
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))
BASELINES = ("Proportional", "UniMax-8", "Olmix")
MARINER = "MARINER"
LADDER_RUNGS = (3e18, 2e19, 3e20, 1e21)
POWER_LAW_MIN_RUNGS = 4
COMPUTE_UNIT = 1e18
# Search one decade beyond the measured rungs so an extrapolated crossing is still located and flagged.
BRACKET_DECADES = 1.0
PLACEHOLDER = "\\placeholder{TODO}"


@dataclass(frozen=True)
class BaselineFit:
    target: str
    baseline: str
    form: str
    parameters: dict[str, float]
    rmse: float
    log_compute_min: float
    log_compute_max: float

    def loss(self, log_compute: float) -> float:
        if self.form == "power":
            return power_law(log_compute, self.parameters["E"], self.parameters["A"], self.parameters["alpha"])
        return self.parameters["intercept"] + self.parameters["slope"] * log_compute

    def latex(self) -> str:
        if self.form == "power":
            p = self.parameters
            return f"${p['E']:.3f} + {p['A']:.3f}\\,c^{{-{p['alpha']:.3f}}}$"
        p = self.parameters
        intercept_at_unit = p["intercept"] + p["slope"] * np.log10(COMPUTE_UNIT)
        return f"${intercept_at_unit:.3f} {p['slope']:+.3f}\\,\\log_{{10}} c$"


@dataclass(frozen=True)
class Equivalence:
    target: str
    baseline: str
    mariner_compute: float
    mariner_loss: float
    baseline_compute_to_match: float
    multiplier: float
    compute_saved_pct: float
    loss_improvement_pct: float
    extrapolated: bool


def power_law(log_compute: float, e: float, a: float, alpha: float) -> float:
    return e + a * 10 ** (-alpha * (log_compute - np.log10(COMPUTE_UNIT)))


def fit_baseline(target: str, baseline: str, curve: dict[float, float]) -> BaselineFit:
    compute = np.array(sorted(curve))
    loss = np.array([curve[c] for c in compute])
    log_compute = np.log10(compute)
    if len(compute) >= POWER_LAW_MIN_RUNGS:
        fitted = curve_fit(
            power_law, log_compute, loss, p0=[0.5, 0.5, 0.3], bounds=([0, 0, 0.01], [2, 5, 2]), maxfev=20000
        )
        e, a, alpha = (float(value) for value in fitted[0])
        parameters = {"E": float(e), "A": float(a), "alpha": float(alpha)}
        predicted = power_law(log_compute, e, a, alpha)
        form = "power"
    else:
        slope, intercept = np.polyfit(log_compute, loss, 1)
        parameters = {"intercept": float(intercept), "slope": float(slope)}
        predicted = intercept + slope * log_compute
        form = "loglinear"
    rmse = float(np.sqrt(np.mean((predicted - loss) ** 2)))
    return BaselineFit(target, baseline, form, parameters, rmse, float(log_compute.min()), float(log_compute.max()))


def _loss_deficit(log_compute: float, fit: BaselineFit, loss: float) -> float:
    return fit.loss(log_compute) - loss


def equivalences(fit: BaselineFit, mariner: dict[float, float]) -> list[Equivalence]:
    rows = []
    low, high = fit.log_compute_min - BRACKET_DECADES, fit.log_compute_max + BRACKET_DECADES
    for compute, loss in sorted(mariner.items()):
        if (fit.loss(low) - loss) * (fit.loss(high) - loss) > 0:
            continue
        log_match = float(brentq(_loss_deficit, low, high, args=(fit, loss)))
        match = 10**log_match
        baseline_loss_here = fit.loss(np.log10(compute))
        rows.append(
            Equivalence(
                target=fit.target,
                baseline=fit.baseline,
                mariner_compute=compute,
                mariner_loss=loss,
                baseline_compute_to_match=match,
                multiplier=match / compute,
                compute_saved_pct=100 * (1 - compute / match),
                loss_improvement_pct=100 * (baseline_loss_here - loss) / baseline_loss_here,
                extrapolated=not fit.log_compute_min <= log_match <= fit.log_compute_max,
            )
        )
    return rows


def latex_rows(fits: list[BaselineFit], rows: list[Equivalence]) -> str:
    """Appendix table rows: one per objective and baseline, a cell per ladder rung."""
    by_key = {(r.target, r.baseline, r.mariner_compute): r for r in rows}
    lines = []
    for target, target_label in TARGETS:
        lines.append(f"\\multicolumn{{{3 + len(LADDER_RUNGS)}}}{{l}}{{\\emph{{{target_label}}}}} \\\\")
        for fit in fits:
            if fit.target != target:
                continue
            cells = []
            for rung in LADDER_RUNGS:
                row = by_key.get((target, fit.baseline, rung))
                if row is None:
                    cells.append(PLACEHOLDER)
                elif row.extrapolated:
                    cells.append("--")
                else:
                    cells.append(f"${row.multiplier:.2f}\\times$ ({row.compute_saved_pct:.0f}\\%)")
            lines.append(f"\\quad {fit.baseline} & {fit.latex()} & {fit.rmse:.4f} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(PLOTTED_POINTS)
    curves: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for row in points.to_dict("records"):
        curves[(str(row["target"]), str(row["label"]))][float(row["flops"])] = float(row["mean"])
    fits, rows = [], []
    for target, _ in TARGETS:
        for baseline in BASELINES:
            fit = fit_baseline(target, baseline, curves[(target, baseline)])
            fits.append(fit)
            rows.extend(equivalences(fit, curves[(target, MARINER)]))
    pd.DataFrame([asdict(r) for r in rows]).to_csv(args.output_dir / "compute_equivalence.csv", index=False)
    with open(args.output_dir / "fits.json", "w") as handle:
        json.dump([asdict(f) for f in fits], handle, indent=1)
    (args.output_dir / "compute_equivalence_rows.tex").write_text(latex_rows(fits, rows))
    summary = []
    for fit in fits:
        cells = [
            f"{r.mariner_compute:.0e}: {r.multiplier:.2f}x ({r.compute_saved_pct:.0f}% less; "
            f"loss {-r.loss_improvement_pct:+.1f}%)" + (" *" if r.extrapolated else "")
            for r in rows
            if (r.target, r.baseline) == (fit.target, fit.baseline)
        ]
        summary.append(f"{fit.target:12s} vs {fit.baseline:13s} [{fit.form}; rmse {fit.rmse:.4f}]  " + ", ".join(cells))
    summary.append("* = beyond the baseline's measured rungs (extrapolated)")
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
