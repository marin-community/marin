# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Figure 13 of the Analysis: MARINER's predicted against measured loss for the held-out optima, with the best 30 inset.

One panel per objective. Points are the held-out optima of the frozen Delphi bank (the coordinates the top-band ordering
uses: the optima stratum without the intervention sources), each the mean over its runs, with MARINER's frozen
predictions (`delphi_kappa_floor_flat15_selection_20260907/predictions.csv`, external_development population). The 30
best-measured coordinates are outlined and enlarged in an inset. Spearman correlations over all coordinates and over the
best 30 are printed on the panels.

Writes `analysis_calibration.pdf/.png` and `plotted_values.csv` to
`reference_outputs/analysis_calibration_figure_20260929/`.

usage: uv run --offline --no-sync python plot_analysis_calibration_20260929.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
BENCHMARK = REFERENCE / "delphi_offline_selection_20260906"
PREDICTIONS = REFERENCE / "delphi_kappa_floor_flat15_selection_20260907" / "predictions.csv"
OUTPUT = REFERENCE / "analysis_calibration_figure_20260929"
METHOD = "weibull_softplus_unscaled@kappa_floor_link_flat15"
POPULATION = "external_development"
INTERVENTIONS = ("conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623")
BAND = 30
TARGETS = (("uncheatable", "A. Uncheatable"), ("table9", "B. OlmoBaseEval Easy"))
COLORS = {"uncheatable": "#0072B2", "table9": "#D55E00"}
GRID = "#DDDDDD"
DIAGONAL = "#B5B5B5"


def held_out(target: str) -> pd.DataFrame:
    labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
    labels = labels[~labels.sources.str.contains("|".join(INTERVENTIONS))].sort_values("measured_mean_bpb")
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[
        predictions.population.eq(POPULATION) & predictions.method.eq(METHOD) & predictions.target.eq(target)
    ]
    predicted = predictions.set_index("row_id").loc[labels.coordinate_id].prediction.to_numpy(float)
    table = pd.DataFrame(
        {
            "target": target,
            "coordinate_id": labels.coordinate_id.to_numpy(),
            "runs": labels.run_count.to_numpy(),
            "measured_bpb": labels.measured_mean_bpb.to_numpy(float),
            "predicted_bpb": predicted,
        }
    )
    table["best_band"] = np.arange(len(table)) < BAND
    return table


def draw(axis, table: pd.DataFrame, color: str, title: str) -> dict[str, float]:
    measured, predicted, best = table.measured_bpb.to_numpy(), table.predicted_bpb.to_numpy(), table.best_band.to_numpy()
    lo, hi = min(measured.min(), predicted.min()), max(measured.max(), predicted.max())
    pad = 0.04 * (hi - lo)
    axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=DIAGONAL, linewidth=0.8, zorder=1)
    axis.scatter(measured[~best], predicted[~best], s=6, color=color, alpha=0.35, linewidths=0, zorder=2)
    axis.scatter(measured[best], predicted[best], s=9, color=color, edgecolor="#222222", linewidths=0.4, zorder=3)
    axis.set_xlim(lo - pad, hi + pad)
    axis.set_ylim(lo - pad, hi + pad)
    axis.set_aspect("equal")
    axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
    axis.set_xlabel("Measured loss (BPB)", fontsize=7)
    axis.set_ylabel("Predicted loss (BPB)", fontsize=7)
    axis.tick_params(labelsize=6.3)
    axis.grid(True, color=GRID, linewidth=0.5, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    rho_all = float(stats.spearmanr(measured, predicted).statistic)
    rho_best = float(stats.spearmanr(measured[best], predicted[best]).statistic)
    axis.text(
        0.07,
        0.46,
        f"Spearman {rho_all:.2f} (all), {rho_best:.2f} (best {BAND})",
        transform=axis.transAxes,
        fontsize=6.3,
        ha="left",
        va="top",
    )
    # Inset: the best band alone.
    blo, bhi = min(measured[best].min(), predicted[best].min()), max(measured[best].max(), predicted[best].max())
    bpad = 0.1 * (bhi - blo)
    axis.add_patch(
        Rectangle(
            (blo - bpad, blo - bpad),
            bhi - blo + 2 * bpad,
            bhi - blo + 2 * bpad,
            fill=False,
            edgecolor="#555555",
            linewidth=0.5,
            zorder=4,
        )
    )
    inset = axis.inset_axes([0.07, 0.5, 0.46, 0.46])
    inset.plot([blo - bpad, bhi + bpad], [blo - bpad, bhi + bpad], color=DIAGONAL, linewidth=0.7, zorder=1)
    inset.scatter(measured[best], predicted[best], s=8, color=color, edgecolor="#222222", linewidths=0.4, zorder=3)
    inset.set_xlim(blo - bpad, bhi + bpad)
    inset.set_ylim(blo - bpad, bhi + bpad)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"best {BAND}", fontsize=6, pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#555555")
    return {
        "n": len(table),
        "spearman_all": rho_all,
        "spearman_best": rho_best,
        "bias_all_bpb": float((predicted - measured).mean()),
        "bias_best_bpb": float((predicted[best] - measured[best]).mean()),
        "best_band_width_bpb": float(measured[best].max() - measured[best].min()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "text.usetex": False, "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(4.4, 2.0), gridspec_kw={"wspace": 0.4, "left": 0.09, "right": 0.99, "top": 0.88, "bottom": 0.19}
    )
    tables, summary = [], {}
    for axis, (target, title) in zip(axes, TARGETS, strict=True):
        table = held_out(target)
        tables.append(table)
        summary[target] = draw(axis, table, COLORS[target], title)
    pd.concat(tables, ignore_index=True).to_csv(OUTPUT / "plotted_values.csv", index=False)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"analysis_calibration.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(pd.DataFrame(summary).T.round(4).to_string())
    print(OUTPUT / "analysis_calibration.pdf")


if __name__ == "__main__":
    main()
