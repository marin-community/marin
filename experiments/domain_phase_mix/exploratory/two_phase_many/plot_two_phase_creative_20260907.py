# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "matplotlib==3.10.8"]
# ///
"""Plot all creative-screen formulations against the same HPR prediction baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_creative_sweep_20260907"
LABELS = (
    "01  Joint phase-blind control",
    "02  Recency benefit / cumulative harm",
    "03  Retained acquisition / prefix harm",
    "04  Phase renewal / reset",
    "05  Rehearsal-protected benefit",
    "06  Early damage recovery",
    "07  Competitive retention",
    "08  Continuous benefit aging",
    "09  Raw order + positive even cost",
    "10  Reduced-rank component response",
    "11  WSPU semantic kernel",
    "12  Raw-policy kernel",
    "13  Expanded recency range",
    "14  Separate acquisition / shared harm",
    "15  Linear semantic response",
    "16  Semantic order + positive cost",
)


def main() -> None:
    endpoint_path = ROOT / "comparison/endpoint_metrics.csv"
    phase_path = ROOT / "comparison/pair_metrics.csv"
    endpoint = pd.read_csv(endpoint_path)
    phase = pd.read_csv(phase_path)
    endpoint = endpoint[(endpoint.context == "oof") & (endpoint.population == "all")]
    phase = phase[phase.context == "oof"]
    ids = [f"CRE2-{i:03d}" for i in range(1, 17)]
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 4, figsize=(17, 7.8), sharey=True)
    y = np.arange(len(ids))
    colors = mpl.colormaps["RdYlGn_r"]
    normalization = mpl.colors.TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=2.4)
    records = []
    for axis, (objective, metric, table) in zip(
        axes,
        (
            ("uncheatable", "Endpoint RMSE", endpoint),
            ("table9", "Endpoint RMSE", endpoint),
            ("uncheatable", "Phase-difference RMSE", phase),
            ("table9", "Phase-difference RMSE", phase),
        ),
        strict=True,
    ):
        selected = table[table.objective == objective].set_index("model")
        reference = float(selected.loc["hpr", "rmse"])
        ratio = selected.loc[ids, "rmse"].to_numpy() / reference
        axis.axvline(1, color="#263238", linewidth=1.2, linestyle="--", zorder=1)
        for row, value in zip(y, ratio, strict=True):
            color = colors(normalization(value))
            axis.plot([1, value], [row, row], color=color, linewidth=3, solid_capstyle="round", zorder=2)
            axis.scatter(value, row, color=color, edgecolor="#3b454b", linewidth=0.5, s=52, zorder=3)
            records.append({"model": ids[row], "objective": objective, "metric": metric, "ratio_to_hpr": float(value)})
        for line in (3.5, 7.5, 11.5):
            axis.axhline(line, color="#dce1e5", linewidth=0.8)
        axis.axhspan(11.5, 15.5, color="#f1f5f8", zorder=0)
        axis.set_xlim(0.8, 2.45)
        axis.set_xticks([1, 1.5, 2])
        axis.set_ylim(15.7, -0.7)
        axis.set_title(("Uncheatable" if objective == "uncheatable" else "Table 9") + "\n" + metric, fontsize=12, pad=13)
        axis.set_xlabel("Ratio to HPR  ↓", labelpad=10)
        axis.grid(axis="x", color="#e9ecef", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#a4aeb5")
        axis.tick_params(axis="y", length=0, pad=12)
    axes[0].set_yticks(y, LABELS)
    figure.suptitle("Prediction error across sixteen two-phase formulations", fontsize=17, x=0.51, y=0.985)
    figure.text(
        0.285,
        0.034,
        "Dashed line: matched HPR. Lower is better. Shaded rows: adaptive follow-ups.\n"
        "Grouped out-of-fold development results on 518 endpoints / 238 exact phase pairs; no prospective validation.",
        fontsize=10,
        color="#42515b",
        ha="left",
        va="bottom",
    )
    figure.subplots_adjust(left=0.285, right=0.985, top=0.87, bottom=0.145, wspace=0.19)
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"formulation_screen.{extension}", dpi=180, facecolor="white")
    plt.close(figure)
    pd.DataFrame(records).to_csv(output / "formulation_screen_data.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), endpoint_path, phase_path)
                }
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
