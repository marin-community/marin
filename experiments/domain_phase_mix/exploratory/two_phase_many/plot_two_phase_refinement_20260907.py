# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "matplotlib==3.10.8"]
# ///
"""Render the matched refinement comparison as ratios to the HPR baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

OUTPUT = Path(__file__).resolve().parent / "reference_outputs/two_phase_refinement_20260907"
MODELS = ["hpr", "CRE2-013", "CRE2-016", "CRE2-017", "CRE2-018", "CRE2-019", "CRE2-020", "CRE2-021", "CRE2-022"]
LABELS = [
    "HPR reference",
    "013  Recency WSPU",
    "016  Cross-component response",
    "017  BPB fitting",
    "018  Fourfold phase weight",
    "019  Objective response",
    "020  Objective log link",
    "021  Local net response",
    "022  Local benefit / harm",
]


def main() -> None:
    endpoint_path = OUTPUT / "comparison/endpoint_metrics.csv"
    pair_path = OUTPUT / "comparison/pair_metrics.csv"
    endpoints = pd.read_csv(endpoint_path)
    pairs = pd.read_csv(pair_path)
    endpoints = endpoints[(endpoints.context == "oof") & (endpoints.population == "all")]
    pairs = pairs[pairs.context == "oof"]
    ratios, records = [], []
    for data, metric in [(endpoints, "rmse"), (pairs, "rmse"), (pairs, "mean_binary_decision_regret")]:
        for target in ["uncheatable", "table9"]:
            selected = data[data.objective == target].set_index("model")[metric]
            column = selected.loc[MODELS].to_numpy()
            ratios.append(column / selected["hpr"])
            for model, value in zip(MODELS, column, strict=True):
                records.append(
                    {
                        "model": model,
                        "objective": target,
                        "metric": metric,
                        "population": "endpoint" if data is endpoints else "paired",
                        "value": value,
                        "hpr_ratio": value / selected["hpr"],
                    }
                )
    ratios = np.column_stack(ratios)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    fig.subplots_adjust(left=0.29, right=0.91, top=0.81, bottom=0.16)
    norm = TwoSlopeNorm(vmin=0.8, vcenter=1, vmax=max(2.2, float(ratios.max())))
    rendered = ax.imshow(ratios, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(6), ["U", "Table 9", "U", "Table 9", "U", "Table 9"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(MODELS)), LABELS)
    ax.tick_params(length=0, pad=10)
    for x, title in [(0.5, "Endpoint RMSE"), (2.5, "Paired RMSE"), (4.5, "Paired regret")]:
        ax.text(x, -1.5, title, ha="center", va="center", fontsize=12, fontweight="bold")
    for row in range(len(MODELS)):
        for col in range(6):
            value = ratios[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}\N{MULTIPLICATION SIGN}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if value > 1.9 else "#17252b",
            )
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(False, which="major")
    ax.grid(which="minor", color="white", linewidth=2, linestyle="-")
    ax.tick_params(which="minor", length=0)
    ax.axhline(2.5, color="#17252b", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes((0.94, 0.19, 0.015, 0.55))
    colorbar = fig.colorbar(rendered, cax=cax)
    cax.grid(False)
    colorbar.set_label("Ratio to HPR", labelpad=7)
    fig.text(
        0.04, 0.965, "Six controlled refinements of the two-phase surrogate", fontsize=17, fontweight="bold", va="top"
    )
    fig.text(
        0.04, 0.911, "Lower is better. 1.00 matches HPR; rows below the divider are this round's six fits.", fontsize=11
    )
    fig.text(0.04, 0.07, "518 held-out endpoints · 238 matched phase pairs · U = Uncheatable", fontsize=10)
    fig.text(
        0.04,
        0.036,
        "Fixed development folds; these ratios do not establish prospective policy improvement.",
        fontsize=10,
        color="#455a64",
    )
    dest = OUTPUT / "figures"
    dest.mkdir(exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(dest / f"refinement_comparison.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    pd.DataFrame(records).to_csv(dest / "refinement_comparison.csv", index=False)
    sources = [Path(__file__), endpoint_path, pair_path]
    (dest / "manifest.json").write_text(
        json.dumps({"source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
