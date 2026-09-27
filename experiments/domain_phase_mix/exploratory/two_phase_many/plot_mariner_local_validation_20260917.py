# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///

"""Plot audited local policy-selection and conditional-response comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"
MODELS = ("hs17_mariner", "hs17_log1", "hs17_bounded_exp_c1", "cv17_smooth_d1", "cv17_huber_d8")
LABELS = ("MARINER", "Logarithmic", "Bounded exponential", "Convex smooth hinge", "Convex Huber hinge")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8B6F19")


def main() -> None:
    for style in ("Regular", "Bold"):
        font_manager.fontManager.addfont(Path.home() / "Library/Fonts" / f"NotoSans-{style}.ttf")
    plt.rcParams["font.family"] = "Noto Sans"
    plt.rcParams["text.usetex"] = False
    banks = pd.concat([pd.read_csv(ROOT / sub / "bank_metrics.csv") for sub in ("banks", "banks/convex")])
    banks = banks[banks.labels.eq("current_weights_complete") & banks.stratum.eq("all")]
    curves = pd.concat([pd.read_csv(ROOT / sub / "dose_summary.csv") for sub in ("banks", "banks/convex")])
    values = [
        banks[banks.panel.eq("delphi_3e18_39bucket") & banks.target.eq("table9")]
        .set_index("model")
        .loc[list(MODELS), "regret_at_1"]
        .to_numpy(),
        banks[banks.panel.eq("300m_39bucket") & banks.target.eq("table9")]
        .set_index("model")
        .loc[list(MODELS), "regret_at_1"]
        .to_numpy(),
        curves[curves.target.eq("uncheatable")].set_index("model").loc[list(MODELS), "rmse"].to_numpy(),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    titles = (
        "Qwen suite\nMeasured-policy regret",
        "Llama 200M suite\nMeasured-policy regret",
        "Qwen Uncheatable\nConditional-curve RMSE",
    )
    for ax, numbers, title in zip(axes, values, titles, strict=True):
        ax.barh(np.arange(5), numbers, color=COLORS, height=0.56)
        ax.axvline(numbers[0], color=COLORS[0], linestyle=":", lw=1, alpha=0.6)
        limit = max(numbers) * 1.31
        ax.set_xlim(0, limit)
        for i, value in enumerate(numbers):
            ax.text(value + limit * 0.022, i, f"{value:.4f}", va="center", fontsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=14)
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("BPB · lower is better", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_yticks(np.arange(5), LABELS, fontsize=11)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Better curve prediction does not guarantee better policy selection", x=0.025, ha="left", fontsize=16, y=0.99
    )
    fig.text(
        0.025,
        0.015,
        "Archived, shared-support comparisons. Regret = selected measured loss minus best measured loss; "
        "curve RMSE averages 39 focal-bucket paths.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.175, right=0.975, bottom=0.18, top=0.74, wspace=0.29)
    fig.savefig(ROOT / "selection_vs_curve_prediction.png", dpi=180)
    fig.savefig(ROOT / "selection_vs_curve_prediction.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
