# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "matplotlib==3.10.8"]
# ///
"""Plot measured and held-out predicted BPB contrasts for three matched models.

Reads existing counterpart predictions only; no model is fitted. Axes share
limits within each objective and use the same scale for observed/predicted BPB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "reference_outputs/two_phase_link_transfer_20260907/matched_pairs.csv"
OBJECTIVES = (("uncheatable", "Uncheatable", 0.02), ("table9", "Table-9", 0.05))
# Okabe-Ito colors; model identity is also encoded by column headings.
MODELS = (
    ("aggregate", "Aggregate only", "#777777"),
    ("benefit+damage", "Benefit + damage", "#0072B2"),
    ("hierarchical_phase_replay", "HPR", "#D55E00"),
)
FIGURE_NAME = "matched_pair_contrasts"


def tick_label(value: float, _position: float) -> str:
    return "0" if abs(value) < 1e-12 else f"{value:.2f}".replace("-", "\N{MINUS SIGN}")


def make_figure(pairs: Path, output: Path) -> dict[str, object]:
    """Export one figure with complete pairs, shared scales, and source metadata."""
    frame = pd.read_csv(pairs)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.usetex": False,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.6,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(2, 3, figsize=(10.8, 8.0), sharex="row", sharey="row")
        figure.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.85, wspace=0.16, hspace=0.22)
        figure.suptitle("Measured phase effects vs. held-out predictions", x=0.545, y=0.97, fontsize=15)
        figure.text(
            0.545,
            0.925,
            "Exact counterpart pairs · negative contrast means the asymmetric schedule helps",
            ha="center",
            fontsize=10,
            color="#444444",
        )
        for row, (objective, label, step) in enumerate(OBJECTIVES):
            selected = frame.loc[frame["objective"].eq(objective) & frame["model"].isin([item[0] for item in MODELS])]
            if selected.empty or selected.duplicated(["model", "group"]).any():
                raise ValueError(f"missing or duplicate counterpart records for {objective}")
            reference = selected.loc[selected["model"].eq(MODELS[0][0])].sort_values("group")
            values = selected[["measured_delta", "predicted_delta"]].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite counterpart contrasts for {objective}")
            limit = step * np.ceil(1.08 * float(np.max(np.abs(values))) / step)
            tick_step = 2 * step if 2 * limit / step > 6 + 1e-9 else step
            ticks = np.arange(-limit, limit + tick_step / 2, tick_step)
            for column, (model, title, color) in enumerate(MODELS):
                axis = axes[row, column]
                records = selected.loc[selected["model"].eq(model)].sort_values("group")
                for field in ("group", "fold", "measured_delta"):
                    if not np.array_equal(records[field].to_numpy(), reference[field].to_numpy()):
                        raise ValueError(f"models do not use matching {field} records for {objective}")
                observed = records["measured_delta"].to_numpy()
                predicted = records["predicted_delta"].to_numpy()
                rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
                axis.axhline(0, color="#C5C5C5", linewidth=0.7, zorder=0)
                axis.axvline(0, color="#C5C5C5", linewidth=0.7, zorder=0)
                axis.plot(
                    [-limit, limit], [-limit, limit], color="#666666", linewidth=1, linestyle=(0, (4, 3)), zorder=0
                )
                axis.scatter(observed, predicted, s=17, color=color, alpha=0.78, edgecolors="white", linewidths=0.25)
                axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
                axis.xaxis.set_major_locator(FixedLocator(ticks))
                axis.yaxis.set_major_locator(FixedLocator(ticks))
                axis.xaxis.set_major_formatter(FuncFormatter(tick_label))
                axis.yaxis.set_major_formatter(FuncFormatter(tick_label))
                axis.tick_params(labelsize=9)
                axis.text(
                    0.035,
                    0.96,
                    f"n = {len(records)}\nRMSE = {rmse:.4f}",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=9,
                    color="#333333",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                )
                if row == 0:
                    axis.set_title(title, fontsize=12, color="#222222", pad=10)
                if column == 0:
                    axis.set_ylabel(f"{label}\nOOF predicted contrast (BPB)", fontsize=10, labelpad=9)
                metrics.append(
                    {"objective": objective, "model": model, "pairs": len(records), "rmse": rmse, "axis_limit": limit}
                )
        figure.supxlabel("Observed asymmetric minus tied BPB", x=0.545, y=0.067, fontsize=11)
        figure.text(
            0.545,
            0.025,
            "Dashed line: perfect prediction.  Solid gray lines: zero phase effect.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        for extension in ("svg", "pdf", "png"):
            figure.savefig(output / f"{FIGURE_NAME}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
    metadata: dict[str, object] = {
        "source": str(pairs.resolve()),
        "source_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "panels": metrics,
        "sign_convention": "asymmetric minus tied BPB; negative means asymmetric helps",
        "output_sha256": {
            f"{FIGURE_NAME}.{extension}": (
                hashlib.sha256((output / f"{FIGURE_NAME}.{extension}").read_bytes()).hexdigest()
            )
            for extension in ("svg", "pdf", "png")
        },
    }
    (output / f"{FIGURE_NAME}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, nargs="?", default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = make_figure(args.pairs, args.output or args.pairs.parent / "figures")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
