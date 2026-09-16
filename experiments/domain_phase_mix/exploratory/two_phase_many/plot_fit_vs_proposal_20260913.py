# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Swarm fit against trained-proposal loss: the surrogates that fit the swarm best do not propose the best mixtures.

One point per surrogate and objective: x is the out-of-fold Spearman correlation on the Qwen3 swarm (the complexity
ladder, five mixture-blocked folds; the released RegMix recipe from its own out-of-fold records), y is the measured
loss of the surrogate's trained proposal minus MARINER's at the same data and trainer seeds (the paired table of the
baseline proposals; the released RegMix endpoints against MARINER's seed-zero run). Error bars are paired SEs where
three seeds were trained.

usage: uv run --offline --no-sync python plot_fit_vs_proposal_20260913.py [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE / "fit_vs_proposal_20260913"
LADDER = REFERENCE / "complexity_ladder_20260909" / "complexity_ladder.csv"
PAIRED = REFERENCE / "delphi_comparator_proposals_3e18_20260909" / "paired_results.csv"
RELEASED_REGMIX = REFERENCE / "delphi_regmix_reference_3e18_20260913" / "measured_results.csv"
RELEASED_OOF = REFERENCE / "regmix_official_oof_20260913"
MARINER_SEED0 = REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908" / "measured_results.csv"
DRIVE_STEM = "r6_fit_vs_proposal"
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))
OBJECTIVE_COLUMN = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
MARINER_POLICY = {"uncheatable": "lwspu_u_snc_cap06", "table9": "lwspu_t9_snc_cap08"}
# Ladder model id, paired-table candidate key, short label, colour (the learning-curve and diagnostics palettes).
SURROGATES = (
    ("weibull_softplus_unscaled@kappa_floor_link_flat15_nocap", "MARINER", "MARINER", "#469C76"),
    ("olmix_loglinear_taskwise", "Olmix", "Olmix", "#CC79A7"),
    ("quadratic_log_epoch@kappa_floor_link_flat15_nocap", "Quadratic in log-epochs, floor link", "Quadratic", "#56B4E9"),
    (
        "spline_log_epoch@kappa_floor_link_flat15_nocap",
        "Natural cubic spline in log-epochs, floor link",
        "Cubic spline",
        "#D55E00",
    ),
    ("hellinger_krr", "Hellinger kernel ridge", "Kernel ridge", "#E69F00"),
    ("lightgbm_regmix", "LightGBM (RegMix)", "RegMix, tuned trees", "#6C6F7D"),
)
# Label offsets in points, chosen by hand so nothing overlaps.
LABEL_OFFSETS = {
    ("uncheatable", "MARINER"): (6, -3),
    ("uncheatable", "Olmix"): (-6, 6),
    ("uncheatable", "Quadratic"): (-6, 6),
    ("uncheatable", "Cubic spline"): (6, 3),
    ("uncheatable", "Kernel ridge"): (6, 3),
    ("uncheatable", "RegMix, tuned trees"): (6, -8),
    ("uncheatable", "RegMix, released recipe"): (6, -3),
    ("table9", "MARINER"): (6, -3),
    ("table9", "Olmix"): (-6, 6),
    ("table9", "Quadratic"): (6, 3),
    ("table9", "Cubic spline"): (6, -3),
    ("table9", "Kernel ridge"): (6, -8),
    ("table9", "RegMix, tuned trees"): (6, 3),
    ("table9", "RegMix, released recipe"): (6, -3),
}
RELEASED_LABEL = "RegMix, released recipe"
RELEASED_COLOR = "#0072B2"
PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
DPI = 300


def released_recipe_rank_correlation(panel: benchmark.BenchPanel, target: str) -> float:
    """Mean out-of-fold Spearman correlation of the released recipe over its ten fold assignments."""
    group = panel.group(target)
    values = []
    for draw in range(10):
        payload = np.load(RELEASED_OOF / f"{target}_draw{draw}.npz")
        predicted = payload["oof"] @ group.aggregation_weights
        values.append(stats.spearmanr(predicted, group.aggregate[payload["rows"]]).statistic)
    return float(np.mean(values))


def points_table() -> pd.DataFrame:
    ladder = pd.read_csv(LADDER)
    # The ladder lists a model once per extra results directory; keep the row that carries the Qwen correlations.
    ladder = ladder.dropna(subset=["spearman:delphi_3e18_39bucket:uncheatable"]).drop_duplicates("model")
    ladder = ladder.set_index("model")
    paired = pd.read_csv(PAIRED)
    released = pd.read_csv(RELEASED_REGMIX)
    mariner_seed0 = pd.read_csv(MARINER_SEED0)
    panel = benchmark.load_panel(fits.PANEL)
    rows = []
    for target, target_label in TARGETS:
        for model_id, comparator, label, color in SURROGATES:
            row = paired[paired["target"].eq(target) & paired["comparator"].eq(comparator)]
            if len(row) != 1:
                raise ValueError(f"{target}/{comparator}: {len(row)} paired rows")
            row = row.iloc[0]
            rows.append(
                {
                    "target": target,
                    "target_label": target_label,
                    "label": label,
                    "color": color,
                    "spearman": float(ladder.loc[model_id, f"spearman:delphi_3e18_39bucket:{target}"]),
                    "difference": float(row["difference_mean"]),
                    "difference_se": float(row["difference_se"]) if row["seeds"] > 1 else float("nan"),
                    "seeds": int(row["seeds"]),
                }
            )
        endpoint = released[
            released["candidate_id"].str.contains(f"rgref_{'u' if target == 'uncheatable' else 't9'}_endpoint")
        ]
        reference = mariner_seed0[mariner_seed0["candidate_id"].eq(MARINER_POLICY[target])]
        rows.append(
            {
                "target": target,
                "target_label": target_label,
                "label": RELEASED_LABEL,
                "color": RELEASED_COLOR,
                "spearman": released_recipe_rank_correlation(panel, target),
                "difference": float(
                    endpoint[OBJECTIVE_COLUMN[target]].item() - reference[OBJECTIVE_COLUMN[target]].item()
                ),
                "difference_se": float("nan"),
                "seeds": 1,
            }
        )
    return pd.DataFrame(rows)


def draw(points: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    for axis, (target, target_label) in zip(axes, TARGETS, strict=True):
        part = points[points["target"].eq(target)]
        axis.axhline(0.0, color=INK, linewidth=0.6, zorder=1)
        axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
        for row in part.itertuples():
            if np.isfinite(row.difference_se):
                axis.errorbar(
                    row.spearman,
                    row.difference,
                    yerr=row.difference_se,
                    fmt="none",
                    ecolor=row.color,
                    elinewidth=0.8,
                    capsize=2,
                    zorder=3,
                )
            axis.plot(row.spearman, row.difference, "o", color=row.color, markeredgecolor=INK, markersize=5.5, zorder=4)
            offset = LABEL_OFFSETS[(target, row.label)]
            axis.annotate(
                row.label,
                (row.spearman, row.difference),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                fontsize=6.2,
                color=INK,
            )
        axis.set_title(
            f"{'AB'[TARGETS.index((target, target_label))]}. {target_label}",
            loc="left",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
        axis.set_xlabel("Out-of-fold Spearman $\\rho$ on the swarm", fontsize=7.5, color=INK)
        axis.tick_params(labelsize=7, colors=INK)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        low, high = part["spearman"].min(), part["spearman"].max()
        axis.set_xlim(low - 0.03, min(high + 0.06, 1.0))
        axis.set_ylim(-0.006, part["difference"].max() + 0.008)
    axes[0].set_ylabel("Trained proposal, BPB above MARINER's", fontsize=7.5, color=INK)
    figure.tight_layout(w_pad=1.5)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)
    points = points_table()
    points.to_csv(OUTPUT_DIR / "points.csv", index=False)
    figure = draw(points)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT_DIR / f"fit_vs_proposal.{extension}", dpi=DPI, bbox_inches="tight")
        if args.drive_dir is not None:
            shutil.copy(OUTPUT_DIR / f"fit_vs_proposal.{extension}", args.drive_dir / f"{DRIVE_STEM}.{extension}")
    print(points.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
