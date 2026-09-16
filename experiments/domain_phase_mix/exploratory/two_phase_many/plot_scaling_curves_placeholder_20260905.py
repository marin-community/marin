# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas"]
# ///

"""Plot completed fixed-mixture compute ladders with matched-Qwen Olmix policies.

Proportional and UniMax-8 use the archived ladder. MARINER and Olmix use their frozen
Qwen-fitted policies at every rung. At 3e18, proportional pools eleven runs and each
fitted policy has three trainer seeds; error bars show one run SD. Later rungs are
single runs. Missing final objective measurements are omitted.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT = SCRIPT_DIR / "reference_outputs" / "delphi_scaling_progress_20260625" / "delphi_scaling_completed_wandb.csv"
FAIRNESS_SUMMARY = SCRIPT_DIR / "reference_outputs" / "delphi_fairness_repeats_3e18_20260908" / "fairness_summary.csv"
MATCHED_FIRST_RUNG = SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_3e18_20260908" / "measured_results.csv"
MATCHED_OLMIX_RESULTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_scaling_v6e_20260910" / "measured_results.csv"
)
LADDER_RESULTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_frozen_procedure_scaling_v6e_20260908" / "measured_results.csv"
)
RELIABILITY_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
PROPORTIONAL_REPEATS = 10
NOISE_SUMMARIES = {
    "uncheatable": RELIABILITY_DIR / "snr_fit_uncheatable_delphi.csv",
    "table9": RELIABILITY_DIR / "snr_fit_tasks_delphi.csv",
}
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "scaling_curves_placeholder_20260905"
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
MARINER_COLOR = "#469C76"
OLMIX_COLOR = "#CC79A7"
PROPORTIONAL_COLOR = "#6C6F7D"
UNIMAX_COLOR = "#4C78A8"
SCALES = (3e18, 2e19, 3e20, 1e21)
PANELS = (
    (
        "uncheatable",
        "eval_uncheatable_eval_bpb",
        "Uncheatable BPB",
        (
            ("proportional", "Proportional", PROPORTIONAL_COLOR, "-"),
            ("unimax8", "UniMax-8", UNIMAX_COLOR, "-"),
            (
                "olmixq_u_kl0p05_cap04",
                "Olmix",
                OLMIX_COLOR,
                "-",
            ),
        ),
        "lwspu_u_snc_cap06",
        "mean",
    ),
    (
        "table9",
        "olmo_base_easy_table9_51_component_macro_bpb",
        "OlmoBaseEval Easy mean BPB",
        (
            ("proportional", "Proportional", PROPORTIONAL_COLOR, "-"),
            ("unimax8", "UniMax-8", UNIMAX_COLOR, "-"),
            (
                "olmixq_t9_kl0p005_cap04",
                "Olmix",
                OLMIX_COLOR,
                "-",
            ),
        ),
        "lwspu_t9_snc_cap08",
        "mean",
    ),
)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
DPI = 300


def first_rung_statistics(
    snapshot: pd.DataFrame, fairness: pd.DataFrame, noise: pd.DataFrame, matched: pd.DataFrame
) -> pd.DataFrame:
    """Pool proportional repeats and use only the specified objective's fitted policies."""
    rows = []
    for target, column, _y_label, series, mariner_id, _mean_column in PANELS:
        metric = "uncheatable_bpb" if target == "uncheatable" else "table9_macro_bpb"
        anchor = snapshot[snapshot["mixture"].eq("proportional") & snapshot["flops"].eq(SCALES[0])]
        assert len(anchor) == 1
        anchor_value = float(anchor[column].iloc[0])
        repeat = noise[noise["target"].eq(target)].iloc[0]
        repeat_mean, repeat_sd = float(repeat["proportional_mean"]), float(repeat["repeat_sd"])
        n = PROPORTIONAL_REPEATS
        pooled_mean = (n * repeat_mean + anchor_value) / (n + 1)
        pooled_sd = (((n - 1) * repeat_sd**2 + n / (n + 1) * (anchor_value - repeat_mean) ** 2) / n) ** 0.5
        rows.append(
            {
                "target": target,
                "mixture": "proportional",
                "mean": pooled_mean,
                "sd": pooled_sd,
                "n": n + 1,
                "source_csv": f"{SNAPSHOT};{NOISE_SUMMARIES[target]}",
                "source_rows": f"{anchor['run_base'].iloc[0]};proportional repeats",
                "source_metric": f"{column};proportional_mean",
            }
        )
        mariner = fairness[fairness["kind"].eq("policy") & fairness["candidate_id"].eq(mariner_id)]
        assert len(mariner) == 1, mariner_id
        row = mariner.iloc[0]
        assert row["metric"] == metric and int(row["n"]) == 3
        rows.append(
            {
                "target": target,
                "mixture": mariner_id,
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "n": int(row["n"]),
                "source_csv": str(FAIRNESS_SUMMARY),
                "source_rows": mariner_id,
                "source_metric": metric,
            }
        )
        olmix_id = series[-1][0]
        olmix = matched[
            matched["candidate_id"].eq(olmix_id) & matched["target"].eq(target) & matched["status"].eq("measured")
        ]
        assert len(olmix) == 3 and set(olmix["trainer_seed"]) == {0, 1, 2}, olmix_id
        assert olmix[metric].notna().all(), olmix_id
        rows.append(
            {
                "target": target,
                "mixture": olmix_id,
                "mean": float(olmix[metric].mean()),
                "sd": float(olmix[metric].std(ddof=1)),
                "n": len(olmix),
                "source_csv": str(MATCHED_FIRST_RUNG),
                "source_rows": ";".join(olmix["group"]),
                "source_metric": metric,
            }
        )
    return pd.DataFrame(rows)


def plotted_points(snapshot: pd.DataFrame, repeats: pd.DataFrame, ladder_path: Path, matched_path: Path) -> pd.DataFrame:
    """Collect every visible point with its run count and source before rendering."""
    ladders = {"MARINER": pd.read_csv(ladder_path), "Olmix": pd.read_csv(matched_path)}
    rows = []
    for target, column, _y_label, series, mariner_id, _mean_column in PANELS:
        for mixture, label, _color, _style in series[:2]:
            frame = snapshot[snapshot["mixture"].eq(mixture) & snapshot["is_completed"] & snapshot[column].notna()]
            for _, row in frame.iterrows():
                if mixture == "proportional" and float(row["flops"]) == SCALES[0]:
                    continue
                rows.append(
                    {
                        "target": target,
                        "mixture": mixture,
                        "label": label,
                        "flops": float(row["flops"]),
                        "mean": float(row[column]),
                        "sd": float("nan"),
                        "n": 1,
                        "source_csv": str(SNAPSHOT),
                        "source_rows": row["run_base"],
                        "source_metric": column,
                    }
                )
        labels = {"proportional": "Proportional", mariner_id: "MARINER", series[-1][0]: "Olmix"}
        for _, row in repeats[repeats["target"].eq(target)].iterrows():
            rows.append({**row.to_dict(), "label": labels[row["mixture"]], "flops": SCALES[0]})
        metric = "uncheatable_bpb" if target == "uncheatable" else "table9_macro_bpb"
        for label, policy, source in (("MARINER", mariner_id, ladder_path), ("Olmix", series[-1][0], matched_path)):
            ladder = ladders[label]
            complete = ladder[
                ladder["policy"].eq(policy)
                & ladder["target"].eq(target)
                & ladder["status"].eq("measured")
                & ladder[metric].notna()
                & ladder["target_flops"].gt(SCALES[0])
            ]
            for _, row in complete.iterrows():
                rows.append(
                    {
                        "target": target,
                        "mixture": policy,
                        "label": label,
                        "flops": float(row["target_flops"]),
                        "mean": float(row[metric]),
                        "sd": float("nan"),
                        "n": 1,
                        "source_csv": str(source),
                        "source_rows": row["run_name"],
                        "source_metric": metric,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["target", "label", "flops"])
    assert not frame.duplicated(["target", "mixture", "flops"]).any()
    assert frame["mean"].notna().all() and frame["flops"].isin(SCALES).all()
    return frame


def build_figure(points: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    for axis, (target, _column, y_label, series, mariner_id, _mean_column), letter in zip(
        axes, PANELS, "AB", strict=True
    ):
        tracks = (*series, (mariner_id, "MARINER optimum", MARINER_COLOR, "-"))
        for mixture, label, color, style in tracks:
            frame = points[points["target"].eq(target) & points["mixture"].eq(mixture)].sort_values("flops")
            axis.plot(
                frame["flops"],
                frame["mean"],
                color=color,
                linestyle=style,
                linewidth=1.3,
                marker="o",
                markersize=3.2,
                markerfacecolor=color,
                markeredgecolor=PAPER,
                zorder=4 if mixture == mariner_id else 3,
                label=label,
            )
            repeated = frame[frame["n"].gt(1)]
            axis.errorbar(
                repeated["flops"],
                repeated["mean"],
                yerr=repeated["sd"],
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=4,
            )
            if mixture == mariner_id:
                axis.plot(
                    repeated["flops"],
                    repeated["mean"],
                    marker="o",
                    markersize=3.8,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=INK,
                    markeredgewidth=0.5,
                    linestyle="none",
                    zorder=5,
                )
        axis.set_xscale("log")
        axis.set_xticks(SCALES)
        axis.set_xticklabels(["3e18", "2e19", "3e20", "1e21"])
        axis.minorticks_off()
        axis.set_xlabel("Training compute (FLOPs; log scale)", color=INK, fontsize=7.5)
        axis.set_ylabel(y_label, color=INK, fontsize=7.5)
        axis.set_title(
            f"{letter}. {'Uncheatable' if target == 'uncheatable' else 'OlmoBaseEval Easy'}",
            loc="left",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
        axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
        axis.tick_params(colors=INK, labelsize=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK)
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["Olmix (cap 4)\nKL 0.05 (A), 0.005 (B)" if label == "Olmix" else label for label in labels]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.6,
        columnspacing=1.2,
    )
    figure.tight_layout(w_pad=1.5, rect=(0, 0.09, 1, 1))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--ladder-results", type=Path, default=LADDER_RESULTS)
    parser.add_argument("--matched-olmix-results", type=Path, default=MATCHED_OLMIX_RESULTS)
    args = parser.parse_args()
    snapshot = pd.read_csv(SNAPSHOT)
    fairness = pd.read_csv(FAIRNESS_SUMMARY)
    noise = pd.concat([pd.read_csv(path).iloc[[0]].assign(target=target) for target, path in NOISE_SUMMARIES.items()])
    matched = pd.read_csv(MATCHED_FIRST_RUNG)
    repeats = first_rung_statistics(snapshot, fairness, noise, matched)
    points = plotted_points(snapshot, repeats, args.ladder_results, args.matched_olmix_results)
    plt.rcParams.update(PLOT_STYLE)
    figure = build_figure(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repeats.to_csv(args.output_dir / "first_rung_statistics.csv", index=False)
    points.to_csv(args.output_dir / "plotted_points.csv", index=False)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"scaling_curves_placeholder.{extension}", dpi=DPI, bbox_inches="tight")
        if args.drive_dir is not None:
            shutil.copyfile(
                args.output_dir / f"scaling_curves_placeholder.{extension}",
                args.drive_dir / f"r1_scaling_curves_placeholder.{extension}",
            )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
