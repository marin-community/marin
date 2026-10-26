# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Which OlmoBaseEval Easy components and groups does MARINER predict well among the held-out optima?

Uses the held-out optima's per-component values (`heldout_table9_components_20260929`, from their W&B evaluations)
and the frozen reference fit's per-task predictions at each coordinate's mixture (as in
`preview_band_ordering_snr_subsets_20260929.py`). Per component: SNR (swarm std over proportional-repeat SD),
Spearman over all coordinates, and within the 30 best-measured coordinates on that component the pairwise sign
accuracy (pairs whose measured difference exceeds the component's repeat SD), the oracle's accuracy under that
noise, and the prediction bias. Per group (equal-weight mean of its components): the same band statistics for
k = 10 to all, as in Figure 13B.

Writes `component_predictability.csv`, `group_band_ordering.csv`, `group_band_ordering.png` and
`component_predictability.png` to `reference_outputs/heldout_table9_components_20260929/`.

usage: uv run --offline --no-sync python analyze_heldout_component_predictability_20260929.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import plot_analysis_floor_20260928 as floor  # noqa: E402
import plot_component_tradeoffs_obe_20260929 as obe  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402

OUTPUT = preview.OUTPUT
OLMIX_LAWS = preview.REFERENCE / "delphi_matched_olmix_3e18_20260908" / "laws_table9.json"
KERNEL_BANDWIDTH = 0.05
OLMIX_COLOR = "#C0397B"
BAND = 30
BAND_SIZES = (10, 20, 30, 50, 100, 0)
GROUPS = ("Code", "Math", "Basic Skills", "MMLU", "QA")
GROUP_COLORS = obe.GROUP_COLORS


def band_statistics(
    measured: np.ndarray, predicted: np.ndarray, blocks: np.ndarray, run_sd: float, counts: np.ndarray, k: int
) -> dict:
    order = np.argsort(measured, kind="stable")[:k]
    y, p, b, c = measured[order], predicted[order], blocks[order], counts[order]
    sd_run = run_sd / np.sqrt(c)
    pairs = np.array(list(itertools.combinations(np.arange(k), 2)))
    sd_diff = np.sqrt(sd_run[pairs[:, 0]] ** 2 + sd_run[pairs[:, 1]] ** 2)
    scores = top_band.band_scores(y, p, run_sd, sd_diff, b)
    return {
        "sign_accuracy": scores["sign_accuracy"],
        "noise_ceiling": scores["noise_ceiling"],
        "pairs": scores["pairs"],
        "spearman": float(stats.spearmanr(y, p).statistic),
        "bias_bpb": float((p - y).mean()),
        "band_width_bpb": float(y.max() - y.min()),
    }


def main() -> None:
    labels = pd.read_csv(preview.BENCHMARK / "inputs" / "table9_bank_labels.csv")
    labels = (
        labels[~labels.sources.str.contains("|".join(preview.INTERVENTIONS))]
        .sort_values("measured_mean_bpb")
        .reset_index(drop=True)
    )
    blocks = pd.read_csv(preview.BENCHMARK / "source_blocks.csv")
    blocks = (
        blocks[blocks.target.eq("table9")].set_index("coordinate_id").loc[labels.coordinate_id].source_block.to_numpy()
    )
    counts = labels.run_count.to_numpy(float)
    components = pd.read_csv(preview.CACHE)
    names = [c for c in components.columns if c not in ("row_id", "coordinate_id", "eval_run_id")]
    measured = components.groupby("coordinate_id")[names].mean().loc[labels.coordinate_id]
    panel = mat.bench.load_panel(mat.PANEL)
    swarm, to_panel = mat.reference_swarm_in_panel_order(panel)
    features = np.load(preview.BENCHMARK / "inputs" / "table9_bank_features.npz", allow_pickle=True)
    weights = pd.DataFrame(
        features["weights"], index=[str(c) for c in features["coordinate_id"]], columns=[str(b) for b in panel.buckets]
    ).loc[labels.coordinate_id]
    fit = mat.reference_fit(swarm, "table9", None)
    full = mat.ms.read_objectives(mat.ms.DATA / "objectives.csv")["table9"]
    predicted = pd.DataFrame(
        fit.predict_tasks(weights.to_numpy(float)[:, np.argsort(to_panel)]),
        index=labels.coordinate_id,
        columns=[preview.short_name(c) for c in full.components],
    )[names]
    laws = json.loads(OLMIX_LAWS.read_text())
    olmix_predicted = pd.DataFrame(
        {
            preview.short_name(law["component"]): olmix.OlmixLoglinearFit(
                law["log_c"], tuple(law["coefficients"]), law["huber_loss"]
            ).predict(weights.to_numpy(float))
            for law in laws
        },
        index=labels.coordinate_id,
    )[names]
    noise = pd.read_csv(preview.NOISE_MATRIX, index_col=0)
    noise.columns = [c.split("/")[2] for c in noise.columns]
    snr_table = pd.read_csv(preview.SNR_TABLE)
    snr = pd.Series(snr_table.snr.to_numpy(), index=snr_table.component.map(preview.short_name))

    rows = []
    for name in names:
        y, p = measured[name].to_numpy(), predicted[name].to_numpy()
        run_sd = float(noise[name].std(ddof=1))
        band = band_statistics(y, p, blocks, run_sd, counts, BAND)
        rows.append(
            {
                "component": name,
                "group": obe.group_of(name),
                "snr": float(snr[name]),
                "run_sd": run_sd,
                "spearman_all": float(stats.spearmanr(y, p).statistic),
                "bias_all_bpb": float((p - y).mean()),
                **{f"band30_{k}": v for k, v in band.items()},
            }
        )
    table = pd.DataFrame(rows).sort_values(["group", "band30_sign_accuracy"])
    table.to_csv(OUTPUT / "component_predictability.csv", index=False)

    group_rows = []
    for group in GROUPS:
        members = [n for n in names if obe.group_of(n) == group]
        y, p = measured[members].mean(axis=1).to_numpy(), predicted[members].mean(axis=1).to_numpy()
        po = olmix_predicted[members].mean(axis=1).to_numpy()
        kernel = top_band.kernel_loso(weights.to_numpy(float), y, blocks, KERNEL_BANDWIDTH)
        run_sd = float(noise[members].mean(axis=1).std(ddof=1))
        for size in BAND_SIZES:
            k = len(y) if size == 0 else size
            row = {"group": group, "components": len(members), "run_sd": run_sd, "band": k}
            row.update(band_statistics(y, p, blocks, run_sd, counts, k))
            row["olmix_sign_accuracy"] = band_statistics(y, po, blocks, run_sd, counts, k)["sign_accuracy"]
            row["kernel_sign_accuracy"] = band_statistics(y, kernel, blocks, run_sd, counts, k)["sign_accuracy"]
            group_rows.append(row)
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUTPUT / "group_band_ordering.csv", index=False)

    plt.rcParams.update(floor.original.PLOT_STYLE)
    plt.rcParams.update({"text.usetex": False})
    figure, axes = plt.subplots(1, len(GROUPS), figsize=(9.5, 2.1), gridspec_kw={"wspace": 0.3})
    for axis, group in zip(axes, GROUPS, strict=True):
        rows_g = groups[groups.group.eq(group)]
        x = rows_g.band.to_numpy(float)
        axis.plot(x, rows_g.noise_ceiling, color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        axis.plot(x, rows_g.kernel_sign_accuracy, color=floor.MUTED, linewidth=0.9, marker="o", markersize=2.2, zorder=3)
        axis.plot(x, rows_g.olmix_sign_accuracy, color=OLMIX_COLOR, linewidth=1.1, marker="s", markersize=2.4, zorder=3)
        axis.plot(
            x, rows_g.sign_accuracy, color=GROUP_COLORS[group], linewidth=1.4, marker="o", markersize=3.0, zorder=4
        )
        axis.axhline(0.5, color=floor.original.GRID, linewidth=0.7, zorder=1)
        axis.set_xscale("log")
        axis.set_xticks(list(x))
        axis.set_xticklabels([str(int(b)) if b < x.max() else "all" for b in x], fontsize=6.5)
        axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        axis.set_ylim(0, 1.02)
        axis.set_title(
            f"{group} ({rows_g.components.iloc[0]}), run SD {rows_g.run_sd.iloc[0]:.4f}",
            loc="left",
            fontsize=7.2,
            fontweight="bold",
        )
        axis.set_xlabel("best $k$ held-out optima", fontsize=7)
        axis.grid(True, axis="y", color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("Pairwise sign accuracy", fontsize=7)
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", marker="o", markersize=3, linewidth=1.4, label="MARINER (group colour)"),
            Line2D([], [], color=OLMIX_COLOR, marker="s", markersize=2.4, linewidth=1.1, label="Olmix"),
            Line2D([], [], color=floor.MUTED, marker="o", markersize=2.2, linewidth=0.9, label="Neighbour kernel"),
            Line2D([], [], color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Oracle under noise"),
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=5.6,
    )
    figure.savefig(OUTPUT / "group_band_ordering.png", dpi=200, bbox_inches="tight", pad_inches=0.02)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.3})
    for axis, column, label in zip(
        axes,
        ("band30_sign_accuracy", "band30_bias_bpb"),
        ("Sign accuracy, best 30 on the component", "Bias in the best 30 (predicted minus measured, BPB)"),
        strict=True,
    ):
        for group in GROUPS:
            rows_g = table[table.group.eq(group)]
            axis.scatter(
                rows_g.snr,
                rows_g[column],
                s=16,
                color=GROUP_COLORS[group],
                label=group,
                zorder=3,
                edgecolor="white",
                linewidths=0.4,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Component SNR (swarm std / repeat SD)", fontsize=7)
        axis.set_ylabel(label, fontsize=7)
        axis.tick_params(labelsize=6.5)
        axis.grid(True, color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].axhline(0.5, color="#888888", linewidth=0.7)
    axes[1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    figure.savefig(OUTPUT / "component_predictability.png", dpi=200, bbox_inches="tight", pad_inches=0.02)

    print(
        table[
            [
                "component",
                "group",
                "snr",
                "run_sd",
                "spearman_all",
                "band30_sign_accuracy",
                "band30_noise_ceiling",
                "band30_spearman",
                "band30_bias_bpb",
                "band30_band_width_bpb",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    columns = [
        "group",
        "components",
        "band",
        "sign_accuracy",
        "olmix_sign_accuracy",
        "kernel_sign_accuracy",
        "noise_ceiling",
    ]
    print(groups[groups.band.isin((30, 157))][columns].round(3).to_string(index=False))
    summary = table.groupby("group").agg(
        n=("component", "size"),
        snr_median=("snr", "median"),
        spearman_all=("spearman_all", "median"),
        band30_sign_accuracy=("band30_sign_accuracy", "median"),
        band30_noise_ceiling=("band30_noise_ceiling", "median"),
        band30_bias=("band30_bias_bpb", "median"),
    )
    print(summary.round(3).to_string())
    print(
        "Spearman(SNR, band30 sign accuracy) over components:",
        round(float(stats.spearmanr(table.snr, table.band30_sign_accuracy).statistic), 3),
    )


if __name__ == "__main__":
    main()
