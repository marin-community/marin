# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Best-k ordering of the held-out optima: MARINER against its one-threshold-per-bucket ablation.

Both models are fitted on the full 280-run panel by the harness's heldout-stage protocol (`fit_component` of
`materialize_delphi_comparator_proposals_20260909.py`, the protocol behind Table 5's per-bucket-threshold row) and
scored on the held-out optima of each objective: pairwise sign accuracy among the k best-measured optima over pairs
whose measured difference exceeds the run standard deviation, with the source-block bootstrap interval and the oracle
under run noise of `analyze_delphi_top_band_ordering_20260906.py`.

Writes `pbt_band_ordering.csv` and `pbt_band_ordering.png` to `reference_outputs/delphi_top_band_ordering_20260928/`.

usage: uv run --offline --no-sync python analyze_heldout_band_ordering_pbt_20260929.py [--workers 8]
"""

import argparse
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import analyze_heldout_component_ordering_20260929 as ordering  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import plot_analysis_floor_20260928 as floor  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

OUTPUT = floor.REFERENCE / "delphi_top_band_ordering_20260928"
MODELS = {
    "MARINER": mat.MARINER_ID,
    "MARINER, one threshold per bucket": f"{mat.MARINER_ID}_per_bucket_threshold",
}
COLORS = {"MARINER": floor.original.MARINER_COLOR, "MARINER, one threshold per bucket": "#E69F00"}
TARGETS = (("uncheatable", "A. Uncheatable"), ("table9", "B. OlmoBaseEval Easy"))
BAND_SIZES = (10, 20, 30, 50, 100, 0)


def held_out_predictions(panel, target: str, model_id: str, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """The held-out optima's labels and the model's aggregate prediction at each, in the labels' order."""
    inner = mat.bench.heldout_inner_folds(panel)
    group = panel.group(target)
    fits = Parallel(n_jobs=workers)(
        delayed(mat.fit_component)(panel, model_id, target, index, inner) for index in range(len(group.components))
    )
    surrogate = mat.ObservatorySurrogate(panel, model_id, target, list(fits))
    labels = ordering.held_out_labels(target)
    features = np.load(preview.BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
    weights = pd.DataFrame(
        features["weights"],
        index=[str(c) for c in features["coordinate_id"]],
        columns=[str(b) for b in panel.buckets],
    ).loc[labels.coordinate_id]
    return labels, surrogate.predict(weights.to_numpy(float))


def band_rows(target: str, label: str, labels: pd.DataFrame, predicted: np.ndarray) -> list[dict]:
    blocks_table = pd.read_csv(preview.BENCHMARK / "source_blocks.csv")
    blocks = (
        blocks_table[blocks_table.target.eq(target)]
        .set_index("coordinate_id")
        .loc[labels.coordinate_id]
        .source_block.to_numpy()
    )
    measured = labels.measured_mean_bpb.to_numpy(float)
    run_sd = top_band.RUN_SD[target]
    sd_run = run_sd / np.sqrt(labels.run_count.to_numpy(float))
    rows = []
    for size in BAND_SIZES:
        k = len(measured) if size == 0 else size
        band = np.arange(k)
        pairs = np.array(list(itertools.combinations(band, 2)))
        sd_diff = np.sqrt(sd_run[pairs[:, 0]] ** 2 + sd_run[pairs[:, 1]] ** 2)
        scores = top_band.band_scores(measured[band], predicted[band], run_sd, sd_diff, blocks[band])
        low, high = top_band.bootstrap_accuracy(measured[band], predicted[band], blocks[band], run_sd, None)
        rows.append(
            {
                "target": target,
                "method": label,
                "band": k,
                **scores,
                "sign_accuracy_ci_low": low,
                "sign_accuracy_ci_high": high,
            }
        )
    return rows


def draw(axis, table: pd.DataFrame, target: str, title: str) -> None:
    for label, color in COLORS.items():
        rows = table[table.target.eq(target) & table.method.eq(label)].sort_values("band")
        x = rows.band.to_numpy(float)
        axis.fill_between(x, rows.sign_accuracy_ci_low, rows.sign_accuracy_ci_high, color=color, alpha=0.16, linewidth=0)
        axis.plot(x, rows.sign_accuracy, color=color, linewidth=1.4, marker="o", markersize=3.0, zorder=4)
    rows = table[table.target.eq(target) & table.method.eq("MARINER")].sort_values("band")
    x = rows.band.to_numpy(float)
    axis.plot(x, rows.noise_ceiling, color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
    axis.axhline(0.5, color=floor.original.GRID, linewidth=0.7, zorder=1)
    axis.set_xscale("log")
    axis.set_xticks(list(x))
    axis.set_xticklabels([str(int(b)) if b < x.max() else "all" for b in x], fontsize=7)
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_ylim(0, 1.02)
    axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
    axis.set_xlabel("best $k$ held-out optima", fontsize=7.5)
    axis.grid(True, axis="y", color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    panel = mat.bench.load_panel(mat.PANEL)
    rows = []
    for target, _ in TARGETS:
        for label, model_id in MODELS.items():
            labels, predicted = held_out_predictions(panel, target, model_id, args.workers)
            rows.extend(band_rows(target, label, labels, predicted))
            print(f"{target} {label}: fitted and scored", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "pbt_band_ordering.csv", index=False)
    plt.rcParams.update(floor.original.PLOT_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), gridspec_kw={"wspace": 0.3})
    for axis, (target, title) in zip(axes, TARGETS, strict=True):
        draw(axis, table, target, title)
    axes[0].set_ylabel("Pairwise sign accuracy", fontsize=7.5)
    axes[0].legend(
        handles=[
            Line2D([], [], color=color, marker="o", markersize=3, linewidth=1.4, label=label)
            for label, color in COLORS.items()
        ]
        + [Line2D([], [], color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Oracle under noise")],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=6,
    )
    figure.savefig(OUTPUT / "pbt_band_ordering.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(
        table[
            [
                "target",
                "method",
                "band",
                "sign_accuracy",
                "sign_accuracy_ci_low",
                "sign_accuracy_ci_high",
                "noise_ceiling",
                "spearman",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
