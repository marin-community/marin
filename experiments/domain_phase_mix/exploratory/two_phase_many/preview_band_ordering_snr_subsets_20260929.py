# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Preview: Figure 13B (ordering of the best k held-out optima) with OlmoBaseEval Easy cut to its higher-SNR components.

The held-out optima's per-component OlmoBaseEval Easy values come from their W&B evaluation runs (fetched once into
`reference_outputs/heldout_table9_components_20260929/heldout_table9_components.csv`); MARINER's per-component
predictions come from the frozen reference fit evaluated at each coordinate's mixture, so the subset objectives need
no refit. For each objective (all 51 components, SNR >= 5, SNR >= 8) the k best-measured coordinates form the band and
MARINER is scored by pairwise sign accuracy as in `analyze_delphi_top_band_ordering_20260906.py`, with the noise
reference and the leave-source-out neighbour kernel; the pair threshold and noise use the objective's proportional-repeat
SD from the ten 3e18 repeats.

usage: (WANDB_API_KEY in the environment for the first run)
  uv run --offline --no-sync python preview_band_ordering_snr_subsets_20260929.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import itertools
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import wandb

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import plot_analysis_floor_20260928 as floor  # noqa: E402

REFERENCE = SCRIPT_DIR / "reference_outputs"
BENCHMARK = REFERENCE / "delphi_offline_selection_20260906"
HELDOUT_RUNS = REFERENCE / "single_phase_heldout_benchmark_20260902" / "heldout_runs.csv"
NOISE_MATRIX = REFERENCE / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
SNR_TABLE = REFERENCE / "table9_snr_subset_optima_20260928" / "component_predictions.csv"
OUTPUT = REFERENCE / "heldout_table9_components_20260929"
CACHE = OUTPUT / "heldout_table9_components.csv"
PROJECT = "marin-community/marin-eval"
KEY_PREFIX = "olmo_base_easy/table9/"
INTERVENTIONS = top_band.INTERVENTIONS
BAND_SIZES = (10, 20, 30, 50, 100, 0)
OBJECTIVES = (("all 51 components", 0.0), ("SNR $\\geq$ 5 (32)", 5.0), ("SNR $\\geq$ 8 (17)", 8.0))
KERNEL_BANDWIDTH = 0.05


def short_name(component: str) -> str:
    return component.split("/")[2] if "/" in component else component


def fetch_components(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-component BPB of each held-out evaluation run from its W&B summary (cached on disk)."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["row_id"])
    missing = runs[~runs.row_id.isin(cached.row_id)]
    if len(missing):
        api = wandb.Api(timeout=60)
        rows = []
        for index, row in enumerate(missing.itertuples(), start=1):
            summary = api.run(f"{PROJECT}/{row.table9_eval_run_id}").summary
            values = {
                k[len(KEY_PREFIX) : -len("/bpb")]: float(v)
                for k, v in summary.items()
                if k.startswith(KEY_PREFIX) and k.endswith("/bpb")
            }
            if len(values) != 51:
                raise ValueError(f"{row.row_id}: {len(values)} components in {row.table9_eval_run_id}")
            rows.append(
                {
                    "row_id": row.row_id,
                    "coordinate_id": row.coordinate_id,
                    "eval_run_id": row.table9_eval_run_id,
                    **values,
                }
            )
            if index % 20 == 0:
                print(f"fetched {index}/{len(missing)}", flush=True)
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True) if len(cached) else pd.DataFrame(rows)
        cached.to_csv(CACHE, index=False)
    return cached[cached.row_id.isin(runs.row_id)]


def main() -> None:
    labels = pd.read_csv(BENCHMARK / "inputs" / "table9_bank_labels.csv")
    labels = (
        labels[~labels.sources.str.contains("|".join(INTERVENTIONS))]
        .sort_values("measured_mean_bpb")
        .reset_index(drop=True)
    )
    blocks = pd.read_csv(BENCHMARK / "source_blocks.csv")
    blocks = (
        blocks[blocks.target.eq("table9")].set_index("coordinate_id").loc[labels.coordinate_id].source_block.to_numpy()
    )
    heldout = pd.read_csv(HELDOUT_RUNS, low_memory=False)
    runs = heldout[heldout.coordinate_id.isin(labels.coordinate_id) & heldout.table9_eval_run_id.notna()]
    components = fetch_components(runs)
    names = [c for c in components.columns if c not in ("row_id", "coordinate_id", "eval_run_id")]
    measured = components.groupby("coordinate_id")[names].mean().loc[labels.coordinate_id]
    check = measured.mean(axis=1).to_numpy() - labels.measured_mean_bpb.to_numpy()
    print(f"51-component mean vs bank label: max abs diff {np.abs(check).max():.2e}")
    # Predictions: the frozen reference fit's per-task heads at each coordinate's mixture.
    panel = mat.bench.load_panel(mat.PANEL)
    swarm, to_panel = mat.reference_swarm_in_panel_order(panel)
    features = np.load(BENCHMARK / "inputs" / "table9_bank_features.npz", allow_pickle=True)
    weights = pd.DataFrame(
        features["weights"], index=[str(c) for c in features["coordinate_id"]], columns=[str(b) for b in panel.buckets]
    ).loc[labels.coordinate_id]
    from_panel = np.argsort(to_panel)
    fit = mat.reference_fit(swarm, "table9", None)
    task_predictions = fit.predict_tasks(weights.to_numpy(float)[:, from_panel])
    full = mat.ms.read_objectives(mat.ms.DATA / "objectives.csv")["table9"]
    predicted = pd.DataFrame(
        task_predictions, index=labels.coordinate_id, columns=[short_name(c) for c in full.components]
    )[names]
    frozen = pd.read_csv(REFERENCE / "delphi_kappa_floor_flat15_selection_20260907" / "predictions.csv")
    frozen = (
        frozen[
            frozen.population.eq("external_development")
            & frozen.method.eq("weibull_softplus_unscaled@kappa_floor_link_flat15")
            & frozen.target.eq("table9")
        ]
        .set_index("row_id")
        .loc[labels.coordinate_id]
        .prediction.to_numpy()
    )
    frozen_gap = np.abs(predicted.mean(axis=1).to_numpy() - frozen).max()
    print(f"predicted 51-mean vs frozen predictions: max abs diff {frozen_gap:.2e}")
    snr = pd.read_csv(SNR_TABLE)
    snr = pd.Series(snr.snr.to_numpy(), index=snr.component.map(short_name))
    noise = pd.read_csv(NOISE_MATRIX, index_col=0)
    noise.columns = [c.split("/")[2] for c in noise.columns]
    run_count = labels.run_count.to_numpy(float)

    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.3), gridspec_kw={"wspace": 0.35})
    plt.rcParams.update(floor.original.PLOT_STYLE)
    for axis, (title, threshold) in zip(axes, OBJECTIVES, strict=True):
        kept = [n for n in names if snr[n] >= threshold]
        y = measured[kept].mean(axis=1).to_numpy()
        p = predicted[kept].mean(axis=1).to_numpy()
        run_sd = float(noise[kept].mean(axis=1).std(ddof=1))
        order = np.argsort(y, kind="stable")
        y, p, blk, counts = y[order], p[order], blocks[order], run_count[order]
        kernel = top_band.kernel_loso(weights.to_numpy(float)[order], y, blk, KERNEL_BANDWIDTH)
        sd_run = run_sd / np.sqrt(counts)
        sd_diff = np.sqrt(sd_run[:, None] ** 2 + sd_run[None, :] ** 2)
        for size in BAND_SIZES:
            k = len(y) if size == 0 else size
            band = np.arange(k)
            pairs = np.array(list(itertools.combinations(band, 2)))
            scores = top_band.band_scores(y[band], p[band], run_sd, sd_diff[pairs[:, 0], pairs[:, 1]], blk[band])
            kscores = top_band.band_scores(y[band], kernel[band], run_sd, sd_diff[pairs[:, 0], pairs[:, 1]], blk[band])
            rows.append(
                {
                    "objective": title,
                    "components": len(kept),
                    "run_sd": run_sd,
                    "band": k,
                    "band_width_bpb": float(y[band].max() - y[band].min()),
                    "sign_accuracy": scores["sign_accuracy"],
                    "kernel_sign_accuracy": kscores["sign_accuracy"],
                    "noise_ceiling": scores["noise_ceiling"],
                    "pairs": scores["pairs"],
                }
            )
        table = pd.DataFrame([r for r in rows if r["objective"] == title])
        x = table.band.to_numpy(float)
        axis.plot(x, table.noise_ceiling, color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        axis.plot(x, table.kernel_sign_accuracy, color=floor.MUTED, linewidth=0.9, marker="o", markersize=2.2, zorder=3)
        axis.plot(
            x,
            table.sign_accuracy,
            color=floor.original.MARINER_COLOR,
            linewidth=1.4,
            marker="o",
            markersize=3.0,
            zorder=4,
        )
        axis.axhline(0.5, color=floor.original.GRID, linewidth=0.7, zorder=1)
        axis.set_xscale("log")
        axis.set_xticks(list(x))
        axis.set_xticklabels([str(int(b)) if b < x.max() else "all" for b in x], fontsize=7)
        axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        axis.set_ylim(0, 1.02)
        axis.set_title(f"OlmoBaseEval Easy, {title}\nrun SD {run_sd:.4f}", loc="left", fontsize=8, fontweight="bold")
        axis.set_xlabel("best $k$ held-out optima", fontsize=7.5)
        axis.grid(True, axis="y", color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("Pairwise sign accuracy", fontsize=7.5)
    handles = [
        Line2D([], [], color=floor.original.MARINER_COLOR, marker="o", markersize=3, linewidth=1.4, label="MARINER"),
        Line2D([], [], color=floor.MUTED, marker="o", markersize=2.2, linewidth=0.9, label="Neighbour kernel"),
        Line2D([], [], color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Oracle under noise"),
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=6)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "band_ordering_snr_subsets.csv", index=False)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"band_ordering_snr_subsets.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
