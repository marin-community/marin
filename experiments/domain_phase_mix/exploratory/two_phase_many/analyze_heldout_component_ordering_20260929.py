# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Ordering of the k best held-out optima on single components of similar SNR, across the two objectives.

For each requested component, the held-out optima of its objective are scored on that component alone: measured value
per coordinate (mean over its runs; OlmoBaseEval Easy values from the cached W&B evaluation summaries, Uncheatable values
from the training runs' summaries, fetched once into `heldout_uncheatable_components.csv`), MARINER's per-task prediction
from the frozen reference fit at the coordinate's mixture, Olmix's matched per-task law, the leave-source-out neighbour
kernel, and the oracle under the component's repeat noise, with the pair threshold at one repeat SD; each predictor's
accuracy carries the source-block bootstrap interval of `analyze_delphi_top_band_ordering_20260906.py`.
Each row also carries the component's MARINER head (shape, ridge, kappa, inner-CV RMSE) and the swarm's measured range
against the held-out minimum, for the appendix table. `component_frame` is shared with
`plot_sciq_head_diagnosis_20260929.py`.

Writes `component_ordering.csv` and `component_ordering.png` to `reference_outputs/heldout_table9_components_20260929/`.

usage: (WANDB_API_KEY in the environment for the first run)
  uv run --offline --no-sync python analyze_heldout_component_ordering_20260929.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import dataclasses
import functools
import json
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
import analyze_heldout_component_predictability_20260929 as predictability  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import plot_analysis_floor_20260928 as floor  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402

OUTPUT = preview.OUTPUT
UNCHEATABLE_CACHE = OUTPUT / "heldout_uncheatable_components.csv"
STATUS = preview.REFERENCE / "single_phase_heldout_benchmark_20260902" / "input" / "wandb_component_status.csv"
LAWS = {
    t: preview.REFERENCE / "delphi_matched_olmix_3e18_20260908" / f"laws_{t}.json" for t in ("uncheatable", "table9")
}
UNCHEATABLE_KEY = "eval/uncheatable_eval/{}/bpb"
BAND_SIZES = (10, 20, 30, 50, 100, 0)
REQUESTS = (
    ("uncheatable", "arxiv_physics", "Uncheatable: arXiv physics"),
    ("table9", "hellaswag", "OlmoBaseEval Easy: HellaSwag"),
    ("table9", "sciq", "OlmoBaseEval Easy: SciQ"),
    ("table9", "arc_easy", "OlmoBaseEval Easy: ARC-Easy"),
    ("table9", "coqa", "OlmoBaseEval Easy: CoQA"),
    ("table9", "drop", "OlmoBaseEval Easy: DROP"),
    ("uncheatable", "arxiv_computer_science", "Uncheatable: arXiv CS"),
    ("table9", "socialiqa", "OlmoBaseEval Easy: SocialIQA"),
    ("table9", "piqa", "OlmoBaseEval Easy: PIQA"),
    ("table9", "minerva_math_geometry", "OlmoBaseEval Easy: Minerva geometry"),
    ("table9", "mmlu_other", "OlmoBaseEval Easy: MMLU Other"),
    ("table9", "mt_mbpp_typescript", "OlmoBaseEval Easy: MT-MBPP TypeScript"),
    ("table9", "mt_mbpp_csharp", "OlmoBaseEval Easy: MT-MBPP C#"),
)
COLORS = {"uncheatable": "#0072B2", "table9": "#D55E00"}


@dataclasses.dataclass(frozen=True)
class ComponentFrame:
    """One component's held-out optima: measured values, the three predictors, blocks and the fitted head."""

    target: str
    component: str
    coordinate_id: np.ndarray
    measured: np.ndarray
    mariner: np.ndarray
    olmix: np.ndarray
    kernel: np.ndarray
    blocks: np.ndarray
    counts: np.ndarray
    run_sd: float
    snr: float
    task: mat.ms.TaskFit
    swarm_measured: np.ndarray
    exposures: pd.DataFrame  # held-out optima, epochs per bucket
    swarm_exposures: pd.DataFrame  # swarm runs, epochs per bucket


@functools.cache
def swarm_inputs() -> tuple:
    panel = mat.bench.load_panel(mat.PANEL)
    swarm, to_panel = mat.reference_swarm_in_panel_order(panel)
    return panel, swarm, np.argsort(to_panel)


@functools.cache
def reference_fit(target: str) -> mat.ms.ObjectiveFit:
    _, swarm, _ = swarm_inputs()
    return mat.reference_fit(swarm, target, None)


def held_out_labels(target: str) -> pd.DataFrame:
    labels = pd.read_csv(preview.BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
    labels = labels[~labels.sources.str.contains("|".join(preview.INTERVENTIONS))]
    return labels.sort_values("measured_mean_bpb").reset_index(drop=True)


def uncheatable_components(labels: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per-component Uncheatable BPB of each held-out run from its training run's W&B summary (cached)."""
    heldout = pd.read_csv(preview.HELDOUT_RUNS, low_memory=False)
    runs = heldout[heldout.coordinate_id.isin(labels.coordinate_id)]
    status = pd.read_csv(STATUS)
    status = status[status.target.eq("uncheatable")].set_index("row_id").loc[runs.row_id]
    cached = pd.read_csv(UNCHEATABLE_CACHE) if UNCHEATABLE_CACHE.exists() else pd.DataFrame(columns=["row_id"])
    missing = runs[~runs.row_id.isin(cached.row_id)]
    if len(missing):
        api = wandb.Api(timeout=60)
        rows = []
        for index, row in enumerate(missing.itertuples(), start=1):
            project, run_id = status.loc[row.row_id, "project"], status.loc[row.row_id, "wandb_run_id"]
            summary = api.run(f"{project}/{run_id}").summary
            values = {name: summary.get(UNCHEATABLE_KEY.format(name)) for name in names}
            if any(v is None for v in values.values()):
                print(f"skipping {row.row_id}: missing components", flush=True)
                continue
            rows.append(
                {"row_id": row.row_id, "coordinate_id": row.coordinate_id, **{k: float(v) for k, v in values.items()}}
            )
            if index % 40 == 0:
                print(f"fetched {index}/{len(missing)}", flush=True)
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True) if len(cached) else pd.DataFrame(rows)
        cached.to_csv(UNCHEATABLE_CACHE, index=False)
    return cached[cached.row_id.isin(runs.row_id)]


def component_frame(target: str, component: str) -> ComponentFrame:
    panel, swarm, from_panel = swarm_inputs()
    labels = held_out_labels(target)
    blocks_table = pd.read_csv(preview.BENCHMARK / "source_blocks.csv")
    blocks = (
        blocks_table[blocks_table.target.eq(target)]
        .set_index("coordinate_id")
        .loc[labels.coordinate_id]
        .source_block.to_numpy()
    )
    counts = labels.run_count.to_numpy(float)
    full = mat.ms.read_objectives(mat.ms.DATA / "objectives.csv")[target]
    names = [preview.short_name(c) for c in full.components]
    full_name = next(c for c in full.components if preview.short_name(c) == component)
    if target == "table9":
        components = pd.read_csv(preview.CACHE)
        measured_all = components.groupby("coordinate_id")[names].mean().loc[labels.coordinate_id]
        noise = pd.read_csv(preview.NOISE_MATRIX, index_col=0)
        noise.columns = [c.split("/")[2] for c in noise.columns]
        run_sd = float(noise[component].std(ddof=1))
    else:
        components = uncheatable_components(labels, names)
        measured_all = components.groupby("coordinate_id")[names].mean().reindex(labels.coordinate_id)
        keep = measured_all[component].notna().to_numpy()
        labels, blocks, counts, measured_all = (
            labels[keep].reset_index(drop=True),
            blocks[keep],
            counts[keep],
            measured_all[keep],
        )
        anchors = mat.ms.read_anchors(mat.ms.DATA / "anchors.csv", target)
        run_sd = float(anchors[full_name].repeat_sd)
    features = np.load(preview.BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
    weights = pd.DataFrame(
        features["weights"],
        index=[str(c) for c in features["coordinate_id"]],
        columns=[str(b) for b in panel.buckets],
    ).loc[labels.coordinate_id]
    fit = reference_fit(target)
    predicted = pd.DataFrame(
        fit.predict_tasks(weights.to_numpy(float)[:, from_panel]), index=labels.coordinate_id, columns=names
    )[component].to_numpy()
    law = next(law for law in json.loads(LAWS[target].read_text()) if preview.short_name(law["component"]) == component)
    olmix_predicted = olmix.OlmixLoglinearFit(law["log_c"], tuple(law["coefficients"]), law["huber_loss"]).predict(
        weights.to_numpy(float)
    )
    y = measured_all[component].to_numpy()
    kernel = top_band.kernel_loso(weights.to_numpy(float), y, blocks, predictability.KERNEL_BANDWIDTH)
    buckets = list(swarm.buckets)
    inventory = pd.Series(swarm.inventory, index=buckets)
    return ComponentFrame(
        target=target,
        component=component,
        coordinate_id=labels.coordinate_id.to_numpy(),
        measured=y,
        mariner=predicted,
        olmix=olmix_predicted,
        kernel=kernel,
        blocks=blocks,
        counts=counts,
        run_sd=run_sd,
        snr=float(swarm.outcomes[full_name].std(ddof=1) / run_sd),
        task=next(t for t in fit.tasks if t.component == full_name),
        swarm_measured=swarm.outcomes[full_name].to_numpy(float),
        exposures=weights[buckets] * inventory,
        swarm_exposures=pd.DataFrame(swarm.weights, columns=buckets) * inventory,
    )


def ordering_rows(frame: ComponentFrame) -> list[dict]:
    head = {
        "threshold": frame.task.shape["threshold"],
        "rate": frame.task.shape["rate"],
        "power": frame.task.shape["power"],
        "ridge": frame.task.ridge,
        "kappa": frame.task.kappa,
        "inner_cv_rmse": frame.task.inner_cv_rmse,
        "swarm_min": float(frame.swarm_measured.min()),
        "swarm_max": float(frame.swarm_measured.max()),
        "heldout_min": float(frame.measured.min()),
    }
    y, blocks, counts, run_sd = frame.measured, frame.blocks, frame.counts, frame.run_sd
    rows = []
    for size in BAND_SIZES:
        k = len(y) if size == 0 else size
        row = {
            "target": frame.target,
            "component": frame.component,
            "snr": frame.snr,
            "run_sd": run_sd,
            "n": len(y),
            "band": k,
            **head,
        }
        row.update(predictability.band_statistics(y, frame.mariner, blocks, run_sd, counts, k))
        order = np.argsort(y, kind="stable")[:k]
        row["sign_accuracy_ci_low"], row["sign_accuracy_ci_high"] = top_band.bootstrap_accuracy(
            y[order], frame.mariner[order], blocks[order], run_sd, None
        )
        for label, prediction in (("olmix", frame.olmix), ("kernel", frame.kernel)):
            row[f"{label}_sign_accuracy"] = predictability.band_statistics(y, prediction, blocks, run_sd, counts, k)[
                "sign_accuracy"
            ]
            row[f"{label}_sign_accuracy_ci_low"], row[f"{label}_sign_accuracy_ci_high"] = top_band.bootstrap_accuracy(
                y[order], prediction[order], blocks[order], run_sd, None
            )
        rows.append(row)
    return rows


def draw(axis, table: pd.DataFrame, target: str, title: str) -> None:
    x = table.band.to_numpy(float)
    axis.plot(x, table.noise_ceiling, color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
    axis.plot(x, table.kernel_sign_accuracy, color=floor.MUTED, linewidth=0.9, marker="o", markersize=2.2, zorder=3)
    axis.plot(
        x,
        table.olmix_sign_accuracy,
        color=predictability.OLMIX_COLOR,
        linewidth=1.1,
        marker="s",
        markersize=2.4,
        zorder=3,
    )
    axis.plot(x, table.sign_accuracy, color=COLORS[target], linewidth=1.4, marker="o", markersize=3.0, zorder=4)
    axis.axhline(0.5, color=floor.original.GRID, linewidth=0.7, zorder=1)
    axis.set_xscale("log")
    axis.set_xticks(list(x))
    axis.set_xticklabels([str(int(b)) if b < x.max() else "all" for b in x], fontsize=7)
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_ylim(0, 1.02)
    snr, run_sd, n = float(table.snr.iloc[0]), float(table.run_sd.iloc[0]), int(table.n.iloc[0])
    axis.set_title(f"{title}\nSNR {snr:.1f}, run SD {run_sd:.4f}, n = {n}", loc="left", fontsize=7.5, fontweight="bold")
    axis.set_xlabel("best $k$ held-out optima", fontsize=7.5)
    axis.grid(True, axis="y", color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def main() -> None:
    rows = []
    plt.rcParams.update(floor.original.PLOT_STYLE)
    figure, axes = plt.subplots(1, len(REQUESTS), figsize=(2.5 * len(REQUESTS), 2.2), gridspec_kw={"wspace": 0.3})
    for axis, (target, component, title) in zip(axes, REQUESTS, strict=True):
        frame = component_frame(target, component)
        component_rows = ordering_rows(frame)
        rows.extend(component_rows)
        draw(axis, pd.DataFrame(component_rows), target, title)
    axes[0].set_ylabel("Pairwise sign accuracy", fontsize=7.5)
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", marker="o", markersize=3, linewidth=1.4, label="MARINER"),
            Line2D([], [], color=predictability.OLMIX_COLOR, marker="s", markersize=2.4, linewidth=1.1, label="Olmix"),
            Line2D([], [], color=floor.MUTED, marker="o", markersize=2.2, linewidth=0.9, label="Neighbour kernel"),
            Line2D([], [], color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Oracle under noise"),
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=6,
    )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "component_ordering.csv", index=False)
    figure.savefig(OUTPUT / "component_ordering.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(
        table[
            [
                "target",
                "component",
                "snr",
                "run_sd",
                "n",
                "band",
                "sign_accuracy",
                "olmix_sign_accuracy",
                "kernel_sign_accuracy",
                "noise_ceiling",
                "spearman",
                "bias_bpb",
                "band_width_bpb",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
