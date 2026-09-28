# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Why the generic baselines overshoot: extrapolation along the segment to their proposals, and support coverage.

Two figures from the baseline proposals (`materialize_delphi_comparator_proposals_20260909.py`) and their
measured runs (`collect_delphi_3e18_validation_results_20260906.py --launch comparator_proposals`; the paper calls
the models baselines, the scripts keep "comparator" in their names):

1. ``paths``: each surrogate's predicted objective along the segment from MARINER's mixture to the baseline's
   proposal, with the measured runs at both ends and the heaviest bucket at each end relative to the swarm's heaviest
   exposure of that bucket. The smooth baselines keep predicting gains along the segment; MARINER's harm term turns
   it up.
2. ``support``: every proposal's per-bucket epochs against the swarm's exposure quantiles.

usage: uv run --offline --no-sync --with lightgbm python plot_comparator_proposal_diagnostics_20260909.py \
    [--drive-dir DIR] [--skip-paths]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

# One OpenMP thread: see materialize_delphi_comparator_proposals_20260909.py.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_comparator_proposals_20260909 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    summarize_delphi_comparator_proposals_20260909 as summarize,
)

PROPOSAL_DIR = proposals.OUTPUT
OUTPUT_DIR = PROPOSAL_DIR / "diagnostics"
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))
# Olmix's proposals come from its own log-linear laws (materialize_delphi_matched_olmix_20260908.py), not the harness.
OLMIX_DIR = proposals.SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_3e18_20260908"
OLMIX_POLICY = {"uncheatable": "olmixq_u_kl0p05_cap04", "table9": "olmixq_t9_kl0p005_cap04"}
OLMIX_MODEL = "olmix_laws"
# Baselines drawn in the path figure, in the order of the tables: every model with a trained proposal per objective.
PATH_COMPARATORS = (
    ("olmix", "Olmix", OLMIX_MODEL),
    ("quad", "Quadratic", registry.QUADRATIC_LINKED_ID),
    ("spline", "Cubic spline", registry.SPLINE_LINKED_ID),  # the caption gives the full name
    ("lgbm", "RegMix trees", "lightgbm_regmix"),
    ("krr", "Kernel ridge", registry.KRR_ID),
)
# Same colours as the learning-curve figures.
COLORS = {
    "mariner": "#469C76",
    "olmix": "#CC79A7",
    "quad": "#56B4E9",
    "spline": "#D55E00",
    "krr": "#E69F00",
    "lgbm": "#6C6F7D",
    "mk1": "#4C78A8",
}
LABELS = {
    "mariner": "MARINER",
    "olmix": "Olmix",
    "quad": "Quadratic in log-epochs, floor link",
    "spline": "Natural cubic spline in log-epochs, floor link",
    "krr": "Hellinger kernel ridge",
    "lgbm": "LightGBM (RegMix)",
    "mk1": "MARINER, exponential benefit",
}
SUPPORT_QUANTILE = 0.95
PATH_POINTS = 61
PATH_EXTENSION = 1.0
# The paper's figure style (plot_starcoder_matched_scaling_b5_preview_20260903.py); usetex off overrides ~/.matplotlib.
PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
}
DPI = 300
DRIVE_STEMS = {"paths": "a_comparator_paths", "support": "a_comparator_support"}


def candidate_key(candidate_id: str) -> str:
    return "olmix" if candidate_id.startswith("olmixq_") else candidate_id.split("_")[2]


def proposal_weights(buckets: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """Runtime weights of every proposal, keyed by target then comparator key; MARINER's from the reference package."""
    table = pd.read_csv(PROPOSAL_DIR / "solutions.csv")
    out: dict[str, dict[str, np.ndarray]] = {target: {} for target, _ in TARGETS}
    for candidate_id, rows in table.groupby("candidate_id", sort=False):
        target = "uncheatable" if candidate_id.startswith("cmp_u_") else "table9"
        weights = rows.set_index("bucket").loc[list(buckets), "runtime"].to_numpy(float)
        out[target][candidate_key(candidate_id)] = weights
    olmix = pd.read_csv(OLMIX_DIR / "solutions.csv")
    for target, _ in TARGETS:
        out[target]["mariner"] = proposals.reference_policy(proposals.MARINER_POLICY[target], buckets)
        rows = olmix[olmix.candidate_id == OLMIX_POLICY[target]].set_index("bucket")
        out[target]["olmix"] = rows.loc[list(buckets), "runtime"].to_numpy(float)
    return out


def measured_table() -> pd.DataFrame:
    """Measured runs of the comparator proposals and of the Olmix policies, with the comparator ``key``."""
    frames = []
    for path in (PROPOSAL_DIR / "measured_results.csv", OLMIX_DIR / "measured_results.csv"):
        table = pd.read_csv(path)
        table = table[table.status == "measured"].copy()
        table["measured"] = [
            row.uncheatable_bpb if row.target == "uncheatable" else row.table9_macro_bpb for row in table.itertuples()
        ]
        frames.append(table.dropna(subset=["measured"]))
    table = pd.concat(frames, ignore_index=True)
    keep = table.candidate_id.str.startswith("cmp_") | table.candidate_id.isin(OLMIX_POLICY.values())
    table = table[keep].copy()
    table["key"] = [candidate_key(candidate_id) for candidate_id in table.candidate_id]
    return table


Predictor = Callable[[np.ndarray], np.ndarray]


def olmix_predictor(panel: bench.BenchPanel, target: str) -> Predictor:
    """Olmix's objective from its saved per-component log-linear laws, in the panel's bucket order."""
    laws = json.loads((OLMIX_DIR / f"laws_{target}.json").read_text())
    group = panel.group(target)
    if [law["component"] for law in laws] != list(group.components):
        raise ValueError(f"Olmix laws for {target} do not match the panel's components")
    intercepts = np.asarray([np.exp(law["log_c"]) for law in laws])
    coefficients = np.asarray([law["coefficients"] for law in laws], float)
    aggregation = np.asarray(group.aggregation_weights, float)

    def predict(w: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(w)
        return (intercepts[None, :] + np.exp(rows @ coefficients.T)) @ aggregation

    return predict


def fit_surrogates(panel: bench.BenchPanel, swarm, target: str) -> dict[str, Predictor]:
    inner = bench.heldout_inner_folds(panel)
    surrogates: dict[str, Predictor] = {}
    for key, _label, model_id in PATH_COMPARATORS:
        if model_id == OLMIX_MODEL:
            surrogates[key] = olmix_predictor(panel, target)
            continue
        fits = [
            proposals.fit_component(panel, model_id, target, index, inner)
            for index in range(len(panel.group(target).components))
        ]
        surrogates[key] = proposals.ObservatorySurrogate(panel, model_id, target, fits).predict
    reference = proposals.reference_fit(swarm, target, None)
    to_panel = np.asarray([swarm.buckets.index(bucket) for bucket in panel.buckets])
    from_panel = np.argsort(to_panel)
    surrogates["mariner"] = lambda w, fit=reference: fit.predict(np.atleast_2d(w)[:, from_panel])
    return surrogates


def path_mixtures(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mixtures along the ray through ``start`` and ``end``, extended past ``end`` while every weight stays >= 0."""
    direction = end - start
    negative = direction < 0
    limit = PATH_EXTENSION
    if negative.any():
        limit = min(limit, float(np.min(-start[negative] / direction[negative])))
    positions = np.linspace(0.0, max(1.0, limit), PATH_POINTS)
    rows = start[None, :] + positions[:, None] * direction[None, :]
    rows = np.clip(rows, 0.0, None)
    rows /= rows.sum(axis=1, keepdims=True)
    return positions, rows


def support_boundary(exposures: np.ndarray) -> np.ndarray:
    return exposures.max(axis=0)


def draw_paths(weights, surrogates, measured, mariner_runs):
    fig, axes = plt.subplots(len(TARGETS), len(PATH_COMPARATORS), figsize=(7.2, 3.9), sharex=False)
    for row, (target, target_label) in enumerate(TARGETS):
        mariner_w = weights[target]["mariner"]
        base = mariner_runs[target].to_numpy(float)
        for column, (key, label, _model_id) in enumerate(PATH_COMPARATORS):
            ax = axes[row, column]
            end = weights[target][key]
            positions, rows = path_mixtures(mariner_w, end)
            plotted: list[float] = []
            for predictor, style in (("mariner", "-"), (key, "--")):
                values = surrogates[target][predictor](rows)
                plotted.extend(float(v) for v in values)
                ax.plot(
                    positions,
                    values,
                    style,
                    color=COLORS[predictor],
                    linewidth=1.3,
                    label=LABELS[predictor] if column == 0 else None,
                )
            plotted.extend(base)
            ax.plot(
                [0.0] * len(base), base, "o", color=COLORS["mariner"], markeredgecolor="black", markersize=4, zorder=5
            )
            own = measured[(measured.target == target) & (measured.key == key)]
            if len(own):
                plotted.extend(own.measured.to_numpy(float))
                ax.plot(
                    [1.0] * len(own),
                    own.measured.to_numpy(float),
                    "o",
                    color=COLORS[key],
                    markeredgecolor="black",
                    markersize=4,
                    zorder=5,
                )
            # Pad the range so the measured markers at either end are never clipped.
            low, high = min(plotted), max(plotted)
            ax.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
            ax.axvline(1.0, color="black", linewidth=0.5, linestyle=":")
            ax.set_title(
                f"{'ABCDEFGHIJ'[row * len(PATH_COMPARATORS) + column]}. {label}",
                loc="left",
                fontsize=8,
                fontweight="bold",
                color=INK,
            )
            ax.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["0%", "50%", "100%"])
            ax.tick_params(labelsize=6.5)
            if column == 0:
                ax.set_ylabel(f"{target_label}\nBPB", fontsize=8)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS["mariner"], linewidth=1.3, label="MARINER's prediction"),
        mpl.lines.Line2D([], [], color="black", linewidth=1.3, linestyle="--", label="baseline's own prediction"),
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="none", color="gray", markeredgecolor="black", label="measured run"
        ),
    ]
    fig.supxlabel("interpolation from MARINER's mixture (0%) to the baseline's proposal (100%)", fontsize=8, y=0.075)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def draw_support(panel, weights, inventory):
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(7.2, 4.6), sharex=True)
    exposures = panel.features.exposures
    low = np.quantile(exposures, 0.05, axis=0)
    high = np.quantile(exposures, SUPPORT_QUANTILE, axis=0)
    top = exposures.max(axis=0)
    for ax, (target, target_label) in zip(axes, TARGETS, strict=True):
        order = np.argsort(-weights[target]["mariner"] * inventory)
        x = np.arange(len(order))
        ax.fill_between(
            x, low[order], high[order], color="#DDDDDD", step="mid", linewidth=0, label="swarm 5th to 95th percentile"
        )
        ax.plot(x, top[order], "_", color="gray", markersize=6, label="swarm maximum")
        for key in ("mariner", "quad", "spline", "krr", "lgbm"):
            if key not in weights[target]:
                continue
            epochs = weights[target][key] * inventory
            ax.plot(
                x,
                epochs[order],
                "o" if key == "mariner" else "x",
                color=COLORS[key],
                markersize=4 if key == "mariner" else 4.5,
                markeredgecolor="black" if key == "mariner" else COLORS[key],
                linestyle="none",
                label=LABELS[key],
            )
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("materialized epochs")
        ax.set_title(f"{target_label}: proposals against the swarm's exposure range", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                panel.buckets[i].replace("dolma3_cc/", "cc/").replace("dolma3_", "").replace("dolmino_", "")
                for i in order
            ],
            rotation=90,
            fontsize=5.5,
        )
    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--skip-paths", action="store_true", help="skip the surrogate fits and the path figure")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)
    panel = bench.load_panel(proposals.PANEL)
    buckets = tuple(str(b) for b in panel.buckets)
    inventory = panel.features.inventory
    weights = proposal_weights(buckets)
    measured = measured_table()
    figures = {}
    figures["support"] = draw_support(panel, weights, inventory)
    if not args.skip_paths:
        swarm, _ = proposals.reference_swarm_in_panel_order(panel)
        surrogates = {target: fit_surrogates(panel, swarm, target) for target, _ in TARGETS}
        figures["paths"] = draw_paths(weights, surrogates, measured, summarize.mariner_runs())
        rows = []
        boundary = support_boundary(panel.features.exposures)
        for target, _ in TARGETS:
            for key, _label, _model_id in PATH_COMPARATORS:
                positions, mixtures = path_mixtures(weights[target]["mariner"], weights[target][key])
                epochs = mixtures * inventory[None, :]
                for predictor in ("mariner", key):
                    values = surrogates[target][predictor](mixtures)
                    for position, value, row_epochs in zip(positions, values, epochs, strict=True):
                        rows.append(
                            {
                                "target": target,
                                "comparator": key,
                                "predictor": predictor,
                                "position": float(position),
                                "prediction": float(value),
                                "max_epochs": float(row_epochs.max()),
                                "outside_support": bool((row_epochs > boundary).any()),
                            }
                        )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "path_predictions.csv", index=False)
    for name, fig in figures.items():
        for suffix in ("pdf", "png"):
            fig.savefig(OUTPUT_DIR / f"{name}.{suffix}", dpi=DPI)
        if args.drive_dir is not None:
            for suffix in ("pdf", "png"):
                shutil.copy(OUTPUT_DIR / f"{name}.{suffix}", args.drive_dir / f"{DRIVE_STEMS[name]}.{suffix}")
        print("wrote", OUTPUT_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
