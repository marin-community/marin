# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Create DS-RE-CEQ diagnostics plots for the two-phase many-domain sweep."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_dsre_ceq import (
    OBJECTIVE_METRIC,
    _cross_validate,
    _fit_dsre,
    _load_spec,
    _sample_predicted_optimum,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
OOF_DIAGNOSTICS_CSV = SCRIPT_DIR / "dsre_ceq_oof_diagnostics.csv"
PARAMETER_TABLE_CSV = SCRIPT_DIR / "dsre_ceq_parameter_table.csv"
RESTART_STABILITY_CSV = SCRIPT_DIR / "dsre_ceq_restart_stability.csv"
SUMMARY_JSON = SCRIPT_DIR / "dsre_ceq_debug_summary.json"
SCATTER_PNG = SCRIPT_DIR / "dsre_ceq_oof_actual_vs_predicted.png"
TOPK_PNG = SCRIPT_DIR / "dsre_ceq_topk_best_actual.png"
RESIDUAL_PANELS_PNG = SCRIPT_DIR / "dsre_ceq_residual_panels.png"
PARAMETERS_PNG = SCRIPT_DIR / "dsre_ceq_parameter_heatmaps.png"
STABILITY_PNG = SCRIPT_DIR / "dsre_ceq_restart_stability.png"

TOP_FRONTIER_K = 20
RANDOM_BASELINE_DRAWS = 1024
STABILITY_SEEDS = tuple(range(8))
MAX_STABILITY_WORKERS = min(4, max(1, (os.cpu_count() or 1) - 2), len(STABILITY_SEEDS))


def _log(message: str) -> None:
    print(message, flush=True)


def _short_domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        return domain_name.removeprefix("dolma3_cc/")
    if domain_name.startswith("dolma3_"):
        return domain_name.removeprefix("dolma3_")
    if domain_name.startswith("dolmino_"):
        return domain_name.removeprefix("dolmino_")
    return domain_name


def _domain_sort_key(domain_name: str) -> tuple[int, str, int]:
    if domain_name.startswith("dolma3_cc/"):
        tail = domain_name.removeprefix("dolma3_cc/")
        if tail.endswith("_high"):
            return (0, tail.removesuffix("_high"), 0)
        if tail.endswith("_low"):
            return (0, tail.removesuffix("_low"), 1)
        return (0, tail, 2)
    if domain_name.startswith("dolma3_"):
        return (1, domain_name.removeprefix("dolma3_"), 0)
    if domain_name.startswith("dolmino_"):
        return (2, domain_name.removeprefix("dolmino_"), 0)
    return (3, domain_name, 0)


def _ordered_domain_names(domain_names: tuple[str, ...]) -> list[str]:
    return sorted(domain_names, key=_domain_sort_key)


def _group_boundaries(domain_names: list[str]) -> list[int]:
    boundaries: list[int] = []
    prev_group: tuple[int, str] | None = None
    for idx, domain_name in enumerate(domain_names):
        key = _domain_sort_key(domain_name)
        group = (key[0], key[1])
        if prev_group is not None and group != prev_group:
            boundaries.append(idx)
        prev_group = group
    return boundaries


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _unpack_dsre_parameters(final_p: np.ndarray, spec: DatasetSpec) -> dict[str, np.ndarray | float]:
    n_phases = spec.N
    n_domains = spec.M
    idx = 0

    c0 = float(final_p[idx])
    log_a = float(final_p[idx + 1])
    log_b = float(final_p[idx + 2])
    idx += 3

    logits = np.zeros(n_domains, dtype=float)
    if n_domains > 1:
        logits[: n_domains - 1] = final_p[idx : idx + (n_domains - 1)]
        idx += n_domains - 1
    ces_weights = _softmax_logits(logits)

    rho = float(np.clip(5.0 * np.tanh(final_p[idx]), -10.0, 0.99))
    idx += 1

    phase_importance = np.zeros((n_domains, n_phases), dtype=float)
    for domain_idx in range(n_domains):
        full = np.zeros(n_phases, dtype=float)
        if n_phases > 1:
            full[: n_phases - 1] = final_p[idx : idx + (n_phases - 1)]
            idx += n_phases - 1
        phase_importance[domain_idx] = _softmax_logits(full)

    interference = np.zeros((n_phases, n_domains), dtype=float)
    if n_phases > 1:
        raw = final_p[idx : idx + (n_phases - 1) * n_domains].reshape(n_phases - 1, n_domains)
        idx += (n_phases - 1) * n_domains
        interference[1:] = np.exp(np.clip(raw, -8.0, 8.0))

    satiety = _sigmoid(final_p[idx : idx + n_domains])
    idx += n_domains

    gates = _sigmoid(final_p[idx : idx + n_phases])
    idx += n_phases
    if n_phases > 0:
        gates[0] = 0.0

    tau = float(np.exp(np.clip(final_p[idx], -8.0, 8.0)))

    return {
        "c0": c0,
        "A": float(np.exp(np.clip(log_a, -10.0, 10.0))),
        "B": float(np.exp(np.clip(log_b, -10.0, 10.0))),
        "rho": rho,
        "tau": tau,
        "ces_weights": ces_weights,
        "phase_importance": phase_importance,
        "interference": interference,
        "gates": gates,
        "effective_interference": interference * gates[:, None],
        "satiety": satiety,
    }


def _coarse_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    def column_sum(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
        return frame[columns].sum(axis=1)

    p0_cc_high = [col for col in frame.columns if col.startswith("phase_0_dolma3_cc/") and col.endswith("_high")]
    p0_cc_low = [col for col in frame.columns if col.startswith("phase_0_dolma3_cc/") and col.endswith("_low")]
    p1_cc_high = [col for col in frame.columns if col.startswith("phase_1_dolma3_cc/") and col.endswith("_high")]
    p1_cc_low = [col for col in frame.columns if col.startswith("phase_1_dolma3_cc/") and col.endswith("_low")]
    p0_synth = [col for col in frame.columns if col.startswith("phase_0_dolmino_synth_")]
    p1_synth = [col for col in frame.columns if col.startswith("phase_1_dolmino_synth_")]
    p1_dolma_non_cc = [
        "phase_1_dolma3_arxiv",
        "phase_1_dolma3_finemath_3plus",
        "phase_1_dolma3_stack_edu",
        "phase_1_dolma3_wikipedia",
    ]

    p0_high = column_sum(p0_cc_high)
    p0_low = column_sum(p0_cc_low)
    p1_high = column_sum(p1_cc_high)
    p1_low = column_sum(p1_cc_low)

    return pd.DataFrame(
        {
            "run_name": frame["run_name"],
            "phase0_cc_high_mass": p0_high,
            "phase0_cc_low_mass": p0_low,
            "phase1_cc_high_mass": p1_high,
            "phase1_cc_low_mass": p1_low,
            "phase0_dolmino_synth_mass": column_sum(p0_synth),
            "phase1_dolmino_synth_mass": column_sum(p1_synth),
            "phase1_dolma3_noncc_mass": column_sum([col for col in p1_dolma_non_cc if col in frame.columns]),
            "total_cc_high_minus_low": (p0_high + p1_high) - (p0_low + p1_low),
        }
    )


def _oof_diagnostics_frame(frame: pd.DataFrame, spec: DatasetSpec, predictions: np.ndarray) -> pd.DataFrame:
    diagnostics = frame.copy()
    diagnostics["actual_bpb"] = spec.y
    diagnostics["predicted_bpb_oof"] = predictions
    diagnostics["residual_actual_minus_pred"] = diagnostics["actual_bpb"] - diagnostics["predicted_bpb_oof"]
    diagnostics["actual_rank"] = diagnostics["actual_bpb"].rank(method="first", ascending=True).astype(int)
    diagnostics["predicted_rank"] = diagnostics["predicted_bpb_oof"].rank(method="first", ascending=True).astype(int)
    coarse = _coarse_feature_frame(frame)
    return diagnostics.merge(coarse, on="run_name", how="left")


def _plot_oof_scatter(diagnostics: pd.DataFrame, *, best_observed: float) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    actual = diagnostics["actual_bpb"].to_numpy()
    predicted = diagnostics["predicted_bpb_oof"].to_numpy()

    norm = plt.Normalize(vmin=float(actual.min()), vmax=float(actual.max()))
    point_colors = cmap(norm(actual))

    actual_frontier = diagnostics["actual_rank"] <= TOP_FRONTIER_K
    predicted_frontier = diagnostics["predicted_rank"] <= TOP_FRONTIER_K

    fig, ax = plt.subplots(figsize=(8.6, 7.0), dpi=180)
    ax.scatter(actual, predicted, c=point_colors, s=40, edgecolors="none", alpha=0.78, zorder=2)

    both = diagnostics[actual_frontier & predicted_frontier]
    actual_only = diagnostics[actual_frontier & ~predicted_frontier]
    predicted_only = diagnostics[~actual_frontier & predicted_frontier]
    for subset, marker, label, color in (
        (both, "o", f"Top-{TOP_FRONTIER_K} actual and predicted", cmap(0.15)),
        (actual_only, "s", f"Top-{TOP_FRONTIER_K} actual only", cmap(0.55)),
        (predicted_only, "D", f"Top-{TOP_FRONTIER_K} predicted only", cmap(0.9)),
    ):
        if subset.empty:
            continue
        ax.scatter(
            subset["actual_bpb"],
            subset["predicted_bpb_oof"],
            s=82,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            zorder=3,
            label=label,
        )

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1.5, label="Ideal diagonal")

    best_row = diagnostics.sort_values("actual_bpb", ascending=True).iloc[0]
    worst_false_positive = diagnostics.sort_values("predicted_bpb_oof", ascending=True).iloc[0]
    for row, color, label in (
        (best_row, cmap(0.08), "Best observed"),
        (worst_false_positive, cmap(0.96), "Top predicted"),
    ):
        ax.annotate(
            f"{label}\n{row['run_name']}",
            xy=(row["actual_bpb"], row["predicted_bpb_oof"]),
            xytext=(10, 12 if label == "Best observed" else -30),
            textcoords="offset points",
            fontsize=9,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.88, "edgecolor": color},
        )

    spearman = float(spearmanr(actual, predicted).statistic)
    ax.set_title("DS-RE-CEQ OOF actual vs predicted BPB")
    ax.set_xlabel("Actual BPB")
    ax.set_ylabel("OOF predicted BPB")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(
        0.98,
        0.02,
        f"n = {len(diagnostics)}\nBest observed = {best_observed:.4f}\nSpearman = {spearman:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, bbox_inches="tight")
    plt.close(fig)


def _random_topk_envelope(
    y: np.ndarray, *, draws: int = RANDOM_BASELINE_DRAWS, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    traces = np.empty((draws, len(y)), dtype=float)
    for draw_idx in range(draws):
        permuted = rng.permutation(y)
        traces[draw_idx] = np.minimum.accumulate(permuted)
    return traces.mean(axis=0), np.quantile(traces, 0.1, axis=0), np.quantile(traces, 0.9, axis=0)


def _plot_topk_retrieval(diagnostics: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    ranked = diagnostics.sort_values("predicted_bpb_oof", ascending=True, ignore_index=True)
    best_true_prefix = np.minimum.accumulate(ranked["actual_bpb"].to_numpy())
    k_values = np.arange(1, len(ranked) + 1)
    random_mean, random_lo, random_hi = _random_topk_envelope(diagnostics["actual_bpb"].to_numpy())
    oracle = np.full_like(best_true_prefix, diagnostics["actual_bpb"].min())

    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=180)
    ax.plot(k_values, best_true_prefix, color=cmap(0.12), linewidth=2.6, label="DS-RE-CEQ OOF ranking")
    ax.fill_between(k_values, random_lo, random_hi, color=cmap(0.82), alpha=0.18, label="Random 10-90% band")
    ax.plot(k_values, random_mean, color=cmap(0.82), linewidth=1.8, linestyle="--", label="Random mean")
    ax.plot(k_values, oracle, color="#444444", linewidth=1.6, linestyle=":", label="Oracle")

    ax.set_title("Best actual BPB among top-k DS-RE-CEQ predictions")
    ax.set_xlabel("Top-k predicted runs inspected")
    ax.set_ylabel("Best actual BPB seen so far")
    ax.set_xlim(1, len(ranked))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.text(
        0.98,
        0.98,
        (
            f"Top-1 actual at DS-RE-CEQ top-1 = {best_true_prefix[0]:.4f}\n"
            f"Top-5 best actual = {best_true_prefix[4]:.4f}\n"
            f"Top-10 best actual = {best_true_prefix[9]:.4f}\n"
            f"Global best = {oracle[0]:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(TOPK_PNG, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_panels(diagnostics: pd.DataFrame) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    features = (
        ("phase0_cc_high_mass", "Phase 0 CC high mass"),
        ("phase0_cc_low_mass", "Phase 0 CC low mass"),
        ("phase1_cc_high_mass", "Phase 1 CC high mass"),
        ("phase1_cc_low_mass", "Phase 1 CC low mass"),
        ("phase1_dolmino_synth_mass", "Phase 1 Dolmino synth mass"),
        ("total_cc_high_minus_low", "Total CC high-minus-low"),
    )

    actual = diagnostics["actual_bpb"].to_numpy()
    residual = diagnostics["residual_actual_minus_pred"].to_numpy()
    norm = plt.Normalize(vmin=float(actual.min()), vmax=float(actual.max()))
    point_colors = cmap(norm(actual))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=180, sharey=True)
    for ax, (feature_name, label) in zip(axes.flat, features, strict=True):
        x = diagnostics[feature_name].to_numpy()
        ax.scatter(x, residual, c=point_colors, s=32, edgecolors="none", alpha=0.82)
        if np.unique(x).size > 1:
            slope, intercept = np.polyfit(x, residual, deg=1)
            x_line = np.linspace(float(x.min()), float(x.max()), 64)
            ax.plot(x_line, intercept + slope * x_line, color="#333333", linewidth=1.5, linestyle="--")
            rho = float(spearmanr(x, residual).statistic)
        else:
            rho = float("nan")
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
        ax.set_title(f"{label}\nSpearman = {rho:.3f}")
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.18)

    axes[0, 0].set_ylabel("Residual (actual - predicted)")
    axes[1, 0].set_ylabel("Residual (actual - predicted)")
    fig.suptitle("DS-RE-CEQ OOF residuals against coarse schedule aggregates", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESIDUAL_PANELS_PNG, bbox_inches="tight")
    plt.close(fig)


def _parameter_table(spec: DatasetSpec, params: dict[str, np.ndarray | float]) -> pd.DataFrame:
    rows = []
    effective_interference = np.asarray(params["effective_interference"], dtype=float)
    phase_importance = np.asarray(params["phase_importance"], dtype=float)
    ces_weights = np.asarray(params["ces_weights"], dtype=float)
    satiety = np.asarray(params["satiety"], dtype=float)

    for domain_idx, domain_name in enumerate(spec.domain_names):
        phase1_pi = phase_importance[domain_idx, 1] if spec.N > 1 else phase_importance[domain_idx, 0]
        effective = effective_interference[1, domain_idx] if spec.N > 1 else effective_interference[0, domain_idx]
        rows.append(
            {
                "domain": domain_name,
                "label": _short_domain_label(domain_name),
                "ces_weight": float(ces_weights[domain_idx]),
                "phase1_pi": float(phase1_pi),
                "effective_interference_phase1": float(effective),
                "satiety_phi": float(satiety[domain_idx]),
            }
        )
    return pd.DataFrame(rows)


def _plot_parameter_heatmaps(spec: DatasetSpec, params: dict[str, np.ndarray | float]) -> None:
    table = _parameter_table(spec, params)
    ordered_domains = _ordered_domain_names(spec.domain_names)
    ordered = table.set_index("domain").loc[ordered_domains].reset_index()
    ordered.to_csv(PARAMETER_TABLE_CSV, index=False)

    display_labels = ordered["label"].tolist()
    boundaries = _group_boundaries(ordered_domains)

    columns = (
        ("ces_weight", "CES weight $a_d$"),
        ("phase1_pi", "Phase-1 importance $\\pi_{1,d}$"),
        ("effective_interference_phase1", "Effective interference $g_1\\lambda_{1,d}$"),
        ("satiety_phi", "Satiety $\\phi_d$"),
    )

    cmap = plt.colormaps["RdYlGn_r"]
    fig, axes = plt.subplots(1, len(columns), figsize=(11.5, 12.0), dpi=180, sharey=True)

    for ax, (column, title) in zip(axes, columns, strict=True):
        values = ordered[column].to_numpy()[:, None]
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(display_labels)))
            ax.set_yticklabels(display_labels, fontsize=8)
        else:
            ax.set_yticks(np.arange(len(display_labels)))
            ax.set_yticklabels([])
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color="white", linewidth=1.0, alpha=0.9)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        (
            "DS-RE-CEQ full-fit parameter heatmaps\n"
            f"rho = {float(params['rho']):.3f}, tau = {float(params['tau']):.3f}, "
            f"gate_1 = {float(np.asarray(params['gates'])[1] if spec.N > 1 else np.asarray(params['gates'])[0]):.3f}"
        ),
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PARAMETERS_PNG, bbox_inches="tight")
    plt.close(fig)


def _phase_weight_matrix(phase_weights: dict[str, dict[str, float]], spec: DatasetSpec) -> np.ndarray:
    matrix = np.zeros((spec.N, spec.M), dtype=float)
    for phase_idx, phase_name in enumerate(spec.phase_names):
        for domain_idx, domain_name in enumerate(spec.domain_names):
            matrix[phase_idx, domain_idx] = phase_weights[phase_name][domain_name]
    return matrix


def _mean_phase_tv(left: np.ndarray, right: np.ndarray) -> float:
    per_phase = 0.5 * np.abs(left - right).sum(axis=1)
    return float(np.mean(per_phase))


def _fit_seed(payload: tuple[DatasetSpec, tuple[str, ...], int]) -> dict[str, object]:
    spec, run_names, seed = payload
    fit = _fit_dsre(spec, seed=seed)
    optimum = _sample_predicted_optimum(fit.predict_fn, spec, seed=1000 + seed)
    observed_predictions = np.asarray(fit.predict_fn(spec.weights), dtype=float)
    chosen_idx = int(np.argmin(observed_predictions))
    unpacked = _unpack_dsre_parameters(np.asarray(fit.info["final_p"], dtype=float), spec)
    optimum_matrix = _phase_weight_matrix(optimum.phase_weights, spec)
    return {
        "seed": seed,
        "predicted_optimum_bpb": float(optimum.predicted_objective),
        "observed_regret_at_1": float(spec.y[chosen_idx] - np.min(spec.y)),
        "chosen_run_name": run_names[chosen_idx] if chosen_idx < len(run_names) else str(chosen_idx),
        "optimum_matrix": optimum_matrix,
        "ces_weights": np.asarray(unpacked["ces_weights"], dtype=float),
        "phase1_pi": (
            np.asarray(unpacked["phase_importance"], dtype=float)[:, 1]
            if spec.N > 1
            else np.asarray(unpacked["phase_importance"], dtype=float)[:, 0]
        ),
        "effective_interference_phase1": (
            np.asarray(unpacked["effective_interference"], dtype=float)[1]
            if spec.N > 1
            else np.asarray(unpacked["effective_interference"], dtype=float)[0]
        ),
        "satiety_phi": np.asarray(unpacked["satiety"], dtype=float),
    }


def _restart_stability(spec: DatasetSpec, run_names: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    parameter_families: dict[str, list[np.ndarray]] = {
        "ces_weights": [],
        "phase1_pi": [],
        "effective_interference_phase1": [],
        "satiety_phi": [],
        "optimum_matrix": [],
    }

    payloads = [(spec, run_names, seed) for seed in STABILITY_SEEDS]
    with ProcessPoolExecutor(max_workers=MAX_STABILITY_WORKERS) as executor:
        futures = [executor.submit(_fit_seed, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            for family in parameter_families:
                parameter_families[family].append(np.asarray(result.pop(family)))
            rows.append(result)
            _log(f"Finished DS-RE-CEQ stability seed {rows[-1]['seed']}")

    stability = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    parameter_arrays = {family: np.stack(values, axis=0) for family, values in parameter_families.items()}
    stability.to_csv(RESTART_STABILITY_CSV, index=False)
    return stability, parameter_arrays


def _plot_restart_stability(stability: pd.DataFrame, parameter_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    cmap = plt.colormaps["RdYlGn_r"]
    optimum_matrices = parameter_arrays["optimum_matrix"]
    n_runs = optimum_matrices.shape[0]
    tv_matrix = np.zeros((n_runs, n_runs), dtype=float)
    for i in range(n_runs):
        for j in range(n_runs):
            tv_matrix[i, j] = _mean_phase_tv(optimum_matrices[i], optimum_matrices[j])

    family_spreads = {
        "CES weights": np.std(parameter_arrays["ces_weights"], axis=0),
        "Phase-1 $\\pi$": np.std(parameter_arrays["phase1_pi"], axis=0),
        "Eff. interference": np.std(parameter_arrays["effective_interference_phase1"], axis=0),
        "Satiety $\\phi$": np.std(parameter_arrays["satiety_phi"], axis=0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)

    axes[0, 0].hist(stability["predicted_optimum_bpb"], bins=min(8, len(stability)), color=cmap(0.18), edgecolor="white")
    axes[0, 0].set_title("Predicted optimum BPB across seeds")
    axes[0, 0].set_xlabel("Predicted optimum BPB")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(stability["observed_regret_at_1"], bins=min(8, len(stability)), color=cmap(0.82), edgecolor="white")
    axes[0, 1].set_title("Observed Regret@1 across seeds")
    axes[0, 1].set_xlabel("Observed Regret@1 on 241-run pool")
    axes[0, 1].set_ylabel("Count")

    im = axes[1, 0].imshow(tv_matrix, cmap=cmap, vmin=0.0, vmax=max(0.25, float(tv_matrix.max())))
    axes[1, 0].set_title("Pairwise TV distance between predicted optima")
    axes[1, 0].set_xlabel("Seed index")
    axes[1, 0].set_ylabel("Seed index")
    axes[1, 0].set_xticks(np.arange(n_runs))
    axes[1, 0].set_yticks(np.arange(n_runs))
    axes[1, 0].set_xticklabels(stability["seed"].tolist(), fontsize=8)
    axes[1, 0].set_yticklabels(stability["seed"].tolist(), fontsize=8)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    boxplot = axes[1, 1].boxplot(
        list(family_spreads.values()),
        tick_labels=list(family_spreads.keys()),
        patch_artist=True,
    )
    for patch, color_position in zip(boxplot["boxes"], np.linspace(0.15, 0.85, len(family_spreads)), strict=True):
        patch.set_facecolor(cmap(color_position))
    axes[1, 1].set_title("Across-seed std by parameter family")
    axes[1, 1].set_ylabel("Std across seeds")
    axes[1, 1].tick_params(axis="x", rotation=12)

    fig.suptitle("DS-RE-CEQ restart stability", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(STABILITY_PNG, bbox_inches="tight")
    plt.close(fig)

    off_diag = tv_matrix[~np.eye(n_runs, dtype=bool)]
    return {
        "predicted_optimum_bpb_mean": float(stability["predicted_optimum_bpb"].mean()),
        "predicted_optimum_bpb_std": float(stability["predicted_optimum_bpb"].std(ddof=0)),
        "observed_regret_at_1_mean": float(stability["observed_regret_at_1"].mean()),
        "observed_regret_at_1_std": float(stability["observed_regret_at_1"].std(ddof=0)),
        "pairwise_tv_mean": float(off_diag.mean()) if len(off_diag) else 0.0,
        "pairwise_tv_max": float(off_diag.max()) if len(off_diag) else 0.0,
    }


def main() -> None:
    start = perf_counter()
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    _log("Loading two_phase_many dataset and dataset spec")
    frame, spec, _loop = _load_spec()
    diagnostics_start = perf_counter()
    metrics, oof_predictions = _cross_validate(spec, fit_kind="dsre")
    diagnostics = _oof_diagnostics_frame(frame, spec, oof_predictions)
    diagnostics.to_csv(OOF_DIAGNOSTICS_CSV, index=False)
    _log(f"Wrote OOF diagnostics to {OOF_DIAGNOSTICS_CSV}")

    _log("Fitting full-data DS-RE-CEQ for parameter inspection")
    full_fit = _fit_dsre(spec)
    unpacked = _unpack_dsre_parameters(np.asarray(full_fit.info["final_p"], dtype=float), spec)

    _plot_oof_scatter(diagnostics, best_observed=float(spec.y.min()))
    _plot_topk_retrieval(diagnostics)
    _plot_residual_panels(diagnostics)
    _plot_parameter_heatmaps(spec, unpacked)
    _log("Finished OOF scatter, top-k retrieval, residual panels, and parameter heatmaps")

    _log("Running restart-stability sweep")
    stability, parameter_arrays = _restart_stability(spec, tuple(frame["run_name"].tolist()))
    stability_summary = _plot_restart_stability(stability, parameter_arrays)
    _log("Finished restart-stability plots")

    topk_prefix = np.minimum.accumulate(
        diagnostics.sort_values("predicted_bpb_oof", ascending=True)["actual_bpb"].to_numpy()
    )
    summary = {
        "objective_metric": OBJECTIVE_METRIC,
        "n_runs": int(spec.R),
        "oof_metrics": {
            "r2": metrics.r2,
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "regret_at_1": metrics.regret_at_1,
            "n_params": metrics.n_params,
        },
        "frontier_retrieval": {
            "top1_best_actual": float(topk_prefix[0]),
            "top5_best_actual": float(topk_prefix[4]),
            "top10_best_actual": float(topk_prefix[9]),
            "top20_best_actual": float(topk_prefix[19]),
            "global_best_actual": float(spec.y.min()),
        },
        "full_fit": {
            "rho": float(unpacked["rho"]),
            "tau": float(unpacked["tau"]),
            "gate_phase1": float(np.asarray(unpacked["gates"], dtype=float)[1] if spec.N > 1 else 0.0),
        },
        "restart_stability": stability_summary,
        "artifacts": {
            "oof_diagnostics_csv": str(OOF_DIAGNOSTICS_CSV),
            "parameter_table_csv": str(PARAMETER_TABLE_CSV),
            "restart_stability_csv": str(RESTART_STABILITY_CSV),
            "scatter_png": str(SCATTER_PNG),
            "topk_png": str(TOPK_PNG),
            "residual_panels_png": str(RESIDUAL_PANELS_PNG),
            "parameter_heatmaps_png": str(PARAMETERS_PNG),
            "restart_stability_png": str(STABILITY_PNG),
        },
        "durations": {
            "diagnostics_seconds": float(perf_counter() - diagnostics_start),
            "total_seconds": float(perf_counter() - start),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _log(f"Wrote summary to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
