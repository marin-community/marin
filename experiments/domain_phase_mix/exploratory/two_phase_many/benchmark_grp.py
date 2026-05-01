# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Benchmark GRP and Olmix loglinear on the two-phase many-domain swarm."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many.convergence_plot_style import (
    BEST_OBSERVED_BPB_COLOR,
    PREDICTED_LINESTYLE,
    model_bpb_color,
)
from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_dsre_ceq import _fit_olmix_loglinear
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "two_phase_many.csv"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
SUBSET_SIZES = tuple(range(20, 241, 20))
FEATURE_POLICY = "feature_bayes_linear_observed"
GRP_MODEL_NAME = "GRP"
OLMIX_MODEL_NAME = "Olmix loglinear"
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_grp_vs_olmix_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_summary.json"
GRP_PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_convergence_tracks.png"
OLMIX_PLOT_PATH = SCRIPT_DIR / "two_phase_many_olmix_convergence_tracks.png"

_SCRIPT_START = perf_counter()


@dataclass(frozen=True)
class PredictedOptimum:
    predicted_objective: float
    phase_weights: dict[str, dict[str, float]]


def _log(message: str) -> None:
    elapsed = perf_counter() - _SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def _load_spec() -> tuple[pd.DataFrame, DatasetSpec, GenericFamilyPacket]:
    frame, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_analysis",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    return frame, spec, packet


def _subset_packet(packet: GenericFamilyPacket, indices: np.ndarray) -> GenericFamilyPacket:
    indices = np.asarray(indices, dtype=int)
    return replace(
        packet,
        base=replace(
            packet.base,
            frame=packet.base.frame.iloc[indices].reset_index(drop=True),
            y=packet.base.y[indices],
            w=packet.base.w[indices],
        ),
    )


def _phase_weights_from_point(point: np.ndarray, packet: GenericFamilyPacket) -> dict[str, dict[str, float]]:
    return {
        f"phase_{phase_idx}": {
            packet.base.domain_names[domain_idx]: float(point[phase_idx, domain_idx])
            for domain_idx in range(point.shape[1])
        }
        for phase_idx in range(point.shape[0])
    }


def _phase_weight_matrix(
    phase_weights: dict[str, dict[str, float]],
    *,
    phase_names: tuple[str, ...],
    domain_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [[float(phase_weights[phase_name][domain_name]) for domain_name in domain_names] for phase_name in phase_names],
        dtype=float,
    )


def _mean_phase_tv_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return mean per-phase total-variation distance between two phase schedules."""
    return 0.5 * float(np.mean(np.sum(np.abs(lhs - rhs), axis=1)))


def _fit_grp(packet: GenericFamilyPacket) -> GenericFamilyRetainedTotalSurrogate:
    return GenericFamilyRetainedTotalSurrogate(packet).fit(packet.base.w, packet.base.y)


def _optimize_grp(
    model: GenericFamilyRetainedTotalSurrogate,
    packet: GenericFamilyPacket,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    alpha = float(model.params["alpha"])
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    beta = float(model.params["beta"])
    rng = np.random.default_rng(seed)

    n_singletons = len(packet.singletons)
    n_pairs = len(packet.pairs)
    n_families = len(model.family_totals)
    single_coef = np.asarray(model.coef_[:n_singletons], dtype=float)
    pair_coef = np.asarray(model.coef_[n_singletons : n_singletons + n_pairs], dtype=float)
    family_coef = np.asarray(model.coef_[n_singletons + n_pairs : n_singletons + n_pairs + n_families], dtype=float)
    penalty_coef = float(model.coef_[-1])
    family_indices = [np.asarray(packet.family_map[name], dtype=int) for name in model.family_totals]

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0, logits1 = z[:n_domains], z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 = p0 / np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 = p1 / np.sum(p1)

        e0 = c0 * p0
        e1 = c1 * p1
        retained = np.exp(-lam * (1.0 - p1))
        d_retained = lam * retained
        x = retained * e0 + eta * e1
        dx0 = retained * c0
        dx1 = d_retained * e0 + eta * c1

        value = float(model.intercept_)
        grad0 = np.zeros(n_domains, dtype=float)
        grad1 = np.zeros(n_domains, dtype=float)

        for coef, idx in zip(single_coef, packet.singletons, strict=True):
            signal = np.log1p(alpha * x[idx])
            deriv = alpha / (1.0 + alpha * x[idx])
            value -= float(coef * signal)
            grad0[idx] -= float(coef * deriv * dx0[idx])
            grad1[idx] -= float(coef * deriv * dx1[idx])

        for coef, (hi, lo) in zip(pair_coef, packet.pairs, strict=True):
            total = x[hi] + (beta * x[lo] if model.quality_discount else x[lo])
            signal = np.log1p(alpha * total)
            deriv = alpha / (1.0 + alpha * total)
            value -= float(coef * signal)
            grad0[hi] -= float(coef * deriv * dx0[hi])
            grad1[hi] -= float(coef * deriv * dx1[hi])
            lo_scale = beta if model.quality_discount else 1.0
            grad0[lo] -= float(coef * deriv * lo_scale * dx0[lo])
            grad1[lo] -= float(coef * deriv * lo_scale * dx1[lo])

        for coef, indices in zip(family_coef, family_indices, strict=True):
            total = float(np.sum(x[indices]))
            signal = np.log1p(alpha * total)
            deriv = alpha / (1.0 + alpha * total)
            value -= float(coef * signal)
            grad0[indices] -= float(coef * deriv) * dx0[indices]
            grad1[indices] -= float(coef * deriv) * dx1[indices]

        penalty = 0.0
        for idx in packet.singletons:
            u = np.log1p(x[idx]) - tau
            sp = float(np.log1p(np.exp(min(u, 20.0))) if u <= 20.0 else u)
            penalty += sp**2
            sigma = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
            deriv = 2.0 * sp * sigma / (1.0 + x[idx])
            grad0[idx] += penalty_coef * deriv * dx0[idx]
            grad1[idx] += penalty_coef * deriv * dx1[idx]

        for hi, lo in packet.pairs:
            total = x[hi] + x[lo]
            u = np.log1p(total) - tau
            sp = float(np.log1p(np.exp(min(u, 20.0))) if u <= 20.0 else u)
            penalty += sp**2
            sigma = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
            deriv = 2.0 * sp * sigma / (1.0 + total)
            grad0[hi] += penalty_coef * deriv * dx0[hi]
            grad1[hi] += penalty_coef * deriv * dx1[hi]
            grad0[lo] += penalty_coef * deriv * dx0[lo]
            grad1[lo] += penalty_coef * deriv * dx1[lo]

        value += penalty_coef * penalty
        dz0 = p0 * (grad0 - np.dot(grad0, p0))
        dz1 = p1 * (grad1 - np.dot(grad1, p1))
        return value, np.concatenate([dz0, dz1])

    starts: list[np.ndarray] = []
    uniform = np.full(n_domains, 1.0 / n_domains, dtype=float)
    starts.append(np.concatenate([np.log(uniform), np.log(uniform)]))

    best_observed = packet.base.w[int(np.argmin(packet.base.y))]
    starts.append(
        np.concatenate(
            [
                np.log(np.clip(best_observed[0], 1e-12, None)),
                np.log(np.clip(best_observed[1], 1e-12, None)),
            ]
        )
    )

    for run_name in ("baseline_unimax", "baseline_proportional"):
        idxs = packet.base.frame.index[packet.base.frame[packet.base.name_col] == run_name]
        if len(idxs) == 0:
            continue
        observed = packet.base.w[int(idxs[0])]
        starts.append(
            np.concatenate(
                [
                    np.log(np.clip(observed[0], 1e-12, None)),
                    np.log(np.clip(observed[1], 1e-12, None)),
                ]
            )
        )

    for _ in range(n_random):
        phase0 = rng.gamma(1.0, 1.0, size=n_domains)
        phase1 = rng.gamma(1.0, 1.0, size=n_domains)
        starts.append(np.concatenate([np.log(phase0 / phase0.sum()), np.log(phase1 / phase1.sum())]))

    best_result = None
    for start in starts:
        result = minimize(value_grad_logits, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 = p0 / np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 = p1 / np.sum(p1)
    return best_result, p0, p1


def _predicted_optimum_grp(model: GenericFamilyRetainedTotalSurrogate, packet: GenericFamilyPacket) -> PredictedOptimum:
    result, phase0, phase1 = _optimize_grp(model, packet, seed=0)
    return PredictedOptimum(
        predicted_objective=float(result.fun),
        phase_weights=_phase_weights_from_point(np.stack([phase0, phase1], axis=0), packet),
    )


def _subset_curve_rows(
    spec: DatasetSpec,
    packet: GenericFamilyPacket,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    best_observed_bpb = float(np.min(packet.base.y))
    phase_names = tuple(spec.phase_names)
    domain_names = tuple(spec.domain_names)
    previous_optima: dict[str, np.ndarray] = {}

    for subset_size in SUBSET_SIZES:
        start = perf_counter()
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_spec = spec.subset(subset_indices)
        train_packet = _subset_packet(packet, subset_indices)

        grp_model = _fit_grp(train_packet)
        grp_predictions = grp_model.predict(packet.base.w)
        grp_chosen_idx = int(np.argmin(grp_predictions))
        grp_optimum = _predicted_optimum_grp(grp_model, train_packet)
        grp_optimum_matrix = _phase_weight_matrix(
            grp_optimum.phase_weights,
            phase_names=phase_names,
            domain_names=domain_names,
        )
        grp_optimum_move = (
            np.nan
            if GRP_MODEL_NAME not in previous_optima
            else _mean_phase_tv_distance(grp_optimum_matrix, previous_optima[GRP_MODEL_NAME])
        )
        rows.append(
            {
                "subset_size": subset_size,
                "policy": FEATURE_POLICY,
                "model_name": GRP_MODEL_NAME,
                "predicted_bpb": float(grp_optimum.predicted_objective),
                "regret_at_1": float(packet.base.y[grp_chosen_idx] - best_observed_bpb),
                "optimum_move_mean_phase_tv": float(grp_optimum_move),
                "n_params": int(1 + len(grp_model.coef_) if grp_model.coef_ is not None else 1),
            }
        )
        previous_optima[GRP_MODEL_NAME] = grp_optimum_matrix

        olmix_fit = _fit_olmix_loglinear(train_spec, seed=0)
        olmix_predictions = np.asarray(olmix_fit.predict_fn(spec.weights), dtype=float)
        olmix_chosen_idx = int(np.argmin(olmix_predictions))
        olmix_optimum = olmix_fit.info["fit"].optimum(spec)
        olmix_optimum_matrix = _phase_weight_matrix(
            olmix_optimum.phase_weights,
            phase_names=phase_names,
            domain_names=domain_names,
        )
        olmix_optimum_move = (
            np.nan
            if OLMIX_MODEL_NAME not in previous_optima
            else _mean_phase_tv_distance(olmix_optimum_matrix, previous_optima[OLMIX_MODEL_NAME])
        )
        rows.append(
            {
                "subset_size": subset_size,
                "policy": FEATURE_POLICY,
                "model_name": OLMIX_MODEL_NAME,
                "predicted_bpb": float(olmix_optimum.predicted_objective),
                "regret_at_1": float(packet.base.y[olmix_chosen_idx] - best_observed_bpb),
                "optimum_move_mean_phase_tv": float(olmix_optimum_move),
                "n_params": int(olmix_fit.n_params),
            }
        )
        previous_optima[OLMIX_MODEL_NAME] = olmix_optimum_matrix

        pd.DataFrame(rows).to_csv(CURVE_POINTS_CSV, index=False)
        _log(f"Finished subset size k={subset_size} in {perf_counter() - start:.1f}s")

    return pd.DataFrame(rows)


def _plot_model_convergence(
    curves: pd.DataFrame,
    *,
    model_name: str,
    best_observed_bpb: float,
    output_path: Path,
) -> None:
    frame = curves[curves["model_name"] == model_name].sort_values("subset_size")
    cmap = plt.colormaps["RdYlGn_r"]
    model_color = model_bpb_color(model_name)
    fig, (ax_bpb, ax_regret, ax_move) = plt.subplots(
        3,
        1,
        figsize=(10.2, 8.4),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0], "hspace": 0.08},
    )

    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_bpb"],
        color=model_color,
        marker="o",
        linewidth=2.2,
        linestyle=PREDICTED_LINESTYLE,
        label="Predicted BPB",
    )
    ax_bpb.axhline(
        best_observed_bpb,
        color=BEST_OBSERVED_BPB_COLOR,
        linewidth=1.8,
        linestyle=":",
        label=f"Best observed BPB ({best_observed_bpb:.4f})",
    )
    ax_regret.plot(
        frame["subset_size"],
        frame["regret_at_1"],
        color=cmap(0.82),
        marker="s",
        linewidth=2.2,
        linestyle="-",
        label="Regret@1",
    )
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        linestyle="-",
        label="Optimum movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title(f"Two-phase many-domain: {model_name} convergence")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(list(SUBSET_SIZES))
    ax_move.set_xlim(min(SUBSET_SIZES), max(SUBSET_SIZES))
    ax_bpb.grid(True, alpha=0.25)
    ax_regret.grid(True, alpha=0.25)
    ax_move.grid(True, alpha=0.25)

    for axis in (ax_bpb, ax_regret, ax_move):
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        if handles:
            axis.legend(handles, labels, loc="best", frameon=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _log(f"Loading {CSV_PATH}")
    _frame, spec, packet = _load_spec()
    curve_points = _subset_curve_rows(spec, packet)
    curve_points.to_csv(CURVE_POINTS_CSV, index=False)
    _log(f"Wrote {CURVE_POINTS_CSV}")

    best_observed_bpb = float(np.min(packet.base.y))
    _plot_model_convergence(
        curve_points,
        model_name=GRP_MODEL_NAME,
        best_observed_bpb=best_observed_bpb,
        output_path=GRP_PLOT_PATH,
    )
    _plot_model_convergence(
        curve_points,
        model_name=OLMIX_MODEL_NAME,
        best_observed_bpb=best_observed_bpb,
        output_path=OLMIX_PLOT_PATH,
    )

    summary = {
        "objective_metric": OBJECTIVE_METRIC,
        "subset_sizes": list(SUBSET_SIZES),
        "curve_points_csv": str(CURVE_POINTS_CSV),
        "grp_plot": str(GRP_PLOT_PATH),
        "olmix_plot": str(OLMIX_PLOT_PATH),
        "best_observed_bpb": best_observed_bpb,
        "models": {
            model_name: [
                {
                    "subset_size": int(row["subset_size"]),
                    "predicted_bpb": float(row["predicted_bpb"]),
                    "regret_at_1": float(row["regret_at_1"]),
                    "optimum_move_mean_phase_tv": (
                        None if pd.isna(row["optimum_move_mean_phase_tv"]) else float(row["optimum_move_mean_phase_tv"])
                    ),
                    "n_params": int(row["n_params"]),
                }
                for _, row in (
                    curve_points[curve_points["model_name"] == model_name].sort_values("subset_size").iterrows()
                )
            ]
            for model_name in (GRP_MODEL_NAME, OLMIX_MODEL_NAME)
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _log(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
