# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
# ///
"""Benchmark CAMEL-style intrinsic-domain GRP adaptations locally."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    TUNED_GENERIC_FAMILY_PARAMS,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.intrinsic_domain_followup import (
    DEFAULT_INTRINSIC_DOMAIN_COUNT,
    IntrinsicDomainRetainedTotalSurrogate,
    IntrinsicFeatureMode,
    IntrinsicGroupBasis,
    IntrinsicPenaltyMode,
    intrinsic_param_count,
    learn_intrinsic_group_basis,
    optimize_intrinsic_domain_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    regression_metrics,
)

plt.switch_backend("Agg")
matplotlib.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = SCRIPT_DIR / "two_phase_many_grp_intrinsic_domain_summary.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_intrinsic_domain_summary.json"
COMPARISON_PNG = SCRIPT_DIR / "two_phase_many_grp_intrinsic_domain_local_comparison.png"
HEATMAP_PNG = SCRIPT_DIR / "two_phase_many_grp_intrinsic_domain_memberships.png"

TOPK_ACTUAL = 8
TRUSTBLEND_LINE_GRID = 81
CV_SEED = 0


@dataclass(frozen=True)
class IntrinsicVariantSummary:
    """Local benchmark summary for one CAMEL-style GRP adaptation."""

    variant: str
    model_scope: str
    deployment_rule: str
    intrinsic_domains: int
    n_params: int
    train_r2: float
    cv_r2: float
    cv_rmse: float
    cv_spearman: float
    cv_regret_at_1: float
    cv_foldmean_regret_at_1: float
    predicted_optimum_value: float
    deployment_delta: float | None
    deployment_gain_budget: float | None
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    phase0_max_weight: float
    phase1_max_weight: float


def _latent_hull_objective(
    coeff_logits: np.ndarray,
    anchors: np.ndarray,
    model: IntrinsicDomainRetainedTotalSurrogate,
) -> float:
    shifted = coeff_logits - np.max(coeff_logits)
    coeffs = np.exp(shifted)
    coeffs /= np.sum(coeffs)
    latent = np.tensordot(coeffs, anchors, axes=1)[None, :]
    return float(model.predict_from_latent_totals(latent)[0])


def optimize_latent_convex_hull(
    anchors: np.ndarray,
    model: IntrinsicDomainRetainedTotalSurrogate,
    *,
    start_indices: np.ndarray | None = None,
    maxiter: int = 100,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimize a latent-only model over convex combinations of latent anchors."""
    num_anchors = anchors.shape[0]
    if start_indices is None:
        vertex_indices = range(num_anchors)
    else:
        vertex_indices = np.asarray(start_indices, dtype=int).tolist()
    starts = [np.zeros(num_anchors, dtype=float)] + [
        np.eye(num_anchors, dtype=float)[idx] * 4.0 for idx in vertex_indices
    ]
    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(
            _latent_hull_objective,
            start,
            args=(anchors, model),
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result
    if best_result is None:
        raise RuntimeError("Latent convex-hull optimization failed")
    shifted = np.asarray(best_result.x, dtype=float) - np.max(best_result.x)
    coeffs = np.exp(shifted)
    coeffs /= np.sum(coeffs)
    latent = np.tensordot(coeffs, anchors, axes=1)
    predicted = float(model.predict_from_latent_totals(latent[None, :])[0])
    return predicted, coeffs, np.asarray(latent, dtype=float)


def _trustblend_select(
    model,
    packet,
    tuning_cv_rmse: float,
    tuning_cv_foldmean_regret_at_1: float,
    raw_weights: np.ndarray,
    hull_weights: np.ndarray,
    hull_predicted_value: float,
) -> tuple[float, float, float, np.ndarray]:
    gain_budget = float(tuning_cv_rmse + tuning_cv_foldmean_regret_at_1)
    raw_predicted_value = float(model.predict(raw_weights[None, :, :])[0])
    target_gain = min(float(hull_predicted_value) - raw_predicted_value, gain_budget)
    best: tuple[tuple[int, float, float], float, float, np.ndarray] | None = None
    for delta in np.linspace(0.0, 1.0, TRUSTBLEND_LINE_GRID):
        weights = (1.0 - delta) * hull_weights + delta * raw_weights
        predicted_value = float(model.predict(weights[None, :, :])[0])
        realized_gain = float(hull_predicted_value) - predicted_value
        feasible = realized_gain <= gain_budget + 1e-12
        key = (0 if feasible else 1, predicted_value, abs(realized_gain - target_gain))
        if best is None or key < best[0]:
            best = (key, float(delta), predicted_value, weights)
    if best is None:
        raise RuntimeError("Trustblend selection failed")
    _, delta, predicted_value, weights = best
    return predicted_value, delta, gain_budget, np.asarray(weights, dtype=float)


def _evaluate_model(
    variant: str,
    model,
    packet,
    *,
    deployment_rule: str,
) -> IntrinsicVariantSummary:
    weights = packet.base.w
    y = packet.base.y
    frame = packet.base.frame
    name_col = packet.base.name_col

    train_pred = model.predict(weights)
    train = regression_metrics(frame, name_col, y, train_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y, dtype=float)
    fold_regrets: list[float] = []
    for tr, te in kf.split(weights):
        if isinstance(model, GenericFamilyRetainedTotalSurrogate):
            fold_model = GenericFamilyRetainedTotalSurrogate(
                packet,
                params=dict(model.params),
                family_totals=model.family_totals,
                quality_discount=model.quality_discount,
                pair_cc_domains=model.pair_cc_domains,
                include_penalty=model.include_penalty,
                signal_transform=model.signal_transform,
            ).fit(weights[tr], y[tr])
        else:
            fold_model = IntrinsicDomainRetainedTotalSurrogate(
                packet,
                model.basis,
                params=dict(model.params),
                feature_mode=model.feature_mode,
                penalty_mode=model.penalty_mode,
                quality_discount=model.quality_discount,
                pair_cc_domains=model.pair_cc_domains,
                include_penalty=model.include_penalty,
                signal_transform=model.signal_transform,
            ).fit(weights[tr], y[tr])
        pred = fold_model.predict(weights[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

    cv = regression_metrics(frame, name_col, y, oof)
    tuning_cv_rmse = float(cv["rmse"])
    tuning_cv_foldmean_regret_at_1 = float(np.mean(fold_regrets))

    chosen_idx = int(np.argmin(train_pred))
    top_indices = np.argsort(y)[:TOPK_ACTUAL]

    if isinstance(model, GenericFamilyRetainedTotalSurrogate):
        _raw_result, phase0, phase1 = optimize_generic_family_model(packet, model, seed=0)
        raw_weights = np.stack([phase0, phase1], axis=0)
        hull_predicted_value, _hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
            model,
            packet.base.w[top_indices],
            start_indices=np.arange(min(len(top_indices), 8), dtype=int),
        )
        if deployment_rule != "trustblend":
            raise ValueError(f"Unsupported baseline deployment rule: {deployment_rule}")
        predicted_optimum_value, deployment_delta, deployment_gain_budget, optimum = _trustblend_select(
            model,
            packet,
            tuning_cv_rmse,
            tuning_cv_foldmean_regret_at_1,
            raw_weights,
            hull_weights,
            hull_predicted_value,
        )
    else:
        if deployment_rule == "trustblend":
            _raw_result, phase0, phase1 = optimize_intrinsic_domain_model(packet, model, seed=0)
            raw_weights = np.stack([phase0, phase1], axis=0)
            hull_predicted_value, _hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
                model,
                packet.base.w[top_indices],
                start_indices=np.arange(min(len(top_indices), 8), dtype=int),
            )
            predicted_optimum_value, deployment_delta, deployment_gain_budget, optimum = _trustblend_select(
                model,
                packet,
                tuning_cv_rmse,
                tuning_cv_foldmean_regret_at_1,
                raw_weights,
                hull_weights,
                hull_predicted_value,
            )
        elif deployment_rule == "latent_top8actual_hull":
            latent_anchors = model.latent_totals(packet.base.w[top_indices])
            predicted_optimum_value, latent_coeffs, _latent_optimum = optimize_latent_convex_hull(
                latent_anchors,
                model,
                start_indices=np.arange(min(len(top_indices), 8), dtype=int),
            )
            optimum = np.tensordot(latent_coeffs, packet.base.w[top_indices], axes=1)
            deployment_delta = None
            deployment_gain_budget = None
        else:
            raise ValueError(f"Unsupported intrinsic deployment rule: {deployment_rule}")

    distances = 0.5 * np.abs(packet.base.w - optimum[None, :, :]).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))

    if isinstance(model, GenericFamilyRetainedTotalSurrogate):
        n_params = len(model.coef_) + 1 + len(model.params)
        model_scope = "current_grp"
        intrinsic_domains = 0
    else:
        n_params = intrinsic_param_count(model)
        model_scope = model.feature_mode.value
        intrinsic_domains = model.basis.num_intrinsic_domains

    return IntrinsicVariantSummary(
        variant=variant,
        model_scope=model_scope,
        deployment_rule=deployment_rule,
        intrinsic_domains=intrinsic_domains,
        n_params=n_params,
        train_r2=float(train["r2"]),
        cv_r2=float(cv["r2"]),
        cv_rmse=tuning_cv_rmse,
        cv_spearman=float(cv["spearman"]),
        cv_regret_at_1=float(cv["regret_at_1"]),
        cv_foldmean_regret_at_1=tuning_cv_foldmean_regret_at_1,
        predicted_optimum_value=float(predicted_optimum_value),
        deployment_delta=deployment_delta,
        deployment_gain_budget=deployment_gain_budget,
        fullswarm_chosen_run_name=str(frame.iloc[chosen_idx][name_col]),
        fullswarm_chosen_value=float(y[chosen_idx]),
        nearest_observed_run_name=str(frame.iloc[nearest_idx][name_col]),
        nearest_observed_value=float(y[nearest_idx]),
        nearest_observed_tv_distance=float(distances[nearest_idx]),
        phase0_max_weight=float(np.max(optimum[0])),
        phase1_max_weight=float(np.max(optimum[1])),
    )


def _comparison_dataframe() -> tuple[pd.DataFrame, IntrinsicGroupBasis]:
    packet = load_generic_family_packet(target=MANY_DOMAIN_TARGET)
    basis = learn_intrinsic_group_basis(
        packet,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        num_intrinsic_domains=DEFAULT_INTRINSIC_DOMAIN_COUNT,
    )

    baseline = GenericFamilyRetainedTotalSurrogate(packet, params=TUNED_GENERIC_FAMILY_PARAMS).fit(
        packet.base.w, packet.base.y
    )
    soft_family = IntrinsicDomainRetainedTotalSurrogate(
        packet,
        basis,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        feature_mode=IntrinsicFeatureMode.SOFT_FAMILY,
        penalty_mode=IntrinsicPenaltyMode.GROUP,
    ).fit(packet.base.w, packet.base.y)
    latent_bottleneck = IntrinsicDomainRetainedTotalSurrogate(
        packet,
        basis,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        feature_mode=IntrinsicFeatureMode.LATENT_BOTTLENECK,
        penalty_mode=IntrinsicPenaltyMode.LATENT,
    ).fit(packet.base.w, packet.base.y)

    rows = [
        _evaluate_model("GRP", baseline, packet, deployment_rule="trustblend"),
        _evaluate_model("GRP + intrinsic families", soft_family, packet, deployment_rule="trustblend"),
        _evaluate_model("GRP + latent bottleneck", latent_bottleneck, packet, deployment_rule="trustblend"),
        _evaluate_model(
            "GRP + latent bottleneck + intrinsic-space deploy",
            latent_bottleneck,
            packet,
            deployment_rule="latent_top8actual_hull",
        ),
    ]
    frame = pd.DataFrame([asdict(row) for row in rows])
    return frame, basis


def _plot_comparison(frame: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    labels = frame["variant"].tolist()
    y_pos = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("cv_r2", "CV R$^2$", True),
        ("cv_foldmean_regret_at_1", "CV Mean Regret@1", False),
        ("predicted_optimum_value", "Predicted Deployment BPB", False),
        ("nearest_observed_tv_distance", "Nearest Observed TV", False),
    ]

    for ax, (column, title, higher_is_better) in zip(axes.flat, metrics, strict=True):
        values = frame[column].to_numpy(dtype=float)
        if np.allclose(values.max(), values.min()):
            colors = [cmap(0.5)] * len(values)
        else:
            normalized = (values - values.min()) / (values.max() - values.min())
            scores = 1.0 - normalized if higher_is_better else normalized
            colors = [cmap(float(score)) for score in scores]
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()
        ax.set_title(title)
        for y_idx, value in enumerate(values):
            ax.text(value, y_idx, f" {value:.4f}", va="center", ha="left", fontsize=8)
        if column == "predicted_optimum_value":
            ax.axvline(1.0572, color="black", linestyle="--", linewidth=1, label="best observed")
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("CAMEL-Style Intrinsic-Domain GRP Variants (Local, Exploratory)")
    fig.savefig(COMPARISON_PNG, dpi=200)
    plt.close(fig)


def _plot_memberships(basis: IntrinsicGroupBasis) -> None:
    frame = pd.DataFrame(
        basis.memberships,
        index=basis.group_names,
        columns=[f"intrinsic_{idx+1}" for idx in range(basis.num_intrinsic_domains)],
    )
    order = np.argsort(-frame.max(axis=1).to_numpy())
    ordered = frame.iloc[order]

    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    image = ax.imshow(ordered.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(ordered.shape[1]), labels=list(ordered.columns), rotation=0)
    ax.set_yticks(np.arange(ordered.shape[0]), labels=list(ordered.index))
    ax.set_title("Learned Intrinsic-Domain Memberships over 26 GRP Groups")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("membership")
    fig.savefig(HEATMAP_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    frame, basis = _comparison_dataframe()
    frame.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    _plot_comparison(frame)
    _plot_memberships(basis)


if __name__ == "__main__":
    main()
