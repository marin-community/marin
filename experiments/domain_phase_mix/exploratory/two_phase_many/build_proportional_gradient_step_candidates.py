# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build local gradient-step candidates from proportional perturbation effects.

The Stage 1 proportional perturbation experiment measured one-domain-at-a-time
finite differences around `baseline_proportional`. This script converts the
domain-only perturbations into candidate one-step mixtures and scores those
mixtures under the fitted local linear finite-difference model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

ROOT = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
PERTURBATION_DIR = ROOT / "reference_outputs/proportional_perturbation_scale_transfer_20260507"
DEFAULT_OUTPUT_DIR = PERTURBATION_DIR / "gradient_step_candidates_domain_only"
DOMAIN_PREFIX = "phase_0_"
EFFECT_COLUMNS = {
    "60m": "effect_60_bpb",
    "100m": "effect_100_bpb",
}
TRUST_REGION_TVS = (0.05, 0.075, 0.10)
RIDGE = 1e-5


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_scale: str
    construction: str
    target_tv: float
    weights: np.ndarray
    notes: str


def simplex_project(values: np.ndarray) -> np.ndarray:
    """Project `values` onto the probability simplex."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("simplex_project expects a one-dimensional vector")
    if len(vector) == 0:
        raise ValueError("Cannot project an empty vector")

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    rho_candidates = sorted_values * np.arange(1, len(vector) + 1) > (cumulative - 1.0)
    if not np.any(rho_candidates):
        raise ValueError("Simplex projection failed to find an active set")
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    total = projected.sum()
    if total <= 0:
        raise ValueError("Simplex projection produced zero mass")
    return projected / total


def tv_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log(positive)))


def weight_summary(weights: np.ndarray, domains: list[str]) -> tuple[str, str]:
    changes = sorted(zip(domains, weights, strict=True), key=lambda item: item[1], reverse=True)
    top = "; ".join(f"{domain}={weight:.4f}" for domain, weight in changes[:6])
    bottom = "; ".join(f"{domain}={weight:.4f}" for domain, weight in changes[-6:])
    return top, bottom


def scale_direction_to_tv(base: np.ndarray, direction: np.ndarray, target_tv: float) -> np.ndarray:
    """Move from base along direction, project, and bisection-scale to target TV."""
    centered = np.asarray(direction, dtype=float)
    centered = centered - centered.mean()
    if np.linalg.norm(centered, ord=1) <= 1e-15:
        raise ValueError("Cannot scale a near-zero direction")

    lower = 0.0
    upper = 1.0
    for _ in range(80):
        candidate = simplex_project(base + upper * centered)
        if tv_distance(candidate, base) >= target_tv or upper > 1e6:
            break
        upper *= 2.0

    for _ in range(80):
        middle = (lower + upper) / 2.0
        candidate = simplex_project(base + middle * centered)
        if tv_distance(candidate, base) < target_tv:
            lower = middle
        else:
            upper = middle
    return simplex_project(base + upper * centered)


def mirror_step_to_tv(base: np.ndarray, gradient: np.ndarray, target_tv: float) -> np.ndarray:
    """Mirror-descent step on the simplex with bisection over step size."""
    centered_gradient = gradient - gradient.mean()
    if np.linalg.norm(centered_gradient, ord=1) <= 1e-15:
        raise ValueError("Cannot mirror-step using a near-zero gradient")

    def step(eta: float) -> np.ndarray:
        logits = np.log(np.maximum(base, 1e-15)) - eta * centered_gradient
        logits = logits - logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()

    lower = 0.0
    upper = 1.0
    for _ in range(80):
        if tv_distance(step(upper), base) >= target_tv or upper > 1e6:
            break
        upper *= 2.0

    for _ in range(80):
        middle = (lower + upper) / 2.0
        if tv_distance(step(middle), base) < target_tv:
            lower = middle
        else:
            upper = middle
    return step(upper)


def fit_local_gradient(displacements: np.ndarray, effects: np.ndarray, ridge: float) -> np.ndarray:
    """Fit the local simplex gradient from finite-difference displacements."""
    lhs = displacements.T @ displacements + ridge * np.eye(displacements.shape[1])
    rhs = displacements.T @ effects
    gradient = np.linalg.solve(lhs, rhs)
    return gradient - gradient.mean()


def normalized_positive(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        raise ValueError("Positive weights sum to zero")
    return values / total


def capped_proportional_removal(base: np.ndarray, scores: np.ndarray, total_mass: float) -> np.ndarray:
    """Remove `total_mass` proportional to scores without making weights negative."""
    if total_mass <= 0:
        raise ValueError("Removal mass must be positive")
    if base.sum() + 1e-12 < total_mass:
        raise ValueError("Cannot remove more mass than exists")

    removal = np.zeros_like(base)
    active = scores > 0
    remaining = float(total_mass)
    for _ in range(len(base) + 1):
        if remaining <= 1e-14:
            break
        if not np.any(active):
            raise ValueError("Ran out of active donor domains while removing mass")
        active_scores = scores[active]
        allocation = remaining * active_scores / active_scores.sum()
        active_indices = np.flatnonzero(active)
        capacities = base[active_indices] - removal[active_indices]
        capped = allocation >= capacities
        if not np.any(capped):
            removal[active_indices] += allocation
            remaining = 0.0
            break
        capped_indices = active_indices[capped]
        removal[capped_indices] = base[capped_indices]
        remaining = total_mass - float(removal.sum())
        active[capped_indices] = False

    if abs(float(removal.sum()) - total_mass) > 1e-10:
        raise ValueError(f"Failed to remove requested mass: got {removal.sum()} vs {total_mass}")
    return removal


def balanced_transfer(
    base: np.ndarray,
    effects: np.ndarray,
    target_tv: float,
    *,
    top_k: int | None,
) -> np.ndarray:
    """Move target_tv mass from harmful domains to helpful domains."""
    good_scores = np.maximum(-effects, 0.0)
    bad_scores = np.maximum(effects, 0.0)
    all_bad_scores = bad_scores.copy()
    if top_k is not None:
        good_mask = np.zeros_like(good_scores, dtype=bool)
        bad_mask = np.zeros_like(bad_scores, dtype=bool)
        good_indices = np.argsort(effects)[:top_k]
        bad_indices = np.argsort(effects)[-top_k:]
        good_mask[good_indices] = True
        bad_mask[bad_indices] = True
        good_scores = np.where(good_mask, good_scores, 0.0)
        bad_scores = np.where(bad_mask, bad_scores, 0.0)
        if float(base[bad_scores > 0.0].sum()) < target_tv:
            bad_scores = all_bad_scores

    add = target_tv * normalized_positive(good_scores)
    remove = capped_proportional_removal(base, bad_scores, target_tv)
    candidate = base + add - remove
    if candidate.min() < -1e-12:
        raise ValueError(f"Balanced candidate has negative mass: min={candidate.min()}")
    candidate = np.maximum(candidate, 0.0)
    return candidate / candidate.sum()


def unscaled_balanced_transfer(
    base: np.ndarray,
    effects: np.ndarray,
    *,
    top_k: int | None,
) -> np.ndarray:
    """Move from harmful domains to helpful domains using raw effect magnitudes."""
    good_scores = np.maximum(-effects, 0.0)
    bad_scores = np.maximum(effects, 0.0)
    all_bad_scores = bad_scores.copy()
    if top_k is not None:
        good_mask = np.zeros_like(good_scores, dtype=bool)
        bad_mask = np.zeros_like(bad_scores, dtype=bool)
        good_mask[np.argsort(effects)[:top_k]] = True
        bad_mask[np.argsort(effects)[-top_k:]] = True
        good_scores = np.where(good_mask, good_scores, 0.0)
        bad_scores = np.where(bad_mask, bad_scores, 0.0)

    add = good_scores.copy()
    target_mass = float(add.sum())
    if target_mass <= 0:
        raise ValueError("No helpful domains for unscaled balanced transfer")
    if top_k is not None and float(base[bad_scores > 0.0].sum()) < target_mass:
        bad_scores = all_bad_scores
    remove = capped_proportional_removal(base, bad_scores, target_mass)
    candidate = base + add - remove
    if candidate.min() < -1e-12:
        raise ValueError(f"Unscaled balanced candidate has negative mass: min={candidate.min()}")
    candidate = np.maximum(candidate, 0.0)
    return candidate / candidate.sum()


def observed_convex_direction(
    displacements: np.ndarray,
    effects: np.ndarray,
    *,
    top_k: int | None,
) -> np.ndarray:
    """Average helpful observed displacement vectors using effect-weighted weights."""
    scores = np.maximum(-effects, 0.0)
    if top_k is not None:
        mask = np.zeros_like(scores, dtype=bool)
        mask[np.argsort(effects)[:top_k]] = True
        scores = np.where(mask, scores, 0.0)
    return normalized_positive(scores) @ displacements


def load_domain_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    effects = pd.read_csv(PERTURBATION_DIR / "paired_bpb_effects.csv")
    manifest = pd.read_csv(PERTURBATION_DIR / "intervention_manifest.csv")

    domain_manifest = manifest[manifest["intervention_type"] == "domain_bump"].copy()
    domain_effects = effects[effects["intervention_type"] == "domain_bump"].copy()
    merged = domain_effects.merge(
        domain_manifest,
        on=[
            "intervention_id",
            "intervention_type",
            "target_unit",
            "target_domain",
            "target_family",
            "quality_high_domain",
            "quality_low_domain",
            "tv_distance",
            "target_mass_before",
            "target_mass_after",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if len(merged) != 39:
        raise ValueError(f"Expected 39 domain bump rows, found {len(merged)}")

    phase_columns = [column for column in manifest.columns if column.startswith(DOMAIN_PREFIX)]
    domains = [column.removeprefix(DOMAIN_PREFIX) for column in phase_columns]
    if len(domains) != 39:
        raise ValueError(f"Expected 39 phase domains, found {len(domains)}")

    base_by_domain = dict(zip(merged["target_domain"], merged["target_mass_before"], strict=True))
    missing = sorted(set(domains) - set(base_by_domain))
    if missing:
        raise ValueError(f"Missing base weights for domains: {missing}")
    base = np.array([base_by_domain[domain] for domain in domains], dtype=float)
    base = base / base.sum()

    perturbed = merged[phase_columns].to_numpy(dtype=float)
    displacements = perturbed - base[None, :]
    max_row_sum = float(np.abs(displacements.sum(axis=1)).max())
    if max_row_sum > 1e-10:
        raise ValueError(f"Domain displacement rows do not sum to zero: max {max_row_sum}")

    return merged, manifest, domains, base, displacements


def build_candidates(
    merged: pd.DataFrame,
    domains: list[str],
    base: np.ndarray,
    displacements: np.ndarray,
) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    gradients: dict[str, np.ndarray] = {}
    candidates: list[Candidate] = []

    for scale, effect_column in EFFECT_COLUMNS.items():
        effects = merged[effect_column].to_numpy(dtype=float)
        ordered_effect_by_domain = merged.set_index("target_domain").loc[domains, effect_column].to_numpy(dtype=float)
        gradient = fit_local_gradient(displacements, effects, RIDGE)
        gradients[scale] = gradient

        best_index = int(np.argmin(effects))
        candidates.append(
            Candidate(
                candidate_id=f"{scale}_best_single_domain_bump",
                source_scale=scale,
                construction="best_single_observed",
                target_tv=tv_distance(base + displacements[best_index], base),
                weights=base + displacements[best_index],
                notes=f"Observed best single domain bump: {merged.iloc[best_index]['target_domain']}",
            )
        )

        for target_tv in TRUST_REGION_TVS:
            ridge_linear = scale_direction_to_tv(base, -gradient, target_tv)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_ridge_linear_tv{target_tv:.3f}",
                    source_scale=scale,
                    construction="ridge_gradient_linear_projected",
                    target_tv=target_tv,
                    weights=ridge_linear,
                    notes="Projected linear step using fitted ridge local gradient.",
                )
            )

            ridge_mirror = mirror_step_to_tv(base, gradient, target_tv)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_ridge_mirror_tv{target_tv:.3f}",
                    source_scale=scale,
                    construction="ridge_gradient_mirror_descent",
                    target_tv=target_tv,
                    weights=ridge_mirror,
                    notes="Exponentiated-gradient step using fitted ridge local gradient.",
                )
            )

            balanced_all = balanced_transfer(base, ordered_effect_by_domain, target_tv, top_k=None)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_balanced_all_tv{target_tv:.3f}",
                    source_scale=scale,
                    construction="balanced_good_bad_all",
                    target_tv=target_tv,
                    weights=balanced_all,
                    notes="Transfer mass from all harmful-effect domains to all helpful-effect domains.",
                )
            )

            balanced_top8 = balanced_transfer(base, ordered_effect_by_domain, target_tv, top_k=8)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_balanced_top8_tv{target_tv:.3f}",
                    source_scale=scale,
                    construction="balanced_good_bad_top8",
                    target_tv=target_tv,
                    weights=balanced_top8,
                    notes="Transfer mass between top-8 helpful and top-8 harmful domains.",
                )
            )

            for top_k in (None, 8):
                direction = observed_convex_direction(displacements, effects, top_k=top_k)
                weights = scale_direction_to_tv(base, direction, target_tv)
                suffix = "all" if top_k is None else f"top{top_k}"
                candidates.append(
                    Candidate(
                        candidate_id=f"{scale}_observed_good_{suffix}_tv{target_tv:.3f}",
                        source_scale=scale,
                        construction=f"observed_helpful_convex_{suffix}",
                        target_tv=target_tv,
                        weights=weights,
                        notes="Scaled convex combination of observed helpful domain-bump directions.",
                    )
                )

        for top_k in (None, 8):
            direction = observed_convex_direction(displacements, effects, top_k=top_k)
            suffix = "all" if top_k is None else f"top{top_k}"
            weights = simplex_project(base + direction)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_observed_good_{suffix}_unscaled",
                    source_scale=scale,
                    construction=f"observed_helpful_convex_{suffix}_unscaled",
                    target_tv=tv_distance(weights, base),
                    weights=weights,
                    notes="Unscaled convex combination of observed helpful domain-bump directions.",
                )
            )
            balanced = unscaled_balanced_transfer(base, ordered_effect_by_domain, top_k=top_k)
            candidates.append(
                Candidate(
                    candidate_id=f"{scale}_balanced_{suffix}_unscaled",
                    source_scale=scale,
                    construction=f"balanced_good_bad_{suffix}_unscaled",
                    target_tv=tv_distance(balanced, base),
                    weights=balanced,
                    notes="Unscaled transfer using raw helpful and harmful effect magnitudes.",
                )
            )

    # One cross-scale candidate moves in the average descent direction. This is useful as a stability diagnostic.
    average_gradient = 0.5 * (gradients["60m"] + gradients["100m"])
    for target_tv in TRUST_REGION_TVS:
        candidates.append(
            Candidate(
                candidate_id=f"avg_ridge_mirror_tv{target_tv:.3f}",
                source_scale="average",
                construction="average_ridge_gradient_mirror_descent",
                target_tv=target_tv,
                weights=mirror_step_to_tv(base, average_gradient, target_tv),
                notes="Mirror-descent step using the average 60M and 100M local gradients.",
            )
        )

    return candidates, gradients


def summarize_candidates(
    candidates: list[Candidate],
    domains: list[str],
    base: np.ndarray,
    gradients: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []

    for candidate in candidates:
        weights = candidate.weights / candidate.weights.sum()
        delta = weights - base
        top_weights, bottom_weights = weight_summary(weights, domains)
        row: dict[str, object] = {
            "candidate_id": candidate.candidate_id,
            "source_scale": candidate.source_scale,
            "construction": candidate.construction,
            "target_tv": candidate.target_tv,
            "actual_tv": tv_distance(weights, base),
            "entropy": entropy(weights),
            "support_gt_0p001": int((weights > 0.001).sum()),
            "max_domain": domains[int(np.argmax(weights))],
            "max_weight": float(weights.max()),
            "min_weight": float(weights.min()),
            "top_weights": top_weights,
            "bottom_weights": bottom_weights,
            "notes": candidate.notes,
        }
        for scale, gradient in gradients.items():
            row[f"predicted_{scale}_bpb_effect"] = float(delta @ gradient)
        row["predicted_scale_interaction_bpb"] = float(delta @ gradients["100m"]) - float(delta @ gradients["60m"])
        summary_rows.append(row)

        weight_row: dict[str, object] = {
            "candidate_id": candidate.candidate_id,
            "source_scale": candidate.source_scale,
            "construction": candidate.construction,
            "phase_mode": "both_phases",
        }
        for domain, weight in zip(domains, weights, strict=True):
            weight_row[f"phase_0_{domain}"] = float(weight)
            weight_row[f"phase_1_{domain}"] = float(weight)
        weight_rows.append(weight_row)

        for domain, base_weight, candidate_weight, change in zip(domains, base, weights, delta, strict=True):
            change_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_scale": candidate.source_scale,
                    "construction": candidate.construction,
                    "domain": domain,
                    "base_weight": float(base_weight),
                    "candidate_weight": float(candidate_weight),
                    "delta_weight": float(change),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(weight_rows), pd.DataFrame(change_rows)


def write_plots(summary: pd.DataFrame, changes: pd.DataFrame, output_dir: Path) -> None:
    long = summary.melt(
        id_vars=["candidate_id", "source_scale", "construction", "actual_tv"],
        value_vars=["predicted_60m_bpb_effect", "predicted_100m_bpb_effect"],
        var_name="prediction_scale",
        value_name="predicted_bpb_effect",
    )
    long["prediction_scale"] = (
        long["prediction_scale"].str.replace("predicted_", "", regex=False).str.replace("_bpb_effect", "", regex=False)
    )
    figure = px.bar(
        long.sort_values(["prediction_scale", "predicted_bpb_effect"]),
        x="candidate_id",
        y="predicted_bpb_effect",
        color="prediction_scale",
        barmode="group",
        hover_data=["construction", "source_scale", "actual_tv"],
        title="Predicted BPB effect for domain-only gradient-step candidates",
        color_discrete_map={"60m": "#1b9e77", "100m": "#d95f02"},
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.update_layout(xaxis_tickangle=45, height=800)
    figure.write_html(output_dir / "candidate_predicted_effects.html")

    top_candidates = summary.sort_values("predicted_100m_bpb_effect").head(8)["candidate_id"].tolist()
    heatmap_frame = changes[changes["candidate_id"].isin(top_candidates)].copy()
    heatmap_frame["domain_short"] = heatmap_frame["domain"].str.replace("dolma3_cc/", "cc/", regex=False)
    heatmap = px.imshow(
        heatmap_frame.pivot(index="candidate_id", columns="domain_short", values="delta_weight"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Domain weight deltas for top predicted 100M candidates",
        labels={"color": "delta weight"},
    )
    heatmap.update_layout(height=700)
    heatmap.write_html(output_dir / "top_100m_candidate_domain_deltas.html")


def write_report(summary: pd.DataFrame, merged: pd.DataFrame, output_dir: Path) -> None:
    diagnostics = pd.read_csv(output_dir / "local_linear_diagnostics.csv")
    recommendations = pd.read_csv(output_dir / "recommended_candidates.csv")
    best_60 = summary.sort_values("predicted_60m_bpb_effect").head(8)
    best_100 = summary.sort_values("predicted_100m_bpb_effect").head(8)
    domain_rank = merged[
        [
            "target_domain",
            "effect_60_bpb",
            "effect_100_bpb",
            "scale_interaction_bpb",
        ]
    ].sort_values("effect_100_bpb")

    lines = [
        "# Domain-Only Gradient-Step Candidates",
        "",
        "Negative BPB effect means predicted improvement over `baseline_proportional`.",
        "",
        "## Best Candidates By 60M Local Model",
        "",
        best_60[
            [
                "candidate_id",
                "construction",
                "actual_tv",
                "predicted_60m_bpb_effect",
                "predicted_100m_bpb_effect",
                "max_domain",
                "max_weight",
                "support_gt_0p001",
            ]
        ].to_markdown(index=False),
        "",
        "## Best Candidates By 100M Local Model",
        "",
        best_100[
            [
                "candidate_id",
                "construction",
                "actual_tv",
                "predicted_60m_bpb_effect",
                "predicted_100m_bpb_effect",
                "max_domain",
                "max_weight",
                "support_gt_0p001",
            ]
        ].to_markdown(index=False),
        "",
        "## Recommended Launch Candidates",
        "",
        recommendations.to_markdown(index=False),
        "",
        "## Local Linear Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Domain Bump Effects Used",
        "",
        domain_rank.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Candidate weights are constant across phases (`phase_0 == phase_1`).",
        "- Only the 39 domain-bump interventions are used; family bumps and quality swaps are excluded.",
        "- Predictions come from a ridge local linear model fit to the 39 observed domain displacement vectors.",
        "- The ridge model is high-leverage; the `observed_helpful_convex_all` recommendations are the conservative launch candidates because they stay in the cone spanned by measured helpful perturbations.",
        "- `best_single_observed` rows are the actual measured one-domain bumps, included as anchors.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def local_linear_diagnostics(merged: pd.DataFrame, displacements: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ridge in (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        for scale, effect_column in EFFECT_COLUMNS.items():
            effects = merged[effect_column].to_numpy(dtype=float)
            gradient = fit_local_gradient(displacements, effects, ridge)
            in_sample = displacements @ gradient
            loo_predictions: list[float] = []
            for heldout in range(len(effects)):
                mask = np.ones(len(effects), dtype=bool)
                mask[heldout] = False
                loo_gradient = fit_local_gradient(displacements[mask], effects[mask], ridge)
                loo_predictions.append(float(displacements[heldout] @ loo_gradient))
            loo = np.array(loo_predictions, dtype=float)
            rows.append(
                {
                    "scale": scale,
                    "ridge": ridge,
                    "in_sample_rmse": float(np.sqrt(np.mean((in_sample - effects) ** 2))),
                    "loo_rmse": float(np.sqrt(np.mean((loo - effects) ** 2))),
                    "loo_spearman": float(spearmanr(loo, effects).statistic),
                    "prediction_min": float(in_sample.min()),
                    "prediction_max": float(in_sample.max()),
                }
            )
    return pd.DataFrame(rows)


def recommended_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    recommendation_ids = [
        "60m_observed_good_all_tv0.050",
        "100m_observed_good_all_tv0.050",
        "60m_balanced_all_tv0.050",
        "100m_balanced_all_tv0.050",
        "100m_balanced_top8_tv0.050",
        "avg_ridge_mirror_tv0.050",
    ]
    reasons = {
        "60m_observed_good_all_tv0.050": (
            "Primary 60M-derived candidate: conservative convex combination of helpful observed domain bumps at the measured perturbation radius."
        ),
        "100m_observed_good_all_tv0.050": (
            "Primary 100M-derived candidate: conservative convex combination of helpful observed domain bumps at the measured perturbation radius."
        ),
        "60m_balanced_all_tv0.050": (
            "Balanced 60M candidate: explicitly transfers mass from harmful-effect domains to helpful-effect domains."
        ),
        "100m_balanced_all_tv0.050": (
            "Balanced 100M candidate: explicitly transfers mass from harmful-effect domains to helpful-effect domains."
        ),
        "100m_balanced_top8_tv0.050": (
            "Sharper 100M balanced candidate: uses top helpful domains and concentrated harmful-domain removal."
        ),
        "avg_ridge_mirror_tv0.050": (
            "Optional fitted-gradient stress test; higher model-risk than observed/balanced candidates."
        ),
    }
    subset = summary[summary["candidate_id"].isin(recommendation_ids)].copy()
    subset["recommendation_reason"] = subset["candidate_id"].map(reasons)
    order = {candidate_id: index for index, candidate_id in enumerate(recommendation_ids)}
    subset["recommendation_order"] = subset["candidate_id"].map(order)
    columns = [
        "recommendation_order",
        "candidate_id",
        "source_scale",
        "construction",
        "actual_tv",
        "predicted_60m_bpb_effect",
        "predicted_100m_bpb_effect",
        "predicted_scale_interaction_bpb",
        "max_domain",
        "max_weight",
        "support_gt_0p001",
        "recommendation_reason",
    ]
    return subset.sort_values("recommendation_order")[columns]


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, _manifest, domains, base, displacements = load_domain_data()
    candidates, gradients = build_candidates(merged, domains, base, displacements)
    summary, weights, changes = summarize_candidates(candidates, domains, base, gradients)

    summary = summary.sort_values(["source_scale", "target_tv", "construction", "candidate_id"])
    diagnostics = local_linear_diagnostics(merged, displacements)
    recommendations = recommended_candidates(summary)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    weights.to_csv(output_dir / "candidate_weights.csv", index=False)
    changes.to_csv(output_dir / "candidate_domain_changes.csv", index=False)
    diagnostics.to_csv(output_dir / "local_linear_diagnostics.csv", index=False)
    recommendations.to_csv(output_dir / "recommended_candidates.csv", index=False)
    pd.DataFrame(
        {
            "domain": domains,
            "base_weight": base,
            "local_gradient_60m": gradients["60m"],
            "local_gradient_100m": gradients["100m"],
        }
    ).to_csv(output_dir / "domain_local_gradients.csv", index=False)

    metadata = {
        "input_dir": str(PERTURBATION_DIR),
        "output_dir": str(output_dir),
        "domain_count": len(domains),
        "candidate_count": len(summary),
        "trust_region_tvs": list(TRUST_REGION_TVS),
        "ridge": RIDGE,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    write_report(summary, merged, output_dir)
    write_plots(summary, changes, output_dir)
    print(f"Wrote {len(summary)} candidates to {output_dir}")
    print(summary.sort_values("predicted_60m_bpb_effect").head(5).to_string(index=False))
    print(summary.sort_values("predicted_100m_bpb_effect").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
