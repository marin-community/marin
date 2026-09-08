# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Materialize a preregistered KRR/HPR support-radius validation panel.

Candidate generation uses only the canonical 280-row Delphi 3e18 fit swarm.
Five independent proposal banks estimate optimizer stability. Existing
heldouts and the running KRR pilot are used only for coordinate deduplication;
their target values are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_hellinger_krr_delphi_3e18_20260727 as krr  # noqa: E402
import benchmark_hierarchical_coverage_grp_20260715 as hierarchical  # noqa: E402
import compare_krr_hpr_support_radius_3e18_20260727 as comparison  # noqa: E402
from support_radius_regularization import SupportGeometry, build_support_geometry, support_distance_batch  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, load_scale  # noqa: E402

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "krr_hpr_support_radius_validation_panel_20260727"
RUNNING_KRR_PANEL = SCRIPT_DIR / "reference_outputs" / "hellinger_krr_delphi_3e18_20260727" / "validation_panel.csv"
BANK_SEEDS = (101, 211, 307, 401, 503)
DEFAULT_BANK_SIZE = 100_000
RADIUS_SPECS = (
    ("raw", np.inf),
    ("r100", 1.00),
    ("r050", 0.50),
    ("r025", 0.25),
    ("r010", 0.10),
)
MODELS = comparison.MODELS
TARGETS = (UNCHEATABLE, TABLE9)
PREDICTED_SPREAD_LIMIT = 0.005
MEDIAN_POLICY_TV_LIMIT = 0.20
COORDINATE_ALIAS_L1_TOLERANCE = 2e-6
MIN_NEW_CANDIDATES_PER_CELL = 2
MAX_PANEL_RUNS = 20
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
MODEL_LABELS = {
    "content_krr": "Content-Hellinger KRR",
    "hierarchical_phase_replay": "Hierarchical phase replay",
}
MODEL_COLORS = {
    "content_krr": "#1a9850",
    "hierarchical_phase_replay": "#d73027",
}
MODEL_SLUGS = {
    "content_krr": "krr",
    "hierarchical_phase_replay": "hpr",
}
TARGET_SLUGS = {
    UNCHEATABLE: "unch",
    TABLE9: "t9",
}


@dataclass(frozen=True)
class BankWinner:
    """One bank-specific constrained optimum."""

    model: str
    target: str
    radius_label: str
    requested_radius: float
    seed: int
    predicted_bpb: float
    support_distance: float
    normalized_support_distance: float
    bank_kind: str
    weights: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One pooled-bank candidate and its preregistered gate diagnostics."""

    winner: BankWinner
    policy_hash: str
    predicted_spread: float
    median_policy_tv: float
    max_policy_tv: float
    max_weight: float
    max_simulated_epoch: float
    phase_tv: float
    support_gate_pass: bool
    stability_gate_pass: bool
    coordinate_gate_pass: bool
    eligible: bool
    exclusion_reason: str
    alias_source: str
    alias_row: str


@dataclass(frozen=True)
class KnownCoordinates:
    """Policies unavailable for new validation because they already exist."""

    source: str
    row_ids: np.ndarray
    weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE)
    return parser.parse_args()


def policy_hash(weights: np.ndarray) -> str:
    """Return a stable coordinate hash after harmless numerical rounding."""
    values = np.round(np.asarray(weights, dtype="<f8"), decimals=12)
    return hashlib.sha256(values.tobytes()).hexdigest()


def phase_fraction_weighted_tv(left: np.ndarray, right: np.ndarray, alpha: float) -> float:
    """Return the phase-duration-weighted total variation between policies."""
    phase_tv = 0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum(axis=1)
    return float(alpha * phase_tv[0] + (1.0 - alpha) * phase_tv[1])


def predict_hpr(model: hierarchical.Model, weights: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """Predict HPR in bounded chunks to keep candidate-bank memory stable."""
    predicted = np.empty(len(weights), dtype=float)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        predicted[start:stop] = model.predict(weights[start:stop])
    return predicted


def select_bank_winner(
    *,
    model: str,
    target: str,
    radius_label: str,
    requested_radius: float,
    seed: int,
    predicted: np.ndarray,
    distances: np.ndarray,
    bank_weights: np.ndarray,
    bank_kind: np.ndarray,
    geometry: SupportGeometry,
) -> BankWinner:
    """Select the exact best bank member inside one normalized radius."""
    if np.isinf(requested_radius):
        feasible = np.arange(len(bank_weights))
    else:
        threshold = requested_radius * geometry.loo_radius_q95 + 1e-12
        feasible = np.flatnonzero(distances <= threshold)
    if not len(feasible):
        raise ValueError(f"No candidate for {model}/{target}/{radius_label} in seed {seed}")
    index = int(feasible[np.argmin(predicted[feasible])])
    return BankWinner(
        model=model,
        target=target,
        radius_label=radius_label,
        requested_radius=requested_radius,
        seed=seed,
        predicted_bpb=float(predicted[index]),
        support_distance=float(distances[index]),
        normalized_support_distance=float(distances[index] / geometry.loo_radius_q95),
        bank_kind=str(bank_kind[index]),
        weights=np.asarray(bank_weights[index], dtype=float),
    )


def known_coordinates(fit: Panel, heldout: Panel) -> list[KnownCoordinates]:
    """Load coordinates used only for duplicate detection."""
    known = [
        KnownCoordinates(
            source="fit_swarm",
            row_ids=np.asarray(fit.row_id, dtype=str),
            weights=np.stack([fit.phase0, fit.phase1], axis=1),
        ),
        KnownCoordinates(
            source="heldout_archive",
            row_ids=np.asarray(heldout.row_id, dtype=str),
            weights=np.stack([heldout.phase0, heldout.phase1], axis=1),
        ),
    ]
    if RUNNING_KRR_PANEL.exists():
        frame = pd.read_csv(RUNNING_KRR_PANEL)
        phase0 = frame[[f"phase_0_{bucket}" for bucket in fit.buckets]].to_numpy(float)
        phase1 = frame[[f"phase_1_{bucket}" for bucket in fit.buckets]].to_numpy(float)
        known.append(
            KnownCoordinates(
                source="running_krr_pilot",
                row_ids=frame["run_name"].astype(str).to_numpy(),
                weights=np.stack([phase0, phase1], axis=1),
            )
        )
    return known


def coordinate_alias(weights: np.ndarray, known: list[KnownCoordinates]) -> tuple[str, str]:
    """Return an existing coordinate alias, if one is present."""
    for group in known:
        distance = np.abs(group.weights - weights[None]).sum(axis=(1, 2))
        index = int(np.argmin(distance))
        if float(distance[index]) <= COORDINATE_ALIAS_L1_TOLERANCE:
            return group.source, str(group.row_ids[index])
    return "", ""


def fit_maxima(panel: Panel) -> dict[str, float]:
    """Return fit-panel plausibility maxima used by the frozen gate."""
    return {
        "max_weight": float(np.maximum(panel.phase0.max(axis=1), panel.phase1.max(axis=1)).max()),
        "max_simulated_epoch": float(panel.epochs.max()),
        "phase_tv": float(panel.phase_tv.max()),
    }


def candidate_diagnostics(
    winners: list[BankWinner],
    panel: Panel,
    geometry: SupportGeometry,
    maxima: dict[str, float],
    known: list[KnownCoordinates],
) -> Candidate:
    """Apply the preregistered stability, support, and coordinate gates."""
    pooled = min(winners, key=lambda winner: winner.predicted_bpb)
    predicted = np.asarray([winner.predicted_bpb for winner in winners], dtype=float)
    policy_tv = np.asarray(
        [phase_fraction_weighted_tv(winner.weights, pooled.weights, panel.alpha) for winner in winners],
        dtype=float,
    )
    max_weight = float(pooled.weights.max())
    phase_tv = float(0.5 * np.abs(pooled.weights[1] - pooled.weights[0]).sum())
    aggregate_epochs = panel.c0 * pooled.weights[0] + panel.c1 * pooled.weights[1]
    max_epoch = float(aggregate_epochs.max())
    radius_pass = np.isinf(pooled.requested_radius) or (
        pooled.support_distance <= pooled.requested_radius * geometry.loo_radius_q95 + 1e-10
    )
    plausibility_pass = (
        max_weight <= maxima["max_weight"] + 1e-9
        and max_epoch <= maxima["max_simulated_epoch"] + 1e-9
        and phase_tv <= maxima["phase_tv"] + 1e-9
    )
    predicted_spread = float(predicted.max() - predicted.min())
    median_tv = float(np.median(policy_tv))
    stability_pass = predicted_spread <= PREDICTED_SPREAD_LIMIT and median_tv <= MEDIAN_POLICY_TV_LIMIT
    alias_source, alias_row = coordinate_alias(pooled.weights, known)
    coordinate_pass = not alias_source

    reasons = []
    if not radius_pass:
        reasons.append("requested support radius violated")
    if not plausibility_pass:
        reasons.append("fit-panel plausibility maximum exceeded")
    if predicted_spread > PREDICTED_SPREAD_LIMIT:
        reasons.append("bank optimum value did not converge")
    if median_tv > MEDIAN_POLICY_TV_LIMIT:
        reasons.append("bank optimum coordinate did not converge")
    if alias_source:
        reasons.append(f"existing coordinate: {alias_source}/{alias_row}")
    support_pass = bool(radius_pass and plausibility_pass)
    eligible = bool(support_pass and stability_pass and coordinate_pass)
    return Candidate(
        winner=pooled,
        policy_hash=policy_hash(pooled.weights),
        predicted_spread=predicted_spread,
        median_policy_tv=median_tv,
        max_policy_tv=float(policy_tv.max()),
        max_weight=max_weight,
        max_simulated_epoch=max_epoch,
        phase_tv=phase_tv,
        support_gate_pass=support_pass,
        stability_gate_pass=stability_pass,
        coordinate_gate_pass=coordinate_pass,
        eligible=eligible,
        exclusion_reason="; ".join(reasons),
        alias_source=alias_source,
        alias_row=alias_row,
    )


def mixture_frame(panel: Panel, weights: np.ndarray) -> pd.DataFrame:
    """Return the canonical phase-mixture CSV representation."""
    available = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] for bucket in panel.buckets], dtype=float)
    proportional = available / available.sum()
    aggregate = panel.alpha * weights[0] + (1.0 - panel.alpha) * weights[1]
    simulated_epochs = TARGET_BUDGET_DOLMA3_COMMON_CRAWL * aggregate / available
    frame = pd.DataFrame(
        {
            "domain": panel.buckets,
            "proportional": proportional,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": aggregate,
            "available_tokens": available,
            "simulated_epochs": simulated_epochs,
            "phase_0_epoch_multiplier": weights[0] / proportional,
            "phase_1_epoch_multiplier": weights[1] / proportional,
            "phase_0_delta": weights[0] - proportional,
            "phase_1_delta": weights[1] - proportional,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    return frame


def validation_row(
    panel: Panel,
    candidate_id: str,
    aliases: list[Candidate],
    run_order: int,
) -> dict[str, object]:
    """Return one launcher-ready row for a coordinate-disjoint policy."""
    representative = aliases[0]
    weights = representative.winner.weights
    available = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] for bucket in panel.buckets], dtype=float)
    proportional = available / available.sum()
    aggregate = panel.alpha * weights[0] + (1.0 - panel.alpha) * weights[1]
    nominal_epochs = TARGET_BUDGET_DOLMA3_COMMON_CRAWL * aggregate / available
    mean_phase_tv = float(np.mean([0.5 * np.abs(weights[phase] - proportional).sum() for phase in range(2)]))
    targets = sorted({candidate.winner.target for candidate in aliases})
    proposal_target = targets[0] if len(targets) == 1 else "joint"
    row: dict[str, object] = {
        "run_order": run_order,
        "run_id": 2_026_072_800 + run_order,
        "run_name": candidate_id,
        "source_experiment": "krr_hpr_support_radius_validation_20260727",
        "panel_source": "preregistered_support_radius",
        "proposal_target": proposal_target,
        "proposal_targets": "|".join(targets),
        "candidate_kind": "support_radius_ladder",
        "model_names": "|".join(sorted({candidate.winner.model for candidate in aliases})),
        "radius_labels": "|".join(sorted({candidate.winner.radius_label for candidate in aliases})),
        "candidate_aliases": "|".join(
            f"{candidate.winner.model}:{candidate.winner.target}:{candidate.winner.radius_label}"
            for candidate in aliases
        ),
        "policy_hash": representative.policy_hash,
        "predicted_bpb": representative.winner.predicted_bpb,
        "nearest_fit_hellinger_sq": representative.winner.support_distance,
        "normalized_support_distance": representative.winner.normalized_support_distance,
        "max_weight": representative.max_weight,
        "max_simulated_epoch": representative.max_simulated_epoch,
        "nominal_0p8_max_simulated_epoch": float(nominal_epochs.max()),
        "phase_tv": representative.phase_tv,
        "mean_phase_tv_to_proportional": mean_phase_tv,
        "fit_phase_0_fraction": panel.alpha,
        "fit_phase_1_fraction": 1.0 - panel.alpha,
        "data_seed": 2_026_072_800 + run_order,
        "trainer_seed": 0,
    }
    for phase in range(2):
        row.update(
            {
                f"phase_{phase}_{bucket}": float(weight)
                for bucket, weight in zip(panel.buckets, weights[phase], strict=True)
            }
        )
    return row


def candidate_rows(candidates: list[Candidate]) -> pd.DataFrame:
    """Return one audit row per model-target-radius candidate."""
    rows = []
    for candidate in candidates:
        winner = candidate.winner
        rows.append(
            {
                "model": winner.model,
                "target": winner.target,
                "radius_label": winner.radius_label,
                "requested_normalized_radius": winner.requested_radius,
                "winning_seed": winner.seed,
                "predicted_bpb": winner.predicted_bpb,
                "support_distance": winner.support_distance,
                "normalized_support_distance": winner.normalized_support_distance,
                "bank_kind": winner.bank_kind,
                "policy_hash": candidate.policy_hash,
                "predicted_spread": candidate.predicted_spread,
                "median_policy_tv": candidate.median_policy_tv,
                "max_policy_tv": candidate.max_policy_tv,
                "max_weight": candidate.max_weight,
                "max_simulated_epoch": candidate.max_simulated_epoch,
                "phase_tv": candidate.phase_tv,
                "support_gate_pass": candidate.support_gate_pass,
                "stability_gate_pass": candidate.stability_gate_pass,
                "coordinate_gate_pass": candidate.coordinate_gate_pass,
                "eligible": candidate.eligible,
                "exclusion_reason": candidate.exclusion_reason,
                "alias_source": candidate.alias_source,
                "alias_row": candidate.alias_row,
            }
        )
    return pd.DataFrame(rows)


def bank_winner_rows(winners: list[BankWinner]) -> pd.DataFrame:
    """Return compact per-bank stability evidence."""
    return pd.DataFrame(
        [
            {
                "model": winner.model,
                "target": winner.target,
                "radius_label": winner.radius_label,
                "requested_normalized_radius": winner.requested_radius,
                "seed": winner.seed,
                "predicted_bpb": winner.predicted_bpb,
                "support_distance": winner.support_distance,
                "normalized_support_distance": winner.normalized_support_distance,
                "bank_kind": winner.bank_kind,
                "policy_hash": policy_hash(winner.weights),
            }
            for winner in winners
        ]
    )


def plot_stability(
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
    destination: Path,
) -> None:
    """Plot bank-specific values and coordinate stability by target."""
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: bank optima",
            "Table-9: bank optima",
            "Uncheatable: coordinate stability",
            "Table-9: coordinate stability",
        ),
        vertical_spacing=0.14,
    )
    radius_order = [label for label, _ in RADIUS_SPECS]
    for column, target in enumerate(TARGETS, start=1):
        for model in MODELS:
            bank_group = winners[(winners["target"] == target) & (winners["model"] == model)].copy()
            bank_group["radius_label"] = pd.Categorical(
                bank_group["radius_label"],
                categories=radius_order,
                ordered=True,
            )
            candidate_group = candidates[(candidates["target"] == target) & (candidates["model"] == model)].copy()
            candidate_group["radius_label"] = pd.Categorical(
                candidate_group["radius_label"],
                categories=radius_order,
                ordered=True,
            )
            bank_group = bank_group.sort_values(["radius_label", "seed"])
            candidate_group = candidate_group.sort_values("radius_label")
            figure.add_trace(
                go.Scatter(
                    x=bank_group["radius_label"],
                    y=bank_group["predicted_bpb"],
                    mode="markers",
                    name=f"{MODEL_LABELS[model]} bank winners",
                    legendgroup=model,
                    marker={"color": MODEL_COLORS[model], "size": 7, "opacity": 0.45},
                    showlegend=column == 1,
                    customdata=bank_group[["seed", "normalized_support_distance", "bank_kind"]],
                    hovertemplate=(
                        "radius: %{x}<br>predicted BPB: %{y:.6f}<br>seed: %{customdata[0]}<br>"
                        "realized radius: %{customdata[1]:.3f}<br>source: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=candidate_group["radius_label"],
                    y=candidate_group["predicted_bpb"],
                    mode="lines+markers",
                    name=f"{MODEL_LABELS[model]} pooled winner",
                    legendgroup=model,
                    marker={"color": MODEL_COLORS[model], "size": 12, "symbol": "star"},
                    line={"color": MODEL_COLORS[model], "width": 2},
                    showlegend=column == 1,
                    customdata=candidate_group[
                        ["eligible", "predicted_spread", "median_policy_tv", "normalized_support_distance"]
                    ],
                    hovertemplate=(
                        "radius: %{x}<br>predicted BPB: %{y:.6f}<br>eligible: %{customdata[0]}<br>"
                        "value spread: %{customdata[1]:.6f}<br>median TV: %{customdata[2]:.3f}<br>"
                        "realized radius: %{customdata[3]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=candidate_group["radius_label"],
                    y=candidate_group["median_policy_tv"],
                    mode="lines+markers",
                    name=MODEL_LABELS[model],
                    legendgroup=model,
                    line={"color": MODEL_COLORS[model], "width": 2},
                    marker={"color": MODEL_COLORS[model], "size": 8},
                    showlegend=False,
                    customdata=candidate_group[["predicted_spread", "eligible"]],
                    hovertemplate=(
                        "radius: %{x}<br>median TV to pooled: %{y:.3f}<br>"
                        "value spread: %{customdata[0]:.6f}<br>eligible: %{customdata[1]}<extra></extra>"
                    ),
                ),
                row=2,
                col=column,
            )
    figure.add_hline(
        y=MEDIAN_POLICY_TV_LIMIT,
        line={"color": "#666666", "dash": "dash"},
        row=2,
        col="all",
    )
    figure.update_yaxes(title_text="Predicted BPB", row=1)
    figure.update_yaxes(title_text="Median weighted TV to pooled winner", row=2)
    figure.update_xaxes(title_text="Requested normalized support radius", row=2)
    figure.update_layout(
        title={
            "text": (
                "KRR/HPR support-radius candidate stability"
                "<br><sup>Five independent 100k-policy banks; no heldout target values used</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=980,
        width=1500,
        margin={"t": 120, "r": 220},
        legend={"orientation": "v", "y": 1.0, "x": 1.01, "xanchor": "left"},
    )
    figure.write_html(destination, include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    destination: Path,
    candidate_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
    gate: dict[str, object],
    geometry: SupportGeometry,
    bank_size: int,
) -> None:
    """Write the frozen local-gate report."""
    rows = [
        "# KRR/HPR support-radius validation panel",
        "",
        "## Frozen protocol",
        "",
        "- Fit source: canonical 280-row Delphi 3e18 two-phase swarm.",
        f"- Independent candidate banks: {len(BANK_SEEDS)} x {bank_size:,} policies.",
        f"- Bank seeds: `{BANK_SEEDS}`.",
        "- Requested normalized radii: raw, 1.0, 0.5, 0.25, 0.1.",
        f"- Fit leave-one-out q95 Hellinger radius: `{geometry.loo_radius_q95:.8f}`.",
        "- Existing target values were not used; heldout coordinates were loaded only for deduplication.",
        "",
        "## Gate",
        "",
        f"- Predicted optimum spread <= {PREDICTED_SPREAD_LIMIT:.3f} BPB.",
        f"- Median weighted policy TV <= {MEDIAN_POLICY_TV_LIMIT:.2f}.",
        "- Requested radius and fit-panel weight/epoch/phase-TV maxima respected.",
        "- No fit, heldout, or running-KRR-pilot coordinate aliases.",
        f"- At least {MIN_NEW_CANDIDATES_PER_CELL} new candidates per model-target cell.",
        f"- At most {MAX_PANEL_RUNS} deduplicated training runs.",
        "",
        f"**Panel gate: {'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        f"Deduplicated runs: `{gate['panel_runs']}`.",
        "",
        "## Candidate summary",
        "",
        (
            "| Model | Target | Radius | Predicted | Realized radius | Value spread | "
            "Median TV | Eligible | Exclusion |"
        ),
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in candidate_frame.itertuples(index=False):
        rows.append(
            f"| {MODEL_LABELS[row.model]} | {row.target} | {row.radius_label} | "
            f"{row.predicted_bpb:.6f} | {row.normalized_support_distance:.3f} | "
            f"{row.predicted_spread:.6f} | {row.median_policy_tv:.3f} | "
            f"{'yes' if row.eligible else 'no'} | {row.exclusion_reason or ''} |"
        )
    rows.extend(
        [
            "",
            "## Launcher panel",
            "",
            "| Run | Proposal targets | Models | Radius aliases |",
            "|---|---|---|---|",
        ]
    )
    for row in panel_frame.itertuples(index=False):
        rows.append(f"| {row.run_name} | {row.proposal_targets} | {row.model_names} | {row.radius_labels} |")
    rows.extend(
        [
            "",
            "## Gate details",
            "",
            "```json",
            json.dumps(gate, indent=2, sort_keys=True),
            "```",
        ]
    )
    destination.write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mixtures").mkdir(exist_ok=True)

    fit, heldout = load_scale("delphi_3e18")
    content_basis, basis_provenance = krr.load_embedding_basis(
        fit.buckets,
        krr.DEFAULT_HISTOGRAM_DIR,
        krr.DEFAULT_LOOKUP,
    )
    geometry = build_support_geometry(
        np.stack([fit.phase0, fit.phase1], axis=1),
        content_basis,
        np.asarray([fit.alpha, 1.0 - fit.alpha]),
    )
    known = known_coordinates(fit, heldout)
    maxima = fit_maxima(fit)

    models: dict[tuple[str, str], krr.KernelFit | hierarchical.Model] = {}
    hpr_provenance = {}
    for target in TARGETS:
        models[("content_krr", target)] = krr.fit_kernel_model(fit, content_basis, "content", target, 0)
        models[("hierarchical_phase_replay", target)], hpr_provenance[target] = comparison.fit_hpr(fit, target)

    all_winners: list[BankWinner] = []
    for target_index, target in enumerate(TARGETS):
        for seed in BANK_SEEDS:
            bank_seed = seed + 1000 * target_index
            phase0, phase1, bank_kind = krr.sample_candidate_bank(fit, target, args.bank_size, bank_seed)
            bank_weights = np.stack([phase0, phase1], axis=1).astype(float)
            distances = support_distance_batch(bank_weights, geometry)
            krr_predicted, _ = krr.evaluate_bank(
                models[("content_krr", target)],
                bank_weights[:, 0],
                bank_weights[:, 1],
            )
            hpr_predicted = predict_hpr(models[("hierarchical_phase_replay", target)], bank_weights)
            predictions = {
                "content_krr": krr_predicted,
                "hierarchical_phase_replay": hpr_predicted,
            }
            for model in MODELS:
                for radius_label, requested_radius in RADIUS_SPECS:
                    all_winners.append(
                        select_bank_winner(
                            model=model,
                            target=target,
                            radius_label=radius_label,
                            requested_radius=requested_radius,
                            seed=bank_seed,
                            predicted=predictions[model],
                            distances=distances,
                            bank_weights=bank_weights,
                            bank_kind=bank_kind,
                            geometry=geometry,
                        )
                    )

    candidates = []
    for target in TARGETS:
        for model in MODELS:
            for radius_label, _requested_radius in RADIUS_SPECS:
                group = [
                    winner
                    for winner in all_winners
                    if winner.target == target and winner.model == model and winner.radius_label == radius_label
                ]
                if len(group) != len(BANK_SEEDS):
                    raise ValueError(f"Missing bank winners for {model}/{target}/{radius_label}")
                candidates.append(candidate_diagnostics(group, fit, geometry, maxima, known))

    candidate_frame = candidate_rows(candidates)
    winner_frame = bank_winner_rows(all_winners)

    eligible = [candidate for candidate in candidates if candidate.eligible]
    coordinate_groups: dict[str, list[Candidate]] = {}
    for candidate in eligible:
        coordinate_groups.setdefault(candidate.policy_hash, []).append(candidate)

    validation_rows = []
    for run_order, (coordinate_hash, aliases) in enumerate(sorted(coordinate_groups.items())):
        first = aliases[0]
        candidate_id = (
            f"sr_{MODEL_SLUGS[first.winner.model]}_{TARGET_SLUGS[first.winner.target]}_"
            f"{first.winner.radius_label}_{coordinate_hash[:6]}"
        )
        validation_rows.append(validation_row(fit, candidate_id, aliases, run_order))
        mixture_frame(fit, first.winner.weights).to_csv(
            args.output_dir / "mixtures" / f"{candidate_id}.csv",
            index=False,
            float_format="%.17g",
        )

    panel_frame = pd.DataFrame(validation_rows)
    cell_counts = {
        f"{model}/{target}": len(
            {
                candidate.policy_hash
                for candidate in candidates
                if candidate.winner.model == model and candidate.winner.target == target and candidate.eligible
            }
        )
        for model in MODELS
        for target in TARGETS
    }
    gate = {
        "passed": bool(
            all(count >= MIN_NEW_CANDIDATES_PER_CELL for count in cell_counts.values())
            and 0 < len(panel_frame) <= MAX_PANEL_RUNS
        ),
        "panel_runs": len(panel_frame),
        "eligible_aliases": len(eligible),
        "cell_counts": cell_counts,
        "bank_seeds": BANK_SEEDS,
        "bank_size": args.bank_size,
        "predicted_spread_limit": PREDICTED_SPREAD_LIMIT,
        "median_policy_tv_limit": MEDIAN_POLICY_TV_LIMIT,
        "coordinate_alias_l1_tolerance": COORDINATE_ALIAS_L1_TOLERANCE,
        "fit_maxima": maxima,
    }
    if len(panel_frame):
        panel_frame.to_csv(args.output_dir / "validation_panel.csv", index=False, float_format="%.17g")
    candidate_frame.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    winner_frame.to_csv(args.output_dir / "bank_winners.csv", index=False)
    (args.output_dir / "panel_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    provenance = {
        "fit_rows": len(fit),
        "heldout_coordinate_rows_used_for_deduplication": len(heldout),
        "heldout_target_values_used": False,
        "models": MODELS,
        "targets": TARGETS,
        "radius_specs": [(label, None if np.isinf(radius) else radius) for label, radius in RADIUS_SPECS],
        "bank_seeds": BANK_SEEDS,
        "bank_size": args.bank_size,
        "loo_radius_q95": geometry.loo_radius_q95,
        "content_basis": basis_provenance,
        "hpr": hpr_provenance,
        "running_panel_deduplication_source": str(RUNNING_KRR_PANEL),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    plot_stability(candidate_frame, winner_frame, args.output_dir / "candidate_stability.html")
    write_report(
        args.output_dir / "report.md",
        candidate_frame,
        panel_frame,
        gate,
        geometry,
        args.bank_size,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
