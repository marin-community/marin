# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Materialize and inspect GRP and compact-retained-state raw optimum paths.

The upstream learning-curve audit fits each frozen model at increasing 3e18
fit-row counts. This script turns those raw continuous optima into canonical
mixture CSVs, collapses only numerically indistinguishable policies, and
renders a quantitative Observatory-style comparison against proportional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_grp_compact_raw_optimum_paths_20260721"
TARGETS = ("uncheatable", "table9")
MODELS = ("grp", "compact_retained_state")
DESIGNS = ("two_phase_only", "tied_spine_plus_two_phase")
FIT_ROW_COUNTS = {
    "two_phase_only": (280, 340, 400, 520, 560, 640, 760),
    "tied_spine_plus_two_phase": (518, 578, 638, 758, 798, 878, 998),
}
DEDUPLICATION_POLICY_TV = 1e-3
COORDINATE_DECIMALS = 12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
MODEL_COLORS = {"grp": "#d65f2e", "compact_retained_state": "#174f67"}
DESIGN_DASHES = {"two_phase_only": "solid", "tied_spine_plus_two_phase": "dash"}
TARGET_LABELS = {"uncheatable": "Uncheatable BPB", "table9": "Table-9 macro BPB"}
MODEL_LABELS = {"grp": "Original GRP", "compact_retained_state": "Compact retained state"}
DESIGN_LABELS = {
    "two_phase_only": "two-phase rows only",
    "tied_spine_plus_two_phase": "tied spine + two-phase rows",
}


@dataclass(frozen=True)
class Policy:
    """One normalized two-phase mixture proposed by a fitted surrogate."""

    phase0: np.ndarray
    phase1: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        return np.stack([self.phase0, self.phase1], axis=0)


@dataclass(frozen=True)
class EmpiricalEnvelope:
    """Data-relative scale for screening raw optimizer extrapolation."""

    max_bucket_weight_q99: float
    max_simulated_epochs_q99: float
    phase_total_variation_q99: float
    support_distance_q95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dedup-tv", type=float, default=DEDUPLICATION_POLICY_TV)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def weighted_policy_tv(left: Policy, right: Policy, alpha0: float, alpha1: float) -> float:
    phase0 = 0.5 * np.abs(left.phase0 - right.phase0).sum()
    phase1 = 0.5 * np.abs(left.phase1 - right.phase1).sum()
    return float(alpha0 * phase0 + alpha1 * phase1)


def policy_hash(policy: Policy) -> str:
    rounded = np.round(policy.weights.astype(np.float64), decimals=COORDINATE_DECIMALS)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def parse_policy(row: pd.Series, domains: int) -> Policy:
    phase0 = np.asarray(json.loads(str(row["phase_0_weights_json"])), dtype=float)
    phase1 = np.asarray(json.loads(str(row["phase_1_weights_json"])), dtype=float)
    if phase0.shape != (domains,) or phase1.shape != (domains,):
        raise ValueError(f"Expected {domains} weights per phase, found {phase0.shape} and {phase1.shape}")
    if np.any(phase0 < 0.0) or np.any(phase1 < 0.0):
        raise ValueError("Mixture weights must be non-negative")
    if not np.isclose(phase0.sum(), 1.0) or not np.isclose(phase1.sum(), 1.0):
        raise ValueError("Each phase must sum to one")
    return Policy(phase0, phase1)


def validate_path_frame(frame: pd.DataFrame, allow_incomplete: bool) -> None:
    required = {
        "target",
        "model",
        "design",
        "total_unique_training_rows",
        "predicted_bpb",
        "phase_0_weights_json",
        "phase_1_weights_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Raw optimum table is missing columns: {missing}")
    if frame.duplicated(["target", "model", "design", "total_unique_training_rows", "seed"]).any():
        raise ValueError("Raw optimum table has duplicate path coordinates")
    unknown_targets = sorted(set(frame["target"]) - set(TARGETS))
    unknown_models = sorted(set(frame["model"]) - set(MODELS))
    unknown_designs = sorted(set(frame["design"]) - set(DESIGNS))
    if unknown_targets or unknown_models or unknown_designs:
        raise ValueError(
            f"Unexpected path values: targets={unknown_targets}, models={unknown_models}, designs={unknown_designs}"
        )
    if not allow_incomplete:
        expected = {
            (target, model, design, rows)
            for target in TARGETS
            for model in MODELS
            for design in DESIGNS
            for rows in FIT_ROW_COUNTS[design]
        }
        observed = set(
            frame[["target", "model", "design", "total_unique_training_rows"]].itertuples(index=False, name=None)
        )
        missing_paths = sorted(expected - observed)
        extra_paths = sorted(observed - expected)
        if missing_paths or extra_paths:
            raise ValueError(f"Raw path is incomplete: missing={missing_paths}, extra={extra_paths}")


def fit_envelope(dataset: pooled.Dataset) -> EmpiricalEnvelope:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    maximum_weights = dataset.weights.max(axis=(1, 2))
    epochs = dataset.weights[:, 0, :] * dataset.c0 + dataset.weights[:, 1, :] * dataset.c1
    maximum_epochs = epochs.max(axis=1)
    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)

    flattened = dataset.weights.reshape(dataset.n, -1)
    scale = np.maximum(np.std(flattened, axis=0), 1e-3)
    normalized = flattened / scale
    squared = np.sum(normalized**2, axis=1, keepdims=True)
    pairwise_squared = np.maximum(squared + squared.T - 2.0 * normalized @ normalized.T, 0.0)
    np.fill_diagonal(pairwise_squared, np.inf)
    nearest = np.sqrt(pairwise_squared.min(axis=1))
    if not np.isclose(alpha0 + alpha1, 1.0):
        raise ValueError("Phase fractions must sum to one")
    return EmpiricalEnvelope(
        max_bucket_weight_q99=float(np.quantile(maximum_weights, 0.99)),
        max_simulated_epochs_q99=float(np.quantile(maximum_epochs, 0.99)),
        phase_total_variation_q99=float(np.quantile(phase_tv, 0.99)),
        support_distance_q95=float(np.quantile(nearest, 0.95)),
    )


def mixture_frame(dataset: pooled.Dataset, natural: np.ndarray, policy: Policy) -> pd.DataFrame:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    aggregate = alpha0 * policy.phase0 + alpha1 * policy.phase1
    proportional_epochs = natural * dataset.c0 + natural * dataset.c1
    return pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "natural_weight": natural,
            "phase_0_weight": policy.phase0,
            "phase_1_weight": policy.phase1,
            "aggregate_weight": aggregate,
            "simulated_epochs": policy.phase0 * dataset.c0 + policy.phase1 * dataset.c1,
            "proportional_simulated_epochs": proportional_epochs,
        }
    )


def cluster_is_compatible(
    candidate: int,
    cluster: list[int],
    policies: list[Policy],
    alpha0: float,
    alpha1: float,
    tolerance: float,
) -> bool:
    candidate_policy = policies[candidate]
    distances = (weighted_policy_tv(candidate_policy, policies[member], alpha0, alpha1) for member in cluster)
    return all(distance <= tolerance for distance in distances)


def cluster_paths(
    frame: pd.DataFrame,
    policies: list[Policy],
    alpha0: float,
    alpha1: float,
    tolerance: float,
) -> list[list[int]]:
    """Complete-link clustering prevents a chain of small drifts becoming one policy."""
    order = frame.sort_values(["target", "model", "design", "total_unique_training_rows", "seed"]).index.to_list()
    clusters: list[list[int]] = []
    for index in order:
        compatible = []
        for cluster in clusters:
            if cluster_is_compatible(
                index,
                cluster,
                policies,
                alpha0,
                alpha1,
                tolerance,
            ):
                compatible.append(cluster)
        if compatible:
            compatible[0].append(index)
        else:
            clusters.append([index])
    return clusters


def medoid(cluster: list[int], policies: list[Policy], alpha0: float, alpha1: float) -> int:
    totals = {
        index: sum(weighted_policy_tv(policies[index], policies[other], alpha0, alpha1) for other in cluster)
        for index in cluster
    }
    return min(cluster, key=lambda index: (totals[index], -index))


def path_label(row: pd.Series) -> str:
    return (
        f"{TARGET_LABELS[str(row['target'])]} · {MODEL_LABELS[str(row['model'])]} · "
        f"{DESIGN_LABELS[str(row['design'])]} · {int(row['total_unique_training_rows'])} rows"
    )


def write_mixture_explorer(
    paths: pd.DataFrame,
    policies: list[Policy],
    datasets: dict[str, pooled.Dataset],
    output_path: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.045,
        subplot_titles=("Phase 0 weights", "Phase 1 weights", "Aggregate weights", "Aggregate exposure"),
    )
    candidate_color = "#e76f35"
    proportional_color = "#8b9ba5"
    traces_per_path = 8
    ordered = (
        paths.assign(
            target_order=paths["target"].map({"uncheatable": 0, "table9": 1}),
            model_order=paths["model"].map({"compact_retained_state": 0, "grp": 1}),
            design_order=paths["design"].map({"tied_spine_plus_two_phase": 0, "two_phase_only": 1}),
        )
        .sort_values(
            ["target_order", "model_order", "design_order", "total_unique_training_rows"],
            ascending=[True, True, True, False],
        )
        .reset_index()
    )
    buttons: list[dict[str, Any]] = []
    for path_number, row in ordered.iterrows():
        original_index = int(row["index"])
        dataset = datasets[str(row["target"])]
        alpha0, _alpha1 = observatory.phase_fractions(dataset)
        natural = observatory.natural_weights(dataset, alpha0)
        local = mixture_frame(dataset, natural, policies[original_index])
        visible = path_number == 0
        custom = np.column_stack(
            [
                local["domain"],
                local["simulated_epochs"],
                local["proportional_simulated_epochs"],
            ]
        )
        panels = (
            ("phase_0_weight", "natural_weight", "weight"),
            ("phase_1_weight", "natural_weight", "weight"),
            ("aggregate_weight", "natural_weight", "weight"),
            ("simulated_epochs", "proportional_simulated_epochs", "epochs"),
        )
        for column, (candidate_column, baseline_column, unit) in enumerate(panels, start=1):
            figure.add_trace(
                go.Bar(
                    x=local[candidate_column],
                    y=local["domain"],
                    orientation="h",
                    marker_color=candidate_color,
                    name="predicted optimum",
                    legendgroup="candidate",
                    offsetgroup="candidate",
                    showlegend=column == 1,
                    visible=visible,
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>candidate=%{x:.6f} "
                        + unit
                        + "<br>candidate exposure=%{customdata[1]:.3f} epochs"
                        "<br>proportional exposure=%{customdata[2]:.3f} epochs<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Bar(
                    x=local[baseline_column],
                    y=local["domain"],
                    orientation="h",
                    marker_color=proportional_color,
                    name="proportional",
                    legendgroup="proportional",
                    offsetgroup="proportional",
                    showlegend=column == 1,
                    visible=visible,
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>proportional=%{x:.6f} "
                        + unit
                        + "<br>proportional exposure=%{customdata[2]:.3f} epochs<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        visibility = [False] * (len(ordered) * traces_per_path)
        start = path_number * traces_per_path
        visibility[start : start + traces_per_path] = [True] * traces_per_path
        title = (
            f"{path_label(row)}<br><sup>predicted BPB={float(row['predicted_bpb']):.6f} · "
            f"unique policy={row['candidate_id']} · grouped path rows={row['grouped_fit_row_counts']} · "
            f"max weight={float(row['max_bucket_weight']):.3f} · max epoch={float(row['max_simulated_epochs']):.2f} · "
            f"phase TV={float(row['phase_total_variation']):.3f} · support distance="
            f"{float(row['standardized_fit_support_distance']):.2f} · empirical screen="
            f"{'inside' if bool(row['within_empirical_shape_envelope']) else 'outside'}</sup>"
        )
        buttons.append(
            {
                "label": path_label(row),
                "method": "update",
                "args": [{"visible": visibility}, {"title": {"text": title, "x": 0.01}}],
            }
        )

    first = ordered.iloc[0]
    first_title = (
        f"{path_label(first)}<br><sup>predicted BPB={float(first['predicted_bpb']):.6f} · "
        f"unique policy={first['candidate_id']} · grouped path rows={first['grouped_fit_row_counts']} · "
        f"max weight={float(first['max_bucket_weight']):.3f} · max epoch={float(first['max_simulated_epochs']):.2f} · "
        f"phase TV={float(first['phase_total_variation']):.3f} · support distance="
        f"{float(first['standardized_fit_support_distance']):.2f} · empirical screen="
        f"{'inside' if bool(first['within_empirical_shape_envelope']) else 'outside'}</sup>"
    )
    figure.update_layout(
        title={"text": first_title, "x": 0.01},
        template="plotly_white",
        barmode="group",
        width=1900,
        height=1420,
        margin={"l": 235, "r": 50, "t": 210, "b": 90},
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.17,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(title_text="mixture probability", row=1, col=1)
    figure.update_xaxes(title_text="mixture probability", row=1, col=2)
    figure.update_xaxes(title_text="token-weighted probability", row=1, col=3)
    figure.update_xaxes(title_text="realized simulated epochs", row=1, col=4)
    html = pio.to_html(figure, include_plotlyjs=True, full_html=False, config=PLOT_CONFIG)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>GRP and compact raw optimum mixtures</title>"
        "<style>body{margin:0;background:#fbfaf6;color:#173042;font-family:Georgia,serif}"
        ".facts{max-width:1500px;margin:20px auto 60px;padding:24px 32px;border:1px solid #d8d1c2;"
        "background:#fffdf8;line-height:1.5}.facts h2{margin-top:0}.facts code{font-family:ui-monospace,monospace}"
        "</style></head><body>" + html + "<section class='facts'><h2>Interpretation</h2>"
        "<p>Each dropdown entry is the unregularized continuous optimum of the named surrogate fitted at the "
        "shown 3e18 row count. Orange is the predicted optimum; gray is proportional. Aggregate weights use "
        "the 80%/20% phase fractions. Aggregate exposure is the realized simulated epoch count across both phases.</p>"
        "<p>Near-identical policies are materialized once when every pair in a group is within weighted policy "
        "TV <code>1e-3</code>. This does not certify deployment safety: the title separately reports max weight, "
        "max epoch, phase divergence, and standardized distance from the fit swarm.</p></section></body></html>"
    )


def write_path_diagnostics(
    paths: pd.DataFrame,
    envelopes: dict[str, EmpiricalEnvelope],
    output_path: Path,
) -> None:
    metrics = (
        ("predicted_bpb", "Predicted BPB"),
        ("max_simulated_epochs", "Maximum simulated epochs"),
        ("standardized_fit_support_distance", "Standardized fit-support distance"),
        ("phase_total_variation", "Phase total variation"),
    )
    figure = make_subplots(
        rows=len(metrics),
        cols=2,
        shared_xaxes=True,
        subplot_titles=tuple(f"{TARGET_LABELS[target]} · {label}" for _metric, label in metrics for target in TARGETS),
        vertical_spacing=0.07,
    )
    for row_number, (metric, _label) in enumerate(metrics, start=1):
        for column, target in enumerate(TARGETS, start=1):
            for model in MODELS:
                for design in DESIGNS:
                    local = paths.loc[
                        paths["target"].eq(target) & paths["model"].eq(model) & paths["design"].eq(design)
                    ].sort_values("total_unique_training_rows")
                    figure.add_trace(
                        go.Scatter(
                            x=local["total_unique_training_rows"],
                            y=local[metric],
                            mode="lines+markers",
                            line={"color": MODEL_COLORS[model], "dash": DESIGN_DASHES[design], "width": 2},
                            marker={"size": 8, "symbol": "circle" if design == "two_phase_only" else "diamond"},
                            name=f"{MODEL_LABELS[model]} · {DESIGN_LABELS[design]}",
                            legendgroup=f"{model}-{design}",
                            showlegend=row_number == 1 and column == 1,
                            customdata=np.column_stack(
                                [local["candidate_id"], local["grouped_fit_row_counts"], local["optimizer_converged"]]
                            ),
                            hovertemplate=(
                                "%{customdata[0]}<br>fit rows=%{x}<br>value=%{y:.6f}<br>"
                                "grouped rows=%{customdata[1]}<br>optimizer converged=%{customdata[2]}<extra></extra>"
                            ),
                        ),
                        row=row_number,
                        col=column,
                    )
    figure.update_xaxes(title_text="unique 3e18 fit policies", row=len(metrics))
    for row_number, (_metric, label) in enumerate(metrics, start=1):
        figure.update_yaxes(title_text=label, row=row_number, col=1)
    for column, target in enumerate(TARGETS, start=1):
        envelope = envelopes[target]
        figure.add_hline(
            y=envelope.max_simulated_epochs_q99,
            line={"color": "#8b9ba5", "dash": "dot", "width": 1.5},
            row=2,
            col=column,
        )
        figure.add_hline(
            y=2.0 * envelope.support_distance_q95,
            line={"color": "#8b9ba5", "dash": "dot", "width": 1.5},
            row=3,
            col=column,
        )
        figure.add_hline(
            y=envelope.phase_total_variation_q99,
            line={"color": "#8b9ba5", "dash": "dot", "width": 1.5},
            row=4,
            col=column,
        )
        figure.update_yaxes(type="log", row=2, col=column)
    figure.update_layout(
        title="Original GRP and compact retained-state raw optimum paths",
        template="plotly_white",
        width=1550,
        height=1450,
        legend={"orientation": "h", "y": 1.04},
        margin={"l": 100, "r": 50, "t": 130, "b": 70},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def grouped_fit_rows(paths: pd.DataFrame, cluster: list[int]) -> str:
    return "; ".join(
        f"{target}/{model}/{design}:" + ",".join(str(value) for value in sorted(local["total_unique_training_rows"]))
        for (target, model, design), local in paths.loc[cluster].groupby(["target", "model", "design"], sort=True)
    )


def write_report(
    paths: pd.DataFrame,
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
    envelopes: dict[str, EmpiricalEnvelope],
    output_path: Path,
    dedup_tv: float,
) -> None:
    endpoint = (
        paths.sort_values("total_unique_training_rows")
        .groupby(["target", "model", "design"], as_index=False)
        .tail(1)
        .sort_values(["target", "model", "design"])
    )
    columns = [
        "target",
        "model_label",
        "design",
        "total_unique_training_rows",
        "predicted_bpb",
        "max_bucket_weight",
        "max_simulated_epochs",
        "phase_total_variation",
        "standardized_fit_support_distance",
        "within_empirical_shape_envelope",
        "optimizer_converged",
        "successful_starts",
        "candidate_id",
    ]
    convergence = (
        paths.groupby(["target", "model", "design"], as_index=False)
        .agg(
            path_points=("candidate_id", "size"),
            unique_policies=("candidate_id", "nunique"),
            final_successive_tv=("successive_optimum_tv", "last"),
            max_successive_tv=("successive_optimum_tv", "max"),
        )
        .sort_values(["target", "model", "design"])
    )
    envelope_table = pd.DataFrame([{"target": target, **vars(envelope)} for target, envelope in envelopes.items()])
    unsafe = candidates.loc[~candidates["within_empirical_shape_envelope"]]
    inside_count = int(candidates["within_empirical_shape_envelope"].sum())
    converged_count = int(candidates["optimizer_converged"].sum())
    text = f"""# Original GRP and compact retained-state raw optimum paths

## Protocol

- Fit evidence: Delphi 3e18 only. The two designs use the original 280 two-phase rows,
  optionally the 238 independent tied policies, and increasing frozen-order phase-varying extensions.
- Hyperparameters: frozen from each model's Observatory fit; no 3e18 validation outcome is
  used to tune these paths.
- Optimization: unregularized continuous surrogate objective with multistart softmax-coordinate L-BFGS-B.
- Materialization: {len(paths)} fitted path points collapse to {len(candidates)} unique policies
  at complete-link weighted policy TV <= {dedup_tv:g}.
- The empirical shape screen is descriptive, not deployment regularization. It asks whether max
  weight, max epoch, and phase divergence stay within the fit panel's 99th-percentile envelope,
  and whether support distance stays within twice the fit panel's leave-one-out 95th percentile.

## Endpoint optima

{endpoint[columns].to_markdown(index=False, floatfmt=".6f")}

## Path convergence

{convergence.to_markdown(index=False, floatfmt=".6f")}

## Fit-swarm empirical envelopes

{envelope_table.to_markdown(index=False, floatfmt=".6f")}

## Sanity assessment

- {inside_count}/{len(candidates)} unique policies remain inside the descriptive empirical
  shape/support envelope.
- {converged_count}/{len(candidates)} representative policies came from an optimizer start that
  reported convergence.
- {len(unsafe)} unique policies exceed at least one empirical envelope diagnostic. These may still
  be useful as deliberate stress tests, but they are not ordinary validation candidates.
- Deduplication only removes numerically indistinguishable coordinates; it does not turn a stable
  but extrapolative endpoint into a trustworthy optimum.

## Validation shortlist

Original GRP is excluded: every endpoint is a distant, high-repetition corner. The four Compact
endpoints are geometrically plausible enough for a small 3e18 validation panel, while their large
predicted gains remain explicitly unconfirmed.

{shortlist.to_markdown(index=False, floatfmt=".6f")}

## Artifacts

- `grp_compact_raw_optimum_mixtures.html`: quantitative per-bucket policy explorer.
- `raw_optimum_path_diagnostics.html`: fit-row learning curves for prediction and geometry.
- `path_manifest.csv`: every fit-row proposal and its unique materialized policy.
- `candidate_manifest.csv`: one row per unique policy.
- `recommended_validation_manifest.csv`: four Compact endpoint candidates; no job has been submitted.
- `mixtures/*.csv`: canonical phase-weight and simulated-exposure files.
"""
    output_path.write_text(text)


def main() -> None:
    args = parse_args()
    if args.dedup_tv <= 0.0:
        raise ValueError("--dedup-tv must be positive")
    raw_path = args.input_dir / "raw_optimum_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    paths = pd.read_csv(raw_path)
    validate_path_frame(paths, args.allow_incomplete)
    paths = paths.sort_values(["target", "model", "design", "total_unique_training_rows", "seed"]).reset_index(drop=True)
    paths["model_label"] = paths["model"].map(MODEL_LABELS)

    datasets = {target: observatory.load_delphi_3e18_fit_dataset(target) for target in TARGETS}
    domains = datasets[TARGETS[0]].m
    policies = [parse_policy(row, domains) for _, row in paths.iterrows()]
    alpha0, alpha1 = observatory.phase_fractions(datasets[TARGETS[0]])
    for target in TARGETS[1:]:
        target_alpha0, target_alpha1 = observatory.phase_fractions(datasets[target])
        if not np.isclose(alpha0, target_alpha0) or not np.isclose(alpha1, target_alpha1):
            raise ValueError("Targets use inconsistent phase fractions")

    clusters = cluster_paths(paths, policies, alpha0, alpha1, args.dedup_tv)
    candidate_rows: list[dict[str, Any]] = []
    path_candidate_ids: dict[int, str] = {}
    path_grouped_rows: dict[int, str] = {}
    mixture_dir = args.output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    for cluster in clusters:
        representative = medoid(cluster, policies, alpha0, alpha1)
        candidate_id = f"grpcompact_rawopt_{policy_hash(policies[representative])[:12]}"
        if candidate_id in {row["candidate_id"] for row in candidate_rows}:
            raise ValueError(f"Candidate hash collision for {candidate_id}")
        grouped_rows = grouped_fit_rows(paths, cluster)
        for index in cluster:
            path_candidate_ids[index] = candidate_id
            path_grouped_rows[index] = grouped_rows
        row = paths.iloc[representative]
        dataset = datasets[str(row["target"])]
        local = mixture_frame(
            dataset,
            observatory.natural_weights(dataset, alpha0),
            policies[representative],
        )
        local.to_csv(mixture_dir / f"{candidate_id}.csv", index=False)
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "coordinate_hash": policy_hash(policies[representative]),
                "representative_target": row["target"],
                "representative_model": row["model"],
                "representative_design": row["design"],
                "representative_fit_rows": int(row["total_unique_training_rows"]),
                "proposal_count": len(cluster),
                "proposal_paths": grouped_rows,
                "optimizer_converged": bool(row["optimizer_converged"]),
                "successful_starts": int(row["successful_starts"]),
                "predicted_bpb": float(row["predicted_bpb"]),
                "max_bucket_weight": float(local[["phase_0_weight", "phase_1_weight"]].to_numpy().max()),
                "max_simulated_epochs": float(local["simulated_epochs"].max()),
                "phase_total_variation": float(
                    0.5 * np.abs(policies[representative].phase0 - policies[representative].phase1).sum()
                ),
                "standardized_fit_support_distance": float(row["standardized_fit_support_distance"]),
                "mixture_csv": f"mixtures/{candidate_id}.csv",
            }
        )

    paths["candidate_id"] = [path_candidate_ids[index] for index in paths.index]
    paths["grouped_fit_row_counts"] = [path_grouped_rows[index] for index in paths.index]
    candidates = pd.DataFrame(candidate_rows)
    envelopes = {target: fit_envelope(dataset) for target, dataset in datasets.items()}
    candidates["within_empirical_shape_envelope"] = [
        bool(
            row.max_bucket_weight <= envelopes[str(row.representative_target)].max_bucket_weight_q99
            and row.max_simulated_epochs <= envelopes[str(row.representative_target)].max_simulated_epochs_q99
            and row.phase_total_variation <= envelopes[str(row.representative_target)].phase_total_variation_q99
            and row.standardized_fit_support_distance
            <= 2.0 * envelopes[str(row.representative_target)].support_distance_q95
        )
        for row in candidates.itertuples(index=False)
    ]
    paths["within_empirical_shape_envelope"] = [
        bool(
            row.max_bucket_weight <= envelopes[str(row.target)].max_bucket_weight_q99
            and row.max_simulated_epochs <= envelopes[str(row.target)].max_simulated_epochs_q99
            and row.phase_total_variation <= envelopes[str(row.target)].phase_total_variation_q99
            and row.standardized_fit_support_distance <= 2.0 * envelopes[str(row.target)].support_distance_q95
        )
        for row in paths.itertuples(index=False)
    ]
    shortlist = (
        paths.loc[paths["model"].eq("compact_retained_state")]
        .sort_values("total_unique_training_rows")
        .groupby(["target", "design"], as_index=False)
        .tail(1)
        .sort_values(["target", "design"])
        .copy()
    )
    if len(shortlist) != 4 or not shortlist["within_empirical_shape_envelope"].all():
        raise ValueError("Expected four geometrically admissible Compact endpoint candidates")
    shortlist["validation_role"] = shortlist["design"].map(
        {
            "tied_spine_plus_two_phase": "primary heterogeneous-fit endpoint",
            "two_phase_only": "two-phase-only design control",
        }
    )
    shortlist["mixture_csv"] = shortlist["candidate_id"].map(lambda value: f"mixtures/{value}.csv")
    shortlist_columns = [
        "target",
        "design",
        "total_unique_training_rows",
        "predicted_bpb",
        "max_bucket_weight",
        "max_simulated_epochs",
        "phase_total_variation",
        "standardized_fit_support_distance",
        "candidate_id",
        "validation_role",
        "mixture_csv",
    ]
    shortlist = shortlist[shortlist_columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.output_dir / "path_manifest.csv", index=False)
    candidates.sort_values("candidate_id").to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    shortlist.to_csv(args.output_dir / "recommended_validation_manifest.csv", index=False)
    protocol = {
        "input": str(raw_path),
        "path_points": len(paths),
        "unique_policies": len(candidates),
        "deduplication_weighted_policy_tv": args.dedup_tv,
        "deduplication_linkage": "complete",
        "phase_fractions": [alpha0, alpha1],
        "fit_row_counts": FIT_ROW_COUNTS,
        "empirical_envelopes": {target: vars(envelope) for target, envelope in envelopes.items()},
    }
    (args.output_dir / "materialization_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_mixture_explorer(
        paths,
        policies,
        datasets,
        args.output_dir / "grp_compact_raw_optimum_mixtures.html",
    )
    write_path_diagnostics(paths, envelopes, args.output_dir / "raw_optimum_path_diagnostics.html")
    write_report(paths, candidates, shortlist, envelopes, args.output_dir / "report.md", args.dedup_tv)


if __name__ == "__main__":
    main()
