# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Evaluate distance-to-swarm regularization for KRR and DSP at Delphi 3e18.

The deployment objective is

    w(lambda) = argmin_w L_hat(w) + lambda D(w, S) / r_95,

where ``D`` is nearest-swarm squared Hellinger distance and ``r_95`` is its
leave-one-out 95th percentile on the 280-row fit swarm. Lambda therefore has
BPB units: it is the price charged for moving one empirical support radius.

Content-Hellinger is the primary geometry and is shared by KRR and DSP. Exact
weight-Hellinger is an embedding-free ablation. Existing coordinate-disjoint
heldouts are used only after the model, geometry, and lambda grid are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
STANDALONE_CODE = SCRIPT_DIR / "standalone_code"
for path in (REPO_ROOT, SCRIPT_DIR, STANDALONE_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_hellinger_krr_delphi_3e18_20260727 as krr  # noqa: E402
import dsp_exact as dsp  # noqa: E402
from dsp_exact_baseline_20260726 import _fit_once as fit_dsp_once  # noqa: E402
from support_radius_regularization import (  # noqa: E402
    PathPoint,
    SupportGeometry,
    build_support_geometry,
    logits_to_weights,
    optimize_regularization_path,
    support_distance,
    support_distance_batch,
)
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, load_scale, metric_row  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "support_radius_regularization_3e18_20260727"
REGULARIZATION_VALUES = (
    0.0,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.128,
    0.256,
    0.512,
)
NORMALIZED_RADII = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 4.00)
CALIBRATION_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf)
TARGETS = (UNCHEATABLE, TABLE9)
MODELS = ("content_krr", "effective_exposure_dsp")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-count", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-continuous", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def krr_loss_and_gradient(fit: krr.KernelFit, logits: np.ndarray) -> tuple[float, np.ndarray]:
    """Content-KRR prediction and analytic gradient in full phase logits."""
    bucket_count = fit.basis.shape[0]
    weights = logits_to_weights(logits, bucket_count)
    histograms = np.clip(weights @ fit.basis, 1e-15, None)
    phase_sqrt = np.sqrt(histograms)
    feature = (np.sqrt(fit.phase_fractions)[:, None] * phase_sqrt).reshape(1, -1)
    distance = krr.squared_hellinger(feature, fit.train_sqrt_features)[0]
    kernel = np.exp(-fit.gamma * distance)
    predicted = fit.target_mean + float(kernel @ fit.dual)

    weighted_dual = fit.dual * kernel
    train_phase_sqrt = np.sqrt(np.clip(fit.train_phase_histograms, 0.0, None))
    gradient_weights = np.empty_like(weights)
    for phase in range(2):
        signal = weighted_dual @ train_phase_sqrt[:, phase]
        histogram_gradient = signal / phase_sqrt[phase]
        gradient_weights[phase] = 0.5 * fit.gamma * fit.phase_fractions[phase] * (fit.basis @ histogram_gradient)
    gradient_logits = weights * (gradient_weights - np.sum(gradient_weights * weights, axis=1, keepdims=True))
    return predicted, gradient_logits.reshape(-1)


def predict_model(
    model_name: str,
    model: krr.KernelFit | dsp.FittedDSPModel,
    weights: np.ndarray,
) -> np.ndarray:
    """Predict a batch of phase policies with either frozen surrogate."""
    if model_name == "content_krr":
        assert isinstance(model, krr.KernelFit)
        predicted, _, _ = krr.predict_weights(model, weights[:, 0], weights[:, 1])
        return predicted
    assert isinstance(model, dsp.FittedDSPModel)
    return dsp.predict(model, weights)


def loss_function(
    model_name: str,
    model: krr.KernelFit | dsp.FittedDSPModel,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    if model_name == "content_krr":
        assert isinstance(model, krr.KernelFit)
        return lambda logits: krr_loss_and_gradient(model, logits)
    assert isinstance(model, dsp.FittedDSPModel)
    return lambda logits: dsp.value_grad_logits(model, logits)


def candidate_bank(panel: Panel, target: str, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Use one common proposal bank for every model and distance geometry."""
    phase0, phase1, kind = krr.sample_candidate_bank(panel, target, count, seed)
    return np.stack([phase0, phase1], axis=1).astype(float), kind


def unique_start_weights(
    bank_weights: np.ndarray,
    predicted: np.ndarray,
    distances: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Collect path-relevant starts from the frozen bank."""
    indices: list[int] = []
    for regularization in REGULARIZATION_VALUES:
        score = predicted + regularization * distances / radius
        indices.extend(np.argsort(score)[:2].tolist())
    indices.extend(np.argsort(predicted)[:6].tolist())
    unique = list(dict.fromkeys(indices))
    return bank_weights[np.asarray(unique[:24], dtype=int)]


def global_path_envelope(
    bank_predicted: np.ndarray,
    bank_distances: np.ndarray,
    bank_weights: np.ndarray,
    refined_points: list[PathPoint],
    geometry: SupportGeometry,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Take the exact lower envelope over every discovered policy."""
    pool_predicted = np.asarray(bank_predicted, dtype=float)
    pool_distances = np.asarray(bank_distances, dtype=float)
    pool_weights = list(np.asarray(bank_weights, dtype=float))
    pool_sources = ["candidate_bank"] * len(pool_weights)
    pool_success = [True] * len(pool_weights)
    pool_messages = ["frozen candidate bank"] * len(pool_weights)
    if refined_points:
        pool_predicted = np.concatenate([pool_predicted, np.asarray([point.surrogate_loss for point in refined_points])])
        pool_distances = np.concatenate(
            [pool_distances, np.asarray([point.support_distance for point in refined_points])]
        )
        pool_weights.extend(point.weights for point in refined_points)
        pool_sources.extend(["continuous_refinement"] * len(refined_points))
        pool_success.extend(point.optimizer_success for point in refined_points)
        pool_messages.extend(point.optimizer_message for point in refined_points)

    rows = []
    selected_weights = []
    for regularization in REGULARIZATION_VALUES:
        objective = pool_predicted + regularization * pool_distances / geometry.loo_radius_q95
        index = int(np.argmin(objective))
        _, nearest = support_distance(pool_weights[index], geometry)
        rows.append(
            {
                "regularization": regularization,
                "surrogate_loss": float(pool_predicted[index]),
                "support_distance": float(pool_distances[index]),
                "normalized_support_distance": float(pool_distances[index] / geometry.loo_radius_q95),
                "objective": float(objective[index]),
                "optimizer_success": pool_success[index],
                "optimizer_message": pool_messages[index],
                "nearest_fit_index": nearest,
                "path_source": pool_sources[index],
            }
        )
        selected_weights.append(pool_weights[index])
    frame = pd.DataFrame(rows)
    distance = frame["normalized_support_distance"].to_numpy(float)
    assert np.all(np.diff(distance) <= 1e-8), "global regularization path must move monotonically toward support"
    return frame, selected_weights


def add_policy_diagnostics(frame: pd.DataFrame, weights: list[np.ndarray], panel: Panel) -> pd.DataFrame:
    frame = frame.copy()
    frame["max_weight"] = [float(weight.max()) for weight in weights]
    frame["phase_tv"] = [float(0.5 * np.abs(weight[1] - weight[0]).sum()) for weight in weights]
    frame["max_epoch"] = [float((panel.c0 * weight[0] + panel.c1 * weight[1]).max()) for weight in weights]
    frame["nearest_fit_row"] = [
        str(panel.row_id[int(index)]) if np.isfinite(index) else ""
        for index in frame.get("nearest_fit_index", pd.Series([np.nan] * len(frame)))
    ]
    return frame


def heldout_selection_path(
    heldout: Panel,
    observed: np.ndarray,
    predicted: np.ndarray,
    distances: np.ndarray,
    radius: float,
) -> pd.DataFrame:
    rows = []
    best = float(observed.min())
    for regularization in REGULARIZATION_VALUES:
        score = predicted + regularization * distances / radius
        index = int(np.argmin(score))
        rows.append(
            {
                "regularization": regularization,
                "row_id": heldout.row_id[index],
                "series": heldout.series[index],
                "observed": float(observed[index]),
                "predicted": float(predicted[index]),
                "support_distance": float(distances[index]),
                "normalized_support_distance": float(distances[index] / radius),
                "observed_regret": float(observed[index] - best),
            }
        )
    return pd.DataFrame(rows)


def heldout_calibration(
    observed: np.ndarray,
    predicted: np.ndarray,
    normalized_distance: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for lower, upper in pairwise(CALIBRATION_BINS):
        mask = (normalized_distance >= lower) & (normalized_distance < upper)
        if int(mask.sum()) < 5:
            continue
        label = f"[{lower:g}, {upper:g})" if np.isfinite(upper) else f"[{lower:g}, inf)"
        rows.append(
            {
                "radius_bin": label,
                "radius_lower": lower,
                "radius_upper": upper,
                **metric_row(observed[mask], predicted[mask]),
            }
        )
    return pd.DataFrame(rows)


def hard_radius_frontier(
    predicted: np.ndarray,
    distances: np.ndarray,
    radius: float,
) -> pd.DataFrame:
    rows = []
    for normalized_radius in NORMALIZED_RADII:
        feasible = np.flatnonzero(distances <= normalized_radius * radius)
        if not len(feasible):
            continue
        index = int(feasible[np.argmin(predicted[feasible])])
        rows.append(
            {
                "normalized_radius": normalized_radius,
                "support_distance": float(distances[index]),
                "surrogate_loss": float(predicted[index]),
                "candidate_index": index,
            }
        )
    return pd.DataFrame(rows)


def plot_paths(path: pd.DataFrame, destination: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: predicted loss versus support",
            "Table-9: predicted loss versus support",
            "Uncheatable: heldout selected performance",
            "Table-9: heldout selected performance",
        ),
    )
    colors = {
        "content_krr": "#2166ac",
        "effective_exposure_dsp": "#d6604d",
    }
    dashes = {"content": "solid", "weight": "dash"}
    for target_index, target in enumerate(TARGETS, start=1):
        subset = path[path["target"] == target]
        for (model, geometry), group in subset.groupby(["model", "geometry"], sort=False):
            group = group.sort_values("regularization")
            name = f"{model} · {geometry}"
            figure.add_trace(
                go.Scatter(
                    x=group["normalized_support_distance"],
                    y=group["surrogate_loss"],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    line={"color": colors[model], "dash": dashes[geometry]},
                    customdata=group[["regularization", "max_weight", "phase_tv", "max_epoch"]],
                    hovertemplate=(
                        "radius: %{x:.3f}<br>predicted: %{y:.6f}<br>"
                        "lambda: %{customdata[0]:.6f}<br>max weight: %{customdata[1]:.3f}<br>"
                        "phase TV: %{customdata[2]:.3f}<br>max epoch: %{customdata[3]:.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=target_index,
            )
            figure.add_trace(
                go.Scatter(
                    x=group["regularization"].map(lambda value: f"{value:g}"),
                    y=group["heldout_selected_observed"],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    line={"color": colors[model], "dash": dashes[geometry]},
                    customdata=group[["heldout_selected_regret", "heldout_selected_radius"]],
                    hovertemplate=(
                        "lambda: %{x:.6f}<br>observed: %{y:.6f}<br>"
                        "archive regret: %{customdata[0]:.6f}<br>"
                        "radius: %{customdata[1]:.3f}<extra></extra>"
                    ),
                ),
                row=2,
                col=target_index,
            )
    figure.update_xaxes(title_text="Nearest-swarm distance / fit LOO q95", row=1)
    figure.update_yaxes(title_text="Predicted BPB", row=1)
    figure.update_xaxes(title_text="Regularization lambda (BPB per q95 radius)", type="category", row=2)
    figure.update_yaxes(title_text="Observed heldout BPB selected offline", row=2)
    figure.update_layout(
        title="Support-radius regularization paths",
        template="plotly_white",
        height=940,
        width=1500,
        hovermode="closest",
    )
    figure.write_html(destination, include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    destination: Path,
    path: pd.DataFrame,
    calibration: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    rows = [
        "# Support-radius regularization at Delphi 3e18",
        "",
        "## Definition",
        "",
        r"\(w(\lambda)=\arg\min_w \widehat L(w)+\lambda D(w,\mathcal S)/r_{95}\).",
        "",
        "This is distance to the nearest observed fit policy, not KL to proportional or a phase-asymmetry charge.",
        "The 3e18 heldout archive is exposed development evidence; these results are not confirmatory.",
        "",
        "## Endpoint summary",
        "",
        (
            "| Model | Target | Geometry | lambda=0 radius | strongest-lambda radius | "
            "heldout regret at lambda=0 | best heldout regret on path |"
        ),
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (model, target, geometry), group in path.groupby(["model", "target", "geometry"], sort=False):
        group = group.sort_values("regularization")
        rows.append(
            f"| {model} | {target} | {geometry} | "
            f"{group.iloc[0]['normalized_support_distance']:.3f} | "
            f"{group.iloc[-1]['normalized_support_distance']:.3f} | "
            f"{group.iloc[0]['heldout_selected_regret']:.6f} | "
            f"{group['heldout_selected_regret'].min():.6f} |"
        )
    rows.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A decreasing radius with increasing lambda verifies that the penalty controls extrapolation.",
            (
                "- Better archive selection along the path is development evidence for the regularizer, "
                "not evidence that the surrogate form is correct."
            ),
            (
                "- Content and weight geometries are compared as preregistered alternatives; "
                "no post-hoc output calibration is used."
            ),
            (
                "- The hard-radius frontier and calibration bins must be inspected before selecting "
                "policies for new training."
            ),
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, indent=2, sort_keys=True),
            "```",
            "",
            f"Calibration rows: {len(calibration)}.",
        ]
    )
    destination.write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_panel, heldout_all = load_scale("delphi_3e18")
    heldout = heldout_all.subset(heldout_all.policy_class == "two_phase")
    heldout = krr.remove_fit_aliases(fit_panel, heldout)
    phase_fractions = np.asarray([fit_panel.alpha, 1.0 - fit_panel.alpha])
    content_basis, basis_provenance = krr.load_embedding_basis(
        fit_panel.buckets,
        krr.DEFAULT_HISTOGRAM_DIR,
        krr.DEFAULT_LOOKUP,
    )
    train_weights = np.stack([fit_panel.phase0, fit_panel.phase1], axis=1)
    geometries = {
        "content": build_support_geometry(train_weights, content_basis, phase_fractions),
        "weight": build_support_geometry(train_weights, np.eye(len(fit_panel.buckets)), phase_fractions),
    }

    models: dict[tuple[str, str], krr.KernelFit | dsp.FittedDSPModel] = {}
    for target in TARGETS:
        models[("content_krr", target)] = krr.fit_kernel_model(
            fit_panel,
            content_basis,
            "content",
            target,
            args.seed,
        )
        models[("effective_exposure_dsp", target)] = fit_dsp_once(fit_panel, target)[0]

    path_frames = []
    calibration_frames = []
    hard_radius_frames = []
    weight_rows = []
    for target_index, target in enumerate(TARGETS):
        bank_weights, bank_kind = candidate_bank(
            fit_panel,
            target,
            args.candidate_count,
            args.seed + 1000 * target_index,
        )
        observed = heldout.targets[target]
        heldout_weights = np.stack([heldout.phase0, heldout.phase1], axis=1)
        for model_name in MODELS:
            model = models[(model_name, target)]
            bank_predicted = predict_model(model_name, model, bank_weights)
            heldout_predicted = predict_model(model_name, model, heldout_weights)
            for geometry_name, geometry in geometries.items():
                bank_distances = support_distance_batch(bank_weights, geometry)
                heldout_distances = support_distance_batch(heldout_weights, geometry)
                refined_points: list[PathPoint] = []
                if geometry_name == "content" and not args.skip_continuous:
                    starts = unique_start_weights(
                        bank_weights,
                        bank_predicted,
                        bank_distances,
                        geometry.loo_radius_q95,
                    )
                    refined_points = optimize_regularization_path(
                        loss_function(model_name, model),
                        geometry,
                        REGULARIZATION_VALUES,
                        starts,
                    )

                selected, selected_weights = global_path_envelope(
                    bank_predicted,
                    bank_distances,
                    bank_weights,
                    refined_points,
                    geometry,
                )
                selected = add_policy_diagnostics(selected, selected_weights, fit_panel)
                archive_path = heldout_selection_path(
                    heldout,
                    observed,
                    heldout_predicted,
                    heldout_distances,
                    geometry.loo_radius_q95,
                )
                selected["heldout_selected_observed"] = archive_path["observed"]
                selected["heldout_selected_predicted"] = archive_path["predicted"]
                selected["heldout_selected_regret"] = archive_path["observed_regret"]
                selected["heldout_selected_radius"] = archive_path["normalized_support_distance"]
                selected["heldout_selected_row_id"] = archive_path["row_id"]
                selected["heldout_selected_series"] = archive_path["series"]
                selected.insert(0, "geometry", geometry_name)
                selected.insert(0, "target", target)
                selected.insert(0, "model", model_name)
                path_frames.append(selected)

                for path_index, weights in enumerate(selected_weights):
                    for phase in range(2):
                        for bucket_index, bucket in enumerate(fit_panel.buckets):
                            weight_rows.append(
                                {
                                    "model": model_name,
                                    "target": target,
                                    "geometry": geometry_name,
                                    "regularization": selected.iloc[path_index]["regularization"],
                                    "phase": phase,
                                    "bucket": bucket,
                                    "weight": float(weights[phase, bucket_index]),
                                }
                            )

                calibration = heldout_calibration(
                    observed,
                    heldout_predicted,
                    heldout_distances / geometry.loo_radius_q95,
                )
                calibration.insert(0, "geometry", geometry_name)
                calibration.insert(0, "target", target)
                calibration.insert(0, "model", model_name)
                calibration_frames.append(calibration)

                hard = hard_radius_frontier(
                    bank_predicted,
                    bank_distances,
                    geometry.loo_radius_q95,
                )
                hard["bank_kind"] = bank_kind[hard["candidate_index"].to_numpy(int)]
                hard.insert(0, "geometry", geometry_name)
                hard.insert(0, "target", target)
                hard.insert(0, "model", model_name)
                hard_radius_frames.append(hard)

    path = pd.concat(path_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    hard_radius = pd.concat(hard_radius_frames, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    path.to_csv(args.output_dir / "regularization_path.csv", index=False)
    calibration.to_csv(args.output_dir / "heldout_calibration_by_radius.csv", index=False)
    hard_radius.to_csv(args.output_dir / "hard_radius_frontier.csv", index=False)
    weights.to_csv(args.output_dir / "path_mixture_weights.csv", index=False)
    plot_paths(path, args.output_dir / "support_radius_paths.html")

    provenance = {
        "fit_panel_sha256": sha256(krr.CANONICAL / "delphi_3e18_two_phase_fit.csv"),
        "heldout_registry_sha256": sha256(
            SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
        ),
        "fit_rows": len(fit_panel),
        "heldout_rows": len(heldout),
        "targets": TARGETS,
        "models": MODELS,
        "primary_geometry": "content Hellinger",
        "ablation_geometry": "weight Hellinger",
        "content_basis": basis_provenance,
        "regularization_values": REGULARIZATION_VALUES,
        "normalized_radii": NORMALIZED_RADII,
        "candidate_count": args.candidate_count,
        "seed": args.seed,
        "continuous_refinement": not args.skip_continuous,
        "data_use": "existing coordinate-disjoint 3e18 archive is exposed development evidence",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(args.output_dir / "report.md", path, calibration, provenance)

    print(
        path[
            [
                "model",
                "target",
                "geometry",
                "regularization",
                "surrogate_loss",
                "normalized_support_distance",
                "heldout_selected_observed",
                "heldout_selected_regret",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
