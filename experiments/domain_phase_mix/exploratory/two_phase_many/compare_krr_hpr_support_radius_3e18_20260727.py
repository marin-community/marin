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
"""Compare KRR and HPR under one shared distance-to-swarm objective.

Both surrogates are fit on the same 280-row Delphi 3e18 two-phase panel, scored
on the same candidate bank, and regularized by the same content-Hellinger
nearest-swarm distance. This isolates the surrogate from proposal and solver
differences. Existing 3e18 heldouts are exposed development evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
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
import benchmark_partially_pooled_phase_bowls as pooled  # noqa: E402
import evaluate_support_radius_regularization_3e18_20260727 as support_eval  # noqa: E402
import export_mixture_fit_observatory as observatory  # noqa: E402
import fit_production_grp_quality_variants as family_grp  # noqa: E402
from support_radius_regularization import SupportGeometry, build_support_geometry, support_distance_batch  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, load_scale  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "krr_hpr_support_radius_comparison_3e18_20260727"
HPR_CACHE = SCRIPT_DIR / "reference_outputs" / "mixture_fit_observatory_cache_20260713" / "delphi_3e18"
TARGET_CACHE_NAMES = {UNCHEATABLE: "uncheatable", TABLE9: "table9"}
MODELS = ("content_krr", "hierarchical_phase_replay")
COLORS = {"content_krr": "#1a9850", "hierarchical_phase_replay": "#d73027"}
LABELS = {"content_krr": "Content-Hellinger KRR", "hierarchical_phase_replay": "Hierarchical phase replay"}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-count", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def hpr_dataset(panel: Panel, target: str) -> pooled.Dataset:
    """Convert the canonical fit panel to the exact Observatory input type."""
    frame = pd.read_csv(krr.CANONICAL / "delphi_3e18_two_phase_fit.csv")
    if len(frame) != len(panel):
        raise ValueError(f"Expected {len(panel)} canonical fit rows, found {len(frame)}")
    if tuple(frame["row_id"].astype(str)) != tuple(panel.row_id.astype(str)):
        raise ValueError("Canonical fit row order differs from the loaded Delphi panel")
    return pooled.Dataset(
        name=f"delphi_3e18_{target}",
        frame=frame,
        y=np.asarray(panel.targets[target], dtype=float),
        weights=np.stack([panel.phase0, panel.phase1], axis=1),
        c0=np.asarray(panel.c0, dtype=float),
        c1=np.asarray(panel.c1, dtype=float),
        domain_names=list(panel.buckets),
    )


def selected_hpr_config(target: str) -> tuple[hierarchical.Config, dict[str, object]]:
    """Load the frozen Observatory-selected HPR configuration."""
    cache_path = HPR_CACHE / TARGET_CACHE_NAMES[target] / "two_phase" / "hierarchical_phase_bucket_replay.json"
    cached = json.loads(cache_path.read_text())
    detail = cached["fitDetail"]
    tuning = detail["tuning"]
    shape = tuning["shapeParameters"]
    config = hierarchical.Config(
        variant=hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        shape_index=-1,
        shape=family_grp.Shape(
            exponent=float(shape["exponent"]),
            late_multiplier=float(shape["lateMultiplier"]),
            forgetting_rate=float(shape["forgettingRate"]),
            penalty_threshold=float(shape["penaltyThreshold"]),
            quality_discount=1.0,
        ),
        l2=float(tuning["l2"]),
        residual_shrink=float(tuning["residualShrink"]),
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )
    return config, {
        "cache_path": str(cache_path.relative_to(REPO_ROOT)),
        "cache_fingerprint": cached["fingerprint"],
        "parameter_count": int(detail["parameterCount"]),
        "shape": shape,
        "l2": float(tuning["l2"]),
        "residual_shrink": float(tuning["residualShrink"]),
    }


def fit_hpr(panel: Panel, target: str) -> tuple[hierarchical.Model, dict[str, object]]:
    dataset = hpr_dataset(panel, target)
    config, provenance = selected_hpr_config(target)
    structured = observatory.family_dataset(dataset)
    model = hierarchical.fit_model(structured, config, np.arange(dataset.n))

    detail = json.loads(
        (HPR_CACHE / TARGET_CACHE_NAMES[target] / "two_phase" / "hierarchical_phase_bucket_replay.json").read_text()
    )["fitDetail"]
    expected = {row["key"]: float(row["value"]) for row in detail["parameters"] if row["key"] != "intercept"}
    design = hierarchical.build_design(structured, config)
    fitted = dict(zip(design.names, model.coefficients, strict=True))
    shared = sorted(set(expected) & set(fitted))
    if not shared:
        raise ValueError("No HPR coefficients matched the frozen Observatory fit")
    maximum_difference = max(abs(expected[name] - fitted[name]) for name in shared)
    expected_intercept = next(float(row["value"]) for row in detail["parameters"] if row["key"] == "intercept")
    intercept_difference = abs(expected_intercept - model.intercept)
    provenance["refit_max_coefficient_difference"] = maximum_difference
    provenance["refit_intercept_difference"] = intercept_difference
    return model, provenance


def predict(
    model_name: str,
    model: krr.KernelFit | hierarchical.Model,
    weights: np.ndarray,
) -> np.ndarray:
    if model_name == "content_krr":
        assert isinstance(model, krr.KernelFit)
        predicted, _, _ = krr.predict_weights(model, weights[:, 0], weights[:, 1])
        return predicted
    assert isinstance(model, hierarchical.Model)
    return model.predict(weights)


def shared_bank_path(
    model_name: str,
    target: str,
    model: krr.KernelFit | hierarchical.Model,
    bank_weights: np.ndarray,
    bank_kind: np.ndarray,
    bank_distances: np.ndarray,
    heldout: Panel,
    heldout_weights: np.ndarray,
    heldout_distances: np.ndarray,
    geometry: SupportGeometry,
    fit_panel: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Take the exact path over one frozen bank shared by both models."""
    bank_predicted = predict(model_name, model, bank_weights)
    heldout_predicted = predict(model_name, model, heldout_weights)
    selected, selected_weights = support_eval.global_path_envelope(
        bank_predicted,
        bank_distances,
        bank_weights,
        [],
        geometry,
    )
    selected = support_eval.add_policy_diagnostics(selected, selected_weights, fit_panel)
    archive = support_eval.heldout_selection_path(
        heldout,
        heldout.targets[target],
        heldout_predicted,
        heldout_distances,
        geometry.loo_radius_q95,
    )
    selected["heldout_selected_observed"] = archive["observed"]
    selected["heldout_selected_predicted"] = archive["predicted"]
    selected["heldout_selected_regret"] = archive["observed_regret"]
    selected["heldout_selected_radius"] = archive["normalized_support_distance"]
    selected["heldout_selected_row_id"] = archive["row_id"]
    selected["heldout_selected_series"] = archive["series"]
    selected["bank_kind"] = [
        bank_kind[int(np.argmin(bank_predicted + value * bank_distances / geometry.loo_radius_q95))]
        for value in selected["regularization"]
    ]
    selected.insert(0, "target", target)
    selected.insert(0, "model", model_name)

    calibration = support_eval.heldout_calibration(
        heldout.targets[target],
        heldout_predicted,
        heldout_distances / geometry.loo_radius_q95,
    )
    calibration.insert(0, "target", target)
    calibration.insert(0, "model", model_name)
    return selected, calibration, selected_weights


def plot_comparison(path: pd.DataFrame, destination: Path) -> None:
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Uncheatable: predicted support frontier",
            "Table-9: predicted support frontier",
            "Uncheatable: radius versus regularization",
            "Table-9: radius versus regularization",
            "Uncheatable: exposed archive selection regret",
            "Table-9: exposed archive selection regret",
        ),
        vertical_spacing=0.09,
    )
    for column, target in enumerate((UNCHEATABLE, TABLE9), start=1):
        for model_name in MODELS:
            group = path[(path["target"] == target) & (path["model"] == model_name)].sort_values("regularization")
            custom = group[["regularization", "max_weight", "phase_tv", "max_epoch"]]
            common = {
                "mode": "lines+markers",
                "name": LABELS[model_name],
                "legendgroup": model_name,
                "line": {"color": COLORS[model_name], "width": 3},
                "marker": {"size": 8},
            }
            figure.add_trace(
                go.Scatter(
                    x=group["normalized_support_distance"],
                    y=group["surrogate_loss"],
                    customdata=custom,
                    hovertemplate=(
                        "radius: %{x:.3f}<br>predicted BPB: %{y:.6f}<br>"
                        "lambda: %{customdata[0]:.6f}<br>max weight: %{customdata[1]:.3f}<br>"
                        "phase TV: %{customdata[2]:.3f}<br>max epoch: %{customdata[3]:.2f}<extra></extra>"
                    ),
                    showlegend=column == 1,
                    **common,
                ),
                row=1,
                col=column,
            )
            lambda_label = group["regularization"].map(lambda value: f"{value:g}")
            figure.add_trace(
                go.Scatter(
                    x=lambda_label,
                    y=group["normalized_support_distance"],
                    customdata=group[["surrogate_loss"]],
                    hovertemplate=(
                        "lambda: %{x}<br>radius: %{y:.3f}<br>" "predicted BPB: %{customdata[0]:.6f}<extra></extra>"
                    ),
                    showlegend=False,
                    **common,
                ),
                row=2,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=lambda_label,
                    y=group["heldout_selected_regret"],
                    customdata=group[["heldout_selected_observed", "heldout_selected_radius"]],
                    hovertemplate=(
                        "lambda: %{x}<br>archive regret: %{y:.6f}<br>"
                        "observed BPB: %{customdata[0]:.6f}<br>"
                        "selected radius: %{customdata[1]:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                    **common,
                ),
                row=3,
                col=column,
            )
    figure.update_xaxes(title_text="Nearest-swarm distance / fit LOO q95", row=1)
    figure.update_yaxes(title_text="Predicted BPB", row=1)
    figure.update_xaxes(title_text="Regularization lambda", type="category", row=2)
    figure.update_yaxes(title_text="Normalized support radius", row=2)
    figure.update_xaxes(title_text="Regularization lambda", type="category", row=3)
    figure.update_yaxes(title_text="Observed selection regret (BPB)", row=3)
    figure.update_layout(
        title={
            "text": (
                "KRR versus HPR under an identical content-support objective"
                "<br><sup>Same 280 fit rows, 60k candidate bank, and lambda grid; "
                "heldout outcomes are exposed development evidence</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=1320,
        width=1500,
        hovermode="closest",
        margin={"t": 120, "r": 190},
        legend={"orientation": "v", "y": 1.0, "x": 1.01, "xanchor": "left"},
    )
    figure.write_html(destination, include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    destination: Path,
    path: pd.DataFrame,
    hpr_provenance: dict[str, dict[str, object]],
    radius: float,
    candidate_count: int,
) -> None:
    rows = [
        "# KRR versus HPR under support-radius regularization",
        "",
        f"Both models use the same 280-row Delphi 3e18 fit panel, {candidate_count:,}-policy candidate bank, ",
        "content-Hellinger nearest-swarm distance, and frozen regularization grid. No continuous ",
        "refinement is used, so model comparison is not confounded by optimizer gradients.",
        "",
        f"Fit-panel leave-one-out q95 content-Hellinger radius: `{radius:.8f}`.",
        "",
        "| Model | Target | Raw radius | Raw predicted | Best archive regret | Lambda at best regret |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (model, target), group in path.groupby(["model", "target"], sort=False):
        group = group.sort_values("regularization")
        best = group.loc[group["heldout_selected_regret"].idxmin()]
        rows.append(
            f"| {LABELS[model]} | {target} | {group.iloc[0]['normalized_support_distance']:.3f} | "
            f"{group.iloc[0]['surrogate_loss']:.6f} | {best['heldout_selected_regret']:.6f} | "
            f"{best['regularization']:.6g} |"
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The upper row compares each surrogate's predicted loss-support tradeoff.",
            "- The middle row verifies how quickly the selected policy returns toward empirical support.",
            "- The lower row is an exposed historical archive diagnostic, not confirmatory evidence.",
            "- A support penalty controls where the model proposes; it does not calibrate either model's BPB output.",
            "",
            "## Frozen HPR fits",
            "",
            "```json",
            json.dumps(hpr_provenance, indent=2, sort_keys=True),
            "```",
        ]
    )
    destination.write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_panel, heldout_all = load_scale("delphi_3e18")
    heldout = heldout_all.subset(heldout_all.policy_class == "two_phase")
    heldout = krr.remove_fit_aliases(fit_panel, heldout)
    content_basis, basis_provenance = krr.load_embedding_basis(
        fit_panel.buckets,
        krr.DEFAULT_HISTOGRAM_DIR,
        krr.DEFAULT_LOOKUP,
    )
    geometry = build_support_geometry(
        np.stack([fit_panel.phase0, fit_panel.phase1], axis=1),
        content_basis,
        np.asarray([fit_panel.alpha, 1.0 - fit_panel.alpha]),
    )
    heldout_weights = np.stack([heldout.phase0, heldout.phase1], axis=1)
    heldout_distances = support_distance_batch(heldout_weights, geometry)

    path_frames = []
    calibration_frames = []
    weight_rows = []
    hpr_provenance: dict[str, dict[str, object]] = {}
    for target_index, target in enumerate((UNCHEATABLE, TABLE9)):
        bank_weights, bank_kind = support_eval.candidate_bank(
            fit_panel,
            target,
            args.candidate_count,
            args.seed + 1000 * target_index,
        )
        bank_distances = support_distance_batch(bank_weights, geometry)
        models: dict[str, krr.KernelFit | hierarchical.Model] = {
            "content_krr": krr.fit_kernel_model(fit_panel, content_basis, "content", target, args.seed),
        }
        models["hierarchical_phase_replay"], hpr_provenance[target] = fit_hpr(fit_panel, target)

        for model_name, model in models.items():
            selected, calibration, selected_weights = shared_bank_path(
                model_name,
                target,
                model,
                bank_weights,
                bank_kind,
                bank_distances,
                heldout,
                heldout_weights,
                heldout_distances,
                geometry,
                fit_panel,
            )
            path_frames.append(selected)
            calibration_frames.append(calibration)
            for path_index, weights in enumerate(selected_weights):
                for phase in range(2):
                    for bucket_index, bucket in enumerate(fit_panel.buckets):
                        weight_rows.append(
                            {
                                "model": model_name,
                                "target": target,
                                "regularization": selected.iloc[path_index]["regularization"],
                                "phase": phase,
                                "bucket": bucket,
                                "weight": float(weights[phase, bucket_index]),
                            }
                        )

    path = pd.concat(path_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    path.to_csv(args.output_dir / "regularization_path.csv", index=False)
    calibration.to_csv(args.output_dir / "heldout_calibration_by_radius.csv", index=False)
    weights.to_csv(args.output_dir / "path_mixture_weights.csv", index=False)
    plot_comparison(path, args.output_dir / "krr_hpr_support_radius.html")
    write_report(
        args.output_dir / "report.md",
        path,
        hpr_provenance,
        geometry.loo_radius_q95,
        args.candidate_count,
    )
    provenance = {
        "fit_rows": len(fit_panel),
        "heldout_rows": len(heldout),
        "candidate_count": args.candidate_count,
        "seed": args.seed,
        "models": MODELS,
        "geometry": "phase-weighted content-Hellinger nearest-swarm distance",
        "loo_radius_q95": geometry.loo_radius_q95,
        "regularization_values": support_eval.REGULARIZATION_VALUES,
        "content_basis": basis_provenance,
        "continuous_refinement": False,
        "data_use": "existing coordinate-disjoint 3e18 archive is exposed development evidence",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        path[
            [
                "model",
                "target",
                "regularization",
                "surrogate_loss",
                "normalized_support_distance",
                "heldout_selected_regret",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
