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
"""Falsify homeostatic replicator capacity on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    homeostatic_replicator_models as replicator,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round7_replicator_starcoder"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[replicator.Config]:
    state_pairs = [(0.0, 0.0)] + [
        (selection, homeostasis) for selection in (0.03, 0.1, 0.3, 1.0, 3.0) for homeostasis in (0.0, 0.1, 0.3, 1.0, 3.0)
    ]
    return [
        replicator.Config(selection, homeostasis, replay, l2)
        for selection, homeostasis in state_pairs
        for replay in (0.5, 1.5, 3.0, 5.0)
        for l2 in (0.0, 0.01, 0.1, 1.0)
    ]


def fit_design(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    train_design = design[train]
    train_target = target[train]
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1], dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=50 * design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
    return intercept, coefficients


def oof_prediction(
    panel: paired.PairedPanel,
    config: replicator.Config,
) -> tuple[np.ndarray, np.ndarray]:
    design, _names = replicator.build_design(panel, panel.weights, config)
    prediction = np.full(panel.n, np.nan, dtype=float)
    active = []
    for train, test in starcoder.surface_folds(panel):
        intercept, coefficients = fit_design(design, panel.two_phase_target, train, config.l2)
        prediction[test] = intercept + design[test] @ coefficients
        active.append(int(np.sum(coefficients > 1e-10)))
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {panel.name} {config.key}")
    return prediction, np.asarray(active, dtype=int)


def select_config(panel: paired.PairedPanel) -> tuple[replicator.Config, pd.DataFrame, np.ndarray]:
    rows = []
    predictions = {}
    for config in configs():
        prediction, active = oof_prediction(panel, config)
        predictions[config.key] = prediction
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "mean_active_coefficients": float(np.mean(active)),
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    selected = replicator.Config(**json.loads(table.iloc[0]["config_json"]))
    return selected, table, predictions[selected.key]


def leave_region_out(
    panel: paired.PairedPanel,
    config: replicator.Config,
) -> list[dict[str, Any]]:
    design, _names = replicator.build_design(panel, panel.weights, config)
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    rows = []
    for region, mask in {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
    }.items():
        train = np.flatnonzero(~mask)
        test = np.flatnonzero(mask)
        if len(test) < 3:
            continue
        intercept, coefficients = fit_design(design, panel.two_phase_target, train, config.l2)
        prediction = intercept + design[test] @ coefficients
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **paired_screen.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def tied_restriction_audit(
    panel: paired.PairedPanel,
    config: replicator.Config,
    algebraic_oof: np.ndarray,
) -> dict[str, Any]:
    tied = np.flatnonzero(panel.paired_mask)
    design, _names = replicator.build_design(panel, panel.weights, config)
    refit = np.full(panel.n, np.nan, dtype=float)
    for test in tied:
        train = tied[tied != test]
        intercept, coefficients = fit_design(design, panel.two_phase_target, train, config.l2)
        refit[test] = intercept + design[test] @ coefficients
    observed = panel.two_phase_target[tied]
    return {
        "surface": panel.name,
        "n_tied": len(tied),
        "algebraic_tied_oof_rmse": paired_screen.scalar_metrics(observed, algebraic_oof[tied])["rmse"],
        "independent_tied_loocv_rmse": paired_screen.scalar_metrics(observed, refit[tied])["rmse"],
        "median_prediction_disagreement": float(np.median(np.abs(algebraic_oof[tied] - refit[tied]))),
    }


def optimum(
    panel: paired.PairedPanel,
    config: replicator.Config,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = replicator.fit_model(panel, np.arange(panel.n), config)
    grid = np.linspace(0.0, 1.0, 201)
    rare0, rare1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - rare0.ravel(), rare0.ravel()]),
            np.column_stack([1.0 - rare1.ravel(), rare1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    record = {
        "surface": panel.name,
        "phase0_rare": float(rare0.ravel()[best]),
        "phase1_rare": float(rare1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                rare0.ravel()[best] - panel.weights[observed_best, 0, 1],
                rare1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
    }
    return record, pd.DataFrame(
        {"phase0_rare": rare0.ravel(), "phase1_rare": rare1.ravel(), "predicted_bpb": prediction}
    )


def render_surface(
    panel: paired.PairedPanel, surface: pd.DataFrame, optimum_record: dict[str, Any], output: Path
) -> None:
    grid_size = round(np.sqrt(len(surface)))
    grid = surface["phase0_rare"].to_numpy().reshape(grid_size, grid_size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(grid_size, grid_size)
    figure = go.Figure(
        [
            go.Surface(x=grid, y=grid, z=z.T, colorscale="RdYlGn_r", opacity=0.72, name="Predicted"),
            go.Scatter3d(
                x=panel.weights[:, 0, 1],
                y=panel.weights[:, 1, 1],
                z=panel.two_phase_target,
                mode="markers",
                marker={"size": 4, "color": panel.two_phase_target, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
            go.Scatter3d(
                x=[optimum_record["phase0_rare"]],
                y=[optimum_record["phase1_rare"]],
                z=[optimum_record["predicted_bpb"]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#111827"},
                name="Predicted optimum",
            ),
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: homeostatic replicator capacity",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    grids = []
    metrics = []
    predictions = []
    regions = []
    restrictions = []
    optima = []
    for panel in panels:
        config, grid, prediction = select_config(panel)
        grids.append(grid)
        selected_metrics = paired_screen.scalar_metrics(panel.two_phase_target, prediction)
        no_capacity = grid.loc[grid["config"].str.startswith("selection=0,")].sort_values("rmse").iloc[0]
        metrics.append(
            {
                "surface": panel.name,
                "selected_config": config.key,
                "tied_semigroup_error": replicator.tied_semigroup_error(panel, config),
                "no_capacity_rmse": float(no_capacity["rmse"]),
                "relative_rmse_vs_no_capacity": float(selected_metrics["rmse"] / no_capacity["rmse"] - 1.0),
                **selected_metrics,
            }
        )
        predictions.extend(
            {
                "surface": panel.name,
                "row_index": index,
                "phase_tied": bool(panel.paired_mask[index]),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for index, (observed, predicted) in enumerate(zip(panel.two_phase_target, prediction, strict=True))
        )
        regions.extend(leave_region_out(panel, config))
        restrictions.append(tied_restriction_audit(panel, config, prediction))
        optimum_record, surface = optimum(panel, config)
        optima.append(optimum_record)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, optimum_record, args.output_dir / f"{panel.name}__surface.html")

    grid_frame = pd.concat(grids, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    region_frame = pd.DataFrame(regions)
    restriction_frame = pd.DataFrame(restrictions)
    optimum_frame = pd.DataFrame(optima)
    grid_frame.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    metric_frame.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "surface_oof_predictions.csv", index=False)
    region_frame.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    restriction_frame.to_csv(args.output_dir / "one_phase_restriction_audit.csv", index=False)
    optimum_frame.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    report = [
        "# Round-seven homeostatic replicator capacity",
        "",
        "The latent state is a conserved representation allocation on the bucket simplex. Selection is multiplicative and homeostasis restores the proportional allocation. The exact nested ablation is selection rate zero.",
        "",
        "## Surface OOF",
        "",
        metric_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        region_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Predicted optima",
        "",
        optimum_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## One-phase restriction",
        "",
        restriction_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "No multi-swarm, historical, or adversarial outcome was read in this first gate.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metric_frame.to_string(index=False))
    print(region_frame.to_string(index=False))


if __name__ == "__main__":
    main()
