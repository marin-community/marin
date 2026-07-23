# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Audit early-allocation-gated plasticity on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    potential_phase_models as potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    primed_plasticity_models as primed,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round4_primed_plasticity_starcoder"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[primed.PrimerConfig]:
    result = []
    for primer_rate in (0.1, 0.3, 1.0, 3.0, 10.0):
        for residual_plasticity in (0.0, 0.1, 0.3, 0.6, 1.0):
            for response in (potential.DebtResponse.INVERSE_POWER, potential.DebtResponse.LOGARITHMIC):
                curvatures = (0.25, 0.5, 1.0) if response is potential.DebtResponse.INVERSE_POWER else (1.0,)
                for curvature in curvatures:
                    for offset in (0.03, 0.1, 0.3, 1.0):
                        for l2 in (0.01, 0.1, 1.0):
                            result.append(
                                primed.PrimerConfig(
                                    primer_rate,
                                    residual_plasticity,
                                    response,
                                    curvature,
                                    offset,
                                    l2,
                                )
                            )
    return result


def oof_prediction(panel: Any, config: primed.PrimerConfig) -> np.ndarray:
    prediction = np.full(panel.n, np.nan, dtype=float)
    geometry = primed.primer_geometry(panel)
    for train, test in starcoder.surface_folds(panel):
        model = primed.fit_primed_plasticity(geometry, panel.weights, panel.two_phase_target, train, config)
        prediction[test] = model.predict(panel.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete primer OOF prediction for {panel.name}")
    return prediction


def select_config(panel: Any) -> tuple[primed.PrimerConfig, np.ndarray, pd.DataFrame]:
    rows = []
    predictions = {}
    for config in configs():
        prediction = oof_prediction(panel, config)
        predictions[config.key] = prediction
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                "primer_rate": config.primer_rate,
                "residual_plasticity": config.residual_plasticity,
                "response": config.response.value,
                "curvature": config.curvature,
                "offset": config.offset,
                "l2": config.l2,
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    best = table.iloc[0]
    selected = primed.PrimerConfig(
        float(best["primer_rate"]),
        float(best["residual_plasticity"]),
        potential.DebtResponse(best["response"]),
        float(best["curvature"]),
        float(best["offset"]),
        float(best["l2"]),
    )
    return selected, predictions[selected.key], table


def one_phase_audit(panel: Any, config: primed.PrimerConfig) -> dict[str, Any]:
    tied = np.flatnonzero(panel.paired_mask)
    geometry = primed.primer_geometry(panel)
    refit_prediction = np.full(len(tied), np.nan, dtype=float)
    for local_test, test_index in enumerate(tied):
        train = tied[tied != test_index]
        model = primed.fit_primed_plasticity(geometry, panel.weights, panel.two_phase_target, train, config)
        refit_prediction[local_test] = model.predict(panel.weights[[test_index]])[0]
    two_phase_model = primed.fit_primed_plasticity(
        geometry,
        panel.weights,
        panel.two_phase_target,
        np.arange(panel.n),
        config,
    )
    restricted = two_phase_model.predict(panel.weights[tied])
    observed = panel.two_phase_target[tied]
    return {
        "surface": panel.name,
        "n_tied": len(tied),
        **{
            f"independent_refit_{key}": value
            for key, value in paired_screen.scalar_metrics(observed, refit_prediction).items()
        },
        **{
            f"algebraic_restriction_{key}": value
            for key, value in paired_screen.scalar_metrics(observed, restricted).items()
        },
        "median_restriction_disagreement": float(np.median(np.abs(restricted - refit_prediction))),
        "maximum_restriction_disagreement": float(np.max(np.abs(restricted - refit_prediction))),
    }


def leave_region_out(panel: Any, config: primed.PrimerConfig) -> list[dict[str, Any]]:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    geometry = primed.primer_geometry(panel)
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        model = primed.fit_primed_plasticity(geometry, panel.weights, panel.two_phase_target, train, config)
        prediction = model.predict(panel.weights[test])
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


def optimum_and_surface(
    panel: Any,
    config: primed.PrimerConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = primed.fit_primed_plasticity(
        primed.primer_geometry(panel),
        panel.weights,
        panel.two_phase_target,
        np.arange(panel.n),
        config,
    )
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
    surface = pd.DataFrame({"phase0_rare": rare0.ravel(), "phase1_rare": rare1.ravel(), "predicted_bpb": prediction})
    return record, surface


def render_surface(panel: Any, surface: pd.DataFrame, optimum: dict[str, Any], output: Path) -> None:
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
                x=[optimum["phase0_rare"]],
                y=[optimum["phase1_rare"]],
                z=[optimum["predicted_bpb"]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#111827"},
                name="Predicted optimum",
            ),
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: early-allocation-gated plasticity",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    grid_tables = []
    metric_rows = []
    one_phase_rows = []
    region_rows = []
    optimum_rows = []
    for panel in panels:
        config, prediction, grid = select_config(panel)
        grid_tables.append(grid)
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": config.key,
                "parameter_count": len(primed.primed_design(primed.primer_geometry(panel), panel.weights[:1], config)[1])
                + 3,
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
        one_phase_rows.append(one_phase_audit(panel, config))
        region_rows.extend(leave_region_out(panel, config))
        optimum, surface = optimum_and_surface(panel, config)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, optimum, args.output_dir / f"{panel.name}__surface.html")

    grid = pd.concat(grid_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    one_phase = pd.DataFrame(one_phase_rows)
    regions = pd.DataFrame(region_rows)
    optima = pd.DataFrame(optimum_rows)
    grid.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    metrics.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    one_phase.to_csv(args.output_dir / "one_phase_restriction_audit.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    report = [
        "# Early-allocation-gated plasticity: StarCoder audit",
        "",
        "All configurations were selected by full-surface OOF. The phase-tied restriction was also refitted independently using only phase-tied rows. No Delphi historical or adversarial outcome was read.",
        "",
        "## Surface OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## One-phase restriction",
        "",
        one_phase.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(metrics.to_string(index=False))
    print("\nOne-phase")
    print(one_phase.to_string(index=False))
    print("\nLeave-region-out")
    print(regions.to_string(index=False))
    print("\nOptima")
    print(optima.to_string(index=False))


if __name__ == "__main__":
    main()
