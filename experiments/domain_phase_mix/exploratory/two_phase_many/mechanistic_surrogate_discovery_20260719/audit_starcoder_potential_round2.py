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
"""Falsify convex-potential phase laws on dense StarCoder surfaces."""

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
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    potential_phase_models as potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_potential_phase_round2 as potential_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round2_potential_starcoder"
)
SEED = 20260719
N_SPLITS = 5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_from_dataset(dataset: Any) -> paired.PairedPanel:
    tied = np.max(np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]), axis=1) < 1e-10
    one_phase = np.full(dataset.n, np.nan, dtype=float)
    one_phase[tied] = dataset.y[tied]
    return paired.PairedPanel(
        name=dataset.name,
        target="starcoder_bpb",
        frame=dataset.frame.copy(),
        domain_names=tuple(dataset.domain_names),
        family_names=tuple(dataset.domain_names),
        family_members=tuple(np.asarray([index], dtype=int) for index in range(dataset.m)),
        weights=np.asarray(dataset.weights, dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        two_phase_target=np.asarray(dataset.y, dtype=float),
        one_phase_target=one_phase,
    )


def load_refined_wsd80(cosine: Any) -> Any:
    """Load the persisted 107-coordinate WSD panel used by discovery audits."""

    return starcoder_refined_data.load_refined_wsd80_starcoder(cosine)


def geometry(panel: paired.PairedPanel) -> potential.PotentialGeometry:
    return potential.PotentialGeometry(
        domain_names=panel.domain_names,
        family_names=panel.family_names,
        family_members=panel.family_members,
        proportional_weights=panel.proportional_weights,
        total_epoch_coefficients=panel.c0 + panel.c1,
    )


def surface_folds(panel: paired.PairedPanel) -> list[tuple[np.ndarray, np.ndarray]]:
    tied = np.flatnonzero(panel.paired_mask)
    untied = np.flatnonzero(~panel.paired_mask)
    tied_folds = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(tied))
    untied_folds = list(KFold(N_SPLITS, shuffle=True, random_state=SEED + 1).split(untied))
    result = []
    for fold in range(N_SPLITS):
        tied_train, tied_test = tied_folds[fold]
        untied_train, untied_test = untied_folds[fold]
        train = np.concatenate([tied[tied_train], untied[untied_train]])
        test = np.concatenate([tied[tied_test], untied[untied_test]])
        result.append((np.sort(train), np.sort(test)))
    return result


def tied_loocv(panel: paired.PairedPanel, config: potential.PotentialConfig) -> np.ndarray:
    tied = np.flatnonzero(panel.paired_mask)
    prediction = np.full(panel.n, np.nan, dtype=float)
    geom = geometry(panel)
    for test_index in tied:
        train = tied[tied != test_index]
        model = potential.fit_potential(
            geom,
            panel.aggregate_weights,
            panel.one_phase_target,
            train,
            config,
        )
        prediction[test_index] = model.predict(panel.aggregate_weights[[test_index]])[0]
    return prediction


def select_potential(panel: paired.PairedPanel) -> tuple[potential.PotentialConfig, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tied = panel.paired_mask
    for config in potential_screen.potential_configs():
        prediction = tied_loocv(panel, config)
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                "config_json": json.dumps({**asdict(config), "response": config.response.value}, sort_keys=True),
                **paired_screen.scalar_metrics(panel.two_phase_target[tied], prediction[tied]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    values = json.loads(table.iloc[0]["config_json"])
    values["response"] = potential.DebtResponse(values["response"])
    return potential.PotentialConfig(**values), table


def phase_oof(
    panel: paired.PairedPanel,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
    phase_config: potential.PhaseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.full(panel.n, np.nan, dtype=float)
    correction = np.full(panel.n, np.nan, dtype=float)
    geom = geometry(panel)
    for train, test in surface_folds(panel):
        tied_train = train[panel.paired_mask[train]]
        tied_model = potential.fit_potential(
            geom,
            panel.aggregate_weights,
            panel.one_phase_target,
            tied_train,
            potential_config,
        )
        aggregate_train = tied_model.predict(panel.aggregate_weights[train])
        phase_target = np.full(panel.n, np.nan, dtype=float)
        phase_target[train] = panel.two_phase_target[train] - aggregate_train
        phase_model = potential.fit_phase_potential(
            tied_model,
            panel.weights,
            phase_target,
            train,
            panel.alpha0,
            law,
            phase_config,
        )
        aggregate_test = tied_model.predict(panel.aggregate_weights[test])
        correction[test] = phase_model.predict_delta(panel.weights[test])
        combined[test] = aggregate_test + correction[test]
    if not np.isfinite(combined).all():
        raise RuntimeError(f"Incomplete StarCoder OOF prediction for {panel.name} {law.value}")
    return combined, correction


def select_phase(
    panel: paired.PairedPanel,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
) -> tuple[potential.PhaseConfig, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for config in potential_screen.phase_configs(law):
        prediction, correction = phase_oof(panel, potential_config, law, config)
        predictions[config.key] = prediction
        tied_correction = correction[panel.paired_mask]
        rows.append(
            {
                "surface": panel.name,
                "law": law.value,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "maximum_tied_correction": float(np.max(np.abs(tied_correction))),
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    values = json.loads(table.iloc[0]["config_json"])
    config_type = {
        potential.PhaseLaw.WORK_DISSIPATION: potential.WorkDissipationConfig,
        potential.PhaseLaw.RELAXATION: potential.RelaxationConfig,
    }[law]
    config = config_type(**values)
    return config, predictions[config.key], table


def fit_full_model(
    panel: paired.PairedPanel,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
    phase_config: potential.PhaseConfig,
) -> potential.PhasePotentialModel:
    tied = np.flatnonzero(panel.paired_mask)
    tied_model = potential.fit_potential(
        geometry(panel),
        panel.aggregate_weights,
        panel.one_phase_target,
        tied,
        potential_config,
    )
    phase_target = panel.two_phase_target - tied_model.predict(panel.aggregate_weights)
    return potential.fit_phase_potential(
        tied_model,
        panel.weights,
        phase_target,
        np.arange(panel.n),
        panel.alpha0,
        law,
        phase_config,
    )


def leave_region_out(
    panel: paired.PairedPanel,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
    phase_config: potential.PhaseConfig,
) -> list[dict[str, Any]]:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
    }
    rows = []
    geom = geometry(panel)
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        tied_train = train[panel.paired_mask[train]]
        if len(test) < 3 or len(tied_train) < 3:
            continue
        tied_model = potential.fit_potential(
            geom,
            panel.aggregate_weights,
            panel.one_phase_target,
            tied_train,
            potential_config,
        )
        phase_target = np.full(panel.n, np.nan, dtype=float)
        phase_target[train] = panel.two_phase_target[train] - tied_model.predict(panel.aggregate_weights[train])
        phase_model = potential.fit_phase_potential(
            tied_model,
            panel.weights,
            phase_target,
            train,
            panel.alpha0,
            law,
            phase_config,
        )
        prediction = phase_model.predict(panel.weights[test])
        rows.append(
            {
                "surface": panel.name,
                "law": law.value,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **paired_screen.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def optimum_record(
    panel: paired.PairedPanel,
    model: potential.PhasePotentialModel,
    law: potential.PhaseLaw,
) -> tuple[dict[str, Any], pd.DataFrame]:
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
        "law": law.value,
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
    panel: paired.PairedPanel,
    law: potential.PhaseLaw,
    surface: pd.DataFrame,
    optimum: dict[str, Any],
    output: Path,
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
        title=f"{panel.name}: {law.value.replace('_', ' ')}",
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
    panels = [panel_from_dataset(cosine), panel_from_dataset(load_refined_wsd80(cosine))]
    potential_tables = []
    phase_tables = []
    metric_rows = []
    prediction_rows = []
    region_rows = []
    optimum_rows = []
    for panel in panels:
        selected_potential, potential_table = select_potential(panel)
        potential_tables.append(potential_table)
        for law in potential.PhaseLaw:
            selected_phase, prediction, phase_table = select_phase(panel, selected_potential, law)
            phase_tables.append(phase_table)
            metric_rows.append(
                {
                    "surface": panel.name,
                    "law": law.value,
                    "n_tied": int(panel.paired_mask.sum()),
                    "selected_potential": selected_potential.key,
                    "selected_phase": selected_phase.key,
                    **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
                }
            )
            for index, (observed, predicted) in enumerate(zip(panel.two_phase_target, prediction, strict=True)):
                prediction_rows.append(
                    {
                        "surface": panel.name,
                        "law": law.value,
                        "row_index": index,
                        "phase_tied": bool(panel.paired_mask[index]),
                        "observed": observed,
                        "predicted": predicted,
                    }
                )
            region_rows.extend(leave_region_out(panel, selected_potential, law, selected_phase))
            model = fit_full_model(panel, selected_potential, law, selected_phase)
            optimum, surface = optimum_record(panel, model, law)
            optimum_rows.append(optimum)
            surface.to_csv(args.output_dir / f"{panel.name}__{law.value}__surface.csv", index=False)
            render_surface(panel, law, surface, optimum, args.output_dir / f"{panel.name}__{law.value}__surface.html")

    potential_grid = pd.concat(potential_tables, ignore_index=True)
    phase_grid = pd.concat(phase_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    regions = pd.DataFrame(region_rows)
    optima = pd.DataFrame(optimum_rows)
    potential_grid.to_csv(args.output_dir / "tied_potential_grid.csv", index=False)
    phase_grid.to_csv(args.output_dir / "phase_grid.csv", index=False)
    metrics.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "surface_oof_predictions.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    report = [
        "# Round-two potential phase laws: StarCoder falsification",
        "",
        "The tied potential was selected by leave-one-out prediction on phase-tied rows only. The phase law was then selected by five-fold OOF prediction over the full surface. Historical and adversarial Delphi outcomes were not read.",
        "",
        "## Surface OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
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
    print("\nLeave-region-out")
    print(regions.to_string(index=False))
    print("\nOptima")
    print(optima.to_string(index=False))


if __name__ == "__main__":
    main()
