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
"""Falsify moment-closed SGD drift-diffusion on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    sgd_drift_diffusion_models as diffusion,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round23_sgd_drift_diffusion_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
CURVATURE_GRID = (0.5, 1.0, 2.0, 4.0)
DRIFT_GRID = (0.25, 1.0, 4.0, 16.0)
DIFFUSION_GRID = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
EVALUATION_GRID = (0.2, 0.5, 0.8)
L2_GRID = (0.1, 1.0)
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.065388405808633, "starcoder_wsd_80_20": 0.0457725108696099}
SEED = 20260719
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[diffusion.DriftDiffusionConfig]:
    return [
        diffusion.DriftDiffusionConfig(curvature, drift, noise, evaluation, l2)
        for curvature in CURVATURE_GRID
        for drift in DRIFT_GRID
        for noise in DIFFUSION_GRID
        for evaluation in EVALUATION_GRID
        for l2 in L2_GRID
    ]


def schedule_for(panel: paired.PairedPanel) -> diffusion.Schedule:
    if panel.name.startswith("starcoder_cosine"):
        return diffusion.Schedule.COSINE
    if panel.name.startswith("starcoder_wsd"):
        return diffusion.Schedule.WSD
    raise ValueError(f"Unknown StarCoder schedule for {panel.name}")


def feature_matrix(
    panel: paired.PairedPanel,
    all_configs: list[diffusion.DriftDiffusionConfig],
) -> np.ndarray:
    schedule = schedule_for(panel)
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for config in all_configs:
        key = (config.curvature_ratio, config.drift_rate, config.diffusion_scale)
        if key not in cache:
            cache[key] = diffusion.terminal_moments(panel.weights, panel.alpha0, schedule, config)
        mean, variance = cache[key]
        broad = 0.5 * ((mean + 0.5) ** 2 + variance)
        rare = 0.5 * config.curvature_ratio * ((mean - 0.5) ** 2 + variance)
        rows.append((1.0 - config.evaluation_mix) * broad + config.evaluation_mix * rare)
    return np.asarray(rows, dtype=float)


def select_surface(
    panel: paired.PairedPanel,
    all_configs: list[diffusion.DriftDiffusionConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features, panel.two_phase_target, starcoder.surface_folds(panel), l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": float(rmse[index])}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    all_configs: list[diffusion.DriftDiffusionConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        local_folds = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner
        ]
        scores, _inner_predictions = scalar_audit.score_configs(
            features[:, outer_train], panel.two_phase_target[outer_train], local_folds, l2
        )
        selected_index = int(np.argmin(scores))
        selected = all_configs[selected_index]
        prediction[outer_test] = scalar_audit.fit_predict_all(
            features[[selected_index]],
            panel.two_phase_target,
            outer_train,
            outer_test,
            np.asarray([selected.l2]),
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": fold,
                "selected_config": selected.key,
                "inner_rmse": float(scores[selected_index]),
                **asdict(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def tied_selection(
    panel: paired.PairedPanel,
    all_configs: list[diffusion.DriftDiffusionConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    folds = list(KFold(min(5, len(tied)), shuffle=True, random_state=SEED + 77).split(tied))
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features[:, tied], panel.two_phase_target[tied], folds, l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": float(rmse[index])}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def leave_region_out(
    panel: paired.PairedPanel,
    config: diffusion.DriftDiffusionConfig,
    feature: np.ndarray,
) -> pd.DataFrame:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    for region, mask in regions.items():
        train = np.flatnonzero(~mask)
        test = np.flatnonzero(mask)
        prediction = scalar_audit.fit_predict_all(
            feature[None, :], panel.two_phase_target, train, test, np.asarray([config.l2])
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **metrics.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return pd.DataFrame(rows)


def algebraic_audit(all_configs: list[diffusion.DriftDiffusionConfig]) -> dict[str, float | bool]:
    rng = np.random.default_rng(SEED)
    rare = rng.uniform(0.0, 1.0, 32)
    errors = []
    minimum_variance = np.inf
    for config in all_configs[:: max(1, len(all_configs) // 64)]:
        for schedule, split in ((diffusion.Schedule.COSINE, 0.5), (diffusion.Schedule.WSD, 0.8)):
            errors.append(diffusion.tied_policy_error(rare, split, schedule, config))
            tied = np.stack([np.column_stack([1.0 - rare, rare]), np.column_stack([1.0 - rare, rare])], axis=1)
            _mean, variance = diffusion.terminal_moments(tied, split, schedule, config)
            minimum_variance = min(minimum_variance, float(np.min(variance)))
    return {
        "maximum_tied_boundary_error": float(max(errors)),
        "minimum_terminal_variance": float(minimum_variance),
        "tied_boundary_invariant": bool(max(errors) < 1e-10),
        "variance_nonnegative": bool(minimum_variance >= -1e-12),
    }


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: diffusion.DriftDiffusionConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    schedule = schedule_for(panel)
    model = diffusion.fit_model(
        panel.weights, panel.two_phase_target, np.arange(panel.n), panel.alpha0, schedule, config
    )
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    mean, variance = diffusion.terminal_moments(weights[[best]], panel.alpha0, schedule, config)
    row = {
        "surface": panel.name,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "terminal_mean": float(mean[0]),
        "terminal_variance": float(variance[0]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                p1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
        "response_amplitude": model.head.amplitude,
    }
    surface = pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction})
    return row, surface


def render_surface(panel: paired.PairedPanel, surface: pd.DataFrame, output: Path) -> None:
    size = round(np.sqrt(len(surface)))
    axis = surface["phase0_rare"].to_numpy().reshape(size, size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(size, size)
    figure = go.Figure(
        [
            go.Surface(x=axis, y=axis, z=z.T, colorscale="RdYlGn_r", opacity=0.72, name="Predicted"),
            go.Scatter3d(
                x=panel.weights[:, 0, 1],
                y=panel.weights[:, 1, 1],
                z=panel.two_phase_target,
                mode="markers",
                marker={"size": 4, "color": panel.two_phase_target, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: SGD drift-diffusion task dynamics",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
        width=1000,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def render_predictions(frame: pd.DataFrame, output: Path) -> None:
    figure = go.Figure()
    for surface, group in frame.groupby("surface"):
        figure.add_trace(
            go.Scatter(
                x=group["observed"],
                y=group["predicted"],
                mode="markers",
                name=surface,
                text=group["coordinate"],
                hovertemplate="%{text}<br>observed=%{x:.5f}<br>predicted=%{y:.5f}<extra></extra>",
            )
        )
    low = float(min(frame["observed"].min(), frame["predicted"].min()))
    high = float(max(frame["observed"].max(), frame["predicted"].max()))
    figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Identity", line={"dash": "dash"}))
    figure.update_layout(
        title="Nested OOF predictions: SGD drift-diffusion",
        xaxis_title="Observed BPB",
        yaxis_title="Predicted BPB",
        template="plotly_white",
        width=1000,
        height=760,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def update_registry_and_ledger(gates: dict[str, bool], output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    passed = all(gates.values())
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = "; ".join(f"{key}={value}" for key, value in gates.items())
    registry.loc[registry["id"].eq("SGDDD"), "status"] = status
    registry.loc[registry["id"].eq("SGDDD"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_23_starcoder_gate",
        "candidate_id": "SGDDD",
        "candidate_family": "SGD drift-diffusion task dynamics",
        "hyperparameters": "Frozen preregistered grid with nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_23_preregistration",
        "novelty_class": "Physical SGD trajectory covariance with eta-squared injection",
        "evaluation_status": status,
        "evidence_path": str(output_dir.relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_configs = configs()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    algebra = algebraic_audit(all_configs)
    grid_tables = []
    nested_rows = []
    nested_selection_rows = []
    tied_rows = []
    leave_rows = []
    optima = []
    selected_rows = []
    for panel in panels:
        features = feature_matrix(panel, all_configs)
        selected_index, selected_oof, grid = select_surface(panel, all_configs, features)
        selected = all_configs[selected_index]
        deterministic = grid.loc[grid["diffusion_scale"].eq(0.0)].iloc[0]
        nested_prediction, nested_selections = nested_selection(panel, all_configs, features)
        tied_index, tied_prediction, tied_grid = tied_selection(panel, all_configs, features)
        optimum, surface = optimum_and_surface(panel, selected)
        render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        grid_tables.append(grid)
        nested_selection_rows.append(nested_selections)
        leave_rows.append(leave_region_out(panel, selected, features[selected_index]))
        optima.append(optimum)
        selected_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                **asdict(selected),
                "selected_rmse": float(grid.iloc[0]["rmse"]),
                "deterministic_rmse": float(deterministic["rmse"]),
                "relative_gain_over_deterministic": float(
                    (float(deterministic["rmse"]) - float(grid.iloc[0]["rmse"])) / float(deterministic["rmse"])
                ),
            }
        )
        nested_rows.extend(
            {
                "surface": panel.name,
                "coordinate": f"p0={panel.weights[index, 0, 1]:.4f},p1={panel.weights[index, 1, 1]:.4f}",
                "observed": float(panel.two_phase_target[index]),
                "predicted": float(nested_prediction[index]),
                "global_selected_prediction": float(selected_oof[index]),
            }
            for index in range(panel.n)
        )
        tied = np.flatnonzero(panel.paired_mask)
        tied_rows.append(
            {
                "surface": panel.name,
                "n_tied": len(tied),
                "selected_config": all_configs[tied_index].key,
                **asdict(all_configs[tied_index]),
                **{
                    f"tied_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target[tied], tied_prediction).items()
                },
                "algebraic_fit_note": "This fits the two-phase observations at tied coordinates; no independent one-phase outcomes exist on these StarCoder panels.",
                "best_tied_grid_rmse": float(tied_grid.iloc[0]["rmse"]),
            }
        )

    grid_frame = pd.concat(grid_tables, ignore_index=True)
    nested_frame = pd.DataFrame(nested_rows)
    nested_selections = pd.concat(nested_selection_rows, ignore_index=True)
    selected_frame = pd.DataFrame(selected_rows)
    optima_frame = pd.DataFrame(optima)
    leave_frame = pd.concat(leave_rows, ignore_index=True)
    tied_frame = pd.DataFrame(tied_rows)
    nested_metrics = pd.DataFrame(
        [
            {"surface": surface, **metrics.scalar_metrics(group["observed"], group["predicted"])}
            for surface, group in nested_frame.groupby("surface")
        ]
    )
    render_predictions(nested_frame, args.output_dir / "nested_oof_predictions.html")

    selected_by_surface = selected_frame.set_index("surface")
    nested_by_surface = nested_metrics.set_index("surface")
    nonzero_fold_fraction = nested_selections.groupby("surface")["diffusion_scale"].apply(
        lambda values: float(np.mean(values.to_numpy() > 0.0))
    )
    cosine_noise = float(selected_by_surface.loc["starcoder_cosine_50_50", "diffusion_scale"])
    wsd_noise = float(selected_by_surface.loc["starcoder_wsd_80_20", "diffusion_scale"])
    positive_noises = [value for value in (cosine_noise, wsd_noise) if value > 0.0]
    diffusion_ratio = max(positive_noises) / min(positive_noises) if len(positive_noises) == 2 else np.inf
    gates = {
        "algebra_ok": bool(algebra["tied_boundary_invariant"] and algebra["variance_nonnegative"]),
        "diffusion_global_both": bool(
            (selected_frame["diffusion_scale"] > 0.0).all()
            and (selected_frame["relative_gain_over_deterministic"] >= 0.01).all()
        ),
        "diffusion_fold_majority_both": bool((nonzero_fold_fraction >= 0.6).all()),
        "diffusion_regime_transfer": bool(diffusion_ratio <= 10.0),
        "diffusion_not_boundary": bool(
            (selected_frame["diffusion_scale"] > min(DIFFUSION_GRID)).all()
            and (selected_frame["diffusion_scale"] < max(DIFFUSION_GRID)).all()
        ),
        "within_5pct_shape_reference": bool(
            all(
                float(nested_by_surface.loc[name, "rmse"]) <= 1.05 * reference
                for name, reference in SHAPE_REFERENCE.items()
            )
        ),
        "optimum_distance_ok": bool((optima_frame["distance_to_observed_best"] <= 0.15).all()),
        "response_amplitude_positive": bool((optima_frame["response_amplitude"] > 1e-8).all()),
    }
    update_registry_and_ledger(gates, args.output_dir)

    artifacts = {
        "hyperparameter_grid.csv": grid_frame,
        "selected_configs.csv": selected_frame,
        "nested_oof_predictions.csv": nested_frame,
        "nested_oof_metrics.csv": nested_metrics,
        "nested_selections.csv": nested_selections,
        "leave_region_out_metrics.csv": leave_frame,
        "one_phase_restriction_audit.csv": tied_frame,
        "predicted_optima.csv": optima_frame,
    }
    for name, frame in artifacts.items():
        frame.to_csv(args.output_dir / name, index=False)
    summary = {
        "candidate": "SGDDD",
        "algebra": algebra,
        "gates": gates,
        "diffusion_ratio": float(diffusion_ratio),
        "nonzero_fold_fraction": nonzero_fold_fraction.to_dict(),
        "status": "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = f"""# Round 23: SGD drift-diffusion StarCoder falsification

## Frozen mechanism

The candidate propagates the mean and across-training-run variance of a scalar representation under stochastic domain gradients. Mean drift scales with the learning rate, covariance injection with learning-rate squared, and the response is expected terminal task loss. The zero-diffusion model is an exact deterministic-gradient-flow ablation; variance has no free output coefficient.

## Selected configurations

{selected_frame.to_markdown(index=False)}

## Nested shape metrics

{nested_metrics.to_markdown(index=False)}

Corrected fixed shape references: {SHAPE_REFERENCE}.

## Raw optima

{optima_frame.to_markdown(index=False)}

## Gate

{pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_markdown(index=False)}

Status: **{summary["status"]}**.

The candidate was frozen before this audit. Historical, adversarial, and sealed-confirmation outcomes were not read or scored. Failure at this stage closes the physical SGD-diffusion route unless a new observed state or scaling invariant identifies the diffusion coefficient independently.
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
