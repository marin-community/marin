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
"""Falsify activated annealing and recoverable memorization as a frozen batch."""

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

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    activated_and_memorization_models as candidate,
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
    sgd_drift_diffusion_models as schedule_models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round24_activated_memorization_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.065388405808633, "starcoder_wsd_80_20": 0.0457725108696099}
CURVATURE_GRID = (0.5, 1.0, 2.0, 4.0)
SPEED_GRID = (0.25, 1.0, 4.0, 16.0)
BARRIER_GRID = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
EVALUATION_GRID = (0.2, 0.5, 0.8)
ACCUMULATION_GRID = (0.0, 0.25, 1.0, 4.0, 16.0)
RECOVERY_GRID = (0.25, 1.0, 4.0, 16.0)
OFFSET_GRID = (0.03, 0.1, 0.3)
L2_GRID = (0.1, 1.0)
SEED = 20260719
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def activated_configs() -> list[candidate.ActivatedFlowConfig]:
    return [
        candidate.ActivatedFlowConfig(curvature, speed, barrier, evaluation, l2)
        for curvature in CURVATURE_GRID
        for speed in SPEED_GRID
        for barrier in BARRIER_GRID
        for evaluation in EVALUATION_GRID
        for l2 in L2_GRID
    ]


def memorization_configs() -> list[candidate.MemorizationConfig]:
    return [
        candidate.MemorizationConfig(accumulation, recovery, offset, l2)
        for accumulation in ACCUMULATION_GRID
        for recovery in RECOVERY_GRID
        for offset in OFFSET_GRID
        for l2 in L2_GRID
    ]


def schedule_for(panel: paired.PairedPanel) -> schedule_models.Schedule:
    if panel.name.startswith("starcoder_cosine"):
        return schedule_models.Schedule.COSINE
    return schedule_models.Schedule.WSD


def memorization_geometry(panel: paired.PairedPanel) -> candidate.MemorizationGeometry:
    return candidate.MemorizationGeometry(
        panel.domain_names,
        np.asarray(panel.c0, dtype=float),
        np.asarray(panel.c1, dtype=float),
        np.asarray(panel.proportional_weights, dtype=float),
    )


def activated_features(
    panel: paired.PairedPanel,
    configs: list[candidate.ActivatedFlowConfig],
) -> np.ndarray:
    schedule = schedule_for(panel)
    cache: dict[tuple[float, float, float], np.ndarray] = {}
    rows = []
    for config in configs:
        key = (config.curvature_ratio, config.speed, config.barrier)
        if key not in cache:
            cache[key] = candidate.activated_terminal_state(panel.weights, panel.alpha0, schedule, config)
        state = cache[key]
        broad = 0.5 * (state + 0.5) ** 2
        rare = 0.5 * config.curvature_ratio * (state - 0.5) ** 2
        rows.append((1.0 - config.evaluation_mix) * broad + config.evaluation_mix * rare)
    return np.asarray(rows, dtype=float)


def memorization_designs(
    panel: paired.PairedPanel,
    configs: list[candidate.MemorizationConfig],
) -> list[np.ndarray]:
    geometry = memorization_geometry(panel)
    return [candidate.memorization_design(geometry, panel.weights, config)[0] for config in configs]


def score_scalar_configs(
    panel: paired.PairedPanel,
    configs: list[candidate.ActivatedFlowConfig],
    features: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    return scalar_audit.score_configs(
        features,
        panel.two_phase_target,
        folds,
        np.asarray([config.l2 for config in configs], dtype=float),
    )


def fit_memorization_head(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    l2: float,
) -> paired.LinearHead:
    return paired.fit_linear_head(
        design[train],
        target[train],
        [f"feature_{index}" for index in range(design.shape[1])],
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=l2,
    )


def score_memorization_configs(
    panel: paired.PairedPanel,
    configs: list[candidate.MemorizationConfig],
    designs: list[np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    all_predictions = np.full((len(configs), panel.n), np.nan, dtype=float)
    for index, (config, design) in enumerate(zip(configs, designs, strict=True)):
        for train, test in folds:
            head = fit_memorization_head(design, panel.two_phase_target, train, config.l2)
            all_predictions[index, test] = head.predict(design[test])
    if not np.isfinite(all_predictions).all():
        raise RuntimeError(f"Incomplete memorization OOF predictions for {panel.name}")
    rmse = np.sqrt(np.mean((all_predictions - panel.two_phase_target[None, :]) ** 2, axis=1))
    return rmse, all_predictions


def nested_scalar(
    panel: paired.PairedPanel,
    configs: list[candidate.ActivatedFlowConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        local_folds = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner
        ]
        local_panel = paired.PairedPanel(
            panel.name,
            panel.target,
            panel.frame.iloc[outer_train].reset_index(drop=True),
            panel.domain_names,
            panel.family_names,
            panel.family_members,
            panel.weights[outer_train],
            panel.c0,
            panel.c1,
            panel.two_phase_target[outer_train],
            panel.one_phase_target[outer_train],
        )
        scores, _inner_predictions = score_scalar_configs(local_panel, configs, features[:, outer_train], local_folds)
        selected_index = int(np.argmin(scores))
        selected = configs[selected_index]
        prediction[outer_test] = scalar_audit.fit_predict_all(
            features[[selected_index]],
            panel.two_phase_target,
            outer_train,
            outer_test,
            np.asarray([selected.l2]),
        )[0]
        rows.append(
            {"surface": panel.name, "outer_fold": fold, "inner_rmse": float(scores[selected_index]), **asdict(selected)}
        )
    return prediction, pd.DataFrame(rows)


def nested_memorization(
    panel: paired.PairedPanel,
    configs: list[candidate.MemorizationConfig],
    designs: list[np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        scores = []
        for config, design in zip(configs, designs, strict=True):
            inner_prediction = np.full(len(outer_train), np.nan, dtype=float)
            for train, test in inner:
                local_test = np.flatnonzero(np.isin(outer_train, test))
                head = fit_memorization_head(design, panel.two_phase_target, train, config.l2)
                inner_prediction[local_test] = head.predict(design[test])
            scores.append(float(np.sqrt(np.mean((inner_prediction - panel.two_phase_target[outer_train]) ** 2))))
        selected_index = int(np.argmin(scores))
        selected = configs[selected_index]
        design = designs[selected_index]
        head = fit_memorization_head(design, panel.two_phase_target, outer_train, selected.l2)
        prediction[outer_test] = head.predict(design[outer_test])
        rows.append(
            {"surface": panel.name, "outer_fold": fold, "inner_rmse": scores[selected_index], **asdict(selected)}
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested memorization prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def raw_optimum(
    panel: paired.PairedPanel,
    candidate_id: str,
    config: candidate.ActivatedFlowConfig | candidate.MemorizationConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    train = np.arange(panel.n)
    if candidate_id == "AAGF":
        assert isinstance(config, candidate.ActivatedFlowConfig)
        model = candidate.fit_activated_model(
            panel.weights, panel.two_phase_target, train, panel.alpha0, schedule_for(panel), config
        )
    else:
        assert isinstance(config, candidate.MemorizationConfig)
        model = candidate.fit_memorization_model(
            memorization_geometry(panel), panel.weights, panel.two_phase_target, train, config
        )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    row = {
        "candidate": candidate_id,
        "surface": panel.name,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                p1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
    }
    surface = pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction})
    return row, surface


def render_surface(
    panel: paired.PairedPanel,
    candidate_id: str,
    surface: pd.DataFrame,
    output: Path,
) -> None:
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
        title=f"{panel.name}: {candidate_id}",
        template="plotly_white",
        scene={"xaxis_title": "Phase 0 rare", "yaxis_title": "Phase 1 rare", "zaxis_title": "BPB"},
        height=850,
        width=1000,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def gate_and_record(
    candidate_id: str,
    selected: pd.DataFrame,
    folds: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    optima: pd.DataFrame,
    output_dir: Path,
) -> dict[str, bool]:
    if candidate_id == "AAGF":
        parameter = "barrier"
        secondary_parameter = "speed"
        lower = min(BARRIER_GRID)
        upper = max(BARRIER_GRID)
        secondary_lower = min(SPEED_GRID)
        secondary_upper = max(SPEED_GRID)
        ablation = selected["ablation_rmse"]
    else:
        parameter = "accumulation_rate"
        secondary_parameter = "recovery_rate"
        lower = min(ACCUMULATION_GRID)
        upper = max(ACCUMULATION_GRID)
        secondary_lower = min(RECOVERY_GRID)
        secondary_upper = max(RECOVERY_GRID)
        ablation = selected["ablation_rmse"]
    values = selected[parameter].to_numpy(dtype=float)
    positive = values[values > 0.0]
    regime_ratio = float(np.max(positive) / np.min(positive)) if len(positive) == 2 else np.inf
    nested_lookup = nested_metrics.set_index("surface")
    gates = {
        "mechanism_global_both": bool((values > 0.0).all() and (selected["selected_rmse"] <= 0.99 * ablation).all()),
        "mechanism_fold_majority_both": bool(
            (folds.groupby("surface")[parameter].apply(lambda x: float(np.mean(x > 0.0))) >= 0.6).all()
        ),
        "mechanism_not_boundary": bool((values > lower).all() and (values < upper).all()),
        "secondary_rate_not_boundary": bool(
            (selected[secondary_parameter] > secondary_lower).all()
            and (selected[secondary_parameter] < secondary_upper).all()
        ),
        "mechanism_regime_transfer": bool(regime_ratio <= 4.0),
        "within_5pct_shape_reference": bool(
            all(
                float(nested_lookup.loc[name, "rmse"]) <= 1.05 * reference for name, reference in SHAPE_REFERENCE.items()
            )
        ),
        "optimum_distance_ok": bool((optima["distance_to_observed_best"] <= 0.15).all()),
    }
    status = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    evidence = "; ".join(f"{key}={value}" for key, value in gates.items())
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq(candidate_id), "status"] = status
    registry.loc[registry["id"].eq(candidate_id), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_24_starcoder_gate",
        "candidate_id": candidate_id,
        "candidate_family": "Activation-barrier annealing gradient flow"
        if candidate_id == "AAGF"
        else "Recoverable replay memorization",
        "hyperparameters": "Frozen round-24 batch grid with nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_24_batch_preregistration",
        "novelty_class": "Kramers transition clock"
        if candidate_id == "AAGF"
        else "Recoverable replay-memorization state",
        "evaluation_status": status,
        "evidence_path": str(output_dir.relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    return gates


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activated = activated_configs()
    memorization = memorization_configs()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    selected_rows = []
    fold_frames = []
    prediction_rows = []
    optimum_rows = []
    grid_frames = []
    for panel in panels:
        folds = starcoder.surface_folds(panel)
        a_features = activated_features(panel, activated)
        a_rmse, _ = score_scalar_configs(panel, activated, a_features, folds)
        a_best = int(np.argmin(a_rmse))
        a_selected = activated[a_best]
        a_ablation = float(np.min(a_rmse[np.asarray([config.barrier == 0.0 for config in activated])]))
        a_nested, a_folds = nested_scalar(panel, activated, a_features)
        a_folds.insert(0, "candidate", "AAGF")
        fold_frames.append(a_folds)
        selected_rows.append(
            {
                "candidate": "AAGF",
                "surface": panel.name,
                **asdict(a_selected),
                "selected_rmse": float(a_rmse[a_best]),
                "ablation_rmse": a_ablation,
            }
        )
        grid_frames.append(
            pd.DataFrame(
                [
                    {"candidate": "AAGF", "surface": panel.name, **asdict(config), "rmse": float(a_rmse[i])}
                    for i, config in enumerate(activated)
                ]
            )
        )
        a_optimum, a_surface = raw_optimum(panel, "AAGF", a_selected)
        optimum_rows.append(a_optimum)
        a_surface.to_csv(args.output_dir / f"AAGF__{panel.name}__surface.csv", index=False)
        render_surface(panel, "AAGF", a_surface, args.output_dir / f"AAGF__{panel.name}__surface.html")
        prediction_rows.extend(
            {
                "candidate": "AAGF",
                "surface": panel.name,
                "observed": float(panel.two_phase_target[i]),
                "predicted": float(a_nested[i]),
            }
            for i in range(panel.n)
        )

        m_designs = memorization_designs(panel, memorization)
        m_rmse, _ = score_memorization_configs(panel, memorization, m_designs, folds)
        m_best = int(np.argmin(m_rmse))
        m_selected = memorization[m_best]
        m_ablation = float(np.min(m_rmse[np.asarray([config.accumulation_rate == 0.0 for config in memorization])]))
        m_nested, m_folds = nested_memorization(panel, memorization, m_designs)
        m_folds.insert(0, "candidate", "RMR")
        fold_frames.append(m_folds)
        selected_rows.append(
            {
                "candidate": "RMR",
                "surface": panel.name,
                **asdict(m_selected),
                "selected_rmse": float(m_rmse[m_best]),
                "ablation_rmse": m_ablation,
            }
        )
        grid_frames.append(
            pd.DataFrame(
                [
                    {"candidate": "RMR", "surface": panel.name, **asdict(config), "rmse": float(m_rmse[i])}
                    for i, config in enumerate(memorization)
                ]
            )
        )
        m_optimum, m_surface = raw_optimum(panel, "RMR", m_selected)
        optimum_rows.append(m_optimum)
        m_surface.to_csv(args.output_dir / f"RMR__{panel.name}__surface.csv", index=False)
        render_surface(panel, "RMR", m_surface, args.output_dir / f"RMR__{panel.name}__surface.html")
        prediction_rows.extend(
            {
                "candidate": "RMR",
                "surface": panel.name,
                "observed": float(panel.two_phase_target[i]),
                "predicted": float(m_nested[i]),
            }
            for i in range(panel.n)
        )

    selected_frame = pd.DataFrame(selected_rows)
    folds_frame = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    optima = pd.DataFrame(optimum_rows)
    grid_frame = pd.concat(grid_frames, ignore_index=True)
    nested_metrics = pd.DataFrame(
        [
            {
                "candidate": candidate_id,
                "surface": surface,
                **metrics.scalar_metrics(group["observed"], group["predicted"]),
            }
            for (candidate_id, surface), group in predictions.groupby(["candidate", "surface"])
        ]
    )
    gates = {}
    for candidate_id in ("AAGF", "RMR"):
        gates[candidate_id] = gate_and_record(
            candidate_id,
            selected_frame.loc[selected_frame["candidate"].eq(candidate_id)],
            folds_frame.loc[folds_frame["candidate"].eq(candidate_id)],
            nested_metrics.loc[nested_metrics["candidate"].eq(candidate_id)],
            optima.loc[optima["candidate"].eq(candidate_id)],
            args.output_dir,
        )
    selected_frame.to_csv(args.output_dir / "selected_configs.csv", index=False)
    folds_frame.to_csv(args.output_dir / "nested_selections.csv", index=False)
    predictions.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    nested_metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    grid_frame.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    summary = {
        candidate_id: {
            "gates": candidate_gates,
            "status": "promoted_to_multi_swarm" if all(candidate_gates.values()) else "blocked_before_multi_swarm",
        }
        for candidate_id, candidate_gates in gates.items()
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = f"""# Round 24: frozen activated-annealing and replay-memorization batch

## Selected configurations

{selected_frame.to_markdown(index=False)}

## Nested shape metrics

{nested_metrics.to_markdown(index=False)}

## Raw optima

{optima.to_markdown(index=False)}

## Frozen gates

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

Both candidates were preregistered before either fit. Historical, adversarial, and sealed-confirmation outcomes were not read or scored. A failed candidate cannot be rescued by retuning this exposed mechanism.
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
