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
"""Falsify preregistered spherical projected flow on StarCoder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_clipped_task_flow_round35 as clock,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_shared_private_round25 as shape_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    spherical_projected_flow_models as candidate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round36_spherical_projected_flow_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ANGULAR_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)
RARE_CURVATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
RELAXATION_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
EVALUATION_GRID = (0.1, 0.2, 0.5, 0.8, 1.0)
L2_GRID = (0.0, 0.1, 1.0)
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[candidate.SphericalProjectedFlowConfig]:
    return [
        candidate.SphericalProjectedFlowConfig(angular, rare, relaxation, evaluation)
        for angular in ANGULAR_GRID
        for rare in RARE_CURVATURE_GRID
        for relaxation in RELAXATION_GRID
        for evaluation in EVALUATION_GRID
    ]


def base_features(
    panel: paired.PairedPanel,
    all_configs: list[candidate.SphericalProjectedFlowConfig],
    weights: np.ndarray | None = None,
    *,
    steps_per_unit: int = candidate.INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    phase0 = clock.optimizer_phase0_fraction(panel)
    return np.asarray(
        [
            candidate.response_feature(
                policies,
                phase0,
                config,
                steps_per_unit=steps_per_unit,
            )
            for config in all_configs
        ],
        dtype=float,
    )


def expanded_variants(
    all_configs: list[candidate.SphericalProjectedFlowConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config_index = np.repeat(np.arange(len(all_configs)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID, dtype=float), len(all_configs))
    return features[config_index], config_index, l2


def score_all(
    panel: paired.PairedPanel,
    all_configs: list[candidate.SphericalProjectedFlowConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expanded, config_index, l2 = expanded_variants(all_configs, features)
    rmse, predictions = scalar_audit.score_configs(
        expanded,
        panel.two_phase_target,
        starcoder.surface_folds(panel),
        l2,
    )
    return rmse, predictions, config_index, l2


def nested_prediction(
    panel: paired.PairedPanel,
    all_configs: list[candidate.SphericalProjectedFlowConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    expanded, config_index, l2 = expanded_variants(all_configs, features)
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = scalar_audit.stratified_folds(panel, outer_train, 4, 20260719 + 100 * outer_fold)
        local_folds = [
            (
                np.flatnonzero(np.isin(outer_train, train)),
                np.flatnonzero(np.isin(outer_train, test)),
            )
            for train, test in inner_folds
        ]
        inner_rmse, _ = scalar_audit.score_configs(
            expanded[:, outer_train],
            panel.two_phase_target[outer_train],
            local_folds,
            l2,
        )
        selected = int(np.argmin(inner_rmse))
        prediction[outer_test] = scalar_audit.fit_predict_all(
            expanded[[selected]],
            panel.two_phase_target,
            outer_train,
            outer_test,
            l2[[selected]],
        )[0]
        config = all_configs[int(config_index[selected])]
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": outer_fold,
                "inner_rmse": float(inner_rmse[selected]),
                "l2": float(l2[selected]),
                **asdict(config),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def raw_optimum(
    panel: paired.PairedPanel,
    config: candidate.SphericalProjectedFlowConfig,
    l2: float,
    fit_feature: np.ndarray,
) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    feature = base_features(panel, [config], weights, steps_per_unit=384)[0]
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_phase0_rare": float(panel.weights[observed, 0, 1]),
        "observed_phase1_rare": float(panel.weights[observed, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed, 0, 1],
                p1.ravel()[best] - panel.weights[observed, 1, 1],
            )
        ),
        "response_amplitude": float(head.coefficient / head.feature_scale),
    }


def algebraic_audit(all_configs: list[candidate.SphericalProjectedFlowConfig]) -> dict[str, float]:
    maximum_semigroup_error = 0.0
    maximum_integration_error = 0.0
    for config in all_configs[::71]:
        maximum_semigroup_error = max(
            maximum_semigroup_error,
            candidate.tied_semigroup_error(config, 0.37),
        )
        rng = np.random.default_rng(20260719)
        rare = rng.uniform(size=64)
        state = rng.uniform(-0.25, 0.25, size=64)
        coarse = candidate.phase_update(state, rare, 1.0, config, steps_per_unit=192)
        fine = candidate.phase_update(state, rare, 1.0, config, steps_per_unit=384)
        maximum_integration_error = max(maximum_integration_error, float(np.max(np.abs(coarse - fine))))
    return {
        "maximum_tied_semigroup_error": maximum_semigroup_error,
        "maximum_192_vs_384_integration_error": maximum_integration_error,
    }


def update_registry(status: str, evidence: str) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Registry has no header")
    for row in rows:
        if row["id"] == "SPTF":
            row["status"] = status
            row["status_evidence"] = evidence
    temporary = REGISTRY.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def record_ledger(status: str, evidence: str) -> None:
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_36_starcoder_decision",
        "candidate_id": "SPTF",
        "candidate_family": "Spherical projected task flow",
        "hyperparameters": "Frozen round-36 grid with angular curvature zero exact ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-36 preregistration.",
        "novelty_class": "Constant-norm projected transition geometry",
        "evaluation_status": status,
        "evidence_path": "round36_spherical_projected_flow_starcoder/report.md",
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True).to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_configs = configs()
    algebraic = algebraic_audit(all_configs)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebraic, indent=2) + "\n")

    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    stage1_rows: list[dict[str, Any]] = []
    cache: dict[str, np.ndarray] = {}
    selected: dict[str, tuple[candidate.SphericalProjectedFlowConfig, float]] = {}
    stage1_passed = True
    for panel in panels:
        features = base_features(panel, all_configs)
        cache[panel.name] = features
        rmse, predictions, config_index, l2 = score_all(panel, all_configs, features)
        active = np.flatnonzero([all_configs[index].angular_curvature > 0.0 for index in config_index])
        ablation = np.flatnonzero([all_configs[index].angular_curvature == 0.0 for index in config_index])
        active_best = int(active[np.argmin(rmse[active])])
        ablation_best = int(ablation[np.argmin(rmse[ablation])])
        config = all_configs[int(config_index[active_best])]
        selected[panel.name] = (config, float(l2[active_best]))
        active_metrics = metrics.scalar_metrics(panel.two_phase_target, predictions[active_best])
        ablation_metrics = metrics.scalar_metrics(panel.two_phase_target, predictions[ablation_best])
        stage1_rows.append(
            {
                "surface": panel.name,
                "optimizer_phase0_fraction": clock.optimizer_phase0_fraction(panel),
                "active_config": json.dumps(asdict(config), sort_keys=True),
                "active_l2": float(l2[active_best]),
                "ablation_config": json.dumps(asdict(all_configs[int(config_index[ablation_best])]), sort_keys=True),
                "ablation_l2": float(l2[ablation_best]),
                **{f"active_{key}": value for key, value in active_metrics.items()},
                **{f"ablation_{key}": value for key, value in ablation_metrics.items()},
                "rmse_delta_active_minus_ablation": float(rmse[active_best] - rmse[ablation_best]),
            }
        )
        stage1_passed &= bool(rmse[active_best] < rmse[ablation_best])

    stage1 = pd.DataFrame(stage1_rows)
    stage1.to_csv(args.output_dir / "stage1_curvature_ablation.csv", index=False)
    report = [
        "# Round 36: spherical projected task flow",
        "",
        "The MuonH projection mechanism, finite grid, and exact zero-curvature ablation were frozen before this audit. No Delphi historical, adversarial, or sealed-confirmation outcome was read.",
        "",
        "## Algebraic audit",
        "",
        pd.DataFrame([algebraic]).to_markdown(index=False),
        "",
        "## Stage 1: projected curvature versus Euclidean flow",
        "",
        stage1.to_markdown(index=False),
        "",
    ]
    if not stage1_passed:
        status = "blocked_before_nested_starcoder"
        evidence = (
            "Nonzero projected curvature failed to beat the exact Euclidean flow ablation on both StarCoder schedules."
        )
        update_registry(status, evidence)
        record_ledger(status, evidence)
        report.extend(
            ["## Decision", "", f"**Blocked.** {evidence}", "", "No Delphi or adversarial outcomes were evaluated."]
        )
        (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
        print(stage1.to_string(index=False))
        print(status, evidence)
        return

    nested_rows: list[dict[str, Any]] = []
    nested_folds: list[pd.DataFrame] = []
    optima_rows: list[dict[str, Any]] = []
    for panel in panels:
        prediction, folds = nested_prediction(panel, all_configs, cache[panel.name])
        nested_rows.append({"surface": panel.name, **metrics.scalar_metrics(panel.two_phase_target, prediction)})
        nested_folds.append(folds)
        config, l2 = selected[panel.name]
        config_index = all_configs.index(config)
        optima_rows.append(raw_optimum(panel, config, l2, cache[panel.name][config_index]))

    nested = pd.DataFrame(nested_rows)
    folds = pd.concat(nested_folds, ignore_index=True)
    optima = pd.DataFrame(optima_rows)
    nested.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    folds.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)

    stability = folds.groupby("surface")["angular_curvature"].agg(["min", "max", "median"]).reset_index()
    stability["span"] = stability["max"] / np.maximum(stability["min"], 1e-12)
    stability["active_fraction"] = (
        folds.groupby("surface")["angular_curvature"].apply(lambda values: float((values > 0.0).mean())).to_numpy()
    )
    stability.to_csv(args.output_dir / "curvature_stability.csv", index=False)

    shape_ok = all(
        float(nested.loc[nested["surface"].eq(surface), "rmse"].iloc[0]) <= 1.05 * reference
        for surface, reference in SHAPE_REFERENCE.items()
    )
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    fold_active = bool((stability["active_fraction"] >= 0.6).all())
    curvature_stable = bool((stability["span"] <= 4.0).all())
    globally_interior = all(
        config.angular_curvature not in {min(value for value in ANGULAR_GRID if value > 0), max(ANGULAR_GRID)}
        for config, _ in selected.values()
    )
    passed = (
        algebraic["maximum_tied_semigroup_error"] < 1e-6
        and algebraic["maximum_192_vs_384_integration_error"] < 1e-6
        and shape_ok
        and optimum_ok
        and fold_active
        and curvature_stable
        and globally_interior
    )
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"curvature_beats_ablation={stage1_passed}; within_5pct_shape={shape_ok}; optimum_distance_ok={optimum_ok}; "
        f"fold_curvature_active={fold_active}; curvature_regime_stable={curvature_stable}; global_curvature_interior={globally_interior}."
    )
    update_registry(status, evidence)
    record_ledger(status, evidence)
    report.extend(
        [
            "## Nested StarCoder audit",
            "",
            nested.to_markdown(index=False),
            "",
            "## Curvature stability",
            "",
            stability.to_markdown(index=False),
            "",
            "## Raw optima",
            "",
            optima.to_markdown(index=False),
            "",
            "## Decision",
            "",
            f"Status: **{status}**. {evidence}",
            "",
            "No Delphi or adversarial outcomes were evaluated.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(stage1.to_string(index=False))
    print(nested.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
