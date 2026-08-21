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
"""Falsify competence-triggered gradient interference on StarCoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    competence_triggered_interference_models as conflict,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    hessian_equilibrium_models as scalar_head,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round30_competence_triggered_interference_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv"
SEED = 20260719
GENERAL_RATE_GRID = (1.0, 4.0, 16.0)
RARE_GENERAL_GRID = (0.3, 1.0)
SPECIALIST_RATE_GRID = (1.0, 4.0, 16.0)
INTERFERENCE_GRID = (0.0, 2.0, 8.0)
THRESHOLD_GRID = (0.25, 0.6)
SOFTNESS = 0.1
EVALUATION_GRID = (0.2, 0.5, 0.8)
L2_GRID = (0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs(interference_values: tuple[float, ...]) -> list[conflict.InterferenceConfig]:
    rows = []
    for general_rate in GENERAL_RATE_GRID:
        for rare_general in RARE_GENERAL_GRID:
            for specialist_rate in SPECIALIST_RATE_GRID:
                for interference_rate in interference_values:
                    thresholds = THRESHOLD_GRID if interference_rate > 0.0 else (THRESHOLD_GRID[0],)
                    for threshold in thresholds:
                        for evaluation in EVALUATION_GRID:
                            for l2 in L2_GRID:
                                rows.append(
                                    conflict.InterferenceConfig(
                                        general_rate,
                                        rare_general,
                                        specialist_rate,
                                        interference_rate,
                                        threshold,
                                        SOFTNESS,
                                        evaluation,
                                        l2,
                                    )
                                )
    return rows


def feature_matrix(panel: paired.PairedPanel, all_configs: list[conflict.InterferenceConfig]) -> np.ndarray:
    cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows = []
    for config in all_configs:
        key = (
            config.general_rate,
            config.rare_general_efficiency,
            config.specialist_rate,
            config.interference_rate,
            config.threshold,
            config.softness,
            config.evaluation_specialist_weight,
        )
        if key not in cache:
            cache[key] = conflict.response_feature(panel.weights, panel.alpha0, config)
        rows.append(cache[key])
    return np.asarray(rows, dtype=float)


def select_global(
    panel: paired.PairedPanel,
    all_configs: list[conflict.InterferenceConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2_values = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(
        features,
        panel.two_phase_target,
        starcoder.surface_folds(panel),
        l2_values,
    )
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        {"surface": panel.name, "config": [config.key for config in all_configs], "rmse": rmse}
    ).sort_values("rmse")
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    all_configs: list[conflict.InterferenceConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    l2_values = np.asarray([config.l2 for config in all_configs], dtype=float)
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + 100 * outer_fold)
        local_folds = [
            (
                np.flatnonzero(np.isin(outer_train, train)),
                np.flatnonzero(np.isin(outer_train, test)),
            )
            for train, test in inner_folds
        ]
        inner_rmse, _ = scalar_audit.score_configs(
            features[:, outer_train],
            panel.two_phase_target[outer_train],
            local_folds,
            l2_values,
        )
        selected_index = int(np.argmin(inner_rmse))
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
                "outer_fold": outer_fold,
                "selected_config": selected.key,
                "inner_rmse": float(inner_rmse[selected_index]),
                "general_rate": selected.general_rate,
                "rare_general_efficiency": selected.rare_general_efficiency,
                "specialist_rate": selected.specialist_rate,
                "interference_rate": selected.interference_rate,
                "threshold": selected.threshold,
                "evaluation_specialist_weight": selected.evaluation_specialist_weight,
                "l2": selected.l2,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested predictions for {panel.name}")
    return prediction, pd.DataFrame(rows)


def algebraic_audit(config: conflict.InterferenceConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    rare_weight = rng.uniform(size=64)
    general = rng.uniform(size=64)
    specialist = rng.uniform(size=64)
    first_general, first_specialist = conflict.phase_update(
        general,
        specialist,
        rare_weight,
        0.37,
        config,
    )
    split_general, split_specialist = conflict.phase_update(
        first_general,
        first_specialist,
        rare_weight,
        0.63,
        config,
    )
    whole_general, whole_specialist = conflict.phase_update(
        general,
        specialist,
        rare_weight,
        1.0,
        config,
    )
    return {
        "maximum_tied_semigroup_error": float(
            max(
                np.max(np.abs(split_general - whole_general)),
                np.max(np.abs(split_specialist - whole_specialist)),
            )
        ),
        "minimum_state": float(min(split_general.min(), split_specialist.min())),
        "maximum_state": float(max(split_general.max(), split_specialist.max())),
    }


def raw_optimum(panel: paired.PairedPanel, config: conflict.InterferenceConfig) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 101)
    phase0, phase1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - phase0.ravel(), phase0.ravel()]),
            np.column_stack([1.0 - phase1.ravel(), phase1.ravel()]),
        ],
        axis=1,
    )
    train_feature = conflict.response_feature(panel.weights, panel.alpha0, config)
    head = scalar_head.fit_quadratic_head(train_feature, panel.two_phase_target, np.arange(panel.n), config.l2)
    prediction = head.predict(conflict.response_feature(weights, panel.alpha0, config))
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "phase0_rare": float(phase0.ravel()[best]),
        "phase1_rare": float(phase1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                phase0.ravel()[best] - panel.weights[observed_best, 0, 1],
                phase1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
        "late_specialization": bool(phase1.ravel()[best] > phase0.ravel()[best]),
        "response_amplitude": head.natural_curvature,
    }


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("CTGI")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_30_starcoder_gate",
        "candidate_id": "CTGI",
        "candidate_family": "Competence-triggered gradient interference",
        "hyperparameters": "Frozen Round 30 grid; exact interference-zero ablation; nested selection only after stage-1 survival",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_30_batch_preregistration",
        "novelty_class": "Competence-dependent sign change in broad-to-specialist gradient interaction",
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
    active_grid = configs(INTERFERENCE_GRID[1:])
    ablation_grid = configs((0.0,))
    all_configs = ablation_grid + active_grid
    representative = conflict.InterferenceConfig(4.0, 0.3, 4.0, 2.0, 0.6, SOFTNESS, 0.5, 0.1)
    algebra = algebraic_audit(representative)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    stage1_rows = []
    active_tables = []
    ablation_tables = []
    selected_active: dict[str, conflict.InterferenceConfig] = {}
    feature_cache: dict[str, np.ndarray] = {}
    for panel in panels:
        active_features = feature_matrix(panel, active_grid)
        ablation_features = feature_matrix(panel, ablation_grid)
        active_index, active_prediction, active_table = select_global(panel, active_grid, active_features)
        ablation_index, ablation_prediction, ablation_table = select_global(panel, ablation_grid, ablation_features)
        selected_active[panel.name] = active_grid[active_index]
        feature_cache[panel.name] = feature_matrix(panel, all_configs)
        active_metrics = paired_screen.scalar_metrics(panel.two_phase_target, active_prediction)
        ablation_metrics = paired_screen.scalar_metrics(panel.two_phase_target, ablation_prediction)
        stage1_rows.append(
            {
                "surface": panel.name,
                "active_config": active_grid[active_index].key,
                "ablation_config": ablation_grid[ablation_index].key,
                **{f"active_{key}": value for key, value in active_metrics.items()},
                **{f"ablation_{key}": value for key, value in ablation_metrics.items()},
                "rmse_delta_active_minus_ablation": active_metrics["rmse"] - ablation_metrics["rmse"],
            }
        )
        active_tables.append(active_table)
        ablation_tables.append(ablation_table)

    stage1 = pd.DataFrame(stage1_rows)
    stage1.to_csv(args.output_dir / "stage1_global_oof_comparison.csv", index=False)
    pd.concat(active_tables, ignore_index=True).to_csv(args.output_dir / "active_grid.csv", index=False)
    pd.concat(ablation_tables, ignore_index=True).to_csv(args.output_dir / "ablation_grid.csv", index=False)
    stage1_passed = bool((stage1["rmse_delta_active_minus_ablation"] < 0.0).all())
    report = [
        "# Round 30: competence-triggered gradient interference",
        "",
        "## Frozen mechanism",
        "",
        r"General competence follows $\dot g=a[(1-p)+rp](1-g)$. Specialist competence follows $\dot s=bp g(1-s)-c(1-p)s\,\sigma((g-\theta)/\delta)$. Thus broad-to-specialist interference activates only after general competence matures. The exact ablation sets $c=0$.",
        "",
        "## Algebraic audit",
        "",
        f"- Maximum tied semigroup error: `{algebra['maximum_tied_semigroup_error']:.3e}`.",
        f"- State range: `[{algebra['minimum_state']:.6f}, {algebra['maximum_state']:.6f}]`.",
        "",
        "## Stage 1: active conflict versus exact zero-conflict ablation",
        "",
        stage1.to_markdown(index=False),
        "",
    ]
    if not stage1_passed:
        evidence = "The zero-interference ablation matched or beat active CTGI on at least one StarCoder schedule; stage 2 was not run."
        update_status("blocked_before_nested_starcoder", evidence, args.output_dir)
        report.extend(
            ["## Decision", "", f"**Blocked.** {evidence}", "", "No Delphi or adversarial outcomes were evaluated."]
        )
        (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
        print(stage1.to_string(index=False))
        print(evidence)
        return

    shape_reference = pd.read_csv(SHAPE_REFERENCE).groupby("surface", as_index=False)["rmse"].min()
    nested_rows = []
    nested_tables = []
    optimum_rows = []
    for panel in panels:
        nested_prediction, nested_table = nested_selection(panel, all_configs, feature_cache[panel.name])
        nested_rows.append(
            {"surface": panel.name, **paired_screen.scalar_metrics(panel.two_phase_target, nested_prediction)}
        )
        nested_tables.append(nested_table)
        optimum_rows.append(raw_optimum(panel, selected_active[panel.name]))

    nested_metrics = pd.DataFrame(nested_rows).merge(shape_reference, on="surface", suffixes=("", "_reference"))
    nested_metrics["relative_to_reference"] = nested_metrics["rmse"] / nested_metrics["rmse_reference"] - 1.0
    nested_folds = pd.concat(nested_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    nested_metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    nested_folds.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)

    active_fraction = nested_folds.groupby("surface")["interference_rate"].apply(
        lambda values: float((values > 0).mean())
    )
    threshold_range = (
        nested_folds.loc[nested_folds["interference_rate"] > 0].groupby("surface")["threshold"].agg(["min", "max"])
    )
    stability = pd.concat([active_fraction.rename("active_fraction"), threshold_range], axis=1).reset_index()
    stability.to_csv(args.output_dir / "mechanism_stability.csv", index=False)
    shape_ok = bool((nested_metrics["relative_to_reference"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    direction_ok = bool(optima.loc[optima["surface"].eq("starcoder_wsd_80_20"), "late_specialization"].all())
    active_ok = bool((stability["active_fraction"] >= 0.5).all())
    global_interior = bool(
        all(selected_active[panel.name].interference_rate < max(INTERFERENCE_GRID) for panel in panels)
    )
    passed = shape_ok and optimum_ok and direction_ok and active_ok and global_interior
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"active_beats_ablation={stage1_passed}; within_5pct_shape={shape_ok}; "
        f"raw_optimum_distance_ok={optimum_ok}; wsd_late_specialization={direction_ok}; "
        f"fold_active={active_ok}; interference_interior={global_interior}."
    )
    update_status(status, evidence, args.output_dir)
    report.extend(
        [
            "## Nested StarCoder audit",
            "",
            nested_metrics.to_markdown(index=False),
            "",
            "## Mechanism stability",
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
    print(nested_metrics.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
