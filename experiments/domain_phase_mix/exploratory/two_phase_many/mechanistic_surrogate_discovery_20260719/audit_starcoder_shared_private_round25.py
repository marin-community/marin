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
"""Falsify the frozen shared/private competence batch on StarCoder."""

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
import plotly.graph_objects as go
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metric_lib,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    shared_private_models as candidate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round25_shared_private_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.065388405808633, "starcoder_wsd_80_20": 0.0457725108696099}
FOUNDATION_RATE_GRID = (0.5, 2.0, 8.0, 32.0)
SPECIALIST_RATE_GRID = (0.5, 2.0, 8.0, 32.0)
RARE_FOUNDATION_GRID = (0.0, 0.1, 0.3, 1.0)
PREREQUISITE_GRID = (0.0, 0.5, 1.0, 2.0)
FLOW_SPEED_GRID = (0.5, 2.0, 8.0, 32.0)
BROAD_DECAY_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
L2_GRID = (0.1, 1.0)
SEED = 20260719
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def cascade_configs() -> list[candidate.CascadeConfig]:
    return [
        candidate.CascadeConfig(foundation, specialist, rare_efficiency, prerequisite, l2)
        for foundation in FOUNDATION_RATE_GRID
        for specialist in SPECIALIST_RATE_GRID
        for rare_efficiency in RARE_FOUNDATION_GRID
        for prerequisite in PREREQUISITE_GRID
        for l2 in L2_GRID
    ]


def factorized_configs() -> list[candidate.FactorizedFlowConfig]:
    return [
        candidate.FactorizedFlowConfig(speed, decay, rare_efficiency, l2)
        for speed in FLOW_SPEED_GRID
        for decay in BROAD_DECAY_GRID
        for rare_efficiency in RARE_FOUNDATION_GRID[1:]
        for l2 in L2_GRID
    ]


def build_designs(
    panel: paired.PairedPanel,
    candidate_id: str,
    configs: list[candidate.CascadeConfig] | list[candidate.FactorizedFlowConfig],
    weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    policies = panel.weights if weights is None else weights
    if candidate_id == "FSC":
        return [candidate.cascade_design(policies, panel.alpha0, config)[0] for config in configs]
    return [candidate.factorized_design(policies, panel.alpha0, config)[0] for config in configs]


def fit_head(design: np.ndarray, target: np.ndarray, train: np.ndarray, l2: float) -> paired.LinearHead:
    return paired.fit_linear_head(
        design[train],
        target[train],
        [f"state_error_{index}" for index in range(design.shape[1])],
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=l2,
    )


def score_configs(
    panel: paired.PairedPanel,
    configs: list[candidate.CascadeConfig] | list[candidate.FactorizedFlowConfig],
    designs: list[np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    required_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full((len(configs), panel.n), np.nan, dtype=float)
    for config_index, (config, design) in enumerate(zip(configs, designs, strict=True)):
        for train, test in folds:
            head = fit_head(design, panel.two_phase_target, train, config.l2)
            predictions[config_index, test] = head.predict(design[test])
    required = np.arange(panel.n) if required_indices is None else required_indices
    if not np.isfinite(predictions[:, required]).all():
        raise RuntimeError(f"Incomplete OOF predictions for {panel.name}")
    rmse = np.sqrt(np.mean((predictions[:, required] - panel.two_phase_target[None, required]) ** 2, axis=1))
    return rmse, predictions


def inner_folds(panel: paired.PairedPanel, indices: np.ndarray, seed_offset: int) -> list[tuple[np.ndarray, np.ndarray]]:
    tied = indices[panel.paired_mask[indices]]
    untied = indices[~panel.paired_mask[indices]]
    splits = min(4, len(tied), len(untied))
    if splits < 2:
        raise ValueError(f"Insufficient tied/untied rows for nested CV on {panel.name}")
    tied_folds = list(KFold(splits, shuffle=True, random_state=SEED + seed_offset).split(tied))
    untied_folds = list(KFold(splits, shuffle=True, random_state=SEED + 100 + seed_offset).split(untied))
    return [
        (
            np.sort(np.concatenate([tied[tied_train], untied[untied_train]])),
            np.sort(np.concatenate([tied[tied_test], untied[untied_test]])),
        )
        for (tied_train, tied_test), (untied_train, untied_test) in zip(tied_folds, untied_folds, strict=True)
    ]


def nested_prediction(
    panel: paired.PairedPanel,
    configs: list[candidate.CascadeConfig] | list[candidate.FactorizedFlowConfig],
    designs: list[np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        scores, _ = score_configs(
            panel,
            configs,
            designs,
            inner_folds(panel, outer_train, outer_fold),
            outer_train,
        )
        selected_index = int(np.argmin(scores))
        selected = configs[selected_index]
        design = designs[selected_index]
        head = fit_head(design, panel.two_phase_target, outer_train, selected.l2)
        prediction[outer_test] = head.predict(design[outer_test])
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": outer_fold,
                "inner_rmse": float(scores[selected_index]),
                **asdict(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def leave_region_out(
    panel: paired.PairedPanel,
    config: candidate.CascadeConfig | candidate.FactorizedFlowConfig,
    design: np.ndarray,
) -> list[dict[str, Any]]:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        head = fit_head(design, panel.two_phase_target, train, config.l2)
        prediction = head.predict(design[test])
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **metric_lib.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def independently_fit_tied(
    panel: paired.PairedPanel,
    configs: list[candidate.CascadeConfig] | list[candidate.FactorizedFlowConfig],
    designs: list[np.ndarray],
) -> tuple[dict[str, Any], int]:
    tied = np.flatnonzero(panel.paired_mask)
    if len(tied) < 4:
        return {"n": len(tied), "rmse": np.nan}, 0
    config_predictions = np.full((len(configs), len(tied)), np.nan, dtype=float)
    for local_test, test in enumerate(tied):
        train = tied[tied != test]
        for config_index, (config, design) in enumerate(zip(configs, designs, strict=True)):
            head = fit_head(design, panel.two_phase_target, train, config.l2)
            config_predictions[config_index, local_test] = head.predict(design[[test]])[0]
    rmse = np.sqrt(np.mean((config_predictions - panel.two_phase_target[tied][None, :]) ** 2, axis=1))
    selected = int(np.argmin(rmse))
    return metric_lib.scalar_metrics(panel.two_phase_target[tied], config_predictions[selected]), selected


def raw_optimum(
    panel: paired.PairedPanel,
    candidate_id: str,
    config: candidate.CascadeConfig | candidate.FactorizedFlowConfig,
    fit_design: np.ndarray,
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
    design = build_designs(panel, candidate_id, [config], weights)[0]
    head = fit_head(fit_design, panel.two_phase_target, np.arange(panel.n), config.l2)
    prediction = head.predict(design)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    record = {
        "surface": panel.name,
        "candidate": candidate_id,
        "predicted_p0": float(p0.ravel()[best]),
        "predicted_p1": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_p0": float(panel.weights[observed, 0, 1]),
        "observed_p1": float(panel.weights[observed, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed]),
        "optimum_distance": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed, 0, 1],
                p1.ravel()[best] - panel.weights[observed, 1, 1],
            )
        ),
    }
    surface = pd.DataFrame(
        {"phase0_rare_weight": p0.ravel(), "phase1_rare_weight": p1.ravel(), "predicted_bpb": prediction}
    )
    return record, surface


def render_surface(panel: paired.PairedPanel, candidate_id: str, surface: pd.DataFrame, output: Path) -> None:
    figure = go.Figure(
        [
            go.Mesh3d(
                x=surface["phase0_rare_weight"],
                y=surface["phase1_rare_weight"],
                z=surface["predicted_bpb"],
                intensity=surface["predicted_bpb"],
                colorscale="RdYlGn_r",
                opacity=0.55,
                name="Predicted surface",
            ),
            go.Scatter3d(
                x=panel.weights[:, 0, 1],
                y=panel.weights[:, 1, 1],
                z=panel.two_phase_target,
                mode="markers",
                marker={"size": 5, "color": panel.two_phase_target, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
        ]
    )
    figure.update_layout(
        title=f"{candidate_id}: {panel.name}",
        template="plotly_white",
        height=850,
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def regime_compatible(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return max(left, right) / min(left, right) <= 4.0


def update_registry(status_rows: pd.DataFrame) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Registry has no header")
    for row in rows:
        match = status_rows[status_rows["candidate"] == row["id"]]
        if match.empty:
            continue
        gates = match.iloc[0].to_dict()
        passed = bool(gates["passes_shape_gate"])
        row["status"] = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
        row["status_evidence"] = "; ".join(
            f"{key}={value}" for key, value in gates.items() if key not in {"candidate", "passes_shape_gate"}
        )
    temporary = REGISTRY.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    batches: list[tuple[str, list[Any]]] = [("FSC", cascade_configs()), ("FPCGF", factorized_configs())]
    metric_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    nested_rows: list[pd.DataFrame] = []
    region_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    one_phase_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str], Any] = {}
    selected_indices: dict[tuple[str, str], int] = {}
    for candidate_id, configs in batches:
        for panel in panels:
            designs = build_designs(panel, candidate_id, configs)
            rmse, _ = score_configs(panel, configs, designs, starcoder.surface_folds(panel))
            selected_index = int(np.argmin(rmse))
            selected = configs[selected_index]
            selections[(candidate_id, panel.name)] = selected
            selected_indices[(candidate_id, panel.name)] = selected_index
            for config_index, config in enumerate(configs):
                config_rows.append(
                    {
                        "candidate": candidate_id,
                        "surface": panel.name,
                        "config_index": config_index,
                        "oof_rmse": float(rmse[config_index]),
                        **asdict(config),
                    }
                )
            nested, nested_selection = nested_prediction(panel, configs, designs)
            nested_selection.insert(0, "candidate", candidate_id)
            nested_rows.append(nested_selection)
            metric_rows.append(
                {
                    "candidate": candidate_id,
                    "surface": panel.name,
                    "selected_config": json.dumps(asdict(selected), sort_keys=True),
                    "global_oof_rmse": float(rmse[selected_index]),
                    **{
                        f"nested_{key}": value
                        for key, value in metric_lib.scalar_metrics(panel.two_phase_target, nested).items()
                    },
                }
            )
            region_rows.extend(
                {"candidate": candidate_id, **row} for row in leave_region_out(panel, selected, designs[selected_index])
            )
            tied_metrics, tied_config_index = independently_fit_tied(panel, configs, designs)
            one_phase_rows.append(
                {
                    "candidate": candidate_id,
                    "surface": panel.name,
                    "selected_config_index": tied_config_index,
                    "selected_config": json.dumps(asdict(configs[tied_config_index]), sort_keys=True),
                    **tied_metrics,
                }
            )
            optimum, surface = raw_optimum(panel, candidate_id, selected, designs[selected_index])
            optimum_rows.append(optimum)
            surface.to_csv(args.output_dir / f"{candidate_id}__{panel.name}__surface.csv", index=False)
            render_surface(panel, candidate_id, surface, args.output_dir / f"{candidate_id}__{panel.name}__surface.html")

    metrics = pd.DataFrame(metric_rows)
    configs_table = pd.DataFrame(config_rows)
    nested_table = pd.concat(nested_rows, ignore_index=True)
    regions = pd.DataFrame(region_rows)
    optima = pd.DataFrame(optimum_rows)
    one_phase = pd.DataFrame(one_phase_rows)
    status_rows = []
    for candidate_id, _configs in batches:
        candidate_metrics = metrics[metrics["candidate"] == candidate_id].set_index("surface")
        candidate_optima = optima[optima["candidate"] == candidate_id].set_index("surface")
        selected_cosine = selections[(candidate_id, "starcoder_cosine_50_50")]
        selected_wsd = selections[(candidate_id, "starcoder_wsd_80_20")]
        if candidate_id == "FSC":
            mechanism_global_both = selected_cosine.prerequisite_power > 0.0 and selected_wsd.prerequisite_power > 0.0
            fold_table = nested_table[nested_table["candidate"] == candidate_id]
            mechanism_fold_majority_both = all(
                (fold_table[fold_table["surface"] == surface]["prerequisite_power"] > 0.0).mean() >= 0.6
                for surface in SHAPE_REFERENCE
            )
            rates_not_boundary = all(
                config.foundation_rate not in {min(FOUNDATION_RATE_GRID), max(FOUNDATION_RATE_GRID)}
                and config.specialist_rate not in {min(SPECIALIST_RATE_GRID), max(SPECIALIST_RATE_GRID)}
                for config in (selected_cosine, selected_wsd)
            )
            regime_transfer = (
                regime_compatible(selected_cosine.foundation_rate, selected_wsd.foundation_rate)
                and regime_compatible(selected_cosine.specialist_rate, selected_wsd.specialist_rate)
                and regime_compatible(
                    selected_cosine.rare_foundation_efficiency,
                    selected_wsd.rare_foundation_efficiency,
                )
            )
        else:
            mechanism_global_both = (
                selected_cosine.broad_specialist_decay > 0.0 and selected_wsd.broad_specialist_decay > 0.0
            )
            fold_table = nested_table[nested_table["candidate"] == candidate_id]
            mechanism_fold_majority_both = all(
                (fold_table[fold_table["surface"] == surface]["broad_specialist_decay"] > 0.0).mean() >= 0.6
                for surface in SHAPE_REFERENCE
            )
            rates_not_boundary = all(
                config.speed not in {min(FLOW_SPEED_GRID), max(FLOW_SPEED_GRID)}
                and config.broad_specialist_decay not in {min(BROAD_DECAY_GRID), max(BROAD_DECAY_GRID)}
                for config in (selected_cosine, selected_wsd)
            )
            regime_transfer = (
                regime_compatible(selected_cosine.speed, selected_wsd.speed)
                and regime_compatible(
                    selected_cosine.broad_specialist_decay,
                    selected_wsd.broad_specialist_decay,
                )
                and regime_compatible(
                    selected_cosine.rare_foundation_efficiency,
                    selected_wsd.rare_foundation_efficiency,
                )
            )
        within_reference = all(
            float(candidate_metrics.loc[surface, "nested_rmse"]) <= 1.05 * reference
            for surface, reference in SHAPE_REFERENCE.items()
        )
        optimum_ok = bool((candidate_optima["optimum_distance"] <= 0.15).all())
        gates = {
            "candidate": candidate_id,
            "algebraic_tied_error": (
                candidate.tied_policy_error(candidate.cascade_terminal_state, selected_cosine)
                if candidate_id == "FSC"
                else candidate.tied_policy_error(candidate.factorized_terminal_state, selected_cosine)
            ),
            "mechanism_global_both": mechanism_global_both,
            "mechanism_fold_majority_both": mechanism_fold_majority_both,
            "rates_not_boundary": rates_not_boundary,
            "regime_transfer": regime_transfer,
            "within_5pct_shape_reference": within_reference,
            "optimum_distance_ok": optimum_ok,
        }
        gates["passes_shape_gate"] = bool(
            gates["algebraic_tied_error"] < 1e-8
            and mechanism_global_both
            and mechanism_fold_majority_both
            and rates_not_boundary
            and regime_transfer
            and within_reference
            and optimum_ok
        )
        status_rows.append(gates)
    status = pd.DataFrame(status_rows)

    metrics.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    configs_table.to_csv(args.output_dir / "config_grid.csv", index=False)
    nested_table.to_csv(args.output_dir / "nested_selections.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    one_phase.to_csv(args.output_dir / "independent_one_phase_refit.csv", index=False)
    status.to_csv(args.output_dir / "gate_status.csv", index=False)
    update_registry(status)
    report = [
        "# Round 25: shared/private competence batch",
        "",
        "Both candidate equations and all grids were frozen before either surface was fit. Historical and adversarial Delphi outcomes were not read.",
        "",
        "## Surface and nested OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Independently fitted one-phase restriction",
        "",
        one_phase.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Frozen gate",
        "",
        status.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(metrics.to_string(index=False))
    print("\nGate status")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
