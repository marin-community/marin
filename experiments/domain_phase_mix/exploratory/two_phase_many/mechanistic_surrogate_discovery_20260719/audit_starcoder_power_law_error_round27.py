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
"""Falsify the frozen power-law error-kinetics batch on StarCoder."""

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
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_shared_private_round25 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    power_law_error_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metric_lib,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round27_power_law_error_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
SHAPE_REFERENCE = audit.SHAPE_REFERENCE
FOUNDATION_RATE_GRID = (0.5, 2.0, 8.0, 32.0)
SPECIALIST_RATE_GRID = (0.5, 2.0, 8.0, 32.0)
RARE_FOUNDATION_GRID = (0.0, 0.1, 0.3, 1.0)
PREREQUISITE_GRID = (0.0, 0.5, 1.0)
FORGETTING_GRID = (0.0, 0.5, 2.0, 8.0)
POWER_GRID = (0.0, 0.5, 1.0)
L2_GRID = (0.1, 1.0)
FEATURE_NAMES = ("foundation_error", "specialist_error", "broad_replay", "specialist_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def gated_configs() -> list[candidate.GatedPowerConfig]:
    return [
        candidate.GatedPowerConfig(foundation, specialist, rare_efficiency, prerequisite, power, l2)
        for foundation in FOUNDATION_RATE_GRID
        for specialist in SPECIALIST_RATE_GRID
        for rare_efficiency in RARE_FOUNDATION_GRID
        for prerequisite in PREREQUISITE_GRID
        for power in POWER_GRID
        for l2 in L2_GRID
    ]


def forgetting_configs() -> list[candidate.ForgettingPowerConfig]:
    return [
        candidate.ForgettingPowerConfig(foundation, specialist, rare_efficiency, forgetting, power, l2)
        for foundation in FOUNDATION_RATE_GRID
        for specialist in SPECIALIST_RATE_GRID
        for rare_efficiency in RARE_FOUNDATION_GRID[1:]
        for forgetting in FORGETTING_GRID
        for power in POWER_GRID
        for l2 in L2_GRID
    ]


def build_designs(
    panel: paired.PairedPanel,
    candidate_id: str,
    configs: list[candidate.GatedPowerConfig] | list[candidate.ForgettingPowerConfig],
    weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    policies = panel.weights if weights is None else weights
    if candidate_id == "PLSC":
        return [
            candidate.gated_power_design(policies, panel.alpha0, panel.c0, panel.c1, config)[0] for config in configs
        ]
    return [
        candidate.forgetting_power_design(policies, panel.alpha0, panel.c0, panel.c1, config)[0] for config in configs
    ]


def raw_optimum(
    panel: paired.PairedPanel,
    candidate_id: str,
    config: candidate.GatedPowerConfig | candidate.ForgettingPowerConfig,
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
    head = audit.fit_head(fit_design, panel.two_phase_target, np.arange(panel.n), config.l2)
    prediction = head.predict(design)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    record = {
        "candidate": candidate_id,
        "surface": panel.name,
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
    return record, pd.DataFrame(
        {"phase0_rare_weight": p0.ravel(), "phase1_rare_weight": p1.ravel(), "predicted_bpb": prediction}
    )


def best_constrained_rmse(configs: list[Any], rmse: np.ndarray, field: str, value: float) -> float:
    mask = np.asarray([getattr(config, field) == value for config in configs], dtype=bool)
    if not mask.any():
        raise ValueError(f"No configs satisfy {field}={value}")
    return float(np.min(rmse[mask]))


def update_registry(status: pd.DataFrame) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Registry has no header")
    for row in rows:
        match = status[status["candidate"] == row["id"]]
        if match.empty:
            continue
        record = match.iloc[0].to_dict()
        row["status"] = "promoted_after_starcoder" if bool(record["passes_shape_gate"]) else "blocked_before_multi_swarm"
        row["status_evidence"] = "; ".join(
            f"{key}={value}" for key, value in record.items() if key not in {"candidate", "passes_shape_gate"}
        )
    temporary = REGISTRY.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def mechanism_field(candidate_id: str) -> str:
    return "prerequisite_power" if candidate_id == "PLSC" else "forgetting_rate"


def mechanism_active(candidate_id: str, config: Any) -> bool:
    return getattr(config, mechanism_field(candidate_id)) > 0.0


def rates_inside_grid(config: Any) -> bool:
    return config.foundation_rate not in {
        min(FOUNDATION_RATE_GRID),
        max(FOUNDATION_RATE_GRID),
    } and config.specialist_rate not in {min(SPECIALIST_RATE_GRID), max(SPECIALIST_RATE_GRID)}


def compatible_configs(left: Any, right: Any) -> bool:
    return (
        audit.regime_compatible(left.foundation_rate, right.foundation_rate)
        and audit.regime_compatible(left.specialist_rate, right.specialist_rate)
        and audit.regime_compatible(left.rare_foundation_efficiency, right.rare_foundation_efficiency)
        and audit.regime_compatible(left.learning_curve_power, right.learning_curve_power)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    batches: list[tuple[str, list[Any]]] = [("PLSC", gated_configs()), ("PLAFK", forgetting_configs())]
    metrics_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    nested_rows: list[pd.DataFrame] = []
    region_rows: list[dict[str, Any]] = []
    one_phase_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str], Any] = {}
    constrained_rows: list[dict[str, Any]] = []

    for candidate_id, configs in batches:
        for panel in panels:
            designs = build_designs(panel, candidate_id, configs)
            rmse, _ = audit.score_configs(panel, configs, designs, starcoder.surface_folds(panel))
            selected_index = int(np.argmin(rmse))
            selected = configs[selected_index]
            selections[(candidate_id, panel.name)] = selected
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
            constrained_rows.append(
                {
                    "candidate": candidate_id,
                    "surface": panel.name,
                    "selected_rmse": float(rmse[selected_index]),
                    "best_power_zero_rmse": best_constrained_rmse(configs, rmse, "learning_curve_power", 0.0),
                    "best_mechanism_zero_rmse": best_constrained_rmse(configs, rmse, mechanism_field(candidate_id), 0.0),
                }
            )
            nested, nested_selection = audit.nested_prediction(panel, configs, designs)
            nested_selection.insert(0, "candidate", candidate_id)
            nested_rows.append(nested_selection)
            metrics_rows.append(
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
                {"candidate": candidate_id, **row}
                for row in audit.leave_region_out(panel, selected, designs[selected_index])
            )
            tied_metrics, tied_index = audit.independently_fit_tied(panel, configs, designs)
            one_phase_rows.append(
                {
                    "candidate": candidate_id,
                    "surface": panel.name,
                    "selected_config_index": tied_index,
                    "selected_config": json.dumps(asdict(configs[tied_index]), sort_keys=True),
                    **tied_metrics,
                }
            )
            head = audit.fit_head(designs[selected_index], panel.two_phase_target, np.arange(panel.n), selected.l2)
            for name, value in zip(FEATURE_NAMES, head.coefficients_in_natural_units, strict=True):
                coefficient_rows.append(
                    {"candidate": candidate_id, "surface": panel.name, "feature": name, "coefficient": float(value)}
                )
            optimum, surface = raw_optimum(panel, candidate_id, selected, designs[selected_index])
            optimum_rows.append(optimum)
            surface.to_csv(args.output_dir / f"{candidate_id}__{panel.name}__surface.csv", index=False)
            audit.render_surface(
                panel, candidate_id, surface, args.output_dir / f"{candidate_id}__{panel.name}__surface.html"
            )

    metrics = pd.DataFrame(metrics_rows)
    configs_table = pd.DataFrame(config_rows)
    nested = pd.concat(nested_rows, ignore_index=True)
    regions = pd.DataFrame(region_rows)
    one_phase = pd.DataFrame(one_phase_rows)
    optima = pd.DataFrame(optimum_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    constrained = pd.DataFrame(constrained_rows)
    status_rows = []
    for candidate_id, _configs in batches:
        chosen_cosine = selections[(candidate_id, "starcoder_cosine_50_50")]
        chosen_wsd = selections[(candidate_id, "starcoder_wsd_80_20")]
        candidate_metrics = metrics[metrics["candidate"] == candidate_id].set_index("surface")
        candidate_optima = optima[optima["candidate"] == candidate_id]
        candidate_constrained = constrained[constrained["candidate"] == candidate_id]
        candidate_nested = nested[nested["candidate"] == candidate_id]
        replay_active_both = all(
            float(
                coefficients[
                    (coefficients["candidate"] == candidate_id)
                    & (coefficients["surface"] == surface)
                    & (coefficients["feature"].isin(["broad_replay", "specialist_replay"]))
                ]["coefficient"].max()
            )
            > 1e-10
            for surface in SHAPE_REFERENCE
        )
        within_reference = all(
            float(candidate_metrics.loc[surface, "nested_rmse"]) <= 1.05 * reference
            for surface, reference in SHAPE_REFERENCE.items()
        )
        status = {
            "candidate": candidate_id,
            "algebraic_tied_error": (
                candidate.tied_error(candidate.gated_power_terminal_errors, chosen_cosine)
                if candidate_id == "PLSC"
                else candidate.tied_error(candidate.forgetting_power_terminal_errors, chosen_cosine)
            ),
            "power_global_both": chosen_cosine.learning_curve_power > 0.0 and chosen_wsd.learning_curve_power > 0.0,
            "power_fold_majority_both": all(
                (candidate_nested[candidate_nested["surface"] == surface]["learning_curve_power"] > 0.0).mean() >= 0.6
                for surface in SHAPE_REFERENCE
            ),
            "mechanism_global_both": mechanism_active(candidate_id, chosen_cosine)
            and mechanism_active(candidate_id, chosen_wsd),
            "mechanism_fold_majority_both": all(
                (candidate_nested[candidate_nested["surface"] == surface][mechanism_field(candidate_id)] > 0.0).mean()
                >= 0.6
                for surface in SHAPE_REFERENCE
            ),
            "beats_power_zero_both": bool(
                (candidate_constrained["selected_rmse"] < candidate_constrained["best_power_zero_rmse"] - 1e-6).all()
            ),
            "beats_mechanism_zero_both": bool(
                (candidate_constrained["selected_rmse"] < candidate_constrained["best_mechanism_zero_rmse"] - 1e-6).all()
            ),
            "replay_active_both": replay_active_both,
            "rates_not_boundary": rates_inside_grid(chosen_cosine) and rates_inside_grid(chosen_wsd),
            "regime_transfer": compatible_configs(chosen_cosine, chosen_wsd),
            "within_5pct_shape_reference": within_reference,
            "optimum_distance_ok": bool((candidate_optima["optimum_distance"] <= 0.15).all()),
        }
        status["passes_shape_gate"] = bool(
            status["algebraic_tied_error"] < 1e-8
            and all(value for key, value in status.items() if key not in {"candidate", "algebraic_tied_error"})
        )
        status_rows.append(status)
    status_table = pd.DataFrame(status_rows)
    update_registry(status_table)

    metrics.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    configs_table.to_csv(args.output_dir / "config_grid.csv", index=False)
    nested.to_csv(args.output_dir / "nested_selections.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    one_phase.to_csv(args.output_dir / "independent_one_phase_refit.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    coefficients.to_csv(args.output_dir / "full_fit_coefficients.csv", index=False)
    constrained.to_csv(args.output_dir / "nested_ablation_metrics.csv", index=False)
    status_table.to_csv(args.output_dir / "gate_status.csv", index=False)
    report = [
        "# Round 27: power-law error kinetics",
        "",
        "Both candidates and all grids were frozen before either StarCoder surface was fit. No Delphi heldout or adversarial outcome was read.",
        "",
        "## Surface and nested OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Exact nested ablations",
        "",
        constrained.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full-fit response coefficients",
        "",
        coefficients.to_markdown(index=False, floatfmt=".6f"),
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
        status_table.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(metrics.to_string(index=False))
    print("\nGate status")
    print(status_table.to_string(index=False))


if __name__ == "__main__":
    main()
