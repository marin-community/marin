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
"""Falsify the frozen replay-complete foundation cascade on StarCoder."""

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
    audit_starcoder_shared_private_round25 as round25,
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
DEFAULT_OUTPUT = OUTPUT_ROOT / "round26_cascade_replay_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
ROUND25_METRICS = OUTPUT_ROOT / "round25_shared_private_starcoder/surface_metrics.csv"
SHAPE_REFERENCE = round25.SHAPE_REFERENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def replay_designs(
    panel: paired.PairedPanel,
    configs: list[candidate.CascadeConfig],
    weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    policies = panel.weights if weights is None else weights
    return [candidate.cascade_replay_design(policies, panel.alpha0, panel.c0, panel.c1, config)[0] for config in configs]


def raw_optimum(
    panel: paired.PairedPanel,
    config: candidate.CascadeConfig,
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
    design = replay_designs(panel, [config], weights)[0]
    head = round25.fit_head(fit_design, panel.two_phase_target, np.arange(panel.n), config.l2)
    prediction = head.predict(design)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    record = {
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
    surface = pd.DataFrame(
        {"phase0_rare_weight": p0.ravel(), "phase1_rare_weight": p1.ravel(), "predicted_bpb": prediction}
    )
    return record, surface


def update_registry(status: dict[str, Any]) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Registry has no header")
    for row in rows:
        if row["id"] != "FSCR":
            continue
        passed = bool(status["passes_shape_gate"])
        row["status"] = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
        row["status_evidence"] = "; ".join(
            f"{key}={value}" for key, value in status.items() if key != "passes_shape_gate"
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
    configs = round25.cascade_configs()
    metric_rows = []
    config_rows = []
    nested_rows = []
    region_rows = []
    one_phase_rows = []
    optimum_rows = []
    coefficient_rows = []
    selections: dict[str, candidate.CascadeConfig] = {}
    for panel in panels:
        designs = replay_designs(panel, configs)
        rmse, _ = round25.score_configs(panel, configs, designs, starcoder.surface_folds(panel))
        selected_index = int(np.argmin(rmse))
        selected = configs[selected_index]
        selections[panel.name] = selected
        for config_index, config in enumerate(configs):
            config_rows.append(
                {
                    "surface": panel.name,
                    "config_index": config_index,
                    "oof_rmse": float(rmse[config_index]),
                    **asdict(config),
                }
            )
        nested, nested_selection = round25.nested_prediction(panel, configs, designs)
        nested_rows.append(nested_selection)
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": json.dumps(asdict(selected), sort_keys=True),
                "global_oof_rmse": float(rmse[selected_index]),
                **{
                    f"nested_{key}": value
                    for key, value in metric_lib.scalar_metrics(panel.two_phase_target, nested).items()
                },
            }
        )
        region_rows.extend(round25.leave_region_out(panel, selected, designs[selected_index]))
        tied_metrics, tied_index = round25.independently_fit_tied(panel, configs, designs)
        one_phase_rows.append(
            {
                "surface": panel.name,
                "selected_config_index": tied_index,
                "selected_config": json.dumps(asdict(configs[tied_index]), sort_keys=True),
                **tied_metrics,
            }
        )
        full_head = round25.fit_head(designs[selected_index], panel.two_phase_target, np.arange(panel.n), selected.l2)
        for name, value in zip(
            ("foundation_error", "specialist_error", "broad_replay", "specialist_replay"),
            full_head.coefficients_in_natural_units,
            strict=True,
        ):
            coefficient_rows.append({"surface": panel.name, "feature": name, "coefficient": float(value)})
        optimum, surface = raw_optimum(panel, selected, designs[selected_index])
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        round25.render_surface(panel, "FSCR", surface, args.output_dir / f"{panel.name}__surface.html")

    metrics = pd.DataFrame(metric_rows)
    config_table = pd.DataFrame(config_rows)
    nested_table = pd.concat(nested_rows, ignore_index=True)
    regions = pd.DataFrame(region_rows)
    one_phase = pd.DataFrame(one_phase_rows)
    optima = pd.DataFrame(optimum_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    round25_metrics = pd.read_csv(ROUND25_METRICS)
    fsc_metrics = round25_metrics[round25_metrics["candidate"] == "FSC"].set_index("surface")
    selected_cosine = selections["starcoder_cosine_50_50"]
    selected_wsd = selections["starcoder_wsd_80_20"]
    fold_mechanism = all(
        (nested_table[nested_table["surface"] == surface]["prerequisite_power"] > 0.0).mean() >= 0.6
        for surface in SHAPE_REFERENCE
    )
    replay_active = all(
        float(
            coefficients[(coefficients["surface"] == surface) & (coefficients["feature"] == "specialist_replay")][
                "coefficient"
            ].iloc[0]
        )
        > 1e-10
        for surface in SHAPE_REFERENCE
    )
    rates_not_boundary = all(
        config.foundation_rate not in {min(round25.FOUNDATION_RATE_GRID), max(round25.FOUNDATION_RATE_GRID)}
        and config.specialist_rate not in {min(round25.SPECIALIST_RATE_GRID), max(round25.SPECIALIST_RATE_GRID)}
        for config in (selected_cosine, selected_wsd)
    )
    regime_transfer = (
        round25.regime_compatible(selected_cosine.foundation_rate, selected_wsd.foundation_rate)
        and round25.regime_compatible(selected_cosine.specialist_rate, selected_wsd.specialist_rate)
        and round25.regime_compatible(
            selected_cosine.rare_foundation_efficiency,
            selected_wsd.rare_foundation_efficiency,
        )
    )
    indexed_metrics = metrics.set_index("surface")
    within_reference = all(
        float(indexed_metrics.loc[surface, "nested_rmse"]) <= 1.05 * reference
        for surface, reference in SHAPE_REFERENCE.items()
    )
    beats_no_replay = all(
        float(indexed_metrics.loc[surface, "nested_rmse"]) < float(fsc_metrics.loc[surface, "nested_rmse"])
        for surface in SHAPE_REFERENCE
    )
    status = {
        "algebraic_tied_error": candidate.tied_policy_error(candidate.cascade_terminal_state, selected_cosine),
        "prerequisite_global_both": (selected_cosine.prerequisite_power > 0.0 and selected_wsd.prerequisite_power > 0.0),
        "prerequisite_fold_majority_both": fold_mechanism,
        "specialist_replay_active_both": replay_active,
        "beats_no_replay_nested_both": beats_no_replay,
        "rates_not_boundary": rates_not_boundary,
        "regime_transfer": regime_transfer,
        "within_5pct_shape_reference": within_reference,
        "optimum_distance_ok": bool((optima["optimum_distance"] <= 0.15).all()),
    }
    status["passes_shape_gate"] = bool(
        status["algebraic_tied_error"] < 1e-8
        and all(value for key, value in status.items() if key != "algebraic_tied_error")
    )
    update_registry(status)

    metrics.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    config_table.to_csv(args.output_dir / "config_grid.csv", index=False)
    nested_table.to_csv(args.output_dir / "nested_selections.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    one_phase.to_csv(args.output_dir / "independent_one_phase_refit.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    coefficients.to_csv(args.output_dir / "full_fit_coefficients.csv", index=False)
    pd.DataFrame([status]).to_csv(args.output_dir / "gate_status.csv", index=False)
    report = [
        "# Round 26: foundation cascade with literal replay",
        "",
        "The physical replay invariant and complete response were frozen before this candidate was fit. Historical and adversarial Delphi outcomes were not read.",
        "",
        "## Surface and nested OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
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
        pd.DataFrame([status]).to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(metrics.to_string(index=False))
    print("\nGate status")
    print(pd.DataFrame([status]).to_string(index=False))


if __name__ == "__main__":
    main()
