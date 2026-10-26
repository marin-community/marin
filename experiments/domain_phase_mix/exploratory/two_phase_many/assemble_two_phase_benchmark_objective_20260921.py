# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Assemble complete component predictions with the benchmark's frozen weights.

Missing components fail assembly rather than changing the objective weights.
Only prepared held-out identities are admissible. This script fits no model.
"""

import argparse
import json
from pathlib import Path

import benchmark_two_phase_20260921 as benchmark
import numpy as np
import pandas as pd

IDENTITY = ["row_id", "policy_id", "group_id", "source", "region", "role"]


def assemble(args: argparse.Namespace) -> None:
    """Validate complete native component menus and save weighted objective scores."""
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark.verify_hashes(manifest["sources"])
    benchmark.verify_hashes(manifest["outputs"], args.prepared)
    if manifest["identity"]["code"] != benchmark.code_identity():
        raise ValueError("Benchmark code differs from its prepared snapshot")
    if manifest["identity"]["environment"] != benchmark.environment_identity():
        raise ValueError("Numerical environment differs from its prepared snapshot")
    panel = benchmark.read_panel(args.prepared, args.panel)
    spec = panel.metadata["objective_components"][args.objective]
    components = spec["components"]
    weights = np.asarray(spec["weights"], dtype=float)
    frame = pd.read_csv(args.predictions, keep_default_na=False, float_precision="round_trip")
    if set(frame.panel) != {args.panel} or set(frame.target) != set(components):
        raise ValueError("Prediction input must contain exactly the requested panel and objective components")
    benchmark.scoring.score_predictions(frame)
    folds = pd.read_csv(args.prepared / args.panel / "folds.csv", dtype={"row_id": str})
    prepared_rows = panel.rows.set_index("row_id")
    position = {row: index for index, row in enumerate(panel.rows.row_id)}
    objective_rows = []
    maximum_component_error = 0.0
    maximum_objective_error = 0.0
    for (model, repeat, fold), group in frame.groupby(["model", "repeat", "outer_fold"], sort=True):
        expected = folds[(folds.repeat == repeat) & (folds.outer_fold == fold) & (folds.role != "train")]
        expected = expected.set_index("row_id").sort_index()
        if len(expected) == 0:
            raise ValueError("Predictions have no corresponding prepared held-out menu")
        observed = []
        predicted = []
        template = None
        for component in components:
            part = group[group.target == component].set_index("row_id").sort_index()
            if list(part.index) != list(expected.index):
                raise ValueError(f"Missing component rows or changed held-out menu for {component}")
            if not np.array_equal(part.role, expected.role):
                raise ValueError("A component has changed prepared roles")
            for column in IDENTITY[1:-1]:
                if not np.array_equal(part[column], prepared_rows.loc[part.index, column]):
                    raise ValueError(f"A component has changed prepared {column} identities")
            indices = np.array([position[row] for row in part.index])
            error = float(np.max(np.abs(part.observed.to_numpy() - panel.targets[component][indices])))
            maximum_component_error = max(error, maximum_component_error)
            if error > 2e-13:
                raise ValueError(f"Component observations differ from the prepared outcomes: {component}")
            observed.append(part.observed.to_numpy())
            predicted.append(part.predicted.to_numpy())
            if template is None:
                template = part.reset_index()[IDENTITY].copy()
        if template is None:
            raise ValueError("The objective has no components")
        template["model"] = model
        template["panel"] = args.panel
        template["target"] = args.objective
        template["repeat"] = repeat
        template["outer_fold"] = fold
        template["observed"] = np.column_stack(observed) @ weights
        template["predicted"] = np.column_stack(predicted) @ weights
        indices = np.array([position[row] for row in template.row_id])
        error = float(np.max(np.abs(template.observed.to_numpy() - panel.targets[args.objective][indices])))
        maximum_objective_error = max(error, maximum_objective_error)
        if error > 2e-13:
            raise ValueError("The fixed-weight reconstruction differs from the prepared objective")
        objective_rows.append(template)
    assembled = pd.concat(objective_rows, ignore_index=True)
    for (_, repeat), group in assembled.groupby(["model", "repeat"]):
        expected = folds[(folds.repeat == repeat) & (folds.role != "train")]
        observed_keys = set(zip(group.outer_fold, group.row_id, strict=True))
        expected_keys = set(zip(expected.outer_fold, expected.row_id, strict=True))
        if observed_keys != expected_keys:
            raise ValueError("The component predictions do not cover every prepared outer fold")
    args.output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([frame, assembled], ignore_index=True)
    for name, predictions in (("component", frame), ("objective", assembled), ("all", combined)):
        predictions.to_csv(args.output / f"{name}_predictions.csv", index=False)
        benchmark.scoring.score_predictions(predictions).to_csv(args.output / f"{name}_metrics.csv", index=False)
        benchmark.scoring.summarize_predictions(predictions).to_csv(args.output / f"{name}_summary.csv", index=False)
    benchmark.write_json(
        args.output / "fixed_weights_receipt.json",
        {
            "status": "complete_component_assembly_verified",
            "claim": "adaptive_retrospective_reference_predictions_only",
            "panel": args.panel,
            "objective": args.objective,
            "components": components,
            "weights": weights.tolist(),
            "weights_sum": float(np.sum(weights)),
            "renormalization": False,
            "component_observed_max_abs_error": maximum_component_error,
            "objective_observed_max_abs_error": maximum_objective_error,
            "component_rows": len(frame),
            "objective_rows": len(assembled),
            "sources": {
                str(args.predictions.resolve()): benchmark.sha256(args.predictions),
                str(manifest_path.resolve()): benchmark.sha256(manifest_path),
                str((args.prepared / args.panel / "metadata.json").resolve()): benchmark.sha256(
                    args.prepared / args.panel / "metadata.json"
                ),
                str(Path(__file__).resolve()): benchmark.sha256(Path(__file__)),
            },
            "benchmark_code": benchmark.code_identity(),
            "outputs": {
                path.name: benchmark.sha256(path)
                for path in sorted(args.output.glob("*_*.csv"))
                if path.name.startswith(("component_", "objective_", "all_"))
            },
            "gate_status": "reference_artifacts_ready; TPP40 numerical promotion policy not frozen",
            "limitation": "Arbitrary held-out menus cannot establish the existence or absence of matched 2p gain.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--objective", required=True)
    assemble(parser.parse_args())


if __name__ == "__main__":
    main()
