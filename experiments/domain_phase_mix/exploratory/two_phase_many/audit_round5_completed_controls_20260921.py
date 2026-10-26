# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Audit completed round-five cells and saved profiles without fitting anything."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import audit_round5_floor_recovery_20260921 as audit
import benchmark_two_phase_20260921 as benchmark
import pandas as pd

METRICS = (
    "matched_gain_rmse",
    "uniform_101_tied_curve_rmse",
    "raw_policy_regret",
    "absolute_selected_optimism",
    "selected_optimism",
    "predicted_max_abs_matched_gain",
    "endpoint_rmse",
)


def profile_row(cell: dict[str, Any], partition: str, profile: dict[str, Any]) -> dict[str, Any]:
    refinements = profile["refinements"]
    trace = profile["trace"]
    return cell | {
        "partition": partition,
        "gamma": profile["gamma"],
        "floor": profile["floor"],
        "bound_hits": ";".join(profile["bound_hits"]),
        "selected_deficit_clamp_count": profile["deficit_clamp_count"],
        "trial_count": len(trace),
        "conditional_solves": profile["conditional_solves"],
        "trials_with_clamps": sum(trial["deficit_clamp_count"] > 0 for trial in trace),
        "maximum_trial_clamp_count": max(trial["deficit_clamp_count"] for trial in trace),
        "refinement_count": len(refinements),
        "unsuccessful_refinements": sum(not result["success"] for result in refinements),
        "refinement_statuses": json.dumps(sorted({result["status"] for result in refinements})),
        "refinement_messages": json.dumps(sorted({result["message"] for result in refinements})),
        "constant_response": profile["constant_response"],
        "sampled_training_sse_spread": profile["sampled_training_sse_spread"],
        "training_sse": profile["training_sse"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=audit.previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, default=audit.ROOT / "synthetic_v1")
    args = parser.parse_args()
    identity = audit.identity(args.prepared)
    budgets = audit.candidate_prerequisites(args.prepared)
    replay_path = args.output / "replay.json"
    replay = json.loads(replay_path.read_text())
    if replay["identity"] != identity:
        raise ValueError("Independent replay did not cover the frozen source identity")
    pairs, profiles, receipts, decisions = [], [], [], {}
    for model in audit.CANDIDATES:
        path = args.output / model / "decision.json"
        decision = json.loads(path.read_text())
        if decision["status"] not in {"rejected", "synthetic_survivor"} or decision["identity"] != identity:
            raise ValueError("Both candidate decisions must be terminal and match the frozen sources")
        expected = audit.contexts()[: decision["completed_contexts"]]
        if [dataclasses.asdict(cell) for cell in expected] != [item["context"] for item in decision["decisions"]]:
            raise ValueError("Completed cells are not the registered ordered prefix")
        if replay["counts_by_model"][model] != len(expected):
            raise ValueError("Replay has a different completed candidate count")
        decisions[model] = {"status": decision["status"], "completed_contexts": len(expected)}
        receipts.append(benchmark.receipt(path))
        for cell, cell_decision in zip(expected, decision["decisions"], strict=True):
            candidate = audit.replay_cell(args.prepared, args.output, cell, model)
            control = audit.replay_cell(args.prepared, args.output, cell, audit.CONTROLS[model])
            actual_failures = audit.budget_failures(candidate, budgets)
            translation = audit.translation_audit(args.output, model, cell)
            if translation is not None and not translation["passed"]:
                actual_failures.append(
                    {
                        "metric": "translation_max_error",
                        "observed": translation["maximum_prediction_shift_error"],
                        "limit": audit.TRANSLATION_TOLERANCE,
                    }
                )
            if actual_failures != cell_decision["failures"]:
                raise ValueError("Saved rejection metrics differ from replay and the frozen budgets")
            paired = dataclasses.asdict(cell) | {"model": model, "control": audit.CONTROLS[model], "stage": cell.stage}
            paired["failed_metrics"] = ";".join(item["metric"] for item in actual_failures)
            for metric in METRICS:
                paired[f"candidate_{metric}"] = candidate[metric]
                paired[f"control_{metric}"] = control[metric]
                paired[f"delta_{metric}"] = candidate[metric] - control[metric]
            for parameter in ("floor", "gamma", "ridge"):
                paired[f"candidate_{parameter}"] = candidate[parameter]
                paired[f"control_{parameter}"] = control[parameter]
            pairs.append(paired)
            for fitted_model in (model, audit.CONTROLS[model]):
                receipts.append(benchmark.receipt(cell.folder(args.output, fitted_model) / "complete.json"))
            fit = json.loads((cell.folder(args.output, model) / "fit.json").read_text())
            description = dataclasses.asdict(cell) | {"model": model}
            profiles.append(profile_row(description, "outer", fit["floor_search"]["outer"]))
            inner = fit["floor_search"]["selected_configuration_inner_profiles"]
            fold_labels = sorted(set(fit["inner_folds"]))
            if len(inner) != len(fold_labels):
                raise ValueError("Selected-law profile traces omit an actual inner fold")
            for fold, record in zip(fold_labels, inner, strict=True):
                fitting = [
                    row for row, label in zip(fit["training_rows"], fit["inner_folds"], strict=True) if label != fold
                ]
                validation = [
                    row for row, label in zip(fit["training_rows"], fit["inner_folds"], strict=True) if label == fold
                ]
                if record["training_rows"] != fitting or record["validation_rows"] != validation:
                    raise ValueError("Saved selected-law profile rows differ from actual inner training/validation rows")
                profiles.append(profile_row(description, f"inner_{fold}", record["profile"]))
    pair_frame, profile_frame = pd.DataFrame(pairs), pd.DataFrame(profiles)
    pair_path = args.output / "completed_cell_pairs.csv"
    profile_path = args.output / "selected_profile_diagnostics.csv"
    pair_frame.to_csv(pair_path, index=False)
    profile_frame.to_csv(profile_path, index=False)
    profile_summary = {}
    for model, frame in profile_frame.groupby("model"):
        profile_summary[model] = {
            "profiles": len(frame),
            "outer_profiles": int(frame.partition.eq("outer").sum()),
            "inner_profiles": int(frame.partition.ne("outer").sum()),
            "profiles_selected_on_bound": int(frame.bound_hits.ne("").sum()),
            "selected_clamp_count": int(frame.selected_deficit_clamp_count.sum()),
            "trials_with_clamps": int(frame.trials_with_clamps.sum()),
            "refinement_count": int(frame.refinement_count.sum()),
            "unsuccessful_refinements": int(frame.unsuccessful_refinements.sum()),
            "constant_response_profiles": int(frame.constant_response.sum()),
        }
    record = {
        "scientific_identity": identity,
        "analysis_source": benchmark.receipt(Path(__file__)),
        "source_freeze": benchmark.receipt(audit.ROOT / "source_freeze.json"),
        "control_budgets": benchmark.receipt(audit.ROOT / "control_budgets.json"),
        "fixed_law_preflight": benchmark.receipt(audit.ROOT / "fixed_law_preflight.json"),
        "independent_replay": benchmark.receipt(replay_path),
        "decisions": decisions,
        "completed_candidate_cells": len(pairs),
        "maximum_serialized_prediction_replay_error": replay["maximum_serialized_prediction_replay_error"],
        "profile_summary": profile_summary,
        "profile_scope": "Selected outer and selected-configuration inner profiles only; not every CV configuration",
        "clamp_scope": "Selected-head clamp counts separate from clamped search trials, including the gamma1 endpoint",
        "metric_scope": {
            "matched_gain": "Prediction difference against the tied counterfactual at the identical aggregate",
            "raw_policy_regret": "Truth at the selected policy minus the minimum truth on the fixed 101x101 menu",
            "aggregate_only_selection": (
                "At kappa0 equal-aggregate schedules are indistinguishable; selected coordinates depend on numerical "
                "ties. A 2D-grid minimum can beat the 101 tied-grid minimum through finer aggregate resolution; "
                "that grid-minimum difference is not matched phase gain."
            ),
        },
        "inputs": receipts,
        "outputs": {str(path.resolve()): benchmark.sha256(path) for path in (pair_path, profile_path)},
        "new_fits": 0,
    }
    benchmark.write_json(args.output / "completed_audit.json", record)
    print(json.dumps({"decisions": decisions, "profile_summary": profile_summary}, indent=2))


if __name__ == "__main__":
    main()
