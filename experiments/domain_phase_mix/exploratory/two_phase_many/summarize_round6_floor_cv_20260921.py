# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Replay completed round-six cells and compare their existing paired controls.

This readout never fits a model or changes a scientific gate. Its snapshot uses
only the ordered prefix recorded in each candidate's saved decision. Run with
uv run --no-sync python; repeat safely as new candidate decisions arrive.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import audit_round6_floor_cv_20260921 as audit
import numpy as np
import pandas as pd

benchmark = audit.benchmark
PRIOR = {"NG2-016": "NG2-014", "NG2-017": "NG2-015"}
MAIN_METRICS = (
    "matched_gain_rmse",
    "uniform_101_tied_curve_rmse",
    "raw_policy_regret",
    "absolute_selected_optimism",
)


def prerequisites(prepared: Path) -> dict[str, Any]:
    current = audit.identity(prepared)
    budgets = audit.candidate_prerequisites(prepared)
    preflight_path = audit.ROOT / "fixed_law_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    generic = json.loads((audit.ROOT / "generic_structural_mapping.json").read_text())
    reuse = json.loads((audit.ROOT / "control_reuse.json").read_text())
    if preflight["fits"] != 8 or len(preflight["cases"]) != 8 or len(generic["generic"]) != 48:
        raise ValueError("Known-law or generic-mapping coverage changed")
    if not generic["passed"] or not all(row["passed"] for row in generic["generic"]):
        raise ValueError("Generic native tied mappings did not pass")
    if reuse["original_identity"] != budgets["identity"] or reuse["new_control_fits"] != 0:
        raise ValueError("The readout requires the original, unchanged round-five controls")
    return {
        "identity": current,
        "budgets": budgets,
        "preflight": {
            "passed": preflight["passed"],
            "known_law_cases": 8,
            "generic_mapping_cases": 48,
            "verified_output_files": len(preflight["outputs"]),
            "maximum_floor_error": max(row["absolute_floor_error"] for row in preflight["cases"]),
            "maximum_gain_rmse": max(row["raw_menu"]["matched_gain_rmse"] for row in preflight["cases"]),
            "maximum_uniform_tied_rmse": max(
                row["raw_menu"]["uniform_101_tied_curve_rmse"] for row in preflight["cases"]
            ),
            "maximum_fitted_mapping_error": max(
                value for row in preflight["cases"] for value in row["mapping"].values()
            ),
            "maximum_generic_mapping_error": max(
                value for row in generic["generic"] for value in row["checks"].values()
            ),
            "maximum_prediction_replay_error": max(
                row["maximum_serialized_prediction_replay_error"] for row in preflight["cases"]
            ),
        },
        "controls": {
            "contexts": reuse["control_context_count"],
            "new_fits": reuse["new_control_fits"],
            "maximum_prediction_replay_error": reuse["maximum_serialized_prediction_replay_error"],
        },
        "receipts": [
            benchmark.receipt(preflight_path),
            benchmark.receipt(audit.ROOT / "control_reuse.json"),
            benchmark.receipt(audit.BUDGET_PATH),
        ],
    }


def policy_metrics(folder: Path) -> dict[str, Any]:
    frame = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    truth, predicted = frame.truth.to_numpy(), frame.prediction.to_numpy()
    tied = frame.tied.to_numpy(bool)
    selected = int(np.argmin(predicted))
    true_best = int(np.argmin(truth))
    best_tied_truth = float(np.min(truth[tied]))
    best_counterfactual_tied_truth = float(np.min(frame.tied_truth))
    true_gain = frame.tied_truth.to_numpy() - truth
    predicted_gain = frame.tied_prediction.to_numpy() - predicted
    maximum_effect = int(np.argmax(np.abs(predicted_gain)))
    code = frame.phase0_code.to_numpy()
    pure = tied & ((code == 0) | (code == 1))
    # The represented 0.95 menu coordinate is one float step above 0.95.
    interior = tied & (code >= 0.05 - 1e-12) & (code <= 0.95 + 1e-12)
    squared_errors = (predicted - truth) ** 2
    tied_sse = float(np.sum(squared_errors[tied]))
    return {
        "selected_phase0_code": float(frame.phase0_code.iloc[selected]),
        "selected_phase1_code": float(frame.phase1_code.iloc[selected]),
        "true_best_phase0_code": float(frame.phase0_code.iloc[true_best]),
        "true_best_phase1_code": float(frame.phase1_code.iloc[true_best]),
        "true_best_tied_bpb": best_tied_truth,
        "true_best_counterfactual_tied_bpb": best_counterfactual_tied_truth,
        "selected_true_bpb": float(truth[selected]),
        "selected_predicted_bpb": float(predicted[selected]),
        "diagonal_minus_counterfactual_tied_grid_gap": best_tied_truth - best_counterfactual_tied_truth,
        "true_grid_advantage_over_counterfactual_tied_grid": best_counterfactual_tied_truth - float(truth[true_best]),
        "selected_true_benefit_over_counterfactual_tied_grid": best_counterfactual_tied_truth - float(truth[selected]),
        "predicted_grid_advantage_over_counterfactual_tied_grid": float(
            np.min(frame.tied_prediction) - predicted[selected]
        ),
        "true_grid_advantage_over_true_best_tied": best_tied_truth - float(truth[true_best]),
        "selected_true_benefit_over_true_best_tied": best_tied_truth - float(truth[selected]),
        "predicted_grid_advantage": float(np.min(predicted[tied]) - predicted[selected]),
        "selected_true_matched_gain": float(true_gain[selected]),
        "selected_predicted_matched_gain": float(predicted_gain[selected]),
        "maximum_true_matched_gain": float(np.max(true_gain)),
        "maximum_absolute_true_matched_gain": float(np.max(np.abs(true_gain))),
        "maximum_predicted_matched_gain": float(np.max(predicted_gain)),
        "minimum_predicted_matched_gain": float(np.min(predicted_gain)),
        "maximum_absolute_predicted_matched_gain": float(np.max(np.abs(predicted_gain))),
        "maximum_absolute_effect_phase0_code": float(frame.phase0_code.iloc[maximum_effect]),
        "maximum_absolute_effect_phase1_code": float(frame.phase1_code.iloc[maximum_effect]),
        "true_gain_at_maximum_absolute_effect": float(true_gain[maximum_effect]),
        "uniform_tied_sse": tied_sse,
        "pure_endpoint_tied_sse": float(np.sum(squared_errors[pure])),
        "pure_endpoint_tied_sse_fraction": float(np.sum(squared_errors[pure]) / tied_sse) if tied_sse > 0 else 0.0,
        "tied_rmse_within_training_aggregate_range": float(np.sqrt(np.mean(squared_errors[interior]))),
        "pure_code0_error": float((predicted - truth)[tied & (code == 0)][0]),
        "pure_code1_error": float((predicted - truth)[tied & (code == 1)][0]),
    }


def estimator_row(folder: Path, row: dict[str, Any], role: str, candidate: str, index: int) -> dict[str, Any]:
    fitted = json.loads((folder / "fit.json").read_text())
    diagnostics = {}
    if role == "candidate":
        search = fitted["floor_search"]["selected_configuration_search"]
        selected = next(trial for trial in search["trace"] if trial["floor"] == fitted["floor"])
        training = pd.read_csv(folder / "training_predictions.csv", float_precision="round_trip")
        diagnostics = {
            "selected_search_bound_hits": json.dumps(search["bound_hits"]),
            "selected_search_refinements": len(search["refinements"]),
            "selected_search_failed_refinements": sum(not trial["success"] for trial in search["refinements"]),
            "selected_search_conditional_solves": search["conditional_solves"],
            "selected_inner_deficit_clamps": sum(fold["deficit_clamp_count"] for fold in selected["folds"]),
            "selected_outer_deficit_clamps": int(np.sum(training.response.to_numpy() - fitted["floor"] < 1e-9)),
            "inner_rmse": fitted["inner_rmse"],
            "final_floor_unchanged": fitted["floor_search"]["final_floor_unchanged"],
        }
    return {
        "candidate": candidate,
        "context_index": index,
        "role": role,
        "estimator": row["model"],
        **{name: row[name] for name in ("world", "sampler", "seed", "noise_sd", "floor_shift")},
        **{metric: row[metric] for metric in MAIN_METRICS},
        "signed_selected_optimism": row["selected_optimism"],
        "training_endpoint_rmse": row["training_endpoint_rmse"],
        "floor": fitted["floor"],
        "shape": json.dumps(fitted["shape"], sort_keys=True),
        "q": fitted["q"],
        "kappa": fitted["kappa"],
        "ridge": fitted["ridge"],
        **policy_metrics(folder),
        **diagnostics,
    }


def snapshot_decision(model: str, source: Path, output: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = source / model / "decision.json"
    if not path.exists():
        return None, None
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    snapshot = output / "decision_snapshots" / f"{model}_{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != payload:
        raise ValueError("An immutable decision snapshot changed")
    if not snapshot.exists():
        snapshot.write_bytes(payload)
    return json.loads(payload), {
        "original_path": str(path.resolve()),
        "original_sha256": digest,
        "snapshot": benchmark.receipt(snapshot),
    }


def summarize(prepared: Path, source: Path, output: Path) -> dict[str, Any]:
    initial = prerequisites(prepared)
    budgets = initial.pop("budgets")
    output.mkdir(parents=True, exist_ok=True)
    long_rows, paired_rows, statuses, snapshots = [], [], {}, []
    receipts = initial.pop("receipts")
    for model in audit.CANDIDATES:
        decision, snapshot = snapshot_decision(model, source, output)
        if decision is None:
            statuses[model] = {"status": "not_started", "completed_contexts": 0, "unrun_contexts": 32}
            continue
        snapshots.append(snapshot)
        if decision["identity"] != initial["identity"]:
            raise ValueError("Candidate decision has a different frozen source identity")
        count = decision["completed_contexts"]
        cells = audit.old.contexts()[:count]
        if [item["context"] for item in decision["decisions"]] != [dataclasses.asdict(cell) for cell in cells]:
            raise ValueError("Candidate decision is not an ordered registered prefix")
        statuses[model] = {key: decision[key] for key in ("status", "completed_contexts", "unrun_contexts")}
        statuses[model]["phase_improvement"] = decision.get("phase_improvement")
        statuses[model]["terminal_failures"] = decision["decisions"][-1]["failures"] if count else []
        for index, cell in enumerate(cells, start=1):
            candidate = audit.replay_cell(prepared, source, cell, model)
            control = audit.old.replay_cell(prepared, audit.OLD_OUTPUT, cell, audit.CONTROLS[model])
            failure = audit.budget_failures(candidate, budgets)
            translation = audit.translation_audit(source, model, cell)
            if translation is not None and not translation["passed"]:
                failure.append(
                    {
                        "metric": "translation_max_error",
                        "observed": translation["maximum_prediction_shift_error"],
                        "limit": audit.TRANSLATION_LIMIT,
                    }
                )
            if failure != decision["decisions"][index - 1]["failures"]:
                raise ValueError("Recorded gate decision differs from replayed metrics")
            if failure and index != count:
                raise ValueError("A candidate continued after its first failed context")
            estimator_rows = []
            for role, row, directory in (
                ("candidate", candidate, source),
                ("same_loss_control", control, audit.OLD_OUTPUT),
            ):
                folder = cell.folder(directory, row["model"])
                estimator_rows.append(estimator_row(folder, row, role, model, index))
                receipts.append(benchmark.receipt(folder / "complete.json"))
            prior_folder = cell.folder(audit.OLD_OUTPUT, PRIOR[model])
            if (prior_folder / "complete.json").exists():
                prior = audit.old.replay_cell(prepared, audit.OLD_OUTPUT, cell, PRIOR[model])
                estimator_rows.append(estimator_row(prior_folder, prior, "prior_round5_candidate", model, index))
                receipts.append(benchmark.receipt(prior_folder / "complete.json"))
            cohort = next(
                item
                for item in budgets["cohorts"]
                if item["control"] == audit.CONTROLS[model]
                and item["world"] == cell.world
                and item["noise_sd"] == cell.noise_sd
            )
            paired = {
                "candidate": model,
                "context_index": index,
                **dataclasses.asdict(cell),
                "failures": json.dumps(failure, sort_keys=True),
                "translation_max_error": None if translation is None else translation["maximum_prediction_shift_error"],
                **{f"limit_{key}": value for key, value in cohort["limits"].items()},
            }
            for row in estimator_rows:
                for metric in (
                    *MAIN_METRICS,
                    "floor",
                    "selected_true_benefit_over_true_best_tied",
                    "selected_true_matched_gain",
                    "selected_predicted_matched_gain",
                    "selected_phase0_code",
                    "selected_phase1_code",
                    "true_best_tied_bpb",
                    "true_best_counterfactual_tied_bpb",
                    "selected_true_benefit_over_counterfactual_tied_grid",
                    "true_grid_advantage_over_counterfactual_tied_grid",
                    "diagonal_minus_counterfactual_tied_grid_gap",
                ):
                    paired[f"{row['role']}_{metric}"] = row[metric]
            paired["true_grid_advantage_over_true_best_tied"] = estimator_rows[0][
                "true_grid_advantage_over_true_best_tied"
            ]
            paired_rows.append(paired)
            long_rows.extend(estimator_rows)
    if audit.identity(prepared) != initial["identity"]:
        raise ValueError("Frozen source identity changed during the readout")
    if not benchmark.valid_receipts(receipts):
        raise ValueError("A completed input changed during the readout")
    pd.DataFrame(long_rows).to_csv(output / "completed_estimators.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "completed_pairs.csv", index=False)
    report = {
        **initial,
        "readout_source": benchmark.receipt(Path(__file__)),
        "candidate_status": statuses,
        "completed_candidate_contexts": len(paired_rows),
        "new_fits": 0,
        "decision_snapshots": snapshots,
        "input_receipts": list({item["path"]: item for item in receipts}.values()),
        "outputs": {
            name: benchmark.sha256(output / name) for name in ("completed_estimators.csv", "completed_pairs.csv")
        },
        "metric_scope": {
            "matched_gain": "Tied counterfactual at the selected policy's same aggregate minus that policy",
            "selected_true_benefit_over_true_best_tied": (
                "Minimum truth on the 101 tied grid points minus truth at model-selected raw policy"
            ),
            "true_grid_advantage_over_true_best_tied": (
                "Minimum truth on 101 tied grid points minus minimum truth on the 101x101 raw menu"
            ),
            "grid_caution": (
                "The raw grid has finer aggregate spacing than the 101-point tied diagonal. A diagonal-referenced "
                "advantage can be a discretization artifact even in a zero-phase-gain world. Gates remain unchanged."
            ),
            "true_best_counterfactual_tied_bpb": (
                "Minimum raw_menu.tied_truth across all 10201 rows: tied policies at every aggregate "
                "represented in the raw menu"
            ),
            "true_grid_advantage_over_counterfactual_tied_grid": (
                "Best counterfactual tied truth minus best raw-menu truth; both search the same aggregate grid"
            ),
            "diagonal_minus_counterfactual_tied_grid_gap": (
                "Best101diagonal tied truth minus best all-row counterfactual tied truth; isolates grid discretization"
            ),
            "aggregate_policy_caution": (
                "Kappa-zero models predict zero matched phase gain; asymmetric selections and their realized benefit "
                "can arise from equal-aggregate ties and numerical tie behavior, not learned phase ordering"
            ),
            "prior_round5_candidate": (
                "Only already-completed same-context fits are compared; missing contexts remain absent"
            ),
            "scope": (
                "Adaptive synthetic rejection evidence; snapshots may be partial and do not establish certification"
            ),
        },
    }
    benchmark.write_json(output / "completed_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=audit.old.previous_audit.PREPARED)
    parser.add_argument("--source", type=Path, default=audit.ROOT / "synthetic_v1")
    parser.add_argument("--output", type=Path, default=audit.ROOT / "readout")
    args = parser.parse_args()
    report = summarize(args.prepared, args.source, args.output)
    print(
        json.dumps(
            {
                "candidate_status": report["candidate_status"],
                "completed_contexts": report["completed_candidate_contexts"],
                "new_fits": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
