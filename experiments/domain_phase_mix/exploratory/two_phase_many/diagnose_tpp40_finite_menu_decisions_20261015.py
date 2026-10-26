# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Read frozen TP038 means and diagnose three explicitly finite decision menus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_finite_menu_risk_20261015 as risk

BASE = Path(__file__).resolve().parent
PARENT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015"
INPUT = PARENT / "gaussian_pair_diagnostic"
OUTPUT = PARENT / "finite_menu_decisions"
PREREQUISITE_SHA = "d9dba546a0e2a0d1a3b9c6beb390a6a90306ff0b65cc672efb92f9cafb1b10a3"
MENUS = {
    "held_asymmetric": ("asymmetric",),
    "held_joint": ("asymmetric", "actual_tied"),
    "held_asymmetric_with_counterfactual_ties": ("asymmetric", "matched_tied"),
}
HEAD_FIELDS = (
    "pair_id",
    "source",
    "world",
    "noise",
    "outer_fold",
    "candidate",
    "distance",
    "bayes_error",
    "analytic_pair_eligible",
    "selected_fit_unresolved",
    "original_numerical_status",
    "within_class_exclusion",
)
CONTRACT = {
    "candidate_id": "TP40-042",
    "scope": "all_450_TP038_pairs_on_three_saved_finite_menus",
    "menus": {key: list(value) for key, value in MENUS.items()},
    "head_count": 450,
    "decision_problems": 1350,
    "gaussian_noise_sd": 0.003,
    "prior": [0.5, 0.5],
    "regret": "function_value_at_action_minus_that_functions_minimum_over_the_same_finite_menu",
    "overlap_bound": "Phi(-d/2) * min_a(regret_0(a)+regret_1(a))",
    "exact_bayes_risk": "integrate_the_exact_rational_lower_envelope_of_r0+t*r1_against_Gaussian_log_likelihood_ratio",
    "numeric_scope": "exact_supplied_float_means_and_rational_breakpoints_floating_point_normal_probabilities",
    "eligibility": "preserve_TP038_analytic_pair_eligibility_for_within_class_bounds",
    "misspecified_pairs": "retain_their_finite_pair_decision_problem_without_within_class_bound",
    "adaptivity": "fixed_pairs_describe_fresh_hypothetical_experiments_not_conditional_posteriors",
    "dependence": "duplicates_and_overlapping_experiments_are_not_independent_confirmations",
    "scientific_gate": None,
    "prior_failures_unchanged": True,
    "continuous_policy_claim": False,
    "new_fits": False,
    "new_policy_queries": False,
    "empirical_outcomes": False,
}


def registration_contents() -> dict[str, Any]:
    sources = [
        Path(__file__),
        Path(risk.__file__),
        BASE / "test_tpp40_finite_menu_risk_20261015.py",
        OUTPUT / "proposal_v1.md",
    ]
    inputs = [INPUT / name for name in ("result_manifest_v1.json", "heads.csv", "endpoints.csv")]
    return {
        "contract": CONTRACT,
        "runtime": (
            {"python": sys.version} | {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")}
        ),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "input_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }


def verify_inputs() -> dict[str, Any]:
    registration = json.loads((OUTPUT / "registration.json").read_text())
    expected = registration_contents()
    if registration.get("status") != "registered" or any(
        registration.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("The registered source, runtime or scope changed")
    seal = INPUT / "result_manifest_v1.json"
    if hashlib.sha256(seal.read_bytes()).hexdigest() != PREREQUISITE_SHA:
        raise ValueError("The prerequisite seal changed")
    for name, record in json.loads(seal.read_text())["files"].items():
        if hashlib.sha256((INPUT / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Prerequisite changed: {name}")
    return {"registration_sha256": hashlib.sha256((OUTPUT / "registration.json").read_bytes()).hexdigest(), **expected}


def head_metadata(head: dict[str, Any]) -> dict[str, Any]:
    return {key: head[key] for key in HEAD_FIELDS} | {"prerequisite_heads_path": str(INPUT / "heads.csv")}


def menu_rows(head: dict[str, Any], endpoints: pd.DataFrame, menu: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = (
        endpoints.loc[endpoints["role"].isin(MENUS[menu])]
        .sort_values(["role", "source_index", "row_id"])
        .reset_index(drop=True)
    )
    labels = [f"{row.role}:{row.source_index}:{row.row_id}" for row in values.itertuples()]
    if len(labels) != len(set(labels)):
        raise ValueError("Repeated role/source action identifiers")
    truth = values["truth"].to_numpy(float)
    alternative = values["predicted"].to_numpy(float)
    result = risk.finite_menu_risk(truth, alternative, float(head["distance"]))
    eligible = bool(head["analytic_pair_eligible"])
    row = head_metadata(head)
    row.update(
        {
            "menu": menu,
            "status": "finite",
            "action_count": len(values),
            "envelope_intervals": len(result.intervals),
            "distinct_action_value_pairs": len(set(zip(truth, alternative, strict=True))),
            "truth_best_mean": float(np.min(truth)),
            "alternative_best_mean": float(np.min(alternative)),
            "optimal_truth_actions": json.dumps([labels[i] for i in result.optimal_actions_zero]),
            "optimal_alternative_actions": json.dumps([labels[i] for i in result.optimal_actions_one]),
            "shared_optimum": result.shared_optimum,
            "compromise_sum_regret": result.compromise_sum_regret,
            "finite_pair_bayes_regret": result.bayes_regret,
            "log_finite_pair_bayes_regret": result.log_bayes_regret,
            "overlap_bound": result.overlap_bound,
            "log_overlap_bound": result.log_overlap_bound,
            "within_class_menu_regret_bound": result.bayes_regret if eligible else None,
            "log_within_class_menu_regret_bound": result.log_bayes_regret if eligible else None,
            "truth_regret_of_first_alternative_best": float(truth[result.optimal_actions_one[0]] - np.min(truth)),
            "alternative_regret_of_first_truth_best": float(
                alternative[result.optimal_actions_zero[0]] - np.min(alternative)
            ),
            "probability_representation": (
                "point_mass_at_likelihood_ratio_one" if head["distance"] == 0 else "continuous_gaussian_log_ratio"
            ),
        }
    )
    intervals = []
    for position, interval in enumerate(result.intervals):
        probabilities = (
            result.interval_log_probabilities[position] if result.interval_log_probabilities else (None, None)
        )
        intervals.append(
            {
                "pair_id": head["pair_id"],
                "menu": menu,
                "interval": position,
                "action": labels[interval.action],
                "truth_regret": float(truth[interval.action] - np.min(truth)),
                "alternative_regret": float(alternative[interval.action] - np.min(alternative)),
                "lower_ratio_numerator": str(interval.lower.numerator),
                "lower_ratio_denominator": str(interval.lower.denominator),
                "upper_ratio_numerator": None if interval.upper is None else str(interval.upper.numerator),
                "upper_ratio_denominator": None if interval.upper is None else str(interval.upper.denominator),
                "log_probability_zero": probabilities[0],
                "log_probability_one": probabilities[1],
            }
        )
    return row, intervals


def diagnose_menu(
    head: dict[str, Any], endpoints: pd.DataFrame, menu: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retain a numerical failure and its original pair status without filtering it."""
    try:
        return menu_rows(head, endpoints, menu)
    except ArithmeticError as error:
        return head_metadata(head) | {"menu": menu, "status": "numerical_failure", "message": str(error)}, []


def run() -> None:
    identity = verify_inputs()
    receipt = OUTPUT / "complete.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            raise ValueError("Existing completion has different inputs")
        for name, expected in saved["outputs"].items():
            if hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Completed output changed: {name}")
        print(json.dumps({"status": "reused_complete", "problems": saved["problems"]}))
        return
    heads = pd.read_csv(INPUT / "heads.csv", float_precision="round_trip", keep_default_na=False)
    endpoints = pd.read_csv(INPUT / "endpoints.csv", float_precision="round_trip")
    if len(heads) != 450 or heads["pair_id"].duplicated().any() or set(endpoints["pair_id"]) != set(heads["pair_id"]):
        raise ValueError("The frozen pair inventory changed")
    grouped = {key: frame for key, frame in endpoints.groupby("pair_id", sort=False)}
    rows, intervals = [], []
    for head in heads.to_dict("records"):
        subset = grouped[head["pair_id"]]
        if set(subset["role"]) != {"asymmetric", "matched_tied", "actual_tied"}:
            raise ValueError("Every pair must retain all three query roles")
        for menu in MENUS:
            row, cells = diagnose_menu(head, subset, menu)
            rows.append(row)
            intervals.extend(cells)
    pd.DataFrame(rows).to_csv(OUTPUT / "problems.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUTPUT / "intervals.csv", index=False)
    finite = [row for row in rows if row["status"] == "finite"]
    summary = {
        "problems": len(rows),
        "finite": len(finite),
        "failed": len(rows) - len(finite),
        "heads": len(heads),
        "envelope_intervals": len(intervals),
        "scientific_gate": None,
        "promotion": False,
        "scope": "Only the named finite menus; no larger-menu or continuous optimum bound.",
        "bounds": (
            "Fixed-pair equal-prior Bayes risk lower bounds its finite-action minimax risk, "
            "not typical or conditional error."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    outputs = {
        name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        for name in ("problems.csv", "intervals.csv", "summary.json")
    }
    receipt.write_text(
        json.dumps({"identity": identity, "outputs": outputs, **summary}, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["draft", "run"])
    args = parser.parse_args()
    if args.mode == "draft":
        print(json.dumps({"status": "draft", **registration_contents()}, indent=2, sort_keys=True, allow_nan=False))
    else:
        run()


if __name__ == "__main__":
    main()
