# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""TP041: fixed-state hazard/additive readout comparison on invented covariates.

No fitting, empirical inputs, cache adoption, or policy optimization. Execution
requires an exact registered contract and approval. Pure helpers are available
for independent toy tests before that scientific factorial is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_causal_acquisition_retention_20261015 as state

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015/readout_isolation"
PURE_SHA = "95980e5615c3c015ecc9720762bbfe23d20cfdaf41e1d3c4e42d21c217701e7c"
SCOPE = "tp041_invented_fixed_state_312_cases_624_heads"
CONTRACT: dict[str, Any] = {
    "candidate_id": "TP40-041",
    "scope": SCOPE,
    "pool_inventories": {"balanced": [1.0, 1.0], "mixed": [1.0, 100.0]},
    "clocks": ["equal_group", "global_pool"],
    "rate": 1.0,
    "power": 0.7,
    "phase_fractions": [0.8, 0.2],
    "main_token_totals": [0.2, 2.0, 20.0, 200.0],
    "main_aggregate_first_shares": [0.2, 0.5, 0.8],
    "timings": ["early_heavy", "tied", "late_heavy"],
    "strengths": [0.0, 3.0],
    "main_cases": 288,
    "zero_total_cases": 8,
    "absent_bucket_cases": 16,
    "state_cases": 312,
    "fixed_heads": {"first_lighter": [1.0, 3.0], "first_heavier": [3.0, 1.0]},
    "floor": 0.6,
    "intercept": 1.0,
    "harm_coefficients": [0.1, 0.1],
    "harm_threshold": 3.0,
    "head_rows": 624,
    "quadrature_relative_tolerance": 1e-11,
    "quadrature_max_panels": 512,
    "floating_reconstruction_tolerance": 1e-10,
    "scientific_acceptance_gate": None,
    "interpretation": "Readout attenuation only; no empirical gain or estimator qualification",
    "fits": False,
    "policy_optimization": False,
    "empirical_inputs": False,
    "cap_or_clipping": False,
    "failed_cells": "retain every numerical failure; no silent retry, order increase, or omission",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    stratum: str
    inventory: str
    clock: str
    token_total: float
    aggregate_first: float
    timing: str
    mu: float
    phase_first: tuple[float, float]


def phase_first(aggregate: float, timing: str) -> tuple[float, float]:
    """Compute exact feasible endpoints with rational inputs, without clipping."""
    a, alpha, beta = Fraction(str(aggregate)), Fraction(4, 5), Fraction(1, 5)
    if timing == "tied":
        return float(a), float(a)
    if timing == "early_heavy":
        early, late = (a / alpha, Fraction(0)) if a <= alpha else (Fraction(1), (a - alpha) / beta)
    elif timing == "late_heavy":
        early, late = (Fraction(0), a / beta) if a <= beta else ((a - beta) / alpha, Fraction(1))
    else:
        raise ValueError("Unknown phase direction")
    return float(early), float(late)


def cases() -> tuple[Case, ...]:
    result = []
    for inventory, clock, total, aggregate, timing, mu in itertools.product(
        CONTRACT["pool_inventories"],
        CONTRACT["clocks"],
        CONTRACT["main_token_totals"],
        CONTRACT["main_aggregate_first_shares"],
        CONTRACT["timings"],
        CONTRACT["strengths"],
    ):
        result.append(
            Case(
                f"case{len(result):03d}",
                "main",
                inventory,
                clock,
                total,
                aggregate,
                timing,
                mu,
                phase_first(aggregate, timing),
            )
        )
    for inventory, clock, mu in itertools.product(
        CONTRACT["pool_inventories"], CONTRACT["clocks"], CONTRACT["strengths"]
    ):
        result.append(
            Case(
                f"case{len(result):03d}",
                "zero_total",
                inventory,
                clock,
                0.0,
                0.35,
                "asymmetric_zero_dose",
                mu,
                (0.25, 0.75),
            )
        )
        for aggregate in (0.0, 1.0):
            result.append(
                Case(
                    f"case{len(result):03d}",
                    "absent_bucket",
                    inventory,
                    clock,
                    2.0,
                    aggregate,
                    "tied",
                    mu,
                    (aggregate, aggregate),
                )
            )
    return tuple(result)


def covariates(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pools = np.asarray(CONTRACT["pool_inventories"][case.inventory], float)
    weights = np.array([[value, 1.0 - value] for value in case.phase_first])
    fractions = np.asarray(CONTRACT["phase_fractions"], float)
    masses = np.full(2, 0.5) if case.clock == "equal_group" else pools / pools.sum()
    return weights, fractions, case.token_total / pools, masses


def signed_exp_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Subtract positive log-domain quantities while retaining tiny differences."""
    output = np.zeros_like(left)
    positive, negative = left > right, left < right
    output[positive] = np.exp(left[positive]) * -np.expm1(right[positive] - left[positive])
    output[negative] = -np.exp(right[negative]) * -np.expm1(left[negative] - right[negative])
    return output


def readouts(result: state.PolicyResult, law: state.Law) -> dict[str, np.ndarray]:
    hazard = state.native_hazard(result.path.epochs, law)
    native = -np.expm1(-hazard)
    deficit = result.tied.deficit(law)
    # m_path-m_tied = q_tied-q_path, with both log survivals evaluated stably.
    contrast = signed_exp_difference(-result.tied.survival_hazard(law), -result.path.survival_hazard(law))
    attenuation = np.exp(-deficit)
    departure = attenuation * contrast
    return {
        "epochs": result.path.epochs,
        "native_hazard": hazard,
        "native_benefit": native,
        "path_mastery": np.exp(result.path.log_mastery),
        "tied_mastery": np.exp(result.tied.log_mastery),
        "tied_deficit": deficit,
        "log_attenuation": -deficit,
        "attenuation": attenuation,
        "state_contrast": contrast,
        "hazard_departure": departure,
        "additive_departure": contrast,
        "hazard_benefit": result.benefit,
        "hazard_benefit_reconstructed": native + departure,
        "additive_benefit": native + contrast,
    }


def head_values(features: dict[str, np.ndarray], amplitudes: np.ndarray) -> dict[str, float]:
    harm = np.logaddexp(np.log1p(features["epochs"]) - CONTRACT["harm_threshold"], 0.0) ** 2
    native_log_deficit = float(
        CONTRACT["intercept"]
        - amplitudes @ features["native_benefit"]
        + np.asarray(CONTRACT["harm_coefficients"]) @ harm
    )
    native_deficit = math.exp(native_log_deficit)
    output = {"native_log_deficit": native_log_deficit, "native_tied_loss": CONTRACT["floor"] + native_deficit}
    for name in ("hazard", "additive"):
        improvement = float(amplitudes @ features[f"{name}_departure"])
        predicted = CONTRACT["floor"] + math.exp(native_log_deficit - improvement)
        # Preserve gains below the precision of subtracting two complete losses.
        gain = native_deficit * -math.expm1(-improvement)
        output |= {
            f"{name}_log_deficit_improvement": improvement,
            f"{name}_loss": predicted,
            f"{name}_gain": gain,
            f"{name}_direct_gain": output["native_tied_loss"] - predicted,
        }
    output["log_deficit_improvement_difference"] = (
        output["additive_log_deficit_improvement"] - output["hazard_log_deficit_improvement"]
    )
    return output


def evaluate_case(case: Case, control: state.Quadrature) -> dict[str, Any]:
    weights, fractions, scales, masses = covariates(case)
    law = state.Law(CONTRACT["rate"], CONTRACT["power"], case.mu, ("first", "second"), masses)
    result = state.evaluate_policy(weights, fractions, scales, law, control)
    features = readouts(result, law)
    pools = np.asarray(CONTRACT["pool_inventories"][case.inventory], float)
    epoch_path = state.policy_epochs(weights, fractions, scales)
    tolerance = CONTRACT["floating_reconstruction_tolerance"]
    parity = float(np.max(abs(features["hazard_benefit"] - features["hazard_benefit_reconstructed"])))
    violations = [
        float(np.max(-features["path_mastery"])),
        float(np.max(features["path_mastery"] - features["native_benefit"])),
        float(np.max(-features["tied_mastery"])),
        float(np.max(features["tied_mastery"] - features["native_benefit"])),
        float(np.max(-features["additive_benefit"])),
        float(np.max(features["additive_benefit"] - 2 * features["native_benefit"])),
        float(np.max(features["attenuation"] - 1)),
        float(np.max(np.exp(-features["native_hazard"]) - features["attenuation"])),
    ]
    errors = {
        "hazard_reconstruction_error": parity,
        "maximum_bound_violation": max(0.0, *violations),
        "phase_budget_max_abs_error": float(np.max(abs(epoch_path @ pools - fractions * case.token_total))),
        "aggregate_first_abs_error": abs(float(fractions @ weights[:, 0]) - case.aggregate_first),
        "weight_box_max_violation": max(0.0, float(np.max(-weights)), float(np.max(weights - 1))),
        "weight_simplex_max_abs_error": float(np.max(abs(weights.sum(axis=1) - 1))),
    }
    native_departure = None
    if result.exact_native_branch:
        native_departure = max(
            float(np.max(abs(features[name]))) for name in ("state_contrast", "hazard_departure", "additive_departure")
        )
        errors["native_branch_departure_max_abs_error"] = native_departure
    return {
        "case": asdict(case),
        "status": "complete",
        "weights": weights.tolist(),
        "masses": masses.tolist(),
        "phase_epochs": epoch_path.tolist(),
        "phase_token_totals": (epoch_path @ pools).tolist(),
        "expected_phase_token_totals": (fractions * case.token_total).tolist(),
        "aggregate_first_reconstructed": float(fractions @ weights[:, 0]),
        "features": {name: values.tolist() for name, values in features.items()},
        "head_values": {
            name: head_values(features, np.asarray(amplitudes)) for name, amplitudes in CONTRACT["fixed_heads"].items()
        },
        "additive_above_one": (features["additive_benefit"] > 1).tolist(),
        "negative_departure": (features["state_contrast"] < 0).tolist(),
        "exact_native_branch": result.exact_native_branch,
        **errors,
        "native_branch_departure_max_abs_error": native_departure,
        "numerical_checks_passed": all(error <= tolerance for error in errors.values()),
        "path_evaluations": result.path.evaluations,
        "tied_evaluations": result.tied.evaluations,
        "shared_path_and_tied_state": result.path is result.tied,
        "maximum_relative_error_estimate": max(
            result.path.maximum_relative_error_estimate, result.tied.maximum_relative_error_estimate
        ),
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Preserve differing artifact {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError(f"Preserve interrupted artifact {temporary}")
    temporary.write_text(text)
    temporary.replace(path)


def runtime() -> dict[str, str]:
    return {"python": sys.version} | {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        BASE / "test_tpp40_readout_isolation_20261015.py",
        Path(state.__file__).resolve(),
        BASE / "test_tpp40_causal_acquisition_retention_20261015.py",
        OUTPUT / "proposal_v1.md",
        OUTPUT.parent / "reviews/additive_readout_attenuation_assessment_v1.md",
        Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py"),
    )


def payload_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity() -> dict[str, Any]:
    registration = json.loads((OUTPUT / "registration.json").read_text())
    if registration["status"] != "registered" or registration["contract"] != CONTRACT:
        raise ValueError("Require the exact registered invented-covariate scope")
    if digest(Path(state.__file__)) != PURE_SHA or registration["runtime"] != runtime():
        raise ValueError("The frozen pure source/runtime changed")
    if set(registration["source_hashes"]) != {str(path) for path in source_paths()}:
        raise ValueError("Source provenance must match the exact reviewed whitelist")
    for name, expected in registration["source_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Source changed: {name}")
    return {
        "registration_sha256": digest(OUTPUT / "registration.json"),
        "source_hashes": registration["source_hashes"],
        "runtime": runtime(),
    }


def run() -> None:
    current = identity()
    if current != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen execution identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    for key, expected in {
        "scope": SCOPE,
        "registration_sha256": current["registration_sha256"],
        "execution_identity_sha256": digest(OUTPUT / "execution_identity.json"),
    }.items():
        if approval.get(key) != expected:
            raise ValueError("Require exact scientific execution approval")
    control = state.Quadrature(CONTRACT["quadrature_relative_tolerance"], CONTRACT["quadrature_max_panels"])
    expected_cases = {f"{case.case_id}.json" for case in cases()}
    if {path.name for path in (OUTPUT / "cells").glob("*.json")} - expected_cases:
        raise ValueError("Preserve unexpected case artifacts outside the registered scope")
    results: list[dict[str, Any]] = []
    for case in cases():
        if identity() != current:
            raise ValueError("Execution sources changed")
        target = OUTPUT / "cells" / f"{case.case_id}.json"
        if target.exists():
            saved = json.loads(target.read_text())
            if saved["identity"] != current or saved["result"]["case"] != json.loads(json.dumps(asdict(case))):
                raise ValueError("Existing cell scope differs")
            if saved["result_sha256"] != payload_digest(saved["result"]):
                raise ValueError("Existing cell payload changed")
            record = saved["result"]
        else:
            try:
                record = evaluate_case(case, control)
            except ArithmeticError as error:
                record = {
                    "case": asdict(case),
                    "status": "numerical_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            write(target, {"identity": current, "result": record, "result_sha256": payload_digest(record)})
        results.append(record)
    write(
        OUTPUT / "summary.json",
        {
            "scope": SCOPE,
            "cases": len(results),
            "head_rows_expected": CONTRACT["head_rows"],
            "completed_cases": sum(item["status"] == "complete" for item in results),
            "numerical_failure_cases": [item["case"]["case_id"] for item in results if item["status"] != "complete"],
            "failed_checks": [
                item["case"]["case_id"]
                for item in results
                if item["status"] == "complete" and not item["numerical_checks_passed"]
            ],
            "scientific_qualification": False,
            "all_cells_retained": True,
        },
    )
    write(
        OUTPUT / "complete.json",
        {
            "identity": current,
            "files": (
                {str(path.relative_to(OUTPUT)): digest(path) for path in sorted((OUTPUT / "cells").glob("*.json"))}
                | {"summary.json": digest(OUTPUT / "summary.json")}
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        write(OUTPUT / "execution_identity.json", identity())
    else:
        run()


if __name__ == "__main__":
    main()
