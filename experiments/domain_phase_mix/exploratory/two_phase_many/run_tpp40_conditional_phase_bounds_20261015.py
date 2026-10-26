# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Evaluate exactly twenty sealed synthetic endpoints; no fitting or policy search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_late_refresh_conditional_diagnostic_20261015 as diagnostic,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_conditional_phase_bounds_20261015"
PRIOR = ROOT / "reference_outputs/tpp40_late_refresh_20261015/stage2"
SEAL_SHA256 = "1a1b7fa05d98649554caa8124a4f602b312af7ff1d7d4714baf59c7172b2bf7f"
ARRAYS = ROOT / "reference_outputs/tpp40_temporal_round_20261014/data/arrays.npz"
ARRAYS_SHA256 = "1e153e80bb5cc2a49f9ec7bc86df08401767c0fe5dae50ca478237cd04ef3919"
INTERIOR_FRACTION = 1e-6
FEASIBILITY_TOLERANCE = 1e-10
PREDICTION_PARITY_TOLERANCE = 1e-10


def cases() -> list[dict[str, str]]:
    """Return the frozen twenty evaluator/policy pairs, in deterministic order."""
    result = []
    for world in ("null", "positive"):
        truth = f"contexts/{world}/truth.json"
        for field in ("best", "best_static"):
            result.append(
                {
                    "case_id": f"{world}__truth__{field}__truth",
                    "world": world,
                    "policy_artifact": f"truth_optima/{world}/optimization.json",
                    "policy_field": field,
                    "source_head_artifact": truth,
                    "source_head_kind": "truth",
                    "evaluator_artifact": truth,
                    "evaluator_kind": "truth",
                }
            )
        for family in ("aggregate_mariner", "late_refresh_hazard"):
            model = f"contexts/{world}/cells/{family}/model.json"
            for field in ("best", "best_static"):
                for evaluator in ("own", "truth"):
                    result.append(
                        {
                            "case_id": f"{world}__{family}__{field}__{evaluator}",
                            "world": world,
                            "family": family,
                            "policy_artifact": f"contexts/{world}/optimization/{family}/optimization.json",
                            "policy_field": field,
                            "source_head_artifact": model,
                            "source_head_kind": "fitted",
                            "evaluator_artifact": model if evaluator == "own" else truth,
                            "evaluator_kind": "fitted" if evaluator == "own" else "truth",
                            "truth_scores_artifact": f"contexts/{world}/optimization/{family}/truth_scores.json",
                        }
                    )
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write a new deterministic artifact atomically; never replace different evidence."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Preserve existing evidence before revising: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def environment() -> dict[str, str]:
    return {
        "python": sys.version,
        **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "threadpoolctl")},
    }


def source_paths() -> set[Path]:
    return {
        Path(__file__).resolve(),
        Path(diagnostic.__file__).resolve(),
        Path(law.__file__).resolve(),
        Path(law.temporal.__file__).resolve(),
        Path(law.mariner.__file__).resolve(),
        ROOT / "test_tpp40_conditional_phase_bounds_20261015.py",
        ROOT / "test_tpp40_late_refresh_conditional_diagnostic_20261015.py",
        ROOT / "run_tpp40_late_refresh_stage2_20261015.py",
        ROOT / "tpp40_tangent_optimization_20261015.py",
        ROOT / "tpp40_late_refresh_joint_models_20261015.py",
        ROOT / "reference_outputs/tpp40_late_refresh_20261015/reviews/mathematical_diagnostics_independent_v1.md",
        ROOT / "reference_outputs/tpp40_temporal_round_20261014/data/manifest.json",
        ARRAYS,
        PRIOR / "result_manifest.json",
    }


def verify_seal(root: Path, expected: str) -> dict[str, dict[str, Any]]:
    """Verify every sealed file, including artifacts unused by this diagnostic."""
    manifest = root / "result_manifest.json"
    if sha256(manifest) != expected:
        raise ValueError("Prior result manifest differs from the required seal")
    entries = json.loads(manifest.read_text())["files"]
    for relative, record in entries.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Sealed path escapes its result directory")
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Sealed prior artifact changed: {path}")
    return entries


def execution_contract() -> dict[str, Any]:
    return {
        "cases": cases(),
        "count": 20,
        "interior_fraction": INTERIOR_FRACTION,
        "feasibility_tolerance_relative_total_budget": FEASIBILITY_TOLERANCE,
        "prediction_parity_tolerance_absolute_bpb": PREDICTION_PARITY_TOLERANCE,
        "geometry_keys": ["c0", "c1", "alpha"],
        "total_budget": 1.0,
        "prior_seal_sha256": SEAL_SHA256,
        "arrays_sha256": ARRAYS_SHA256,
        "truth_objective": "unclipped exponential",
        "fitted_objective": "native clipped exponential",
        "new_fit_or_nonlinear_optimization": False,
    }


def registration() -> dict[str, Any]:
    config = json.loads((OUTPUT / "registration.json").read_text())
    if config["candidate_id"] != "TP40-031" or config["status"] != "registered":
        raise ValueError("A separately registered TP40-031 protocol is required")
    if config["execution_contract"] != execution_contract() or config["environment"] != environment():
        raise ValueError("Registered cases, numerical settings, or runtime differ")
    pins = config["source_hashes"]
    if set(pins) != {str(path) for path in source_paths()}:
        raise ValueError("Registration must pin the exact executable and input set")
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Registered source or input changed: {path}")
    if sha256(ARRAYS) != ARRAYS_SHA256:
        raise ValueError("Frozen covariate archive changed")
    return config


def current_identity() -> dict[str, Any]:
    registration()
    return {
        "registration_sha256": sha256(OUTPUT / "registration.json"),
        "source_hashes": {str(path): sha256(path) for path in sorted(source_paths())},
        "environment": environment(),
    }


def freeze() -> None:
    """Verify source, covariate bytes and the entire prior seal without loading heads."""
    registration()
    entries = verify_seal(PRIOR, SEAL_SHA256)
    previous = json.loads((PRIOR / "registration.json").read_text())["frozen_reference_inputs"]
    adopted = (
        Path(law.__file__).resolve(),
        Path(law.temporal.__file__).resolve(),
        Path(law.mariner.__file__).resolve(),
        ROOT / "run_tpp40_late_refresh_stage2_20261015.py",
        ROOT / "tpp40_tangent_optimization_20261015.py",
        ROOT / "tpp40_late_refresh_joint_models_20261015.py",
    )
    for path in adopted:
        if sha256(path) != previous[str(path)]:
            raise ValueError(f"Adopted prediction or optimizer source differs from prior execution: {path}")
    for case in cases():
        for key in ("policy_artifact", "source_head_artifact", "evaluator_artifact", "truth_scores_artifact"):
            if key in case and case[key] not in entries:
                raise ValueError("A named diagnostic input is not part of the prior seal")
    write_json(OUTPUT / "execution_identity.json", current_identity())


def require_execution() -> dict[str, Any]:
    identity = current_identity()
    if identity != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen diagnostic identity differs")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "tp031_twenty_saved_synthetic_endpoints",
        "registration_sha256": sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Separate exact execution approval is required")
    return identity


@dataclass(frozen=True)
class Geometry:
    alpha: float
    scales: np.ndarray
    costs: np.ndarray


def geometry(path: Path) -> Geometry:
    """Load only the three named covariates, without indexing any outcome arrays."""
    with np.load(path, allow_pickle=False) as arrays:
        c0, c1 = arrays["c0"], arrays["c1"]
        alpha = float(arrays["alpha"])
    return Geometry(alpha, c0 + c1, diagnostic.normalized_exposure_cost(c0, c1, alpha))


def task_head(payload: dict[str, Any], kind: str, dimensions: Geometry) -> diagnostic.TaskHead:
    raw, shape = payload["head"], payload["shape"]
    coefficients = np.asarray(raw["coefficients"], dtype=float)
    width = len(dimensions.scales)
    if coefficients.shape != (2 * width,):
        raise ValueError("Saved head width differs from frozen bucket inventory")
    if kind == "fitted" and float(payload["alpha"]) != dimensions.alpha:
        raise ValueError("Fitted head phase fraction differs from frozen geometry")
    strength = 0.0 if kind == "fitted" and payload["model"] == "aggregate_mariner" else payload["parameters"]["lambda"]
    return diagnostic.TaskHead(
        rate=float(shape["rate"]),
        power=float(shape["power"]),
        threshold=float(shape["threshold"]),
        strength=float(strength),
        intercept=float(raw["intercept"]),
        benefit=coefficients[:width],
        harm=coefficients[width:],
        floor=float(raw["floor"]),
    )


def frozen_prediction(head: diagnostic.TaskHead, kind: str, early: np.ndarray, late: np.ndarray, alpha: float) -> float:
    """Replay the frozen law with each evaluator's original exponential convention."""
    matrix, _ = law.design_and_offset(
        early[None],
        late[None],
        alpha,
        {"rate": head.rate, "power": head.power, "threshold": head.threshold},
        "late_refresh_hazard",
        {"lambda": head.strength},
    )
    native = law.mariner.Head(head.intercept, np.r_[head.benefit, head.harm], head.floor)
    if kind == "truth":
        return float(head.floor + np.exp(head.intercept + matrix @ native.coefficients)[0])
    return float(native.predict(matrix)[0])


def evaluate_case(
    case: dict[str, str],
    dimensions: Geometry,
    policy_document: dict[str, Any],
    source_payload: dict[str, Any],
    evaluator_payload: dict[str, Any],
    expected_evaluator_prediction: float,
) -> dict[str, Any]:
    """Evaluate one supplied saved case; this function performs no artifact reads."""
    policy = policy_document[case["policy_field"]]
    phase0, phase1 = np.asarray(policy["phase0"], dtype=float), np.asarray(policy["phase1"], dtype=float)
    if phase0.shape != dimensions.scales.shape or phase1.shape != phase0.shape:
        raise ValueError("Saved policy width differs from frozen geometry")
    early = dimensions.alpha * dimensions.scales * phase0
    late = (1 - dimensions.alpha) * dimensions.scales * phase1
    total = early + late
    source_head = task_head(source_payload, case["source_head_kind"], dimensions)
    evaluator = task_head(evaluator_payload, case["evaluator_kind"], dimensions)
    bounds = diagnostic.conditional_diagnostic(
        (evaluator,),
        np.ones(1),
        total,
        late,
        dimensions.costs,
        1 - dimensions.alpha,
        1.0,
        interior_fraction=INTERIOR_FRACTION,
        feasibility_tolerance=FEASIBILITY_TOLERANCE,
    )
    result: dict[str, Any] = {
        "case": case,
        "saved_policy": policy,
        "saved_phase_blind_search": policy_document["phase_blind_search"],
        "saved_predicted_gain_vs_static": policy_document["predicted_gain_vs_static"],
        "saved_predicted_gain_same_aggregate": policy_document["predicted_gain_same_aggregate"],
        "phase0_sum_residual": float(phase0.sum() - 1),
        "phase1_sum_residual": float(phase1.sum() - 1),
        "minimum_phase_weight": float(min(phase0.min(), phase1.min())),
        "evaluator_convention": (
            "unclipped exponential" if case["evaluator_kind"] == "truth" else "native clipped exponential"
        ),
        "expected_evaluator_prediction": expected_evaluator_prediction,
        "diagnostic": bounds,
        "matching_objective_bound_available": False,
        "bound_parity_and_feasibility_checks_pass": False,
        "scope": "Fixed aggregate only; floating-point estimate, not a rigorous or outer-global certificate",
    }
    if np.any(early < 0) or np.any(late < 0):
        result.update(
            frozen_prediction_parity_pass=False,
            frozen_prediction_unavailable_reason="Negative physical exposure; frozen law is undefined, no policy repair",
        )
        return result
    source_value = frozen_prediction(source_head, case["source_head_kind"], early, late, dimensions.alpha)
    evaluator_value = frozen_prediction(evaluator, case["evaluator_kind"], early, late, dimensions.alpha)
    source_error = source_value - float(policy["predicted"])
    evaluator_error = evaluator_value - expected_evaluator_prediction
    result.update(
        source_prediction_replay=source_value,
        source_prediction_error=source_error,
        evaluator_prediction_replay=evaluator_value,
        evaluator_prediction_error=evaluator_error,
        frozen_prediction_parity_pass=max(abs(source_error), abs(evaluator_error)) <= PREDICTION_PARITY_TOLERANCE,
    )
    if not bounds["bounds_available"]:
        return result
    point = bounds["task_results"][0]
    smooth_logit = point["original"]["logit"]
    smooth_native = point["original"]["native_loss"]
    with np.errstate(over="ignore", under="ignore"):
        smooth_unclipped = float(evaluator.floor + np.exp(smooth_logit))
    result["smooth_native_prediction"] = smooth_native
    result["smooth_unclipped_prediction"] = smooth_unclipped if np.isfinite(smooth_unclipped) else None
    smooth_matching = smooth_unclipped if case["evaluator_kind"] == "truth" else smooth_native
    result["smooth_minus_frozen_evaluator_prediction"] = (
        smooth_matching - evaluator_value if np.isfinite(smooth_matching) else None
    )
    if case["evaluator_kind"] == "truth":
        with np.errstate(over="ignore", under="ignore"):
            lower = float(evaluator.floor + np.exp(point["lower_logit"]))
        weighted = bounds["weighted"]
        if weighted["unclipped_support_available"]:
            lower = max(lower, weighted["unclipped_lower_bound"])
    else:
        lower = bounds["weighted"]["native_combined_lower_bound"]
    if np.isfinite(lower) and np.isfinite(smooth_matching):
        result.update(
            matching_objective_bound_available=True,
            bound_parity_and_feasibility_checks_pass=(
                result["frozen_prediction_parity_pass"]
                and abs(smooth_matching - evaluator_value) <= PREDICTION_PARITY_TOLERANCE
            ),
            matching_objective_lower_bound=lower,
            smooth_matching_remaining_improvement_upper_estimate=smooth_matching - lower,
            frozen_matching_remaining_improvement_upper_estimate=evaluator_value - lower,
        )
    else:
        result["matching_bound_unavailable_reason"] = "Matching-objective exponential is outside floating-point range"
    return result


def cached_case(folder: Path, spec: dict[str, str], identity: dict[str, Any]) -> bool:
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if (
        saved["case"] != spec
        or saved["identity"] != identity
        or saved["result_sha256"] != sha256(folder / "result.json")
    ):
        raise ValueError(f"Existing diagnostic receipt or result differs: {folder}")
    return True


def checked_input(relative: str, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = PRIOR / relative
    if sha256(path) != entries[relative]["sha256"]:
        raise ValueError(f"Sealed input changed before reading: {path}")
    return json.loads(path.read_text())


def run() -> None:
    identity = require_execution()
    entries = verify_seal(PRIOR, SEAL_SHA256)
    dimensions = geometry(ARRAYS)
    records, receipts = [], {}
    for spec in cases():
        require_execution()
        folder = OUTPUT / "cases" / spec["case_id"]
        if not cached_case(folder, spec, identity):
            policies = checked_input(spec["policy_artifact"], entries)
            source = checked_input(spec["source_head_artifact"], entries)
            evaluator = checked_input(spec["evaluator_artifact"], entries)
            if spec["evaluator_kind"] == "truth" and spec["source_head_kind"] == "fitted":
                scores = checked_input(spec["truth_scores_artifact"], entries)
                key = "true_selected_free" if spec["policy_field"] == "best" else "true_selected_static"
                expected = float(scores[key])
            else:
                expected = float(policies[spec["policy_field"]]["predicted"])
            result = evaluate_case(spec, dimensions, policies, source, evaluator, expected)
            result["input_hashes"] = {
                value: entries[value]["sha256"] for key, value in spec.items() if key.endswith("_artifact")
            }
            write_json(folder / "result.json", result)
            write_json(
                folder / "complete.json",
                {"case": spec, "identity": identity, "result_sha256": sha256(folder / "result.json")},
            )
        records.append(json.loads((folder / "result.json").read_text()))
        receipts[spec["case_id"]] = sha256(folder / "complete.json")
    require_execution()
    verify_seal(PRIOR, SEAL_SHA256)
    write_json(
        OUTPUT / "summary.json",
        {
            "identity": identity,
            "case_count": len(records),
            "case_receipts": receipts,
            "frozen_prediction_parity_pass": all(item["frozen_prediction_parity_pass"] for item in records),
            "bounds_available_count": sum(item["matching_objective_bound_available"] for item in records),
            "bounds_passing_parity_and_feasibility_count": sum(
                item["bound_parity_and_feasibility_checks_pass"] for item in records
            ),
            "cases": records,
            "scope": "Saved-only descriptive fixed-aggregate bounds; no fitting, optimization, or qualification change",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
