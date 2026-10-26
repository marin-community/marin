# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""TP044: paired invented-state clock/readout isolation; no fitting or outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_readout_isolation_20261015 as reference
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_competitive_hazard_clock_20261015 as competitive,
)

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015/competitive_clock_isolation"
REFERENCE = reference.OUTPUT
SCOPE = "tp044_exact_tp041_mixed_aggregate_half_mu3_24_paired_states"
CONTRACT: dict[str, Any] = {
    key: reference.CONTRACT[key]
    for key in (
        "pool_inventories",
        "clocks",
        "rate",
        "power",
        "phase_fractions",
        "main_token_totals",
        "fixed_heads",
        "floor",
        "intercept",
        "harm_coefficients",
        "harm_threshold",
        "quadrature_relative_tolerance",
        "quadrature_max_panels",
        "floating_reconstruction_tolerance",
    )
} | {
    "candidate_id": "TP40-044",
    "scope": SCOPE,
    "selection": {"stratum": "main", "inventory": "mixed", "aggregate_first": 0.5, "mu": 3.0},
    "interference_clocks": ["novelty", "donor_hazard"],
    "readouts": ["hazard", "additive"],
    "new_state_cases": 24,
    "sealed_reference_state_cases": 24,
    "new_fixed_head_rows": 48,
    "sealed_reference_fixed_head_rows": 48,
    "scalar_head_readout_values": 192,
    "novelty_reference": "Reuse exact sealed TP041 cells; do not re-integrate or refit",
    "native_restriction": "Exact tied and zero-mu benefit; physical cumulative harm unchanged",
    "overlap": "p1/global-pool/singleton donor hazard equals NG005 after fixed-budget mu rescaling",
    "interpretation": "Distinct donor clock and readout effects only; no empirical or estimator qualification",
    "scientific_acceptance_gate": None,
    "fits": False,
    "policy_optimization": False,
    "empirical_inputs": False,
    "cap_or_clipping": False,
    "failed_cells": "Retain every numerical failure; no silent retry, budget increase or omission",
}


def load_readout_backend() -> ModuleType:
    """Load the frozen TP041 evaluator privately, replacing only its state law."""
    name = "_tp044_frozen_readout_backend"
    spec = importlib.util.spec_from_file_location(name, Path(reference.__file__))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen readout evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.state = competitive
    module.CONTRACT = CONTRACT
    return module


BACKEND = load_readout_backend()


def cases() -> tuple[reference.Case, ...]:
    selected = tuple(
        case
        for case in reference.cases()
        if all(getattr(case, key) == value for key, value in CONTRACT["selection"].items())
    )
    if len(selected) != 24 or len({case.case_id for case in selected}) != 24:
        raise ValueError("The exact TP041 subset changed")
    return selected


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        BASE / "test_tpp40_competitive_clock_isolation_20261015.py",
        Path(competitive.__file__).resolve(),
        BASE / "test_tpp40_competitive_hazard_clock_20261015.py",
        Path(reference.__file__).resolve(),
        Path(reference.state.__file__).resolve(),
        OUTPUT / "proposal_v1.md",
        OUTPUT.parent / "reviews/competitive_hazard_clock_concept_v1.md",
        OUTPUT.parent / "reviews/competitive_hazard_clock_independent_assessment_v1.md",
        OUTPUT.parent / "pure_qualification/result_manifest_v1.json",
        REFERENCE / "result_manifest_v1.json",
    )


def verify_reference_seal() -> int:
    """Require complete immutable TP041 numerical evidence before any new state."""
    manifest = json.loads((REFERENCE / "result_manifest_v1.json").read_text())
    for relative, entry in manifest["files"].items():
        if reference.digest(REFERENCE / relative) != entry["sha256"]:
            raise ValueError(f"Sealed TP041 artifact changed: {relative}")
    audit = json.loads((REFERENCE / "audit/independent_replay_report_v1.json").read_text())
    if not audit["passed"]:
        raise ValueError("Require the independent TP041 replay")
    return len(manifest["files"])


def identity() -> dict[str, Any]:
    registered = json.loads((OUTPUT / "registration.json").read_text())
    if registered["status"] != "registered" or registered["contract"] != CONTRACT:
        raise ValueError("Require the exact TP044 registration")
    if registered["runtime"] != reference.runtime():
        raise ValueError("Frozen runtime changed")
    if set(registered["source_hashes"]) != {str(path) for path in source_paths()}:
        raise ValueError("Source pins must match the exact whitelist")
    for path, expected in registered["source_hashes"].items():
        if reference.digest(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return {
        "registration_sha256": reference.digest(OUTPUT / "registration.json"),
        "source_hashes": registered["source_hashes"],
        "runtime": reference.runtime(),
        "sealed_reference_files": verify_reference_seal(),
    }


def load_reference(case: reference.Case, root: Path) -> tuple[dict[str, Any], str]:
    """Adopt one exact reference cell without regenerating its state or head."""
    path = root / "cells" / f"{case.case_id}.json"
    saved = json.loads(path.read_text())
    record = saved["result"]
    if record["case"] != json.loads(json.dumps(asdict(case))):
        raise ValueError("Reference case covariates differ")
    if saved["result_sha256"] != reference.payload_digest(record):
        raise ValueError("Reference payload hash differs")
    return record, reference.digest(path)


def evaluate_pair(case: reference.Case, old: dict[str, Any], control: competitive.Quadrature) -> dict[str, Any]:
    """Evaluate only the new clock, retaining any failed side of the comparison."""
    new: dict[str, Any]
    try:
        new = BACKEND.evaluate_case(case, control)
    except ArithmeticError as error:
        new = {
            "case": asdict(case),
            "status": "numerical_failure",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    result: dict[str, Any] = {"case": asdict(case), "novelty": old, "donor_hazard": new}
    complete = old["status"] == new["status"] == "complete"
    result["status"] = "complete" if complete else "paired_numerical_failure"
    result["numerical_checks_passed"] = bool(
        complete and old["numerical_checks_passed"] and new["numerical_checks_passed"]
    )
    if not complete:
        return result
    for name in ("weights", "masses", "phase_epochs", "phase_token_totals", "expected_phase_token_totals"):
        if old[name] != new[name]:
            raise ValueError(f"Paired physical inputs differ: {name}")
    result["feature_changes_donor_minus_novelty"] = {
        name: (np.asarray(new["features"][name]) - np.asarray(old["features"][name])).tolist()
        for name in ("state_contrast", "log_attenuation", "hazard_departure", "additive_departure")
    }
    result["head_changes_donor_minus_novelty"] = {
        head: {name: value - old["head_values"][head][name] for name, value in values.items()}
        for head, values in new["head_values"].items()
    }
    return result


def run() -> None:
    current = identity()
    if current != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen execution identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    for key, expected in {
        "scope": SCOPE,
        "registration_sha256": current["registration_sha256"],
        "execution_identity_sha256": reference.digest(OUTPUT / "execution_identity.json"),
    }.items():
        if approval.get(key) != expected:
            raise ValueError("Require exact scientific execution approval")
    control = competitive.Quadrature(CONTRACT["quadrature_relative_tolerance"], CONTRACT["quadrature_max_panels"])
    expected_cases = {f"{case.case_id}.json" for case in cases()}
    if {path.name for path in (OUTPUT / "cells").glob("*.json")} - expected_cases:
        raise ValueError("Unexpected case artifacts outside the registered scope")
    results = []
    for case in cases():
        if identity() != current:
            raise ValueError("Execution sources or reference seal changed")
        old, source_sha = load_reference(case, REFERENCE)
        path = OUTPUT / "cells" / f"{case.case_id}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            result = saved["result"]
            if saved["identity"] != current or result["case"] != json.loads(json.dumps(asdict(case))):
                raise ValueError("Existing cell scope changed")
            if result["reference_cell_sha256"] != source_sha or result["novelty"] != old:
                raise ValueError("Existing reference adoption changed")
            if saved["result_sha256"] != reference.payload_digest(result):
                raise ValueError("Existing result payload changed")
        else:
            result = evaluate_pair(case, old, control)
            result["reference_cell_sha256"] = source_sha
            result["reference_cell_path"] = str(REFERENCE / "cells" / f"{case.case_id}.json")
            reference.write(
                path, {"identity": current, "result": result, "result_sha256": reference.payload_digest(result)}
            )
        results.append(result)
    reference.write(
        OUTPUT / "summary.json",
        {
            "scope": SCOPE,
            "paired_cases": len(results),
            "new_head_rows_expected": 48,
            "reference_head_rows_expected": 48,
            "scalar_head_readout_values_expected": 192,
            "completed_pairs": sum(item["status"] == "complete" for item in results),
            "numerical_failure_cases": [item["case"]["case_id"] for item in results if item["status"] != "complete"],
            "failed_checks": [item["case"]["case_id"] for item in results if not item["numerical_checks_passed"]],
            "scientific_qualification": False,
            "all_cells_retained": True,
        },
    )
    if identity() != current:
        raise ValueError("Execution source changed at completion")
    reference.write(
        OUTPUT / "complete.json",
        {
            "identity": current,
            "files": (
                {
                    str(path.relative_to(OUTPUT)): reference.digest(path)
                    for path in sorted((OUTPUT / "cells").glob("*.json"))
                }
                | {"summary.json": reference.digest(OUTPUT / "summary.json")}
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        reference.write(OUTPUT / "execution_identity.json", identity())
    else:
        run()


if __name__ == "__main__":
    main()
