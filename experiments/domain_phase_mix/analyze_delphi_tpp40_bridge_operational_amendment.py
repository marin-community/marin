# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the dated TPP40 operational amendment to a completed v4 bridge report.

The frozen v4 result remains authoritative for the original equivalence claim.
This analyzer adds a separate, fail-closed large-effect screen for deciding
whether to launch a region-balanced production panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = (
    SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "delphi_tpp40_europe_readiness_20260830"
)
AMENDMENT_PATH = REFERENCE_DIR / "bridge_operational_amendment_v2.json"
HISTORICAL_AMENDMENT_PATH = REFERENCE_DIR / "bridge_operational_amendment_v1.json"
V4_REPORT_PATH = REFERENCE_DIR / "bridge_acceptance_report_v3.json"
V4_CONTRACT_PATH = REFERENCE_DIR / "bridge_acceptance_contract_v4.json"
NOISE_AUDIT_PATH = REFERENCE_DIR / "bridge_uncheatable_noise_audit_v2.json"
OUTPUT_PATH = REFERENCE_DIR / "bridge_operational_authorization_v2.json"
EXPECTED_AMENDMENT_SHA256 = "178ca635b116ee1b6bf1c09a80e1de43d639a99594300f57402ad55fdb9dcd5a"
EXPECTED_HISTORICAL_AMENDMENT_SHA256 = "32b6fd0b2a27dbacde1168c1bca297ff6528c447b045c754f9f4a67e9f609765"
EXPECTED_V4_CONTRACT_SHA256 = "f0441b8927e3e7d32bbdbe781ed3008dbb46a1cd98ff540661423e850ee936df"
LEGACY_EAST5_PARENT = "/calvinxu/dm-delphi-augmented-swarm-tpp40-phase0ckpt-interactive-retry8-20260825"
TERMINAL_IRIS_STATES = frozenset({"failed", "killed", "succeeded"})
EXPECTED_EAST5_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
)
EXPECTED_EUROPE_ROOT = (
    "gs://marin-eu-west4/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _load_amendment(path: Path) -> tuple[dict[str, Any], str]:
    historical_sha256 = _sha256_bytes(HISTORICAL_AMENDMENT_PATH.read_bytes())
    if historical_sha256 != EXPECTED_HISTORICAL_AMENDMENT_SHA256:
        raise ValueError(
            f"Historical operational amendment changed: {historical_sha256} "
            f"!= {EXPECTED_HISTORICAL_AMENDMENT_SHA256}"
        )
    encoded = path.read_bytes()
    observed_sha256 = _sha256_bytes(encoded)
    if observed_sha256 != EXPECTED_AMENDMENT_SHA256:
        raise ValueError(f"Operational amendment changed: {observed_sha256} != {EXPECTED_AMENDMENT_SHA256}")
    amendment = json.loads(encoded)
    if amendment["superseded_decision"]["contract_sha256"] != EXPECTED_V4_CONTRACT_SHA256:
        raise ValueError("Operational amendment refers to the wrong v4 contract")
    if amendment["frozen_inputs"]["run_orders"] != [2]:
        raise ValueError("Operational amendment must remain bound to run_order=2")
    return amendment, observed_sha256


def _threshold_result(section: dict[str, Any], *, maximum: float) -> dict[str, Any]:
    threshold = section.get("threshold")
    pairs = section.get("pairs")
    if not isinstance(threshold, dict) or not isinstance(pairs, list):
        return {"passed": False, "reason": "missing threshold or pair inventory"}
    observed_count = threshold.get("observed_pair_count")
    expected_count = threshold.get("expected_pair_count")
    deltas = threshold.get("signed_paired_deltas")
    if observed_count != 1 or expected_count != 1 or not isinstance(deltas, list) or len(deltas) != 1:
        return {
            "passed": False,
            "reason": "expected exactly one complete paired delta",
            "observed_pair_count": observed_count,
            "expected_pair_count": expected_count,
        }
    delta = deltas[0]
    if isinstance(delta, bool) or not isinstance(delta, int | float) or not math.isfinite(float(delta)):
        return {"passed": False, "reason": "paired delta is not finite"}
    if len(pairs) != 1 or pairs[0].get("run_order") != 2:
        return {"passed": False, "reason": "paired result is not the frozen run_order=2"}
    absolute_delta = abs(float(delta))
    component_deltas = pairs[0].get("component_deltas")
    if not isinstance(component_deltas, dict):
        component_deltas = {}
    component_large_effect_warnings = {
        name: float(value)
        for name, value in component_deltas.items()
        if not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and abs(float(value)) > maximum
    }
    return {
        "passed": absolute_delta <= maximum,
        "run_order": 2,
        "signed_paired_delta": float(delta),
        "absolute_paired_delta": absolute_delta,
        "absolute_paired_delta_max": maximum,
        "component_deltas": component_deltas,
        "component_large_effect_warnings": component_large_effect_warnings,
        "component_deltas_are_gating": False,
    }


def _production_assignment_binding(
    assignment: dict[str, Any] | None,
    *,
    file_sha256: str | None,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if assignment is None or file_sha256 is None:
        return {"passed": False, "reason": "production assignment is missing"}

    claimed_sha256 = assignment.get("assignment_sha256")
    canonical_payload = dict(assignment)
    canonical_payload.pop("assignment_sha256", None)
    observed_sha256 = _sha256_bytes(json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode())
    assignments = assignment.get("assignments")
    expected_runs = assignment.get("expected_runs")
    freeze = assignment.get("freeze")
    errors: list[str] = []
    expected_file_sha256 = expected.get("file_sha256")
    expected_assignment_sha256 = expected.get("assignment_sha256")
    if expected_file_sha256 == "UNFROZEN" or expected_assignment_sha256 == "UNFROZEN":
        errors.append("production assignment digests remain unfrozen in the operational amendment")
    if file_sha256 != expected_file_sha256:
        errors.append("production assignment file digest does not match the operational amendment")
    if claimed_sha256 != expected_assignment_sha256:
        errors.append("production assignment semantic digest does not match the operational amendment")
    if not isinstance(claimed_sha256, str) or claimed_sha256 != observed_sha256:
        errors.append("production assignment semantic digest does not match its payload")
    if expected_runs != 280:
        errors.append("production assignment does not cover the frozen 280-row panel")
    if assignment.get("east5_root") != EXPECTED_EAST5_ROOT:
        errors.append("production assignment names the wrong East5 root")
    if assignment.get("europe_root") != EXPECTED_EUROPE_ROOT:
        errors.append("production assignment names the wrong Europe root")
    if not isinstance(freeze, dict):
        errors.append("production assignment lacks a legacy-parent quiescence snapshot")
    else:
        if freeze.get("legacy_parent_job") != LEGACY_EAST5_PARENT:
            errors.append("production assignment names the wrong legacy East5 parent")
        if freeze.get("legacy_parent_state") not in TERMINAL_IRIS_STATES:
            errors.append("production assignment was not frozen after the legacy East5 parent became terminal")
        observed_at = freeze.get("legacy_parent_observed_at_utc")
        try:
            parsed_observed_at = datetime.fromisoformat(observed_at) if isinstance(observed_at, str) else None
        except ValueError:
            parsed_observed_at = None
        if (
            parsed_observed_at is None
            or parsed_observed_at.tzinfo is None
            or parsed_observed_at.utcoffset() != UTC.utcoffset(parsed_observed_at)
        ):
            errors.append("production assignment lacks a UTC legacy-parent observation timestamp")
    if not isinstance(assignments, dict):
        errors.append("production assignment lacks assignment partitions")
    else:
        partitions: dict[str, set[int]] = {}
        for name in ("completed", "east5", "europe"):
            values = assignments.get(name)
            if not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in values
            ):
                errors.append(f"production assignment partition {name} is malformed")
                continue
            partitions[name] = set(values)
            if len(partitions[name]) != len(values):
                errors.append(f"production assignment partition {name} contains duplicates")
        if len(partitions) == 3:
            names = tuple(partitions)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    if partitions[left] & partitions[right]:
                        errors.append(f"production assignment partitions {left} and {right} overlap")
            if set().union(*partitions.values()) != set(range(280)):
                errors.append("production assignment partitions do not cover exactly run_order 0..279")
            for name in names:
                values = assignments[name]
                if values != sorted(values):
                    errors.append(f"production assignment partition {name} is not sorted")
            resumable = assignments.get("resumable_east5")
            if not isinstance(resumable, list) or not set(resumable) <= partitions["east5"]:
                errors.append("production assignment resumable East5 rows are malformed or routed outside East5")
    observed = assignment.get("observed")
    if not isinstance(observed, dict):
        errors.append("production assignment lacks the materializer observation inventory")
    else:
        for name in (
            "europe_final",
            "europe_phase0",
            "europe_executor_info",
            "europe_executor_status",
            "europe_success",
            "europe_final_without_success",
            "east5_final_without_success",
        ):
            if observed.get(name) != []:
                errors.append(f"production assignment observation {name} must be empty at freeze time")
        completed = assignments.get("completed") if isinstance(assignments, dict) else None
        if observed.get("east5_final") != completed or observed.get("east5_success") != completed:
            errors.append("production assignment completed rows do not match successful East5 final artifacts")
    return {
        "passed": not errors,
        "file_sha256": file_sha256,
        "assignment_sha256": claimed_sha256,
        "run_counts": (
            {
                name: len(values)
                for name, values in assignments.items()
                if name in {"completed", "east5", "europe"} and isinstance(values, list)
            }
            if isinstance(assignments, dict)
            else {}
        ),
        "errors": errors,
    }


def analyze_operational_authorization(
    *,
    amendment: dict[str, Any],
    amendment_sha256: str,
    v4_report: dict[str, Any],
    v4_report_sha256: str,
    v4_contract_sha256: str,
    noise_audit_sha256: str,
    production_assignment: dict[str, Any] | None,
    production_assignment_file_sha256: str | None,
) -> dict[str, Any]:
    """Return the fail-closed operational launch decision."""
    errors: list[str] = []
    frozen_inputs = amendment["frozen_inputs"]
    if v4_contract_sha256 != EXPECTED_V4_CONTRACT_SHA256:
        errors.append("v4 acceptance contract changed")
    if v4_report.get("acceptance_contract_sha256") != EXPECTED_V4_CONTRACT_SHA256:
        errors.append("v4 report refers to the wrong acceptance contract")
    if v4_report.get("path_manifest_sha256") != frozen_inputs["path_manifest_sha256"]:
        errors.append("v4 report refers to the wrong path manifest")
    evaluation_identity = v4_report.get("evaluation_data_identity")
    if not isinstance(evaluation_identity, dict) or evaluation_identity.get("passed") is not True:
        errors.append("v4 evaluation-data identity did not pass")
    elif evaluation_identity.get("audit_sha256") != frozen_inputs["evaluation_data_audit_sha256"]:
        errors.append("v4 evaluation-data audit changed")
    training_identity = v4_report.get("training_data_identity")
    if not isinstance(training_identity, dict) or training_identity.get("passed") is not True:
        errors.append("v4 training-data identity did not pass")
    elif training_identity.get("audit_sha256") != frozen_inputs["training_data_audit_sha256"]:
        errors.append("v4 training-data audit changed")
    if noise_audit_sha256 != frozen_inputs["uncheatable_noise_audit_sha256"]:
        errors.append("frozen Uncheatable noise audit changed")
    idempotence = v4_report.get("idempotence")
    if not isinstance(idempotence, dict) or idempotence.get("passed") is not True:
        errors.append("v4 idempotence audit did not pass")
    blocking_errors = v4_report.get("blocking_errors")
    if not isinstance(blocking_errors, list):
        errors.append("v4 report lacks a blocking-error inventory")
    elif blocking_errors:
        errors.append("v4 report retains non-numerical blocking errors")
    if v4_report.get("numerical_acceptance_passed") is not False:
        errors.append("v4 numerical verdict must remain the recorded failure")
    if v4_report.get("production_launch_authorized") is not False:
        errors.append("v4 production verdict must remain unauthorized")

    limits = amendment["decision"]["absolute_paired_delta_max_bpb"]
    uncheatable = v4_report.get("uncheatable")
    table9 = v4_report.get("table9")
    if not isinstance(uncheatable, dict):
        uncheatable = {}
    if not isinstance(table9, dict):
        table9 = {}
    screens = {
        "uncheatable_phase_0": _threshold_result(
            uncheatable.get("phase_0", {}),
            maximum=float(limits["uncheatable_phase_0"]),
        ),
        "uncheatable_endpoint": _threshold_result(
            uncheatable.get("endpoint", {}),
            maximum=float(limits["uncheatable_endpoint"]),
        ),
        "table9_macro_endpoint": _threshold_result(
            table9,
            maximum=float(limits["table9_macro_endpoint"]),
        ),
    }
    for name, screen in screens.items():
        if screen["passed"] is not True:
            errors.append(f"operational large-effect screen failed: {name}")

    bridge_screen_passed = not errors
    production_assignment_binding = _production_assignment_binding(
        production_assignment,
        file_sha256=production_assignment_file_sha256,
        expected=frozen_inputs["production_assignment"],
    )
    if production_assignment_binding["passed"] is not True:
        errors.append("production assignment is not frozen and valid")
    authorized = not errors
    return {
        "schema_version": 1,
        "amendment_sha256": amendment_sha256,
        "historical_amendment_v1_sha256": EXPECTED_HISTORICAL_AMENDMENT_SHA256,
        "v4_acceptance_contract_sha256": EXPECTED_V4_CONTRACT_SHA256,
        "v4_acceptance_contract_file_sha256": v4_contract_sha256,
        "v4_acceptance_report_sha256": v4_report_sha256,
        "uncheatable_noise_audit_sha256": noise_audit_sha256,
        "v4_numerical_acceptance_passed": v4_report.get("numerical_acceptance_passed"),
        "v4_production_launch_authorized": v4_report.get("production_launch_authorized"),
        "decision_name": amendment["decision"]["name"],
        "interpretation": amendment["decision"]["interpretation"],
        "threshold_origin": amendment["decision"]["threshold_origin"],
        "component_policy": amendment["decision"]["component_policy"],
        "preserved_findings": amendment["preserved_findings"],
        "v4_thresholds": {
            "uncheatable_phase_0": uncheatable.get("phase_0", {}).get("threshold"),
            "uncheatable_endpoint": uncheatable.get("endpoint", {}).get("threshold"),
            "table9_macro_endpoint": table9.get("threshold"),
        },
        "screens": screens,
        "idempotence": idempotence,
        "operational_bridge_screen_passed": bridge_screen_passed,
        "production_assignment": production_assignment_binding,
        "operational_production_launch_authorized": authorized,
        "blocking_errors": errors,
        "required_production_followup": amendment["production_followup"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=AMENDMENT_PATH)
    parser.add_argument("--v4-report", type=Path, default=V4_REPORT_PATH)
    parser.add_argument("--v4-contract", type=Path, default=V4_CONTRACT_PATH)
    parser.add_argument("--noise-audit", type=Path, default=NOISE_AUDIT_PATH)
    parser.add_argument("--production-assignment", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    amendment, amendment_sha256 = _load_amendment(args.amendment)
    v4_report_bytes = args.v4_report.read_bytes()
    v4_contract_bytes = args.v4_contract.read_bytes()
    noise_audit_bytes = args.noise_audit.read_bytes()
    assignment_bytes = args.production_assignment.read_bytes() if args.production_assignment is not None else None
    assignment = json.loads(assignment_bytes) if assignment_bytes is not None else None
    assignment_file_sha256 = _sha256_bytes(assignment_bytes) if assignment_bytes is not None else None
    report = analyze_operational_authorization(
        amendment=amendment,
        amendment_sha256=amendment_sha256,
        v4_report=json.loads(v4_report_bytes),
        v4_report_sha256=_sha256_bytes(v4_report_bytes),
        v4_contract_sha256=_sha256_bytes(v4_contract_bytes),
        noise_audit_sha256=_sha256_bytes(noise_audit_bytes),
        production_assignment=assignment,
        production_assignment_file_sha256=assignment_file_sha256,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["operational_production_launch_authorized"]:
        raise RuntimeError(f"Operational bridge screen failed closed; see {args.output}")


if __name__ == "__main__":
    main()
