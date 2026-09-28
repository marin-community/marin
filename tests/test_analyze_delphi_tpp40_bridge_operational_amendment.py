# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from experiments.domain_phase_mix import analyze_delphi_tpp40_bridge_operational_amendment as operational


def _amendment(assignment: dict | None = None) -> dict:
    assignment = assignment or _assignment()
    return {
        "frozen_inputs": {
            "run_orders": [2],
            "path_manifest_sha256": "a" * 64,
            "evaluation_data_audit_sha256": "b" * 64,
            "training_data_audit_sha256": "c" * 64,
            "uncheatable_noise_audit_sha256": "e" * 64,
            "production_assignment": {
                "file_sha256": "1" * 64,
                "assignment_sha256": assignment["assignment_sha256"],
            },
        },
        "decision": {
            "name": "single-pair large-effect screen",
            "interpretation": "operational screen",
            "threshold_origin": "frozen any-row bound",
            "component_policy": "diagnostic only",
            "absolute_paired_delta_max_bpb": {
                "uncheatable_phase_0": 0.005,
                "uncheatable_endpoint": 0.005,
                "table9_macro_endpoint": 0.005,
            },
        },
        "preserved_findings": ["v4 failed"],
        "production_followup": {"required": True},
    }


def _section(delta: float, *, component_delta: float = 0.0) -> dict:
    return {
        "pairs": [{"run_order": 2, "component_deltas": {"component": component_delta}}],
        "threshold": {
            "expected_pair_count": 1,
            "observed_pair_count": 1,
            "signed_paired_deltas": [delta],
        },
    }


def _assignment() -> dict:
    assignment = {
        "expected_runs": 280,
        "east5_root": operational.EXPECTED_EAST5_ROOT,
        "europe_root": operational.EXPECTED_EUROPE_ROOT,
        "freeze": {
            "legacy_parent_job": operational.LEGACY_EAST5_PARENT,
            "legacy_parent_state": "killed",
            "legacy_parent_observed_at_utc": "2026-08-31T05:00:00+00:00",
        },
        "assignments": {
            "completed": list(range(20)),
            "east5": list(range(20, 150)),
            "europe": list(range(150, 280)),
            "resumable_east5": [20],
        },
        "observed": {
            "east5_final": list(range(20)),
            "east5_success": list(range(20)),
            "east5_final_without_success": [],
            "europe_final": [],
            "europe_phase0": [],
            "europe_executor_info": [],
            "europe_executor_status": [],
            "europe_success": [],
            "europe_final_without_success": [],
        },
    }
    canonical = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
    assignment["assignment_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return assignment


def _v4_report(delta: float = 0.004) -> dict:
    return {
        "acceptance_contract_sha256": operational.EXPECTED_V4_CONTRACT_SHA256,
        "path_manifest_sha256": "a" * 64,
        "evaluation_data_identity": {"passed": True, "audit_sha256": "b" * 64},
        "training_data_identity": {"passed": True, "audit_sha256": "c" * 64},
        "idempotence": {"passed": True},
        "blocking_errors": [],
        "numerical_acceptance_passed": False,
        "production_launch_authorized": False,
        "uncheatable": {
            "phase_0": _section(delta),
            "endpoint": _section(delta),
        },
        "table9": _section(delta),
    }


def _analyze(v4_report: dict, *, assignment: dict | None = None, noise_audit_sha256: str = "e" * 64) -> dict:
    assignment = _assignment() if assignment is None else assignment
    return operational.analyze_operational_authorization(
        amendment=_amendment(assignment),
        amendment_sha256="d" * 64,
        v4_report=v4_report,
        v4_report_sha256="f" * 64,
        v4_contract_sha256=operational.EXPECTED_V4_CONTRACT_SHA256,
        noise_audit_sha256=noise_audit_sha256,
        production_assignment=assignment,
        production_assignment_file_sha256="1" * 64,
    )


def test_complete_report_below_large_effect_limit_authorizes_operational_launch() -> None:
    report = _analyze(_v4_report())

    assert report["v4_numerical_acceptance_passed"] is False
    assert report["operational_production_launch_authorized"] is True
    assert report["blocking_errors"] == []


def test_delta_above_large_effect_limit_blocks_operational_launch() -> None:
    report = _analyze(_v4_report(delta=0.005001))

    assert report["operational_production_launch_authorized"] is False
    assert set(report["blocking_errors"]) == {
        "operational large-effect screen failed: uncheatable_phase_0",
        "operational large-effect screen failed: uncheatable_endpoint",
        "operational large-effect screen failed: table9_macro_endpoint",
    }


def test_large_component_delta_is_reported_but_does_not_change_macro_gate() -> None:
    v4_report = _v4_report()
    v4_report["uncheatable"]["phase_0"] = _section(0.004, component_delta=0.012)

    report = _analyze(v4_report)

    assert report["operational_production_launch_authorized"] is True
    assert report["screens"]["uncheatable_phase_0"]["component_large_effect_warnings"] == {"component": 0.012}
    assert report["screens"]["uncheatable_phase_0"]["component_deltas_are_gating"] is False


def test_missing_pair_blocks_operational_launch() -> None:
    v4_report = _v4_report()
    v4_report["uncheatable"]["endpoint"]["pairs"] = []
    v4_report["uncheatable"]["endpoint"]["threshold"]["observed_pair_count"] = 0
    v4_report["uncheatable"]["endpoint"]["threshold"]["signed_paired_deltas"] = []

    report = _analyze(v4_report)

    assert report["operational_production_launch_authorized"] is False
    assert "operational large-effect screen failed: uncheatable_endpoint" in report["blocking_errors"]


def test_non_numerical_v4_blocker_is_not_waived() -> None:
    v4_report = _v4_report()
    v4_report["blocking_errors"] = ["result inventory changed"]

    report = _analyze(v4_report)

    assert report["operational_production_launch_authorized"] is False
    assert "v4 report retains non-numerical blocking errors" in report["blocking_errors"]


def test_v4_verdict_cannot_be_rewritten_as_pass() -> None:
    v4_report = deepcopy(_v4_report())
    v4_report["numerical_acceptance_passed"] = True
    v4_report["production_launch_authorized"] = True

    report = _analyze(v4_report)

    assert report["operational_production_launch_authorized"] is False
    assert "v4 numerical verdict must remain the recorded failure" in report["blocking_errors"]
    assert "v4 production verdict must remain unauthorized" in report["blocking_errors"]


def test_noise_audit_digest_is_bound() -> None:
    report = _analyze(_v4_report(), noise_audit_sha256="0" * 64)

    assert report["operational_production_launch_authorized"] is False
    assert "frozen Uncheatable noise audit changed" in report["blocking_errors"]


def test_missing_production_assignment_blocks_only_production_authorization() -> None:
    report = operational.analyze_operational_authorization(
        amendment=_amendment(),
        amendment_sha256="d" * 64,
        v4_report=_v4_report(),
        v4_report_sha256="f" * 64,
        v4_contract_sha256=operational.EXPECTED_V4_CONTRACT_SHA256,
        noise_audit_sha256="e" * 64,
        production_assignment=None,
        production_assignment_file_sha256=None,
    )

    assert report["operational_bridge_screen_passed"] is True
    assert report["operational_production_launch_authorized"] is False
    assert report["production_assignment"]["passed"] is False


def test_overlapping_production_assignment_is_rejected() -> None:
    assignment = _assignment()
    assignment["assignments"]["europe"].append(20)
    canonical_payload = dict(assignment)
    canonical_payload.pop("assignment_sha256")
    assignment["assignment_sha256"] = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    report = _analyze(_v4_report(), assignment=assignment)

    assert report["operational_production_launch_authorized"] is False
    assert "production assignment is not frozen and valid" in report["blocking_errors"]
    assert "production assignment partitions east5 and europe overlap" in report["production_assignment"]["errors"]


def test_assignment_digest_must_be_frozen_by_the_amendment() -> None:
    assignment = _assignment()
    amendment = _amendment(assignment)
    amendment["frozen_inputs"]["production_assignment"] = {
        "file_sha256": "UNFROZEN",
        "assignment_sha256": "UNFROZEN",
    }

    report = operational.analyze_operational_authorization(
        amendment=amendment,
        amendment_sha256="d" * 64,
        v4_report=_v4_report(),
        v4_report_sha256="f" * 64,
        v4_contract_sha256=operational.EXPECTED_V4_CONTRACT_SHA256,
        noise_audit_sha256="e" * 64,
        production_assignment=assignment,
        production_assignment_file_sha256="1" * 64,
    )

    assert report["operational_production_launch_authorized"] is False
    assert (
        "production assignment digests remain unfrozen in the operational amendment"
        in report["production_assignment"]["errors"]
    )


def test_pre_quiescence_assignment_is_rejected() -> None:
    assignment = _assignment()
    assignment.pop("freeze")
    canonical_payload = dict(assignment)
    canonical_payload.pop("assignment_sha256")
    assignment["assignment_sha256"] = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    report = _analyze(_v4_report(), assignment=assignment)

    assert report["operational_production_launch_authorized"] is False
    assert "production assignment lacks a legacy-parent quiescence snapshot" in report["production_assignment"]["errors"]


def test_on_disk_amendment_matches_pinned_digest() -> None:
    amendment, observed_sha256 = operational._load_amendment(operational.AMENDMENT_PATH)

    assert observed_sha256 == operational.EXPECTED_AMENDMENT_SHA256
    assert amendment["frozen_inputs"]["run_orders"] == [2]
    assert (
        operational._sha256_bytes(operational.HISTORICAL_AMENDMENT_PATH.read_bytes())
        == operational.EXPECTED_HISTORICAL_AMENDMENT_SHA256
    )
