# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the narrowly scoped Gen19 analyzer semantic correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT = REPO_ROOT / (
    "experiments/domain_phase_mix/" "starcoder_wsd80_gradient_conflict_resume_analyzer_revision2_20260813.json"
)
ORIGINAL_ANALYZER = (
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "analyze_starcoder_wsd80_gradient_conflict_resume_canary_20260813.py"
)
REVISION_ANALYZER = (
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "analyze_starcoder_wsd80_gradient_conflict_resume_canary_revision2_20260813.py"
)
REVISION_TESTS = "tests/test_starcoder_wsd80_gradient_conflict_resume_analyzer_revision2.py"
FAILED_REPORT = (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_resume_canary_20260813/runtime_gate.json"
)
ORIGINAL_PREREGISTRATION = (
    "experiments/domain_phase_mix/"
    "starcoder_wsd80_gradient_conflict_resume_canary_preregistration_generation19_20260813.json"
)
ORIGINAL_PREREGISTRATION_SHA256 = "1c3a125f2521116db8a10ebb87a7e52e3bfe6128539af55da1ee9308f45b418f"
ORIGINAL_FAILED_REPORT_SHA256 = "d4271cd8c904cd695269cdf3aba539eacafe46834930cd9a347ca985a14810ea"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_revision() -> dict[str, Any]:
    """Return the revision contract after checking immutable predecessor evidence."""
    if _sha256(REPO_ROOT / ORIGINAL_PREREGISTRATION) != ORIGINAL_PREREGISTRATION_SHA256:
        raise ValueError("Original Gen19 preregistration drifted")
    if _sha256(REPO_ROOT / FAILED_REPORT) != ORIGINAL_FAILED_REPORT_SHA256:
        raise ValueError("Original failed Gen19 runtime report drifted")
    failed_report = json.loads((REPO_ROOT / FAILED_REPORT).read_text())
    if failed_report.get("status") != "fail" or failed_report.get("endpoint_metrics_read") is not False:
        raise ValueError("Original report is not the expected endpoint-blind failure")

    dependencies = (
        ORIGINAL_ANALYZER,
        REVISION_ANALYZER,
        REVISION_TESTS,
        "lib/levanter/src/levanter/callbacks/_core.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/levanter/src/levanter/main/train_lm.py",
    )
    return {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "decision": {
            "next_on_fail": "do not launch the production-cadence C64 recovery gate",
            "next_on_pass": "seek independent review before accepting Gen19",
        },
        "endpoint_metrics_read": False,
        "implementation_sha256": {path: _sha256(REPO_ROOT / path) for path in dependencies},
        "original_failed_report_path": FAILED_REPORT,
        "original_failed_report_sha256": ORIGINAL_FAILED_REPORT_SHA256,
        "original_preregistration_path": ORIGINAL_PREREGISTRATION,
        "original_preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "rationale": {
            "checkpoint": (
                "StepInfo.step is the completed step N, while the checkpointed TrainerState.step is next_step N+1."
            ),
            "wandb": (
                "TrainerHooks suppresses ordinary hooks for info.step <= 1, so complete operational history is "
                f"global steps 2..{stress.TERMINAL_STEP}."
            ),
        },
        "semantic_corrections": {
            "checkpoint_metadata_step_to_restored_state_step": "N_to_N_plus_1",
            "operational_wandb_global_steps": [2, stress.TERMINAL_STEP],
        },
        "unchanged_contract": {
            "all_original_operational_gates_preserved": True,
            "endpoint_metrics_remain_unread": True,
            "fault_plan_unchanged": True,
            "no_threshold_changed": True,
            "runtime_evidence_unchanged": True,
        },
        "version": "2026-08-13-gen19-analyzer-semantic-correction-v2",
    }


def main() -> None:
    payload = json.dumps(build_revision(), indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(payload)
    print(json.dumps({"path": str(OUTPUT), "sha256": _sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
