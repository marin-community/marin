# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize completed C6 runtime histories without selecting a release gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_gradient_conflict_stress_gate_backtest_20260811"
SOURCE_REPORTS = (
    (
        "original-c06",
        REFERENCE_OUTPUTS / "starcoder_wsd80_gradient_conflict_stress_results_20260811/stage-c06/runtime_gate.json",
    ),
    (
        "retry3-c06",
        REFERENCE_OUTPUTS
        / "starcoder_wsd80_gradient_conflict_stress_retry3_results_20260811/stage-c06/runtime_gate.json",
    ),
    (
        "retry5-c06",
        REFERENCE_OUTPUTS
        / "starcoder_wsd80_gradient_conflict_stress_retry5_results_20260811/stage-c06/runtime_gate.json",
    ),
    (
        "retry8-c06",
        REFERENCE_OUTPUTS
        / "starcoder_wsd80_gradient_conflict_stress_retry8_results_20260811/stage-c06/runtime_gate.json",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_summary(label: str, path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text())
    if source["analysis_scope"] != "operational_only_no_endpoint_metrics" or source["endpoint_metrics_read"]:
        raise ValueError(f"{label} is not an endpoint-blind operational report")
    optimizer_decay = next(
        diagnostic for diagnostic in source["stress_event_recovery"] if diagnostic["event"] == "optimizer_decay"
    )
    data_switch = next(
        diagnostic for diagnostic in source["stress_event_recovery"] if diagnostic["event"] == "data_switch"
    )
    return {
        "label": label,
        "source_report": str(path),
        "source_report_sha256": _sha256(path),
        "source_report_version": source["report_version"],
        "source_status_under_then-current_gate": source["status"],
        "optimizer_decay_recovery_median": optimizer_decay["recovery_median"],
        "optimizer_decay_recovery_p25": optimizer_decay["recovery_p25"],
        "optimizer_decay_slowdown_positions_max": optimizer_decay["slowdown_positions_max"],
        "data_switch_recovery_median": data_switch["recovery_median"],
        "data_switch_slowdown_positions_max": data_switch["slowdown_positions_max"],
    }


def main() -> None:
    results = [_historical_summary(label, path) for label, path in SOURCE_REPORTS]
    output = {
        "report_version": "2026-08-11-stress-history-audit-v2",
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "gate_selection_allowed": False,
        "interpretation": (
            "Historical reports retain the pass/fail status of the gate active at the time. "
            "This audit has no expected labels and does not certify the generation-9 gate."
        ),
        "fresh_generation": stress.DEFAULT_GENERATION,
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "backtest.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# WSD80 gradient-conflict C6 history audit",
        "",
        "Endpoint losses were neither requested nor used. This is chronology, not gate selection.",
        "",
        "| Attempt | Historical verdict | Decay median | Decay p25 | Decay max slowdown | Switch median |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['label']} | {result['source_status_under_then-current_gate']} | "
            f"{result['optimizer_decay_recovery_median']:.6f} | "
            f"{result['optimizer_decay_recovery_p25']:.6f} | "
            f"{result['optimizer_decay_slowdown_positions_max']} | "
            f"{result['data_switch_recovery_median']:.6f} |"
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output_path), "sha256": _sha256(output_path)}))


if __name__ == "__main__":
    main()
