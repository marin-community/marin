# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the campaign's paired loop and fail-closed correctness gates to saved audits."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

NON_TARGET_LOOP_ALLOWANCE = 3
MINIMUM_PASS_LOSS_ALLOWANCE = 3
DOMAIN_PASS_LOSS_PERCENT = 5
PANEL_PASS_LOSS_PERCENT = 2
FINAL_REPEAT_LOOP_MARGIN = 2


def campaign_result(manifest: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Require every measured domain and the pooled 32K panel to meet the design."""
    names = set(manifest["cohorts"])
    for audit in (baseline, candidate):
        if any(set(audit[key]) != names for key in ("domains", "target_status", "cohort_hashes")):
            raise ValueError("Success requires a complete audit of every frozen cohort")
        if any(status not in ("target", "non-target") for status in audit["target_status"].values()):
            raise ValueError("Success requires a sealed baseline, including all required reserves")
    if candidate["target_status"] != baseline["target_status"]:
        raise ValueError("Candidate target decisions differ from the sealed baseline")
    if candidate["cohort_hashes"] != baseline["cohort_hashes"]:
        raise ValueError("Candidate prompt hashes differ from the sealed baseline")
    domains = {}
    for name in sorted(names):
        before = baseline["domains"][name]
        after = candidate["domains"][name]
        if before["N"] != after["N"]:
            raise ValueError(f"Candidate cohort size differs for {name}")
        target = baseline["target_status"][name] == "target"
        loop_max = before["confirmed"] // 2 if target else before["confirmed"] + NON_TARGET_LOOP_ALLOWANCE
        passes_min = before["passes"] - max(MINIMUM_PASS_LOSS_ALLOWANCE, before["N"] * DOMAIN_PASS_LOSS_PERCENT // 100)
        domains[name] = {
            "N": before["N"],
            "target": target,
            "baseline_confirmed": before["confirmed"],
            "candidate_UB": after["UB"],
            "loop_max": loop_max,
            "loops_ok": after["UB"] <= loop_max,
            "baseline_passes": before["passes"],
            "candidate_passes": after["passes"],
            "passes_min": passes_min,
            "correctness_ok": after["passes"] >= passes_min,
        }
    panel = {name: row for name, row in domains.items() if name != "math16k"}
    panel_n = sum(row["N"] for row in panel.values())
    panel_before = sum(row["baseline_passes"] for row in panel.values())
    panel_after = sum(row["candidate_passes"] for row in panel.values())
    panel_min = panel_before - panel_n * PANEL_PASS_LOSS_PERCENT // 100
    return {
        "domains": domains,
        "panel": {
            "N": panel_n,
            "baseline_passes": panel_before,
            "candidate_passes": panel_after,
            "passes_min": panel_min,
            "correctness_ok": panel_after >= panel_min,
        },
        "meets_success": (
            all(row["loops_ok"] and row["correctness_ok"] for row in domains.values()) and panel_after >= panel_min
        ),
        "repeat_final_greedy": any(
            abs(row["candidate_UB"] - row["loop_max"]) <= FINAL_REPEAT_LOOP_MARGIN for row in domains.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    baseline = json.loads(args.baseline.read_bytes())
    candidate = json.loads(args.candidate.read_bytes())
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if any(audit["manifest_sha256"] != digest for audit in (baseline, candidate)):
        raise ValueError("Audit manifest digest does not match the frozen manifest")
    result = campaign_result(json.loads(manifest_bytes), baseline, candidate)
    result["baseline"] = str(args.baseline)
    result["candidate"] = str(args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
