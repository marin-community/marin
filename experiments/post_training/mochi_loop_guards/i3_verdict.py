# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved paired I3 audits under the registered measure and user override."""

import argparse
import hashlib
import json
import math

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.mochi_loop_guards.user_overrides import CORRECTNESS_MULTIPLIER, override_cohort

VALID_FRACTION = 0.9
REGISTRATION = "https://github.com/marin-community/marin/issues/12264#issuecomment-7171525899"


def paired_verdict(control: dict, treatment: dict, baseline: dict) -> dict:
    """Apply I3 gates against concurrent control, retaining S0's frozen targets."""
    assert control["manifest_sha256"] == treatment["manifest_sha256"]
    names = tuple(control["domains"])
    assert set(names) == set(treatment["domains"])
    targets = [name for name in names if baseline["target_status"][name] == "target"]
    assert targets
    rows = []
    triggers = []
    for name in names:
        assert control["cohort_hashes"][name] == treatment["cohort_hashes"][name] == baseline["cohort_hashes"][name]
        c, t = control["domains"][name], treatment["domains"][name]
        assert c["N"] == t["N"]
        ca, ta = control["agents"][name], treatment["agents"][name]
        row, boundary = override_cohort(
            {"cohort": name, "passes": t["passes"], "correctness_min_raw": c["passes"] - max(3, 0.05 * c["N"])},
            ub_cap_check=False,
        )
        if name in targets:
            triggers.extend(boundary)
        row.update(
            target=name in targets,
            N=c["N"],
            control_passes=c["passes"],
            original_correctness_pass=t["passes"] >= row["original_correctness_min_raw"],
            control_assessed=c["N"] - c["unassessed"],
            treatment_assessed=t["N"] - t["unassessed"],
            valid=c["N"] - c["unassessed"] >= VALID_FRACTION * c["N"]
            and t["N"] - t["unassessed"] >= VALID_FRACTION * t["N"],
            control_repeat_turn_share=ca["repeat_turn_share"],
            treatment_repeat_turn_share=ta["repeat_turn_share"],
            primary_cap=0.5 * ca["repeat_turn_share"],
            primary_pass=ta["repeat_turn_share"] <= 0.5 * ca["repeat_turn_share"],
            control_action_loop_trials=ca["loop_trials"],
            treatment_action_loop_trials=ta["loop_trials"],
            control_action_loop_UB=ca["loop_trials"] + c["unassessed"],
            treatment_action_loop_UB=ta["loop_trials"] + t["unassessed"],
            control_cost=ca["cost"],
            treatment_cost=ta["cost"],
            S0_repeat_turn_share=baseline["agents"][name]["repeat_turn_share"],
        )
        rows.append(row)
    total = sum(control["domains"][name]["N"] for name in names)
    control_passes = sum(control["domains"][name]["passes"] for name in names)
    passes = sum(treatment["domains"][name]["passes"] for name in names)
    original_floor = control_passes - 0.02 * total
    floor = original_floor * CORRECTNESS_MULTIPLIER
    valid = all(row["valid"] for row in rows)
    correct = passes >= floor and all(row["cohort_correctness_pass"] for row in rows if row["target"])
    primary = all(row["primary_pass"] for row in rows if row["target"])
    return {
        "registration": REGISTRATION,
        "primary_measure": "repeat_turn_share",
        "comparator": "concurrent RL control",
        "target_cohorts": targets,
        "correctness_multiplier": CORRECTNESS_MULTIPLIER,
        "cohorts": rows,
        "pooled": {
            "N": total,
            "control_passes": control_passes,
            "treatment_passes": passes,
            "original_correctness_min_raw": original_floor,
            "original_correctness_pass": passes >= original_floor,
            "correctness_min_raw": floor,
            "correctness_min_integer": math.ceil(floor),
            "correctness_pass": passes >= floor,
        },
        "validity_pass": valid,
        "correctness_pass": correct,
        "primary_pass": primary,
        "repeat_triggers": triggers,
        "repeat_required": bool(triggers),
        "repeat_policy": (
            "Target-cohort correctness within +/-2 of overridden unrounded floor; "
            "primary share has no trial-count window; secondary UB is report-only"
        ),
        "S0_comparison": "report-only",
        "verdict": "PENDING_REPEAT" if triggers else "PASS" if valid and correct and primary else "FAIL",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    paths = {name: getattr(args, name) for name in ("control", "treatment", "baseline")}
    payloads = {name: StoragePath(path).read_bytes() for name, path in paths.items()}
    result = paired_verdict(*(json.loads(payloads[name]) for name in paths))
    result["audit_inputs"] = {
        name: {"path": paths[name], "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in payloads.items()
    }
    StoragePath(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
