# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The three binding user overrides: deadline, correctness thresholds, checkpoint conservation.

Promoted from the ad-hoc ``apply-user-overrides.py`` and ``submit-user-overrides.py``
worker scripts. ``apply-user-overrides`` recomputes both S1 arms' gate receipts under the
registered 0.75 correctness multiplier and refuses to issue receipts when a repeat window
trigger fires; ``publish-user-overrides`` publishes the receipts to the experiment root,
marks the conserved initializer, and inventories it.

    python -m experiments.post_training.mochi_loop_guards apply-user-overrides \\
        --stop-receipt <s1-stop full-gate receipt> --rp-receipt <s1-rp full-gate receipt> \\
        --output-dir <decisions dir>

    … cpu-submit --cpu 8 --memory 32GB --disk 64GB -- \\
      python -m experiments.post_training.mochi_loop_guards publish-user-overrides \\
      --receipts-dir <dir> --hf-inventory <json> --root s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards
"""

import argparse
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

CORRECTNESS_MULTIPLIER = 0.75
POOLED_CORRECTNESS_MIN_RAW = 222.54
BOUNDARY_REPEAT_WINDOW = 2
CLOCK_OVERRIDE = {
    "kind": "user override",
    "T0": "2026-11-25T12:01:36-05:00",
    "extension_hours": 24,
    "T20": "2026-11-27T08:01:36-05:00",
    "T32": "2026-11-27T20:01:36-05:00",
    "deadline": "2026-11-28T12:01:36-05:00",
    "decision": "I3 restored; supersedes original T20 drop; original receipt retained",
    "supersedes": (
        "s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/decisions/i3-t20-budget-drop-2026.11.26.json"
    ),
}
CONSERVATION_OVERRIDE = {
    "kind": "user override",
    "permanent": True,
    "exclude_from_every_pruning_sweep": True,
    "source": "s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
    "HF": "open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407",
    "HF_private": False,
    "private_conversion_blocker": (
        "HF403: Private storage limit reached for org open-athena; org plan/capacity must be resolved"
    ),
    "new_S1_weights": False,
    "new_S2_weights_policy": (
        "Export both arms to PRIVATE open-athena repos with OpenMDW1.1 LICENSE, required model-card warnings and "
        "generation config via in-region CPU job; conserve regional exports permanently"
    ),
    "hash_verification": "pending regional verification",
}
CLOCK_RECEIPT_NAME = "clock-extension-user-override.json"
GATE_RECEIPT_NAME = "correctness-gate-user-override.json"
CONSERVATION_RECEIPT_NAME = "checkpoint-conservation-user-override.json"
RP_REPEAT_DECISION = (
    "No preregistered trigger; cancel exact repeat; originals remain authoritative; "
    "repeat excluded from scientific verdict"
)
FAILURE_REASONS = {
    "stop": ["TBLite repeat-turn share exceeds unchanged I1 cap"],
    "rp": ["All three action UB counts exceed unchanged I2 caps"],
}


def override_cohort(row: dict, *, ub_cap_check: bool) -> tuple[dict, list[str]]:
    """Recompute one cohort's correctness gate under the registered override.

    Returns the overridden row and any repeat-window triggers it fires. Thresholds stay
    unrounded through the boundary-repeat window.
    """
    triggers = []
    raw = row["correctness_min_raw"] * CORRECTNESS_MULTIPLIER
    row["original_correctness_min_raw"] = row["correctness_min_raw"]
    row["correctness_min_raw"] = raw
    row["correctness_min_integer"] = math.ceil(raw)
    row["cohort_correctness_pass"] = row["passes"] >= raw
    row["correctness_boundary_repeat_trigger"] = abs(row["passes"] - raw) <= BOUNDARY_REPEAT_WINDOW
    if row["correctness_boundary_repeat_trigger"]:
        triggers.append(row["cohort"] + " correctness")
    if ub_cap_check and abs(row["action_loop_UB_count"] - row["primary_UB_cap_raw"]) <= BOUNDARY_REPEAT_WINDOW:
        triggers.append(row["cohort"] + " UB")
    return row, triggers


def override_gate_receipt(stop_receipt: Mapping, rp_receipt: Mapping, *, recorded_at: str) -> dict:
    """Recompute both S1 arms' correctness gates under the registered override."""
    results = []
    for arm, receipt in (("stop", stop_receipt), ("rp", rp_receipt)):
        overridden = json.loads(json.dumps(receipt))
        triggers = []
        for row in overridden["cohorts"]:
            row, fired = override_cohort(row, ub_cap_check=arm == "rp")
            triggers.extend(fired)
        pooled_raw = POOLED_CORRECTNESS_MIN_RAW * CORRECTNESS_MULTIPLIER
        overridden["pooled_correctness_min_raw"] = pooled_raw
        overridden["pooled_correctness_min_integer"] = math.ceil(pooled_raw)
        overridden["pooled_correctness_pass"] = overridden["passes"] >= pooled_raw
        overridden["repeat_triggered"] = bool(triggers)
        overridden["triggers"] = triggers
        overridden["final_guard_verdict"] = "FAIL"
        overridden["failure_reasons"] = list(FAILURE_REASONS[arm])
        results.append({"arm": arm, "original": receipt, "under_user_override": overridden})
    return {
        "kind": "user override",
        "recorded_at": recorded_at,
        "correctness_multiplier": CORRECTNESS_MULTIPLIER,
        "loop_gates_unchanged": True,
        "repeat_trigger_uses_unrounded_thresholds": True,
        "results": results,
        "RP_repeat_decision": RP_REPEAT_DECISION,
    }


def override_receipts(stop_receipt: Mapping, rp_receipt: Mapping, *, now: datetime) -> dict[str, dict]:
    """Build the three binding override receipts; refuse when a repeat window triggers."""
    recorded_at = now.isoformat()
    clock = {**CLOCK_OVERRIDE, "kind": "user override", "recorded_at": recorded_at}
    gate = override_gate_receipt(stop_receipt, rp_receipt, recorded_at=recorded_at)
    if any(result["under_user_override"]["repeat_triggered"] for result in gate["results"]):
        raise ValueError("A repeat window fires under the override; no receipts are issued")
    conservation = {**CONSERVATION_OVERRIDE, "kind": "user override", "recorded_at": recorded_at}
    return {CLOCK_RECEIPT_NAME: clock, GATE_RECEIPT_NAME: gate, CONSERVATION_RECEIPT_NAME: conservation}


def apply_receipts(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-receipt", required=True, help="s1-stop full-gate receipt JSON")
    parser.add_argument("--rp-receipt", required=True, help="s1-rp full-gate receipt JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for the three override receipts")
    args = parser.parse_args(argv)
    receipts = override_receipts(
        json.loads(Path(args.stop_receipt).read_text()),
        json.loads(Path(args.rp_receipt).read_text()),
        now=datetime.now(UTC),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, receipt in receipts.items():
        (output / name).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipts": sorted(receipts),
                "raw_correctness": [
                    40.65 * CORRECTNESS_MULTIPLIER,
                    56 * CORRECTNESS_MULTIPLIER,
                    104.2 * CORRECTNESS_MULTIPLIER,
                    POOLED_CORRECTNESS_MIN_RAW * CORRECTNESS_MULTIPLIER,
                ],
                "stop": "FAIL loop gate only",
                "rp": "FAIL loop gates only",
                "repeat_required": False,
            },
            indent=2,
        )
    )


def publish_receipts(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts-dir", required=True, help="Directory of override receipt JSON files")
    parser.add_argument(
        "--hf-inventory", required=True, help="HF repository inventory JSON for the conserved checkpoint"
    )
    parser.add_argument("--root", required=True, help="Regional experiment root to publish into")
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    payload = {path.name: json.loads(path.read_text()) for path in sorted(Path(args.receipts_dir).glob("*.json"))}
    hf = json.loads(Path(args.hf_inventory).read_text())
    root = StoragePath(args.root)
    for name, receipt in payload.items():
        (root / "decisions" / name).write_text(json.dumps(receipt, indent=2) + "\n")
    source = StoragePath(payload[CONSERVATION_RECEIPT_NAME]["source"])
    (source / ".conservation.json").write_text(json.dumps(payload[CONSERVATION_RECEIPT_NAME], indent=2) + "\n")
    (root / "conservation" / "sft-400407-HF-inventory.json").write_text(json.dumps(hf, indent=2) + "\n")
    entries = [{"name": file.name, "size": file.size()} for file in (source / "*").glob() if file.isfile()]
    (root / "conservation" / "sft-400407-regional-inventory.json").write_text(
        json.dumps(entries, indent=2, default=str) + "\n"
    )
    for entry in entries:
        if "manifest" in entry["name"].lower() or "sha" in entry["name"].lower():
            if entry["size"] < 10_000_000:
                text = (source / entry["name"]).read_text()
                (root / "conservation" / entry["name"]).write_text(text)
                print(entry["name"], text[:10000], flush=True)
    print(
        json.dumps(
            {
                "receipts_published": sorted(payload),
                "permanent_marker": str(source / ".conservation.json"),
                "regional_inventory_count": len(entries),
                "inventory": str(root / "conservation" / "sft-400407-regional-inventory.json"),
            },
            flush=True,
        )
    )


def main(argv: list[str] | None = None) -> None:
    handlers = {
        "apply-user-overrides": apply_receipts,
        "publish-user-overrides": publish_receipts,
    }
    handler = handlers.get((argv or [None])[0])
    if handler is None:
        raise SystemExit(f"expected one of {sorted(handlers)} as the first argument")
    handler(argv[1:])


if __name__ == "__main__":
    main()
