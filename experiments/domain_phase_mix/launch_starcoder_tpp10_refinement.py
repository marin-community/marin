# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Release the five approved TPP10 fractions using the frozen dense-grid recipes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from experiments.domain_phase_mix import launch_starcoder_tpp10 as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

PERCENTS = (40, 55, 60, 65, 80)
DENSE_PLAN_SHA256 = "3a20c5e45ae73f31d066216adbd5dbd63af427338aecc0c3999a5215135c2ba5"
PILOT_PLAN_SHA256 = "afa613b25468dd3dadff230544361d815f8524370bd33c14e6c58d3717496ba0"


def build_refinement_plan(design: dict) -> tuple[dict, tuple]:
    """Select only approved coordinates without changing any training identity."""
    dense, steps = launcher.build_plan(design, "dense")
    if dense["plan_sha256"] != DENSE_PLAN_SHA256:
        raise ValueError("Dense recipes differ from the frozen experiment")
    selected = [(row, step) for row, step in zip(dense["runs"], steps, strict=True) if row["percent"] in PERCENTS]
    if len(selected) != 45:
        raise ValueError("Expected five targets, ten unmatched proxies, and thirty matched proxies")
    plan = {
        **{key: value for key, value in dense.items() if key != "plan_sha256"},
        "stage": "refinement",
        "dense_plan_sha256": DENSE_PLAN_SHA256,
        "pilot_plan_sha256": PILOT_PLAN_SHA256,
        "selection_launcher_sha256": file_sha256(Path(__file__)),
        "percents": list(PERCENTS),
        "runs": [row for row, _ in selected],
        "training_flops": sum(
            design["models"]["target" if row["arm"] == experiment.Arm.TARGET else "unmatched"]["training_flops"]
            for row, _ in selected
        ),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan, tuple(step for _, step in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--max-concurrent", type=int, default=45)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect-results", type=Path)
    args = parser.parse_args()
    if args.collect_results:
        launcher.collect_results(json.loads(args.plan_path.read_text()), args.collect_results)
        return
    if args.max_concurrent != 45:
        raise ValueError("Release all 45 selected jobs to the scheduler")
    design = experiment.load_design()
    plan, steps = build_refinement_plan(design)
    if args.plan_path.exists() and json.loads(args.plan_path.read_text()) != plan:
        raise ValueError("Existing refinement plan differs; preserve the archived plan")
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: plan[key] for key in ("plan_sha256", "percents", "training_flops")}))
    if not args.submit:
        return
    if args.release is None:
        raise ValueError("--submit requires the recorded release")
    experiment.require_central1()
    cache_audit = preparation.verify_caches(design, preparation.data_steps(design), experiment.PREFIX)
    launcher.validate_release(plan, json.loads(args.release.read_text()), cache_audit)
    asyncio.run(experiment.audit_allocations(design))
    pilot, _ = launcher.build_plan(design, "pilot")
    if pilot["plan_sha256"] != PILOT_PLAN_SHA256:
        raise ValueError("Pilot recipes differ from the frozen experiment")
    launcher.collect_results(pilot, args.plan_path.parent / "refinement_pilot_gate_metrics.csv")
    uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/{plan['plan_sha256']}/plan.json"
    launcher.persist_submission_plan(plan, uri)
    pending = launcher.pending_training_steps(steps, marin_prefix=experiment.PREFIX)
    print(json.dumps({"selected": len(steps), "pending": len(pending), "plan_uri": uri}))
    if pending:
        launcher.run(*pending, max_concurrent=args.max_concurrent, force_run_failed=True)
    launcher.collect_results(plan, args.plan_path.parent / "refinement_metrics.csv")


if __name__ == "__main__":
    main()
