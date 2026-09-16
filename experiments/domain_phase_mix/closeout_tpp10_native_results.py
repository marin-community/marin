# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Finalize native survey records and corrected BPB without training or evaluation."""

import argparse
import json
import logging
from pathlib import Path

from iris.client.client import get_iris_ctx
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState

from experiments.domain_phase_mix import complete_tpp10_finemath_targets as completion
from experiments.domain_phase_mix import repair_tpp10_evaluation as repair
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

DIRECTORY = Path(".agents/projects/starcoder_tpp10/domain_sweeps/native_closeout_20260912")
REPAIR_PLAN = DIRECTORY.parent / "repairs_20260911/plan.json"
COMPLETION_PLAN = DIRECTORY.parent / "target_completion_20260912/plan.json"
COMPLETION_PARENT = "/calvinxu/tpp10-finemath-target-completion"


def resolved_sources(plan: dict, continuation: dict) -> tuple[dict, list[dict]]:
    """Resolve four resumed children while preserving all frozen scientific identities."""
    repair.validate_plan(plan)
    completion.validate_plan(continuation)
    if continuation["training_plan"] != plan["artifacts"]["plan"]:
        raise ValueError("Repair and completion refer to different training plans")
    if continuation["source_parent"] != plan["source_coordinator"]:
        raise ValueError("Completion has a different original parent")
    selected = {item["request"]["run_name"]: item["request"] for item in continuation["selected"]}
    rows = []
    resolutions = []
    for row in plan["artifacts"]["rows"]:
        request = row["request"]
        name = request["run_name"]
        if name not in selected:
            rows.append(row)
            continue
        if selected[name] != request:
            raise ValueError(f"Resumed request changed: {name}")
        new_job = f"{COMPLETION_PARENT}/{name}"
        rows.append(row | {"iris_job": new_job})
        resolutions.append({"run_name": name, "original_job": row["iris_job"], "completed_job": new_job})
    if len(resolutions) != 4 or len(rows) != 28:
        raise ValueError("Closeout requires exactly 28 points and four resumed sources")
    # This is an operational job lookup, never a replacement for the immutable repair plan.
    resolved = plan | {"artifacts": plan["artifacts"] | {"rows": rows}}
    return resolved, resolutions


def closeout(plan: dict, continuation: dict) -> dict:
    repair.experiment.require_central1()
    resolved, resolutions = resolved_sources(plan, continuation)
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("Record closeout requires the regional Iris coordinator")
    client = context.client
    trees = {}
    for parent in (plan["source_coordinator"], COMPLETION_PARENT):
        tree = client.list_jobs(prefix=parent, limit=None)
        if not tree or any(job.state not in TERMINAL_JOB_STATES for job in tree):
            raise ValueError(f"Source tree is not wholly terminal: {parent}")
        trees[parent] = {job.job_id.to_wire(): str(job.state) for job in tree}
    if client.job_state(JobName.from_wire(COMPLETION_PARENT)) != JobState.SUCCEEDED:
        raise ValueError("FineMath completion parent did not succeed")
    for item in continuation["selected"]:
        request = item["request"]
        receipt = repair.read_json(request["output_path"] + "/target_completion_resume.json")
        if (
            receipt["recovery_sha256"] != continuation["recovery_sha256"]
            or receipt["training_fingerprint"] != request["fingerprint"]
            or receipt["load_checkpoint"] is not True
        ):
            raise ValueError(f"Invalid continuation receipt: {request['run_name']}")
    counts = repair.read_json(repair.root(plan) + "/population_counts.json")
    for audit in plan["audits"]:
        if not repair.audit_verified(plan, counts, audit):
            raise ValueError(f"Existing numerical audit no longer verifies: {audit['request']['run_name']}")
    receipt = {
        "repair_sha256": plan["repair_sha256"],
        "completion_sha256": continuation["recovery_sha256"],
        "wrapper_sha256": file_sha256(Path(__file__)),
        "source_resolutions": resolutions,
        "terminal_trees": trees,
        "counts_sha256": canonical_sha256(counts),
        "reused_audits": [audit["request"]["run_name"] for audit in plan["audits"]],
    }
    receipt["closeout_sha256"] = canonical_sha256(receipt)
    receipt_root = repair.root(plan) + "/record_closeouts/" + receipt["closeout_sha256"]
    repair.write_json(receipt_root + "/source_resolution.json", receipt)
    statuses = repair.recover_records(resolved, client)
    if any(state["state"] != "verified" for state in statuses.values()):
        raise ValueError(f"Native completion records remain unresolved: {statuses}")
    result = repair.collect_corrected(plan, counts, statuses)
    if not result["complete"] or len(result["rows"]) != 28:
        raise ValueError("Corrected survey is incomplete")
    publication = receipt | {
        "result_sha256": canonical_sha256(result),
        "result_snapshot": repair.root(plan) + "/result_snapshots/" + canonical_sha256(result) + ".json",
        "rows": len(result["rows"]),
        "complete": result["complete"],
    }
    repair.write_json(receipt_root + "/publication.json", publication)
    return publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-plan", type=Path, default=REPAIR_PLAN)
    parser.add_argument("--completion-plan", type=Path, default=COMPLETION_PLAN)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    plan = json.loads(args.repair_plan.read_text())
    continuation = json.loads(args.completion_plan.read_text())
    if args.check_only:
        _, resolutions = resolved_sources(plan, continuation)
        print(json.dumps({"verified_source_resolutions": resolutions}))
    else:
        print(json.dumps(closeout(plan, continuation)))


if __name__ == "__main__":
    main()
