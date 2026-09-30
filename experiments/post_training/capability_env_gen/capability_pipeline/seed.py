"""Import frozen proposal evidence for a fresh review/repair experiment."""

import copy
import hashlib
import json
from pathlib import Path

from .inference import digest
from .validation import validate_plan, validate_proposal


def load_seed(root, pilot, capabilities):
    root = Path(root)
    documents, hashes = {}, {}
    for name in ("input_pilot.json", "plans.json", "proposals.json", "report.json"):
        raw = (root / name).read_bytes()
        documents[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    if digest(documents["input_pilot.json"]) != digest(pilot):
        raise ValueError("seed pilot differs from the frozen requested pilot")
    report = documents["report.json"]
    if report.get("stage") != "proposal_review" or report.get("state") not in {
        "complete",
        "needs_iteration",
    }:
        raise ValueError("seed requires a terminal proposal report")
    ids = {cap["capability_id"] for cap in capabilities}
    plans = documents["plans.json"]
    if not isinstance(plans, dict) or not ids <= plans.keys():
        raise ValueError("seed is missing selected capability plans")
    plans = {cid: plans[cid] for cid in sorted(ids)}
    for cid, plan in plans.items():
        # Legacy exclusion rationale is precisely one target of this experiment.
        # Validate every other field now; the original exclusion list remains in
        # reviewer input and must pass the current validator before acceptance.
        shape = copy.deepcopy(plan)
        shape["excluded_combinations"] = []
        validate_plan(shape, cid)
    proposals = {}
    source = documents["proposals.json"]
    if not isinstance(source, list):
        raise TypeError("seed proposals must be a list")
    for proposal in source:
        cid, slot = proposal.get("capability_id"), proposal.get("slot")
        if cid not in ids:
            continue
        validate_proposal(proposal, cid, slot)
        key = f"{cid}:{slot}"
        if key in proposals:
            raise ValueError("seed contains duplicate proposal identity")
        proposals[key] = proposal
    failure_history = copy.deepcopy(report.get("inherited_failures", []))
    if not isinstance(failure_history, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("failures"), dict)
        for entry in failure_history
    ):
        raise TypeError("seed inherited failure history must contain report records")
    current_failures = report.get("failures", {})
    if not isinstance(current_failures, dict):
        raise TypeError("seed failures must be a stage mapping")
    if any(current_failures.values()):
        failure_history.append(
            {
                "source_report_sha256": hashes["report.json"],
                "failures": copy.deepcopy(current_failures),
            }
        )
    prior_proposal_failures = {}
    for entry in failure_history:
        failures = entry["failures"].get("proposal", {})
        if not isinstance(failures, dict):
            raise TypeError("seed proposal failures must be an identity mapping")
        prior_proposal_failures.update(failures)
    return {
        "plans": plans,
        "proposals": proposals,
        "prior_failures": prior_proposal_failures,
        "failure_history": failure_history,
        "provenance": {
            "source_files": hashes,
            "pilot_hash": digest(pilot),
            "inherited_plans": len(plans),
            "inherited_proposals": len(proposals),
            "acceptance_inherited": False,
        },
    }
