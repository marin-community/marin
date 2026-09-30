"""Review selected construction candidates without certifying their whole portfolio."""

import argparse
import fcntl
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .catalog import load_pilot
from .inference import GLMClient, StageStore, atomic_json, digest, parallel_map
from .prompts import SYSTEM, encode, proposal_prompt
from .schema import enum, obj, proposal_schema, response_format, strings
from .select_builds import attach_provenance, select
from .synthesis import _revoked_proposals, load_accepted
from .validation import AXES, validate_proposal, validate_review


def validate_individual(review, proposal):
    slot = proposal["slot"]
    validate_review(
        {
            "capability_id": proposal["capability_id"],
            "portfolio_verdict": "repair",
            "portfolio_issues": [],
            "missing_slots": sorted(set(range(1, 11)) - {slot}),
            "reviews": [review],
        },
        proposal["capability_id"],
        [slot],
        {slot: proposal["status"]},
    )


def review_schema(slot):
    fields = {
        "slot": {"type": "integer", "const": slot},
        "verdict": enum(["accept", "repair", "reject"]),
        "scores": obj(
            {
                axis: {"type": "integer", "minimum": 1, "maximum": 5}
                for axis in sorted(AXES)
            }
        ),
        "critical_failures": strings(),
        "issues": strings(),
        "required_changes": strings(),
    }
    return obj(fields)


def review_prompt(candidate, proposal, source_revocation=None):
    return (
        """Independently decide whether THIS ONE blueprint is ready for agentic construction.
This is construction admission, not approval of the complete ten-slot portfolio and not runtime
certification. You have the exact source capability and all recorded portfolio-level findings.
Check whether EACH portfolio finding affects this candidate. Do not inherit its prior accepting
verdict. Reject weak alignment, reward ambiguity, unsupported environment assumptions, unrealistic
workflows, source fabrication, missing independence boundaries, or a missing path to ground truth.

When a proposal translates ordinal rubric anchors into binary thresholds, compare the actual
predicates against EVERY original partial-credit level, including concrete boundary examples and
alternate valid implementations. Copying the old anchor prose next to stricter new predicates
does not preserve the rubric. The kth cumulative predicate must accept exactly the original
levels k and above. Check negative controls against their actual expected failing fields, not
their labels, and distinguish model-completion counts from complete score-vector counts.
If the source omits intermediate anchors, identify the proposed interpolation explicitly and
review it on its merits; do not claim that newly authored levels were present in the source.

Missing sibling slots and the need to repair other tasks do not block an otherwise valid candidate.
Executing an already well-specified future build or validation gate is expected construction work,
not a missing proposal change. Preserve difficult realistic work: millions of tokens and multiple
builder sessions are permitted. An unsupported premise must be repaired or rejected, never excused
as future work. Evaluate the specified experiments and abandonment conditions critically.

Score realism, alignment, specificity, reward_validity, environment_fit, diversity, source_honesty
from 1=invalid to 5=excellent. Accept only when every score is >=4, critical_failures is empty and
required_changes is empty. Any required BLUEPRINT change means repair. Informational pending build
evidence may remain in issues. Use reject for an unsalvageable premise. Return only the requested
JSON review, with the exact slot and all seven axes.

SOURCE CAPABILITY:\n"""
        + encode(candidate["provenance"]["capability_record"])
        + "\nPORTFOLIO FINDINGS:\n"
        + encode(candidate["construction_context"]["portfolio_issues"])
        + "\nCONTROLLER SOURCE REVOCATION (if present, independently verify that the "
        "current proposal substantively resolves it; an unchanged or cosmetic repair "
        "cannot clear it):\n"
        + encode(source_revocation)
        + "\nCANDIDATE:\n"
        + encode(proposal)
    )


def admit_one(
    candidate,
    store,
    repair_rounds,
    *,
    repair_rejected=False,
    require_changed_hash=False,
):
    proposal = candidate["proposal"]
    initial_hash = digest(proposal)
    key = f"{proposal['capability_id']}:{proposal['slot']}"
    history = []
    revoked = _revoked_proposals()
    for round_number in range(repair_rounds + 1):
        review = store.generate(
            "construction-review",
            key,
            SYSTEM,
            review_prompt(candidate, proposal, revoked.get(initial_hash)),
            lambda value, current=proposal: validate_individual(value, current),
            response_format=response_format(
                "construction_review", review_schema(proposal["slot"])
            ),
        )
        history.append(
            {"round": round_number, "proposal_hash": digest(proposal), "review": review}
        )
        revocation = revoked.get(digest(proposal))
        if revocation is not None:
            history[-1]["controller_block"] = {
                "reason_code": "revoked_proposal_requires_changed_hash",
                "revocation": revocation,
            }
        needs_changed_repair = (
            require_changed_hash
            and digest(proposal) == initial_hash
        )
        if needs_changed_repair and revocation is None:
            history[-1]["controller_block"] = {
                "reason_code": "clarification_requires_changed_hash",
                "source_proposal_hash": initial_hash,
            }
        if (
            review["verdict"] == "accept"
            and revocation is None
            and not needs_changed_repair
        ):
            return {
                "state": "accepted",
                "item": {
                    **candidate,
                    "proposal": proposal,
                    "proposal_hash": digest(proposal),
                    "review": review,
                    "admission": {
                        "state": "accepted",
                        "scope": "individual_construction",
                        "source_proposal_hash": initial_hash,
                        "portfolio_certified": False,
                        "runtime_certified": False,
                        "history": history,
                    },
                },
            }
        rejected_repair_allowed = (
            review["verdict"] == "reject"
            and (repair_rejected or needs_changed_repair)
            and round_number < repair_rounds
        )
        if (
            review["verdict"] == "reject" and not rejected_repair_allowed
        ) or round_number == repair_rounds:
            result = {"state": "rejected", "proposal": proposal, "history": history}
            if revocation is not None:
                result["controller_issue"] = (
                    "Revoked proposal cannot be readmitted unchanged: "
                    + revocation["reason"]
                )
            elif require_changed_hash and digest(proposal) == initial_hash:
                result["controller_issue"] = (
                    "Clarification admission requires a new proposal hash and a "
                    "fresh accepting review"
                )
            return result
        plan = candidate["construction_context"]["plan"]
        slot = next(
            value for value in plan["slots"] if value["slot"] == proposal["slot"]
        )
        previous_hash = digest(proposal)
        proposal = store.generate(
            "construction-repair",
            key,
            SYSTEM,
            proposal_prompt(
                candidate["provenance"]["capability_record"],
                plan,
                slot,
                {
                    "review": review,
                    "previous_proposal": proposal,
                    "portfolio_issues": candidate["construction_context"][
                        "portfolio_issues"
                    ],
                    "scope": "Repair this candidate; no claim of whole-portfolio acceptance.",
                    "controller_constraints": (
                        {
                            "revocation": revocation,
                            "required_action": (
                                "Resolve the recorded source defect through a substantive "
                                "proposal repair and a new hash, then undergo fresh review. "
                                "An accepting model review cannot clear this revocation. "
                                "A cosmetic edit does not resolve the recorded defect."
                            ),
                        }
                        if revocation is not None
                        else {
                            "required_action": (
                                "Resolve the recorded clarification requirements through "
                                "a substantive proposal change and a new hash, then "
                                "undergo fresh independent review. An unchanged accepting "
                                "review cannot satisfy this requirement; a cosmetic edit "
                                "does not resolve the clarification."
                            ),
                            "source_proposal_hash": initial_hash,
                        }
                        if require_changed_hash
                        else None
                    ),
                },
            ),
            lambda value, current=proposal: validate_proposal(
                value, current["capability_id"], current["slot"]
            ),
            response_format=response_format(
                "construction_proposal",
                proposal_schema(proposal["capability_id"], proposal["slot"]),
            ),
        )
        if proposal["status"] == "null":
            return {"state": "null", "proposal": proposal, "history": history}
        if (
            (rejected_repair_allowed or require_changed_hash)
            and digest(proposal) == previous_hash
        ):
            return {
                "state": "rejected",
                "proposal": proposal,
                "history": history,
                "controller_issue": (
                    "Required proposal repair did not produce a new proposal hash"
                ),
            }
    raise AssertionError("unreachable")


def admit(args):
    candidates = load_accepted(
        Path(args.candidates),
        allow_pending_admission=True,
        allow_revoked=True,
    )
    for candidate in candidates:
        if not candidate.get("construction_context", {}).get("source_files"):
            raise ValueError(
                "construction candidates require hash-bound portfolio context"
            )
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        atomic_json(root / "input_candidates.json", candidates)
        atomic_json(
            root / "run.json",
            {
                "stage": "construction_admission",
                "input_hash": digest(candidates),
                "tier": args.tier,
                "concurrency": args.concurrency,
                "repair_rounds": args.repair_rounds,
                "repair_rejected": args.repair_rejected,
                "require_changed_hash": args.require_changed_hash,
                "started": started,
            },
        )
        atomic_json(
            root / "report.json",
            {"stage": "construction_admission", "state": "running"},
        )
        store = StageStore(root, GLMClient(tier=args.tier))
        results, failures = parallel_map(
            [
                (
                    f"{c['proposal']['capability_id']}:{c['proposal']['slot']}",
                    lambda candidate=c: admit_one(
                        candidate,
                        store,
                        args.repair_rounds,
                        repair_rejected=args.repair_rejected,
                        require_changed_hash=args.require_changed_hash,
                    ),
                )
                for c in candidates
            ],
            args.concurrency,
        )
        ordered = [results[key] for key in sorted(results)]
        accepted = [value["item"] for value in ordered if value["state"] == "accepted"]
        rejected = [value for value in ordered if value["state"] == "rejected"]
        nulls = [value for value in ordered if value["state"] == "null"]
        for filename, values in (
            ("accepted.json", accepted),
            ("rejected.json", rejected),
            ("null.json", nulls),
        ):
            atomic_json(root / filename, values)
        report = {
            "stage": "construction_admission",
            "state": "complete" if not failures and not rejected else "needs_iteration",
            "requested": len(candidates),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "null": len(nulls),
            "failures": failures,
            "runtime_validated_tasks": 0,
            "portfolio_certified": False,
            "repair_rejected": args.repair_rejected,
            "require_changed_hash": args.require_changed_hash,
            "environment_counts": dict(
                Counter(c["proposal"]["environment"] for c in accepted)
            ),
            "verifier_counts": dict(
                Counter(c["proposal"]["verification"] for c in accepted)
            ),
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["state"] == "complete" else 2


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "admit", help="Independently admit selected tasks for construction"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tier", choices=("interactive", "bulk"), default="interactive"
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument(
        "--repair-rejected",
        action="store_true",
        help="Allow a bounded repair round after an initial reject verdict",
    )
    parser.add_argument(
        "--require-changed-hash",
        action="store_true",
        help="Require a repair with a new proposal hash before admission can accept",
    )
    parser.set_defaults(func=admit)


def prepare(run, output, count=9, review_round=0):
    run = Path(run)
    pilot = load_pilot(run / "input_pilot.json")
    proposals = json.loads((run / "proposals.json").read_text())
    reviews = json.loads((run / f"reviews-round-{review_round}.json").read_text())
    plans = json.loads((run / "plans.json").read_text())
    source_files = {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for name in (
            "input_pilot.json",
            "proposals.json",
            "plans.json",
            f"reviews-round-{review_round}.json",
        )
    }
    candidates = []
    for proposal in proposals:
        if proposal["status"] != "proposed":
            continue
        cid, slot = proposal["capability_id"], proposal["slot"]
        portfolio = reviews.get(cid)
        review = (
            next(
                (value for value in portfolio["reviews"] if value["slot"] == slot), None
            )
            if portfolio
            else None
        )
        if not review or review["verdict"] != "accept":
            continue
        validate_proposal(proposal, cid, slot)
        validate_individual(review, proposal)
        candidates.append(
            {
                "proposal": proposal,
                "proposal_hash": digest(proposal),
                "review": review,
                "admission": {"state": "pending", "scope": "individual_construction"},
                "construction_context": {
                    "portfolio_verdict": portfolio["portfolio_verdict"],
                    "portfolio_issues": portfolio["portfolio_issues"],
                    "plan": plans[cid],
                    "source_files": source_files,
                },
            }
        )
    chosen, report = select(candidates, count)
    chosen = attach_provenance(chosen, pilot)
    report.update(
        output_hash=digest(chosen),
        scope="candidates only; independent construction admission still required",
    )
    atomic_json(output, chosen)
    atomic_json(Path(output).with_suffix(".selection.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare candidates for a separate construction-admission run"
    )
    parser.add_argument("run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--review-round", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.run, args.out, args.count, args.review_round), indent=2)
    )
