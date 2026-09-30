"""Controller CLI. Inference and generated-code execution belong on cluster workers."""

import argparse
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

from .catalog import load_pilot
from .inference import GLMClient, StageStore, atomic_json, digest, parallel_map
from .prompts import (
    SYSTEM,
    capability_prompt_record,
    plan_prompt,
    proposal_prompt,
    review_prompt,
)
from .schema import plan_schema, proposal_schema, response_format, review_schema
from .seed import load_seed
from .validation import (
    PARTIAL_ADMISSION_POLICY,
    validate_plan,
    validate_proposal,
    validate_review,
)


def _ordered_proposals(proposals):
    return sorted(
        proposals.values(),
        key=lambda proposal: (proposal["capability_id"], proposal["slot"]),
    )


def _stage_response_format(args, name, schema):
    if getattr(args, "structured_output", "json_schema") == "off":
        return None
    return response_format(name, schema)


def _source_provenance(pilot, capability):
    result = {
        "catalog_source": pilot["source"],
        "capability_record": capability,
        "capability_record_hash": digest(capability),
    }
    context = capability_prompt_record(capability, pilot).get("learning_progression")
    if context is not None:
        result["learning_progression"] = context
        result["learning_progression_hash"] = digest(context)
    return result


def _locked_slot_signature(plan, proposals):
    """Describe slots a plan repair may explain but must not replace."""

    proposal_by_slot = {proposal["slot"]: proposal for proposal in proposals}
    signature = {}
    for slot in plan["slots"]:
        number = slot["slot"]
        proposal = proposal_by_slot.get(number)
        if proposal is None:
            source = slot
            status = source["status"]
        else:
            source = proposal
            status = "propose" if source["status"] == "proposed" else "null"
        signature[number] = {
            "status": status,
            "environment": source.get("environment") if status == "propose" else None,
            "verification": source.get("verification") if status == "propose" else None,
        }
    return signature


def _validate_plan_repair(plan, capability_id, locked_signature):
    validate_plan(plan, capability_id)
    if _locked_slot_signature(plan, []) != locked_signature:
        raise ValueError("plan repair changed locked current slot assignments")


def _plan_repair_prompt(capability, plan, proposals, review, locked_signature):
    return (
        plan_prompt(capability)
        + "\nREPAIR THE EXISTING PLAN using the independent review below. Correct invalid "
        "environment/verifier reasoning and portfolio rationale, but do not replace or "
        "renumber current proposal slots. The locked signature is mandatory. Return the "
        "complete plan JSON only.\nLOCKED_SLOT_SIGNATURE:\n"
        + json.dumps(locked_signature, ensure_ascii=False, sort_keys=True)
        + "\nCURRENT_PLAN:\n"
        + json.dumps(plan, ensure_ascii=False, sort_keys=True)
        + "\nCURRENT_PROPOSALS:\n"
        + json.dumps(proposals, ensure_ascii=False, sort_keys=True)
        + "\nREVIEW_FEEDBACK:\n"
        + json.dumps(review, ensure_ascii=False, sort_keys=True)
    )


def _validate_portfolio_review(review, cid, slots, statuses, plan):
    validate_review(review, cid, slots, statuses)
    if review["portfolio_verdict"] == "accept":
        validate_plan(plan, cid)


def propose(args):
    pilot = load_pilot(args.pilot)
    capabilities = pilot["capabilities"]
    ids = [c["capability_id"] for c in capabilities]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("pilot must contain unique capability IDs")
    for c in capabilities:
        if c["capability"]["id"] != c["capability_id"]:
            raise ValueError("pilot source ID mismatch")
    if args.limit:
        capabilities = capabilities[: args.limit]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    # A lock is held by this live process, never interpreted as proof of liveness.
    with (root / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _propose(args, pilot, capabilities, root)


def _propose(args, pilot, capabilities, root):
    prompt_records = {
        cap["capability_id"]: capability_prompt_record(cap, pilot)
        for cap in capabilities
    }
    seed_path = getattr(args, "seed_run", None)
    if seed_path and Path(seed_path).resolve() == root.resolve():
        raise ValueError("seed refinement requires a distinct output directory")
    seed = load_seed(seed_path, pilot, capabilities) if seed_path else None
    client = GLMClient(tier=args.tier, hold_seconds=args.hold_seconds)
    store = StageStore(root, client)
    started = time.time()
    # These files are the public handoff. Clear a restored prior run before any
    # resumed cache work so a crash cannot republish stale accepted proposals.
    atomic_json(root / "accepted.json", [])
    atomic_json(root / "rejected.json", [])
    atomic_json(root / "null.json", [])
    atomic_json(root / "plans.json", {})
    atomic_json(root / "proposals.json", [])
    atomic_json(
        root / "report.json",
        {"stage": "proposal_review", "state": "running", "runtime_validated_tasks": 0},
    )
    atomic_json(root / "input_pilot.json", pilot)
    atomic_json(
        root / "run.json",
        {
            "stage": "propose",
            "seed": seed["provenance"] if seed else None,
            "model": "glm-5.3",
            "tier": args.tier,
            "concurrency": args.concurrency,
            "capability_ids": [c["capability_id"] for c in capabilities],
            "pilot_hash": digest(pilot),
            "started": started,
            "repair_rounds": args.repair_rounds,
            "structured_output": getattr(args, "structured_output", "json_schema"),
            "catalog_status": pilot.get("catalog_audit", {}).get("status", "unknown"),
        },
    )
    if seed:
        plans = seed["plans"]
        all_proposals = seed["proposals"]
        plan_failures = {}
        proposal_failures = {}
        atomic_json(root / "seed-provenance.json", seed["provenance"])
        atomic_json(root / "plans.json", dict(sorted(plans.items())))
        atomic_json(root / "proposals.json", _ordered_proposals(all_proposals))
    else:
        jobs = []
        for cap in capabilities:
            cid = cap["capability_id"]
            jobs.append(
                (
                    cid,
                    lambda c=cap, i=cid: store.generate(
                        "plan",
                        i,
                        SYSTEM,
                        plan_prompt(prompt_records[i]),
                        lambda obj: validate_plan(obj, i),
                        max_tokens=24000,
                        response_format=_stage_response_format(
                            args, "capability_plan", plan_schema(i)
                        ),
                    ),
                )
            )
        plans, plan_failures = parallel_map(jobs, args.concurrency)
        atomic_json(root / "plans.json", dict(sorted(plans.items())))
        jobs = []
        all_proposals = {}
        for cap in capabilities:
            cid = cap["capability_id"]
            if cid not in plans:
                continue
            plan = plans[cid]
            for slot in plan["slots"]:
                key = f"{cid}:{slot['slot']}"
                if slot["status"] == "null":
                    all_proposals[key] = {
                        "capability_id": cid,
                        "slot": slot["slot"],
                        "status": "null",
                        "null_reason": slot["reason"],
                    }
                    continue
                jobs.append(
                    (
                        key,
                        lambda c=cap, p=plan, s=slot, i=cid, k=key: store.generate(
                            "proposal",
                            k,
                            SYSTEM,
                            proposal_prompt(prompt_records[i], p, s),
                            lambda obj: validate_proposal(obj, i, s["slot"]),
                            max_tokens=32000,
                            response_format=_stage_response_format(
                                args,
                                "capability_proposal",
                                proposal_schema(i, s["slot"]),
                            ),
                        ),
                    )
                )
        generated, proposal_failures = parallel_map(jobs, args.concurrency)
        all_proposals.update(generated)
        atomic_json(root / "proposals.json", _ordered_proposals(all_proposals))
    reviews, review_failures = {}, {}
    review_failure_history = {}
    repair_failures = {}
    plan_repair_failures = {}
    unresolved_repairs = {}
    unresolved_plan_repairs = {}
    for round_number in range(args.repair_rounds + 1):
        jobs = []
        for cap in capabilities:
            cid = cap["capability_id"]
            if cid not in plans:
                continue
            group = sorted(
                [v for v in all_proposals.values() if v["capability_id"] == cid],
                key=lambda x: x["slot"],
            )
            if not group:
                continue
            plan_issue = None
            try:
                validate_plan(plans[cid], cid)
            except ValueError as error:
                plan_issue = str(error)
            prompt = review_prompt(prompt_records[cid], plans[cid], group)
            if plan_issue:
                prompt += (
                    "\nDETERMINISTIC PLAN VALIDATION FAILURE: "
                    + plan_issue
                    + "\nThe portfolio cannot be accepted until the plan is repaired. "
                    "Record this defect in portfolio_issues. Do not downgrade valid sibling tasks."
                )
            jobs.append(
                (
                    cid,
                    lambda i=cid, g=group, p=prompt, current_plan=plans[cid]: (
                        store.generate(
                            "review",
                            i,
                            SYSTEM,
                            p,
                            lambda obj: _validate_portfolio_review(
                                obj,
                                i,
                                [p["slot"] for p in g],
                                {p["slot"]: p["status"] for p in g},
                                current_plan,
                            ),
                            # Two of three v1 pilot portfolios exhausted 32K
                            # entirely in reasoning; their full inputs were
                            # 73–78K tokens. Give review enough output room
                            # without shortening its admitted proposals.
                            max_tokens=64000,
                            response_format=_stage_response_format(
                                args,
                                "capability_review",
                                review_schema(i, [p["slot"] for p in g]),
                            ),
                        )
                    ),
                )
            )
        reviews, review_failures = parallel_map(jobs, args.concurrency)
        review_failure_history.update(
            {
                f"round-{round_number}:{key}": value
                for key, value in review_failures.items()
            }
        )
        atomic_json(
            root / f"reviews-round-{round_number}.json", dict(sorted(reviews.items()))
        )
        if round_number == args.repair_rounds:
            break

        plan_jobs = []
        for cap in capabilities:
            cid = cap["capability_id"]
            review = reviews.get(cid)
            if not review or not review["portfolio_issues"]:
                continue
            group = sorted(
                [v for v in all_proposals.values() if v["capability_id"] == cid],
                key=lambda item: item["slot"],
            )
            locked = _locked_slot_signature(plans[cid], group)
            repair_prompt = _plan_repair_prompt(
                prompt_records[cid], plans[cid], group, review, locked
            )
            plan_jobs.append(
                (
                    cid,
                    lambda i=cid, p=repair_prompt, expected=locked: store.generate(
                        "plan-repair",
                        i,
                        SYSTEM,
                        p,
                        lambda obj: _validate_plan_repair(obj, i, expected),
                        max_tokens=32000,
                        response_format=_stage_response_format(
                            args, "capability_plan", plan_schema(i)
                        ),
                    ),
                )
            )
        repaired_plans, errors = parallel_map(plan_jobs, args.concurrency)
        plan_repair_failures.update(
            {f"round-{round_number}:{key}": value for key, value in errors.items()}
        )
        for cid, repaired_plan in repaired_plans.items():
            plans[cid] = repaired_plan
            unresolved_plan_repairs.pop(cid, None)
        unresolved_plan_repairs.update(errors)
        if repaired_plans:
            atomic_json(
                root / f"plans-round-{round_number + 1}.json",
                dict(sorted(plans.items())),
            )

        jobs = []
        recovered_nulls = []
        for cap in capabilities:
            cid = cap["capability_id"]
            review = reviews.get(cid)
            if not review:
                continue
            slot_map = {s["slot"]: s for s in plans[cid]["slots"]}
            for result in review["reviews"]:
                # A portfolio defect does not invalidate every sibling. The
                # reviewer marks each affected slot; preserve accepted bytes
                # while repairing the plan and re-reviewing the whole portfolio.
                if result["verdict"] not in ("repair", "reject"):
                    continue
                slot = slot_map[result["slot"]]
                key = f"{cid}:{slot['slot']}"
                feedback = {
                    "individual": result,
                    "portfolio": review["portfolio_issues"],
                    "previous_proposal": all_proposals[key],
                }
                jobs.append(
                    (
                        key,
                        lambda c=cap, p=plans[cid], s=slot, i=cid, k=key, f=feedback: (
                            store.generate(
                                "repair",
                                k,
                                SYSTEM,
                                proposal_prompt(prompt_records[i], p, s, f),
                                lambda obj: validate_proposal(obj, i, s["slot"]),
                                max_tokens=32000,
                                response_format=_stage_response_format(
                                    args,
                                    "capability_proposal",
                                    proposal_schema(i, s["slot"]),
                                ),
                            )
                        ),
                    )
                )
            for missing_slot in review["missing_slots"]:
                slot = slot_map[missing_slot]
                key = f"{cid}:{missing_slot}"
                if key in all_proposals:
                    continue
                if slot["status"] == "null":
                    all_proposals[key] = {
                        "capability_id": cid,
                        "slot": missing_slot,
                        "status": "null",
                        "null_reason": slot["reason"],
                    }
                    unresolved_repairs.pop(key, None)
                    recovered_nulls.append(key)
                    continue
                feedback = {
                    "missing_slot": {
                        "slot": missing_slot,
                        "reason": "the prior generation produced no valid proposal artifact",
                    },
                    "portfolio": review["portfolio_issues"],
                }
                prior_failure = (
                    unresolved_repairs.get(key)
                    or proposal_failures.get(key)
                    or (seed["prior_failures"].get(key) if seed else None)
                )
                if prior_failure is not None:
                    feedback["prior_failure"] = prior_failure
                jobs.append(
                    (
                        key,
                        lambda c=cap, p=plans[cid], s=slot, i=cid, k=key, f=feedback: (
                            store.generate(
                                "repair",
                                k,
                                SYSTEM,
                                proposal_prompt(prompt_records[i], p, s, f),
                                lambda obj: validate_proposal(obj, i, s["slot"]),
                                max_tokens=32000,
                                response_format=_stage_response_format(
                                    args,
                                    "capability_proposal",
                                    proposal_schema(i, s["slot"]),
                                ),
                            )
                        ),
                    )
                )
        if jobs:
            repaired, errors = parallel_map(jobs, args.concurrency)
            repair_failures.update(
                {f"round-{round_number}:{key}": value for key, value in errors.items()}
            )
            for key in repaired:
                unresolved_repairs.pop(key, None)
            unresolved_repairs.update(errors)
            all_proposals.update(repaired)
            atomic_json(
                root / f"proposals-round-{round_number + 1}.json",
                _ordered_proposals(all_proposals),
            )
        elif not plan_jobs and not recovered_nulls:
            break
    # Canonical files always reflect the final in-memory state, never round zero.
    atomic_json(root / "plans.json", dict(sorted(plans.items())))
    atomic_json(root / "proposals.json", _ordered_proposals(all_proposals))
    accepted, rejected, nulls = [], [], []
    capabilities_by_id = {
        capability["capability_id"]: capability for capability in capabilities
    }
    invalid_final_plans = {}
    for cid in capabilities_by_id:
        plan = plans.get(cid)
        if not isinstance(plan, dict):
            invalid_final_plans[cid] = "final plan is missing or malformed"
            continue
        try:
            validate_plan(plan, cid)
        except ValueError as error:
            invalid_final_plans[cid] = str(error)
    partial_portfolio_admissions = 0
    for prop in _ordered_proposals(all_proposals):
        review = reviews.get(prop["capability_id"])
        slot_review = (
            next((r for r in review["reviews"] if r["slot"] == prop["slot"]), None)
            if review
            else None
        )
        if prop["status"] == "null":
            nulls.append(prop)
        elif (
            review
            and review["portfolio_verdict"] in ("accept", "repair")
            and prop["capability_id"] not in invalid_final_plans
            and slot_review
            and slot_review["verdict"] == "accept"
        ):
            if review["portfolio_verdict"] == "repair":
                partial_portfolio_admissions += 1
            accepted.append(
                {
                    "proposal": prop,
                    "review": slot_review,
                    "proposal_hash": digest(prop),
                    "provenance": _source_provenance(
                        pilot, capabilities_by_id[prop["capability_id"]]
                    ),
                }
            )
        else:
            rejected.append(
                {
                    "proposal": prop,
                    "review": slot_review,
                    "portfolio_verdict": review["portfolio_verdict"]
                    if review
                    else "unreviewed",
                }
            )
    expected = {f"{c['capability_id']}:{s}" for c in capabilities for s in range(1, 11)}
    missing = sorted(expected - all_proposals.keys())
    nonaccepting_portfolios = {
        capability["capability_id"]: reviews.get(capability["capability_id"], {}).get(
            "portfolio_verdict", "missing"
        )
        for capability in capabilities
        if reviews.get(capability["capability_id"], {}).get("portfolio_verdict")
        != "accept"
    }
    complete = not any(
        (
            missing,
            review_failures,
            rejected,
            nonaccepting_portfolios,
            unresolved_repairs,
            unresolved_plan_repairs,
            invalid_final_plans,
        )
    )
    report = {
        "stage": "proposal_review",
        "runtime_validated_tasks": 0,
        "capabilities": len(capabilities),
        "expected_slots": len(expected),
        "accepted": len(accepted),
        "acceptance_policy": PARTIAL_ADMISSION_POLICY,
        "partial_portfolio_admissions": partial_portfolio_admissions,
        "invalid_final_plans": invalid_final_plans,
        "rejected_or_needs_repair": len(rejected),
        "null": len(nulls),
        "missing_slots": missing,
        "elapsed_seconds": time.time() - started,
        "environment_counts": dict(
            Counter(p["proposal"]["environment"] for p in accepted)
        ),
        "verifier_counts": dict(
            Counter(p["proposal"]["verification"] for p in accepted)
        ),
        "catalog_status": pilot.get("catalog_audit", {}).get("status", "unknown"),
        "nonaccepting_portfolios": nonaccepting_portfolios,
        "unresolved_repairs": unresolved_repairs,
        "unresolved_plan_repairs": unresolved_plan_repairs,
        "inherited_failures": seed["failure_history"] if seed else [],
        "failures": {
            "plan": plan_failures,
            "proposal": proposal_failures,
            "review": review_failure_history,
            "repair": repair_failures,
            "plan_repair": plan_repair_failures,
        },
        "state": "complete" if complete else "needs_iteration",
    }
    atomic_json(root / "accepted.json", accepted)
    atomic_json(root / "rejected.json", rejected)
    atomic_json(root / "null.json", nulls)
    atomic_json(root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["state"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "propose",
        help="Plan ten slots/capability, generate, review, repair and re-review",
    )
    p.add_argument("--pilot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--seed-run",
        help="Frozen terminal run to freshly review/repair without regenerating proposals",
    )
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--tier", choices=("interactive", "bulk"), default="interactive")
    p.add_argument("--repair-rounds", type=int, default=1)
    p.add_argument("--hold-seconds", type=int, default=3600)
    p.add_argument(
        "--structured-output",
        choices=("json_schema", "off"),
        default="json_schema",
        help="strict JSON-schema output, or off for a diagnostic comparison",
    )
    p.add_argument(
        "--limit", type=int, help="Small, explicitly bounded prompt pilot only"
    )
    p.set_defaults(func=propose)
    from .admission import add_parser as add_admission_parser
    from .evaluation import add_parser as add_evaluation_parser
    from .generate import add_parser as add_generate_parser
    from .judge import add_parser as add_judge_parser
    from .regrade import add_parser as add_regrade_parser
    from .synthesis import add_parser as add_synthesis_parser

    add_judge_parser(sub)
    add_admission_parser(sub)
    add_evaluation_parser(sub)
    add_generate_parser(sub)
    add_regrade_parser(sub)
    add_synthesis_parser(sub)
    args = parser.parse_args(argv)
    if getattr(args, "concurrency", 1) < 1 or getattr(args, "repair_rounds", 0) < 0:
        parser.error("concurrency must be positive; repair rounds must be nonnegative")
    if (
        getattr(args, "session_time", 1) < 1
        or getattr(args, "max_continuations", 0) < 0
    ):
        parser.error(
            "session time must be positive; max continuations must be nonnegative"
        )
    if getattr(args, "validation_timeout", 1) < 1:
        parser.error("validation timeout must be positive")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("limit must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
