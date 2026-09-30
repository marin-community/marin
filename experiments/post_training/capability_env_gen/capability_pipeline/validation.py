"""Fail-closed structural gates; semantic and runtime gates are separate."""

from .prompts import ENVIRONMENTS, VERIFIERS


class InvalidArtifact(ValueError):
    pass


PARTIAL_ADMISSION_POLICY = "reviewed_slot_on_repair_v1"


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


def text_fields(obj, fields):
    for field in fields:
        require(
            isinstance(obj.get(field), str) and bool(obj[field].strip()),
            f"missing text: {field}",
        )


def list_fields(obj, fields, nonempty=True):
    for field in fields:
        require(
            isinstance(obj.get(field), list) and (not nonempty or bool(obj[field])),
            f"missing list: {field}",
        )


def validate_plan(obj, capability_id):
    require(obj.get("capability_id") == capability_id, "plan capability mismatch")
    text_fields(obj, ["coverage_rationale"])
    list_fields(obj, ["excluded_combinations", "research_priorities"], nonempty=False)
    list_fields(obj, ["slots"])
    excluded_pairs = set()
    for excluded in obj["excluded_combinations"]:
        require(isinstance(excluded, dict), "excluded combination must be an object")
        require(
            excluded.get("environment") in ENVIRONMENTS,
            f"excluded environment {excluded.get('environment')!r} not in {ENVIRONMENTS!r}",
        )
        require(
            excluded.get("verification") in VERIFIERS,
            f"excluded verifier {excluded.get('verification')!r} not in {VERIFIERS!r}",
        )
        text_fields(excluded, ["reason"])
        pair = (excluded["environment"], excluded["verification"])
        require(pair not in excluded_pairs, "duplicate excluded combination")
        excluded_pairs.add(pair)
    require(
        all(
            isinstance(item, str) and item.strip()
            for item in obj["research_priorities"]
        ),
        "research priorities must be non-empty strings",
    )
    require(len(obj["slots"]) == 10, "exactly ten slots required")
    require(
        {s.get("slot") for s in obj["slots"]} == set(range(1, 11)),
        "slots must be 1..10",
    )
    for slot in obj["slots"]:
        require(slot.get("status") in ("propose", "null"), "invalid slot status")
        if slot["status"] == "null":
            text_fields(slot, ["reason"])
        else:
            text_fields(slot, ["title", "workflow", "distinctive_challenge"])
            require(
                slot.get("environment") in ENVIRONMENTS,
                f"environment {slot.get('environment')!r} not in {ENVIRONMENTS!r}",
            )
            require(
                slot.get("verification") in VERIFIERS,
                f"verifier {slot.get('verification')!r} not in {VERIFIERS!r}",
            )
            require(
                (slot["environment"], slot["verification"]) not in excluded_pairs,
                "excluded combination is used by a proposed slot",
            )


def validate_proposal(obj, capability_id, slot):
    require(
        obj.get("capability_id") == capability_id and obj.get("slot") == slot,
        "proposal identity mismatch",
    )
    require(obj.get("status") in ("proposed", "null"), "invalid proposal status")
    if obj["status"] == "null":
        text_fields(obj, ["null_reason"])
        return
    text_fields(
        obj,
        [
            "title",
            "task_family",
            "capability_alignment",
            "environment_rationale",
            "workflow",
            "task_brief",
        ],
    )
    require(
        obj.get("environment") in ENVIRONMENTS,
        f"environment {obj.get('environment')!r} not in {ENVIRONMENTS!r}",
    )
    require(
        obj.get("verification") in VERIFIERS,
        f"verifier {obj.get('verification')!r} not in {VERIFIERS!r}",
    )
    list_fields(
        obj,
        [
            "inputs",
            "deliverables",
            "constraints",
            "difficulty_drivers",
            "builder_plan",
            "validation_plan",
            "risks",
        ],
    )
    grounding = obj.get("grounding", {})
    list_fields(
        grounding, ["known_facts", "research_needed", "sources"], nonempty=False
    )
    for source in grounding["sources"]:
        text_fields(source, ["query_or_url", "purpose", "license_check", "fallback"])
        require(
            source.get("verification_status") == "unverified",
            "text-only author cannot verify sources",
        )
    env = obj.get("environment_spec", {})
    text_fields(
        env, ["initial_state", "reset", "dependency_strategy", "resource_estimate"]
    )
    list_fields(env, ["tools"], nonempty=False)
    verify = obj.get("verification_spec", {})
    text_fields(verify, ["observable_success", "grader_design"])
    list_fields(
        verify, ["positive_controls", "negative_controls", "anti_shortcuts", "rubric"]
    )
    policy = obj.get("data_policy", {})
    text_fields(policy, ["provenance", "license", "split_group", "contamination_check"])
    list_fields(policy, ["private_evaluator_data"])
    seen = set()
    for session in obj["builder_plan"]:
        text_fields(session, ["session", "goal"])
        list_fields(session, ["depends_on"], nonempty=False)
        list_fields(session, ["handoff_artifacts", "acceptance_checks"])
        require(
            set(session["depends_on"]) <= seen,
            "builder DAG must be topologically ordered",
        )
        require(session["session"] not in seen, "duplicate session")
        seen.add(session["session"])
    for risk in obj["risks"]:
        text_fields(risk, ["risk", "mitigation", "abandon_if"])


AXES = {
    "realism",
    "alignment",
    "specificity",
    "reward_validity",
    "environment_fit",
    "diversity",
    "source_honesty",
}


def validate_review(obj, capability_id, slots, proposal_statuses=None):
    slots = set(slots)
    require(obj.get("capability_id") == capability_id, "review capability mismatch")
    require(
        obj.get("portfolio_verdict") in ("accept", "repair", "reject"),
        "invalid portfolio verdict",
    )
    list_fields(obj, ["reviews", "portfolio_issues", "missing_slots"], nonempty=False)
    require(
        set(obj["missing_slots"]) == set(range(1, 11)) - slots,
        "missing_slots must reflect absent portfolio slots",
    )
    require(len(obj["reviews"]) == len(slots), "missing/duplicate reviews")
    require({r.get("slot") for r in obj["reviews"]} == slots, "review slots differ")
    for review in obj["reviews"]:
        require(
            review.get("verdict") in ("accept", "repair", "reject", "null"),
            "invalid review verdict",
        )
        list_fields(
            review, ["critical_failures", "issues", "required_changes"], nonempty=False
        )
        scores = review.get("scores", {})
        require(set(scores) == AXES, "wrong score axes")
        require(
            all(type(s) is int and 1 <= s <= 5 for s in scores.values()),
            "scores must be integers 1..5",
        )
        if proposal_statuses is not None:
            proposal_status = proposal_statuses.get(review.get("slot"))
            require(
                proposal_status in ("proposed", "null"),
                "review has no matching proposal status",
            )
            require(
                (review["verdict"] == "null") == (proposal_status == "null"),
                "null review verdict contradicts proposal status",
            )
        if review["verdict"] == "accept":
            require(
                min(scores.values()) >= 4 and not review["critical_failures"],
                "accept contradicts scores/failures",
            )
            require(
                not review["required_changes"], "accepted review still requires changes"
            )
    if obj["portfolio_verdict"] == "accept":
        require(
            not obj["missing_slots"] and not obj["portfolio_issues"],
            "portfolio acceptance contradicts issues",
        )
        require(
            all(r["verdict"] in ("accept", "null") for r in obj["reviews"]),
            "portfolio acceptance contradicts slot review",
        )
    else:
        require(
            obj["portfolio_issues"]
            or obj["missing_slots"]
            or any(r["verdict"] in ("repair", "reject") for r in obj["reviews"]),
            "nonaccepting portfolio has no actionable feedback",
        )
