# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review with an arm's treatment: class-level repair feedback (L2), the saturated accept (L3) and the shadows.

``treated_review`` runs the stack's ``default_review`` with the band the coordinator passes (the policy's
``band.*``) and then:

* **L3.** A round is *saturated* when ``default_review`` asked for repair only because the solve rate was above the
  band, every one of the ``solver_k`` trials solved the task, and a false-accept search ran to a report on this
  version with no defect needing attention (the FA guard of acceptance.md L3). Arm C accepts it; arms A and B log
  the same decision as the shadow. The stack's coordinator holds any ``Accept`` outside ``IN_BAND``, so the arm
  returns ``Accept(IN_BAND)`` and the band label ``saturated`` lives in ``acceptance-review.json``, which the
  reducer reads; this is the one place the label and the coordinator's band disagree.
* **L2 feedback.** With ``review.feedback == "class"`` a ``Modify`` gains one finding: the class table of
  ``text/class-feedback.md``, after the classes the keyword heuristic assigns to this round's findings. A
  ``Modify`` whose only finding is the solve-rate band names no defect and gets no class table.
* **Shadows** (every arm, never enforced): the decision ``default_review`` would make with the claims that fail
  the L4 preconditions dropped, with the claims the shadow judge contradicted dropped, and with the contrived
  claims dropped (``shadows.py`` writes the per-claim results this reads).

Each call writes ``rounds/<round>/acceptance-review.json``.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter
from taskforge.stage30_author.author import ROUNDS_DIR
from taskforge.stage50_validate.adversary import FALSE_ACCEPT
from taskforge.stage50_validate.round import AdversaryResult, TrialStatus, ValidationResult
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify
from taskforge.stage60_review.rules import BandRules, default_review
from taskforge.stage_coordinator.event_log import WORKPIECES_DIR

REVIEW_FILE = "acceptance-review.json"
SHADOW_FILE = "acceptance-shadow.json"
SATURATED = "saturated"
BAND_FINDING = re.compile(r"\A(\d+) of (\d+) solver trials solved the task: (\w+)\Z")
_DECISION: TypeAdapter[Decision] = TypeAdapter(Decision)


@dataclass(frozen=True)
class ReviewTreatment:
    solver_k: int
    saturated_accept: bool
    class_feedback: str | None
    taxonomy: Mapping[str, Mapping]
    """``kit/taxonomy.json``'s ``classes``."""


def keyword_class(text: str, taxonomy: Mapping[str, Mapping]) -> str:
    """The first taxonomy class one of whose keywords occurs in ``text``; ``other`` when none does."""
    lowered = text.lower()
    for name, entry in taxonomy.items():
        if any(keyword in lowered for keyword in entry["keywords"]):
            return name
    return "other"


def fa_passed(trials: tuple[AdversaryResult, ...]) -> bool:
    """A false-accept search reached a report on this version and asked for nothing to be addressed."""
    searches = [t for t in trials if t.name == FALSE_ACCEPT.name]
    return bool(searches) and all(
        t.status is TrialStatus.GRADED
        and t.report is not None
        and not any(defect.needs_addressing for defect in t.report.defects)
        for t in searches
    )


def saturation(result: ValidationResult, stack: Decision, solver_k: int) -> dict:
    """The L3 reading of a round: whether it is saturated and whether the guard lets it be accepted."""
    band_only = (
        isinstance(stack, Modify)
        and len(stack.findings) == 1
        and (match := BAND_FINDING.match(stack.findings[0])) is not None
        and match.group(3) == BandOutcome.TOO_EASY
    )
    stats = result.solver.stats if result.solver is not None else None
    all_solved = stats is not None and stats.graded == solver_k and stats.solved == stats.graded
    guard = fa_passed(result.adversary)
    return {
        "band_only_modify": band_only,
        "all_solved": all_solved,
        "solved": None if stats is None else stats.solved,
        "graded": None if stats is None else stats.graded,
        "fa_passed": guard,
        "accept": band_only and all_solved and guard,
    }


def _without(result: ValidationResult, drop: Callable[[str, int], bool]) -> ValidationResult:
    """``result`` with each defect whose submissions all satisfy ``drop(trial, ordinal)`` set to not need attention."""
    adversary = []
    for trial in result.adversary:
        if trial.report is None:
            adversary.append(trial)
            continue
        defects = tuple(
            (
                replace(d, needs_addressing=False)
                if d.needs_addressing and all(drop(trial.trial, i) for i in d.submission_ids)
                else d
            )
            for d in trial.report.defects
        )
        adversary.append(replace(trial, report=replace(trial.report, defects=defects)))
    return replace(result, adversary=tuple(adversary))


def _label(decision: Decision) -> str:
    return type(decision).__name__.lower()


async def shadow_decisions(result: ValidationResult, shadow: Mapping | None, band: BandRules) -> dict:
    """The decision ``default_review`` makes when each shadow rule drops the claims it refutes."""
    if shadow is None:
        return {"l4": None, "judge": None, "realistic_only": None}
    claims = {(c["trial"], c["ordinal"]): c for c in shadow["claims"]}

    def rule(test: Callable[[Mapping], bool]) -> Callable[[str, int], bool]:
        return lambda trial, ordinal: (trial, ordinal) in claims and test(claims[(trial, ordinal)])

    rules = {
        "l4": rule(lambda c: c["l4"]["precondition"] == "fail"),
        "judge": rule(lambda c: (c.get("judge") or {}).get("verdict") == "finder_false"),
        "realistic_only": rule(lambda c: (c.get("judge") or {}).get("plausible_solver_output") is False),
    }
    return {name: _label(await default_review(_without(result, test), band=band)) for name, test in rules.items()}


def round_dir(root: Path, workpiece_id: str, round_no: int) -> Path:
    return root / WORKPIECES_DIR / workpiece_id / ROUNDS_DIR / str(round_no)


async def treated_review(
    result: ValidationResult, *, band: BandRules, treatment: ReviewTreatment, root: Path
) -> Decision:
    """The arm's decision on ``result`` against ``band``; writes ``acceptance-review.json`` beside the round's
    ``validation.json``."""
    stack = await default_review(result, band=band)
    l3 = saturation(result, stack, treatment.solver_k)
    decision = stack
    band_label = stack.band if isinstance(stack, Accept) else None
    if treatment.saturated_accept and l3["accept"]:
        decision, band_label = Accept(BandOutcome.IN_BAND), SATURATED
    classes: list[str] = []
    if isinstance(decision, Modify):
        defects = [finding for finding in decision.findings if BAND_FINDING.match(finding) is None]
        classes = [keyword_class(finding, treatment.taxonomy) for finding in defects]
        if treatment.class_feedback is not None and defects:
            named = ", ".join(sorted(set(classes) - {"other"})) or "none recognised"
            lead = f"Defect classes the findings above most likely belong to: {named}."
            text = f"{lead}\n\n{treatment.class_feedback.strip()}"
            decision = Modify((*decision.findings, text))
    directory = round_dir(root, result.workpiece_id, result.round)
    shadow_path = directory / SHADOW_FILE
    shadow = json.loads(shadow_path.read_text()) if shadow_path.exists() else None
    record = {
        "workpiece_id": result.workpiece_id,
        "round": result.round,
        "stack_decision": json.loads(_DECISION.dump_json(stack)),
        "decision": json.loads(_DECISION.dump_json(decision)),
        "band_label": band_label,
        "l3": l3 | {"enforced": treatment.saturated_accept},
        "finding_classes": classes,
        "class_feedback": isinstance(decision, Modify) and isinstance(stack, Modify) and decision != stack,
        "shadow_decisions": (
            await shadow_decisions(result, shadow, band) | {"l3": "accept" if l3["accept"] else _label(stack)}
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REVIEW_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return decision
