# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The acceptance.md section 3.5 rows of one allocation, read back from what the stack and the arm wrote.

``allocation_rows(root, allocation)`` returns the rows of ``records.jsonl``: ``proposal``, ``submission``,
``component``, ``claim``, ``feedback``, ``cost`` and ``terminal``. Nothing here is recomputed by a model; every row
is a reading of a file under the allocation root, so the reducer can be rerun on any capture. ``heldout`` rows are
not produced (held-out evaluation is a separate run).

Statuses are ``pass``, ``fail`` and ``hold``; causes are the stack's enums as text (``Cause``, ``ControlVerdict``,
``RejectKind``), never free text.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from taskforge.ledger.jsonl import read_entries
from taskforge.ledger.records import EntryKind
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.author import ROUNDS_DIR
from taskforge.stage40_build_with_program.agentic_author import AGENTIC_DIR, SUBMIT_FILE
from taskforge.stage50_validate.adversary import FALSE_ACCEPT
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.round import VALIDATION_FILE, TrialStatus, ValidationResult, load_validation
from taskforge.stage_coordinator.event_log import EVENTS_FILE, WORKPIECES_DIR

from experiments.post_training.capability_driven_envs.acceptance_measure.author import EVENTS_FILE as SESSION_EVENTS
from experiments.post_training.capability_driven_envs.acceptance_measure.pipeline import LEDGER_DIR
from experiments.post_training.capability_driven_envs.acceptance_measure.review import REVIEW_FILE, SHADOW_FILE

RECORDS_FILE = "records.jsonl"
DETECTOR_EVENTS = frozenset({"identical_error_hint", "zero_progress_escalation"})
REFUSALS = frozenset({"unchanged_resubmit_refused", "battery_refused"})


@dataclass(frozen=True)
class Allocation:
    arm: str
    capability: str
    domain: str
    rank: int
    slot: int
    proposal: str
    sha256: str

    @property
    def key(self) -> str:
        return f"{self.arm}/{self.capability}"

    @property
    def workpiece_id(self) -> str:
        return f"{self.arm}.{self.capability}"


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _numbered(directory: Path) -> list[int]:
    return sorted(int(p.name) for p in directory.iterdir() if p.name.isdigit()) if directory.exists() else []


def events(root: Path, workpiece_id: str) -> list[dict]:
    stored = _json(root / WORKPIECES_DIR / workpiece_id / EVENTS_FILE)
    return [] if stored is None else stored["events"]


def proposal_row(allocation: Allocation, proposal: TaskProposal, events_: list[dict]) -> dict:
    triaged = next((e for e in events_ if e["kind"] == "triaged"), None)
    header = proposal.header
    return {
        "row": "proposal",
        "proposal_sha256": allocation.sha256,
        "capability": allocation.capability,
        "domain": allocation.domain,
        "slot": allocation.slot,
        "verifier_family": str(header.verification),
        "environment": str(header.environment),
        "no_task": header.null_reason is not None,
        "null_reason": header.null_reason,
        "preflight": None if triaged is None else triaged["attrs"].get("decision"),
    }


def _component(round_no: int, component: str, status: str, **fields: object) -> dict:
    return {"row": "component", "round": round_no, "component": component, "status": status, **fields}


def validation_components(result: ValidationResult) -> Iterator[dict]:
    r = result.round
    for control in result.controls:
        status = {ControlVerdict.MET: "pass", ControlVerdict.VIOLATED: "fail"}.get(control.verdict, "hold")
        yield _component(
            r, "controls", status, id=control.id, kind=str(control.kind), cause=control.cause, reward=control.reward
        )
    if result.solver is not None:
        stats = result.solver.stats
        yield _component(
            r,
            "solver",
            "hold" if result.solver.ungraded else "pass",
            rewards=[t.reward for t in result.solver.trials],
            solved=stats.solved,
            graded=stats.graded,
            ungraded={str(k): v for k, v in result.solver.ungraded.items()},
        )
    for trial in result.adversary:
        name = "fa" if trial.name == FALSE_ACCEPT.name else "fr"
        if trial.status is not TrialStatus.GRADED or trial.report is None:
            yield _component(r, name, "hold", trial=trial.trial, cause=trial.cause, author_visible=False)
            continue
        wanted = [d for d in trial.report.defects if d.needs_addressing]
        yield _component(
            r, name, "fail" if wanted else "pass", trial=trial.trial, defects=len(trial.report.defects),
            needs_addressing=len(wanted), submissions=len(trial.submissions),
        )  # fmt: skip


def round_rows(round_dir: Path, round_no: int) -> Iterator[dict]:
    sessions = round_dir / AGENTIC_DIR
    index = 0
    for k in _numbered(sessions):
        session = sessions / str(k)
        session_events = _jsonl(session / SESSION_EVENTS)
        for event in session_events:
            if event["kind"] == "probe_battery":
                yield _component(
                    round_no, "probe_battery", "pass" if event["all_met"] else "fail",
                    session=f"{round_no}.{k}", program_digest=event["program_digest"], probes=event["probes"],
                )  # fmt: skip
            elif event["kind"] in REFUSALS:
                yield {"row": "submission", "round": round_no, "session": f"{round_no}.{k}", "charged": False,
                       "refusal": event["kind"], "program_digest": event["program_digest"]}  # fmt: skip
            elif event["kind"] in DETECTOR_EVENTS:
                detail = {key: value for key, value in event.items() if key != "kind"}
                yield {"row": "cost", "kind": "detector", "event": event["kind"], "round": round_no,
                       "session": f"{round_no}.{k}", **detail}  # fmt: skip
        submit = _json(session / SUBMIT_FILE)
        if submit is not None and submit.get("accepted"):
            yield {
                "row": "submission",
                "round": round_no,
                "session": f"{round_no}.{k}",
                "index": index,
                "program_digest": submit["digest"],
                "tested": submit["tested"],
                "charged": True,
                "unchanged_from_previous": submit.get("unchanged_from_previous", False),
                "battery_met": submit.get("battery_met"),
            }
            index += 1
    if (round_dir / "draft").exists():
        yield _component(round_no, "build_selfcheck", "pass")
    if (round_dir / VALIDATION_FILE).exists():
        result = load_validation(round_dir)
        yield from validation_components(result)
        yield {"row": "component", "round": round_no, "component": "lowering", "status": str(result.lowering.status),
               "task_digest": result.lowering.task_digest}  # fmt: skip
    shadow = _json(round_dir / SHADOW_FILE)
    for claim in [] if shadow is None else shadow.get("claims", []):
        yield {"row": "claim", "round": round_no, **claim}
    if shadow is not None and "error" in shadow:
        yield {"row": "claim_error", "round": round_no, "error": shadow["error"][-2000:]}
    review = _json(round_dir / REVIEW_FILE)
    if review is not None and review["decision"].get("findings") is not None:
        witnesses = [f"{c['trial']}-s{c['ordinal']}" for c in (shadow or {}).get("claims", []) if c["needs_addressing"]]
        yield {
            "row": "feedback",
            "round": round_no,
            "classes": review["finding_classes"],
            "class_feedback": review["class_feedback"],
            "witness_ids": witnesses,
            "findings": len(review["decision"]["findings"]),
        }


def cost_rows(ledger: Path) -> Iterator[dict]:
    if not ledger.exists():
        return
    for entry in read_entries(ledger):
        if entry.kind is not EntryKind.LLM_CALL:
            continue
        attrs = entry.attrs
        yield {
            "row": "cost",
            "kind": "llm_call",
            "round": entry.round,
            "step": entry.step,
            "stage": stage_of(entry.step, attrs),
            "model": entry.model,
            "tokens_in": entry.tokens_in,
            "tokens_cached": int(attrs.get("cached_tokens", 0)),
            "tokens_out": entry.tokens_out,
            "tokens_reasoning": entry.tokens_reasoning,
            "wall": entry.ended - entry.started,
            "cause": entry.cause,
            "attempts": attrs.get("attempts"),
        }


def stage_of(step: str, attrs: Mapping[str, str]) -> str:
    """The section 1.3 stage of a model call, from its ledger step and tags."""
    if attrs.get("author_kind"):
        return "author"
    if attrs.get("origin") == "author_session":
        return "author_build"
    head = step.split("/", 1)[0]
    if head.startswith("solver"):
        return "solver"
    if head.startswith("adversary"):
        return "adversary"
    if head.startswith("control"):
        return "controls"
    return "build"


def terminal_row(events_: list[dict], last_review: dict | None) -> dict:
    terminal = next((e for e in reversed(events_) if e["kind"] == "terminal"), None)
    attrs = {} if terminal is None else terminal["attrs"]
    return {
        "row": "terminal",
        "decision": attrs.get("terminal"),
        "kind": attrs.get("kind"),
        "reason": attrs.get("reason"),
        "rounds": max((e["round"] for e in events_), default=0) + 1 if events_ else 0,
        "band_label": None if last_review is None else last_review["band_label"],
        "shadow_decisions": None if last_review is None else last_review["shadow_decisions"],
        "l3": None if last_review is None else last_review["l3"],
    }


def allocation_rows(root: Path, allocation: Allocation, proposal: TaskProposal) -> list[dict]:
    """Every section 3.5 row of the allocation under ``root``, each tagged with the allocation key and arm."""
    workpiece = root / WORKPIECES_DIR / allocation.workpiece_id
    events_ = events(root, allocation.workpiece_id)
    rows = [proposal_row(allocation, proposal, events_)]
    last_review = None
    for round_no in _numbered(workpiece / ROUNDS_DIR):
        round_dir = workpiece / ROUNDS_DIR / str(round_no)
        rows += round_rows(round_dir, round_no)
        last_review = _json(round_dir / REVIEW_FILE) or last_review
    rows += cost_rows(root / LEDGER_DIR / f"{allocation.workpiece_id}.jsonl")
    rows.append(terminal_row(events_, last_review))
    return [
        {"allocation": allocation.key, "arm": allocation.arm, "capability": allocation.capability, **row} for row in rows
    ]


def write_records(root: Path, allocation: Allocation, proposal: TaskProposal) -> Path:
    path = root / RECORDS_FILE
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in allocation_rows(root, allocation, proposal)))
    return path
