"""Fail-closed metadata accounting for every catalog capability and ten slots."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .catalog import load_pilot
from .inference import digest


class CoverageError(ValueError):
    """Frozen manifest and terminal receipts disagree."""


def _provider_failure(item_root: Path, issues: Any) -> bool:
    if not isinstance(issues, list) or not any(
        isinstance(issue, str) and issue.startswith("runtime controls failed:")
        for issue in issues
    ):
        return False
    for path in (item_root / "runtime-trials").glob("*/result.json"):
        trial = _read(path)
        exception = trial.get("exception_info")
        if not isinstance(exception, dict) or trial.get("verifier_result") is not None:
            continue
        detail = str(exception.get("exception_type", "")) + " " + str(exception.get("exception_message", ""))
        if any(marker in detail for marker in ("ProviderRateLimitExhausted", "DaytonaRateLimitError", "DaytonaNotFoundError")):
            return True
    return False


def _semantic_rejection(issues: Any) -> bool:
    if not isinstance(issues, list) or not issues or not all(isinstance(issue, str) for issue in issues):
        return False
    prefixes = (
        "invalid task bundle:",
        "invalid controls:",
        "duplicate generated TaskSpec id:",
        "quality review rejected:",
    )
    return all(issue.startswith(prefixes) for issue in issues)


def _exhausted_rejection(row: dict[str, Any]) -> bool:
    marker = row.get("terminal_disposition")
    if marker is None:
        return False
    if marker != "rejected":
        raise CoverageError(f"unknown terminal disposition: {marker}")
    budget = row.get("repair_budget")
    if not isinstance(budget, dict):
        raise CoverageError("terminal rejection lacks repair budget")
    used, maximum, exhausted = budget.get("used"), budget.get("max"), budget.get("exhausted")
    if (
        type(used) is not int or type(maximum) is not int
        or used < 0 or maximum < 0 or used < maximum
        or exhausted is not True
    ):
        raise CoverageError("terminal rejection has invalid exhausted repair budget")
    return True


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageError(f"missing or invalid evidence: {path}") from exc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(proposal: dict, ids: set[str]) -> tuple[str, int]:
    cap, slot = proposal.get("capability_id"), proposal.get("slot")
    if cap not in ids or type(slot) is not int or not 1 <= slot <= 10:
        raise CoverageError(f"out-of-manifest identity: {cap}:{slot}")
    return cap, slot


def coverage_report(
    manifest_path: str | Path,
    proposal_root: str | Path | None = None,
    synthesis_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Reconcile all slots using only terminal reports, lists, and item statuses."""
    manifest_path = Path(manifest_path)
    manifest = load_pilot(manifest_path)
    ids = {row["capability_id"] for row in manifest["capabilities"]}
    proposals: dict[tuple[str, int], dict] = {}
    proposal_source = None
    if proposal_root is not None:
        root = Path(proposal_root)
        report = _read(root / "report.json")
        if report.get("stage") != "proposal_review" or report.get("state") not in {"complete", "needs_iteration"}:
            raise CoverageError("proposal run lacks terminal report")
        if digest(_read(root / "input_pilot.json")) != digest(manifest):
            raise CoverageError("proposal run input differs from frozen manifest")
        run = _read(root / "run.json")
        selected = run.get("capability_ids") if isinstance(run, dict) else None
        if (
            not isinstance(run, dict)
            or run.get("stage") != "propose"
            or run.get("pilot_hash") != digest(manifest)
            or not isinstance(selected, list)
            or not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= ids
            or report.get("capabilities") != len(selected)
            or report.get("expected_slots") != len(selected) * 10
        ):
            raise CoverageError("proposal run selection disagrees with frozen manifest")
        selected_slots = {(cap, slot) for cap in selected for slot in range(1, 11)}
        for category in ("accepted", "rejected", "null"):
            rows = _read(root / f"{category}.json")
            if not isinstance(rows, list):
                raise CoverageError(f"{category} is not a list")
            for row in rows:
                proposal = row if category == "null" else row.get("proposal") if isinstance(row, dict) else None
                if not isinstance(proposal, dict):
                    raise CoverageError(f"{category} row lacks proposal")
                identity = _identity(proposal, ids)
                if identity not in selected_slots:
                    raise CoverageError(f"proposal outside declared run selection: {identity}")
                if identity in proposals:
                    raise CoverageError(f"duplicate proposal slot: {identity}")
                proposal_hash = digest(proposal)
                if row.get("proposal_hash") is not None and row["proposal_hash"] != proposal_hash:
                    raise CoverageError(f"mismatched proposal hash: {identity}")
                if category == "accepted" and (row.get("proposal_hash") != proposal_hash or row.get("review", {}).get("verdict") != "accept"):
                    raise CoverageError(f"invalid accepted proposal: {identity}")
                proposals[identity] = {"category": category, "proposal_hash": proposal_hash}
        counts = Counter(row["category"] for row in proposals.values())
        if (report.get("accepted"), report.get("rejected_or_needs_repair"), report.get("null")) != (counts["accepted"], counts["rejected"], counts["null"]):
            raise CoverageError("proposal report counts disagree")
        missing_inventory = report.get("missing_slots")
        if (
            not isinstance(missing_inventory, list)
            or len(missing_inventory) != len(set(missing_inventory))
            or set(missing_inventory) != {f"{c}:{s}" for c, s in selected_slots - proposals.keys()}
        ):
            raise CoverageError("proposal missing-slot inventory disagrees")
        proposal_source = {"path": str(root), "report_sha256": _sha(root / "report.json")}
    syntheses: dict[tuple[str, int], dict] = {}
    synth_sources = []
    task_ids: set[str] = set()
    for value in synthesis_roots:
        root = Path(value)
        report, tasks = _read(root / "report.json"), _read(root / "tasks.json")
        if report.get("stage") != "synthesize" or report.get("state") not in {"complete", "needs_continuation"} or not isinstance(tasks, list):
            raise CoverageError(f"synthesis run lacks terminal evidence: {root}")
        if report.get("completed_items") != len(tasks) or report.get("states") != dict(Counter(row.get("state") for row in tasks)):
            raise CoverageError(f"synthesis report counts disagree: {root}")
        for row in tasks:
            if not isinstance(row, dict):
                raise CoverageError("malformed synthesis task")
            try:
                cap, slot_text = row["key"].rsplit(":", 1)
                slot = int(slot_text)
            except (KeyError, AttributeError, ValueError) as exc:
                raise CoverageError("invalid synthesis key") from exc
            identity = _identity({"capability_id": cap, "slot": slot}, ids)
            if identity in syntheses:
                raise CoverageError(f"duplicate synthesis slot: {identity}")
            prior = proposals.get(identity)
            if prior is None or prior["category"] != "accepted" or row.get("proposal_hash") != prior["proposal_hash"]:
                raise CoverageError(f"synthesis proposal absent or hash mismatch: {identity}")
            statuses = list((root / "items").glob(f"*-{prior['proposal_hash'][:12]}/status.json"))
            if len(statuses) != 1 or _read(statuses[0]) != row:
                raise CoverageError(f"missing or mismatched authoritative item status: {identity}")
            task = row.get("taskcompendium")
            task_id = task.get("id") if isinstance(task, dict) else task
            if task_id is not None:
                if not isinstance(task_id, str) or task_id in task_ids:
                    raise CoverageError(f"duplicate/invalid task identity: {task_id}")
                task_ids.add(task_id)
            state = row.get("state")
            terminal_rejection = _exhausted_rejection(row)
            if terminal_rejection:
                if state == "quality_accepted":
                    raise CoverageError(f"accepted task has terminal rejection marker: {identity}")
                disposition = "rejected"
            elif state == "quality_accepted":
                if row.get("runtime_validated") is not True:
                    raise CoverageError(f"quality acceptance lacks runtime validation: {identity}")
                disposition = "quality_accepted"
            elif state == "pending_readmission":
                disposition = "needs_readmission"
            elif state == "failed":
                disposition = (
                    "infrastructure_failure" if _provider_failure(statuses[0].parent, row.get("issues"))
                    else "rejected" if _semantic_rejection(row.get("issues"))
                    else "execution_failure_unresolved"
                )
            elif state in {"pending_adversary_retry", "pending_runtime"} and row.get("issues"):
                disposition = "infrastructure_or_retry_pending"
            elif isinstance(state, str) and (state.startswith("pending_") or state == "runtime_controls_passed_pending_adversary"):
                disposition = "pending_construction"
            else:
                raise CoverageError(f"unknown synthesis state: {state}")
            syntheses[identity] = {"disposition": disposition, "state": state, "proposal_hash": prior["proposal_hash"], "status_sha256": _sha(statuses[0])}
        synth_sources.append({"path": str(root), "report_sha256": _sha(root / "report.json")})
    capabilities = []
    aggregate = Counter()
    for item in manifest["capabilities"]:
        cap = item["capability_id"]
        slots = []
        for slot in range(1, 11):
            identity = cap, slot
            proposal, synthesis = proposals.get(identity), syntheses.get(identity)
            disposition = synthesis["disposition"] if synthesis else (
                "pending_construction" if proposal and proposal["category"] == "accepted"
                else proposal["category"] if proposal else "missing_proposal"
            )
            aggregate[disposition] += 1
            slots.append({"slot": slot, "disposition": disposition, "proposal_hash": proposal["proposal_hash"] if proposal else None, "synthesis": synthesis})
        terminal = {"null", "rejected", "quality_accepted", "needs_readmission"}
        capabilities.append({
            "capability_id": cap, "subject_id": item["subject_id"], "slots": slots,
            "accounting_complete": all(row["disposition"] in terminal for row in slots),
            "has_accepted_task": any(row["disposition"] == "quality_accepted" for row in slots),
            "all_slots_quality_accepted": all(row["disposition"] == "quality_accepted" for row in slots),
        })
    accounting_complete = all(row["accounting_complete"] for row in capabilities)
    return {
        "schema_version": "capability-catalog-coverage-v1",
        "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
        "capabilities_total": len(capabilities), "slots_total": len(capabilities) * 10,
        "capabilities_accounted": sum(row["accounting_complete"] for row in capabilities),
        "capabilities_with_accepted_tasks": sum(row["has_accepted_task"] for row in capabilities),
        "accounting_complete": accounting_complete,
        "coverage_complete": accounting_complete,
        "all_slots_quality_accepted": all(row["all_slots_quality_accepted"] for row in capabilities),
        "scope": "terminal catalog accounting only; no training admission assertion",
        "slot_dispositions": dict(sorted(aggregate.items())),
        "proposal_source": proposal_source, "synthesis_sources": synth_sources,
        "capabilities": capabilities,
    }
