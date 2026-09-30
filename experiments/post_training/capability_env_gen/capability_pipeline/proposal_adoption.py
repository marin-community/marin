"""Adopt a frozen, independently checked proposal checkpoint without inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .catalog import load_pilot
from .inference import atomic_json, digest
from .validation import (
    PARTIAL_ADMISSION_POLICY,
    validate_plan,
    validate_proposal,
    validate_review,
)

SCHEMA = "capability-proposal-adoption-v1"
_PROPOSAL_FILES = (
    "accepted.json", "rejected.json", "null.json", "plans.json",
    "proposals.json", "report.json", "run.json",
)
_REQUIRED_SOURCE = (*_PROPOSAL_FILES, "input_pilot.json", "reviews-round-1.json")
_HELD = frozenset({"c21.spreadsheets.calculate"})
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_UNPACKED_BYTES = 256 * 1024 * 1024


class AdoptionError(ValueError):
    """The checkpoint cannot be promoted into a new controller run."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AdoptionError(f"required source file missing or linked: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise AdoptionError(f"invalid source JSON: {path.name}") from error


def _member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise AdoptionError("snapshot contains an unsafe path")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise AdoptionError("snapshot contains a linked path")
    return path


def _ordinary_ancestors(path: Path) -> None:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise AdoptionError("adoption path traverses a symlink")


def _verify_snapshot(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    _ordinary_ancestors(root)
    receipt = _read(root / "pull-manifest.json")
    capture = _read(root / "snapshot-capture.json")
    raw = capture.get("remote_manifest_json")
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != capture.get("remote_manifest_sha256"):
        raise AdoptionError("captured remote manifest fingerprint mismatch")
    try:
        remote = json.loads(raw)
    except ValueError as error:
        raise AdoptionError("captured remote manifest is invalid") from error
    files = remote.get("files")
    if (not isinstance(files, dict) or not files or receipt.get("files") != files
            or receipt.get("remote_manifest_sha256") != capture["remote_manifest_sha256"]
            or receipt.get("remote_snapshot_id") != remote.get("snapshot_id")
            or receipt.get("complete_manifest") is not True
            or receipt.get("complete_snapshot") is not True
            or receipt.get("omitted") or remote.get("omitted")):
        raise AdoptionError("checkpoint is not a complete verified snapshot")
    for name, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise AdoptionError("snapshot inventory has an invalid SHA-256")
        path = _member(root, name)
        if not path.is_file() or _sha(path) != expected:
            raise AdoptionError(f"snapshot member fingerprint mismatch: {name}")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AdoptionError("snapshot contains an undeclared symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise AdoptionError("snapshot contains a special file")
    if actual_files != set(files) | {"snapshot-capture.json", "pull-manifest.json"}:
        raise AdoptionError("snapshot has missing or undeclared files")
    return files, {
        "snapshot_id": remote["snapshot_id"],
        "remote_manifest_sha256": capture["remote_manifest_sha256"],
        "remote_final": remote.get("final") is True,
        "files_verified": len(files),
    }


def _verify_archive(path: Path, launch: dict[str, Any], pilot: Path,
                    original_controller_sha256: str) -> str:
    _ordinary_ancestors(path)
    if path.is_symlink() or not path.is_file():
        raise AdoptionError("original source archive is unavailable")
    if path.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise AdoptionError("source archive exceeds bounded transport size")
    archive_sha = _sha(path)
    if archive_sha != launch.get("source_snapshot_sha256"):
        raise AdoptionError("original source archive differs from launch receipt")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not member.isfile() or member.name.startswith("/") or
               PurePosixPath(member.name).as_posix() != member.name or
               any(part in {"", ".", ".."} for part in member.name.split("/")) or
               member.size > _MAX_ARCHIVE_MEMBER_BYTES for member in members):
            raise AdoptionError("source archive has unsafe members")
        if sum(member.size for member in members) > _MAX_ARCHIVE_UNPACKED_BYTES:
            raise AdoptionError("source archive exceeds bounded unpacked size")
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise AdoptionError("source archive has missing or duplicate manifest")
        contents = {member.name: archive.extractfile(member).read() for member in members}
    try:
        source_manifest = json.loads(contents.pop("manifest.json"))
    except (ValueError, KeyError) as error:
        raise AdoptionError("source archive manifest is invalid") from error
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    if source_manifest.get("files") != expected:
        raise AdoptionError("source archive member hashes disagree")
    if launch.get("source_snapshot_files") != len(expected):
        raise AdoptionError("source archive member count differs from launch")
    if contents.get("inputs/pilot.json") != pilot.read_bytes():
        raise AdoptionError("source archive pilot differs from requested pilot")
    for name, expected_sha in launch.get("controller_files", {}).items():
        if expected.get(name) != expected_sha:
            raise AdoptionError(f"launched controller source differs: {name}")
    package = {
        name.removeprefix("capability_pipeline/"): sha
        for name, sha in expected.items()
        if name.startswith("capability_pipeline/") and
        name.endswith((".py", ".json")) and "__pycache__" not in name
    }
    if digest(dict(sorted(package.items()))) != original_controller_sha256:
        raise AdoptionError("original controller package differs from checkpoint")
    return archive_sha


def _validate_classification(source: Path, pilot_doc: dict[str, Any]) -> dict[str, int]:
    run = _read(source / "run.json")
    report = _read(source / "report.json")
    plans = _read(source / "plans.json")
    proposals = _read(source / "proposals.json")
    reviews = _read(source / "reviews-round-1.json")
    capabilities = {row["capability_id"]: row for row in pilot_doc["capabilities"]}
    ids = list(capabilities)
    if (run.get("stage") != "propose" or run.get("capability_ids") != ids
            or run.get("pilot_hash") != digest(pilot_doc)
            or report.get("stage") != "proposal_review"
            or report.get("state") not in {"complete", "needs_iteration"}
            or report.get("capabilities") != len(ids)
            or report.get("expected_slots") != 10 * len(ids)
            or not isinstance(plans, dict) or set(plans) != set(ids)
            or not isinstance(reviews, dict) or set(reviews) != set(ids)
            or not isinstance(proposals, list)):
        raise AdoptionError("proposal stage identity or terminal accounting differs")
    if _read(source / "input_pilot.json") != pilot_doc:
        raise AdoptionError("proposal input pilot differs")
    acceptance_policy = report.get("acceptance_policy")
    if acceptance_policy not in (None, PARTIAL_ADMISSION_POLICY):
        raise AdoptionError("proposal acceptance policy is unknown")
    for cid, plan in plans.items():
        validate_plan(plan, cid)
    indexed = {}
    for proposal in proposals:
        cid, slot = proposal.get("capability_id"), proposal.get("slot")
        if cid not in capabilities or type(slot) is not int or not 1 <= slot <= 10:
            raise AdoptionError("proposal has an out-of-catalog slot")
        validate_proposal(proposal, cid, slot)
        key = cid, slot
        if key in indexed:
            raise AdoptionError("proposal slot is duplicated")
        indexed[key] = proposal
    accepted, rejected, nulls = [], [], []
    for cid in ids:
        group = {slot: prop for (cap, slot), prop in indexed.items() if cap == cid}
        review = reviews[cid]
        validate_review(review, cid, group, {slot: prop["status"] for slot, prop in group.items()})
        if review["portfolio_verdict"] == "accept":
            # The original controller required a valid plan for a whole-portfolio acceptance.
            validate_plan(plans[cid], cid)
        slot_reviews = {row["slot"]: row for row in review["reviews"]}
        for slot, prop in sorted(group.items()):
            slot_review = slot_reviews[slot]
            if prop["status"] == "null":
                nulls.append(prop)
            elif (
                review["portfolio_verdict"] == "accept"
                or acceptance_policy == PARTIAL_ADMISSION_POLICY
                and review["portfolio_verdict"] == "repair"
            ) and slot_review["verdict"] == "accept":
                if cid in _HELD:
                    raise AdoptionError("held capability cannot be adopted as accepted")
                provenance = {
                    "catalog_source": pilot_doc["source"],
                    "capability_record": capabilities[cid],
                    "capability_record_hash": digest(capabilities[cid]),
                }
                from .prompts import capability_prompt_record
                progression = capability_prompt_record(capabilities[cid], pilot_doc).get("learning_progression")
                if progression is not None:
                    provenance["learning_progression"] = progression
                    provenance["learning_progression_hash"] = digest(progression)
                accepted.append({"proposal": prop, "review": slot_review,
                                 "proposal_hash": digest(prop), "provenance": provenance})
            else:
                rejected.append({"proposal": prop, "review": slot_review,
                                 "portfolio_verdict": review["portfolio_verdict"]})
    for name, rebuilt in (("accepted", accepted), ("rejected", rejected), ("null", nulls)):
        if _read(source / f"{name}.json") != rebuilt:
            raise AdoptionError(f"{name} inventory differs from final review classification")
    missing = sorted(f"{cid}:{slot}" for cid in ids for slot in range(1, 11)
                     if (cid, slot) not in indexed)
    if (report.get("accepted") != len(accepted)
            or report.get("rejected_or_needs_repair") != len(rejected)
            or report.get("null") != len(nulls)
            or report.get("missing_slots") != missing
            or len(accepted) + len(rejected) + len(nulls) + len(missing) != 10 * len(ids)
            or report.get("environment_counts") != dict(Counter(row["proposal"]["environment"] for row in accepted))
            or report.get("verifier_counts") != dict(Counter(row["proposal"]["verification"] for row in accepted))):
        raise AdoptionError("proposal report counts or missing slots disagree")
    if acceptance_policy == PARTIAL_ADMISSION_POLICY and (
        report.get("partial_portfolio_admissions") != sum(
            row["review"]["verdict"] == "accept"
            and reviews[row["proposal"]["capability_id"]]["portfolio_verdict"] == "repair"
            for row in accepted
        )
        or report.get("invalid_final_plans") != {}
    ):
        raise AdoptionError("partial proposal admission accounting differs")
    if not isinstance(report.get("failures"), dict) or not isinstance(run.get("repair_rounds"), int):
        raise AdoptionError("original proposal repair history is missing")
    return {"expected_slots": 10 * len(ids), "accepted": len(accepted),
            "rejected": len(rejected), "null": len(nulls), "missing": len(missing),
            "original_proposal_repair_rounds": run["repair_rounds"]}


def _validated_checkpoint(snapshot_dir: Path, source_archive: Path,
                          launch_receipt: Path, pilot: Path) -> dict[str, Any]:
    snapshot_dir, source_archive, launch_receipt = map(Path, (snapshot_dir, source_archive, launch_receipt))
    pilot = Path(pilot)
    for path in (snapshot_dir, source_archive, launch_receipt, pilot):
        _ordinary_ancestors(path)
    if pilot.is_symlink() or not pilot.is_file():
        raise AdoptionError("adoption pilot is missing or linked")
    launch = _read(launch_receipt)
    pilot_doc = load_pilot(pilot)
    if launch.get("manifest_sha256") != _sha(pilot) or launch.get("manifest_capabilities") != len(pilot_doc["capabilities"]):
        raise AdoptionError("launch pilot identity differs")
    files, snapshot_id = _verify_snapshot(snapshot_dir)
    if (any(name.startswith("construction/") for name in files) or
            any(name in {"construction-receipt.json", "construction-progress.json"}
                for name in files)):
        raise AdoptionError("checkpoint already contains construction state")
    original_run = _read(snapshot_dir / "generate-run.json")
    if (original_run.get("pilot_sha256") != _sha(pilot)
            or original_run.get("settings", {}).get("controller_package_sha256") is None
            or original_run.get("identity_sha256") != digest({
                k: v for k, v in original_run.items() if k != "identity_sha256"
            })):
        raise AdoptionError("original generate identity differs from launch pilot")
    archive_sha = _verify_archive(source_archive, launch, pilot,
                                  original_run["settings"]["controller_package_sha256"])
    source = snapshot_dir / "proposal"
    for name in _REQUIRED_SOURCE:
        if f"proposal/{name}" not in files:
            raise AdoptionError(f"original proposal artifact is absent: {name}")
    counts = _validate_classification(source, pilot_doc)
    if (launch.get("proposal_slots") != counts["expected_slots"]
            or original_run.get("settings", {}).get("proposal_repair_rounds")
            != counts["original_proposal_repair_rounds"]):
        raise AdoptionError("original proposal settings or slot scope differ")
    return {"files": files, "snapshot": snapshot_id, "archive_sha256": archive_sha,
            "launch_sha256": _sha(launch_receipt), "counts": counts,
            "original_run": original_run, "source": source}


def validate_checkpoint(snapshot_dir: Path, source_archive: Path,
                        launch_receipt: Path, pilot: Path) -> dict[str, Any]:
    """Read-only, full-closure preflight for submit and worker transport."""
    checked = _validated_checkpoint(snapshot_dir, source_archive, launch_receipt, pilot)
    return {
        "schema_version": SCHEMA,
        "original_snapshot": checked["snapshot"],
        "original_source_archive_sha256": checked["archive_sha256"],
        "original_launch_receipt_sha256": checked["launch_sha256"],
        "original_controller_package_sha256": checked["original_run"]["settings"]["controller_package_sha256"],
        "slot_accounting": {k: checked["counts"][k] for k in
                            ("expected_slots", "accepted", "rejected", "null", "missing")},
    }


def validate_and_adopt(
    snapshot_dir: Path, source_archive: Path, launch_receipt: Path,
    pilot: Path, destination: Path, target_identity: dict[str, Any],
) -> dict[str, Any]:
    """Verify the original full checkpoint, then copy its proposal stage exactly once."""
    destination, pilot = Path(destination), Path(pilot)
    _ordinary_ancestors(destination)
    if any((destination / name).exists() for name in (
        "proposal", "proposal-receipt.json", "construction",
        "construction-receipt.json", "construction-progress.json",
    )):
        raise AdoptionError("adoption destination already contains proposal state")
    if _read(destination / "generate-run.json") != target_identity:
        raise AdoptionError("target generate identity differs from frozen run")
    if target_identity.get("identity_sha256") != digest({k: v for k, v in target_identity.items() if k != "identity_sha256"}):
        raise AdoptionError("target generate identity fingerprint mismatch")
    if (pilot.is_symlink() or target_identity.get("pilot_sha256") != _sha(pilot) or
            _sha(destination / "input-pilot.json") != _sha(pilot)):
        raise AdoptionError("target pilot differs from frozen input")
    checked = _validated_checkpoint(snapshot_dir, source_archive, launch_receipt, pilot)
    files, snapshot_id = checked["files"], checked["snapshot"]
    archive_sha, counts, original_run = checked["archive_sha256"], checked["counts"], checked["original_run"]
    source = checked["source"]
    source_settings = original_run["settings"]
    target_settings = target_identity.get("settings", {})
    if any(target_settings.get(name) != source_settings.get(name) for name in
           ("proposal_repair_rounds", "construction_max_repair_rounds")):
        raise AdoptionError("adoption would change original repair budgets")
    from .runtime import sha256, tree_sha256
    # All checks above are read-only. Copy into an isolated temporary directory
    # so no partially verified proposal is published as an adoption.
    temporary = destination / "proposal-adoption.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise AdoptionError("adoption temporary directory already exists")
    shutil.copytree(source, temporary)
    expected_proposal = {name.removeprefix("proposal/"): value for name, value in files.items()
                         if name.startswith("proposal/")}
    actual_proposal = {
        path.relative_to(temporary).as_posix(): sha256(path)
        for path in temporary.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if actual_proposal != expected_proposal:
        shutil.rmtree(temporary)
        raise AdoptionError("proposal changed during adoption copy")
    copied = {name: actual_proposal[name] for name in _PROPOSAL_FILES}
    tree_hash = tree_sha256(temporary)
    temporary.replace(destination / "proposal")
    receipt = {
        "schema_version": "capability-generate-v1",
        "stage": "proposal",
        "identity_sha256": target_identity["identity_sha256"],
        "exit_code": 2 if counts["missing"] or counts["rejected"] else 0,
        "accepted_count": counts["accepted"],
        "files": copied,
        "tree_sha256": tree_hash,
        "adoption": {
            "schema_version": SCHEMA,
            "original_snapshot": snapshot_id,
            "original_source_archive_sha256": archive_sha,
            "original_launch_receipt_sha256": _sha(launch_receipt),
            "original_controller_package_sha256": original_run["settings"]["controller_package_sha256"],
            "original_generate_identity_sha256": original_run.get("identity_sha256"),
            "original_proposal_repair_rounds": counts["original_proposal_repair_rounds"],
            "slot_accounting": {k: counts[k] for k in ("expected_slots", "accepted", "rejected", "null", "missing")},
            "proposal_tree_sha256": tree_hash,
            "proposal_report_sha256": copied["report.json"],
            "acceptance_reconstructed": True,
            "inference_replayed": False,
        },
    }
    atomic_json(destination / "proposal-receipt.json", receipt)
    return receipt
