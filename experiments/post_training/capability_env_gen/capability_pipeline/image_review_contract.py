"""Portable validation for evidence-bound generic image plan reviews."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .inference import digest

REVIEW_SCHEMA = "capability-image-capture-review-v1"
CHECKS = frozenset({
    "source_closure", "role_separation", "image_configuration",
    "capture_lifecycle", "provenance",
})


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_input_manifest(manifest: object, *, plan_sha256: str) -> dict:
    """Validate the persisted review packet before trusting any citation."""
    if not isinstance(manifest, dict):
        raise TypeError("image review input manifest is not an object")
    snapshot_hash = manifest.get("snapshot_hash")
    without_snapshot = {key: value for key, value in manifest.items() if key != "snapshot_hash"}
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != "capability-quality-input-v1"
        or not isinstance(snapshot_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot_hash) is None
        or snapshot_hash != digest(without_snapshot)
        or not isinstance(files, dict)
        or files.get("controller/image-plan.json") != plan_sha256
    ):
        raise ValueError("image review input manifest differs from its frozen plan")
    for relative, value in files.items():
        path = Path(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or ".." in path.parts
            or not relative
            or not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise ValueError("image review input manifest has an unsafe file record")
    return manifest


def validate_decision(raw: object, manifest: dict, plan_sha256: str) -> None:
    """Require a complete cited verdict, not a controller's summary claim."""
    if not isinstance(raw, dict):
        raise TypeError("image review decision is not an object")
    if raw.get("plan_sha256") != plan_sha256 or raw.get("snapshot_hash") != manifest["snapshot_hash"]:
        raise ValueError("image review input identity differs")
    if raw.get("decision") not in {"approve", "repair", "reject"}:
        raise ValueError("invalid image plan decision")
    issues = raw.get("issues")
    if not isinstance(issues, list) or any(not isinstance(x, str) or not x for x in issues):
        raise ValueError("invalid image plan issues")
    checks = raw.get("checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(CHECKS)
        or any(not isinstance(row, dict) for row in checks)
        or {row.get("id") for row in checks} != CHECKS
    ):
        raise ValueError("image review lacks required checks")
    for row in checks:
        if row.get("state") not in {"passed", "failed", "missing"}:
            raise ValueError("invalid image plan check state")
        citations = row.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError("image plan check has no evidence")
        for citation in citations:
            if (
                not isinstance(citation, dict)
                or citation.get("path") not in manifest["files"]
                or citation.get("sha256") != manifest["files"][citation["path"]]
                or not isinstance(citation.get("supports"), str)
                or not citation["supports"].strip()
            ):
                raise ValueError("image plan citation is unbound")
    if raw["decision"] == "approve":
        if issues or any(row["state"] != "passed" for row in checks):
            raise ValueError("image plan approval contradicts findings")
    elif not issues:
        raise ValueError("nonapproving image review has no repair feedback")


def validate_retained_packet(
    review_root: Path, *, plan_sha256: str, snapshot_hash: str, manifest_sha256: str
) -> tuple[dict, dict]:
    """Rehash the complete frozen packet and validate its cited approval decision."""
    review_root = Path(review_root)
    manifest_path = review_root / "input-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or sha256(manifest_path) != manifest_sha256:
        raise ValueError("image review input manifest changed")
    manifest = validate_input_manifest(json.loads(manifest_path.read_text()), plan_sha256=plan_sha256)
    if manifest["snapshot_hash"] != snapshot_hash:
        raise ValueError("image review snapshot differs from approval")
    packet = review_root / "input"
    for relative, expected in manifest["files"].items():
        path = packet / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError("image review frozen input changed")
    return manifest, packet
