"""Immutable, byte-bound delivery closures for private verifier replay."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .grading_input import grading_input_fingerprint

SCHEMA = "capability-fixed-grading-capture-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _safe_relative(value: str) -> Path:
    # Do not let ``Path`` normalize a malicious spelling before we inspect it.
    # Captures are portable POSIX trees, even when a controller happens to run
    # on another host.
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("capture member path is unsafe")
    path = Path(*value.split("/"))
    return path


def _copy_tree(source: Path, target: Path) -> list[dict]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("capture workspace must be a real directory")
    entries: list[dict] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError("capture workspace contains an unsafe member")
        destination = target / relative
        if stat.S_ISDIR(mode):
            destination.mkdir(parents=True, exist_ok=False)
            os.chmod(destination, mode & 0o777)
            entries.append(
                {"path": relative, "kind": "directory", "mode": mode & 0o777}
            )
        else:
            data = path.read_bytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            os.chmod(destination, mode & 0o777)
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode & 0o777,
                    "size": len(data),
                    "sha256": _sha(data),
                }
            )
    return entries


def _members(root: Path) -> list[dict]:
    """Describe the complete captured tree, excluding its signed manifest."""
    result: list[dict] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError("capture workspace contains an unsafe member")
        if stat.S_ISDIR(mode):
            result.append({"path": relative, "kind": "directory", "mode": mode & 0o777})
        else:
            data = path.read_bytes()
            result.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode & 0o777,
                    "size": len(data),
                    "sha256": _sha(data),
                }
            )
    return result


def _reject_link_ancestors(root: Path, relative: Path) -> None:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("capture member absent or linked")


def _delivery_fingerprint(captured: dict, *, workspace: Path | None = None) -> dict:
    """Recompute the first-delivery fingerprint from the persisted raw bytes."""
    payload = json.loads(captured["payload"])
    if not isinstance(payload, dict) or set(payload) != {
        "specification", "protocol", "step_index", "attempt"
    }:
        raise ValueError("captured grading payload has an invalid shape")
    attempt = payload["attempt"]
    if not isinstance(attempt, dict) or set(attempt) != {"response", "transcript"}:
        raise ValueError("captured grading payload has an invalid attempt")
    if payload["attempt"]["response"] != captured["response"]:
        raise ValueError("captured payload response differs from captured response")
    if _json(payload["attempt"]["transcript"]) != _json(captured["transcript"]):
        raise ValueError("captured payload transcript differs from captured transcript")
    if json.loads(captured["specification"]) != payload["specification"]:
        raise ValueError("captured payload specification differs from raw specification")
    if json.loads(captured["protocol"]) != payload["protocol"]:
        raise ValueError("captured payload protocol differs from raw protocol")
    step_index = payload["step_index"]
    if type(step_index) is not int or step_index < 0:
        raise ValueError("captured grading payload has an invalid step index")
    return grading_input_fingerprint(
        captured["specification"],
        payload["protocol"],
        captured["response"],
        workspace or captured["workspace"],
        captured["transcript"],
        step_index=step_index,
        payload=captured["payload"],
    )


def materialize_workspace(captured: dict, destination: Path) -> Path:
    """Recreate the manifest-bound workspace from portable stored blobs."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("capture materialization destination must be fresh")
    destination.mkdir(parents=True)
    logical = [
        member
        for member in captured["manifest"]["members"]
        if member["path"].startswith("workspace/")
    ]
    directories = sorted(
        (member for member in logical if member["kind"] == "directory"),
        key=lambda member: (member["path"].count("/"), member["path"]),
    )
    for member in directories:
        relative = _safe_relative(member["path"]).relative_to("workspace")
        (destination / relative).mkdir(parents=True, exist_ok=False)
    for member in logical:
        if member["kind"] != "file":
            continue
        relative = _safe_relative(member["path"]).relative_to("workspace")
        source = captured["root"] / _safe_relative(member["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        os.chmod(target, member["mode"])
    for member in directories:
        relative = _safe_relative(member["path"]).relative_to("workspace")
        os.chmod(destination / relative, member["mode"])
    return destination


def write_capture(
    root: Path,
    *,
    specification: bytes,
    protocol: bytes,
    response: str | None,
    transcript: Any,
    payload: bytes,
    workspace: Path,
    fingerprint: dict,
    source_specification_sha256: str,
    source_renderings_sha256: str,
) -> dict:
    """Write one fresh closure after candidate evidence is downloaded, before grading."""
    if root.exists() or root.is_symlink():
        raise FileExistsError("fixed grading capture destination must be fresh")
    if response is not None and not isinstance(response, str):
        raise TypeError("captured response must be text or null")
    if not isinstance(specification, bytes) or not isinstance(protocol, bytes):
        raise TypeError("captured specification and protocol must be raw bytes")
    if not isinstance(payload, bytes) or not isinstance(fingerprint, dict):
        raise TypeError("captured payload and fingerprint must be bytes and an object")
    if not all(_is_sha256(item) for item in (
        source_specification_sha256, source_renderings_sha256
    )):
        raise ValueError("capture source hashes must be SHA-256 strings")
    root.mkdir(parents=True)
    files = {
        "specification.json": specification,
        "protocol.json": protocol,
        "payload.json": payload,
        "transcript.json": _json(transcript),
    }
    if response is not None:
        files["response.txt"] = response.encode()
    members = []
    for name, data in files.items():
        (root / name).write_bytes(data)
        os.chmod(root / name, 0o644)
        members.append(
            {
                "path": name,
                "kind": "file",
                "mode": 0o644,
                "size": len(data),
                "sha256": _sha(data),
            }
        )
    workspace_root = root / "workspace"
    workspace_root.mkdir()
    members.append({"path": "workspace", "kind": "directory", "mode": 0o755})
    for entry in _copy_tree(workspace, workspace_root):
        members.append({**entry, "path": f"workspace/{entry['path']}"})
    captured = {
        "specification": specification,
        "protocol": protocol,
        "payload": payload,
        "transcript": transcript,
        "response": response,
        "workspace": workspace_root,
    }
    actual_fingerprint = _delivery_fingerprint(captured)
    if actual_fingerprint != fingerprint:
        raise ValueError("capture fingerprint does not bind the supplied delivery")
    manifest = {
        "schema_version": SCHEMA,
        "response_present": response is not None,
        "fingerprint": fingerprint,
        "source_specification_sha256": source_specification_sha256,
        "source_renderings_sha256": source_renderings_sha256,
        # Re-read the completed tree so this list is both complete and ordered
        # exactly as a later verifier will inspect it.
        "members": _members(root),
    }
    (root / "manifest.json").write_bytes(_json(manifest) + b"\n")
    return {"manifest_sha256": _sha((root / "manifest.json").read_bytes()), **manifest}


def load_capture(root: Path, *, expected_manifest_sha256: str | None = None) -> dict:
    """Verify and return a closure without importing TaskCompendium."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("capture root must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("capture manifest is absent or linked")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("capture manifest hash differs from expected identity")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA or not isinstance(
        manifest.get("members"), list
    ):
        raise ValueError("unsupported capture manifest")
    if not all(
        _is_sha256(manifest.get(key))
        for key in ("source_specification_sha256", "source_renderings_sha256")
    ) or not isinstance(manifest.get("fingerprint"), dict):
        raise ValueError("capture manifest identity fields are invalid")
    listed = set()
    expected: dict[str, dict] = {}
    for member in manifest["members"]:
        if not isinstance(member, dict):
            raise TypeError("invalid capture member")
        relative = _safe_relative(member.get("path", ""))
        key = relative.as_posix()
        if key in listed:
            raise ValueError("duplicate capture member")
        listed.add(key)
        expected[key] = member
        path = root / relative
        _reject_link_ancestors(root, relative)
        workspace_member = key == "workspace" or key.startswith("workspace/")
        if path.is_symlink():
            raise ValueError("capture member absent or linked")
        if not path.exists():
            # Empty directories are not retained by the generic object-store
            # synchronizer.  They are rebuilt from this signed logical tree.
            if workspace_member and member.get("kind") == "directory":
                continue
            raise ValueError("capture member absent or linked")
        if member.get("kind") == "directory":
            if not path.is_dir():
                raise ValueError("capture directory changed")
        elif member.get("kind") == "file":
            data = path.read_bytes()
            if (
                not path.is_file()
                or member.get("size") != len(data)
                or member.get("sha256") != _sha(data)
            ):
                raise ValueError("capture file changed")
        else:
            raise ValueError("capture member kind is invalid")
    for key, member in expected.items():
        relative = _safe_relative(key)
        if key == "workspace":
            if member.get("kind") != "directory":
                raise ValueError("capture workspace must be a directory")
            continue
        parent = relative.parent.as_posix()
        if parent == ".":
            continue
        parent_member = expected.get(parent)
        if parent_member is None or parent_member.get("kind") != "directory":
            raise ValueError("capture member parent is not a declared directory")
    for actual in _members(root):
        member = expected.get(actual["path"])
        if member is None or member.get("kind") != actual["kind"]:
            raise ValueError("capture tree differs from manifest")
        if actual["kind"] == "file" and (
            actual.get("size") != member.get("size")
            or actual.get("sha256") != member.get("sha256")
        ):
            raise ValueError("capture file changed")
    required = {
        "specification.json",
        "protocol.json",
        "payload.json",
        "transcript.json",
        "workspace",
    }
    if not required.issubset(listed) or bool(manifest.get("response_present")) != (
        "response.txt" in listed
    ):
        raise ValueError("capture required members differ")
    captured = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "root": root,
        "workspace": root / "workspace",
        "specification": (root / "specification.json").read_bytes(),
        "protocol": (root / "protocol.json").read_bytes(),
        "payload": (root / "payload.json").read_bytes(),
        "transcript": json.loads((root / "transcript.json").read_text()),
        "response": (root / "response.txt").read_text()
        if manifest["response_present"]
        else None,
    }
    with tempfile.TemporaryDirectory(prefix="fixed-grading-capture-check-") as temporary:
        workspace = materialize_workspace(captured, Path(temporary) / "workspace")
        if _delivery_fingerprint(captured, workspace=workspace) != manifest.get("fingerprint"):
            raise ValueError("capture fingerprint differs from persisted delivery")
    return captured
