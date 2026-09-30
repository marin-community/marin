"""Credential-free, relocatable handoff for an isolated task-image publisher.

Two entry shapes share one validator:

* ``*_for_item`` functions take the controller's live pending-publication
  record (``process_image_construction``'s ``pending_publication`` dict, the
  same object synthesis stores as ``status["custom_images"]``) plus the item
  root. The automatic publication exchange uses these from inside the job.
* ``export_handoff`` / ``import_publication`` keep the original status.json
  operator interface and require ``state == pending_image_publication``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path

from .generic_image_capture import _source, snapshot_identity_matches, validate_plan
from .generic_image_cold_pull import _checked_publication
from .generic_image_publication import validate_review
from .image_review_contract import validate_retained_packet

SCHEMA = "capability-image-publication-handoff-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("handoff source is missing or linked")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("handoff source must be an object")
    return value


def _under(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("handoff path leaves its controller-owned root")
    return path


def deterministic_targz(files: dict[str, bytes]) -> bytes:
    """Byte-stable tar.gz: fixed member order, metadata and gzip header.

    Content addressing (the publication queue keys packets by SHA-256) needs
    the same reviewed inputs to yield the same archive bytes on every pass.
    """
    raw = io.BytesIO()
    with (gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as zipped,
          tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive):
        for name, value in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            entry.mode = 0o444
            entry.mtime = 0
            archive.addfile(entry, io.BytesIO(value))
    return raw.getvalue()


def _inputs_for_item(item: Path, images: object, declared_sessions: set[str] | None = None,
                     ) -> tuple[dict, dict, Path, Path, Path, Path]:
    """Validate a live pending-publication record against its frozen attempt."""
    if not isinstance(images, dict) or images.get("state") != "pending_publication":
        raise ValueError("item is not pending reviewed image publication")
    plan_path = _under(item, Path(images["plan_path"]))
    workspace = _under(item, Path(images["workspace"]))
    if not plan_path.is_relative_to(item / "diagnostics/image-capture"):
        raise ValueError("handoff plan is outside its frozen attempt")
    if workspace != plan_path.parent / "workspace":
        raise ValueError("handoff workspace differs from frozen plan")
    capture_tools = Path(images["capture_tools"])
    if capture_tools.is_symlink() or not capture_tools.is_dir():
        raise ValueError("staged capture tools are unavailable")
    approval_path = Path(images["approval_path"])
    if approval_path.is_symlink() or approval_path.name != "approval.json":
        raise ValueError("handoff approval path is invalid")
    plan = _read(plan_path)
    validate_plan(plan, workspace, capture_tools)
    session_ids = images.get("builder_session_ids")
    if (not isinstance(session_ids, list)
            or any(not isinstance(value, str) or not value for value in session_ids)
            or not (declared_sessions or set()).issubset(session_ids)):
        raise ValueError("handoff lacks exact builder session identities")
    sessions = set(session_ids)
    approval = validate_review(approval_path, plan_path, builder_session_ids=sessions)
    manifest, _ = validate_retained_packet(
        approval_path.parent, plan_sha256=_sha(plan_path.read_bytes()),
        snapshot_hash=approval["snapshot_hash"],
        manifest_sha256=approval["input_manifest_sha256"],
    )
    roles = {row["role"] for row in plan["images"]}
    if set(images.get("capture_paths", {})) != roles or set(images.get("publication_paths", {})) != roles:
        raise ValueError("handoff role paths differ from reviewed plan")
    for role in roles:
        expected = plan_path.parent.parent / f"capture-{role}.json"
        if _under(item, Path(images["capture_paths"][role])) != expected:
            raise ValueError("handoff capture path differs from frozen attempt")
        publication = _under(item, Path(images["publication_paths"][role]))
        if publication != plan_path.parent.parent / f"publication-{role}.json":
            raise ValueError("handoff publication path differs from frozen attempt")
    return images, manifest, plan_path, workspace, capture_tools, approval_path


def _inputs(status_path: Path) -> tuple[dict, dict, Path, Path, Path, Path, Path]:
    status = _read(status_path)
    if status.get("state") != "pending_image_publication":
        raise ValueError("item is not pending reviewed image publication")
    item = _under(status_path.parent, Path(status["item_root"]))
    declared = {str(row["session"]) for row in status.get("sessions", [])}
    _, manifest, plan_path, workspace, tools, approval = _inputs_for_item(
        item, status.get("custom_images"), declared)
    return status, manifest, item, plan_path, workspace, tools, approval


def _members_for_item(item: Path, images: object, declared_sessions: set[str] | None = None,
                      ) -> tuple[dict[str, bytes], dict]:
    images, review_manifest, plan_path, workspace, tools, approval_path = _inputs_for_item(
        item, images, declared_sessions)
    plan = _read(plan_path)
    files = {"plan.json": plan_path.read_bytes(),
             "review/approval.json": approval_path.read_bytes(),
             "review/input-manifest.json": (approval_path.parent / "input-manifest.json").read_bytes(),
             "tools/dt.py": (tools.parent / "dt.py").read_bytes()}
    approval = _read(approval_path)
    raw = approval["raw_artifact"]
    files["review/" + raw] = (approval_path.parent / raw).read_bytes()
    for relative in review_manifest["files"]:
        files["review/input/" + relative] = (approval_path.parent / "input" / relative).read_bytes()
    for row in plan["source_files"]:
        source = _source(workspace, row["path"], row["sha256"])
        files["workspace/" + row["path"]] = source.read_bytes()
    for name in ("capture_rootfs.py", "in_sandbox_capture.py", "dtx.py", "cw_presign.py"):
        path = tools / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("capture tool changed during handoff")
        files["tools/capture-tools/" + name] = path.read_bytes()
    capture_hashes = {}
    for image in plan["images"]:
        role = image["role"]
        path = _under(item, Path(images["capture_paths"][role]))
        capture = _read(path)
        if (capture.get("schema_version") != "capability-rootfs-capture-v1"
                or capture.get("state") != "captured_pending_privacy_and_publication"
                or capture.get("role") != role or capture.get("plan_sha256") != _sha(files["plan.json"])
                or capture.get("cleanup", {}).get("absence_verified") is not True
                or not snapshot_identity_matches(capture, image)):
            raise ValueError("capture receipt is incomplete or differs from plan")
        files[f"captures/{role}.json"] = path.read_bytes()
        capture_hashes[role] = _sha(files[f"captures/{role}.json"])
    meta = {"schema_version": SCHEMA, "roles": sorted(capture_hashes),
            "builder_session_ids": sorted(images["builder_session_ids"]),
            "plan_sha256": _sha(files["plan.json"]),
            "approval_sha256": _sha(files["review/approval.json"]),
            "capture_sha256": capture_hashes,
            "files": {name: _sha(value) for name, value in sorted(files.items())}}
    files["manifest.json"] = json.dumps(meta, sort_keys=True, indent=2).encode() + b"\n"
    return files, meta


def handoff_packet(item_root: Path, images: object, *,
                   declared_sessions: set[str] | None = None) -> dict:
    """Freeze exact reviewed metadata into deterministic, credential-free bytes.

    Returns ``archive`` (the tar.gz bytes), ``packet_sha256`` (their SHA-256,
    the queue key), ``manifest_sha256`` and ``roles``. No layer bytes.
    """
    files, manifest = _members_for_item(Path(item_root), images, declared_sessions)
    data = deterministic_targz(files)
    return {"archive": data, "packet_sha256": _sha(data),
            "manifest_sha256": _sha(files["manifest.json"]), "roles": manifest["roles"]}


def _write_new(archive_path: Path, data: bytes) -> None:
    if archive_path.exists() or archive_path.is_symlink():
        raise ValueError("handoff archive already exists")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as stream:
        stream.write(data)


def export_handoff_for_item(item_root: Path, images: object, archive_path: Path) -> dict:
    """Write the deterministic packet for a live pending-publication record."""
    if archive_path.exists() or archive_path.is_symlink():
        raise ValueError("handoff archive already exists")
    packet = handoff_packet(item_root, images)
    _write_new(archive_path, packet["archive"])
    return {"state": "exported", "archive": str(archive_path),
            "archive_sha256": packet["packet_sha256"],
            "manifest_sha256": packet["manifest_sha256"], "roles": packet["roles"]}


def export_handoff(status_path: Path, archive_path: Path) -> dict:
    """Freeze exact reviewed metadata, without layer bytes or credentials."""
    if archive_path.exists() or archive_path.is_symlink():
        raise ValueError("handoff archive already exists")
    status, _, item, *_ = _inputs(status_path)
    declared = {str(row["session"]) for row in status.get("sessions", [])}
    packet = handoff_packet(item, status["custom_images"], declared_sessions=declared)
    _write_new(archive_path, packet["archive"])
    return {"state": "exported", "archive": str(archive_path),
            "archive_sha256": packet["packet_sha256"],
            "manifest_sha256": packet["manifest_sha256"], "roles": packet["roles"]}


def unpack_handoff(archive_path: Path, output: Path) -> dict:
    """Verify and materialize a publication packet at a relocated path."""
    if output.exists() or output.is_symlink():
        raise ValueError("handoff destination already exists")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("handoff archive has duplicate or absent members")
        if any(not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts
               for member in members):
            raise ValueError("handoff archive has unsafe members")
        data = {member.name: archive.extractfile(member).read() for member in members}
    manifest = json.loads(data["manifest.json"])
    if (manifest.get("schema_version") != SCHEMA or set(data) != set(manifest.get("files", {})) | {"manifest.json"}
            or any(_sha(data[name]) != expected for name, expected in manifest["files"].items())):
        raise ValueError("handoff archive differs from its file manifest")
    output.mkdir(parents=True)
    for name, value in data.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        path.chmod(0o444)
    plan = _read(output / "plan.json")
    validate_plan(plan, output / "workspace", output / "tools/capture-tools")
    validate_review(output / "review/approval.json", output / "plan.json",
                    builder_session_ids=set(manifest["builder_session_ids"]))
    return {"state": "unpacked", "root": str(output), "roles": manifest["roles"],
            "manifest_sha256": _sha(data["manifest.json"])}


def _import_checked(item: Path, images: dict, plan_path: Path, approval_path: Path,
                    archive_path: Path, role: str, publication_path: Path) -> dict:
    if role not in images["publication_paths"]:
        raise ValueError("publication role is not pending")
    target = _under(item, Path(images["publication_paths"][role]))
    if target.exists() or target.is_symlink():
        raise ValueError("publication target already exists")
    with tempfile.TemporaryDirectory(prefix="image-publication-handoff-") as temporary:
        packet = Path(temporary) / "packet"
        unpack_handoff(archive_path, packet)
        manifest = _read(packet / "manifest.json")
        if (manifest["plan_sha256"] != _sha(plan_path.read_bytes())
                or manifest["approval_sha256"] != _sha(approval_path.read_bytes())
                or role not in manifest["roles"]):
            raise ValueError("publication handoff differs from live controller input")
        capture_path = Path(images["capture_paths"][role])
        if manifest["capture_sha256"][role] != _sha(capture_path.read_bytes()):
            raise ValueError("captured input changed after handoff")
        publication = _read(publication_path)
        image, reference = _checked_publication(_read(plan_path), plan_path, publication_path)
        capture = _read(capture_path)
        if (image["role"] != role
                or publication.get("review_sha256") != _sha(approval_path.read_bytes())
                or publication.get("capture_receipt_sha256") != manifest["capture_sha256"][role]
                or publication.get("source_snapshot") != image["source_snapshot"]
                or publication.get("image_config_sha256") != capture.get("image_config_sha256")
                or publication.get("layer_digest") != "sha256:" + capture["capture"]["sha256"]):
            raise ValueError("publication receipt differs from captured reviewed bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(publication_path.read_bytes())
    return {"state": "imported", "role": role, "image": reference,
            "publication_sha256": _sha(target.read_bytes()), "target": str(target)}


def import_publication_for_item(item_root: Path, images: object, archive_path: Path,
                                role: str, publication_path: Path) -> dict:
    """Install an exact reviewed receipt for a live pending-publication record."""
    item = Path(item_root)
    images, _, plan_path, _, _, approval_path = _inputs_for_item(item, images)
    return _import_checked(item, images, plan_path, approval_path, archive_path, role, publication_path)


def import_publication(status_path: Path, archive_path: Path, role: str,
                       publication_path: Path) -> dict:
    """Install only an exact reviewed publisher receipt at its controller path."""
    status, _, item, plan_path, _, _, approval_path = _inputs(status_path)
    return _import_checked(item, status["custom_images"], plan_path, approval_path,
                           archive_path, role, publication_path)
