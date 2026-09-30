"""Reviewed generic rootfs publication; registry credentials stay in the caller."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .generic_image_capture import _CREDENTIAL_ROOTS, _PRIVATE_ROOTS, validate_plan
from .image_review_contract import (
    REVIEW_SCHEMA,
    validate_decision,
    validate_retained_packet,
)
from .oci_artifact import validate_layer
from .rootfs_review import review_rootfs

RECEIPT_SCHEMA = "capability-task-image-publication-v1"

# Object-storage endpoints are never OCI registries. A reviewed plan naming one
# is a builder slip in `registry_host` alone: the reviewed repository, config
# and captured bytes do not depend on it. The trusted publisher therefore pushes
# to its credential's registry, keeps the reviewed repository, and records the
# correction in the receipt, so the reviewed plan bytes (and every review and
# acceptance hash bound to them) stay untouched. Any other mismatch rejects.
_NON_REGISTRY_STORAGE_HOST = re.compile(r"(?:[a-z0-9-]+\.)*cwobject\.com")
_REGISTRY_HOST = re.compile(r"[a-z0-9.-]+(?::[0-9]+)?")
REGISTRY_HOST_CORRECTION_REASON = "reviewed_plan_named_object_storage_endpoint_not_registry"


def publication_registry_host(plan_host: object, registry_host: object) -> tuple[str, dict | None]:
    """Return the host to publish to and the correction record, or reject.

    Equal hosts publish unchanged. A plan host that is a known object-storage
    endpoint is corrected to the trusted registry host. Everything else raises.
    """
    if (not isinstance(plan_host, str) or not isinstance(registry_host, str)
            or _REGISTRY_HOST.fullmatch(plan_host) is None
            or _REGISTRY_HOST.fullmatch(registry_host) is None):
        raise ValueError("registry host is invalid")
    if registry_host == plan_host:
        return registry_host, None
    if (_NON_REGISTRY_STORAGE_HOST.fullmatch(plan_host) is not None
            and _NON_REGISTRY_STORAGE_HOST.fullmatch(registry_host) is None):
        return registry_host, {"from": plan_host, "to": registry_host,
                               "reason": REGISTRY_HOST_CORRECTION_REASON}
    raise ValueError("registry host differs from reviewed plan")


def published_registry_host(plan: dict, publication: dict) -> str:
    """Host a publication receipt must name: the plan's, or its exact recorded correction."""
    if "registry_host_corrected" not in publication:
        return publication_registry_host(plan["registry_host"], plan["registry_host"])[0]
    recorded = publication["registry_host_corrected"]
    if not isinstance(recorded, dict) or set(recorded) != {"from", "to", "reason"}:
        raise ValueError("registry host correction is malformed")
    host, expected = publication_registry_host(plan["registry_host"], recorded["to"])
    if expected is None or expected != recorded:
        raise ValueError("registry host correction differs from reviewed plan")
    return host


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_review(
    approval_path: Path, plan_path: Path, *, builder_session_ids: set[str]
) -> dict:
    """Require a separate GLM review tied to the exact immutable plan bytes."""
    if approval_path.is_symlink() or plan_path.is_symlink():
        raise ValueError("image review contains a link")
    approval = json.loads(approval_path.read_text())
    if (not isinstance(approval, dict) or approval.get("schema_version") != REVIEW_SCHEMA
            or approval.get("state") != "approved" or approval.get("plan_sha256") != _sha(plan_path)
            or approval.get("decision") != "approve" or approval.get("issues") != []
            or approval.get("model") != "glm-5.3" or approval.get("independent") is not True
            or not isinstance(approval.get("snapshot_hash"), str)
            or re.fullmatch(r"[0-9a-f]{64}", approval["snapshot_hash"]) is None
            or not isinstance(approval.get("input_manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", approval["input_manifest_sha256"]) is None):
        raise ValueError("image capture lacks exact independent approval")
    session = approval.get("reviewer_session_id")
    if not isinstance(session, str) or not session or session in builder_session_ids:
        raise ValueError("image reviewer is not distinct from builder")
    artifact = approval.get("raw_artifact")
    digest = approval.get("raw_sha256")
    if (not isinstance(artifact, str) or not artifact or Path(artifact).is_absolute()
            or ".." in Path(artifact).parts or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("image review raw evidence binding is malformed")
    raw = approval_path.parent / artifact
    if raw.is_symlink() or not raw.is_file() or _sha(raw) != digest:
        raise ValueError("image review raw evidence changed")
    manifest, _ = validate_retained_packet(
        approval_path.parent,
        plan_sha256=_sha(plan_path),
        snapshot_hash=approval["snapshot_hash"],
        manifest_sha256=approval["input_manifest_sha256"],
    )
    try:
        raw_decision = json.loads(raw.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("image review raw evidence is unreadable") from error
    validate_decision(raw_decision, manifest, _sha(plan_path))
    if raw_decision["decision"] != "approve":
        raise ValueError("raw image review does not approve capture")
    return approval


def publish_generic_capture(
    *, plan_path: Path, workspace: Path, capture_tools: Path,
    approval_path: Path, builder_session_ids: set[str], capture_path: Path,
    layer_path: Path, max_uncompressed_bytes: int, registry_client: Any = None,
) -> dict:
    """Review exact captured bytes and optionally upload them by manifest digest."""
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, workspace, capture_tools)
    approval = validate_review(approval_path, plan_path, builder_session_ids=builder_session_ids)
    if capture_path.is_symlink() or layer_path.is_symlink():
        raise ValueError("capture inputs contain a link")
    capture = json.loads(capture_path.read_text())
    plan_hash = _sha(plan_path)
    if (capture.get("schema_version") != "capability-rootfs-capture-v1"
            or capture.get("state") != "captured_pending_privacy_and_publication"
            or capture.get("plan_sha256") != plan_hash
            or capture.get("cleanup", {}).get("absence_verified") is not True):
        raise ValueError("generic capture is incomplete or belongs to another plan")
    matches = [row for row in plan["images"] if row["role"] == capture.get("role")]
    if len(matches) != 1:
        raise ValueError("capture role is ambiguous")
    image = matches[0]
    from .generic_image_capture import snapshot_identity_matches

    if not snapshot_identity_matches(capture, image):
        raise ValueError("capture source snapshot changed")
    if capture.get("source_recipe_sha256") != image["source_recipe"]["sha256"]:
        raise ValueError("capture source recipe changed")
    from .generic_image_capture import _json_hash

    if capture.get("image_config_sha256") != _json_hash(image["image_config"]):
        raise ValueError("capture OCI config changed")
    archive = capture.get("capture", {})
    process = archive.get("capture", {})
    if (capture.get("archive_process", {}).get("tar_exit_class") != "clean"
            or process.get("tar_exit") != 0 or process.get("gzip_exit") != 0
            or process.get("compressed_bytes") != archive.get("object_bytes")):
        raise ValueError("capture archive process is incomplete")
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
        raise ValueError("invalid maximum rootfs size")
    with tempfile.TemporaryDirectory(prefix="reviewed-generic-image-") as temporary:
        frozen = Path(temporary) / "layer.tar.gz"
        shutil.copyfile(layer_path, frozen)
        frozen.chmod(0o400)
        with frozen.open("rb") as stream:
            layer = validate_layer(
                stream, expected_digest="sha256:" + archive["sha256"],
                expected_bytes=archive["object_bytes"],
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            stream.seek(0)
            candidate = image["role"] == "candidate"
            forbidden = list(_CREDENTIAL_ROOTS)
            if candidate:
                forbidden.extend(_PRIVATE_ROOTS)
                forbidden.extend(row["image_path"] for row in plan["private_assets"])
            review = review_rootfs(
                stream, expected_files=image["required_ready_hashes"],
                reject_paths=forbidden,
                task_roots=["/workspace", "/opt/task", "/fixtures"] if candidate else [],
                reject_hashes=[row["sha256"] for row in plan["private_assets"]] if candidate else [],
                max_file_bytes=max_uncompressed_bytes,
            )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "state": "reviewed_not_published",
            "role": image["role"],
            "plan_sha256": plan_hash,
            "review_sha256": _sha(approval_path),
            "reviewer_session_id": approval["reviewer_session_id"],
            "capture_receipt_sha256": _sha(capture_path),
            "source_snapshot": image["source_snapshot"],
            "image_config_sha256": capture["image_config_sha256"],
            "layer_digest": layer.compressed_digest,
            "rootfs_review": review,
            "cold_pull": "pending",
            "task_acceptance": "not_evaluated",
        }
        if registry_client is not None:
            if registry_client.repository != image["repository"]:
                raise ValueError("registry client target differs from reviewed plan")
            try:
                host, correction = publication_registry_host(plan["registry_host"], registry_client.host)
            except ValueError as error:
                raise ValueError("registry client target differs from reviewed plan") from error
            published = registry_client.publish_layer(
                frozen, expected_digest=layer.compressed_digest,
                expected_bytes=layer.compressed_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                image_config=image["image_config"],
                architecture=image["architecture"],
                operating_system=image["operating_system"],
            )
            prefix = host + "/" + image["repository"] + "@"
            if (published.get("state") != "integrity_verified"
                    or not isinstance(published.get("image"), str)
                    or not published["image"].startswith(prefix)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", published["image"].rpartition("@")[2]) is None
                    or published["image"].rpartition("@")[2] != published.get("manifest_digest")):
                raise ValueError("registry returned an unreviewed manifest identity")
            if correction is not None:
                receipt["registry_host_corrected"] = correction
            receipt.update(state="published_pending_cold_pull", publication=published)
        return receipt
