"""Bind captured task bytes to review evidence before registry publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from capability_pipeline.image_publication_contract import validate_publication_contract
from capability_pipeline.oci_artifact import validate_layer
from capability_pipeline.rootfs_review import review_rootfs


class PublicationEvidenceError(ValueError):
    pass


def publish_captured_image(
    *,
    plan_path: Path,
    workspace: Path,
    capture_path: Path,
    layer_path: Path,
    approved_plan_sha256: str,
    max_uncompressed_bytes: int,
    registry_client=None,
) -> dict:
    """Review a private copy, then optionally publish those exact reviewed bytes.

    The caller downloads the content-addressed S3 object into the trusted worker.
    No credentials are required for review-only execution. Provider cold pulls and
    the complete task gates remain necessary after this function returns.
    """
    plan_bytes, capture_bytes = plan_path.read_bytes(), capture_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    validate_publication_contract(plan, workspace=workspace)
    if plan_digest != approved_plan_sha256 or plan["state"] != "approved_for_capture":
        raise PublicationEvidenceError(
            "publication requires exact approved capture plan"
        )
    implementation = plan.get("capture_implementation", {})
    for name in ("oci_artifact", "oci_registry", "rootfs_review", "image_publication"):
        actual = hashlib.sha256(
            Path(__file__).with_name(name + ".py").read_bytes()
        ).hexdigest()
        if implementation.get(name + "_sha256") != actual:
            raise PublicationEvidenceError(
                "publication implementation differs from plan"
            )
    capture = json.loads(capture_bytes)
    if (
        capture.get("plan_sha256") != plan_digest
        or capture.get("state") != "captured_pending_privacy_and_publication"
    ):
        raise PublicationEvidenceError("capture is not bound to approved plan")
    images = [image for image in plan["images"] if image["role"] == capture.get("role")]
    if len(images) != 1:
        raise PublicationEvidenceError("capture role is ambiguous")
    image = images[0]
    for key in ("name", "id", "ref"):
        if capture.get("source_snapshot", {}).get(key) != image["source_snapshot"][key]:
            raise PublicationEvidenceError("capture source snapshot drift")
    if capture.get("cleanup", {}).get("absence_verified") is not True:
        raise PublicationEvidenceError("owned capture sandbox cleanup is unverified")
    captured = capture.get("capture", {})
    transport = captured.get("capture", {})
    if transport.get("tar_exit") != 0 or transport.get("gzip_exit") != 0:
        raise PublicationEvidenceError("rootfs capture did not exit cleanly")
    if transport.get("compressed_bytes") != captured.get("object_bytes"):
        raise PublicationEvidenceError("capture size evidence disagrees")
    boundary = image["privacy_boundary"]
    candidate = image["role"] == "candidate"
    with tempfile.TemporaryDirectory(prefix="reviewed-task-image-") as temporary:
        frozen = Path(temporary) / "layer.tar.gz"
        shutil.copyfile(layer_path, frozen)
        frozen.chmod(0o400)
        with frozen.open("rb") as stream:
            layer = validate_layer(
                stream,
                expected_digest="sha256:" + captured.get("sha256", ""),
                expected_bytes=captured.get("object_bytes"),
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            stream.seek(0)
            review = review_rootfs(
                stream,
                expected_files=image.get("required_ready_hashes", {}),
                reject_paths=boundary["reject_paths"],
                task_roots=["/workspace", "/opt/task", "/fixtures"]
                if candidate
                else [],
                content_markers=boundary.get("required_marker_scan", [])
                if candidate
                else [],
                max_file_bytes=max_uncompressed_bytes,
            )
        receipt = {
            "schema_version": "capability-task-image-publication-v1",
            "state": "reviewed_not_published",
            "role": image["role"],
            "plan_sha256": plan_digest,
            "capture_receipt_sha256": hashlib.sha256(capture_bytes).hexdigest(),
            "source_snapshot": image["source_snapshot"],
            "layer_digest": layer.compressed_digest,
            "rootfs_review": review,
            "cold_pull": "pending",
            "task_acceptance": "not_evaluated",
        }
        if registry_client is not None:
            if registry_client.repository != image["repository"]:
                raise PublicationEvidenceError("publisher repository differs from plan")
            published = registry_client.publish_layer(
                frozen,
                expected_digest=layer.compressed_digest,
                expected_bytes=layer.compressed_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                image_config=image["image_config"],
                architecture=image["architecture"],
                operating_system=image["operating_system"],
            )
            receipt.update(state="published_pending_cold_pull", publication=published)
        return receipt
