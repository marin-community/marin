import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from test_generic_image_capture import _fixture, _sha

from capability_pipeline import generic_image_capture as capture
from capability_pipeline.generic_image_publication import (
    publish_generic_capture,
    validate_review,
)
from capability_pipeline.image_review_contract import CHECKS
from capability_pipeline.inference import digest
from capability_pipeline.rootfs_review import RootfsReviewError


def _layer(path: Path, payload: bytes, *, credential=False):
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("workspace/public.txt")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
        if credential:
            secret = b"credential"
            entry = tarfile.TarInfo("run/secrets/token")
            entry.size = len(secret)
            archive.addfile(entry, io.BytesIO(secret))


def _artifacts(tmp_path, *, payload=b"public", role="candidate", credential=False, registry_host=None,
               authored_pointer=None):
    workspace, tools, plan, plan_path = _fixture(tmp_path)
    if registry_host is not None:
        plan["registry_host"] = registry_host
    if authored_pointer is not None:
        plan["images"][0]["authored_image_pointer"] = authored_pointer
    plan["images"][0]["role"] = role
    plan["images"][0]["required_ready_hashes"] = {
        "/workspace/public.txt": hashlib.sha256(payload).hexdigest()
    }
    plan_path.write_text(json.dumps(plan))
    review = tmp_path / "review"
    packet = review / "input" / "controller"
    packet.mkdir(parents=True)
    packet_plan = packet / "image-plan.json"
    packet_plan.write_bytes(plan_path.read_bytes())
    manifest = {
        "schema_version": "capability-quality-input-v1",
        "files": {"controller/image-plan.json": _sha(plan_path)},
    }
    manifest["snapshot_hash"] = digest(manifest)
    manifest_path = review / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    raw = review / "review-raw.json"
    raw.write_text(json.dumps({
        "plan_sha256": _sha(plan_path),
        "snapshot_hash": manifest["snapshot_hash"],
        "decision": "approve",
        "issues": [],
        "checks": [
            {"id": check, "state": "passed", "citations": [{
                "path": "controller/image-plan.json",
                "sha256": _sha(plan_path),
                "supports": "Frozen fixture plan.",
            }]}
            for check in CHECKS
        ],
    }))
    approval_path = review / "approval.json"
    approval_path.write_text(json.dumps({
        "schema_version": "capability-image-capture-review-v1",
        "state": "approved", "plan_sha256": _sha(plan_path),
        "decision": "approve", "issues": [], "model": "glm-5.3",
        "independent": True, "reviewer_session_id": "independent-review",
        "snapshot_hash": manifest["snapshot_hash"],
        "input_manifest_sha256": _sha(manifest_path),
        "raw_artifact": raw.name, "raw_sha256": _sha(raw),
    }))
    layer_path = tmp_path / "layer.tar.gz"
    _layer(layer_path, payload, credential=credential)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps({
        "schema_version": "capability-rootfs-capture-v1",
        "state": "captured_pending_privacy_and_publication", "role": role,
        "plan_sha256": _sha(plan_path),
        "source_snapshot": plan["images"][0]["source_snapshot"],
        "source_recipe_sha256": plan["images"][0]["source_recipe"]["sha256"],
        "image_config_sha256": capture._json_hash(plan["images"][0]["image_config"]),
        "cleanup": {"absence_verified": True},
        "archive_process": {"tar_exit_class": "clean"},
        "capture": {"sha256": _sha(layer_path), "object_bytes": layer_path.stat().st_size,
                    "capture": {"tar_exit": 0, "gzip_exit": 0, "compressed_bytes": layer_path.stat().st_size}},
    }))
    return workspace, tools, plan_path, approval_path, capture_path, layer_path


def test_generic_publication_reviews_exact_layer_and_binds_glm_approval(tmp_path):
    workspace, tools, plan_path, approval, captured, layer = _artifacts(tmp_path)
    receipt = publish_generic_capture(
        plan_path=plan_path, workspace=workspace, capture_tools=tools,
        approval_path=approval, builder_session_ids={"builder-1"},
        capture_path=captured, layer_path=layer, max_uncompressed_bytes=1_000_000,
    )
    assert receipt["state"] == "reviewed_not_published"
    assert receipt["rootfs_review"]["state"] == "passed"
    assert receipt["cold_pull"] == "pending"


def test_generic_publication_rejects_private_bytes_at_public_path(tmp_path):
    workspace, tools, plan_path, approval, captured, layer = _artifacts(tmp_path, payload=b"private")
    with pytest.raises(RootfsReviewError, match="private content"):
        publish_generic_capture(
            plan_path=plan_path, workspace=workspace, capture_tools=tools,
            approval_path=approval, builder_session_ids={"builder-1"},
            capture_path=captured, layer_path=layer, max_uncompressed_bytes=1_000_000,
        )


def test_generic_publication_rejects_builder_self_review(tmp_path):
    _, _, plan_path, approval, _, _ = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        validate_review(approval, plan_path, builder_session_ids={"independent-review"})


def test_generic_publication_rejects_hash_bound_raw_without_checks(tmp_path):
    _, _, plan_path, approval, _, _ = _artifacts(tmp_path)
    raw = approval.parent / "review-raw.json"
    raw.write_text('{"decision":"approve"}')
    document = json.loads(approval.read_text())
    document["raw_sha256"] = _sha(raw)
    approval.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="input identity|not an object"):
        validate_review(approval, plan_path, builder_session_ids=set())


def test_private_verifier_publication_rejects_credential_path(tmp_path):
    workspace, tools, plan_path, approval, captured, layer = _artifacts(
        tmp_path, role="private_verifier", credential=True
    )
    with pytest.raises(RootfsReviewError, match="private path|runtime or provider state"):
        publish_generic_capture(
            plan_path=plan_path, workspace=workspace, capture_tools=tools,
            approval_path=approval, builder_session_ids={"builder-1"},
            capture_path=captured, layer_path=layer, max_uncompressed_bytes=1_000_000,
        )


@pytest.mark.parametrize("decision", ["repair", "reject"])
def test_approved_summary_cannot_override_raw_nonapproval(tmp_path, decision):
    _, _, plan_path, approval, _, _ = _artifacts(tmp_path)
    raw = approval.parent / "review-raw.json"
    document = json.loads(raw.read_text())
    document.update(decision=decision, issues=["Image provenance is incomplete."])
    raw.write_text(json.dumps(document))
    summary = json.loads(approval.read_text())
    summary["raw_sha256"] = _sha(raw)
    approval.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="does not approve"):
        validate_review(approval, plan_path, builder_session_ids=set())
