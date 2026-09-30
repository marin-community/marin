import json
import shutil

import pytest
from test_generic_image_publication import _artifacts, _sha

from capability_pipeline.image_publication_handoff import (
    export_handoff,
    import_publication,
    unpack_handoff,
)


def _fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    workspace, tools, plan, approval, capture, _ = _artifacts(source)
    item = tmp_path / "items/task"
    attempt = item / "diagnostics/image-capture/attempt-source"
    frozen = attempt / "input/workspace"
    frozen.parent.mkdir(parents=True)
    shutil.copytree(workspace, frozen)
    plan_path = attempt / "input/plan.json"
    shutil.copy2(plan, plan_path)
    capture_path = attempt / "capture-candidate.json"
    shutil.copy2(capture, capture_path)
    target = attempt / "publication-candidate.json"
    item.mkdir(parents=True, exist_ok=True)
    status = item / "status.json"
    status.write_text(json.dumps({
        "state": "pending_image_publication", "item_root": str(item),
        "sessions": [{"session": "builder-1"}],
        "custom_images": {
            "builder_session_ids": ["builder-1", "builder-transcript-id"],
            "state": "pending_publication", "plan_path": str(plan_path),
            "workspace": str(frozen), "capture_tools": str(tools),
            "approval_path": str(approval),
            "capture_paths": {"candidate": str(capture_path)},
            "publication_paths": {"candidate": str(target)},
        },
    }))
    archive = tmp_path / "handoff.tar.gz"
    return status, archive, plan_path, approval, capture_path, target


def _publication(path, plan_path, approval, capture):
    plan = json.loads(plan_path.read_text())
    image = plan["images"][0]
    reference = "registry.example/" + image["repository"] + "@sha256:" + "a" * 64
    path.write_text(json.dumps({
        "schema_version": "capability-task-image-publication-v1",
        "state": "published_pending_cold_pull", "role": "candidate",
        "plan_sha256": _sha(plan_path), "review_sha256": _sha(approval),
        "capture_receipt_sha256": _sha(capture),
        "source_snapshot": image["source_snapshot"],
        "image_config_sha256": json.loads(capture.read_text())["image_config_sha256"],
        "layer_digest": "sha256:" + json.loads(capture.read_text())["capture"]["sha256"],
        "rootfs_review": {"state": "passed", "required_file_hashes": image["required_ready_hashes"]},
        "publication": {"state": "integrity_verified", "image": reference,
                        "manifest_digest": "sha256:" + "a" * 64},
    }))


def test_export_unpack_and_import_exact_publication(tmp_path):
    status, archive, plan, approval, capture, target = _fixture(tmp_path)
    exported = export_handoff(status, archive)
    assert exported["state"] == "exported"
    packet = tmp_path / "relocated"
    unpacked = unpack_handoff(archive, packet)
    assert unpacked["roles"] == ["candidate"]
    assert json.loads((packet / "manifest.json").read_text())["builder_session_ids"] == [
        "builder-1", "builder-transcript-id"
    ]
    assert (packet / "review/approval.json").read_bytes() == approval.read_bytes()
    assert (packet / "captures/candidate.json").read_bytes() == capture.read_bytes()
    assert not any(path.name.endswith("layer.tar.gz") for path in packet.rglob("*"))
    publication = tmp_path / "returned-publication.json"
    _publication(publication, plan, approval, capture)
    result = import_publication(status, archive, "candidate", publication)
    assert result["state"] == "imported"
    assert target.read_bytes() == publication.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        import_publication(status, archive, "candidate", publication)


def test_import_rejects_changed_capture_or_unreviewed_publication(tmp_path):
    status, archive, plan, approval, capture, target = _fixture(tmp_path)
    export_handoff(status, archive)
    publication = tmp_path / "returned-publication.json"
    _publication(publication, plan, approval, capture)
    altered = json.loads(publication.read_text())
    altered["capture_receipt_sha256"] = "0" * 64
    publication.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="captured reviewed bytes"):
        import_publication(status, archive, "candidate", publication)
    assert not target.exists()
    _publication(publication, plan, approval, capture)
    capture.write_text(capture.read_text() + " ")
    with pytest.raises(ValueError, match="captured input changed"):
        import_publication(status, archive, "candidate", publication)
    assert not target.exists()


def test_export_rejects_non_pending_status(tmp_path):
    status, archive, *_ = _fixture(tmp_path)
    value = json.loads(status.read_text())
    value["state"] = "ready"
    status.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not pending"):
        export_handoff(status, archive)
    assert not archive.exists()
