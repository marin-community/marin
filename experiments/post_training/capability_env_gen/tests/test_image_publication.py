import hashlib
import json
from pathlib import Path

import pytest
from test_image_publication_contract import contract
from test_rootfs_review import archive

from capability_pipeline.image_publication import (
    PublicationEvidenceError,
    publish_captured_image,
)
from capability_pipeline.oci_artifact import ImageArtifactError
from capability_pipeline.rootfs_review import RootfsReviewError


class Publisher:
    repository = "capability-env-gen/task-candidate"

    def __init__(self):
        self.calls = []

    def publish_layer(self, path, **kwargs):
        self.calls.append((path.read_bytes(), kwargs))
        return {
            "state": "integrity_verified",
            "image": "registry/repository@sha256:fixture",
        }


def inputs(tmp_path, content=b"reviewed", extra=()):
    document = contract(tmp_path)
    document.update(state="approved_for_capture", review_blockers=[])
    document["capture_implementation"] = {
        name + "_sha256": hashlib.sha256(
            (
                Path(__file__).parents[1] / "capability_pipeline" / (name + ".py")
            ).read_bytes()
        ).hexdigest()
        for name in (
            "oci_artifact",
            "oci_registry",
            "rootfs_review",
            "image_publication",
        )
    }
    image = document["images"][0]
    image.update(architecture="amd64", operating_system="linux")
    image["required_ready_hashes"] = {
        "/workspace/task.txt": hashlib.sha256(b"reviewed").hexdigest()
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document))
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    layer = tmp_path / "rootfs.tar.gz"
    layer.write_bytes(archive([("./workspace/task.txt", content), *extra]).getvalue())
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "state": "captured_pending_privacy_and_publication",
                "role": "candidate",
                "plan_sha256": digest,
                "source_snapshot": image["source_snapshot"],
                "cleanup": {"absence_verified": True},
                "capture": {
                    "sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
                    "object_bytes": layer.stat().st_size,
                    "capture": {
                        "tar_exit": 0,
                        "gzip_exit": 0,
                        "compressed_bytes": layer.stat().st_size,
                    },
                },
            }
        )
    )
    return {
        "plan_path": plan,
        "workspace": tmp_path,
        "capture_path": capture,
        "layer_path": layer,
        "approved_plan_sha256": digest,
        "max_uncompressed_bytes": 1 << 20,
    }


def test_exact_reviewed_archive_reaches_publisher_with_reviewed_config(tmp_path):
    params = inputs(tmp_path)
    client = Publisher()
    receipt = publish_captured_image(**params, registry_client=client)
    assert receipt["state"] == "published_pending_cold_pull"
    assert receipt["rootfs_review"]["state"] == "passed"
    assert receipt["task_acceptance"] == "not_evaluated"
    assert client.calls[0][0] == params["layer_path"].read_bytes()
    assert client.calls[0][1]["image_config"]["Env"] == ["PATH=/bin"]


def test_review_only_needs_no_registry_client(tmp_path):
    assert (
        publish_captured_image(**inputs(tmp_path))["state"] == "reviewed_not_published"
    )


@pytest.mark.parametrize(
    "change, error",
    [
        ("old_public_content", RootfsReviewError),
        ("provider_file", RootfsReviewError),
        ("modified_layer", ImageArtifactError),
        ("wrong_plan", PublicationEvidenceError),
        ("tar_error", PublicationEvidenceError),
        ("cleanup_unknown", PublicationEvidenceError),
        ("implementation_drift", PublicationEvidenceError),
    ],
)
def test_failed_gate_never_publishes(tmp_path, change, error):
    params = inputs(
        tmp_path,
        content=b"old" if change == "old_public_content" else b"reviewed",
        extra=[("./etc/hostname", b"provider-host")]
        if change == "provider_file"
        else [],
    )
    if change == "modified_layer":
        params["layer_path"].write_bytes(b"wrong")
    if change == "wrong_plan":
        params["approved_plan_sha256"] = "0" * 64
    if change == "implementation_drift":
        path = params["plan_path"]
        document = json.loads(path.read_text())
        document["capture_implementation"]["rootfs_review_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        params["approved_plan_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if change in {"tar_error", "cleanup_unknown"}:
        path = params["capture_path"]
        capture = json.loads(path.read_text())
        if change == "tar_error":
            capture["capture"]["capture"]["tar_exit"] = 2
        else:
            capture["cleanup"] = {"delete_requested": True}
        path.write_text(json.dumps(capture))
    client = Publisher()
    with pytest.raises(error):
        publish_captured_image(**params, registry_client=client)
    assert client.calls == []
