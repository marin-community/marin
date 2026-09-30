"""Publication-time correction of an object-store plan host to the real registry.

A reviewed plan whose `registry_host` names an object-storage endpoint
(`*.cwobject.com`) publishes to the trusted credential's registry, keeps the
reviewed repository, records the correction in the receipt, and never touches
the reviewed plan bytes. Every other host mismatch still rejects.
"""

import hashlib
import json

import pytest
from test_generic_image_publication import _artifacts, _sha
from test_image_publication_handoff import _fixture as _handoff_fixture

from capability_pipeline.generic_image_cold_pull import _checked_publication
from capability_pipeline.generic_image_publication import (
    REGISTRY_HOST_CORRECTION_REASON,
    publication_registry_host,
    publish_generic_capture,
    published_registry_host,
)
from capability_pipeline.image_publication_handoff import (
    export_handoff,
    import_publication,
)
from scripts import probe_published_task_image as probe

OBJECT_STORE = "marin-us-east-02a.cwobject.com"
REGISTRY = "envreg.208261-marin-gpu.coreweave.app"


class _Registry:
    def __init__(self, host, repository):
        self.host, self.repository, self.calls = host, repository, 0

    def publish_layer(self, frozen, *, expected_digest, **_):
        self.calls += 1
        manifest = "sha256:" + "c" * 64
        return {"state": "integrity_verified", "manifest_digest": manifest,
                "image": self.host + "/" + self.repository + "@" + manifest,
                "layer_digest": expected_digest}


def _publish(tmp_path, *, plan_host, client_host):
    workspace, tools, plan_path, approval, captured, layer = _artifacts(
        tmp_path, registry_host=plan_host)
    plan_bytes = plan_path.read_bytes()
    repository = json.loads(plan_bytes)["images"][0]["repository"]
    client = _Registry(client_host, repository)
    receipt = publish_generic_capture(
        plan_path=plan_path, workspace=workspace, capture_tools=tools,
        approval_path=approval, builder_session_ids={"builder-1"},
        capture_path=captured, layer_path=layer, max_uncompressed_bytes=1_000_000,
        registry_client=client,
    )
    return receipt, plan_path, plan_bytes, approval, client


def test_object_store_plan_host_publishes_to_credential_registry(tmp_path):
    receipt, plan_path, plan_bytes, approval, client = _publish(
        tmp_path, plan_host=OBJECT_STORE, client_host=REGISTRY)
    repository = json.loads(plan_bytes)["images"][0]["repository"]
    assert client.calls == 1
    assert receipt["state"] == "published_pending_cold_pull"
    assert receipt["publication"]["image"] == (
        REGISTRY + "/" + repository + "@sha256:" + "c" * 64)
    assert receipt["registry_host_corrected"] == {
        "from": OBJECT_STORE, "to": REGISTRY, "reason": REGISTRY_HOST_CORRECTION_REASON}
    # The reviewed plan bytes and their review binding are untouched.
    assert plan_path.read_bytes() == plan_bytes
    assert json.loads(plan_bytes)["registry_host"] == OBJECT_STORE
    assert receipt["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert json.loads(approval.read_text())["plan_sha256"] == receipt["plan_sha256"]


def test_matching_host_records_no_correction(tmp_path):
    receipt, *_ = _publish(tmp_path, plan_host="registry.example", client_host="registry.example")
    assert receipt["state"] == "published_pending_cold_pull"
    assert "registry_host_corrected" not in receipt


@pytest.mark.parametrize("plan_host,client_host", [
    ("other.example", REGISTRY),               # arbitrary wrong host
    ("registry.example", REGISTRY),            # a different real registry
    (OBJECT_STORE, "cwobject.com"),            # "correcting" to another object store
    (OBJECT_STORE + ":443", REGISTRY),         # host with a port is not the known endpoint
    ("cwobject.com.evil.example", REGISTRY),   # suffix spoof
])
def test_other_host_mismatches_still_reject_before_upload(tmp_path, plan_host, client_host):
    with pytest.raises(ValueError, match="registry client target differs"):
        _publish(tmp_path, plan_host=plan_host, client_host=client_host)


def test_publisher_credential_check_uses_the_same_rule():
    assert publication_registry_host(REGISTRY, REGISTRY) == (REGISTRY, None)
    assert publication_registry_host(OBJECT_STORE, REGISTRY)[0] == REGISTRY
    for plan_host, credential in [("other.example", REGISTRY), (OBJECT_STORE, None),
                                  (OBJECT_STORE, "Bad Host")]:
        with pytest.raises(ValueError):
            publication_registry_host(plan_host, credential)


def _corrected(tmp_path):
    receipt, plan_path, _, _, _ = _publish(tmp_path, plan_host=OBJECT_STORE, client_host=REGISTRY)
    publication_path = tmp_path / "publication-candidate.json"
    publication_path.write_text(json.dumps(receipt))
    return json.loads(plan_path.read_text()), plan_path, publication_path, receipt


def test_downstream_accepts_the_recorded_correction(tmp_path):
    plan, plan_path, publication_path, receipt = _corrected(tmp_path)
    image, reference = _checked_publication(plan, plan_path, publication_path)
    assert reference == receipt["publication"]["image"]
    assert reference.startswith(REGISTRY + "/" + image["repository"] + "@sha256:")


@pytest.mark.parametrize("forge", ["dropped", "retargeted", "other_reason", "extra_key", "null"])
def test_downstream_rejects_absent_or_forged_correction(tmp_path, forge):
    plan, plan_path, publication_path, receipt = _corrected(tmp_path)
    if forge == "dropped":
        del receipt["registry_host_corrected"]
    elif forge == "retargeted":
        receipt["registry_host_corrected"]["to"] = "other.example"
    elif forge == "other_reason":
        receipt["registry_host_corrected"]["reason"] = "because"
    elif forge == "extra_key":
        receipt["registry_host_corrected"]["note"] = "x"
    else:
        receipt["registry_host_corrected"] = None
    publication_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from reviewed image"):
        _checked_publication(plan, plan_path, publication_path)


def test_correction_cannot_launder_a_non_object_store_plan_host():
    plan = {"registry_host": "other.example"}
    with pytest.raises(ValueError):
        published_registry_host(plan, {"registry_host_corrected": {
            "from": "other.example", "to": REGISTRY, "reason": REGISTRY_HOST_CORRECTION_REASON}})


def test_handoff_import_accepts_corrected_publication(tmp_path):
    status, archive, plan_path, approval, capture, target = _handoff_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    assert plan["registry_host"] == "registry.example"
    # Re-root the handoff fixture on an object-store plan host with a fresh review.
    source = tmp_path / "cw-source"
    source.mkdir()
    _, _, cw_plan, cw_approval, cw_capture, _ = _artifacts(source, registry_host=OBJECT_STORE)
    plan_path.write_bytes(cw_plan.read_bytes())
    approval.write_bytes(cw_approval.read_bytes())
    for name in ("review-raw.json", "input-manifest.json"):
        (approval.parent / name).write_bytes((cw_approval.parent / name).read_bytes())
    (approval.parent / "input/controller/image-plan.json").write_bytes(cw_plan.read_bytes())
    capture.write_bytes(cw_capture.read_bytes())
    export_handoff(status, archive)
    image = json.loads(plan_path.read_text())["images"][0]
    captured = json.loads(capture.read_text())
    reference = REGISTRY + "/" + image["repository"] + "@sha256:" + "a" * 64
    publication = tmp_path / "publisher-output.json"
    publication.write_text(json.dumps({
        "schema_version": "capability-task-image-publication-v1",
        "state": "published_pending_cold_pull", "role": "candidate",
        "plan_sha256": _sha(plan_path), "review_sha256": _sha(approval),
        "capture_receipt_sha256": _sha(capture),
        "source_snapshot": image["source_snapshot"],
        "image_config_sha256": captured["image_config_sha256"],
        "layer_digest": "sha256:" + captured["capture"]["sha256"],
        "rootfs_review": {"state": "passed", "required_file_hashes": image["required_ready_hashes"]},
        "registry_host_corrected": {"from": OBJECT_STORE, "to": REGISTRY,
                                    "reason": REGISTRY_HOST_CORRECTION_REASON},
        "publication": {"state": "integrity_verified", "image": reference,
                        "manifest_digest": "sha256:" + "a" * 64},
    }))
    result = import_publication(status, archive, "candidate", publication)
    assert result["image"] == reference
    assert target.read_bytes() == publication.read_bytes()


def test_legacy_probe_accepts_correction_and_rejects_foreign_host():
    plan = {
        "state": "approved_for_capture", "review_blockers": [],
        "registry_host": OBJECT_STORE,
        "images": [{"role": "candidate", "repository": "tasks/candidate",
                    "required_ready_hashes": {"/fixtures/input": "a" * 64}}],
    }
    plan_bytes = json.dumps(plan).encode()
    publication = {
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "state": "published_pending_cold_pull", "role": "candidate",
        "rootfs_review": {"state": "passed", "required_file_hashes": {"/fixtures/input": "a" * 64}},
        "registry_host_corrected": {"from": OBJECT_STORE, "to": REGISTRY,
                                    "reason": REGISTRY_HOST_CORRECTION_REASON},
        "publication": {"state": "integrity_verified",
                        "image": REGISTRY + "/tasks/candidate@sha256:" + "b" * 64,
                        "manifest_digest": "sha256:" + "b" * 64},
    }
    _, reference = probe.checked_publication(plan_bytes, json.dumps(publication).encode())
    assert reference.startswith(REGISTRY + "/")
    publication["publication"]["image"] = "other.example/tasks/candidate@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="differs from reviewed repository"):
        probe.checked_publication(plan_bytes, json.dumps(publication).encode())
