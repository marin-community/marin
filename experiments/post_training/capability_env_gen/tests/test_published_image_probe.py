import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import probe_published_task_image as probe


def evidence():
    plan = {
        "state": "approved_for_capture",
        "review_blockers": [],
        "registry_host": "registry.example",
        "images": [
            {
                "role": "candidate",
                "repository": "tasks/candidate",
                "required_ready_hashes": {"/fixtures/input": "a" * 64},
            }
        ],
    }
    publication = {
        "plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
        "state": "published_pending_cold_pull",
        "role": "candidate",
        "rootfs_review": {
            "state": "passed",
            "required_file_hashes": copy.deepcopy(
                plan["images"][0]["required_ready_hashes"]
            ),
        },
        "publication": {
            "state": "integrity_verified",
            "image": "registry.example/tasks/candidate@sha256:" + "b" * 64,
            "manifest_digest": "sha256:" + "b" * 64,
        },
    }
    return plan, publication


def test_failed_readiness_keeps_bounded_startup_diagnostics(monkeypatch):
    calls = []

    def shell(sandbox, query, timeout):
        calls.append(query)
        return {"exit": 1, "stdout": "x" * 9000, "stderr": "not ready"}

    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    with pytest.raises(probe.ReadinessError) as failed:
        probe.ready(SimpleNamespace(sh=shell), object())
    assert len(calls) == 64
    assert len(failed.value.diagnostics["postgres_startup"]["stdout"]) == 8192
    assert failed.value.diagnostics["postgres_ready"]["exit"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "foreign_registry",
        "foreign_repository",
        "wrong_manifest",
        "changed_files",
        "unpublished",
        "unreviewed",
        "wrong_plan",
        "blocked_plan",
    ],
)
def test_rejects_unbound_publication_before_provider_access(change):
    plan, publication = evidence()
    if change == "foreign_registry":
        publication["publication"]["image"] = publication["publication"][
            "image"
        ].replace("registry.example", "other.example")
    elif change == "foreign_repository":
        publication["publication"]["image"] = publication["publication"][
            "image"
        ].replace("tasks/candidate", "tasks/private")
    elif change == "wrong_manifest":
        publication["publication"]["manifest_digest"] = "sha256:" + "c" * 64
    elif change == "changed_files":
        publication["rootfs_review"]["required_file_hashes"]["/fixtures/input"] = (
            "d" * 64
        )
    elif change == "unpublished":
        publication["state"] = "reviewed_not_published"
    elif change == "unreviewed":
        publication["rootfs_review"]["state"] = "failed"
    elif change == "wrong_plan":
        publication["plan_sha256"] = "e" * 64
    elif change == "blocked_plan":
        plan["review_blockers"] = ["provider state"]
        publication["plan_sha256"] = hashlib.sha256(
            json.dumps(plan).encode()
        ).hexdigest()
    with pytest.raises(ValueError):
        probe.checked_publication(
            json.dumps(plan).encode(), json.dumps(publication).encode()
        )


@pytest.mark.parametrize(
    "network_reachable,cleanup_absent", [(False, True), (True, True), (False, False)]
)
def test_two_boot_gate_and_cleanup_receipt(
    tmp_path, monkeypatch, network_reachable, cleanup_absent
):
    plan, publication = evidence()
    plan_path, publication_path = tmp_path / "plan.json", tmp_path / "publication.json"
    plan_path.write_text(json.dumps(plan))
    publication_path.write_text(json.dumps(publication))
    args = SimpleNamespace(
        plan=plan_path, publication=publication_path, output=tmp_path / "receipt.json"
    )
    recipe = "FROM " + publication["publication"]["image"] + "\n"
    snapshot = SimpleNamespace(
        name="cap-cold-" + hashlib.sha256(recipe.encode()).hexdigest()[:24],
        id="snapshot-id",
        ref="provider-ref",
        build_info=SimpleNamespace(dockerfile_content=recipe),
    )
    client = SimpleNamespace(
        snapshot=SimpleNamespace(get=lambda name: snapshot),
        get=lambda sid: SimpleNamespace(
            network_block_all=True, cpu=4, memory=8, disk=10
        ),
    )
    created, deleted, commands = [], [], []

    def create(*args, **kwargs):
        assert kwargs["block_all"] is True
        sid = "sandbox-" + str(len(created))
        sandbox = SimpleNamespace(id=sid, delete=lambda: deleted.append(sid))
        created.append(sandbox)
        return sandbox, 1.0

    def command(dtx, sandbox, text, timeout=60):
        commands.append((sandbox.id, text))
        if text.startswith("sha256sum"):
            return "a" * 64 + "  /fixtures/input"
        if "SELECT to_regclass" in text:
            return "t"
        if text.startswith("python3"):
            return json.dumps({"reachable": network_reachable})
        return ""

    monkeypatch.setitem(
        sys.modules,
        "daytona",
        SimpleNamespace(CreateSnapshotParams=object, Image=object, Resources=object),
    )
    monkeypatch.setitem(
        sys.modules, "dtx", SimpleNamespace(client=lambda: client, create=create)
    )
    monkeypatch.setattr(probe, "command", command)
    monkeypatch.setattr(
        probe,
        "ready",
        lambda *args: {"public_tables": ["fixture"], "postgis_version": "3.4"},
    )
    monkeypatch.setattr(
        probe,
        "wait_for_sandbox_deletion",
        lambda *args: ("not_found" if cleanup_absent else "present", []),
    )
    result = probe.probe(args)
    assert deleted == [sandbox.id for sandbox in created]
    assert result == json.loads(args.output.read_text())
    assert result["task_gates"] == "pending"
    if not cleanup_absent:
        assert result["state"] == "cleanup_unverified"
    elif network_reachable:
        assert result["state"] == "failed"
        assert result["failed_stage"] == "sandbox_1_network"
        assert len(created) == 1
    else:
        assert result["state"] == "passed"
        assert len(created) == 2
        mutations = [sid for sid, text in commands if "CREATE TABLE" in text]
        assert mutations == ["sandbox-0"]
        assert (
            result["sandboxes"][1]["required_file_hashes"]
            == plan["images"][0]["required_ready_hashes"]
        )
