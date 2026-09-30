import hashlib
import json
from types import SimpleNamespace

from test_generic_image_capture import _sha
from test_generic_image_publication import _artifacts

from capability_pipeline.generic_image_cold_pull import _recipe, cold_pull


def test_generic_cold_pull_uses_two_fresh_blocked_sandboxes(tmp_path):
    workspace, tools, plan_path, approval, _, _ = _artifacts(tmp_path)
    plan = json.loads(plan_path.read_text())
    image = plan["images"][0]
    reference = "registry.example/capability-env-gen/test-candidate@sha256:" + "a" * 64
    publication = tmp_path / "publication.json"
    publication.write_text(json.dumps({
        "schema_version": "capability-task-image-publication-v1",
        "state": "published_pending_cold_pull", "role": "candidate",
        "plan_sha256": _sha(plan_path),
        "rootfs_review": {"state": "passed", "required_file_hashes": image["required_ready_hashes"]},
        "publication": {"state": "integrity_verified", "image": reference, "manifest_digest": "sha256:" + "a" * 64},
    }))
    recipe = _recipe(reference, image["image_config"])
    name = "cap-cold-" + hashlib.sha256(recipe.encode()).hexdigest()[:24]
    snapshot = SimpleNamespace(name=name, id="snapshot-id", ref="snapshot-ref", state="ACTIVE", build_info=SimpleNamespace(dockerfile_content=recipe))
    deleted = set()
    created = []

    class Sandbox:
        def __init__(self, index):
            self.id = f"fresh-{index}"

        def delete(self):
            deleted.add(self.id)

    class Client:
        snapshot = SimpleNamespace(get=lambda requested: snapshot)

        def get(self, sandbox_id):
            if sandbox_id in deleted:
                error = RuntimeError("sandbox not found")
                error.status_code = 404
                raise error
            return SimpleNamespace(network_block_all=True)

    class Dtx:
        client = staticmethod(Client)

        @staticmethod
        def create(client, requested, **kwargs):
            assert kwargs["block_all"] is True
            assert requested == name
            sandbox = Sandbox(len(created) + 1)
            created.append(sandbox.id)
            return sandbox, 1.0

        @staticmethod
        def sh(sandbox, command, timeout):
            if command.startswith("sha256sum"):
                return {"exit": 0, "stdout": next(iter(image["required_ready_hashes"].values())) + "  file\n"}
            return {"exit": 0, "stdout": ""}

    result = cold_pull(
        plan_path=plan_path, workspace=workspace, capture_tools=tools,
        approval_path=approval, builder_session_ids={"builder"},
        publication_path=publication, dtx=Dtx,
        create_snapshot=lambda *args: (_ for _ in ()).throw(AssertionError("existing snapshot should be reused")),
        sleeper=lambda _: None,
    )
    assert result["state"] == "passed_pending_task_gates"
    assert created == ["fresh-1", "fresh-2"]
    assert deleted == set(created)
    assert all(row["state"] == "not_found" for row in result["cleanup"])
    assert result["task_gates"] == "pending"
