import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import generic_image_capture as capture


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_text("FROM example@sha256:" + "a" * 64 + "\n")
    (workspace / "public.txt").write_text("public")
    (workspace / "evaluate.py").write_text("print('public workflow')\n")
    (workspace / "private.txt").write_text("private")
    source_files = [
        {"path": name, "sha256": _sha(workspace / name), "visibility": visibility}
        for name, visibility in (("Dockerfile", "public"), ("public.txt", "public"), ("evaluate.py", "public"), ("private.txt", "private"))
    ]
    image = {
        "role": "candidate",
        "repository": "capability-env-gen/test-candidate",
        "architecture": "amd64",
        "operating_system": "linux",
        "source_snapshot": {"name": "snap", "id": "snapshot-id", "ref": "provider-ref"},
        "authored_image_pointer": "provider-ref",
        "source_recipe": {"path": "Dockerfile", "sha256": _sha(workspace / "Dockerfile")},
        "input_files": ["Dockerfile", "public.txt", "evaluate.py"],
        "image_config": {"Env": ["FOO=bar"], "WorkingDir": "/workspace", "User": "root", "Entrypoint": ["/bin/sh"], "Cmd": None},
        "capture_lifecycle": {"ready_command": "test -f /tmp/task-ready", "ready_attempts": 2, "ready_timeout_seconds": 5, "ready_interval_seconds": 0, "quiesce_command": None, "quiesce_probe": None, "exclude_mounts": [], "exclude_absent_paths": ["/var/lib/postgresql/data"]},
        "required_ready_hashes": {"/workspace/public.txt": "b" * 64},
        "resources": {"cpu": 2, "memory": 4, "disk": 10},
    }
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "capture_rootfs.py").write_text("EXCLUDES = " + repr(sorted(capture.REQUIRED_CAPTURE_EXCLUDES)) + "\n")
    for name in ("in_sandbox_capture.py", "dtx.py", "cw_presign.py"):
        (tools / name).write_text("# pinned synthetic tool\n")
    (tmp_path / "dt.py").write_text("# pinned synthetic dt tool\n")
    plan = {
        "schema_version": capture.SCHEMA,
        "state": "review_required",
        "registry_host": "registry.example",
        "source_files": source_files,
        "private_assets": [{"workspace_path": "private.txt", "image_path": "/opt/task/private.txt", "sha256": _sha(workspace / "private.txt")}],
        "capture_implementation": {
            "capture_rootfs_sha256": _sha(tools / "capture_rootfs.py"),
            "in_sandbox_capture_sha256": _sha(tools / "in_sandbox_capture.py"),
            "dtx_sha256": _sha(tools / "dtx.py"),
            "cw_presign_sha256": _sha(tools / "cw_presign.py"),
            "dt_sha256": _sha(tmp_path / "dt.py"),
        },
        "images": [image],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return workspace, tools, plan, plan_path


def test_generic_plan_accepts_one_role_and_rejects_private_candidate_mapping(tmp_path):
    workspace, tools, plan, _ = _fixture(tmp_path)
    assert capture.validate_plan(plan, workspace, tools) is plan
    plan["images"][0]["input_files"].append("private.txt")
    with pytest.raises(ValueError, match="private source"):
        capture.validate_plan(plan, workspace, tools)


def test_generic_plan_rejects_absolute_and_symlink_recipe(tmp_path):
    workspace, tools, plan, _ = _fixture(tmp_path)
    plan["images"][0]["source_recipe"]["path"] = str(workspace / "Dockerfile")
    with pytest.raises(ValueError, match="workspace-relative"):
        capture.validate_plan(plan, workspace, tools)
    plan["images"][0]["source_recipe"]["path"] = "linked"
    (workspace / "linked").symlink_to(workspace / "Dockerfile")
    with pytest.raises(ValueError, match="symlink"):
        capture.validate_plan(plan, workspace, tools)


def test_generic_plan_rejects_builder_self_approval_and_capture_source_drift(tmp_path):
    workspace, tools, plan, _ = _fixture(tmp_path)
    plan["state"] = "approved_for_capture"
    with pytest.raises(ValueError, match="review_required"):
        capture.validate_plan(plan, workspace, tools)
    plan["state"] = "review_required"
    (workspace / "public.txt").write_text("changed")
    with pytest.raises(ValueError, match="source hash"):
        capture.validate_plan(plan, workspace, tools)


def test_generic_plan_rejects_staged_tool_drift(tmp_path):
    workspace, tools, plan, _ = _fixture(tmp_path)
    (tools / "dtx.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="capture implementation changed"):
        capture.validate_plan(plan, workspace, tools)


def test_generic_capture_keeps_cleanup_and_sanitizes_raw_receipt(tmp_path):
    workspace, tools, _, plan_path = _fixture(tmp_path)
    snapshot = SimpleNamespace(name="snap", id="snapshot-id", ref="provider-ref", cpu=1, mem=2, disk=3, state="ACTIVE", build_info=SimpleNamespace(dockerfile_content=(workspace / "Dockerfile").read_text()))
    calls = []
    commands = []

    class Sandbox:
        id = "fresh-sandbox"

        def delete(self):
            calls.append("delete")

    class Client:
        snapshot = SimpleNamespace(get=lambda name: snapshot)

        def get(self, sandbox_id):
            error = RuntimeError("not found")
            error.status_code = 404
            raise error

    class Dtx:
        client = staticmethod(Client)
        create = staticmethod(lambda client, name, **kwargs: (Sandbox(), 1.0))

        @staticmethod
        def sh(sandbox, command, timeout):
            commands.append(command)
            if command.startswith("sha256sum"):
                return {"exit": 0, "stdout": "b" * 64 + "  /workspace/public.txt\n"}
            if command == "env":
                return {"exit": 0, "stdout": "FOO=bar\nHOME=/root\n"}
            return {"exit": 0, "stdout": ""}

    def runner(command, **kwargs):
        raw_path = Path(command[command.index("--out") + 1])
        raw_path.write_text(json.dumps({
            "source_env": {"FOO": "bar", "DAYTONA_API_KEY": "secret"},
            "ok": True, "object_bytes": 3, "sha256": "c" * 64,
            "capture": {"compressed_bytes": 3, "sha256": "c" * 64, "tar_exit": 0, "gzip_exit": 0, "etags": ["secret"]},
        }))
        return SimpleNamespace(returncode=0)

    output = tmp_path / "receipt.json"
    result = capture.capture_role(plan_path, workspace, "candidate", output, approved_plan_sha256=_sha(plan_path), dtx=Dtx, capture_tools=tools, runner=runner, sleeper=lambda _: None)
    assert result["state"] == "captured_pending_privacy_and_publication"
    assert result["cleanup"]["absence_verified"] is True
    assert calls == ["delete"]
    assert "DAYTONA_API_KEY" not in output.read_text()
    assert "etags" not in output.read_text()
    assert result["capture"]["capture"]["compressed_bytes"] == 3
    assert "/opt/task/private.txt" in result["candidate_private_paths_checked"]
    assert "evaluate.py" not in result["candidate_private_paths_checked"]
    assert not any("*evaluate.py*" in command or "-iname evaluate.py" in command for command in commands)


def test_generic_capture_rejects_reviewed_private_asset_present(tmp_path):
    workspace, tools, _, plan_path = _fixture(tmp_path)
    snapshot = SimpleNamespace(name="snap", id="snapshot-id", ref="provider-ref", cpu=1, mem=2, disk=3, state="ACTIVE", build_info=SimpleNamespace(dockerfile_content=(workspace / "Dockerfile").read_text()))

    class Sandbox:
        id = "fresh-sandbox"

        def delete(self):
            pass

    class Client:
        snapshot = SimpleNamespace(get=lambda name: snapshot)

        def get(self, sandbox_id):
            error = RuntimeError("not found")
            error.status_code = 404
            raise error

    class Dtx:
        client = staticmethod(Client)
        create = staticmethod(lambda client, name, **kwargs: (Sandbox(), 1.0))

        @staticmethod
        def sh(sandbox, command, timeout):
            if command.startswith("sha256sum"):
                return {"exit": 0, "stdout": "b" * 64 + "  /workspace/public.txt\n"}
            if "test ! -e /opt/task/private.txt" in command:
                return {"exit": 1, "stdout": ""}
            return {"exit": 0, "stdout": ""}

    receipt = capture.capture_role(plan_path, workspace, "candidate", tmp_path / "receipt.json", approved_plan_sha256=_sha(plan_path), dtx=Dtx, capture_tools=tools, runner=lambda *args, **kwargs: pytest.fail("rootfs capture must not start after privacy failure"), sleeper=lambda _: None)
    assert receipt["state"] == "privacy_review_required"
    assert receipt["error_type"] == "PrivacyReviewRequired"
    assert receipt["cleanup"]["absence_verified"] is True
