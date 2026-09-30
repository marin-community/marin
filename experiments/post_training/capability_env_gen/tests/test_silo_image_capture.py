"""Custom-image capture on the silo sandbox provider (no sandbox egress, no Daytona)."""

import gzip
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import generic_image_capture as capture
from capability_pipeline import image_pipeline as pipeline
from capability_pipeline import silo_rootfs_capture as silo
from tests.test_generic_image_capture import _fixture, _sha


class _NotFound(Exception):
    status_code = 404


RECIPE = "FROM example@sha256:" + "a" * 64 + "\n"  # the _fixture Dockerfile bytes


def _snapshot(*, snapshot_id="snapshot-id", ref="provider-ref", recipe=RECIPE):
    return SimpleNamespace(
        name="snap", id=snapshot_id, ref=ref, cpu=2, mem=4, disk=10, state="active",
        build_info=SimpleNamespace(dockerfile_content=recipe),
    )


class _SiloWorld:
    """A silo client/sandbox double that records what the capture asked for."""

    def __init__(self, *, snapshot=None, missing=False, env_text="FOO=bar\nHOME=/root\nHOSTNAME=slb-fresh\n"):
        self.snapshot_record = snapshot or _snapshot()
        self.missing = missing
        self.created_snapshots = []
        self.sandbox_params = []
        self.commands = []
        self.deleted = []
        self.env_text = env_text
        self.exit_overrides = {}
        world = self

        class Sandbox:
            id = "slb-fresh"
            network_block_all = True

            def delete(self):
                world.deleted.append(self.id)

        class SnapshotApi:
            def get(self, name):
                if world.missing:
                    raise _NotFound(f"snapshot {name!r} not found")
                return world.snapshot_record

            def create(self, params, timeout):
                world.created_snapshots.append(params)
                world.missing = False

        class Client:
            snapshot = SnapshotApi()

            def create(self, params, timeout):
                world.sandbox_params.append(params)
                return Sandbox()

            def get(self, sandbox_id):
                raise _NotFound(f"sandbox {sandbox_id!r} not found")

        self.client = Client()

    def dtx(self):
        world = self

        class Dtx:
            @staticmethod
            def client():
                raise AssertionError("dtx.client() is Daytona-only and must not be called on silo")

            @staticmethod
            def create(*args, **kwargs):
                raise AssertionError("dtx.create would request an egress allow-list")

            @staticmethod
            def sh(sandbox, command, timeout):
                world.commands.append(command)
                if command.startswith("sha256sum"):
                    return {"exit": 0, "stdout": "b" * 64 + "  /workspace/public.txt\n"}
                if command == "env":
                    return {"exit": 0, "stdout": world.env_text}
                for fragment, code in world.exit_overrides.items():
                    if fragment in command:
                        return {"exit": code, "stdout": ""}
                return {"exit": 0, "stdout": ""}

        return Dtx


def _raw(sha="c" * 64, size=3):
    return {
        "source_env": {"FOO": "bar", "HOSTNAME": "slb-fresh"},
        "ok": True, "object": f"s3://bucket/{sha}.tar.gz", "object_key": f"{sha}.tar.gz",
        "object_bytes": size, "sha256": sha,
        "capture": {"compressed_bytes": size, "sha256": sha, "tar_exit": 0, "gzip_exit": 0,
                    "parts": 1, "bad_parts": {}, "tar_stderr_tail": ""},
    }


def _run_silo_capture(tmp_path, world, **overrides):
    workspace, tools, _, plan_path = _fixture(tmp_path)
    seen = {}

    def silo_capture(**kwargs):
        seen.update(kwargs)
        return _raw()

    receipt = capture.capture_role(
        plan_path, workspace, "candidate", tmp_path / "receipt.json",
        approved_plan_sha256=_sha(plan_path), dtx=world.dtx(), capture_tools=tools,
        runner=lambda *a, **k: pytest.fail("capture_rootfs.py needs sandbox egress; silo has none"),
        sleeper=lambda _: None, provider="silo", client_factory=lambda: world.client,
        s3_factory=lambda: "harness-s3-client", silo_capture=silo_capture, **overrides,
    )
    return receipt, seen, json.loads((tmp_path / "plan.json").read_text())


def test_silo_capture_uses_network_blocked_sandbox_and_harness_transport(tmp_path):
    world = _SiloWorld()
    receipt, seen, plan = _run_silo_capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert receipt["sandbox_provider"] == "silo"
    assert receipt["cleanup"]["absence_verified"] is True
    assert world.deleted == ["slb-fresh"]
    params = world.sandbox_params[0]
    assert params["network_block_all"] is True
    assert "domain_allow_list" not in params and "network_allow_list" not in params
    # The archive leaves through the harness, which alone holds the object-store client.
    assert seen["s3"] == "harness-s3-client"
    assert seen["sandbox"].id == "slb-fresh"
    reviewed = capture._reviewed_excludes(tmp_path / "tools")
    assert seen["excludes"][: len(reviewed)] == reviewed
    assert {"./.silo", "./.silo/*"} <= set(seen["excludes"])
    assert any('stat -c %d /.silo' in command for command in world.commands)
    assert not any("findmnt" in command and "daytona" in command for command in world.commands)
    assert receipt["rootfs_transport"]["transport"] == silo.TRANSPORT
    assert receipt["rootfs_transport"]["helper_sha256"] == silo.helper_sha256()
    assert capture.snapshot_identity_matches(receipt, plan["images"][0])


def test_silo_capture_rematerializes_missing_snapshot_from_reviewed_recipe(tmp_path):
    # A broker that never saw the builder's snapshot hands out a new random id.
    world = _SiloWorld(snapshot=_snapshot(snapshot_id="snap-rebuilt"), missing=True)
    receipt, _, plan = _run_silo_capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    [created] = world.created_snapshots
    assert created["name"] == "snap"
    assert created["image"] == (tmp_path / "workspace/Dockerfile").read_text()
    assert created["resources"] == {"cpu": 2, "memory": 4, "disk": 10}
    assert receipt["source_snapshot_rematerialized"] is True
    assert receipt["source_snapshot"]["id"] == "snap-rebuilt"
    assert receipt["source_snapshot_reviewed"]["id"] == "snapshot-id"
    image = plan["images"][0]
    assert capture.snapshot_identity_matches(receipt, image)
    # A Daytona receipt gets no such latitude, and silo still binds name and ref.
    assert not capture.snapshot_identity_matches({**receipt, "sandbox_provider": "daytona"}, image)
    assert not capture.snapshot_identity_matches(
        {**receipt, "source_snapshot": {**receipt["source_snapshot"], "ref": "other"}}, image)
    assert not capture.snapshot_identity_matches({**receipt, "source_recipe_verified": False}, image)
    assert not capture.snapshot_identity_matches({**receipt, "source_recipe_sha256": "0" * 64}, image)


def test_silo_capture_refuses_changed_recipe_or_ref(tmp_path):
    world = _SiloWorld(snapshot=_snapshot(snapshot_id="x", recipe="FROM other\n"))
    with pytest.raises(capture.CaptureProvisionError, match="recipe changed") as changed:
        _run_silo_capture(tmp_path, world)
    # A reviewed snapshot name rebuilt from another recipe is task content, not infrastructure.
    assert changed.value.failure_class == capture.CONTENT and changed.value.step == "snapshot"
    assert world.sandbox_params == []
    (tmp_path / "second").mkdir()
    world2 = _SiloWorld(snapshot=_snapshot(ref="silo.local/snapshots/other:built"))
    with pytest.raises(capture.CaptureProvisionError, match="identity"):
        _run_silo_capture(tmp_path / "second", world2)
    assert world2.sandbox_params == []


def test_silo_capture_without_harness_object_store_client_fails_closed(tmp_path):
    world = _SiloWorld()
    workspace_, tools, _, plan_path = _fixture(tmp_path)
    with pytest.raises(capture.CaptureProvisionError, match="object-store client") as missing:
        capture.capture_role(
            plan_path, workspace_, "candidate", tmp_path / "receipt.json",
            approved_plan_sha256=_sha(plan_path), dtx=world.dtx(), capture_tools=tools,
            sleeper=lambda _: None, provider="silo", client_factory=lambda: world.client,
        )
    # Our own wiring: refused before any sandbox is spent, and classified as harness.
    assert missing.value.failure_class == capture.HARNESS
    assert world.sandbox_params == [] and not (tmp_path / "receipt.json").exists()


def test_image_pipeline_accepts_silo_receipt_with_rebuilt_snapshot_id(tmp_path, monkeypatch):
    world = _SiloWorld(snapshot=_snapshot(snapshot_id="snap-rebuilt"), missing=True)
    receipt, _, _ = _run_silo_capture(tmp_path, world)

    item = tmp_path / "items/task-1"
    (item / "workspace/task").mkdir(parents=True)
    attempt = item / "diagnostics/image-capture/attempt-source"
    (attempt / "input").mkdir(parents=True)
    plan_path = attempt / "input/plan.json"
    plan_path.write_bytes((tmp_path / "plan.json").read_bytes())
    receipt["plan_sha256"] = pipeline._sha(plan_path)
    review_root = tmp_path / "image-reviews/task-1/attempt-source"
    review_root.mkdir(parents=True)
    (review_root / "approval.json").write_text("{}")
    monkeypatch.setattr(pipeline, "request_needed", lambda _: {"needed": True})
    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda *_: {
        "attempt": str(attempt), "plan_path": str(plan_path), "workspace": str(tmp_path / "workspace")})
    monkeypatch.setattr(pipeline, "validate_review", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "_review_matches_task", lambda *a: None)

    def runner(command):
        Path(command[command.index("--output") + 1]).write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    result = pipeline.process_image_construction(
        item_root=item, capture_tools=tmp_path / "tools", scripts_root=tmp_path / "scripts",
        agent=object(), builder_session_ids={"builder"}, command_runner=runner,
        review_base=tmp_path / "image-reviews/task-1",
    )
    assert result["state"] == "pending_publication", result


# --------------------------------------------------------------------------- #
# The transport itself, run for real: the in-sandbox helper under a local shell.
# --------------------------------------------------------------------------- #


class _LocalSandbox:
    """Maps sandbox paths under a local directory and runs commands with bash."""

    def __init__(self, base: Path, root: Path, *, tamper: bool = False,
                 tools: tuple[str, ...] = ("python3", "tar", "gzip")):
        self.id = "slb-local"
        self.base = base
        self.root = root
        self.tamper = tamper
        self.tools = tools
        self.fs = self
        self.lock = threading.Lock()

    def _local(self, path: str) -> Path:
        return self.base / path.lstrip("/")

    def _rewrite(self, text: str) -> str:
        return text.replace(silo.WORK_DIR, str(self._local(silo.WORK_DIR)))

    def upload_file(self, data: bytes, path: str) -> None:
        target = self._local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".job.json"):
            job = json.loads(data)
            job["out_dir"] = self._rewrite(job["out_dir"])
            job["root"] = str(self.root)
            data = json.dumps(job).encode()
        target.write_bytes(data)

    def download_file(self, path: str) -> bytes:
        data = self._local(path).read_bytes()
        if self.tamper and path.endswith("part-00002"):
            data = data[:-1] + bytes([data[-1] ^ 1])
        return data

    def sh(self, sandbox, command, timeout):
        if command.startswith("env | sort"):
            return {"exit": 0, "stdout": "FOO=bar\n--- DU ---\n12\n"}
        if command == silo._TOOL_PROBE:
            # Never probe the host: the answer decides which helper runs.
            return {"exit": 0, "stdout": "".join(f"have:{tool}\n" for tool in self.tools)}
        if command.startswith("sh ") and " / " in command:
            raise AssertionError("the shell helper would archive the host root; pass root=")
        command = self._rewrite(command)
        if command.startswith("python3 "):
            command = shlex_quote(sys.executable) + command[len("python3"):]
        completed = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout, check=False)
        return {"exit": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def shlex_quote(value):
    return shlex.quote(value)


class _FakeS3:
    def __init__(self):
        self.parts = {}
        self.objects = {}
        self.aborted = []

    def create_multipart_upload(self, Bucket, Key):
        return {"UploadId": "upload-1"}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.parts[PartNumber] = Body
        return {"ETag": hashlib.md5(Body).hexdigest()}  # fake ETag

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        numbers = [row["PartNumber"] for row in MultipartUpload["Parts"]]
        assert numbers == sorted(self.parts)
        self.objects[Key] = b"".join(self.parts[n] for n in numbers)

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.aborted.append(Key)

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_object(self, Bucket, Key):
        del self.objects[Key]

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}


def _rootfs(tmp_path: Path) -> Path:
    root = tmp_path / "rootfs"
    (root / "opt/verifier").mkdir(parents=True)
    (root / "opt/verifier/READY").write_text("ready\n")
    (root / "data.bin").write_bytes(os.urandom(300_000))
    (root / ".silo").mkdir()
    (root / ".silo/busybox").write_text("provider plumbing")
    (root / "tmp").mkdir()
    (root / "tmp/scratch").write_text("scratch")
    return root


@pytest.mark.skipif(shutil.which("tar") is None or shutil.which("gzip") is None, reason="needs tar and gzip")
def test_silo_transport_streams_parts_and_content_addresses_the_archive(tmp_path):
    sandbox = _LocalSandbox(tmp_path / "sandbox", _rootfs(tmp_path))
    s3 = _FakeS3()
    raw = silo.capture_rootfs(
        sandbox=sandbox, sh=sandbox.sh, snapshot_name="snap", tag="cap-test",
        excludes=["./tmp/*", *silo.PROVIDER_EXCLUDES], s3=s3, gzip_level=1,
        part_bytes=64 * 1024, max_outstanding=2, poll_seconds=0.01, sleeper=lambda s: None,
    )
    assert raw["ok"] is True
    assert raw["capture"]["parts"] > 2  # the stream really was cut and reassembled
    data = s3.objects[raw["object_key"]]
    assert raw["object_key"] == f"{silo.PREFIX}/{hashlib.sha256(data).hexdigest()}.tar.gz"
    assert raw["object_bytes"] == len(data) == raw["capture"]["compressed_bytes"]
    assert raw["sha256"] == raw["capture"]["sha256"]
    assert raw["source_env"] == {"FOO": "bar"} and raw["unpacked_kb"] == 12
    assert not any(key.startswith(f"{silo.PREFIX}/_staging-") for key in s3.objects)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as archive:
        names = {name.removeprefix("./") for name in archive.getnames()}
    assert "opt/verifier/READY" in names and "data.bin" in names
    assert not any(name.startswith(".silo/") for name in names)
    assert "tmp/scratch" not in names
    # Every part was consumed and removed from the sandbox.
    leftover = list((tmp_path / "sandbox" / silo.WORK_DIR.lstrip("/") / "cap-test").glob("part-*"))
    assert leftover == []
    # The raw receipt passes the same archive checks as a Daytona capture_rootfs.py receipt.
    from scripts.capture_task_images import (
        _sanitize_legacy_receipt,
        _validate_archive_receipt,
    )

    assert _validate_archive_receipt(raw)["tar_exit_class"] == "clean"
    assert "etags" not in json.dumps(_sanitize_legacy_receipt(raw))


@pytest.mark.skipif(shutil.which("tar") is None or shutil.which("gzip") is None, reason="needs tar and gzip")
def test_silo_transport_rejects_bytes_that_differ_from_the_sandbox_count(tmp_path):
    sandbox = _LocalSandbox(tmp_path / "sandbox", _rootfs(tmp_path), tamper=True)
    s3 = _FakeS3()
    with pytest.raises(silo.SiloCaptureError, match="differ"):
        silo.capture_rootfs(
            sandbox=sandbox, sh=sandbox.sh, snapshot_name="snap", tag="cap-tamper",
            excludes=["./tmp/*"], s3=s3, gzip_level=1, part_bytes=64 * 1024,
            max_outstanding=2, poll_seconds=0.01, sleeper=lambda s: None,
        )
    assert s3.aborted and not s3.objects


def test_silo_transport_fails_closed_when_the_helper_never_reports(tmp_path):
    class Sandbox:
        id = "slb-dead"
        fs = SimpleNamespace(upload_file=lambda data, path: None,
                             download_file=lambda path: pytest.fail("nothing to download"))

    def sh(sandbox, command, timeout):
        if command == silo._TOOL_PROBE:
            return {"exit": 0, "stdout": "have:python3\nhave:tar\nhave:gzip\n"}
        if command.startswith("python3"):
            return {"exit": 127, "stdout": "python3: not found"}
        return {"exit": 0, "stdout": ""}

    s3 = _FakeS3()
    with pytest.raises(silo.SiloCaptureError, match="without a result"):
        silo.capture_rootfs(sandbox=Sandbox(), sh=sh, snapshot_name="snap", tag="cap-dead",
                            excludes=[], s3=s3, poll_seconds=0.01, sleeper=lambda s: None)
    assert len(s3.aborted) == 1 and not s3.objects


# --------------------------------------------------------------------------- #
# The trusted CLI entry point: silo credentials, not Daytona's.
# --------------------------------------------------------------------------- #


def test_capture_cli_on_silo_needs_no_daytona_key_and_uses_silo_client(tmp_path, monkeypatch):
    from scripts import capture_generic_task_image as cli

    tools = tmp_path / "daytona-tools/capture-tools"
    tools.mkdir(parents=True)
    (tools / "dtx.py").write_text("import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
    (tools / "cw_presign.py").write_text("class presigner:\n    def __init__(self):\n        self.c = 'cw-client'\n")
    (tools.parent / "dt.py").write_text("def client():\n    return 'silo-client'\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"images": [{"role": "candidate"}]}))
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"plan_sha256": "p" * 64}))
    monkeypatch.setattr(cli, "validate_plan", lambda *a: None)
    monkeypatch.setattr(cli, "validate_review", lambda *a, **k: None)
    seen = {}

    def fake_capture(*args, **kwargs):
        seen.update(kwargs)
        return {"state": "captured_pending_privacy_and_publication"}

    monkeypatch.setattr(cli, "capture_role", fake_capture)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    monkeypatch.setenv("SILO_API_TOKEN", "token")
    monkeypatch.setenv("SILO_BROKER_RESOLVE_URL", "http://resolve")
    monkeypatch.setenv("CW_KEY_ID", "id")
    monkeypatch.setenv("CW_KEY_SECRET", "secret")
    for name in ("dtx", "dt", "cw_presign"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    argv = ["--plan", str(plan), "--workspace", str(tmp_path), "--capture-tools", str(tools),
            "--role", "candidate", "--approval", str(approval), "--output", str(tmp_path / "out.json"),
            "--execute"]
    assert cli.main(argv) == 0
    assert seen["provider"] == "silo"
    assert seen["client_factory"]() == "silo-client"
    assert seen["s3_factory"]() == "cw-client"

    # Without the silo credentials the trusted capture refuses before any provider call.
    monkeypatch.delenv("SILO_API_TOKEN")
    seen.clear()
    with pytest.raises(ValueError, match="credentials"):
        cli.main(argv)
    assert seen == {}


def test_capture_cli_on_daytona_still_requires_daytona_key(tmp_path, monkeypatch):
    from scripts import capture_generic_task_image as cli

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"images": [{"role": "candidate"}]}))
    monkeypatch.setattr(cli, "validate_plan", lambda *a: None)
    monkeypatch.setattr(cli, "validate_review", lambda *a, **k: None)
    monkeypatch.setattr(cli, "capture_role", lambda *a, **k: pytest.fail("must not capture"))
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "daytona")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("CW_KEY_ID", "id")
    monkeypatch.setenv("CW_KEY_SECRET", "secret")
    with pytest.raises(ValueError, match="credentials"):
        cli.main(["--plan", str(plan), "--workspace", str(tmp_path), "--capture-tools", str(tmp_path),
                  "--role", "candidate", "--approval", str(tmp_path / "a.json"),
                  "--output", str(tmp_path / "o.json"), "--execute"])


def test_cold_pull_cli_on_silo_uses_silo_client(tmp_path, monkeypatch):
    from scripts import probe_generic_task_image as cli

    tools = tmp_path / "daytona-tools/capture-tools"
    tools.mkdir(parents=True)
    (tools / "dtx.py").write_text(
        "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "def client():\n    raise AssertionError('Daytona client on silo')\n"
        "def create(*a, **k):\n    return 'created'\n"
        "def sh(*a, **k):\n    return 'ran'\n")
    (tools.parent / "dt.py").write_text("def client():\n    return 'silo-client'\n")
    seen = {}

    def fake_cold_pull(**kwargs):
        seen["client"] = kwargs["dtx"].client()
        seen["create"] = kwargs["dtx"].create()
        return {"state": "passed_pending_task_gates", "role": "candidate"}

    monkeypatch.setattr(cli, "cold_pull", fake_cold_pull)
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    for name in ("dtx", "dt"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    argv = ["--plan", str(tmp_path / "p"), "--workspace", str(tmp_path), "--capture-tools", str(tools),
            "--approval", str(tmp_path / "a"), "--publication", str(tmp_path / "pub"),
            "--output", str(tmp_path / "cold.json")]
    assert cli.main(argv) == 0
    assert seen == {"client": "silo-client", "create": "created"}


def test_infrastructure_capture_receipt_is_retired_and_retried_to_a_terminal_outcome(tmp_path, monkeypatch):
    """The old controller parked the item forever after three retirements."""
    item = tmp_path / "items/task-1"
    (item / "workspace/task").mkdir(parents=True)
    attempt = item / "diagnostics/image-capture/attempt-source"
    (attempt / "input").mkdir(parents=True)
    plan_path = attempt / "input/plan.json"
    plan_path.write_text(json.dumps({"images": [{"role": "candidate", "source_snapshot": {
        "name": "snap", "id": "snapshot-id", "ref": "provider-ref"}}]}))
    review_root = tmp_path / "image-reviews/task-1/attempt-source"
    review_root.mkdir(parents=True)
    (review_root / "approval.json").write_text("{}")
    monkeypatch.setattr(pipeline, "request_needed", lambda _: {"needed": True})
    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda *_: {
        "attempt": str(attempt), "plan_path": str(plan_path), "workspace": str(tmp_path)})
    monkeypatch.setattr(pipeline, "validate_review", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "_review_matches_task", lambda *a: None)
    monkeypatch.delenv("CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS", raising=False)
    calls = []

    def runner(command):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "schema_version": "capability-rootfs-capture-v1", "state": "infrastructure_error",
            "role": "candidate", "plan_sha256": pipeline._sha(plan_path), "failure_class": "transient",
            "failed_step": "archive_transport",
            "source_snapshot": {"name": "snap", "id": "snapshot-id", "ref": "provider-ref"},
            "cleanup": {"absence_verified": True}, "error_type": "SiloCaptureError"}))
        return SimpleNamespace(returncode=1)

    args = {"item_root": item, "capture_tools": tmp_path, "scripts_root": tmp_path, "agent": object(),
            "builder_session_ids": {"builder"}, "command_runner": runner,
            "review_base": tmp_path / "image-reviews/task-1"}
    results = [pipeline.process_image_construction(**args) for _ in range(8)]
    budget = pipeline.MAX_CAPTURE_INFRASTRUCTURE_RETRIES
    assert budget == 6
    assert [r["state"] for r in results[: budget - 1]] == ["pending_capture"] * (budget - 1)
    assert all(r["retryable"] is True and r["reason"] == "capture_infrastructure_error" for r in results[: budget - 1])
    assert [r["attempts"] for r in results[: budget - 1]] == list(range(1, budget))
    assert results[1]["backoff_seconds"] > results[0]["backoff_seconds"]
    # The budget ends in an explicit terminal outcome, and nothing runs after it.
    assert {r["state"] for r in results[budget - 1:]} == {"failed_terminal"}
    assert results[-1]["failure_stage"] == "image_capture"
    assert results[-1]["reason"] == "image_capture_failed: archive_transport:SiloCaptureError"
    assert len(calls) == budget
    assert not (attempt / "capture-candidate.json").exists()
    assert len(list(attempt.glob("capture-candidate.attempt-*.receipt.json"))) == budget
