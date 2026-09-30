"""Regression tests for image-capture failures seen in catalog-full-construct-003.

Each test is built from a real failure shape recorded in the run's capture
receipts, silo host/broker logs, or plans (2026-09-29).
"""

import gzip
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import generic_image_capture as capture
from capability_pipeline import image_capture_control as control
from capability_pipeline import image_pipeline as pipeline
from capability_pipeline import silo_rootfs_capture as silo
from tests.test_generic_image_capture import _fixture, _sha
from tests.test_silo_image_capture import (
    _FakeS3,
    _LocalSandbox,
    _raw,
    _rootfs,
    _SiloWorld,
    _snapshot,
)

# --------------------------------------------------------------------------- #
# Provider error doubles shaped like silo_embedded's (module is not builtins).
# --------------------------------------------------------------------------- #


class SiloRateLimitError(Exception):
    status_code = 429

    def __init__(self, message="sandbox capacity exhausted across 42 hosts; waited 480s", retry_after=30):
        super().__init__(message)
        self.headers = {"Retry-After": str(retry_after)}
        self.error_code = "capacity_exhausted"


class TransportError(Exception):
    status_code = 503

    def __init__(self, message="cannot reach http://10.0.0.1:1/sandboxes/x: timed out", refused=False):
        super().__init__(message)
        self.refused = refused


class SiloNotFoundError(Exception):
    status_code = 404


class SiloError(Exception):
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code


PYTHON_IMAGE_PATH = "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _with_plan(tmp_path, *, env=None, recipe=None):
    """_fixture with a different reviewed Env or recipe; returns (workspace, tools, plan_path)."""
    workspace, tools, plan, plan_path = _fixture(tmp_path)
    image = plan["images"][0]
    if env is not None:
        image["image_config"]["Env"] = env
    if recipe is not None:
        (workspace / "Dockerfile").write_text(recipe)
        digest = _sha(workspace / "Dockerfile")
        image["source_recipe"]["sha256"] = digest
        for row in plan["source_files"]:
            if row["path"] == "Dockerfile":
                row["sha256"] = digest
    plan_path.write_text(json.dumps(plan))
    return workspace, tools, plan_path


def _capture(tmp_path, world, *, env=None, recipe=None, raw_env=None, silo_capture=None, **overrides):
    workspace, tools, plan_path = _with_plan(tmp_path, env=env, recipe=recipe)
    raw = _raw()
    if raw_env is not None:
        raw["source_env"] = raw_env

    def default_capture(**kwargs):
        return json.loads(json.dumps(raw))

    kwargs = {
        "approved_plan_sha256": _sha(plan_path), "dtx": world.dtx(), "capture_tools": tools,
        "runner": lambda *a, **k: pytest.fail("no Daytona capture on silo"),
        "sleeper": world.sleeps.append if hasattr(world, "sleeps") else (lambda _: None),
        "provider": "silo", "client_factory": lambda: world.client,
        "s3_factory": lambda: "harness-s3", "silo_capture": silo_capture or default_capture,
    }
    kwargs.update(overrides)
    return capture.capture_role(plan_path, workspace, "candidate", tmp_path / "receipt.json", **kwargs)


def _world(**kwargs):
    world = _SiloWorld(**kwargs)
    world.sleeps = []
    return world


# --------------------------------------------------------------------------- #
# Environment: 18 of 39 incomplete receipts were an Env "mismatch".
# --------------------------------------------------------------------------- #


def test_python_image_duplicate_path_entry_is_equivalent_not_a_mismatch(tmp_path):
    # hc1 d19.prevention-implementation.delivery-5: python:3.11-slim PATH carries
    # /usr/local/bin twice; the builder reviewed it without the duplicate.
    reviewed = ["PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin", "PYTHONPATH=/opt"]
    sandbox = {"PATH": PYTHON_IMAGE_PATH, "PYTHONPATH": "/opt", "HOSTNAME": "slb-fresh"}
    world = _world(env_text="".join(f"{k}={v}\n" for k, v in sandbox.items()))
    receipt = _capture(tmp_path, world, env=reviewed, raw_env=sandbox)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert receipt["environment"]["equivalences"] == ["PATH: repeated entries ignored (lookup order unchanged)"]
    assert receipt["environment_precheck"]["reviewed_values_matched"] is True


def test_reviewed_home_is_compared_against_the_raw_environment(tmp_path):
    # shard-036 / shard-068 / shard-081: reviewed HOME=/root never matched because
    # the comparison used the HOME-filtered environment.
    reviewed = ["HOME=/root", "LANG=C.UTF-8", "PATH=/usr/local/bin:/usr/bin:/bin"]
    sandbox = {"HOME": "/root", "LANG": "C.UTF-8", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    world = _world(env_text="".join(f"{k}={v}\n" for k, v in sandbox.items()))
    receipt = _capture(tmp_path, world, env=reviewed, raw_env=sandbox)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert "HOME" not in receipt["environment"]["additional_filtered_names"]


def test_builder_env_typo_is_content_and_stops_before_the_archive(tmp_path):
    # hc1 d19.global-equity.intervention-1: '/usr/bin/sbin' typo in the reviewed PATH.
    reviewed = ["PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin/sbin:/bin"]
    world = _world(env_text=f"PATH={PYTHON_IMAGE_PATH}\n")
    receipt = _capture(tmp_path, world, env=reviewed,
                       silo_capture=lambda **k: pytest.fail("a known mismatch must not stream the rootfs"))
    assert receipt["state"] == "image_config_mismatch"
    assert receipt["failure_class"] == "content"
    [row] = receipt["environment_mismatch"]
    assert row["name"] == "PATH" and row["observed"] == PYTHON_IMAGE_PATH
    assert receipt["cleanup"]["absence_verified"] is True


def test_silo_multi_assignment_env_parse_is_named_in_the_mismatch(tmp_path):
    # shard-076 d20.formulation.compatibility-8: `ENV A=1 B=1 C=1` became A='1 B=1 C=1'.
    reviewed = ["PYTHONUNBUFFERED=1", "PIP_NO_CACHE_DIR=1"]
    world = _world(env_text="PYTHONUNBUFFERED=1      PIP_NO_CACHE_DIR=1\n")
    receipt = _capture(tmp_path, world, env=reviewed)
    assert receipt["state"] == "image_config_mismatch"
    rows = {row["name"]: row for row in receipt["environment_mismatch"]}
    assert "multi-assignment ENV line" in rows["PYTHONUNBUFFERED"]["hint"]
    assert rows["PIP_NO_CACHE_DIR"]["observed"] is None


def test_secret_looking_env_values_are_not_recorded(tmp_path):
    world = _world(env_text="FOO=bar\nAPI_TOKEN=live-token-value-123\n")
    receipt = _capture(tmp_path, world, env=["FOO=bar", "API_TOKEN=reviewed-token"])
    assert receipt["state"] == "image_config_mismatch"
    assert "live-token-value-123" not in json.dumps(receipt)


# --------------------------------------------------------------------------- #
# Cleanup: 20 successful archives were thrown away as cleanup_unverified.
# --------------------------------------------------------------------------- #


def test_delete_timeout_is_followed_by_an_absence_lookup(tmp_path):
    # 17 receipts: archive uploaded, then delete_error:TransportError and no lookup.
    world = _world()
    calls = []

    class Box:
        id = "slb-fresh"
        network_block_all = True

        def delete(self):
            calls.append("delete")
            if len(calls) == 1:
                raise TransportError()

    world.client.create = lambda params, timeout: Box()
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert receipt["cleanup"]["absence_verified"] is True
    assert receipt["cleanup"]["observations"][0] == {"state": "delete_error", "error_type": "TransportError"}


def test_not_found_on_delete_counts_as_already_gone(tmp_path):
    # 3 receipts: delete_error:SiloNotFoundError (the sandbox was already reaped).
    world = _world()

    class Box:
        id = "slb-fresh"
        network_block_all = True

        def delete(self):
            raise SiloNotFoundError("sandbox 'slb-fresh' not found")

    world.client.create = lambda params, timeout: Box()
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert receipt["cleanup"]["observations"][0] == {"state": "not_found_on_delete"}


def test_unverifiable_cleanup_keeps_the_underlying_outcome(tmp_path):
    world = _world()

    def get(sandbox_id):
        raise TransportError()

    world.client.get = get
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "cleanup_unverified"
    assert receipt["state_before_cleanup"] == "captured_pending_privacy_and_publication"
    assert receipt["failure_class"] == "transient" and receipt["failed_step"] == "cleanup"
    assert sum(world.sleeps) >= 150  # about three minutes of lookups, not one


# --------------------------------------------------------------------------- #
# Sandbox creation: broker "AT CAPACITY", 429 after a 480 s placement wait.
# --------------------------------------------------------------------------- #


def test_capacity_refusals_are_retried_with_the_advertised_delay(tmp_path):
    world = _world()
    original = world.client.create
    failures = [SiloRateLimitError(retry_after=30), SiloRateLimitError(retry_after=45)]

    def create(params, timeout):
        if failures:
            raise failures.pop(0)
        return original(params, timeout)

    world.client.create = create
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    attempts = receipt["sandbox_create_attempts"]
    assert [row.get("status_code") for row in attempts] == [429, 429, None]
    assert world.sleeps[:2] == [30.0, 45.0]
    assert world.sandbox_params[0]["labels"]["envgen_capture"] == receipt["capture_label"]


def test_exhausted_capacity_is_a_transient_provision_failure_without_a_receipt(tmp_path):
    world = _world()

    def create(params, timeout):
        raise SiloRateLimitError()

    world.client.create = create
    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, world)
    error = failure.value
    assert error.step == "sandbox_create" and error.failure_class == "transient"
    assert error.error_type == "SiloRateLimitError" and len(error.detail["attempts"]) == 3
    assert not (tmp_path / "receipt.json").exists()


def test_ambiguous_create_timeout_reaps_the_labelled_sandbox_before_retrying(tmp_path):
    world = _world()
    original = world.client.create
    leaked = []

    class Leaked:
        id = "slb-leaked"

        def __init__(self, label):
            self.labels = {"envgen_capture": label}

        def delete(self):
            leaked.append(self.id)

    state = {"first": True}

    def create(params, timeout):
        if state["first"]:
            state["first"] = False
            state["label"] = params["labels"]["envgen_capture"]
            raise TransportError()
        return original(params, timeout)

    world.client.create = create
    world.client.list = lambda: [Leaked(state["label"]), SimpleNamespace(id="other", labels={})]
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert leaked == ["slb-leaked"]
    assert receipt["sandbox_create_attempts"][0]["labelled_sandboxes_reaped"]["matched"] == 1


def test_a_refused_request_is_our_harness_not_capacity(tmp_path):
    world = _world()

    def create(params, timeout):
        raise SiloError("bad request", status_code=400)

    world.client.create = create
    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, world)
    assert failure.value.failure_class == "harness"


# --------------------------------------------------------------------------- #
# Snapshot re-materialisation after the broker restart (05:26Z).
# --------------------------------------------------------------------------- #


def test_provider_local_base_image_is_content_not_infrastructure(tmp_path):
    # broker16.log: "lookup silo.local ... no such host" for FROM silo.local/snapshots/... recipes.
    recipe = "FROM silo.local/snapshots/cap-d01-fva9-solver-base-v2:built\nRUN true\n"
    world = _world(snapshot=_snapshot(recipe=recipe), missing=True)

    def failing_create(params, timeout):
        raise SiloError("snapshot 'snap' failed: nerdctl pull -q exited 1", status_code=500)

    world.client.snapshot.create = failing_create
    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, world, recipe=recipe)
    assert failure.value.failure_class == "content"
    assert failure.value.detail["base_image"].startswith("silo.local/")


def test_building_snapshot_is_waited_for(tmp_path):
    world = _world()
    record = world.snapshot_record
    states = iter(["building", "building", "active"])
    world.client.snapshot.get = lambda name: SimpleNamespace(**{**vars(record), "state": next(states)})
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert world.sleeps[:2] == [capture.SNAPSHOT_POLL_SECONDS] * 2


def test_failed_snapshot_with_the_reviewed_recipe_is_rebuilt_once(tmp_path):
    world = _world()
    record = world.snapshot_record
    deleted = []
    states = iter(["error", "active"])
    world.client.snapshot.get = lambda name: SimpleNamespace(**{**vars(record), "state": next(states)})
    world.client.snapshot.delete = deleted.append
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "captured_pending_privacy_and_publication", receipt
    assert deleted == ["snap"] and len(world.created_snapshots) == 1
    assert receipt["source_snapshot_rematerialized"] is True


# --------------------------------------------------------------------------- #
# In-sandbox checks: an unanswered check is never a verdict about the image.
# --------------------------------------------------------------------------- #


def test_unanswered_readiness_is_transient_and_failed_readiness_is_content(tmp_path):
    world = _world()
    world.exit_overrides["test -f /tmp/task-ready"] = None
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "infrastructure_error" and receipt["failure_class"] == "transient"
    world2 = _world()
    world2.exit_overrides["test -f /tmp/task-ready"] = 1
    (tmp_path / "second").mkdir()
    receipt2 = _capture(tmp_path / "second", world2)
    assert receipt2["state"] == "capture_content_error" and receipt2["failed_step"] == "readiness"
    assert receipt2["ready"]["exit_codes"] == [1, 1]


def test_unanswered_privacy_scan_is_not_a_privacy_finding(tmp_path):
    world = _world()
    world.exit_overrides["test ! -e /opt/evaluator"] = None
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "infrastructure_error", receipt
    assert receipt["failed_step"] == "sensitive_paths"


def test_broken_provider_mount_model_is_a_harness_failure(tmp_path):
    world = _world()
    world.exit_overrides["stat -c %d /.silo"] = 1
    receipt = _capture(tmp_path, world)
    assert receipt["state"] == "capture_harness_error" and receipt["failure_class"] == "harness"


def test_rejected_archive_keeps_the_tar_evidence(tmp_path):
    # shard-026 d02.pl.runtime_memory-5: RuntimeError after the archive, no tar facts kept.
    raw = _raw()
    raw["capture"]["tar_exit"] = 2
    raw["capture"]["tar_stderr_tail"] = "tar: ./proc/1: Cannot open: Permission denied\n"
    world = _world()
    receipt = _capture(tmp_path, world, silo_capture=lambda **k: json.loads(json.dumps(raw)))
    assert receipt["state"] == "infrastructure_error" and receipt["failed_step"] == "archive_process"
    facts = receipt["failure_detail"]["archive_failure"]
    assert facts["tar_exit"] == 2 and "Permission denied" in facts["tar_stderr_tail"]


def test_transport_failure_records_its_own_message_and_kind(tmp_path):
    # hc3 d19.biostatistics...longitudinal-3: SiloCaptureError, nothing else recorded.
    def stalled(**kwargs):
        raise silo.SiloCaptureError("rootfs capture produced no part within the stall window", kind="stall")

    receipt = _capture(tmp_path, _world(), silo_capture=stalled)
    assert receipt["state"] == "infrastructure_error"
    assert receipt["error_type"] == "SiloCaptureError"
    assert receipt["error_message"] == "rootfs capture produced no part within the stall window"
    assert receipt["failure_detail"]["kind"] == "stall"


def test_missing_harness_dependency_fails_before_any_sandbox(tmp_path):
    # shard-081 d22.thermal_process-9: ModuleNotFoundError after the sandbox was made.
    world = _world()

    def s3_factory():
        raise ModuleNotFoundError("No module named 'botocore'")

    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, world, s3_factory=s3_factory)
    assert failure.value.failure_class == "harness" and failure.value.step == "object_store_client"
    assert world.sandbox_params == []


def test_provider_client_exit_on_missing_credentials_is_harness(tmp_path):
    def client_factory():
        raise SystemExit("dt: SILO_API_TOKEN is not set")

    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, _world(), client_factory=client_factory)
    assert failure.value.failure_class == "harness"


# --------------------------------------------------------------------------- #
# Transport: guests without python3 (rocker/r-ver, r-base, node images).
# --------------------------------------------------------------------------- #

_GNUBIN = Path("/opt/homebrew/opt/coreutils/libexec/gnubin")


def _gnu_path() -> str | None:
    for prefix in ("", str(_GNUBIN) + os.pathsep):
        path = prefix + os.environ.get("PATH", "")
        probe = subprocess.run(["sh", "-c", "dd if=/dev/null of=/dev/null iflag=fullblock && stat -c %s /dev/null"],
                               capture_output=True, env={**os.environ, "PATH": path}, check=False)
        if probe.returncode == 0 and shutil.which("sha256sum", path=path) and shutil.which("mkfifo", path=path):
            return path
    return None


@pytest.mark.skipif(_gnu_path() is None or shutil.which("tar") is None, reason="needs GNU coreutils and tar")
def test_shell_helper_streams_a_content_addressed_archive_without_python(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _gnu_path())
    root = _rootfs(tmp_path)
    tools = ("tar", "gzip", "gnu-dd", "stat", "sha256sum", "mkfifo", "tee")
    sandbox = _LocalSandbox(tmp_path / "sandbox", root, tools=tools)
    s3 = _FakeS3()
    raw = silo.capture_rootfs(
        sandbox=sandbox, sh=sandbox.sh, snapshot_name="snap", tag="cap-sh",
        excludes=["./tmp/*", *silo.PROVIDER_EXCLUDES], s3=s3, gzip_level=1,
        part_bytes=64 * 1024, max_outstanding=2, poll_seconds=0.01, sleeper=lambda s: None,
        root=str(root),
    )
    assert raw["stream_helper"] == "posix-sh" and raw["ok"] is True
    assert raw["capture"]["parts"] > 2 and raw["capture"]["tar_exit"] == 0
    data = s3.objects[raw["object_key"]]
    assert raw["object_bytes"] == len(data) == raw["capture"]["compressed_bytes"]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as archive:
        names = {name.removeprefix("./") for name in archive.getnames()}
    assert "opt/verifier/READY" in names and "data.bin" in names
    assert "tmp/scratch" not in names and not any(name.startswith(".silo/") for name in names)


def test_guest_without_stream_tools_is_a_loud_harness_failure(tmp_path):
    sandbox = _LocalSandbox(tmp_path / "sandbox", _rootfs(tmp_path), tools=("tar",))
    s3 = _FakeS3()
    with pytest.raises(silo.SiloCaptureError) as failure:
        silo.capture_rootfs(sandbox=sandbox, sh=sandbox.sh, snapshot_name="snap", tag="cap-none",
                            excludes=[], s3=s3, poll_seconds=0.01, sleeper=lambda s: None, root=str(tmp_path))
    assert failure.value.kind == "guest_tools_missing" and failure.value.failure_class == "harness"
    assert "gnu-dd" in failure.value.detail["missing"]
    assert not s3.parts and not s3.aborted  # no upload was ever started


# --------------------------------------------------------------------------- #
# The capture CLI's structured failure line.
# --------------------------------------------------------------------------- #


def test_cli_failure_line_is_classified_and_secret_free():
    from scripts import capture_generic_task_image as cli

    cli._STAGE["name"] = "capture"
    content = cli.failure_line(capture.CaptureProvisionError(
        "the reviewed recipe builds FROM a provider-local snapshot image", step="snapshot",
        failure_class="content", detail={"base_image": "silo.local/x"}))
    assert content["state"] == "failed" and content["stage"] == "snapshot"
    assert content["failure_class"] == "content" and "provider-local" in content["error_message"]
    cli._STAGE["name"] = "credentials"
    missing = cli.failure_line(ValueError("trusted capture credentials are unavailable"))
    assert missing["failure_class"] == "harness" and missing["stage"] == "credentials"
    cli._STAGE["name"] = "capture"
    provider = cli.failure_line(TransportError("cannot reach http://token@broker"))
    assert provider["failure_class"] == "transient" and "token@broker" not in json.dumps(provider)


# --------------------------------------------------------------------------- #
# The controller: evidence, budget, explicit outcomes, recovery of parked items.
# --------------------------------------------------------------------------- #


def _controller(tmp_path, monkeypatch, name="run-a"):
    item = tmp_path / name / "items/task-1"
    (item / "workspace/task").mkdir(parents=True)
    attempt = item / "diagnostics/image-capture/attempt-source"
    (attempt / "input").mkdir(parents=True)
    plan_path = attempt / "input/plan.json"
    plan_path.write_text(json.dumps({"images": [{"role": "candidate", "source_snapshot": {
        "name": "snap", "id": "snapshot-id", "ref": "provider-ref"}}]}))
    review_base = tmp_path / name / "image-reviews/task-1"
    (review_base / "attempt-source").mkdir(parents=True)
    (review_base / "attempt-source/approval.json").write_text("{}")
    monkeypatch.setattr(pipeline, "request_needed", lambda _: {"needed": True})
    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda root, *_: {
        "attempt": str(Path(root) / "diagnostics/image-capture/attempt-source"),
        "plan_path": str(Path(root) / "diagnostics/image-capture/attempt-source/input/plan.json"),
        "workspace": str(tmp_path)})
    monkeypatch.setattr(pipeline, "validate_review", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "_review_matches_task", lambda *a: None)
    monkeypatch.delenv(control.MAX_ATTEMPTS_ENV, raising=False)
    return item, attempt, plan_path, review_base


def _run(item, review_base, runner):
    return pipeline.process_image_construction(
        item_root=item, capture_tools=item, scripts_root=item, agent=object(),
        builder_session_ids={"builder"}, command_runner=runner, review_base=review_base)


def _receipt_doc(plan_path, **fields):
    doc = {"schema_version": "capability-rootfs-capture-v1", "role": "candidate",
           "plan_sha256": pipeline._sha(plan_path),
           "source_snapshot": {"name": "snap", "id": "snapshot-id", "ref": "provider-ref"},
           "cleanup": {"absence_verified": True}}
    doc.update(fields)
    return doc


def _writer(plan_path, **fields):
    def runner(command):
        Path(command[command.index("--output") + 1]).write_text(json.dumps(_receipt_doc(plan_path, **fields)))
        return SimpleNamespace(returncode=0 if fields.get("state") == capture_ok() else 1, stdout="", stderr="")
    return runner


def capture_ok():
    return "captured_pending_privacy_and_publication"


def test_failed_command_leaves_redacted_evidence_and_a_retryable_status(tmp_path, monkeypatch):
    item, attempt, _, review_base = _controller(tmp_path, monkeypatch)
    monkeypatch.setenv("SILO_API_TOKEN", "super-secret-silo-token-value")
    line = {"state": "failed", "stage": "sandbox_create", "failure_class": "transient",
            "error_type": "SiloRateLimitError", "status_code": 429,
            "error_message": "capture sandbox could not be created"}

    def runner(command):
        return subprocess.CompletedProcess(command, 1, stdout=json.dumps(line) + "\n",
                                           stderr="Traceback: token super-secret-silo-token-value\n"
                                                  "Authorization: Bearer abc.def.ghi\n")

    result = _run(item, review_base, runner)
    assert result["state"] == "pending_capture" and result["retryable"] is True
    assert result["reason"] == "capture_command_failed"
    assert result["error_type"] == "SiloRateLimitError" and result["stage"] == "sandbox_create"
    assert result["exit_code"] == 1 and result["attempts"] == 1 and result["max_attempts"] == 6
    assert result["backoff_seconds"] >= control.CAPACITY_BACKOFF_FLOOR_SECONDS
    log = Path(result["evidence_path"]).read_text()
    assert "super-secret-silo-token-value" not in log and "abc.def.ghi" not in log
    assert "<redacted:SILO_API_TOKEN>" in log and "SiloRateLimitError" in log
    assert Path(result["evidence_path"]).parent == attempt
    record = json.loads((attempt / "capture-candidate.attempt-1.json").read_text())
    assert record["state"] == "finished" and record["failure_class"] == "transient"


def test_item_parked_by_the_old_three_strike_cap_captures_again(tmp_path, monkeypatch):
    item, attempt, plan_path, review_base = _controller(tmp_path, monkeypatch)
    legacy = _receipt_doc(plan_path, state="infrastructure_error", error_type="RuntimeError")
    for index in (1, 2, 3):
        (attempt / f"capture-candidate.infrastructure-{index}.json").write_text(json.dumps(legacy))
    (attempt / "capture-candidate.json").write_text(json.dumps(legacy))  # the parked live receipt
    result = _run(item, review_base, _writer(plan_path, state=capture_ok()))
    assert result["state"] == "pending_publication", result
    assert (attempt / "capture-candidate.infrastructure-4.json").is_file()
    assert json.loads((attempt / "capture-candidate.json").read_text())["state"] == capture_ok()


def test_unverified_cleanup_is_retried_with_the_old_sandbox_to_reap(tmp_path, monkeypatch):
    item, attempt, plan_path, review_base = _controller(tmp_path, monkeypatch)
    (attempt / "capture-candidate.json").write_text(json.dumps(_receipt_doc(
        plan_path, state="cleanup_unverified", sandbox_id="slb-old",
        cleanup={"absence_verified": False, "observations": [{"state": "delete_error", "error_type": "TransportError"}]})))
    seen = []

    def runner(command):
        seen.append(command)
        return _writer(plan_path, state=capture_ok())(command)

    assert _run(item, review_base, runner)["state"] == "pending_publication"
    assert seen[0][seen[0].index("--prior-sandbox-id") + 1] == "slb-old"


def test_harness_failure_is_loud_and_blocked_until_a_relaunch(tmp_path, monkeypatch):
    item, attempt, plan_path, review_base = _controller(tmp_path, monkeypatch)
    line = {"state": "failed", "stage": "credentials", "failure_class": "harness", "error_type": "ValueError",
            "error_message": "trusted capture credentials are unavailable"}
    first = _run(item, review_base, lambda c: subprocess.CompletedProcess(c, 1, json.dumps(line), ""))
    assert first["state"] == "pending_capture" and first["retryable"] is False
    assert first["reason"] == "capture_harness_error" and first["blocked_until"] == "job_relaunch"
    assert first["stage"] == "credentials"
    again = _run(item, review_base, lambda c: pytest.fail("no silent retry inside the same job"))
    assert again["reason"] == "capture_harness_error"
    # A relaunched job restores the same tree under a new results root and retries.
    relaunched = tmp_path / "run-b"
    shutil.copytree(tmp_path / "run-a", relaunched)
    item_b = relaunched / "items/task-1"
    result = _run(item_b, relaunched / "image-reviews/task-1", _writer(
        item_b / "diagnostics/image-capture/attempt-source/input/plan.json", state=capture_ok()))
    assert result["state"] == "pending_publication", result


def test_env_mismatch_receipt_is_a_builder_repair(tmp_path, monkeypatch):
    item, _, plan_path, review_base = _controller(tmp_path, monkeypatch)
    rows = [{"name": "PATH", "reviewed": "/usr/local/sbin:/usr/local/bin", "observed": PYTHON_IMAGE_PATH,
             "hint": "the reviewed image_config.Env value differs from the snapshot's own environment"}]
    result = _run(item, review_base, _writer(plan_path, state="image_config_mismatch", failure_class="content",
                                             environment_mismatch=rows, failed_step="environment"))
    assert result["state"] == "repairable" and result["reason"] == "image_config_env_mismatch"
    assert "PATH" in result["issues"][0] and "image-capture-request.json" in result["issues"][0]


def test_readiness_failure_is_confirmed_once_before_a_repair(tmp_path, monkeypatch):
    item, _, plan_path, review_base = _controller(tmp_path, monkeypatch)
    runner = _writer(plan_path, state="capture_content_error", failure_class="content", failed_step="readiness",
                     error_message="capture sandbox did not become ready", failure_detail={"exit_codes": [1, 1]})
    first = _run(item, review_base, runner)
    assert first["state"] == "pending_capture" and first["reason"] == "capture_content_check_unconfirmed"
    second = _run(item, review_base, runner)
    assert second["state"] == "repairable" and second["reason"] == "image_capture_content_check_failed"
    assert "readiness" in second["issues"][0]


def test_unreproducible_recipe_without_a_receipt_is_a_builder_repair(tmp_path, monkeypatch):
    item, _, _, review_base = _controller(tmp_path, monkeypatch)
    line = {"state": "failed", "stage": "snapshot", "failure_class": "content", "error_type": "SiloError",
            "error_message": "the reviewed recipe builds FROM a provider-local snapshot image that no registry serves",
            "detail": {"base_image": "silo.local/snapshots/base:built"}}
    result = _run(item, review_base, lambda c: subprocess.CompletedProcess(c, 1, json.dumps(line), ""))
    assert result["state"] == "repairable" and result["reason"] == "image_capture_input_unusable"
    assert "silo.local/snapshots/base:built" in result["issues"][0]


def test_killed_command_without_output_is_retryable(tmp_path, monkeypatch):
    item, _, _, review_base = _controller(tmp_path, monkeypatch)
    result = _run(item, review_base, lambda c: subprocess.CompletedProcess(c, 124, "", "killed"))
    assert result["state"] == "pending_capture" and result["retryable"] is True
    assert result["error_type"] == "timeout"


def test_budget_is_configurable_and_ends_terminal(tmp_path, monkeypatch):
    item, _, _, review_base = _controller(tmp_path, monkeypatch)
    monkeypatch.setenv(control.MAX_ATTEMPTS_ENV, "2")
    runner = lambda c: subprocess.CompletedProcess(c, 1, "", "uv: failed to spawn")  # noqa: E731
    assert _run(item, review_base, runner)["state"] == "pending_capture"
    final = _run(item, review_base, runner)
    assert final["state"] == "failed_terminal" and final["max_attempts"] == 2
    assert final["reason"] == "image_capture_failed: capture_command_failed:unparsed_output"


def test_controller_crash_after_the_command_still_counts_the_attempt(tmp_path, monkeypatch):
    item, attempt, plan_path, review_base = _controller(tmp_path, monkeypatch)
    (attempt / "capture-candidate.attempt-1.json").write_text(json.dumps(
        {"schema_version": control.LEDGER_SCHEMA, "role": "candidate", "attempt": 1, "state": "started"}))
    (attempt / "capture-candidate.json").write_text(json.dumps(_receipt_doc(
        plan_path, state="infrastructure_error", failure_class="transient")))
    result = _run(item, review_base, _writer(plan_path, state=capture_ok()))
    assert result["state"] == "pending_publication"
    assert (attempt / "capture-candidate.attempt-1.receipt.json").is_file()
    assert json.loads((attempt / "capture-candidate.attempt-1.json").read_text())["state"] == "finished"
    assert (attempt / "capture-candidate.attempt-2.json").is_file()


def test_default_runner_kills_a_hung_command_and_its_children(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PROJECT", str(tmp_path))
    monkeypatch.setenv(pipeline.COMMAND_TIMEOUT_ENV, "1")
    real = subprocess.Popen

    def popen(argv, **kwargs):
        return real(["sh", "-c", "echo started; sleep 30 & sleep 30"], **kwargs)

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    completed = pipeline.default_cli_runner(["capture_generic_task_image.py"])
    assert completed.returncode == 124
    assert completed.stdout.strip() == "started" and "killed" in completed.stderr


def test_redaction_covers_presigned_urls_and_access_keys():
    text = ("https://b.cwobject.com/k?X-Amz-Credential=AKIAABCDEFGHIJKLMNOP%2F&X-Amz-Signature=deadbeef "
            "aws_secret_access_key=abcd1234secret")
    cleaned = control.redact(text)
    assert "deadbeef" not in cleaned and "AKIAABCDEFGHIJKLMNOP" not in cleaned and "abcd1234secret" not in cleaned


def test_missing_module_name_is_recorded_for_a_harness_import_failure(tmp_path):
    def s3_factory():
        raise ModuleNotFoundError("No module named 'botocore'", name="botocore")

    with pytest.raises(capture.CaptureProvisionError) as failure:
        _capture(tmp_path, _world(), s3_factory=s3_factory)
    assert failure.value.detail["missing_module"] == "botocore"
    assert capture.error_summary(ModuleNotFoundError("x", name="botocore"))["missing_module"] == "botocore"
