import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline.fixed_grading_capture import write_capture
from capability_pipeline.grading_input import grading_input_fingerprint


def _module(monkeypatch):
    class Parent:
        async def _verify(self):
            return "parent-result"

    upstream = types.ModuleType("taskcompendium.harbor.verifier")
    upstream.__file__ = __file__
    upstream.SemanticVerifier = Parent
    taskcompendium = types.ModuleType("taskcompendium")
    taskcompendium.__file__ = __file__
    harbor = types.ModuleType("taskcompendium.harbor")
    harbor.verifier = upstream
    taskcompendium.harbor = harbor
    tasktrove = types.ModuleType("tasktrove_verify")
    daytona = types.ModuleType("daytona")
    daytona.CreateSandboxFromSnapshotParams = lambda **kwargs: kwargs
    daytona.CreateSnapshotParams = lambda **kwargs: kwargs
    daytona.Image = SimpleNamespace(from_dockerfile=lambda value: value)
    daytona.Resources = lambda **kwargs: kwargs
    paths = types.ModuleType("taskcompendium.grading_paths")
    paths.EXTERNAL_DIRECTORY = "__external__"
    lowering = types.ModuleType("taskcompendium.lowering")
    lowering.validate_workspace_submission = lambda *args: None
    models = types.ModuleType("taskcompendium.models")
    models.ContainerRuntime = type("ContainerRuntime", (), {})
    models.Embedded = lambda value: value
    models.GradingResult = lambda *args: SimpleNamespace(
        status=args[0], reward=args[1], detail=args[2]
    )
    models.ImageOverlay = type("ImageOverlay", (), {})
    models.Outcome = SimpleNamespace(INFRA_ERROR="infra")
    models.ResourceRef = type("ResourceRef", (), {})
    models.ResourceRole = SimpleNamespace(VERIFIER="verifier")
    models.TaskSpec = object
    models.verifier_runtime = lambda value: value
    resources = types.ModuleType("taskcompendium.resources")
    resources.resource_bytes = lambda value: value
    serialization = types.ModuleType("taskcompendium.serialization")
    serialization.from_json = lambda value: ("spec", value)
    serialization.rendering_from_json = lambda value: ("protocol", value)
    serialization.to_json = lambda value: value
    msgspec = types.ModuleType("msgspec")
    msgspec.to_builtins = lambda value: value
    msgspec.json = SimpleNamespace(encode=lambda value: json.dumps(value).encode())
    msgspec.structs = SimpleNamespace(replace=lambda value, **changes: value)
    httpx = types.ModuleType("httpx")
    httpx.get = lambda *args, **kwargs: None
    composite = types.ModuleType("capability_pipeline.composite_extension")
    composite.BASE_VERIFIER_SHA256 = "a" * 64
    composite.PATCHED_VERIFIER_SHA256 = "b" * 64
    environment = types.ModuleType("capability_pipeline.daytona_environment")
    environment._dt = lambda: None
    environment._run_portable = lambda *args: None
    policy = types.ModuleType("capability_pipeline.daytona_policy")
    policy.verifier_bootstrap_sha256 = lambda *args: "bootstrap"
    policy.verifier_snapshot_recipe = lambda *args: "recipe"
    dresources = types.ModuleType("capability_pipeline.daytona_resources")
    dresources.VERIFIER_DEFAULT = object()
    dresources.resolve_profile = lambda *args, **kwargs: None
    dresources.snapshot_name = lambda *args: "snapshot"
    snapshot = types.ModuleType("capability_pipeline.daytona_snapshot")
    snapshot.snapshot_conflict = snapshot.snapshot_not_found = lambda error: False
    snapshot.wait_for_snapshot_active = lambda *args, **kwargs: None
    snapshot.wait_for_sandbox_deletion = lambda *args, **kwargs: ("not_found", [{"state": "not_found"}])
    retry = types.ModuleType("capability_pipeline.provider_retry")
    retry.provision_with_rate_limit_retry = lambda callback: (callback(), 1)
    for name, value in {
        "msgspec": msgspec,
        "httpx": httpx,
        "taskcompendium": taskcompendium,
        "taskcompendium.harbor": harbor,
        "taskcompendium.harbor.verifier": upstream,
        "taskcompendium.grading_paths": paths,
        "taskcompendium.lowering": lowering,
        "taskcompendium.models": models,
        "taskcompendium.resources": resources,
        "taskcompendium.serialization": serialization,
        "tasktrove_verify": tasktrove,
        "daytona": daytona,
        "capability_pipeline.composite_extension": composite,
        "capability_pipeline.daytona_environment": environment,
        "capability_pipeline.daytona_policy": policy,
        "capability_pipeline.daytona_resources": dresources,
        "capability_pipeline.daytona_snapshot": snapshot,
        "capability_pipeline.provider_retry": retry,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    sys.modules.pop("capability_pipeline.daytona_verifier", None)
    return importlib.import_module("capability_pipeline.daytona_verifier")


def test_verifier_snapshot_python_incompatibility_uses_provider_log(monkeypatch):
    module = _module(monkeypatch)
    raw = (
        b"Downloading jiter-0.16.0-cp311-manylinux.whl\n"
        b"Ignored versions: numpy 2.5.3 Requires-Python >=3.12\n"
        b"ERROR: No matching distribution found for numpy==2.5.3\n"
    )
    snapshot = SimpleNamespace(
        name="snap", id="provider-id", state="ERROR",
        build_info=SimpleNamespace(dockerfile_content="recipe"),
    )
    api = SimpleNamespace(
        get_snapshot_build_logs_url=lambda _: SimpleNamespace(url="https://daytonaproxy01.net/logs"),
        api_client=SimpleNamespace(default_headers={"Authorization": "Bearer test"}),
    )
    service = SimpleNamespace(get=lambda _: snapshot)
    service._SnapshotService__snapshots_api = api
    client = SimpleNamespace(snapshot=service)
    monkeypatch.setattr(
        module.httpx, "get",
        lambda *args, **kwargs: SimpleNamespace(
            content=raw, raise_for_status=lambda: None
        ),
    )
    error = module._compatibility_failure(client, "snap", "recipe")
    assert isinstance(error, module.VerifierImageCompatibilityError)
    assert error.reason == "supervisor_python_too_old"
    assert error.log_sha256 == __import__("hashlib").sha256(raw).hexdigest()
    assert "daytonaproxy01.net" not in str(error)
    assert module._compatibility_failure(client, "snap", "other recipe") is None
    monkeypatch.setattr(
        module.httpx, "get",
        lambda *args, **kwargs: SimpleNamespace(
            content=b"ERROR: transient network failure", raise_for_status=lambda: None
        ),
    )
    assert module._compatibility_failure(client, "snap", "recipe") is None
    api.get_snapshot_build_logs_url = lambda _: SimpleNamespace(
        url="https://untrusted.invalid/logs"
    )
    monkeypatch.setattr(
        module.httpx, "get",
        lambda *args, **kwargs: pytest.fail("unknown host received an authenticated request"),
    )
    assert module._compatibility_failure(client, "snap", "recipe") is None


def _capture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    specification, protocol, response, transcript = b'{"x":1}', b'{"y":2}', "ok", ()
    payload = json.dumps(
        {
            "specification": {"x": 1},
            "protocol": {"y": 2},
            "step_index": 0,
            "attempt": {"response": response, "transcript": transcript},
        }
    ).encode()
    fingerprint = grading_input_fingerprint(
        specification, {"y": 2}, response, workspace, transcript, payload=payload
    )
    root = tmp_path / "capture"
    receipt = write_capture(
        root,
        specification=specification,
        protocol=protocol,
        response=response,
        transcript=transcript,
        payload=payload,
        workspace=workspace,
        fingerprint=fingerprint,
        source_specification_sha256="a" * 64,
        source_renderings_sha256="b" * 64,
    )
    return root, receipt, payload


def test_grade_captured_passes_raw_payload_and_manifest_identity(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    root, receipt, payload = _capture(tmp_path)
    calls = []

    def grade(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(detail={"grading_input_fingerprint": receipt["fingerprint"]})

    monkeypatch.setattr(module, "grade_in_daytona", grade)
    result = module.grade_captured_in_daytona(
        root, expected_manifest_sha256=receipt["manifest_sha256"]
    )
    assert calls[0][1]["payload"] == payload
    assert calls[0][1]["embedded"] is True
    assert result.detail["fixed_grading_capture_manifest_sha256"] == receipt["manifest_sha256"]


def test_capturing_verifier_scopes_capture_to_one_trial(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    root = tmp_path / "task"
    root.mkdir()
    (root / "specification.json").write_text("{}")
    (root / "renderings.json").write_text("{}")
    verifier = object.__new__(module.CapturingDaytonaSemanticVerifier)
    verifier.task = SimpleNamespace(paths=SimpleNamespace(task_dir=root))
    verifier.trial_paths = SimpleNamespace(verifier_dir=tmp_path / "trial-verifier")
    source_hash = module.hashlib.sha256(Path(module._upstream_verifier.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(module, "BASE_VERIFIER_SHA256", source_hash)
    assert asyncio.run(verifier._verify()) == "parent-result"
    assert module._CAPTURE_ROOT.get() is None
    assert module._CAPTURE_SOURCE.get() is None


def test_grader_rejects_payload_that_does_not_describe_current_delivery(
    monkeypatch, tmp_path
):
    module = _module(monkeypatch)
    runtime = module.ContainerRuntime()
    runtime.image, runtime.supervisor_python, runtime.timeout = "image", "python", 1
    specification = SimpleNamespace(
        steps=(SimpleNamespace(verifier=runtime),),
        requirements=SimpleNamespace(
            state=SimpleNamespace(additional_directories=())
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(module, "to_json", lambda value: b'{"spec":true}')
    with pytest.raises(ValueError, match="does not bind"):
        module.grade_in_daytona(
            specification,
            {"protocol": True},
            "answer",
            workspace,
            embedded=True,
            payload=b"{}",
        )


def test_private_cleanup_transient_error_then_confirmed_absent_never_regrades(monkeypatch):
    module = _module(monkeypatch)
    calls = []

    class DaytonaError(Exception):
        pass

    def delete():
        calls.append("delete")
        if len(calls) == 1:
            raise DaytonaError("provider state change")

    observations = iter([
        ("present", [{"state": "present"}]),
        ("not_found", [{"state": "not_found"}]),
    ])
    monkeypatch.setattr(module, "wait_for_sandbox_deletion", lambda *a, **k: next(observations))
    receipt = module._cleanup_private_sandbox(SimpleNamespace(delete=delete), object(), "id")
    assert calls == ["delete", "delete"]
    assert receipt["state"] == "deleted"
    assert [attempt["state"] for attempt in receipt["attempts"]] == [
        "delete_error", "delete_requested"
    ]


def test_private_cleanup_permanent_error_keeps_bounded_unconfirmed_receipt(monkeypatch):
    module = _module(monkeypatch)
    calls = []

    def delete():
        calls.append("delete")
        raise RuntimeError("provider deletion failed")

    monkeypatch.setattr(module, "wait_for_sandbox_deletion", lambda *a, **k: (
        "present", [{"state": "present"}]
    ))
    receipt = module._cleanup_private_sandbox(SimpleNamespace(delete=delete), object(), "id")
    assert calls == ["delete", "delete"]
    assert receipt["state"] == "unconfirmed"
    assert [attempt["error_type"] for attempt in receipt["attempts"]] == [
        "RuntimeError", "RuntimeError"
    ]


def test_completed_grade_survives_unconfirmed_cleanup(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    module.tasktrove_verify.__file__ = __file__
    runtime = module.ContainerRuntime()
    runtime.image = "image"
    runtime.supervisor_python = "python3"
    runtime.timeout = 60
    runtime.workspace = object()
    specification = SimpleNamespace(
        steps=(SimpleNamespace(verifier=runtime),),
        requirements=SimpleNamespace(state=SimpleNamespace(
            additional_directories=(), workdir="/work"
        )),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SimpleNamespace(id="private-id", fs=SimpleNamespace(upload_file=lambda *a: None))
    client = SimpleNamespace(create=lambda *a, **k: sandbox)
    dt = SimpleNamespace(client=lambda: client, upload_path=lambda *a: {"exit": 0})
    monkeypatch.setattr(module, "_dt", lambda: dt)
    monkeypatch.setattr(module, "_ensure_snapshot", lambda *a: "snapshot")
    monkeypatch.setattr(module, "resolve_profile", lambda *a, **k: SimpleNamespace(
        receipt=lambda: {"cpu": 1, "memory_gb": 1, "disk_gb": 1}
    ))
    monkeypatch.setattr(module, "to_json", lambda *a: b'{"spec":true}')
    monkeypatch.setattr(module, "_run_portable", lambda *a: {
        "exit": 0, "stdout": b'{}', "stderr": b""
    })
    monkeypatch.setattr(module.msgspec.json, "decode", lambda *a, **k: SimpleNamespace(
        status="graded", reward=1.0, detail={}
    ), raising=False)
    cleanup_calls = []

    def cleanup(*args):
        cleanup_calls.append(args)
        return {"state": "unconfirmed", "attempts": [], "observations": []}

    monkeypatch.setattr(module, "_cleanup_private_sandbox", cleanup)
    result = module.grade_in_daytona(
        specification, {"protocol": True}, "answer", workspace, embedded=True
    )
    assert result.status == "graded" and result.reward == 1.0
    assert result.detail["verifier_cleanup"]["state"] == "unconfirmed"
    assert len(cleanup_calls) == 1
