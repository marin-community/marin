import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

from capability_pipeline.daytona_resources import CANDIDATE_DEFAULT
from capability_pipeline.daytona_telemetry import (
    lifecycle_telemetry,
    parse_cgroup_telemetry,
)


def _environment_module(monkeypatch):
    class BaseEnvironment:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class ExecResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "msgspec", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "daytona",
        SimpleNamespace(
            CreateSandboxFromSnapshotParams=object,
            CreateSnapshotParams=object,
            Image=object,
            Resources=object,
        ),
    )
    monkeypatch.setitem(sys.modules, "harbor", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "harbor.environments", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "harbor.environments.base",
        SimpleNamespace(BaseEnvironment=BaseEnvironment, ExecResult=ExecResult),
    )
    monkeypatch.setitem(
        sys.modules,
        "harbor.environments.capabilities",
        SimpleNamespace(EnvironmentCapabilities=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(sys.modules, "taskcompendium", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "taskcompendium.execution",
        SimpleNamespace(DockerEnvironment=object, HarborTaskBinding=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "taskcompendium.models",
        SimpleNamespace(image_digest=lambda image: image),
    )
    sys.modules.pop("capability_pipeline.daytona_environment", None)
    return importlib.import_module("capability_pipeline.daytona_environment")


def test_frozen_daytona_helper_load_does_not_create_bytecode(tmp_path, monkeypatch):
    module = _environment_module(monkeypatch)
    helper = tmp_path / "dt.py"
    helper.write_text("VALUE = 7\n")
    monkeypatch.setenv("CAPABILITY_DAYTONA_TOOLS", str(tmp_path))

    assert module._dt().VALUE == 7
    assert not (tmp_path / "__pycache__").exists()
    helper.write_text("VALUE = 8\n")
    assert module._dt().VALUE == 8
    assert not (tmp_path / "__pycache__").exists()


def test_stop_records_failed_final_probe_before_guaranteed_deletion(
    tmp_path, monkeypatch
):
    module = _environment_module(monkeypatch)
    startup = parse_cgroup_telemetry("", CANDIDATE_DEFAULT)
    telemetry = lifecycle_telemetry(startup)
    lifecycle, provider = tmp_path / "lifecycle.json", tmp_path / "provider.json"
    for path in (lifecycle, provider):
        path.write_text(json.dumps({"resource_telemetry": telemetry}))

    deleted = []
    environment = object.__new__(module.DaytonaHarborEnvironment)
    environment.sandbox = SimpleNamespace(delete=lambda: deleted.append(True))
    environment.resource_profile = CANDIDATE_DEFAULT
    environment._resource_telemetry = telemetry
    environment._resource_receipt_paths = (lifecycle, provider)
    monkeypatch.setattr(
        module,
        "_run_portable",
        lambda *args, **kwargs: {"exit": 1, "stdout": "", "stderr": "failed"},
    )

    asyncio.run(environment.stop(delete=True))

    assert deleted == [True]
    assert environment.sandbox is None
    for path in (lifecycle, provider):
        final = json.loads(path.read_text())["resource_telemetry"]["final"]
        assert final["phase"] == "final"
        assert final["state"] == "probe_failed"


def test_shared_public_startup_applies_task_binding_in_harbor_order(tmp_path, monkeypatch):
    module = _environment_module(monkeypatch)
    calls = []
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    async def execute(command, *, cwd, timeout_sec=None):
        calls.append(("exec", command, cwd, timeout_sec))
        return SimpleNamespace(return_code=0, stderr="", stdout="")

    async def upload(source, target):
        calls.append(("upload", source, target))

    binding = SimpleNamespace(
        workdir="/workspace", additional_directories=("/workspace/cache",),
        setup_commands=("touch /workspace/ready",),
    )
    asyncio.run(module.materialize_docker_binding(binding, inputs, execute, upload))
    assert calls == [
        ("exec", "mkdir -p /workspace /logs/agent /logs/verifier /logs/artifacts /tests", "/", None),
        ("upload", inputs, "/workspace"),
        ("exec", "mkdir -p /workspace/cache", "/", None),
        ("exec", "touch /workspace/ready", "/", 1800),
    ]
