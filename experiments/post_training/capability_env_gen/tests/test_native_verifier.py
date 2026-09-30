import asyncio
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(monkeypatch):
    class Result:
        def __init__(self, status, reward, detail):
            self.status, self.reward, self.detail = status, reward, detail

    upstream = types.ModuleType("taskcompendium.harbor.verifier")
    upstream.__file__ = __file__
    upstream.SemanticVerifier = object
    upstream.ExtractionError = RuntimeError
    upstream.GradingInfrastructureError = RuntimeError
    taskcompendium = types.ModuleType("taskcompendium")
    harbor = types.ModuleType("taskcompendium.harbor")
    harbor.verifier = upstream
    taskcompendium.harbor = harbor
    models = types.ModuleType("taskcompendium.models")
    models.ContainerRuntime = type("ContainerRuntime", (), {})

    class FileSubmission:
        def __init__(self, path):
            self.path = path

    models.FileSubmission = FileSubmission
    models.FinalState = type("FinalState", (), {})

    class ResourceRef:
        pass

    class Embedded:
        def __init__(self, data):
            self.data = data

    models.ResourceRef = ResourceRef
    models.Embedded = Embedded
    models.ResourceRole = SimpleNamespace(VERIFIER="verifier")
    models.TaskTroveVerifier = type("TaskTroveVerifier", (), {})
    models.GradingResult = Result
    models.Outcome = SimpleNamespace(
        EXTRACTION_ERROR="extraction_error", GRADED="graded"
    )
    models.tasktrove_verifier = lambda value: value
    models.verifier_runtime = lambda value: None
    grading = types.ModuleType("taskcompendium.grading")
    grading.grade_attempt = lambda *args: None
    resources = types.ModuleType("taskcompendium.resources")
    resources.resource_bytes = lambda resource: resource.content.value
    paths = types.ModuleType("taskcompendium.grading_paths")
    paths.EXTERNAL_DIRECTORY = "__external__"
    paths.submission_relative = lambda path, *_: path
    serialization = types.ModuleType("taskcompendium.serialization")
    serialization.from_json = lambda value: value
    serialization.renderings_from_json = lambda value: value
    serialization.to_json = lambda value: b"{}"
    tasktrove = types.ModuleType("tasktrove_verify")
    tasktrove_spec = types.ModuleType("tasktrove_verify.spec")
    tasktrove_spec.Mode = SimpleNamespace(
        MCQ="mcq",
        MATH="math",
        NUMERIC="numeric",
        EXACT="exact",
        JSON_SCHEMA="json-schema",
        XML_ELEMENTS="xml-elements",
        CSV_COLUMNS="csv-columns",
        IFEVAL="ifeval",
        JUDGE="judge",
    )
    msgspec = types.ModuleType("msgspec")
    msgspec.json = SimpleNamespace(encode=lambda value: b"{}")
    msgspec.structs = SimpleNamespace(
        replace=lambda value, **changes: SimpleNamespace(**(vars(value) | changes))
    )
    harbor_root = types.ModuleType("harbor")
    harbor_models = types.ModuleType("harbor.models")
    harbor_verifier = types.ModuleType("harbor.models.verifier")
    harbor_result = types.ModuleType("harbor.models.verifier.result")
    harbor_result.VerifierResult = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "harbor", harbor_root)
    monkeypatch.setitem(sys.modules, "harbor.models", harbor_models)
    monkeypatch.setitem(sys.modules, "harbor.models.verifier", harbor_verifier)
    monkeypatch.setitem(sys.modules, "harbor.models.verifier.result", harbor_result)
    monkeypatch.setitem(sys.modules, "msgspec", msgspec)
    monkeypatch.setitem(sys.modules, "tasktrove_verify", tasktrove)
    monkeypatch.setitem(sys.modules, "tasktrove_verify.spec", tasktrove_spec)
    monkeypatch.setitem(sys.modules, "taskcompendium", taskcompendium)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor", harbor)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.verifier", upstream)
    monkeypatch.setitem(sys.modules, "taskcompendium.grading", grading)
    monkeypatch.setitem(sys.modules, "taskcompendium.grading_paths", paths)
    monkeypatch.setitem(sys.modules, "taskcompendium.models", models)
    monkeypatch.setitem(sys.modules, "taskcompendium.resources", resources)
    monkeypatch.setitem(sys.modules, "taskcompendium.serialization", serialization)
    sys.modules.pop("capability_pipeline.native_verifier", None)
    return (
        importlib.import_module("capability_pipeline.native_verifier"),
        Result,
        upstream,
    )


def test_receipt_keeps_source_result_and_binds_fingerprint(monkeypatch, tmp_path):
    module, Result, _ = _module(monkeypatch)
    result = Result("extraction_error", None, {"error": "empty"})
    fingerprint = {"grading_input_sha256": "a" * 64}
    receipt = module._receipt(result, fingerprint, "b" * 64)
    assert receipt.status == "extraction_error"
    assert receipt.reward is None
    assert receipt.detail["error"] == "empty"
    assert receipt.detail["grading_input_fingerprint"] == fingerprint
    native = receipt.detail["native_verifier_receipt"]
    assert native["schema_version"] == "capability-native-verifier-receipt-v1"
    assert native["semantic_verifier_sha256"] == "b" * 64
    assert (
        native["adapter_sha256"]
        == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    )


def test_source_guard_rejects_unknown_and_required_extension(monkeypatch, tmp_path):
    module, _, upstream = _module(monkeypatch)
    root = tmp_path / "task"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"required_extensions": []}))
    upstream.__file__ = __file__
    with pytest.raises(RuntimeError, match="pinned"):
        module._verified_native_source(root)

    source = tmp_path / "verifier.py"
    source.write_text("pinned")
    upstream.__file__ = str(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "BASE_VERIFIER_SHA256", source_hash)
    (root / "manifest.json").write_text(
        json.dumps({"required_extensions": [{"id": "composite"}]})
    )
    with pytest.raises(RuntimeError, match="composite or mandatory"):
        module._verified_native_source(root)
    (root / "manifest.json").write_text(json.dumps({"required_extensions": []}))
    assert module._verified_native_source(root) == source_hash


def _instance(module, tmp_path, mode="exact", judge=None):
    root, agent = tmp_path / "task", tmp_path / "agent"
    root.mkdir()
    agent.mkdir()
    (root / "specification.json").write_text("raw-spec")
    (root / "renderings.json").write_text("raw-renderings")
    (root / "manifest.json").write_text(json.dumps({"step_names": ["step-1"]}))
    (agent / "response.txt").write_text("answer")
    (agent / "transcript.json").write_text(json.dumps([{"role": "assistant"}]))
    verifier = module.TaskTroveVerifier()
    verifier.mode, verifier.judge, verifier.runtime = mode, judge, None
    specification = SimpleNamespace(
        steps=(SimpleNamespace(verifier=verifier, resources=()),),
        resources=(),
        requirements=SimpleNamespace(state=SimpleNamespace(additional_directories=())),
    )
    protocol = SimpleNamespace(submission=module.FileSubmission("answer.txt"))
    instance = object.__new__(module.NativeDiagnosticSemanticVerifier)
    instance.task = SimpleNamespace(
        paths=SimpleNamespace(task_dir=root),
        config=SimpleNamespace(environment=SimpleNamespace(workdir="/app")),
    )
    instance.trial_paths = SimpleNamespace(agent_dir=agent)
    instance.step_name, instance.judge_client = None, None
    return instance, specification, protocol


def test_verify_captures_before_grade_and_writes_graded_receipt(monkeypatch, tmp_path):
    module, Result, _ = _module(monkeypatch)
    instance, specification, protocol = _instance(module, tmp_path)
    events, written = [], []
    monkeypatch.setattr(module, "_verified_native_source", lambda root: "b" * 64)
    monkeypatch.setattr(module, "from_json", lambda value: specification)
    monkeypatch.setattr(module, "renderings_from_json", lambda value: (protocol,))
    monkeypatch.setattr(module, "to_json", lambda value: b"embedded-spec")

    async def download(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("delivered")

    def fingerprint(spec, selected, response, workspace, transcript, **kwargs):
        events.append("fingerprint")
        assert spec == b"embedded-spec" and selected == b"{}"
        assert response == "answer" and transcript == ({"role": "assistant"},)
        assert (workspace / "answer.txt").read_text() == "delivered"
        return {"grading_input_sha256": "f" * 64}

    def grade(*args):
        assert events == ["fingerprint"]
        return Result("graded", 1.0, {})

    instance._download_evidence, instance._write_result = download, written.append
    monkeypatch.setattr(module, "grading_input_fingerprint", fingerprint)
    monkeypatch.setattr(module, "grade_attempt", grade)
    result = asyncio.run(instance._verify())
    assert result["rewards"] == {"reward": 1.0}
    assert (
        written[0].detail["grading_input_fingerprint"]["grading_input_sha256"]
        == "f" * 64
    )


def test_verify_writes_extraction_receipt_before_exception(monkeypatch, tmp_path):
    module, Result, _ = _module(monkeypatch)
    instance, specification, protocol = _instance(module, tmp_path)
    written = []
    monkeypatch.setattr(module, "_verified_native_source", lambda root: "b" * 64)
    monkeypatch.setattr(module, "from_json", lambda value: specification)
    monkeypatch.setattr(module, "renderings_from_json", lambda value: (protocol,))
    monkeypatch.setattr(module, "to_json", lambda value: b"embedded-spec")
    instance._download_evidence = lambda *args: asyncio.sleep(0)
    instance._write_result = written.append
    monkeypatch.setattr(
        module, "grading_input_fingerprint", lambda *args, **kwargs: {"input": "bound"}
    )
    monkeypatch.setattr(
        module, "grade_attempt", lambda *args: Result("extraction_error", None, {})
    )
    with pytest.raises(RuntimeError):
        asyncio.run(instance._verify())
    assert written[0].status == "extraction_error"
    assert written[0].detail["grading_input_fingerprint"] == {"input": "bound"}


def test_verify_rejects_judge_before_native_grade(monkeypatch, tmp_path):
    module, _, _ = _module(monkeypatch)
    instance, specification, protocol = _instance(
        module, tmp_path, mode="judge", judge=object()
    )
    monkeypatch.setattr(module, "_verified_native_source", lambda root: "b" * 64)
    monkeypatch.setattr(module, "from_json", lambda value: specification)
    monkeypatch.setattr(module, "renderings_from_json", lambda value: (protocol,))
    monkeypatch.setattr(
        module, "grade_attempt", lambda *args: pytest.fail("must not grade")
    )
    with pytest.raises(RuntimeError, match="allowed non-judge"):
        asyncio.run(instance._verify())


def test_embedding_verifier_resource_changes_captured_spec_fingerprint(
    monkeypatch, tmp_path
):
    module, _, _ = _module(monkeypatch)
    reference = module.ResourceRef()
    reference.value = b"first"
    resource = SimpleNamespace(roles=("verifier",), content=reference)
    specification = SimpleNamespace(
        resources=(resource,), steps=(SimpleNamespace(resources=()),)
    )
    monkeypatch.setattr(
        module, "to_json", lambda value: value.resources[0].content.data
    )
    first = module.grading_input_fingerprint(
        module.to_json(module._embedded_specification(specification, 0)),
        b"protocol",
        None,
        tmp_path,
        (),
    )
    reference.value = b"second"
    second = module.grading_input_fingerprint(
        module.to_json(module._embedded_specification(specification, 0)),
        b"protocol",
        None,
        tmp_path,
        (),
    )
    assert first["specification_sha256"] != second["specification_sha256"]
