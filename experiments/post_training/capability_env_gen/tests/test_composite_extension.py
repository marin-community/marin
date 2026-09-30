import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline.composite_extension import (
    BASE_REVISION,
    COMPOSITE_SPECIFICATION,
    IMPORT_PATH,
    UNSUPPORTED_SPECIFICATION,
    install_extension_marker,
    resolve_runner_specification,
    runtime_attestation_bindings,
    seal_composite_lowering_package,
    sha256,
    validate_composite_lowering_authorization,
    validate_extension_marker,
)


def test_runtime_attestation_producer_consumer_contract_includes_policy():
    bindings = runtime_attestation_bindings("c" * 64)
    assert set(bindings) == {
        "composite_verifier_adapter_sha256",
        "composite_verifier_policy_sha256",
        "composite_native_judge_protocol_sha256",
        "composite_verifier_config_sha256",
        "composite_taskcompendium_guard_sha256",
        "composite_taskcompendium_runner_sha256",
        "composite_taskcompendium_lowering_sha256",
    }
    assert bindings["composite_verifier_policy_sha256"] == sha256(
        Path(__file__).parents[1] / "capability_pipeline/composite_policy.py"
    )
    assert bindings["composite_native_judge_protocol_sha256"] == sha256(
        Path(__file__).parents[1] / "capability_pipeline/native_judge_protocol.py"
    )


def test_composite_lowering_authorization_requires_exact_pinned_inputs(tmp_path, monkeypatch):
    spec_path = tmp_path / "specification.json"
    spec_path.write_text('{"id":"test"}')
    config_path = tmp_path / "composite-verifier.json"
    lowering_path = tmp_path / "lowering.py"
    lowering_path.write_text("# pinned lowering fixture\n")
    specification = SimpleNamespace(steps=(object(),))
    lowering = types.ModuleType("taskcompendium.lowering")
    lowering.__file__ = str(lowering_path)
    serialization = types.ModuleType("taskcompendium.serialization")
    serialization.from_json = lambda _raw: specification
    taskcompendium = types.ModuleType("taskcompendium")
    taskcompendium.__path__ = []
    taskcompendium.lowering = lowering
    taskcompendium.serialization = serialization
    monkeypatch.setitem(sys.modules, "taskcompendium", taskcompendium)
    monkeypatch.setitem(sys.modules, "taskcompendium.lowering", lowering)
    monkeypatch.setitem(sys.modules, "taskcompendium.serialization", serialization)
    monkeypatch.setattr(
        "capability_pipeline.composite_extension.PATCHED_LOWERING_SHA256",
        sha256(lowering_path),
    )
    module_root = Path(__file__).parents[1] / "capability_pipeline"
    config = {
        "schema_version": "taskcompendium-composite-verifier-v1",
        "implementation": {
            "taskcompendium_revision": BASE_REVISION,
            "adapter_sha256": sha256(module_root / "composite_verifier.py"),
            "policy_sha256": sha256(module_root / "composite_policy.py"),
            "native_judge_protocol_sha256": sha256(module_root / "native_judge_protocol.py"),
        },
        "specification_sha256": sha256(spec_path),
        "steps": [{
            "step_index": 0,
            "machine_checks": [{
                "id": "gate", "role": "gate", "script_path": "gate.py", "args": [],
                "image": "python@sha256:" + "a" * 64, "timeout": 60,
            }],
            "judge": {
                "criterion_weights": [1.0], "critical_indices": [0],
                "critical_min": 1.0, "conditional_caps": [],
            },
        }],
    }
    config_path.write_text(json.dumps(config))
    assert validate_composite_lowering_authorization(
        specification, spec_path, config_path,
    )
    with pytest.raises(ValueError, match="specification and config"):
        validate_composite_lowering_authorization(specification, spec_path, None)
    config["implementation"]["policy_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not bound"):
        validate_composite_lowering_authorization(
            specification, spec_path, config_path,
        )


def test_composite_lowering_seals_native_package_before_return(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "specification.json").write_text('{"id":"same"}')
    (package / "manifest.json").write_text(json.dumps({
        "step_names": ["step-1"],
        "specification_sha256": sha256(package / "specification.json"),
    }))
    source_specification = tmp_path / "source-specification.json"
    source_specification.write_text('{ "id" : "same" }')
    config = tmp_path / "composite-verifier.json"
    config.write_text('{"sealed":true}')
    serialization = types.ModuleType("taskcompendium.serialization")
    serialization.from_json = json.loads
    taskcompendium = types.ModuleType("taskcompendium")
    taskcompendium.__path__ = []
    taskcompendium.serialization = serialization
    monkeypatch.setitem(sys.modules, "taskcompendium", taskcompendium)
    monkeypatch.setitem(sys.modules, "taskcompendium.serialization", serialization)
    monkeypatch.setattr(
        "capability_pipeline.composite_extension.validate_composite_lowering_authorization",
        lambda *_args: True,
    )
    seal_composite_lowering_package(package, source_specification, config)
    assert (package / "specification.json").read_bytes() == UNSUPPORTED_SPECIFICATION
    assert (package / COMPOSITE_SPECIFICATION).read_bytes() == source_specification.read_bytes()
    assert (package / "composite-verifier.json").read_bytes() == config.read_bytes()
    record = json.loads((package / "manifest.json").read_text())["required_extensions"][0]
    assert record["required"] is True
    assert json.loads((package / "manifest.json").read_text())["specification_sha256"] == sha256(
        package / COMPOSITE_SPECIFICATION
    )
    manifest_path = package / "manifest.json"
    bound_manifest = json.loads(manifest_path.read_text())
    tampered_manifest = {**bound_manifest, "specification_sha256": "0" * 64}
    manifest_path.write_text(json.dumps(tampered_manifest))
    with pytest.raises(ValueError, match="manifest specification hash mismatch"):
        validate_extension_marker(
            package,
            adapter_sha256=record["adapter_sha256"],
            policy_sha256=record["policy_sha256"],
            config_sha256=record["config_sha256"],
            supported=True,
        )
    manifest_path.write_text(json.dumps(bound_manifest))
    with pytest.raises(RuntimeError, match="Unsupported mandatory"):
        validate_extension_marker(
            package,
            adapter_sha256=record["adapter_sha256"],
            policy_sha256=record["policy_sha256"],
            config_sha256=record["config_sha256"],
            supported=False,
        )
    unbound = tmp_path / "unbound-package"
    unbound.mkdir()
    (unbound / "specification.json").write_text('{"id":"same"}')
    (unbound / "manifest.json").write_text('{"step_names":["step-1"]}')
    with pytest.raises(ValueError, match="does not bind"):
        seal_composite_lowering_package(unbound, source_specification, config)
    assert not unbound.exists()


def test_export_requires_composite_consumer_and_preserves_bound_specification(
    tmp_path: Path,
):
    package = tmp_path / "package"
    package.mkdir()
    original = b'{"schema_version":"0.9","steps":[]}\n'
    (package / "specification.json").write_bytes(original)
    (package / "manifest.json").write_text('{"step_names":[]}')
    hashes = {"adapter_sha256": "a" * 64, "policy_sha256": "b" * 64}
    install_extension_marker(package, **hashes, config_sha256="c" * 64)

    assert (package / COMPOSITE_SPECIFICATION).read_bytes() == original
    assert (package / "specification.json").read_bytes() == UNSUPPORTED_SPECIFICATION
    with pytest.raises(RuntimeError, match="Unsupported mandatory"):
        validate_extension_marker(
            package, **hashes, config_sha256="c" * 64, supported=False
        )
    record = validate_extension_marker(
        package, **hashes, config_sha256="c" * 64, supported=True
    )
    assert json.loads((package / "manifest.json").read_text())[
        "required_extensions"
    ] == [record]


def test_guard_fails_when_native_specification_is_restored(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "specification.json").write_text("original")
    (package / "manifest.json").write_text('{"step_names":[]}')
    kwargs = {
        "adapter_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "config_sha256": "c" * 64,
    }
    install_extension_marker(package, **kwargs)
    (package / "specification.json").write_text("original")
    with pytest.raises(ValueError, match="fail closed"):
        validate_extension_marker(package, **kwargs, supported=True)


def test_helper_only_change_invalidates_extension_marker(tmp_path: Path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "specification.json").write_text("original")
    (package / "manifest.json").write_text('{"step_names":[]}')
    kwargs = {
        "adapter_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "config_sha256": "c" * 64,
    }
    install_extension_marker(package, **kwargs)
    monkeypatch.setattr(
        "capability_pipeline.composite_extension.native_judge_protocol_sha256",
        lambda: "d" * 64,
    )
    with pytest.raises(ValueError, match="lacks the exact mandatory"):
        validate_extension_marker(package, **kwargs, supported=True)


def test_runner_resolves_preserved_spec_only_for_exact_declared_adapter(
    tmp_path: Path, monkeypatch
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "specification.json").write_text('{"schema_version":"0.9"}')
    (package / "manifest.json").write_text('{"step_names":[]}')
    config = package / "composite-verifier.json"
    config.write_text("{}")
    module_root = Path(__file__).parents[1] / "capability_pipeline"
    hashes = {
        "adapter_sha256": sha256(module_root / "composite_verifier.py"),
        "policy_sha256": sha256(module_root / "composite_policy.py"),
        "config_sha256": sha256(config),
    }

    native_runner_path = tmp_path / "runner.py"
    native_runner_path.write_text("# trusted patched runner fixture\n")
    taskcompendium = types.ModuleType("taskcompendium")
    taskcompendium.__path__ = []
    harbor = types.ModuleType("taskcompendium.harbor")
    harbor.__path__ = []
    runner = types.ModuleType("taskcompendium.harbor.runner")
    runner.__file__ = str(native_runner_path)
    taskcompendium.harbor = harbor
    harbor.runner = runner
    monkeypatch.setitem(sys.modules, "taskcompendium", taskcompendium)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor", harbor)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.runner", runner)
    monkeypatch.setattr(
        "capability_pipeline.composite_extension.PATCHED_RUNNER_SHA256",
        sha256(native_runner_path),
    )
    install_extension_marker(package, **hashes)

    with pytest.raises(RuntimeError, match="declared verifier adapter"):
        resolve_runner_specification(
            package, {"verifier": {"import_path": "ordinary:Verifier"}}
        )
    resolved = resolve_runner_specification(
        package, {"verifier": {"import_path": IMPORT_PATH}}
    )
    assert resolved == package / COMPOSITE_SPECIFICATION
