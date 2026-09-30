"""Mandatory package marker and pinned TaskCompendium guard for composite grading."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

EXTENSION_ID = "taskcompendium-composite-verifier-v1"
IMPORT_PATH = "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
BASE_REVISION = "dc6b501c8604bcd2e3c20c1e9947679845fdfef8"
VERIFIER_RELATIVE = Path("src/taskcompendium/harbor/verifier.py")
RUNNER_RELATIVE = Path("src/taskcompendium/harbor/runner.py")
LOWERING_RELATIVE = Path("src/taskcompendium/lowering.py")
BASE_LOWERING_SHA256 = "86141beaf8a1c5113f504a3cdf71921f004b5040e4cf6d8d9bfb49e5973af6ca"
PATCHED_LOWERING_SHA256 = "9a5212468b1376174166556be2a4d838e0e35329a89bb4a9da196190af0b553f"
BASE_VERIFIER_SHA256 = (
    "208ef5d5e99adc91aec5fce624c280e96401c59a8c4c5a02c6d26735de0fd657"
)
PATCHED_VERIFIER_SHA256 = (
    "cd770d4a5ec2ccfc882de6fce6433095faf6515b8cbe9ad36b51afd7e4470128"
)
BASE_RUNNER_SHA256 = "f7e14c819644469b86929c2770e4e3f02237fda6882f88f48cc0342ddc99ca9d"
PATCHED_RUNNER_SHA256 = (
    "c9540cfde9fae294d9de98229ca97a564ef6b36a29f16906b271b07bc7c4192a"
)
COMPOSITE_SPECIFICATION = "composite-specification.json"
UNSUPPORTED_SPECIFICATION = (
    b'{"required_extension":"taskcompendium-composite-verifier-v1",'
    b'"schema_version":"unsupported-without-mandatory-extension"}\n'
)

_BASE = """        names = json.loads((root / "manifest.json").read_text())["step_names"]
"""
_PATCHED = """        manifest = json.loads((root / "manifest.json").read_text())
        required_extensions = manifest.get("required_extensions", [])
        if required_extensions:
            identifiers = [
                item.get("id") for item in required_extensions if isinstance(item, dict)
            ]
            raise RuntimeError(
                "Unsupported mandatory task extension(s): " + ", ".join(identifiers)
            )
        names = manifest["step_names"]
"""

_RUNNER_BASE = """    specification = from_json((task_dir / "specification.json").read_bytes())
"""
_RUNNER_PATCHED = """    specification_path = task_dir / "specification.json"
    if manifest.get("required_extensions"):
        from capability_pipeline.composite_extension import (
            resolve_runner_specification,
        )

        specification_path = resolve_runner_specification(task_dir, execution)
    specification = from_json(specification_path.read_bytes())
"""

_LOWERING_PATCHES = (
    (
        "    specification: TaskSpec, protocol: Rendering, binding: HarborTaskBinding, step_index: int = 0\n",
        ("    specification: TaskSpec, protocol: Rendering, binding: HarborTaskBinding, step_index: int = 0, *,\n"
        "    allow_composite_judge_final_state: bool = False,\n"),
    ),
    (
        ("    elif isinstance(submission, FinalState):\n"
        "        raise ValueError(\"Final-state submission requires executable workspace verification\")\n"),
        ("    elif isinstance(submission, FinalState) and not (\n"
        "        allow_composite_judge_final_state and source is not None and source.mode == Mode.JUDGE\n"
        "    ):\n"
        "        raise ValueError(\"Final-state submission requires executable workspace verification\")\n"),
    ),
    (
        ("    model_name: str | None = None,\n) -> Path:\n"
        "    \"\"\"Write one task-owned Harbor package without selecting a harness.\"\"\"\n"),
        ("    model_name: str | None = None,\n"
        "    composite_specification_path: Path | None = None,\n"
        "    composite_config_path: Path | None = None,\n) -> Path:\n"
        "    \"\"\"Write one task-owned Harbor package without selecting a harness.\"\"\"\n"),
    ),
    (
        ("    for index, rendering in enumerate(renderings):\n"
        "        validate_lowering(specification, rendering, binding, index)\n"),
        ("    allow_composite_judge_final_state = False\n"
        "    if composite_specification_path is not None or composite_config_path is not None:\n"
        "        from capability_pipeline.composite_extension import validate_composite_lowering_authorization\n"
        "        allow_composite_judge_final_state = validate_composite_lowering_authorization(\n"
        "            specification, composite_specification_path, composite_config_path\n"
        "        )\n"
        "    for index, rendering in enumerate(renderings):\n"
        "        validate_lowering(\n"
        "            specification, rendering, binding, index,\n"
        "            allow_composite_judge_final_state=allow_composite_judge_final_state,\n"
        "        )\n"),
    ),
    (
        ('    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")\n'
        '    return destination\n'),
        ('    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")\n'
        '    if allow_composite_judge_final_state:\n'
        '        from capability_pipeline.composite_extension import seal_composite_lowering_package\n'
        '        seal_composite_lowering_package(destination, composite_specification_path, composite_config_path)\n'
        '    return destination\n'),
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_judge_protocol_sha256() -> str:
    return sha256(Path(__file__).with_name("native_judge_protocol.py"))


def validate_composite_lowering_authorization(
    specification: Any, specification_path: Path | None, config_path: Path | None,
) -> bool:
    """Authorize composite-only FinalState lowering from exact source bytes and pins."""
    if specification_path is None or config_path is None:
        raise ValueError("composite lowering requires specification and config paths")
    import taskcompendium.lowering as native_lowering
    from taskcompendium.serialization import from_json

    from .composite_policy import validate_composite_config

    if sha256(Path(native_lowering.__file__)) != PATCHED_LOWERING_SHA256:
        raise ValueError("composite lowering requires the exact pinned extension")
    if from_json(specification_path.read_bytes()) != specification:
        raise ValueError("composite lowering specification changed after validation")
    config = json.loads(config_path.read_text())
    validate_composite_config(
        config,
        specification_sha256=sha256(specification_path),
        adapter_sha256=sha256(Path(__file__).with_name("composite_verifier.py")),
        policy_sha256=sha256(Path(__file__).with_name("composite_policy.py")),
        step_count=len(specification.steps),
    )
    return True


def seal_composite_lowering_package(
    package: Path, specification_path: Path, config_path: Path,
) -> None:
    """Replace native export with the mandatory composite marker before return."""
    from taskcompendium.serialization import from_json

    try:
        native_specification = package / "specification.json"
        original = specification_path.read_bytes()
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("specification_sha256") != sha256(native_specification):
            raise ValueError("lowered composite manifest does not bind its specification")
        if from_json(native_specification.read_bytes()) != from_json(original):
            raise ValueError("lowered composite specification differs from source")
        validate_composite_lowering_authorization(
            from_json(original), specification_path, config_path,
        )
        native_specification.write_bytes(original)
        config_target = package / "composite-verifier.json"
        config_target.write_bytes(config_path.read_bytes())
        module_root = Path(__file__).parent
        pins = {
            "adapter_sha256": sha256(module_root / "composite_verifier.py"),
            "policy_sha256": sha256(module_root / "composite_policy.py"),
            "config_sha256": sha256(config_target),
        }
        install_extension_marker(package, **pins)
        validate_extension_marker(package, **pins, supported=True)
    except Exception:
        shutil.rmtree(package, ignore_errors=True)
        raise


def runtime_attestation_bindings(config_sha256: str) -> dict[str, str]:
    """One producer/consumer contract for composite runtime evidence."""
    return {
        "composite_verifier_adapter_sha256": sha256(
            Path(__file__).with_name("composite_verifier.py")
        ),
        "composite_verifier_policy_sha256": sha256(
            Path(__file__).with_name("composite_policy.py")
        ),
        "composite_native_judge_protocol_sha256": native_judge_protocol_sha256(),
        "composite_verifier_config_sha256": config_sha256,
        "composite_taskcompendium_guard_sha256": PATCHED_VERIFIER_SHA256,
        "composite_taskcompendium_runner_sha256": PATCHED_RUNNER_SHA256,
        "composite_taskcompendium_lowering_sha256": PATCHED_LOWERING_SHA256,
    }


def apply_taskcompendium_guard(source: Path, *, expected_base: str) -> str:
    """Apply the exact fail-closed guard to a private copy of the pinned source."""
    verifier = source / VERIFIER_RELATIVE
    if sha256(verifier) != expected_base:
        raise ValueError("TaskCompendium verifier does not match the pinned patch base")
    text = verifier.read_text()
    if text.count(_BASE) != 1:
        raise ValueError("TaskCompendium verifier guard patch has no unique anchor")
    verifier.write_text(text.replace(_BASE, _PATCHED))
    runner = source / RUNNER_RELATIVE
    if sha256(runner) != BASE_RUNNER_SHA256:
        raise ValueError("TaskCompendium runner does not match the pinned patch base")
    runner_text = runner.read_text()
    if runner_text.count(_RUNNER_BASE) != 1:
        raise ValueError("TaskCompendium runner extension patch has no unique anchor")
    runner.write_text(runner_text.replace(_RUNNER_BASE, _RUNNER_PATCHED))
    lowering = source / LOWERING_RELATIVE
    if sha256(lowering) != BASE_LOWERING_SHA256:
        raise ValueError("TaskCompendium lowering does not match the pinned patch base")
    lowering_text = lowering.read_text()
    for original, patched in _LOWERING_PATCHES:
        if lowering_text.count(original) != 1:
            raise ValueError("TaskCompendium composite lowering patch has no unique anchor")
        lowering_text = lowering_text.replace(original, patched)
    lowering.write_text(lowering_text)
    if sha256(lowering) != PATCHED_LOWERING_SHA256:
        raise ValueError("TaskCompendium composite lowering patch hash mismatch")
    return sha256(verifier)


def extension_record(
    *,
    adapter_sha256: str,
    policy_sha256: str,
    config_sha256: str,
    specification_sha256: str,
) -> dict[str, Any]:
    return {
        "id": EXTENSION_ID,
        "required": True,
        "import_path": IMPORT_PATH,
        "taskcompendium_base_revision": BASE_REVISION,
        "adapter_sha256": adapter_sha256,
        "policy_sha256": policy_sha256,
        "native_judge_protocol_sha256": native_judge_protocol_sha256(),
        "config_sha256": config_sha256,
        "specification_path": COMPOSITE_SPECIFICATION,
        "specification_sha256": specification_sha256,
        "unsupported_specification_sha256": hashlib.sha256(
            UNSUPPORTED_SPECIFICATION
        ).hexdigest(),
        "taskcompendium_verifier_guard_sha256": PATCHED_VERIFIER_SHA256,
        "taskcompendium_runner_extension_sha256": PATCHED_RUNNER_SHA256,
        "taskcompendium_lowering_extension_sha256": PATCHED_LOWERING_SHA256,
    }


def install_extension_marker(
    package: Path,
    *,
    adapter_sha256: str,
    policy_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("Harbor manifest must be an object")
    source_specification = package / "specification.json"
    composite_specification = package / COMPOSITE_SPECIFICATION
    if not source_specification.is_file() or composite_specification.exists():
        raise ValueError("Harbor package specification cannot be guarded exactly once")
    specification_sha256 = sha256(source_specification)
    source_specification.replace(composite_specification)
    source_specification.write_bytes(UNSUPPORTED_SPECIFICATION)
    record = extension_record(
        adapter_sha256=adapter_sha256,
        policy_sha256=policy_sha256,
        config_sha256=config_sha256,
        specification_sha256=specification_sha256,
    )
    existing = manifest.get("required_extensions")
    if existing not in (None, []):
        raise ValueError("Harbor package already declares mandatory extensions")
    manifest["specification_sha256"] = specification_sha256
    manifest["required_extensions"] = [record]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return record


def validate_extension_marker(
    package: Path,
    *,
    adapter_sha256: str,
    policy_sha256: str,
    config_sha256: str,
    supported: bool,
) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text())
    specification = package / COMPOSITE_SPECIFICATION
    raw_sha256 = sha256(specification)
    if manifest.get("specification_sha256") != raw_sha256:
        raise ValueError("Harbor composite manifest specification hash mismatch")
    expected = extension_record(
        adapter_sha256=adapter_sha256,
        policy_sha256=policy_sha256,
        config_sha256=config_sha256,
        specification_sha256=raw_sha256,
    )
    if not isinstance(manifest, dict) or manifest.get("required_extensions") != [
        expected
    ]:
        raise ValueError("Harbor package lacks the exact mandatory composite extension")
    if (package / "specification.json").read_bytes() != UNSUPPORTED_SPECIFICATION:
        raise ValueError("Harbor package does not fail closed without the extension")
    if not supported:
        raise RuntimeError(f"Unsupported mandatory task extension: {EXTENSION_ID}")
    return expected


def resolve_runner_specification(package: Path, execution: dict[str, Any]) -> Path:
    """Resolve the preserved spec only for the exact maintained runner/adapter."""
    import taskcompendium.harbor.runner as native_runner

    package = Path(package)
    if sha256(Path(native_runner.__file__)) != PATCHED_RUNNER_SHA256:
        raise RuntimeError(
            "composite package requires the pinned TaskCompendium runner"
        )
    verifier = execution.get("verifier") if isinstance(execution, dict) else None
    if not isinstance(verifier, dict) or verifier.get("import_path") != IMPORT_PATH:
        raise RuntimeError("composite package requires its declared verifier adapter")
    adapter = Path(__file__).with_name("composite_verifier.py")
    policy = Path(__file__).with_name("composite_policy.py")
    config = package / "composite-verifier.json"
    if not adapter.is_file() or not policy.is_file() or not config.is_file():
        raise RuntimeError("composite runner dependencies are absent")
    validate_extension_marker(
        package,
        adapter_sha256=sha256(adapter),
        policy_sha256=sha256(policy),
        config_sha256=sha256(config),
        supported=True,
    )
    return package / COMPOSITE_SPECIFICATION
