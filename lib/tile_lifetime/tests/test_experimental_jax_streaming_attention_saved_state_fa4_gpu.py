# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from pathlib import Path
from types import ModuleType

import pytest

from lib.tile_lifetime.benchmarks.experimental_jax_streaming_attention_saved_state_fa4_gpu import (
    FLASH_ATTN_DISTRIBUTION,
    FLASH_ATTN_IMPORT_PACKAGE,
    FLASH_ATTN_SM90_IMPLEMENTATION_MODULES,
    GENERATED_FORWARD_TARGET,
    GENERATED_REVERSE_TARGET,
    _installed_flash_attn_audit,
    _require_fresh_directory,
    _source_audit,
    _verify_compiled_hlo_audit,
)

REPOSITORY = Path(__file__).resolve().parents[3]


class _FakeDistribution:
    def __init__(self, root: Path, *, name: str = FLASH_ATTN_DISTRIBUTION):
        self._root = root
        self.metadata = {"Name": name}
        self.version = "4.0.0b16"

    def locate_file(self, path: str) -> Path:
        return self._root / path


def _fake_loaded_fa4_modules(import_root: Path) -> dict[str, ModuleType]:
    modules = {}
    for index, (_, module_name, symbol) in enumerate(FLASH_ATTN_SM90_IMPLEMENTATION_MODULES):
        path = import_root.joinpath(*module_name.split(".")[1:]).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fake loaded implementation {index}\n")
        module = ModuleType(module_name)
        module.__file__ = str(path)
        setattr(module, symbol, object())
        modules[module_name] = module
    return modules


def test_fa4_source_audit_covers_the_physical_kernel_sources() -> None:
    audit = _source_audit(REPOSITORY)
    audited_paths = {source["path"] for source in audit["sources"]}

    assert "lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py" in audited_paths
    assert "lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py" in audited_paths
    assert all(len(source["sha256"]) == 64 and source["bytes"] > 0 for source in audit["sources"])


def test_benchmark_requires_fresh_artifact_and_build_directories(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    _require_fresh_directory(fresh, label="artifact directory")
    assert fresh.is_dir()

    (fresh / "stale.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh empty directory"):
        _require_fresh_directory(fresh, label="artifact directory")


def test_compiled_boundary_rejects_missing_generated_target() -> None:
    audit = {
        "entry_layout": "HloModule generated",
        "contains_custom_call": True,
        "custom_call_targets": (GENERATED_FORWARD_TARGET,),
    }

    _verify_compiled_hlo_audit(
        audit,
        boundary_name="generated forward",
        expected_target=GENERATED_FORWARD_TARGET,
    )
    with pytest.raises(ValueError, match="does not contain generated target"):
        _verify_compiled_hlo_audit(
            audit,
            boundary_name="generated reverse",
            expected_target=GENERATED_REVERSE_TARGET,
        )


def test_fa4_runtime_audit_records_distribution_import_mapping_and_loaded_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_root = tmp_path / FLASH_ATTN_IMPORT_PACKAGE
    modules = _fake_loaded_fa4_modules(import_root)

    def distribution(name: str) -> _FakeDistribution:
        assert name == "flash-attn-4"
        return _FakeDistribution(tmp_path)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    audit = _installed_flash_attn_audit(loaded_modules=modules)

    assert audit["distribution"] == {"requested": "flash-attn-4", "resolved": "flash-attn-4"}
    assert audit["distribution_to_import"] == {"flash-attn-4": "flash_attn"}
    assert {record["module"] for record in audit["modules"]} == {
        module_name for _, module_name, _ in FLASH_ATTN_SM90_IMPLEMENTATION_MODULES
    }
    assert all(record["sha256"] and record["bytes"] > 0 for record in audit["modules"])


def test_fa4_runtime_audit_rejects_wrong_distribution_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import_root = tmp_path / FLASH_ATTN_IMPORT_PACKAGE
    modules = _fake_loaded_fa4_modules(import_root)
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _FakeDistribution(tmp_path, name="flash-attn"),
    )

    with pytest.raises(ValueError, match="unexpected distribution"):
        _installed_flash_attn_audit(loaded_modules=modules)


def test_fa4_runtime_audit_rejects_missing_dynamic_sm90_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import_root = tmp_path / FLASH_ATTN_IMPORT_PACKAGE
    modules = _fake_loaded_fa4_modules(import_root)
    missing_module = "flash_attn.cute.flash_bwd_sm90"
    del modules[missing_module]
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _FakeDistribution(tmp_path))

    with pytest.raises(ValueError, match=missing_module):
        _installed_flash_attn_audit(loaded_modules=modules)


def test_fa4_runtime_audit_rejects_module_outside_distribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import_root = tmp_path / FLASH_ATTN_IMPORT_PACKAGE
    modules = _fake_loaded_fa4_modules(import_root)
    module_name = "flash_attn.cute.flash_bwd_sm90"
    outside = tmp_path / "other_distribution" / "flash_bwd_sm90.py"
    outside.parent.mkdir()
    outside.write_text("# wrong distribution\n")
    modules[module_name].__file__ = str(outside)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _FakeDistribution(tmp_path))

    with pytest.raises(ValueError, match="outside audited source root"):
        _installed_flash_attn_audit(loaded_modules=modules)
