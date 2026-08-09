# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the persistent CuTeDSL kernel object-code cache.

``cutlass.jax`` is CUDA-only, so these exercise the integration against a stand-in
that mirrors the three symbols the cache touches: the ``get_or_compile_kernel``
entry point bound into ``cutlass.jax.primitive``, the in-process
``_CUTLASS_COMPILE_CACHE`` dict, and the ``CompileResult`` record.
"""

import dataclasses
import hashlib
import sys
import types
from typing import Any

import pytest

from rigging.cache import PersistentKvCache

from levanter import cutlass_kernel_cache
from levanter.cutlass_kernel_cache import cute_launcher_factory, install


def _kernel_store(directory) -> PersistentKvCache:
    """A single-directory store standing in for the cutlass cache's disk tier."""
    return PersistentKvCache.at(str(directory))


@dataclasses.dataclass(frozen=True)
class FakeCompileResult:
    module: bytes
    fingerprint: bytes
    spec: Any


@dataclasses.dataclass(frozen=True)
class FakeFunctionSpec:
    """Stand-in for ``cutlass.jax.compile.FunctionSpec``: frozen, with a stable repr."""

    shape: tuple[int, ...]


class FakeCutlass:
    """Records every CuTeDSL compile the cache does not serve from the store."""

    def __init__(self) -> None:
        self.compiled: list[str] = []
        self.in_process: dict[Any, Any] = {}
        self.compile_module = types.ModuleType("cutlass.jax.compile")
        self.compile_module._CUTLASS_COMPILE_CACHE = self.in_process
        self.compile_module.CompileResult = FakeCompileResult
        self.primitive_module = types.ModuleType("cutlass.jax.primitive")
        self.primitive_module.get_or_compile_kernel = self._compile

    def _compile(self, fn: Any, spec: Any) -> FakeCompileResult:
        name = getattr(fn, "kernel_name", "anonymous")
        self.compiled.append(name)
        module = f"objectcode:{name}:{spec!r}".encode()
        result = FakeCompileResult(module=module, fingerprint=hashlib.sha256(module).digest(), spec=spec)
        self.in_process[(fn, spec)] = result
        return result

    def forget_process_state(self) -> None:
        """Drop everything a process restart would drop, keeping the on-disk store."""
        self.in_process.clear()
        self.primitive_module.get_or_compile_kernel = self._compile

    def compile_kernel(self, fn: Any, spec: Any) -> FakeCompileResult:
        return self.primitive_module.get_or_compile_kernel(fn, spec)


@pytest.fixture
def fake_cutlass(monkeypatch) -> FakeCutlass:
    cutlass = FakeCutlass()
    monkeypatch.setitem(sys.modules, "cutlass.jax.compile", cutlass.compile_module)
    monkeypatch.setitem(sys.modules, "cutlass.jax.primitive", cutlass.primitive_module)
    monkeypatch.setattr(cutlass_kernel_cache, "_kernel_source_revision", lambda: "source-v1")
    return cutlass


def _build_launcher(modules, *, tile: int, dtype: str = "bf16") -> Any:
    def launcher(stream):
        raise AssertionError("a launcher is never called on the host")

    launcher.kernel_name = f"tile{tile}-{dtype}"
    return launcher


_build_launcher.__module__ = "levanter.grug.testing"
build_launcher = cute_launcher_factory(_build_launcher)


def test_launcher_factory_rejects_positional_configuration():
    with pytest.raises(TypeError, match="positional parameter named 'modules'"):

        @cute_launcher_factory
        def invalid_factory(tile: int) -> int:
            return tile


def test_a_restarted_process_loads_the_stored_object_instead_of_compiling(fake_cutlass, tmp_path):
    spec = FakeFunctionSpec(shape=(8, 16))

    install(_kernel_store(tmp_path))
    cold = fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)
    assert fake_cutlass.compiled == ["tile128-bf16"]

    # A restart drops the in-process memory tier, so a fresh cache must read the store.
    fake_cutlass.forget_process_state()
    install(_kernel_store(tmp_path))
    warm = fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert warm.module == cold.module
    assert warm.fingerprint == hashlib.sha256(cold.module).digest()


def test_configuration_and_specification_both_discriminate_stored_kernels(fake_cutlass, tmp_path):
    install(_kernel_store(tmp_path))

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))
    fake_cutlass.compile_kernel(build_launcher(None, tile=256), FakeFunctionSpec(shape=(8, 16)))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 32)))

    assert fake_cutlass.compiled == ["tile128-bf16", "tile256-bf16", "tile128-bf16"]
    assert len(list(tmp_path.iterdir())) == 3

    fake_cutlass.forget_process_state()
    install(_kernel_store(tmp_path))
    served = fake_cutlass.compile_kernel(build_launcher(None, tile=256), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16", "tile256-bf16", "tile128-bf16"]
    assert served.module == b"objectcode:tile256-bf16:FakeFunctionSpec(shape=(8, 16))"


def test_a_launch_with_no_source_revision_is_compiled_but_not_stored(fake_cutlass, tmp_path, monkeypatch):
    def unavailable_revision() -> str:
        raise ValueError("missing compiler file")

    monkeypatch.setattr(cutlass_kernel_cache, "_kernel_source_revision", unavailable_revision)
    install(_kernel_store(tmp_path))

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert list(tmp_path.iterdir()) == []


def test_a_source_revision_change_invalidates_the_stored_object(fake_cutlass, tmp_path, monkeypatch):
    revision = ["source-v1"]
    monkeypatch.setattr(cutlass_kernel_cache, "_kernel_source_revision", lambda: revision[0])
    spec = FakeFunctionSpec(shape=(8, 16))

    install(_kernel_store(tmp_path))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)

    fake_cutlass.forget_process_state()
    revision[0] = "source-v2"
    install(_kernel_store(tmp_path))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)

    assert fake_cutlass.compiled == ["tile128-bf16", "tile128-bf16"]
    assert len(list(tmp_path.iterdir())) == 2


def test_source_revision_combines_internal_source_and_dependency_lock(monkeypatch):
    source_hash = ["source-v1"]
    dependency_hash = ["dependencies-v1"]
    monkeypatch.setattr(cutlass_kernel_cache, "directory_content_hash", lambda _path: source_hash[0])
    monkeypatch.setattr(cutlass_kernel_cache, "workspace_lock_hash", lambda _path: dependency_hash[0])
    cutlass_kernel_cache._kernel_source_revision.cache_clear()
    try:
        original = cutlass_kernel_cache._kernel_source_revision()
        source_hash[0] = "source-v2"
        cutlass_kernel_cache._kernel_source_revision.cache_clear()
        changed_source = cutlass_kernel_cache._kernel_source_revision()
        dependency_hash[0] = "dependencies-v2"
        cutlass_kernel_cache._kernel_source_revision.cache_clear()
        changed_dependency = cutlass_kernel_cache._kernel_source_revision()
    finally:
        cutlass_kernel_cache._kernel_source_revision.cache_clear()

    assert len({original, changed_source, changed_dependency}) == 3


def test_a_launcher_outside_the_covered_source_tree_is_not_stored(fake_cutlass, tmp_path):
    @cute_launcher_factory
    def external_launcher(modules, *, tile: int) -> Any:
        def launcher(_stream):
            raise AssertionError("a launcher is never called on the host")

        launcher.kernel_name = f"external-{tile}"
        return launcher

    install(_kernel_store(tmp_path))
    fake_cutlass.compile_kernel(external_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["external-128"]
    assert list(tmp_path.iterdir()) == []


def test_a_launcher_without_an_identity_is_compiled_but_not_stored(fake_cutlass, tmp_path):
    cache = _kernel_store(tmp_path)
    install(cache)

    def untagged(stream):
        raise AssertionError("a launcher is never called on the host")

    untagged.kernel_name = "untagged"

    fake_cutlass.compile_kernel(untagged, FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["untagged"]
    assert list(tmp_path.iterdir()) == []


def test_a_specification_that_reprs_an_address_is_not_stored(fake_cutlass, tmp_path):
    """Such a key would change every process, so it would miss forever and litter the store."""
    cache = _kernel_store(tmp_path)
    install(cache)

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), object())

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_store_compiles_without_failing(fake_cutlass, tmp_path):
    """A cache directory a task cannot write degrades to a compile rather than failing."""
    blocked = tmp_path / "file"
    blocked.write_bytes(b"")
    install(_kernel_store(blocked / "kernels"))

    result = fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert result.module == b"objectcode:tile128-bf16:FakeFunctionSpec(shape=(8, 16))"


def test_install_is_a_noop_when_cutlass_jax_will_not_import(fake_cutlass, monkeypatch, tmp_path):
    """A CPU task on the GPU image has the package but no CUDA bindings to import it with."""
    monkeypatch.setitem(sys.modules, "cutlass.jax.primitive", None)

    install(_kernel_store(tmp_path))

    # Nothing was patched, so the compile never reaches the store.
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))
    assert list(tmp_path.iterdir()) == []
