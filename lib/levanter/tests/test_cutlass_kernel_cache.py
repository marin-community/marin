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
import importlib
import sys
import textwrap
import types
from typing import Any

import pytest

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_kernel_cache, install


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
    return cutlass


@cute_launcher_factory
def build_launcher(modules, *, tile: int, dtype: str = "bf16") -> Any:
    def launcher(stream):
        raise AssertionError("a launcher is never called on the host")

    launcher.kernel_name = f"tile{tile}-{dtype}"
    return launcher


def test_a_restarted_process_loads_the_stored_object_instead_of_compiling(fake_cutlass, tmp_path):
    spec = FakeFunctionSpec(shape=(8, 16))

    install(cutlass_kernel_cache(str(tmp_path)))
    cold = fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)
    assert fake_cutlass.compiled == ["tile128-bf16"]

    # A restart drops the in-process memory tier, so a fresh cache must read the store.
    fake_cutlass.forget_process_state()
    install(cutlass_kernel_cache(str(tmp_path)))
    warm = fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert warm.module == cold.module
    assert warm.fingerprint == hashlib.sha256(cold.module).digest()


def test_configuration_and_specification_both_discriminate_stored_kernels(fake_cutlass, tmp_path):
    install(cutlass_kernel_cache(str(tmp_path)))

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))
    fake_cutlass.compile_kernel(build_launcher(None, tile=256), FakeFunctionSpec(shape=(8, 16)))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 32)))

    assert fake_cutlass.compiled == ["tile128-bf16", "tile256-bf16", "tile128-bf16"]
    assert len(list(tmp_path.iterdir())) == 3

    fake_cutlass.forget_process_state()
    install(cutlass_kernel_cache(str(tmp_path)))
    served = fake_cutlass.compile_kernel(build_launcher(None, tile=256), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16", "tile256-bf16", "tile128-bf16"]
    assert served.module == b"objectcode:tile256-bf16:FakeFunctionSpec(shape=(8, 16))"


def test_editing_the_launcher_source_invalidates_its_kernels(fake_cutlass, tmp_path, monkeypatch):
    """A launcher body is levanter source: nothing else in the key notices an edit."""
    source = textwrap.dedent(
        """
        from levanter.cutlass_kernel_cache import cute_launcher_factory

        @cute_launcher_factory
        def build(modules, *, tile: int):
            def launcher(stream):
                raise AssertionError("a launcher is never called on the host")

            launcher.kernel_name = f"tile{tile}"
            return launcher
        """
    )
    module_dir = tmp_path / "src"
    module_dir.mkdir()
    monkeypatch.syspath_prepend(str(module_dir))
    store = str(tmp_path / "store")

    compiled_per_revision = []
    for revision in ("original", "edited"):
        name = f"edited_launcher_{revision}"
        (module_dir / f"{name}.py").write_text(f"# revision: {revision}\n{source}")
        # The import system caches each directory's listing against its mtime, so a
        # second file written inside one mtime tick is invisible to the finder.
        importlib.invalidate_caches()
        module = __import__(name)
        fake_cutlass.forget_process_state()
        install(cutlass_kernel_cache(store))
        fake_cutlass.compile_kernel(module.build(None, tile=128), FakeFunctionSpec(shape=(8, 16)))
        compiled_per_revision.append(list(fake_cutlass.compiled))

    assert compiled_per_revision == [["tile128"], ["tile128", "tile128"]]


def test_a_launcher_without_an_identity_is_compiled_but_not_stored(fake_cutlass, tmp_path):
    cache = cutlass_kernel_cache(str(tmp_path))
    install(cache)

    def untagged(stream):
        raise AssertionError("a launcher is never called on the host")

    untagged.kernel_name = "untagged"

    fake_cutlass.compile_kernel(untagged, FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["untagged"]
    assert list(tmp_path.iterdir()) == []


def test_a_specification_that_reprs_an_address_is_not_stored(fake_cutlass, tmp_path):
    """Such a key would change every process, so it would miss forever and litter the store."""
    cache = cutlass_kernel_cache(str(tmp_path))
    install(cache)

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), object())

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_store_compiles_without_failing(fake_cutlass, tmp_path):
    """The store follows the compilation cache dir, which a task may not be able to write."""
    blocked = tmp_path / "file"
    blocked.write_bytes(b"")
    install(cutlass_kernel_cache(str(blocked / "kernels")))

    result = fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16"]
    assert result.module == b"objectcode:tile128-bf16:FakeFunctionSpec(shape=(8, 16))"


def test_install_is_a_noop_when_cutlass_jax_will_not_import(fake_cutlass, monkeypatch, tmp_path):
    """A CPU task on the GPU image has the package but no CUDA bindings to import it with."""
    monkeypatch.setitem(sys.modules, "cutlass.jax.primitive", None)

    install(cutlass_kernel_cache(str(tmp_path)))

    # Nothing was patched, so the compile never reaches the store.
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))
    assert list(tmp_path.iterdir()) == []
