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
from rigging.provenance import Provenance

from levanter import cutlass_kernel_cache
from levanter.cutlass_kernel_cache import cute_launcher_factory, install


def _provenance(*, tree_hash: str) -> Provenance:
    """Launch provenance carrying only the field the cache key reads."""
    return Provenance(tree_hash=tree_hash, base_commit="", dirty=False, branch=None, built_by=None)


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
    return cutlass


@cute_launcher_factory
def build_launcher(modules, *, tile: int, dtype: str = "bf16") -> Any:
    def launcher(stream):
        raise AssertionError("a launcher is never called on the host")

    launcher.kernel_name = f"tile{tile}-{dtype}"
    return launcher


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


def test_a_new_source_revision_invalidates_stored_kernels(fake_cutlass, tmp_path, monkeypatch):
    """The launch tree hash is the whole source identity, so a changed tree must miss.

    A launcher rarely holds its whole kernel in its own file — the segmented backward is
    defined in ``_fa4_cute_kernels`` and built from ``_fa4_cute_segmented_bwd`` — so keying
    on any one file would serve a stale object after the kernel itself changed.
    """
    spec = FakeFunctionSpec(shape=(8, 16))
    store = str(tmp_path / "store")

    compiled_per_revision = []
    for revision in ("treehash-original", "treehash-edited"):
        monkeypatch.setattr(
            cutlass_kernel_cache,
            "launch_provenance",
            lambda revision=revision: _provenance(tree_hash=revision),
        )
        fake_cutlass.forget_process_state()
        install(_kernel_store(store))
        fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)
        compiled_per_revision.append(list(fake_cutlass.compiled))

    assert compiled_per_revision == [["tile128-bf16"], ["tile128-bf16", "tile128-bf16"]]


def test_the_same_source_revision_is_served_from_the_store(fake_cutlass, tmp_path, monkeypatch):
    """A resume runs the same tree, which is the case the store exists to serve."""
    spec = FakeFunctionSpec(shape=(8, 16))
    store = str(tmp_path / "store")
    monkeypatch.setattr(cutlass_kernel_cache, "launch_provenance", lambda: _provenance(tree_hash="treehash-stable"))

    install(_kernel_store(store))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)
    fake_cutlass.forget_process_state()
    install(_kernel_store(store))
    fake_cutlass.compile_kernel(build_launcher(None, tile=128), spec)

    assert fake_cutlass.compiled == ["tile128-bf16"]


def test_a_launch_with_no_source_revision_is_compiled_but_not_stored(fake_cutlass, tmp_path, monkeypatch):
    """Outside a checkout with no stamped provenance there is no source identity to key on.

    Storing anyway would let one revision's object serve another's.
    """
    monkeypatch.setattr(cutlass_kernel_cache, "launch_provenance", lambda: _provenance(tree_hash=""))
    install(_kernel_store(tmp_path))

    fake_cutlass.compile_kernel(build_launcher(None, tile=128), FakeFunctionSpec(shape=(8, 16)))

    assert fake_cutlass.compiled == ["tile128-bf16"]
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
