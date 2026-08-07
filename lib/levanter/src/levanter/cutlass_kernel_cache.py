# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist CuTeDSL kernel object code across processes.

``cutlass.jax`` compiles a CuTe kernel during MLIR lowering and embeds the
resulting object code as an attribute on the HLO custom call. That is upstream of
JAX's persistent compilation cache, which is keyed on the lowered module, so a
compilation-cache hit still pays for every kernel. CuTeDSL's own in-memory and
on-disk caches cannot cover it either: ``cutlass.cute.compile`` forces
``no_cache=True``, leaving only an in-process dict keyed on launcher identity.

:func:`install` wraps ``get_or_compile_kernel`` to consult an object store first,
keyed on the launcher's configuration and defining source, the argument
specification, the device architecture, and the CuTeDSL and QuACK versions.
``cutlass.jax`` derives a kernel's fingerprint from its object code by SHA-256, so
a stored blob reconstructs a compile with nothing else.

Launchers opt in through :func:`cute_launcher_factory`.
"""

import dataclasses
import functools
import hashlib
import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import jax
from rigging.filesystem import StoragePath, atomic_rename

logger = logging.getLogger(__name__)

_KERNEL_IDENTITY_ATTR = "_levanter_cute_kernel_identity"
_VERSIONED_PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels", "jaxlib")
_OBJECT_SUFFIX = ".o"


def cute_launcher_factory(build: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize a ``@cute.jit`` launcher factory and stamp its result with a stable identity.

    ``cutlass.jax`` keys its compile cache on launcher identity, so a factory that
    returns a fresh launcher per call forces a fresh CuTeDSL compile even when the
    resulting kernel is byte-identical. Memoizing collapses shape-identical call
    sites — expert chunks, scanned layers, the repeated backward postprocess —
    onto one compile.

    The stamped identity names the kernel across processes: the factory's
    qualified name, its keyword arguments, and a digest of its defining source.
    Keyword arguments carry the whole kernel configuration; the positional
    argument is the CuTe module bundle, a singleton that configures nothing.
    """

    @functools.lru_cache(maxsize=None)
    @functools.wraps(build)
    def memoized(*args: Any, **kwargs: Any) -> Any:
        launcher = build(*args, **kwargs)
        identity = "|".join(
            [
                f"{build.__module__}.{build.__qualname__}",
                _source_digest(build),
                *(f"{name}={kwargs[name]!r}" for name in sorted(kwargs)),
            ]
        )
        setattr(launcher, _KERNEL_IDENTITY_ATTR, identity)
        return launcher

    return memoized


@functools.lru_cache(maxsize=None)
def cutlass_call(launcher: Any, **kwargs: Any) -> Any:
    """Return ``cutlass.jax.cutlass_call(launcher, **kwargs)``, memoized on its arguments.

    ``cutlass_call`` builds its JAX entry point as a fresh ``@jax.jit`` closure, and
    JAX's tracing cache is keyed on function identity, so rebuilding it per call
    site re-traces and re-lowers the same kernel as its own nested ``pjit``.

    Keyword arguments must be passed in a consistent order: ``lru_cache`` does not
    sort them, so a reordered call misses rather than hits.
    """
    cjax = importlib.import_module("cutlass.jax")
    return cjax.cutlass_call(launcher, **kwargs)


@functools.lru_cache(maxsize=None)
def _source_digest(build: Callable[..., Any]) -> str:
    """Digest the source file that defines ``build``.

    The launcher body is levanter source, so nothing else in the key would notice
    an edit to it. Everything the body calls into is either QuACK or CuTeDSL,
    covered by their package versions, or configuration, covered by the arguments.
    """
    source = importlib.import_module(build.__module__).__file__
    assert source is not None, f"{build.__module__} has no source file to digest"
    with open(source, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class CutlassKernelCache:
    """Content-addressed store of compiled CuTeDSL kernel object code.

    Args:
        directory: Store location, local or any fsspec URL.
    """

    directory: str

    def load(self, key: str) -> bytes | None:
        # A store that cannot be reached is a miss, not a failure: the caller
        # compiles instead. Training must not depend on the cache being healthy.
        try:
            path = self._object(key)
            return path.read_bytes() if path.exists() else None
        except OSError as exc:
            logger.warning("CuTeDSL kernel cache unreadable, compiling instead: %s", exc)
            return None

    def store(self, key: str, module: bytes) -> None:
        # Every process compiles the same kernels at once, so stage and rename
        # rather than let a reader see a half-written object.
        try:
            path = str(self._object(key))
            with atomic_rename(path) as staged:
                StoragePath(staged).write_bytes(module)
        except OSError as exc:
            logger.warning("CuTeDSL kernel cache unwritable, not storing %s: %s", key, exc)

    def _object(self, key: str) -> StoragePath:
        return _root(self.directory) / f"{key}{_OBJECT_SUFFIX}"


@functools.lru_cache(maxsize=None)
def _root(directory: str) -> StoragePath:
    path = StoragePath(directory)
    path.mkdirs()
    return path


def install_if_available(cache: CutlassKernelCache) -> bool:
    """Install ``cache`` when ``cutlass.jax`` can actually be imported.

    The package being present does not mean it imports: ``cutlass.jax`` pulls in
    CUDA bindings, so it fails on a CPU task running the GPU image, or on a host
    whose driver does not match. Returns whether the cache was installed.
    """
    try:
        importlib.import_module("cutlass.jax.primitive")
    except ImportError as exc:
        logger.info("CuTeDSL kernel cache skipped, cutlass.jax unavailable: %s", exc)
        return False

    install(cache)
    return True


def install(cache: CutlassKernelCache) -> None:
    """Route ``cutlass.jax`` kernel compiles through ``cache``.

    Patches ``cutlass.jax.primitive``, which binds ``get_or_compile_kernel`` at
    import time, rather than the function's defining module. Idempotent.
    """
    primitive = importlib.import_module("cutlass.jax.primitive")
    compile_module = importlib.import_module("cutlass.jax.compile")
    if getattr(primitive.get_or_compile_kernel, "_levanter_kernel_cache", None) is not None:
        logger.info("CuTeDSL kernel cache already installed")
        return

    original = primitive.get_or_compile_kernel
    in_process = compile_module._CUTLASS_COMPILE_CACHE
    compile_result = compile_module.CompileResult

    def get_or_compile_kernel(fn: Any, spec: Any) -> Any:
        cached = in_process.get((fn, spec))
        if cached is not None:
            return cached

        key = _kernel_key(fn, spec)
        if key is None:
            return original(fn, spec)

        module = cache.load(key)
        if module is not None:
            logger.info("CuTeDSL kernel cache hit: %s", key)
            result = compile_result(module=module, fingerprint=hashlib.sha256(module).digest(), spec=spec)
            in_process[(fn, spec)] = result
            return result

        logger.info("CuTeDSL kernel cache miss, compiling: %s", key)
        result = original(fn, spec)
        cache.store(key, result.module)
        return result

    get_or_compile_kernel._levanter_kernel_cache = cache
    primitive.get_or_compile_kernel = get_or_compile_kernel
    logger.info("CuTeDSL kernel cache installed at %s", cache.directory)


def _kernel_key(fn: Any, spec: Any) -> str | None:
    """Return the store key for a kernel, or ``None`` if it cannot be named stably.

    Launchers built outside :func:`cute_launcher_factory` carry no identity, and a
    specification whose ``repr`` embeds an object address would key on the address
    and miss forever. Both fall back to compiling.
    """
    identity = getattr(fn, _KERNEL_IDENTITY_ATTR, None)
    if identity is None:
        return None

    specification = repr(spec)
    if " at 0x" in specification:
        logger.warning("CuTeDSL kernel not cacheable, specification repr carries an address: %s", identity)
        return None

    payload = "\n".join([identity, specification, _device_architecture(), _package_versions()])
    return hashlib.sha256(payload.encode()).hexdigest()


def _device_architecture() -> str:
    device = jax.local_devices()[0]
    return f"{device.platform}-{getattr(device, 'compute_capability', device.device_kind)}"


@functools.lru_cache(maxsize=1)
def _package_versions() -> str:
    def installed(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "absent"

    return " ".join(f"{package}={installed(package)}" for package in _VERSIONED_PACKAGES)
