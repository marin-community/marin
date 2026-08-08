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
keyed on the launcher's configuration, the launch tree hash, the argument specification,
the device architecture, and the CuTeDSL, QuACK, and FlashAttention versions.
``cutlass.jax`` derives a kernel's fingerprint from its object code by SHA-256, so
a stored blob reconstructs a compile with nothing else.

Launchers opt in through :func:`cute_launcher_factory`.
"""

import functools
import hashlib
import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import jax
from rigging.cache import PersistentKvCache
from rigging.provenance import launch_provenance

logger = logging.getLogger(__name__)

_KERNEL_IDENTITY_ATTR = "_levanter_cute_kernel_identity"
_VERSIONED_PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels", "jaxlib", "flash-attn-4")
_KERNEL_CACHE_PREFIX = "cutlass-kernels"


def cute_launcher_factory(build: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize a ``@cute.jit`` launcher factory and stamp its result with a stable identity.

    ``cutlass.jax`` keys its compile cache on launcher identity, so a factory that
    returns a fresh launcher per call forces a fresh CuTeDSL compile even when the
    resulting kernel is byte-identical. Memoizing collapses shape-identical call
    sites — expert chunks, scanned layers, the repeated backward postprocess —
    onto one compile.

    The stamped identity names the kernel within one source revision: the factory's
    qualified name and its keyword arguments. Keyword arguments carry the whole kernel
    configuration; the positional argument is the CuTe module bundle, a singleton that
    configures nothing. :func:`_kernel_key` adds the revision.
    """

    @functools.lru_cache(maxsize=None)
    @functools.wraps(build)
    def memoized(*args: Any, **kwargs: Any) -> Any:
        launcher = build(*args, **kwargs)
        identity = "|".join(
            [
                f"{build.__module__}.{build.__qualname__}",
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


def _source_revision() -> str:
    """Content hash of the working tree the process was launched from, or ``""`` if unknown.

    A launcher rarely holds the whole kernel in its own file -- the segmented backward is
    defined in ``_fa4_cute_kernels`` and built from ``_fa4_cute_segmented_bwd`` -- so keying
    on any one file serves a stale object after the kernel changes. The launch tree hash
    covers every source file at once. It is content-addressed rather than a commit hash, so
    it survives rebases and amends that do not change content, and
    :func:`rigging.provenance.launch_provenance` derives it through ``git stash create`` so
    uncommitted edits count. In a bundle with no checkout it comes from ``MARIN_PROVENANCE``,
    stamped by the submitting client.
    """
    return launch_provenance().tree_hash


def cutlass_kernel_cache() -> PersistentKvCache:
    """The standard cache for compiled CuTeDSL kernel object code, one object per key.

    Memory over region-local temp object storage, assembled by
    :meth:`PersistentKvCache.for_prefix`. An unreachable store degrades to a compile.
    """
    return PersistentKvCache.for_prefix(_KERNEL_CACHE_PREFIX)


def install(cache: PersistentKvCache) -> None:
    """Route ``cutlass.jax`` kernel compiles through ``cache``.

    Patches ``cutlass.jax.primitive``, which binds ``get_or_compile_kernel`` at
    import time, rather than the function's defining module. Idempotent, and a
    no-op when ``cutlass.jax`` will not import: a CPU task on the GPU image has
    the package but not the CUDA bindings it pulls in.
    """
    try:
        primitive = importlib.import_module("cutlass.jax.primitive")
        compile_module = importlib.import_module("cutlass.jax.compile")
    except ImportError as exc:
        logger.info("CuTeDSL kernel cache skipped, cutlass.jax unavailable: %s", exc)
        return

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
    logger.info("CuTeDSL kernel cache installed at %s", cache.location())


def _kernel_key(fn: Any, spec: Any) -> str | None:
    """Return the store key for a kernel, or ``None`` if it cannot be named stably.

    Launchers built outside :func:`cute_launcher_factory` carry no identity, a launch
    outside any checkout has no source revision to key on, and a specification whose
    ``repr`` embeds an object address would key on the address and miss forever. All three
    fall back to compiling.
    """
    identity = getattr(fn, _KERNEL_IDENTITY_ATTR, None)
    if identity is None:
        return None

    revision = _source_revision()
    if not revision:
        logger.warning("CuTeDSL kernel not cacheable, no launch tree hash to key on: %s", identity)
        return None

    specification = repr(spec)
    if " at 0x" in specification:
        logger.warning("CuTeDSL kernel not cacheable, specification repr carries an address: %s", identity)
        return None

    payload = "\n".join([identity, revision, specification, _device_architecture(), _package_versions()])
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
