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
keyed on the launcher's configuration, the kernel source and installed compiler
bytes, the argument specification, and the device architecture. ``cutlass.jax``
derives a kernel's fingerprint from its object code by SHA-256, so a stored blob
reconstructs a compile with nothing else.

Launchers opt in through :func:`cute_launcher_factory`.
"""

import functools
import hashlib
import importlib
import importlib.util
import inspect
import logging
import pathlib
from typing import Any, Callable

import jax
from rigging.cache import PersistentKvCache, compile_cache_key, installed_distribution_fingerprint

logger = logging.getLogger(__name__)

_KERNEL_IDENTITY_ATTR = "_levanter_cute_kernel_identity"
_KERNEL_SOURCE_COVERED_ATTR = "_levanter_cute_kernel_source_covered"
_KERNEL_CACHE_PREFIX = "cutlass-kernels"
_KERNEL_CACHE_SCHEMA = "cutlass-object-v3"
_CUTE_TOOLCHAIN_DISTRIBUTIONS = (
    "apache-tvm-ffi",
    "cuda-bindings",
    "cuda-python",
    "flash-attn-4",
    "nvidia-cuda-nvdisasm",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu13",
    "quack-kernels",
)
_CUTE_TOOLCHAIN_PACKAGES = ("cuda", "cutlass", "flash_attn", "quack", "tvm_ffi")


def cute_launcher_factory(build: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize a ``@cute.jit`` launcher factory and stamp its result with a stable identity.

    ``cutlass.jax`` keys its compile cache on launcher identity, so a factory that
    returns a fresh launcher per call forces a fresh CuTeDSL compile even when the
    resulting kernel is byte-identical. Memoizing collapses shape-identical call
    sites — expert chunks, scanned layers, the repeated backward postprocess —
    onto one compile.

    The stamped identity is the factory's qualified name and its keyword arguments, which
    carry the whole kernel configuration. A factory may additionally accept one positional
    argument named ``modules`` for the CuTe dependency bundle; other positional
    configuration is rejected because it would be absent from the persistent identity.
    """
    positional_parameters = [
        parameter
        for parameter in inspect.signature(build).parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    if (
        any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in positional_parameters)
        or len(positional_parameters) > 1
        or (positional_parameters and positional_parameters[0].name != "modules")
    ):
        raise TypeError("CuTe launcher factories may only accept a positional parameter named 'modules'")

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
        setattr(launcher, _KERNEL_SOURCE_COVERED_ATTR, build.__module__.startswith("levanter.grug."))
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
    try:
        source_revision = _kernel_source_revision()
    except (OSError, ValueError) as exc:
        logger.warning("CuTeDSL persistent kernel cache disabled, source identity unavailable: %s", exc)
        source_revision = None

    def get_or_compile_kernel(fn: Any, spec: Any) -> Any:
        cached = in_process.get((fn, spec))
        if cached is not None:
            return cached

        key = _kernel_key(fn, spec, source_revision)
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


def _kernel_key(fn: Any, spec: Any, source_revision: str | None) -> str | None:
    """Hash the kernel identity, covered source, specification, and device.

    ``None`` when any of those is missing or unstable, which falls back to compiling.
    """
    identity = getattr(fn, _KERNEL_IDENTITY_ATTR, None)
    if identity is None:
        return None
    if not getattr(fn, _KERNEL_SOURCE_COVERED_ATTR, False):
        logger.warning("CuTeDSL kernel not cacheable, launcher source is outside levanter.grug: %s", identity)
        return None
    if source_revision is None:
        return None

    specification = repr(spec)
    if " at 0x" in specification:
        logger.warning("CuTeDSL kernel not cacheable, specification repr carries an address: %s", identity)
        return None

    payload = "\n".join([_KERNEL_CACHE_SCHEMA, identity, source_revision, specification, _device_architecture()])
    return hashlib.sha256(payload.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _kernel_source_revision() -> str:
    grug_sources = pathlib.Path(__file__).resolve().parent / "grug"
    toolchain = installed_distribution_fingerprint(_CUTE_TOOLCHAIN_DISTRIBUTIONS)
    return compile_cache_key(
        [grug_sources, *_toolchain_source_roots()],
        environment=[_KERNEL_CACHE_SCHEMA, toolchain, *_CUTE_TOOLCHAIN_PACKAGES],
    )


def _toolchain_source_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for package_name in _CUTE_TOOLCHAIN_PACKAGES:
        specification = importlib.util.find_spec(package_name)
        if specification is None:
            raise ValueError(f"CuTe toolchain package is unavailable: {package_name}")
        if specification.submodule_search_locations is not None:
            roots.extend(pathlib.Path(location) for location in specification.submodule_search_locations)
        elif specification.origin not in (None, "built-in", "frozen"):
            roots.append(pathlib.Path(specification.origin))
        else:
            raise ValueError(f"CuTe toolchain package has no source tree: {package_name}")
    return roots


def _device_architecture() -> str:
    device = jax.local_devices()[0]
    return f"{device.platform}-{getattr(device, 'compute_capability', device.device_kind)}"
