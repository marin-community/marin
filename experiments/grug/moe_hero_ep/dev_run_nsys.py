# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``dev_run`` with an explicit CUDA profiler range around the XProf capture window.

Nsight Systems is launched by ``iris.hooks.nsys_main --capture-range`` as
``--capture-range=cudaProfilerApi --capture-range-end=stop``: it collects nothing until the
application calls ``cudaProfilerStart``/``cuProfilerStart``, and stops at the matching ``Stop``.

The 2026-08-08 r8 round got that range for free, apparently from XLA's GPU profiler. On the current
jaxlib it does not happen: the 2026-08-19 first attempt
(`/muchanem/mokprof-hero-nsys-r1-20260819`) ran the full window, exited 0, and nsys printed
``No reports were generated`` — the range never opened, so there was nothing to write.

This wrapper closes that gap without touching anything the measurement depends on. It patches
``jax.profiler.start_trace`` / ``stop_trace`` — the two calls
``levanter.callbacks.profiler.profile`` already makes on exactly the profiled step boundaries, on
exactly the tracing process — to bracket them with the driver's profiler API. On a rank Nsight is
not attached to, ``cuProfilerStart`` is a no-op.

Run it exactly like ``dev_run``; every option is forwarded.
"""

import ctypes
import logging

import jax

logger = logging.getLogger("mok.nsys_range")

# libcuda is the driver; it is always present when CUDA is. libcudart's name is version-suffixed
# and ships inside the gpu extra's wheels, so it is tried by soname and treated as optional.
_DRIVER_SONAME = "libcuda.so.1"
_RUNTIME_SONAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
_libraries: dict[str, ctypes.CDLL | None] = {}


def _load(soname: str) -> ctypes.CDLL | None:
    if soname not in _libraries:
        try:
            _libraries[soname] = ctypes.CDLL(soname)
        except OSError as exc:
            logger.info("could not load %s: %s", soname, exc)
            _libraries[soname] = None
    return _libraries[soname]


def _call(symbol_driver: str, symbol_runtime: str) -> None:
    """Call the driver entry point, and the runtime one when it is loadable.

    Both are called because Nsight documents the capture range against
    ``cudaProfilerStart``/``cuProfilerStart`` without saying which it hooks; calling both is
    idempotent and removes the guess.
    """
    results = []
    driver = _load(_DRIVER_SONAME)
    if driver is not None:
        try:
            results.append((symbol_driver, getattr(driver, symbol_driver)()))
        except Exception as exc:
            results.append((symbol_driver, repr(exc)))
    for soname in _RUNTIME_SONAMES:
        runtime = _load(soname)
        if runtime is None:
            continue
        try:
            results.append((f"{soname}:{symbol_runtime}", getattr(runtime, symbol_runtime)()))
        except Exception as exc:
            results.append((f"{soname}:{symbol_runtime}", repr(exc)))
        break
    logger.warning("CUDA profiler range %s -> %s", symbol_driver, results)


_original_start = jax.profiler.start_trace
_original_stop = jax.profiler.stop_trace


def _start_trace(*args, **kwargs):
    _call("cuProfilerStart", "cudaProfilerStart")
    return _original_start(*args, **kwargs)


def _stop_trace(*args, **kwargs):
    result = _original_stop(*args, **kwargs)
    _call("cuProfilerStop", "cudaProfilerStop")
    return result


jax.profiler.start_trace = _start_trace
jax.profiler.stop_trace = _stop_trace

from experiments.grug.moe_hero_ep.dev_run import main  # noqa: E402  - patch before the import runs

if __name__ == "__main__":
    main()
