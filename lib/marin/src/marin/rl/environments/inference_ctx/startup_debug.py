# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import Any, cast

_DUMMY_INIT_PROGRESS_EVERY_PARAMS = 25
_DUMMY_INIT_PROGRESS_EVERY_SECONDS = 30.0
_DUMMY_INIT_SLOW_PARAM_SECONDS = 15.0
_DUMMY_INIT_LARGE_PARAM_NUMEL = 50_000_000


def _tensor_numel(param: Any) -> int:
    numel = getattr(param, "numel", None)
    if callable(numel):
        return int(numel())

    shape = getattr(param, "shape", None)
    if shape is None:
        return 0

    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _is_meta_tensor(param: Any) -> bool:
    device = getattr(param, "device", None)
    if device is None:
        return False
    if getattr(device, "type", None) == "meta":
        return True
    return str(device) == "meta"


def _format_shape(param: Any) -> str:
    shape = getattr(param, "shape", None)
    if shape is None:
        return "?"
    return "x".join(str(int(dim)) for dim in shape)


def install_dummy_init_progress_instrumentation(
    *,
    vllm_dummy_loader_module: Any,
    vllm_weight_utils_module: Any,
    emit: Callable[[str], None],
) -> None:
    """Install aggregated progress logging for upstream dummy weight init."""
    original = getattr(vllm_weight_utils_module, "initialize_dummy_weights", None)
    initialize_single_dummy_weight = getattr(vllm_weight_utils_module, "initialize_single_dummy_weight", None)
    if original is None or initialize_single_dummy_weight is None:
        return
    initialize_single_weight = cast(Callable[[Any, float, float, int], None], initialize_single_dummy_weight)
    if getattr(original, "_marin_dummy_init_progress", False):
        return

    def uses_meta_device(module: Any) -> bool:
        quant_method = getattr(module, "quant_method", None)
        return getattr(quant_method, "uses_meta_device", False)

    @functools.wraps(original)
    def wrapped(
        model: Any,
        model_config: Any,
        low: float = -1e-3,
        high: float = 1e-3,
        seed: int = 1234,
    ) -> None:
        del model_config
        has_online_quant = any(uses_meta_device(module) for module in model.modules())
        state_items = tuple(model.state_dict().items())

        tracked_items: list[tuple[str, Any, int]] = []
        total_numel = 0
        for name, param in state_items:
            if has_online_quant and _is_meta_tensor(param):
                continue
            param_numel = _tensor_numel(param)
            tracked_items.append((name, param, param_numel))
            total_numel += param_numel

        total_params = len(tracked_items)
        t_start = time.perf_counter()
        last_progress = t_start
        completed_numel = 0

        emit(f"START dummy-init.loop params={total_params} total_numel={total_numel} pid={os.getpid()}")

        for index, (name, param, param_numel) in enumerate(tracked_items, start=1):
            param_shape = _format_shape(param)
            param_dtype = getattr(param, "dtype", "?")
            should_trace_param = param_numel >= _DUMMY_INIT_LARGE_PARAM_NUMEL
            t_param_start = time.perf_counter()

            if should_trace_param:
                emit(
                    "dummy-init.param-start "
                    f"index={index}/{total_params} name={name} shape={param_shape} "
                    f"dtype={param_dtype} numel={param_numel} pid={os.getpid()}"
                )

            initialize_single_weight(param, low, high, seed)

            param_elapsed = time.perf_counter() - t_param_start
            completed_numel += param_numel
            now = time.perf_counter()
            should_log_progress = (
                index == total_params
                or index % _DUMMY_INIT_PROGRESS_EVERY_PARAMS == 0
                or now - last_progress >= _DUMMY_INIT_PROGRESS_EVERY_SECONDS
            )

            if should_trace_param or param_elapsed >= _DUMMY_INIT_SLOW_PARAM_SECONDS:
                emit(
                    "dummy-init.param-end "
                    f"index={index}/{total_params} name={name} shape={param_shape} "
                    f"dtype={param_dtype} numel={param_numel} elapsed={param_elapsed:.2f}s "
                    f"done_numel={completed_numel}/{total_numel} pid={os.getpid()}"
                )
                last_progress = now
                continue

            if should_log_progress:
                emit(
                    "dummy-init.progress "
                    f"params={index}/{total_params} numel={completed_numel}/{total_numel} "
                    f"elapsed={now - t_start:.2f}s pid={os.getpid()}"
                )
                last_progress = now

        emit(
            "END dummy-init.loop "
            f"in {time.perf_counter() - t_start:.2f}s params={total_params}/{total_params} "
            f"numel={completed_numel}/{total_numel} pid={os.getpid()}"
        )

    wrapped._marin_startup_timed = True
    wrapped._marin_dummy_init_progress = True
    vllm_weight_utils_module.initialize_dummy_weights = wrapped
    vllm_dummy_loader_module.initialize_dummy_weights = wrapped
