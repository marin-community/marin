# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger(__name__)

MARIN_VLLM_FAST_BOOTSTRAP = "MARIN_VLLM_FAST_BOOTSTRAP"
MARIN_ZERO_BOOTSTRAP_LOAD_FORMAT = "marin_zero_bootstrap"
_SUPPORTED_TEXT_ARCHITECTURES = frozenset({"LlamaForCausalLM"})


def install_marin_fast_tpu_bootstrap_patch(*, emit: Callable[[str], None] | None = None) -> None:
    """Install Marin's guarded TPU bootstrap patch for async-native vLLM.

    The patch is intentionally narrow. It only changes the `flax_nnx` Llama
    bootstrap path when Marin explicitly enables fast bootstrap. All other
    architectures and load paths continue to use the dependency's default
    behavior.
    """
    if os.environ.get(MARIN_VLLM_FAST_BOOTSTRAP) != "1":
        return

    try:
        import jax
        from flax import nnx
        from tpu_inference.models.common import model_loader as model_loader_module
    except ImportError as exc:
        logger.warning("Could not import TPU modules for Marin fast bootstrap patch: %s", exc)
        return

    original = getattr(model_loader_module, "_get_nnx_model", None)
    if original is None:
        logger.warning("Could not find TPU _get_nnx_model for Marin fast bootstrap patch")
        return
    if getattr(original, "_marin_fast_bootstrap_patch", False):
        return
    original_get_nnx_model = cast(Callable[[Any, Any, Any, Any], Any], original)

    def _emit(message: str) -> None:
        if emit is not None:
            emit(message)
            return
        logger.info(message)

    @functools.wraps(original)
    def patched(model_class: Any, vllm_config: Any, rng: Any, mesh: Any) -> Any:
        if not _should_use_marin_fast_bootstrap(
            model_class=model_class,
            vllm_config=vllm_config,
            model_loader_module=model_loader_module,
        ):
            return original_get_nnx_model(model_class, vllm_config, rng, mesh)

        _emit(
            "Using Marin abstract-state flax_nnx path "
            f"arch={model_class.__name__} load_format={vllm_config.load_config.load_format!r}"
        )
        return _build_abstract_bootstrap_model(
            model_class=model_class,
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            jax_module=jax,
            nnx_module=nnx,
            emit=_emit,
        )

    patched._marin_fast_bootstrap_patch = True
    model_loader_module._get_nnx_model = patched
    _emit("Applying Marin fast TPU bootstrap patch")


def _should_use_marin_fast_bootstrap(
    *,
    model_class: type[Any],
    vllm_config: Any,
    model_loader_module: Any,
) -> bool:
    if os.environ.get(MARIN_VLLM_FAST_BOOTSTRAP) != "1":
        return False

    load_format = getattr(vllm_config.load_config, "load_format", None)
    if load_format not in {"dummy", MARIN_ZERO_BOOTSTRAP_LOAD_FORMAT}:
        return False

    if model_class.__name__ not in _SUPPORTED_TEXT_ARCHITECTURES:
        return False

    if model_loader_module.apply_qwix_on_abstract_model(vllm_config):
        return False

    quant_config = getattr(vllm_config.model_config.hf_config, "quantization_config", None)
    if quant_config:
        return False

    return True


def _build_abstract_bootstrap_model(
    *,
    model_class: type[Any],
    vllm_config: Any,
    rng: Any,
    mesh: Any,
    jax_module: Any,
    nnx_module: Any,
    emit: Callable[[str], None],
) -> Any:
    def create_abstract_model() -> Any:
        return model_class(vllm_config, rng, mesh)

    t_start = time.perf_counter()
    with jax_module.set_mesh(mesh):
        model = nnx_module.eval_shape(create_abstract_model)
        abstract_state = nnx_module.state(model)

        param_count = 0
        total_numel = 0
        for _, variable in nnx_module.to_flat_state(abstract_state):
            if not isinstance(variable, nnx_module.Param):
                continue
            value = variable.value
            param_count += 1
            total_numel += _parameter_numel(value.shape)

        emit(
            "START Marin abstract-state bootstrap "
            f"arch={model_class.__name__} params={param_count} numel={total_numel}"
        )
        seeded_state = nnx_module.map_state(
            lambda path, variable: _seed_rng_variable(path=path, variable=variable, rng=rng),
            abstract_state,
        )
        nnx_module.update(model, seeded_state)
        emit(
            "END Marin abstract-state bootstrap "
            f"arch={model_class.__name__} params={param_count} numel={total_numel} "
            f"elapsed={time.perf_counter() - t_start:.2f}s"
        )

    elapsed = time.perf_counter() - t_start
    emit(
        "END Marin fast-bootstrap model prep "
        f"arch={model_class.__name__} params={param_count} numel={total_numel} elapsed={elapsed:.2f}s"
    )
    logger.info(
        "Marin abstract-state bootstrap prepared %s parameters (%s scalars) in %.2fs",
        param_count,
        total_numel,
        elapsed,
    )
    return model


def _parameter_numel(shape: Any) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _seed_rng_variable(
    *,
    path: tuple[Any, ...],
    variable: Any,
    rng: Any,
) -> Any:
    if path and path[0] == "rng" and path[-1] == "key":
        return variable.replace(value=rng)
    return variable
