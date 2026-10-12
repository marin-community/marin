# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the kernels of the GPU math-library calls in a compiled program before the program first runs.

cuBLASLt loads its kernel libraries (``cuLibraryLoadData``) the first time a GEMM needs them, whatever
``CUDA_MODULE_LOADING`` says. Inside a multi-process step that load blocks while collectives spin on peers
blocked in the same load, and GB200 first steps deadlock at 100% GPU. Running each GEMM of the program once,
alone on each local device, loads its kernels up front. XLA refuses modules that already hold cuBLAS calls, so
each one is rebuilt as the ``dot`` it came from (same operand shapes and layouts, dimension numbers and output
type) and rewritten again; with the shared autotune results XLA picks the same algorithm.
"""

import json
import logging
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lib import xla_client
from jaxlib import _hlo

logger = logging.getLogger(__name__)

LIBRARY_CALL_TARGETS = ("__cublas$lt$matmul", "__cublas$gemm")

_CALL = re.compile(
    r"^\s*(?:ROOT\s+)?%?[\w.\-]+ = (?P<result>.+?) custom-call\((?P<operands>.*?)\), "
    r'(?P<attrs>custom_call_target="(?P<target>[^"]+)".*)$'
)
_ARRAY_TYPE = r"\w+\[[^\]]*\](?:\{[^}]*\})?"
# Every instruction with an array result: its name and type (operands are printed by name only).
_DEFINITION = re.compile(rf"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+) = (?P<type>{_ARRAY_TYPE}) ")
_SHAPE = re.compile(r"^(\w+)\[([^\]]*)\]")
_FIRST_RESULT = re.compile(r"^\(?(\w+\[[^\]]*\](?:\{[^}]*\})?)")
# The warm-up compiles may autotune: a GEMM missing from the shared results must not fail the run.
_COMPILE_OVERRIDES = [("xla_gpu_require_complete_aot_autotune_results", False)]
_DTYPES = {
    "pred": np.bool_,
    "s8": np.int8,
    "u8": np.uint8,
    "s32": np.int32,
    "f16": np.float16,
    "bf16": jnp.bfloat16,
    "f32": np.float32,
    "f8e4m3fn": jnp.float8_e4m3fn,
    "f8e5m2": jnp.float8_e5m2,
}


def library_call_modules(hlo_text: str) -> list[tuple[str, list[str]]]:
    """One single-``dot`` HLO module per distinct cuBLAS GEMM in ``hlo_text``, with its operand types."""
    lines = hlo_text.splitlines()
    types = {m["name"]: m["type"] for m in map(_DEFINITION.match, lines) if m is not None}
    modules: dict[tuple[str, tuple[str, ...], str], tuple[str, list[str]]] = {}
    for line in lines:
        match = _CALL.match(line)
        if match is None or match["target"] not in LIBRARY_CALL_TARGETS:
            continue
        # The GEMM's own operands come first; a bias or scale operand only feeds its epilogue.
        inline = re.findall(rf"({_ARRAY_TYPE}) %", match["operands"])
        names = re.findall(r"%([\w.\-]+)", match["operands"])
        operand_types = inline[:2] if inline else [types[n] for n in names[:2] if n in types]
        result = _FIRST_RESULT.match(match["result"])
        if result is None or len(operand_types) != 2:
            raise ValueError(f"unparsed cuBLAS call: {line.strip()[:300]}")
        attrs = match["attrs"]
        config_start = attrs.index("backend_config=") + len("backend_config=")
        config, _ = json.JSONDecoder().raw_decode(attrs[config_start:])
        dims = config["gemm_backend_config"]["dot_dimension_numbers"]
        dot_dims = ", ".join(
            f"{name}_dims={{{','.join(dims.get(f'{name}_dimensions', []))}}}"
            for name in ("lhs_batch", "lhs_contracting", "rhs_batch", "rhs_contracting")
        )
        key = (result[1], tuple(operand_types), dot_dims)
        if key in modules:
            continue
        text = (
            "HloModule library_call\n\nENTRY main {\n"
            f"  p0 = {operand_types[0]} parameter(0)\n  p1 = {operand_types[1]} parameter(1)\n"
            f"  ROOT dot = {result[1]} dot(p0, p1), {dot_dims}\n}}\n"
        )
        modules[key] = (text, operand_types)
    return list(modules.values())


def _zeros(type_text: str, device: jax.Device) -> jax.Array:
    match = _SHAPE.match(type_text)
    if match is None or match[1] not in _DTYPES:
        raise ValueError(f"unsupported library-call operand type {type_text!r}")
    shape = tuple(int(d) for d in match[2].split(",")) if match[2] else ()
    return jax.device_put(np.zeros(shape, _DTYPES[match[1]]), device)


def load_library_kernels(hlo_text: str, devices: list[jax.Device]) -> None:
    """Compile and run every distinct library call of ``hlo_text`` once on zeros on each of ``devices``."""
    start = time.perf_counter()
    modules = library_call_modules(hlo_text)
    for text, operand_types in modules:
        mlir = xla_client._xla.mlir.hlo_to_stablehlo(_hlo.hlo_module_from_text(text).as_serialized_hlo_module_proto())
        for device in devices:
            options = xla_client.CompileOptions()
            options.env_option_overrides = _COMPILE_OVERRIDES
            executable = device.client.compile_and_load(mlir, xla_client.DeviceList((device,)), options)
            results = executable.execute_sharded([_zeros(t, device) for t in operand_types])
            jax.block_until_ready(results.consume_with_handlers([lambda arrays: arrays]))
    logger.info(
        "Loaded the kernels of %d distinct cuBLAS GEMMs on %d devices in %.1f s",
        len(modules),
        len(devices),
        time.perf_counter() - start,
    )
