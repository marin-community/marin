# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch-free JAX boundary for generated relation-edge reverse stages."""

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.xla_relation_edge_reverse_ffi import (
    GeneratedRelationEdgeReverseFfi,
    RelationEdgeReverseFfiPlan,
)


@dataclass(frozen=True)
class RelationEdgeReverseCudaCompilePlan:
    """Exact standalone NVCC invocation for one generated handler."""

    source_path: Path
    library_path: Path
    argv: tuple[str, ...]


def register_cuda_relation_edge_reverse_ffi(
    generated: GeneratedRelationEdgeReverseFfi,
    library: ctypes.CDLL,
) -> None:
    """Register one compiled relation-edge reverse handler with JAX CUDA."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cuda_relation_edge_reverse_ffi(
    generated: GeneratedRelationEdgeReverseFfi,
    plan: RelationEdgeReverseFfiPlan,
    received_cotangent: jax.Array,
    route_padded_rows: jax.Array,
    route_weights: jax.Array,
    saved_edge_output: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Invoke the generated edge Map and feature Fold through typed FFI."""
    _validate_inputs(plan, received_cotangent, route_padded_rows, route_weights, saved_edge_output)
    results = jax.ffi.ffi_call(
        generated.target,
        (
            jax.ShapeDtypeStruct((plan.padded_rows, plan.features), jnp.bfloat16),
            jax.ShapeDtypeStruct((plan.received_rows, plan.route_slots), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        received_cotangent,
        route_padded_rows,
        route_weights,
        saved_edge_output,
    )
    return results[0], results[1]


def evaluate_relation_edge_reverse_jax(
    plan: RelationEdgeReverseFfiPlan,
    received_cotangent: jax.Array,
    route_padded_rows: jax.Array,
    route_weights: jax.Array,
    saved_edge_output: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the typed-FFI Map/Fold contract as a one-device JAX program."""
    _validate_inputs(plan, received_cotangent, route_padded_rows, route_weights, saved_edge_output)
    flat_rows = route_padded_rows.reshape(-1)
    valid = flat_rows >= 0
    safe_rows = jnp.where(valid, flat_rows, 0)
    edge_cotangent = (received_cotangent[:, None, :].astype(jnp.float32) * route_weights[:, :, None]).reshape(
        -1, plan.features
    )
    row_ids = jnp.arange(plan.padded_rows, dtype=jnp.int32)
    row_matches = flat_rows[:, None] == row_ids[None, :]
    padded_cotangent = jnp.sum(
        jnp.where(row_matches[:, :, None], edge_cotangent[:, None, :], 0.0),
        axis=0,
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    received_by_edge = jnp.repeat(received_cotangent, plan.route_slots, axis=0).astype(jnp.float32)
    saved_by_edge = saved_edge_output[safe_rows].astype(jnp.float32)

    def ordered_feature_fold(values: tuple[jax.Array, jax.Array]) -> jax.Array:
        saved, cotangent = values

        def update(feature: int, state: jax.Array) -> jax.Array:
            return state + saved[feature] * cotangent[feature]

        return jax.lax.fori_loop(0, plan.features, update, jnp.asarray(0.0, dtype=jnp.float32))

    route_weight_cotangent = jax.vmap(ordered_feature_fold)((saved_by_edge, received_by_edge))
    route_weight_cotangent = jnp.where(valid, route_weight_cotangent, 0.0)
    return padded_cotangent, route_weight_cotangent.reshape(plan.received_rows, plan.route_slots)


def relation_edge_reverse_cuda_compile_plan(
    generated: GeneratedRelationEdgeReverseFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
    jaxlib_include: Path | None = None,
) -> RelationEdgeReverseCudaCompilePlan:
    """Build a Torch-free compile plan using CUDA runtime and XLA FFI only."""
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    include_directory = jaxlib_include or Path(jaxlib.__file__).resolve().parent / "include"
    if not include_directory.is_dir():
        raise ValueError(f"jaxlib include directory does not exist: {include_directory}")
    source_path = directory / f"{generated.handler_symbol}.cu"
    library_path = directory / f"{generated.handler_symbol}.so"
    argv = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        f"-arch={architecture}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(include_directory),
        str(source_path),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
    )
    return RelationEdgeReverseCudaCompilePlan(source_path, library_path, argv)


def compile_cuda_relation_edge_reverse_ffi(
    generated: GeneratedRelationEdgeReverseFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile and load one generated relation-edge reverse handler."""
    plan = relation_edge_reverse_cuda_compile_plan(
        generated,
        directory=directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    directory.mkdir(parents=True, exist_ok=True)
    plan.source_path.write_text(generated.source + "\n")
    subprocess.run(plan.argv, check=True)
    return ctypes.CDLL(str(plan.library_path))


def _validate_inputs(
    plan: RelationEdgeReverseFfiPlan,
    received_cotangent: jax.Array,
    route_padded_rows: jax.Array,
    route_weights: jax.Array,
    saved_edge_output: jax.Array,
) -> None:
    specifications = (
        ("received_cotangent", received_cotangent, (plan.received_rows, plan.features), jnp.bfloat16),
        ("route_padded_rows", route_padded_rows, (plan.received_rows, plan.route_slots), jnp.int32),
        ("route_weights", route_weights, (plan.received_rows, plan.route_slots), jnp.float32),
        ("saved_edge_output", saved_edge_output, (plan.padded_rows, plan.features), jnp.bfloat16),
    )
    for name, value, shape, dtype in specifications:
        if value.shape != shape:
            raise ValueError(f"relation-edge reverse input {name!r} must have shape {shape}, found {value.shape}")
        if np.dtype(value.dtype) != np.dtype(dtype):
            raise ValueError(
                f"relation-edge reverse input {name!r} must have dtype {np.dtype(dtype)}, found {value.dtype}"
            )
