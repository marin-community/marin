# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI wrapper for DeepEP's CUDA dispatch-layout kernel.

This wrapper intentionally targets the narrowest DeepEP entry point that has a
raw CUDA API and no Torch process-group dependency:
`deep_ep::layout::get_dispatch_layout`.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from levanter.kernels.deepep.availability import (
    LAYOUT_REQUIRED_FILES,
    deepep_cache_root,
    deepep_cuda_arch_flag,
    deepep_layout_source,
    deepep_source_root,
)

_TARGET_NAME = "levanter_deepep_get_dispatch_layout"
_BUILD_CACHE_SCHEMA_VERSION = "layout_ffi_int32_topk_v2"


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _ffi_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_layout_ffi.cu"


def _deepep_source_root() -> Path:
    return deepep_source_root(
        required_files=LAYOUT_REQUIRED_FILES,
        purpose="the DeepEP JAX FFI layout kernel",
        requires_layout_source=True,
    )


def _cache_root() -> Path:
    return deepep_cache_root("deepep_layout_ffi")


def _shared_library_path() -> Path:
    deepep_root = _deepep_source_root()
    layout_source = deepep_layout_source(deepep_root)
    key = hashlib.sha256()
    for path in (_ffi_source(), layout_source):
        key.update(str(path).encode("utf-8"))
        key.update(path.read_bytes())
    key.update(str(_jaxlib_include_dir()).encode("utf-8"))
    key.update(str(deepep_root).encode("utf-8"))
    key.update(" ".join(deepep_cuda_arch_flag()).encode("utf-8"))
    key.update(_BUILD_CACHE_SCHEMA_VERSION.encode("utf-8"))
    digest = key.hexdigest()[:16]
    out_dir = _cache_root() / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "libdeepep_layout_ffi.so"


def _build_shared_library(out_path: Path) -> None:
    deepep_root = _deepep_source_root()
    layout_source = deepep_layout_source(deepep_root)
    jax_include = _jaxlib_include_dir()
    cmd = [
        "nvcc",
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "-DDISABLE_NVSHMEM",
        *deepep_cuda_arch_flag(),
        "-I",
        str(jax_include),
        "-I",
        str(deepep_root),
        "-I",
        str(deepep_root / "csrc"),
        str(_ffi_source()),
        str(layout_source),
        "-o",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _load_library() -> ctypes.CDLL:
    lib_path = _shared_library_path()
    if not lib_path.exists():
        _build_shared_library(lib_path)
    return ctypes.cdll.LoadLibrary(str(lib_path))


def _register_target() -> None:
    if getattr(_register_target, "_done", False):
        return
    library = _load_library()
    handler = getattr(library, _TARGET_NAME)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        _TARGET_NAME,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )
    jax.ffi.register_ffi_target_as_batch_partitionable(_TARGET_NAME)
    _register_target._done = True


def deepep_get_dispatch_layout(
    topk_idx: jax.Array,
    *,
    num_ranks: int,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run DeepEP's CUDA dispatch-layout kernel via JAX FFI."""
    _register_target()
    topk_idx_i32 = jnp.asarray(topk_idx, dtype=jnp.int32)
    tokens, topk = topk_idx_i32.shape
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((num_ranks,), jnp.int32),
        jax.ShapeDtypeStruct((num_experts,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, num_ranks), jnp.bool_),
        jax.ShapeDtypeStruct((tokens, topk * 2), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        _TARGET_NAME,
        result_shape_dtypes,
        vmap_method="broadcast_all",
    )(topk_idx_i32)
    return results[:3]
