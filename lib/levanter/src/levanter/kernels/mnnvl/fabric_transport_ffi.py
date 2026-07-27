# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI transport over GB200 MNNVL fabric memory."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import shutil
import subprocess
import sysconfig
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from filelock import FileLock
from jax._src import distributed as jax_distributed


_BUILD_CACHE_SCHEMA_VERSION = "mnnvl_fabric_exchange_v1"
_CUDA_ARCH = "sm_100"
_EXCHANGE_TARGET = "levanter_mnnvl_exchange"
_GATHER_EXCHANGE_TARGET = "levanter_mnnvl_gather_exchange"
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
_RENDEZVOUS_TIMEOUT_MS = 30 * 60 * 1000


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "fabric_transport_ffi.cu"


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _cuda_home() -> Path:
    raw = os.environ.get("CUDA_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    packaged_cuda = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    if (packaged_cuda / "bin" / "nvcc").is_file():
        return packaged_cuda.resolve()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the MNNVL fabric runtime")
    return Path(nvcc).resolve().parent.parent


def _cache_root() -> Path:
    base = Path(os.environ.get("MARIN_MNNVL_CACHE_DIR", Path.home() / ".cache" / "marin"))
    return base.expanduser().resolve() / "mnnvl_fabric_runtime"


def _shared_library_path() -> Path:
    source = _source_path()
    key = hashlib.sha256()
    key.update(source.read_bytes())
    key.update(_BUILD_CACHE_SCHEMA_VERSION.encode())
    key.update(_CUDA_ARCH.encode())
    key.update(str(_cuda_home()).encode())
    key.update(str(_jaxlib_include_dir()).encode())
    out_dir = _cache_root() / key.hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "libmnnvl_fabric_runtime.so"


def _build_library(output_path: Path) -> None:
    cuda_home = _cuda_home()
    cuda_lib = cuda_home / "lib"
    runtime_library = cuda_lib / "libcudart.so.13"
    if not runtime_library.is_file():
        raise RuntimeError(f"CUDA runtime library is missing: {runtime_library}")
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    command = [
        str(cuda_home / "bin" / "nvcc"),
        "-std=c++17",
        "-O3",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "--cudart=none",
        "-gencode=arch=compute_100,code=sm_100",
        "-I",
        str(_jaxlib_include_dir()),
        str(_source_path()),
        "-L",
        str(cuda_lib),
        "-l:libcudart.so.13",
        "-lcuda",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(cuda_lib),
        "-o",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_library() -> ctypes.CDLL:
    cached = getattr(_load_library, "_library", None)
    if cached is not None:
        return cached
    path = _shared_library_path()
    with FileLock(f"{path}.lock"):
        if not path.is_file():
            _build_library(path)
    library = ctypes.CDLL(str(path), mode=_LIBRARY_DLOPEN_MODE)
    _load_library._library = library
    return library


def _register_target() -> None:
    if getattr(_register_target, "_done", False):
        return
    library = _load_library()
    for target in (_EXCHANGE_TARGET, _GATHER_EXCHANGE_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _register_target._done = True


def _last_error() -> str:
    function = _load_library().levanter_mnnvl_last_error
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else "unknown error"


def shutdown_mnnvl_runtime() -> None:
    if getattr(ensure_mnnvl_runtime, "_signature", None) is None:
        return
    function = _load_library().levanter_mnnvl_shutdown
    function.argtypes = []
    function.restype = None
    function()
    ensure_mnnvl_runtime._signature = None


atexit.register(shutdown_mnnvl_runtime)


def _distributed_client():
    client = jax_distributed.global_state.client
    if client is None:
        raise RuntimeError("MNNVL fabric rendezvous requires an initialized JAX distributed client")
    return client


def _exchange_fabric_handles(
    local_handle: np.ndarray,
    *,
    world_size: int,
    rank: int,
    rendezvous: str,
) -> np.ndarray:
    if world_size == 1:
        return local_handle.reshape(1, -1)

    client = _distributed_client()
    handle_size = local_handle.size
    key = f"{rendezvous}/handle/{rank}"
    client.key_value_set_bytes(key, local_handle.tobytes())
    handles = [
        client.blocking_key_value_get_bytes(f"{rendezvous}/handle/{peer}", _RENDEZVOUS_TIMEOUT_MS)
        for peer in range(world_size)
    ]
    for peer, handle in enumerate(handles):
        if len(handle) != handle_size:
            raise RuntimeError(f"MNNVL fabric handle from rank {peer} has {len(handle)} bytes; expected {handle_size}")
    return np.frombuffer(b"".join(handles), dtype=np.uint8).copy().reshape(world_size, handle_size)


def _wait_at_rendezvous(rendezvous: str, phase: str, world_size: int) -> None:
    if world_size == 1:
        return
    _distributed_client().wait_at_barrier(
        f"{rendezvous}/{phase}",
        _RENDEZVOUS_TIMEOUT_MS,
        list(range(world_size)),
    )


def ensure_mnnvl_runtime(*, buffer_rows: int, row_bytes: int) -> None:
    """Allocate a fabric buffer and map every JAX process's allocation."""
    if buffer_rows <= 0:
        raise ValueError(f"buffer_rows must be positive, got {buffer_rows}")
    if row_bytes <= 0 or row_bytes % 16 != 0:
        raise ValueError(f"row_bytes must be a positive multiple of 16, got {row_bytes}")
    world_size = jax.process_count()
    rank = jax.process_index()
    signature = (world_size, rank, buffer_rows, row_bytes)
    if getattr(ensure_mnnvl_runtime, "_signature", None) == signature:
        return
    if getattr(ensure_mnnvl_runtime, "_signature", None) is not None:
        shutdown_mnnvl_runtime()
    generation = getattr(ensure_mnnvl_runtime, "_generation", 0) + 1
    ensure_mnnvl_runtime._generation = generation
    rendezvous = f"levanter_mnnvl_v1/{generation}/{buffer_rows}/{row_bytes}"

    library = _load_library()
    handle_size_function = library.levanter_mnnvl_fabric_handle_size
    handle_size_function.argtypes = []
    handle_size_function.restype = ctypes.c_int
    handle_size = int(handle_size_function())
    local_handle = np.empty((handle_size,), dtype=np.uint8)

    initialize = library.levanter_mnnvl_init_local
    initialize.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
    ]
    initialize.restype = ctypes.c_int
    status = initialize(
        rank,
        world_size,
        buffer_rows,
        row_bytes,
        local_handle.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        handle_size,
    )
    if status != 0:
        raise RuntimeError(f"Failed to allocate the MNNVL fabric buffer: {_last_error()}")

    try:
        gathered = _exchange_fabric_handles(
            local_handle,
            world_size=world_size,
            rank=rank,
            rendezvous=rendezvous,
        )
        synchronize = library.levanter_mnnvl_sync_handles
        synchronize.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
        synchronize.restype = ctypes.c_int
        status = synchronize(
            gathered.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            world_size,
            handle_size,
        )
        if status != 0:
            raise RuntimeError(f"Failed to map the MNNVL fabric buffers: {_last_error()}")
        _wait_at_rendezvous(rendezvous, "mapped", world_size)
    except BaseException:
        function = library.levanter_mnnvl_shutdown
        function.argtypes = []
        function.restype = None
        function()
        raise
    ensure_mnnvl_runtime._signature = signature
    ensure_mnnvl_runtime._rendezvous = rendezvous


def probe_mnnvl_peer_writes() -> np.ndarray:
    """Write one marker to every peer and return the markers received locally."""
    if getattr(ensure_mnnvl_runtime, "_signature", None) is None:
        raise RuntimeError("ensure_mnnvl_runtime must be called before probing peer writes")
    library = _load_library()
    write = library.levanter_mnnvl_probe_write
    write.argtypes = []
    write.restype = ctypes.c_int
    if write() != 0:
        raise RuntimeError(f"MNNVL peer write failed: {_last_error()}")

    rendezvous = getattr(ensure_mnnvl_runtime, "_rendezvous", None)
    if rendezvous is None:
        raise RuntimeError("MNNVL fabric rendezvous is not initialized")
    probe_generation = getattr(probe_mnnvl_peer_writes, "_generation", 0) + 1
    probe_mnnvl_peer_writes._generation = probe_generation
    _wait_at_rendezvous(rendezvous, f"probe/{probe_generation}", jax.process_count())

    output = np.empty((jax.process_count(),), dtype=np.int32)
    read = library.levanter_mnnvl_probe_read
    read.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    read.restype = ctypes.c_int
    status = read(output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), output.size)
    if status != 0:
        raise RuntimeError(f"MNNVL peer read failed: {_last_error()}")
    return output


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(shape, dtype=dtype)
    return jnp.asarray(cotangent, dtype=dtype)


def _exchange_impl(
    values: jax.Array,
    destination_ranks: jax.Array,
    destination_slots: jax.Array,
    *,
    output_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if values.ndim != 2:
        raise ValueError(f"MNNVL values must be rank-2, got shape={values.shape}")
    if values.dtype != jnp.bfloat16:
        raise ValueError(f"MNNVL values must be bfloat16, got dtype={values.dtype}")
    if destination_ranks.shape != (values.shape[0],):
        raise ValueError(f"destination_ranks must have shape {(values.shape[0],)}, got {destination_ranks.shape}")
    if destination_slots.shape != (values.shape[0],):
        raise ValueError(f"destination_slots must have shape {(values.shape[0],)}, got {destination_slots.shape}")
    if output_rows <= 0:
        raise ValueError(f"output_rows must be positive, got {output_rows}")

    _register_target()
    ensure_mnnvl_runtime(
        buffer_rows=max(values.shape[0], output_rows),
        row_bytes=values.shape[1] * values.dtype.itemsize,
    )
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, values.shape[1]), values.dtype),
        jax.ShapeDtypeStruct((output_rows,), jnp.int32),
        jax.ShapeDtypeStruct((output_rows,), jnp.int32),
    )
    outputs = jax.ffi.ffi_call(
        _EXCHANGE_TARGET,
        output_shapes,
        vmap_method="broadcast_all",
    )(
        values,
        jnp.asarray(destination_ranks, dtype=jnp.int32),
        jnp.asarray(destination_slots, dtype=jnp.int32),
    )
    return outputs[0], outputs[1], outputs[2]


def mnnvl_gather_exchange(
    values: jax.Array,
    source_rows: jax.Array,
    destination_ranks: jax.Array,
    destination_slots: jax.Array,
    *,
    output_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather local source rows while sending an envelope into remote receiver slots."""
    if values.ndim != 2:
        raise ValueError(f"MNNVL values must be rank-2, got shape={values.shape}")
    if values.dtype != jnp.bfloat16:
        raise ValueError(f"MNNVL values must be bfloat16, got dtype={values.dtype}")
    send_rows = source_rows.shape[0]
    if source_rows.shape != (send_rows,):
        raise ValueError(f"source_rows must be rank-1, got {source_rows.shape}")
    if destination_ranks.shape != (send_rows,):
        raise ValueError(f"destination_ranks must have shape {(send_rows,)}, got {destination_ranks.shape}")
    if destination_slots.shape != (send_rows,):
        raise ValueError(f"destination_slots must have shape {(send_rows,)}, got {destination_slots.shape}")
    if output_rows <= 0:
        raise ValueError(f"output_rows must be positive, got {output_rows}")

    _register_target()
    ensure_mnnvl_runtime(
        buffer_rows=max(send_rows, output_rows),
        row_bytes=values.shape[1] * values.dtype.itemsize,
    )
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, values.shape[1]), values.dtype),
        jax.ShapeDtypeStruct((output_rows,), jnp.int32),
        jax.ShapeDtypeStruct((output_rows,), jnp.int32),
    )
    outputs = jax.ffi.ffi_call(
        _GATHER_EXCHANGE_TARGET,
        output_shapes,
        vmap_method="broadcast_all",
    )(
        values,
        jnp.asarray(source_rows, dtype=jnp.int32),
        jnp.asarray(destination_ranks, dtype=jnp.int32),
        jnp.asarray(destination_slots, dtype=jnp.int32),
    )
    return outputs[0], outputs[1], outputs[2]


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def mnnvl_dispatch(
    send_values: jax.Array,
    destination_ranks: jax.Array,
    destination_slots: jax.Array,
    receiver_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Send fixed-envelope rows directly into receiver slots over MNNVL."""
    return _exchange_impl(
        send_values,
        destination_ranks,
        destination_slots,
        output_rows=receiver_capacity,
    )


def _mnnvl_dispatch_fwd(
    send_values: jax.Array,
    destination_ranks: jax.Array,
    destination_slots: jax.Array,
    receiver_capacity: int,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, int, int],
]:
    outputs = _exchange_impl(
        send_values,
        destination_ranks,
        destination_slots,
        output_rows=receiver_capacity,
    )
    _, source_ranks, source_slots = outputs
    return outputs, (source_ranks, source_slots, send_values.shape[0], send_values.shape[1])


def _mnnvl_dispatch_bwd(
    receiver_capacity: int,
    residuals: tuple[jax.Array, jax.Array, int, int],
    cotangents: tuple[
        jax.Array | jax.custom_derivatives.SymbolicZero,
        jax.Array | jax.custom_derivatives.SymbolicZero,
        jax.Array | jax.custom_derivatives.SymbolicZero,
    ],
) -> tuple[jax.Array, None, None]:
    del receiver_capacity
    source_ranks, source_slots, send_rows, hidden = residuals
    receiver_cotangent = _materialize_cotangent(
        cotangents[0],
        shape=(source_ranks.shape[0], hidden),
        dtype=jnp.bfloat16,
    )
    send_cotangent, _, _ = _exchange_impl(
        receiver_cotangent,
        source_ranks,
        source_slots,
        output_rows=send_rows,
    )
    return send_cotangent, None, None


mnnvl_dispatch.defvjp(_mnnvl_dispatch_fwd, _mnnvl_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def mnnvl_combine(
    receiver_values: jax.Array,
    source_ranks: jax.Array,
    source_slots: jax.Array,
    dispatch_destination_ranks: jax.Array,
    dispatch_destination_slots: jax.Array,
    send_rows: int,
) -> jax.Array:
    """Return receiver rows to their source envelope over MNNVL."""
    returned_values, _, _ = _exchange_impl(
        receiver_values,
        source_ranks,
        source_slots,
        output_rows=send_rows,
    )
    return returned_values


def _mnnvl_combine_fwd(
    receiver_values: jax.Array,
    source_ranks: jax.Array,
    source_slots: jax.Array,
    dispatch_destination_ranks: jax.Array,
    dispatch_destination_slots: jax.Array,
    send_rows: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, int, int]]:
    returned_values, _, _ = _exchange_impl(
        receiver_values,
        source_ranks,
        source_slots,
        output_rows=send_rows,
    )
    return returned_values, (
        dispatch_destination_ranks,
        dispatch_destination_slots,
        receiver_values.shape[0],
        receiver_values.shape[1],
    )


def _mnnvl_combine_bwd(
    send_rows: int,
    residuals: tuple[jax.Array, jax.Array, int, int],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array, None, None, None, None]:
    destination_ranks, destination_slots, receiver_capacity, hidden = residuals
    send_cotangent = _materialize_cotangent(
        cotangent,
        shape=(send_rows, hidden),
        dtype=jnp.bfloat16,
    )
    receiver_cotangent, _, _ = _exchange_impl(
        send_cotangent,
        destination_ranks,
        destination_slots,
        output_rows=receiver_capacity,
    )
    return receiver_cotangent, None, None, None, None


mnnvl_combine.defvjp(_mnnvl_combine_fwd, _mnnvl_combine_bwd)
