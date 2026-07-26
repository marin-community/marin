# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise HybridEP dispatch/combine through JAX typed FFI."""

import atexit
import ctypes
import importlib
import os
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)
from iris.runtime.jax_init import _poll_for_coordinator, initialize_jax
from levanter.kernels.hybridep import hybridep_combine, hybridep_dispatch

_DISPATCH_TARGET = "levanter_hybridep_dispatch"
_COMBINE_TARGET = "levanter_hybridep_combine"
_COMBINE_WITH_PROBABILITIES_TARGET = "levanter_hybridep_combine_with_probabilities"


def _trace(message: str) -> None:
    rank = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "?")
    print(f"HYBRID_EP_JAX_FFI rank={rank} {message}", flush=True)


def _rank_info(job_info) -> tuple[int, int, int]:
    device_ids = os.environ[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV].split(",")
    if len(device_ids) != 1:
        raise ValueError(f"HybridEP expects one device per process, got {device_ids}")
    return (
        int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV]),
        int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV]),
        int(device_ids[0]),
    )


def _initialize_torch_process_group() -> None:
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("HybridEP JAX smoke must run inside an Iris job")
    rank, world_size, device_index = _rank_info(job_info)
    endpoint_name = f"hybridep-jax-ffi-torch-{job_info.job_id.to_safe_token()}-attempt-{job_info.attempt_id}"
    port = job_info.ports.get("jax", 8476) + 1
    address = f"{job_info.advertise_host}:{port}"
    if rank == 0:
        endpoint_id = iris_ctx().registry.register(endpoint_name, address)
        atexit.register(iris_ctx().registry.unregister, endpoint_id)
    else:
        address = _poll_for_coordinator(
            iris_ctx().resolver,
            endpoint_name,
            timeout=600,
            poll_interval=1,
        )
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{address}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{device_index}"),
    )
    torch.cuda.set_device(device_index)


def _load_hybrid_module(source_root: Path):
    package = types.ModuleType("deep_ep")
    package.__path__ = [str(source_root / "deep_ep")]
    sys.modules["deep_ep"] = package
    sys.path.insert(0, str(source_root))
    return importlib.import_module("hybrid_ep_cpp")


def _register_targets(library: ctypes.CDLL) -> None:
    for target in (_DISPATCH_TARGET, _COMBINE_TARGET, _COMBINE_WITH_PROBABILITIES_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)


def _initialize_runtime(
    module, library: ctypes.CDLL, source_root: Path, *, hidden: int, tokens: int, experts: int
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_experts = experts // world_size
    init = library.levanter_hybridep_init
    init.argtypes = [
        ctypes.py_object,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    init.restype = ctypes.c_int
    status = init(
        dist.group.WORLD,
        rank,
        world_size,
        hidden,
        tokens,
        local_experts,
        32,
        32,
        os.fsencode(source_root / "deep_ep"),
    )
    if status != 0:
        last_error = library.levanter_hybridep_last_error
        last_error.restype = ctypes.c_char_p
        raise RuntimeError(last_error().decode())
    del module


def _dispatch(hidden: jax.Array, routing_map: jax.Array, probabilities: jax.Array, output_rows: int, local_experts: int):
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, hidden.shape[1]), jnp.bfloat16),
        jax.ShapeDtypeStruct((output_rows,), jnp.float32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
    )
    return jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        output_shapes,
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(hidden, routing_map, probabilities)


def _combine(expert_hidden: jax.Array, handle_token: jax.Array, output_rows: int):
    output_shape = jax.ShapeDtypeStruct((output_rows, expert_hidden.shape[1]), jnp.bfloat16)
    return jax.ffi.ffi_call(
        _COMBINE_TARGET,
        output_shape,
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(expert_hidden, handle_token)


def _combine_with_probabilities(
    expert_hidden: jax.Array,
    expert_probabilities: jax.Array,
    handle_token: jax.Array,
    output_rows: int,
    num_experts: int,
):
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, expert_hidden.shape[1]), jnp.bfloat16),
        jax.ShapeDtypeStruct((output_rows, num_experts), jnp.float32),
    )
    return jax.ffi.ffi_call(
        _COMBINE_WITH_PROBABILITIES_TARGET,
        output_shapes,
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(expert_hidden, expert_probabilities, handle_token)


def main() -> None:
    _trace("initialize_jax start")
    initialize_jax()
    _trace("initialize_jax done")
    _initialize_torch_process_group()
    _trace("torch process group done")
    source_root = Path(os.environ["HYBRID_EP_SOURCE"]).resolve()
    module = _load_hybrid_module(source_root)
    _trace("hybrid module loaded")
    library = ctypes.CDLL(module.__file__, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    _register_targets(library)
    _trace("FFI targets registered")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hidden_dim = int(os.environ.get("HYBRID_EP_HIDDEN", "512"))
    tokens = int(os.environ.get("HYBRID_EP_TOKENS", "1024"))
    topk = int(os.environ.get("HYBRID_EP_TOPK", "4"))
    num_experts = int(os.environ.get("HYBRID_EP_EXPERTS", str(world_size * 4)))
    if num_experts % world_size:
        raise ValueError(f"experts={num_experts} must be divisible by ranks={world_size}")
    _initialize_runtime(
        module,
        library,
        source_root,
        hidden=hidden_dim,
        tokens=tokens,
        experts=num_experts,
    )
    _trace(f"HybridEP runtime initialized torch_reserved_bytes={torch.cuda.memory_reserved()}")

    token_indices = np.arange(tokens, dtype=np.int64)[:, None]
    topk_indices = np.arange(topk, dtype=np.int64)[None, :]
    selected = (rank * tokens * topk + token_indices * topk + topk_indices) % num_experts
    drop_every = int(os.environ.get("HYBRID_EP_DROP_EVERY", "0"))
    assignment_indices = token_indices * topk + topk_indices
    assignment_keep = np.ones((tokens, topk), dtype=np.bool_)
    if drop_every > 0:
        assignment_keep = assignment_indices % drop_every != 0
    kept_per_token = np.sum(assignment_keep, axis=1, dtype=np.int32)
    routing_map = np.zeros((tokens, num_experts), dtype=np.bool_)
    probabilities = np.zeros((tokens, num_experts), dtype=np.float32)
    routing_map[token_indices, selected] = assignment_keep
    probabilities[token_indices, selected] = 1.0
    hidden = (jnp.arange(tokens * hidden_dim, dtype=jnp.float32).reshape(tokens, hidden_dim) % 97 / 97).astype(
        jnp.bfloat16
    )
    routing_map_jax = jnp.asarray(routing_map)
    probabilities_jax = jnp.asarray(probabilities)
    output_rows = tokens * topk
    gemm_probe_columns = int(os.environ.get("HYBRID_EP_GEMM_PROBE_COLUMNS", "0"))
    gemm_probe_weight = jnp.ones((hidden_dim, gemm_probe_columns), dtype=jnp.bfloat16)

    @jax.jit
    def roundtrip(x, routes, probs):
        dispatched, dispatched_probs, group_sizes, handle_token = _dispatch(
            x,
            routes,
            probs,
            output_rows,
            num_experts // world_size,
        )
        if gemm_probe_columns:
            projected = dispatched @ gemm_probe_weight
            dispatched = dispatched + (jnp.sum(projected.astype(jnp.float32), axis=1, keepdims=True) * 1e-6).astype(
                dispatched.dtype
            )
        combined = _combine(
            (dispatched.astype(jnp.float32) * dispatched_probs[:, None]).astype(jnp.bfloat16),
            handle_token,
            tokens,
        )
        return combined, group_sizes

    _trace("roundtrip start")
    combined, group_sizes = roundtrip(hidden, routing_map_jax, probabilities_jax)
    _trace("roundtrip done")
    combined_host = np.asarray(combined, dtype=np.float32)
    if gemm_probe_columns:
        assert np.all(np.isfinite(combined_host))
    else:
        np.testing.assert_allclose(
            combined_host,
            np.asarray(hidden, dtype=np.float32) * kept_per_token[:, None],
            rtol=2e-2,
            atol=2e-2,
        )
    received_rows = int(jnp.sum(group_sizes))
    assert 0 < received_rows <= output_rows
    _trace(f"received rows={received_rows} output envelope={output_rows}")

    @jax.jit
    def repeated_roundtrip(x, routes, probs):
        for _ in range(4):
            dispatched, dispatched_probs, _, handle_token = _dispatch(
                x,
                routes,
                probs,
                output_rows,
                num_experts // world_size,
            )
            x = _combine(
                (dispatched.astype(jnp.float32) * dispatched_probs[:, None]).astype(jnp.bfloat16),
                handle_token,
                tokens,
            )
        return x

    _trace("repeated roundtrip start")
    repeated = repeated_roundtrip(hidden, routing_map_jax, probabilities_jax / topk)
    _trace("repeated roundtrip done")
    np.testing.assert_allclose(
        np.asarray(repeated, dtype=np.float32),
        np.asarray(hidden, dtype=np.float32) * (kept_per_token[:, None] / topk) ** 4,
        rtol=2e-2,
        atol=2e-2,
    )
    if os.environ.get("HYBRID_EP_SKIP_GRADIENTS") == "1":
        if rank == 0:
            print(
                "HYBRID_EP_JAX_FFI_FORWARD_PASS "
                f"ranks={world_size} tokens={tokens} hidden={hidden_dim} experts={num_experts} "
                f"topk={topk} drop_every={drop_every} gemm_probe_columns={gemm_probe_columns}",
                flush=True,
            )
        dist.barrier()
        library.levanter_hybridep_shutdown()
        dist.destroy_process_group()
        return

    @jax.custom_vjp
    def differentiable_roundtrip(x, routes, probs):
        dispatched, dispatched_probs, _, handle_token = _dispatch(
            x,
            routes,
            probs,
            output_rows,
            num_experts // world_size,
        )
        return _combine(
            (dispatched.astype(jnp.float32) * dispatched_probs[:, None]).astype(jnp.bfloat16),
            handle_token,
            tokens,
        )

    def differentiable_roundtrip_fwd(x, routes, probs):
        dispatched, dispatched_probs, _, handle_token = _dispatch(
            x,
            routes,
            probs,
            output_rows,
            num_experts // world_size,
        )
        combined = _combine(
            (dispatched.astype(jnp.float32) * dispatched_probs[:, None]).astype(jnp.bfloat16),
            handle_token,
            tokens,
        )
        return combined, (routes, probs, dispatched, dispatched_probs)

    def differentiable_roundtrip_bwd(residuals, combined_cotangent):
        routes, probs, dispatched, dispatched_probs = residuals
        dispatched_cotangent, _, _, backward_handle = _dispatch(
            combined_cotangent,
            routes,
            probs,
            output_rows,
            num_experts // world_size,
        )
        dispatched_hidden_cotangent = (dispatched_cotangent.astype(jnp.float32) * dispatched_probs[:, None]).astype(
            jnp.bfloat16
        )
        dispatched_probability_cotangent = jnp.sum(
            dispatched_cotangent.astype(jnp.float32) * dispatched.astype(jnp.float32),
            axis=1,
        )
        hidden_cotangent, probability_cotangent = _combine_with_probabilities(
            dispatched_hidden_cotangent,
            dispatched_probability_cotangent,
            backward_handle,
            tokens,
            num_experts,
        )
        return hidden_cotangent, None, probability_cotangent

    differentiable_roundtrip.defvjp(differentiable_roundtrip_fwd, differentiable_roundtrip_bwd)

    _trace("gradient roundtrip start")
    hidden_gradient, probability_gradient = jax.jit(
        jax.grad(
            lambda x, probs: jnp.sum(differentiable_roundtrip(x, routing_map_jax, probs).astype(jnp.float32)),
            argnums=(0, 1),
        )
    )(hidden, probabilities_jax)
    _trace("gradient roundtrip done")
    np.testing.assert_allclose(
        np.asarray(hidden_gradient, dtype=np.float32),
        np.broadcast_to(kept_per_token[:, None], hidden.shape),
        rtol=2e-2,
        atol=2e-2,
    )
    expected_probability_gradient = np.zeros_like(probabilities)
    expected_probability_gradient[token_indices, selected] = np.where(
        assignment_keep,
        np.sum(np.asarray(hidden, dtype=np.float32), axis=1)[:, None],
        0,
    )
    np.testing.assert_allclose(
        np.asarray(probability_gradient),
        expected_probability_gradient,
        rtol=2e-2,
        atol=2e-2,
    )

    def separated_roundtrip(x, probs):
        dispatched, dispatched_probs, _, handle_token = hybridep_dispatch(
            x,
            routing_map_jax,
            probs,
            output_rows,
            num_experts // world_size,
        )
        weighted = (dispatched.astype(jnp.float32) * dispatched_probs[:, None]).astype(jnp.bfloat16)
        return hybridep_combine(
            weighted,
            routing_map_jax,
            probs,
            handle_token,
            num_experts // world_size,
        )

    _trace("separated custom VJP start")
    separated_hidden_gradient, separated_probability_gradient = jax.jit(
        jax.grad(
            lambda x, probs: jnp.sum(separated_roundtrip(x, probs).astype(jnp.float32)),
            argnums=(0, 1),
        )
    )(hidden, probabilities_jax)
    _trace("separated custom VJP done")
    np.testing.assert_allclose(
        np.asarray(separated_hidden_gradient, dtype=np.float32),
        np.broadcast_to(kept_per_token[:, None], hidden.shape),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(separated_probability_gradient),
        expected_probability_gradient,
        rtol=2e-2,
        atol=2e-2,
    )
    if rank == 0:
        print(
            "HYBRID_EP_JAX_FFI_PASS "
            f"ranks={world_size} tokens={tokens} hidden={hidden_dim} experts={num_experts} "
            f"topk={topk} drop_every={drop_every}",
            flush=True,
        )
    dist.barrier()
    library.levanter_hybridep_shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
