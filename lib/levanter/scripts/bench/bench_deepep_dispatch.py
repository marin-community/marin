# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark DeepEP dispatch/combine kernels on the sealed MoE comparison shape.

This script is intentionally standalone and Torch-native. It is for a research
thread where the questions are:

1. Can this repo call DeepEP's Torch-targeted MoE kernels at all?
2. If so, how do intranode dispatch/combine timings look on the same fixed
   routed-token regime used in the sealed H100x8 `ragged_a2a` experiment?

`--input-source=jax` exercises a JAX -> Torch DLPack bridge before the DeepEP
call so the interop cost is explicit.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.utils import dlpack as torch_dlpack

Distribution = Literal["random", "runs"]
InputSource = Literal["torch", "jax"]

DEFAULT_NUM_NVL_BYTES = 256 * 1024 * 1024
DEFAULT_NUM_RDMA_BYTES = 0


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    device: torch.device

    @property
    def hybrid_ep(self) -> bool:
        return self.world_size > self.local_world_size


@dataclass(frozen=True)
class TorchBatch:
    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class LayoutCache:
    num_tokens_per_rank: torch.Tensor
    num_tokens_per_rdma_rank: torch.Tensor | None
    num_tokens_per_expert: torch.Tensor
    is_token_in_rank: torch.Tensor


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _dist_env() -> DistEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepEP benchmarking.")

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", torch.cuda.device_count())
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if world_size <= 1:
        raise RuntimeError("Run this script under torchrun with at least 2 ranks.")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        device=device,
    )


def _print0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def _parse_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"f16", "float16", "half"}:
        return torch.float16
    if lowered in {"f32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted, dtype=np.float32)
    denom = np.sum(exp_x, axis=-1, keepdims=True, dtype=np.float32)
    return exp_x / denom


def _sample_router_logits_numpy(
    *,
    seed: int,
    tokens: int,
    experts: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "random":
        return rng.normal(loc=0.0, scale=1.0, size=(tokens, experts)).astype(np.float32)

    if distribution != "runs":
        raise ValueError(f"Unknown distribution: {distribution}")

    mean_run = max(2.0, 1.0 / max(1e-6, 1.0 - run_alpha))
    p = min(0.9, max(0.01, 1.0 / mean_run))
    assigned = np.empty((tokens,), dtype=np.int32)
    loads = np.zeros((experts,), dtype=np.int32)
    prev_expert = -1
    pos = 0
    while pos < tokens:
        run_len = int(rng.geometric(p))
        run_len = min(run_len, tokens - pos)
        min_load = int(np.min(loads))
        candidates = np.flatnonzero(loads == min_load)
        if prev_expert in candidates and candidates.size > 1:
            candidates = candidates[candidates != prev_expert]
        expert = int(rng.choice(candidates))
        assigned[pos : pos + run_len] = expert
        loads[expert] += run_len
        prev_expert = expert
        pos += run_len

    logits = rng.normal(loc=0.0, scale=float(run_noise_scale), size=(tokens, experts)).astype(np.float32)
    logits[np.arange(tokens), assigned] += 6.0
    return logits


def _route_topk_numpy(router_logits: np.ndarray, *, topk: int) -> tuple[np.ndarray, np.ndarray]:
    topk_unsorted = np.argpartition(router_logits, kth=-topk, axis=-1)[:, -topk:]
    topk_logits = np.take_along_axis(router_logits, topk_unsorted, axis=-1)
    order = np.argsort(-topk_logits, axis=-1)
    topk_idx = np.take_along_axis(topk_unsorted, order, axis=-1).astype(np.int64)
    topk_logits = np.take_along_axis(topk_logits, order, axis=-1)
    topk_weights = _softmax_numpy(topk_logits).astype(np.float32)
    return topk_idx, topk_weights


def _global_arrays(
    *,
    seed: int,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(tokens, hidden)).astype(np.float32)
    if dtype == torch.float16:
        x = x.astype(np.float16)
    router_logits = _sample_router_logits_numpy(
        seed=seed + 1,
        tokens=tokens,
        experts=experts,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
    )
    topk_idx, topk_weights = _route_topk_numpy(router_logits, topk=topk)
    return x, topk_idx, topk_weights


def _local_slice(tokens: int, env: DistEnv) -> slice:
    if tokens % env.world_size != 0:
        raise ValueError(f"tokens={tokens} must be divisible by world_size={env.world_size}")
    per_rank = tokens // env.world_size
    start = env.rank * per_rank
    return slice(start, start + per_rank)


def _torch_batch(
    *,
    env: DistEnv,
    seed: int,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
    dtype: torch.dtype,
) -> TorchBatch:
    x_global, topk_idx_global, topk_weights_global = _global_arrays(
        seed=seed,
        tokens=tokens,
        hidden=hidden,
        experts=experts,
        topk=topk,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
        dtype=dtype,
    )
    local = _local_slice(tokens, env)
    x = torch.as_tensor(x_global[local], device=env.device, dtype=dtype)
    topk_idx = torch.as_tensor(topk_idx_global[local], device=env.device, dtype=torch.int64)
    topk_weights = torch.as_tensor(topk_weights_global[local], device=env.device, dtype=torch.float32)
    return TorchBatch(x=x, topk_idx=topk_idx, topk_weights=topk_weights)


def _jax_batch(
    *,
    env: DistEnv,
    seed: int,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
    dtype: torch.dtype,
) -> tuple[TorchBatch, float]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.dlpack as jax_dlpack
    import jax.numpy as jnp

    x_global, topk_idx_global, topk_weights_global = _global_arrays(
        seed=seed,
        tokens=tokens,
        hidden=hidden,
        experts=experts,
        topk=topk,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
        dtype=dtype,
    )
    local = _local_slice(tokens, env)
    device = next(d for d in jax.devices("cuda") if d.id == env.local_rank)

    jax_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float16 if dtype == torch.float16 else jnp.float32
    start = time.perf_counter()
    with jax.default_device(device):
        x = jax.device_put(jnp.asarray(x_global[local], dtype=jax_dtype), device)
        topk_idx = jax.device_put(jnp.asarray(topk_idx_global[local], dtype=jnp.int64), device)
        topk_weights = jax.device_put(jnp.asarray(topk_weights_global[local], dtype=jnp.float32), device)
    jax.block_until_ready(x)
    torch_x = torch_dlpack.from_dlpack(x).to(device=env.device)
    torch_topk_idx = torch_dlpack.from_dlpack(topk_idx).to(device=env.device, dtype=torch.int64)
    torch_topk_weights = torch_dlpack.from_dlpack(topk_weights).to(device=env.device, dtype=torch.float32)
    torch.cuda.synchronize(env.device)
    bridge_time = time.perf_counter() - start
    _ = jax_dlpack
    return TorchBatch(x=torch_x, topk_idx=torch_topk_idx, topk_weights=torch_topk_weights), bridge_time


def _load_deepep():
    try:
        import deep_ep
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DeepEP is not installed in the active environment. Install/build it on the target GPU image first."
        ) from exc
    if not hasattr(deep_ep, "Buffer"):
        raise RuntimeError("DeepEP import succeeded but `deep_ep.Buffer` is missing.")
    return deep_ep


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _time_torch(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)
    _barrier()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    _barrier()
    return (time.perf_counter() - start) / iters


def _layout_cache(buffer, batch: TorchBatch, *, experts: int) -> LayoutCache:
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
        buffer.get_dispatch_layout(batch.topk_idx, experts)
    )
    return LayoutCache(
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
    )


def _dispatch_and_combine(
    buffer,
    batch: TorchBatch,
    layout: LayoutCache,
    *,
    handle: tuple | None = None,
    async_finish: bool,
    allocate_on_comm_stream: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    if handle is None:
        recv_x, _, recv_topk_weights, _, dispatch_handle, _ = buffer.dispatch(
            batch.x,
            num_tokens_per_rank=layout.num_tokens_per_rank,
            num_tokens_per_rdma_rank=layout.num_tokens_per_rdma_rank,
            is_token_in_rank=layout.is_token_in_rank,
            num_tokens_per_expert=layout.num_tokens_per_expert,
            topk_idx=batch.topk_idx,
            topk_weights=batch.topk_weights,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    else:
        recv_x, _, recv_topk_weights, _, _, _ = buffer.dispatch(
            batch.x,
            handle=handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        dispatch_handle = handle
    combined_x, combined_topk_weights, _ = buffer.combine(
        recv_x,
        dispatch_handle,
        topk_weights=recv_topk_weights,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return recv_x, combined_x, combined_topk_weights, dispatch_handle


def _bridge_back_to_jax(tensor: torch.Tensor) -> float:
    import jax
    import jax.dlpack as jax_dlpack

    start = time.perf_counter()
    jax_array = jax_dlpack.from_dlpack(tensor)
    jax.block_until_ready(jax_array)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepEP dispatch/combine on the sealed MoE GPU shape.")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--distribution", choices=("random", "runs"), default="random")
    parser.add_argument("--run-alpha", type=float, default=0.97)
    parser.add_argument("--run-noise-scale", type=float, default=0.05)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--input-source", choices=("torch", "jax"), default="torch")
    parser.add_argument("--return-to-jax", action="store_true")
    parser.add_argument("--num-nvl-bytes", type=int, default=DEFAULT_NUM_NVL_BYTES)
    parser.add_argument("--num-rdma-bytes", type=int, default=DEFAULT_NUM_RDMA_BYTES)
    parser.add_argument("--async-finish", action="store_true")
    parser.add_argument("--allocate-on-comm-stream", action="store_true")
    args = parser.parse_args()

    env = _dist_env()
    dtype = _parse_dtype(args.dtype)
    if args.experts % env.world_size != 0:
        raise ValueError(f"experts={args.experts} must be divisible by world_size={env.world_size}")

    deep_ep = _load_deepep()
    buffer = deep_ep.Buffer(dist.group.WORLD, args.num_nvl_bytes, args.num_rdma_bytes)

    if args.input_source == "jax":
        batch, bridge_to_torch = _jax_batch(
            env=env,
            seed=args.seed,
            tokens=args.tokens,
            hidden=args.hidden,
            experts=args.experts,
            topk=args.topk,
            distribution=args.distribution,
            run_alpha=args.run_alpha,
            run_noise_scale=args.run_noise_scale,
            dtype=dtype,
        )
    else:
        batch = _torch_batch(
            env=env,
            seed=args.seed,
            tokens=args.tokens,
            hidden=args.hidden,
            experts=args.experts,
            topk=args.topk,
            distribution=args.distribution,
            run_alpha=args.run_alpha,
            run_noise_scale=args.run_noise_scale,
            dtype=dtype,
        )
        bridge_to_torch = 0.0

    layout = _layout_cache(buffer, batch, experts=args.experts)
    recv_x, combined_x, _, dispatch_handle = _dispatch_and_combine(
        buffer,
        batch,
        layout,
        async_finish=args.async_finish,
        allocate_on_comm_stream=args.allocate_on_comm_stream,
    )
    torch.cuda.synchronize(env.device)
    _ = recv_x, combined_x
    _barrier()

    layout_time = _time_torch(
        lambda: _layout_cache(buffer, batch, experts=args.experts),
        warmup=args.warmup,
        iters=args.iters,
        device=env.device,
    )

    cached_layout_time = _time_torch(
        lambda: _dispatch_and_combine(
            buffer,
            batch,
            layout,
            handle=dispatch_handle,
            async_finish=args.async_finish,
            allocate_on_comm_stream=args.allocate_on_comm_stream,
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=env.device,
    )

    full_time = _time_torch(
        lambda: _dispatch_and_combine(
            buffer,
            batch,
            _layout_cache(buffer, batch, experts=args.experts),
            async_finish=args.async_finish,
            allocate_on_comm_stream=args.allocate_on_comm_stream,
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=env.device,
    )

    bridge_to_jax = 0.0
    if args.return_to_jax:
        _, combined_x, _, _ = _dispatch_and_combine(
            buffer,
            batch,
            layout,
            handle=dispatch_handle,
            async_finish=args.async_finish,
            allocate_on_comm_stream=args.allocate_on_comm_stream,
        )
        bridge_to_jax = _bridge_back_to_jax(combined_x)

    mode = "hybrid" if env.hybrid_ep else "intranode"
    tokens_per_second = args.tokens / full_time
    _print0(f"devices_per_host={env.local_world_size} world_size={env.world_size} mode={mode}")
    _print0(
        "shape "
        f"tokens={args.tokens} hidden={args.hidden} experts={args.experts} topk={args.topk} "
        f"distribution={args.distribution} dtype={dtype} input_source={args.input_source}"
    )
    _print0(
        "RESULT "
        f"layout_s={layout_time:.6f} "
        f"dispatch_combine_cached_s={cached_layout_time:.6f} "
        f"dispatch_combine_full_s={full_time:.6f} "
        f"bridge_to_torch_s={bridge_to_torch:.6f} "
        f"bridge_to_jax_s={bridge_to_jax:.6f} "
        f"tokens_per_s={tokens_per_second:.2f}"
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
