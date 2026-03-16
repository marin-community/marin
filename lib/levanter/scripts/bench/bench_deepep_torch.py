# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark DeepEP and Hybrid-EP torch kernels on GPU.

This script is intentionally torch-side. It is meant for bring-up and benchmarking
of DeepEP / Hybrid-EP dispatch-combine kernels before attempting any JAX bridge.
Run it under `torchrun --nproc_per_node=<gpus> ...` after installing the
DeepEP `hybrid-ep` branch into the job environment.
"""

from __future__ import annotations

import argparse
import inspect
import os
import time
from collections.abc import Callable
from typing import Literal

import numpy as np

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    torch = None
    dist = None

Distribution = Literal["random", "runs"]
Kernel = Literal["deep_ep", "hybrid_ep", "hybrid_ep_permute"]


def _require_deep_ep():
    try:
        import deep_ep
    except ImportError as exc:
        raise ImportError(
            "deep_ep is not installed. Install the DeepEP `hybrid-ep` branch in the job environment first."
        ) from exc
    return deep_ep


def _require_torch() -> None:
    if torch is None or dist is None:
        raise ImportError(
            "torch is not installed. Run this benchmark with the GPU dependency set, "
            "for example `uv run --extra gpu ...`, or inside the GPU job image."
        )


def _print0(*args, **kwargs) -> None:
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def _init_dist() -> tuple[int, int]:
    _require_torch()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        kwargs: dict[str, object] = {"backend": "nccl"}
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**kwargs)
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _sync_max_time(local_elapsed: float) -> float:
    elapsed = torch.tensor([local_elapsed], device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def _time_fn(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    dist.barrier()
    return _sync_max_time(elapsed)


def _sample_router_logits(
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
        return rng.normal(size=(tokens, experts)).astype(np.float32)

    if distribution == "runs":
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

    raise ValueError(f"Unknown distribution: {distribution}")


def _route_topk(router_logits: torch.Tensor, *, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    topk_logits, topk_idx = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    return topk_idx.to(torch.int64), topk_weights.to(torch.float32)


def _routing_map(topk_idx: torch.Tensor, *, num_experts: int) -> torch.Tensor:
    mapping = torch.zeros(topk_idx.shape[0], num_experts, device=topk_idx.device, dtype=torch.bool)
    ones = torch.ones_like(topk_idx, dtype=torch.bool)
    return mapping.scatter(1, topk_idx, ones)


def _make_hidden(*, tokens: int, hidden: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(tokens, hidden, generator=generator, device="cuda", dtype=torch.bfloat16)


def _make_deepep_buffer(deep_ep, *, group, hidden: int, world_size: int, num_sms: int | None):
    if num_sms is not None:
        deep_ep.Buffer.set_num_sms(num_sms)

    hidden_bytes = hidden * 2
    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (deep_ep.Buffer.get_dispatch_config(world_size), deep_ep.Buffer.get_combine_config(world_size)):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, world_size), num_nvl_bytes)
        try:
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, world_size), num_rdma_bytes)
        except RuntimeError as exc:
            # An intranode H100x8 build can still use the NVLink path even when
            # DeepEP was compiled without NVSHMEM internode support.
            if world_size <= 8 and "NVSHMEM is disable during compilation" in str(exc):
                num_rdma_bytes = max(0, num_rdma_bytes)
            else:
                raise
    return deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)


def _bench_deep_ep(
    deep_ep,
    *,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    warmup: int,
    iters: int,
    num_sms: int | None,
) -> float:
    group = dist.group.WORLD
    buffer = _make_deepep_buffer(
        deep_ep, group=group, hidden=x.shape[1], world_size=dist.get_world_size(), num_sms=num_sms
    )
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
        buffer.get_dispatch_layout(topk_idx, num_experts)
    )
    _, _, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    def step() -> None:
        recv_x, _, recv_topk_weights, _, _, _ = buffer.dispatch(x, handle=handle)
        buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)

    return _time_fn(step, warmup=warmup, iters=iters)


def _bench_hybrid_ep(
    deep_ep,
    *,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    warmup: int,
    iters: int,
    pad_multiple: int,
    num_sms_dispatch: int | None,
    num_sms_combine: int | None,
    num_sms_preprocess: int | None,
    kernel: Kernel,
) -> float:
    buffer = deep_ep.HybridEPBuffer(
        group=dist.group.WORLD,
        hidden_dim=x.shape[1],
        max_num_of_tokens_per_rank=x.shape[0],
        num_local_experts=num_local_experts,
        use_fp8=False,
        num_sms_dispatch_api=num_sms_dispatch,
        num_sms_combine_api=num_sms_combine,
        num_sms_preprocessing_api=num_sms_preprocess,
    )

    if kernel == "hybrid_ep":
        _, _, _, handle = buffer.dispatch(
            hidden=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_of_experts=num_experts,
        )

        def step() -> None:
            dispatched_hidden, dispatched_probs, _, _ = buffer.dispatch(
                hidden=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_of_experts=num_experts,
                handle=handle,
            )
            buffer.combine(hidden=dispatched_hidden.to(torch.bfloat16), probs=dispatched_probs, handle=handle)

        return _time_fn(step, warmup=warmup, iters=iters)

    if kernel == "hybrid_ep_permute":
        _, _, _, tokens_per_expert, handle = buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            pad_multiple=pad_multiple,
        )
        num_permuted_tokens = int(tokens_per_expert.sum().item())

        def step() -> None:
            dispatched_hidden, dispatched_probs, _, _, _ = buffer.dispatch_with_permute(
                hidden=x,
                routing_map=routing_map,
                probs=probs,
                pad_multiple=pad_multiple,
                handle=handle,
                num_permuted_tokens=num_permuted_tokens,
            )
            buffer.combine_with_unpermute(
                hidden=dispatched_hidden.to(torch.bfloat16),
                probs=dispatched_probs,
                handle=handle,
                pad_multiple=pad_multiple,
            )

        return _time_fn(step, warmup=warmup, iters=iters)

    raise ValueError(f"Unsupported kernel: {kernel}")


def _routing_probs(topk_idx: torch.Tensor, topk_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    probs = torch.zeros(topk_idx.shape[0], num_experts, device=topk_idx.device, dtype=torch.float32)
    return probs.scatter(1, topk_idx, topk_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepEP / Hybrid-EP torch kernels.")
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-local-experts", type=int, default=16)
    parser.add_argument("--distribution", choices=["random", "runs"], default="random")
    parser.add_argument("--kernel", choices=["deep_ep", "hybrid_ep", "hybrid_ep_permute", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-alpha", type=float, default=0.97)
    parser.add_argument("--run-noise-scale", type=float, default=0.2)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-sms-dispatch", type=int, default=None)
    parser.add_argument("--num-sms-combine", type=int, default=None)
    parser.add_argument("--num-sms-preprocess", type=int, default=None)
    args = parser.parse_args()

    deep_ep = _require_deep_ep()
    _, world_size = _init_dist()

    try:
        if world_size < 2:
            raise ValueError("DeepEP benchmarking requires at least 2 ranks.")

        num_experts = world_size * args.num_local_experts
        rank_seed = args.seed + dist.get_rank()
        logits_np = _sample_router_logits(
            seed=rank_seed,
            tokens=args.tokens_per_rank,
            experts=num_experts,
            distribution=args.distribution,
            run_alpha=args.run_alpha,
            run_noise_scale=args.run_noise_scale,
        )
        router_logits = torch.as_tensor(logits_np, device="cuda", dtype=torch.float32)
        topk_idx, topk_weights = _route_topk(router_logits, topk=args.topk)
        routing_map = _routing_map(topk_idx, num_experts=num_experts)
        probs = _routing_probs(topk_idx, topk_weights, num_experts)
        x = _make_hidden(tokens=args.tokens_per_rank, hidden=args.hidden, seed=rank_seed)

        kernels: list[Kernel]
        if args.kernel == "all":
            kernels = ["deep_ep", "hybrid_ep", "hybrid_ep_permute"]
        else:
            kernels = [args.kernel]

        _print0(
            f"shape tokens_per_rank={args.tokens_per_rank} hidden={args.hidden} "
            f"global_experts={num_experts} local_experts={args.num_local_experts} "
            f"topk={args.topk} distribution={args.distribution} world_size={world_size}"
        )

        for kernel in kernels:
            if kernel == "deep_ep":
                time_s = _bench_deep_ep(
                    deep_ep,
                    x=x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_experts=num_experts,
                    warmup=args.warmup,
                    iters=args.iters,
                    num_sms=args.num_sms,
                )
            else:
                time_s = _bench_hybrid_ep(
                    deep_ep,
                    x=x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    routing_map=routing_map,
                    probs=probs,
                    num_experts=num_experts,
                    num_local_experts=args.num_local_experts,
                    warmup=args.warmup,
                    iters=args.iters,
                    pad_multiple=args.pad_multiple,
                    num_sms_dispatch=args.num_sms_dispatch,
                    num_sms_combine=args.num_sms_combine,
                    num_sms_preprocess=args.num_sms_preprocess,
                    kernel=kernel,
                )

            global_tokens = args.tokens_per_rank * world_size
            tokens_per_s = global_tokens / time_s
            _print0(
                f"RESULT kernel={kernel} ep={world_size} distribution={args.distribution} "
                f"topk={args.topk} pass=dispatch_combine time_s={time_s:.6f} tokens_per_s={tokens_per_s:.2f}"
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
