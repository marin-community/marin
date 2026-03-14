#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Megatron-style Qwen-shaped MoE layer benchmark."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist

import megatron.core.parallel_state as parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP, HAVE_HYBRIDEP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import RandomSTE
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed

NVTE_ENV_VARS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    seq_length: int
    micro_batch_size: int
    hidden_size: int
    moe_ffn_hidden_size: int
    num_experts: int
    router_topk: int
    num_attention_heads: int
    layers: int

    @property
    def tokens_per_rank(self) -> int:
        return self.seq_length * self.micro_batch_size


@dataclass(frozen=True)
class DispatcherSpec:
    name: str
    token_dispatcher: str
    moe_flex_dispatcher_backend: str


DISPATCHERS: dict[str, DispatcherSpec] = {
    "alltoall": DispatcherSpec(
        name="alltoall",
        token_dispatcher="alltoall",
        moe_flex_dispatcher_backend="deepep",
    ),
    "deepep": DispatcherSpec(
        name="deepep",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="deepep",
    ),
    "hybridep": DispatcherSpec(
        name="hybridep",
        token_dispatcher="flex",
        moe_flex_dispatcher_backend="hybridep",
    ),
}


def _qwen_case_specs() -> dict[str, ModelSpec]:
    qwen30 = ModelSpec(
        name="qwen3_30b_a3b_anchor",
        seq_length=4096,
        micro_batch_size=1,
        hidden_size=2048,
        moe_ffn_hidden_size=768,
        num_experts=128,
        router_topk=8,
        num_attention_heads=32,
        layers=48,
    )
    qwen235 = ModelSpec(
        name="qwen3_235b_a22b_anchor",
        seq_length=4096,
        micro_batch_size=1,
        hidden_size=4096,
        moe_ffn_hidden_size=1536,
        num_experts=128,
        router_topk=8,
        num_attention_heads=64,
        layers=94,
    )
    marin3633 = ModelSpec(
        name="marin_3633_topk_8",
        seq_length=4096,
        micro_batch_size=1,
        hidden_size=2048,
        moe_ffn_hidden_size=768,
        num_experts=128,
        router_topk=8,
        num_attention_heads=32,
        layers=1,
    )
    cases = {
        qwen30.name: qwen30,
        qwen235.name: qwen235,
        marin3633.name: marin3633,
        "marin_3633_topk_2": replace(marin3633, name="marin_3633_topk_2", router_topk=2),
        "qwen3_batch_mb2": replace(qwen30, name="qwen3_batch_mb2", micro_batch_size=2),
        "qwen3_batch_mb4": replace(qwen30, name="qwen3_batch_mb4", micro_batch_size=4),
        "qwen3_hidden_3072": replace(qwen30, name="qwen3_hidden_3072", hidden_size=3072),
        "qwen3_hidden_4096": replace(qwen30, name="qwen3_hidden_4096", hidden_size=4096),
        "qwen3_expert_1152": replace(qwen30, name="qwen3_expert_1152", moe_ffn_hidden_size=1152),
        "qwen3_expert_1536": replace(qwen30, name="qwen3_expert_1536", moe_ffn_hidden_size=1536),
        "qwen3_experts_32": replace(qwen30, name="qwen3_experts_32", num_experts=32),
        "qwen3_experts_64": replace(qwen30, name="qwen3_experts_64", num_experts=64),
        "qwen3_topk_2": replace(qwen30, name="qwen3_topk_2", router_topk=2),
        "qwen3_topk_4": replace(qwen30, name="qwen3_topk_4", router_topk=4),
    }
    return cases


def _parse_csv_arg(raw: str, *, choices: set[str] | None = None) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if choices is not None:
        invalid = [value for value in values if value not in choices]
        if invalid:
            raise ValueError(f"Invalid values {invalid}; expected subset of {sorted(choices)}")
    return values


def _print0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def _clear_nvte_env() -> None:
    for key in NVTE_ENV_VARS:
        os.environ.pop(key, None)


def _init_dist() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        kwargs: dict[str, object] = {"backend": "nccl"}
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**kwargs)
    return local_rank, dist.get_world_size()


def _init_model_parallel(ep_size: int) -> None:
    _clear_nvte_env()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
    )


def _check_dispatcher(dispatcher: DispatcherSpec) -> None:
    if dispatcher.name == "deepep" and not HAVE_DEEP_EP:
        raise RuntimeError("Megatron reports HAVE_DEEP_EP=False")
    if dispatcher.name == "hybridep" and not HAVE_HYBRIDEP:
        raise RuntimeError("Megatron reports HAVE_HYBRIDEP=False")


def _build_transformer_config(spec: ModelSpec, dispatcher: DispatcherSpec) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=spec.hidden_size,
        moe_ffn_hidden_size=spec.moe_ffn_hidden_size,
        num_attention_heads=spec.num_attention_heads,
        num_moe_experts=spec.num_experts,
        moe_router_topk=spec.router_topk,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type=dispatcher.token_dispatcher,
        moe_flex_dispatcher_backend=dispatcher.moe_flex_dispatcher_backend,
        use_cpu_initialization=True,
        add_bias_linear=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=dist.get_world_size(),
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8=None,
        moe_permute_fusion=True,
        moe_router_fusion=True,
        moe_router_force_load_balancing=True,
    )


def _resolve_moe_submodules(spec: ModelSpec):
    return get_gpt_layer_with_transformer_engine_submodules(
        num_experts=spec.num_experts,
        moe_grouped_gemm=True,
    ).mlp.submodules


def _prepare_layer(spec: ModelSpec, dispatcher: DispatcherSpec) -> MoELayer:
    config = _build_transformer_config(spec, dispatcher)
    submodules = _resolve_moe_submodules(spec)
    layer = MoELayer(config=config, submodules=submodules).cuda().to(dtype=torch.bfloat16)
    layer.train()
    return layer


def _sync_max(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _dummy_gemm(size: int) -> None:
    with torch.cuda.nvtx.range("(dummy GEMM)"):
        dummy_tensor = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        torch.matmul(dummy_tensor, dummy_tensor)
        del dummy_tensor


def _benchmark_case(
    layer: MoELayer,
    spec: ModelSpec,
    *,
    warmup_iters: int,
    measure_iters: int,
    dummy_gemm_size: int,
    manual_gc: bool,
) -> dict[str, float | list[float]]:
    set_experimental_flag(True)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input_tensor = torch.randn(
        spec.seq_length,
        spec.micro_batch_size,
        spec.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    input_tensor.requires_grad_(True)

    if manual_gc:
        torch.cuda.empty_cache()
        gc.disable()
        gc.collect()

    forward_timings: list[float] = []
    backward_timings: list[float] = []
    max_allocated_bytes: list[float] = []

    total_iters = warmup_iters + measure_iters
    for iteration in range(total_iters):
        random_ste_generator = getattr(RandomSTE, "generator", None)
        if random_ste_generator is not None:
            random_ste_generator.manual_seed(random_ste_generator.initial_seed())
        dist.barrier()
        _dummy_gemm(dummy_gemm_size)
        input_tensor.grad = None
        layer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output, _ = layer(input_tensor)
        fwd_end.record()

        backward_grad = torch.randn_like(output)
        bwd_start.record()
        output.backward(backward_grad)
        bwd_end.record()

        torch.cuda.synchronize()
        if iteration >= warmup_iters:
            forward_timings.append(fwd_start.elapsed_time(fwd_end))
            backward_timings.append(bwd_start.elapsed_time(bwd_end))
            max_allocated_bytes.append(float(torch.cuda.max_memory_allocated()))

    if manual_gc:
        gc.collect()
        gc.enable()

    forward_ms = statistics.mean(forward_timings)
    backward_ms = statistics.mean(backward_timings)
    forward_std_ms = statistics.pstdev(forward_timings) if len(forward_timings) > 1 else 0.0
    backward_std_ms = statistics.pstdev(backward_timings) if len(backward_timings) > 1 else 0.0
    max_allocated = statistics.mean(max_allocated_bytes)

    return {
        "forward_ms": _sync_max(forward_ms),
        "backward_ms": _sync_max(backward_ms),
        "forward_std_ms": _sync_max(forward_std_ms),
        "backward_std_ms": _sync_max(backward_std_ms),
        "max_allocated_bytes": _sync_max(max_allocated),
        "forward_timings_ms": forward_timings,
        "backward_timings_ms": backward_timings,
    }


def _selected_cases(case_arg: str) -> list[ModelSpec]:
    specs = _qwen_case_specs()
    if case_arg == "all":
        return list(specs.values())
    names = _parse_csv_arg(case_arg, choices=set(specs))
    return [specs[name] for name in names]


def _selected_dispatchers(dispatcher_arg: str) -> list[DispatcherSpec]:
    if dispatcher_arg == "all":
        return list(DISPATCHERS.values())
    names = _parse_csv_arg(dispatcher_arg, choices=set(DISPATCHERS))
    return [DISPATCHERS[name] for name in names]


def _append_result(path: Path | None, record: dict[str, object]) -> None:
    if path is None or dist.get_rank() != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--dispatchers", default="all")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--dummy-gemm-size", type=int, default=8192)
    parser.add_argument("--manual-gc", action="store_true", default=True)
    parser.add_argument("--no-manual-gc", dest="manual_gc", action="store_false")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    local_rank, world_size = _init_dist()
    if world_size != 8:
        raise RuntimeError(f"Expected world_size=8 for the first H100x8 sweep, got {world_size}")

    _init_model_parallel(ep_size=world_size)
    _set_random_seed(seed_=123, data_parallel_random_init=False)

    _print0("DEVICE_NAME", torch.cuda.get_device_name(local_rank))
    _print0("WORLD_SIZE", world_size)
    _print0("HAVE_DEEP_EP", HAVE_DEEP_EP)
    _print0("HAVE_HYBRIDEP", HAVE_HYBRIDEP)

    try:
        for spec in _selected_cases(args.cases):
            for dispatcher in _selected_dispatchers(args.dispatchers):
                _check_dispatcher(dispatcher)
                dist.barrier()
                torch.cuda.empty_cache()
                gc.collect()
                layer = _prepare_layer(spec, dispatcher)
                _print0(
                    "CASE_START",
                    spec.name,
                    dispatcher.name,
                    f"tokens_per_rank={spec.tokens_per_rank}",
                    f"hidden={spec.hidden_size}",
                    f"moe_ffn_hidden={spec.moe_ffn_hidden_size}",
                    f"experts={spec.num_experts}",
                    f"topk={spec.router_topk}",
                )
                metrics = _benchmark_case(
                    layer,
                    spec,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                    dummy_gemm_size=args.dummy_gemm_size,
                    manual_gc=args.manual_gc,
                )
                record: dict[str, object] = {
                    "case_name": spec.name,
                    "dispatcher": dispatcher.name,
                    "world_size": world_size,
                    **asdict(spec),
                    **metrics,
                }
                _append_result(args.output_jsonl, record)
                if dist.get_rank() == 0:
                    print("RESULT", json.dumps(record, sort_keys=True), flush=True)
                del layer
                torch.cuda.empty_cache()
                gc.collect()
                dist.barrier()
    finally:
        dist.barrier()
        parallel_state.destroy_model_parallel()
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
