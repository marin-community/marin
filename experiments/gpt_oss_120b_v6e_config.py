# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-aware GPT-OSS 120B v6e tensor-parallel defaults.

The v6e slice families span from sub-host allocations to large multi-host
meshes. For the current GPT-OSS 120B vLLM path, the pragmatic default is to
keep tensor parallelism within one 8-chip host:

- v6e-4  -> tp=4
- v6e-8  -> tp=8
- v6e-16 and larger -> tp=8

Why cap at 8:
- one v6e host tops out at 8 chips
- GPT-OSS 120B exposes 8 KV heads, so tp>8 does not buy cleaner KV-head
  sharding
- larger TP values would force multi-host collectives in the current local
  vLLM setup before we have a separate data-parallel strategy
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.gpt_oss_120b_tpu import gpt_oss_120b_tpu_vllm_config
from marin.alignment.inference_config import VLLMConfig

GPT_OSS_120B_NUM_ATTENTION_HEADS = 64
GPT_OSS_120B_NUM_KV_HEADS = 8
V6E_CHIPS_PER_HOST = 8


@dataclass(frozen=True)
class V6ESliceShape:
    tpu_type: str
    topology: str
    chip_count: int
    host_layout: str
    vm_count: int
    scope: str


@dataclass(frozen=True)
class GptOss120BV6EStrategy:
    slice_shape: V6ESliceShape
    recommended_tensor_parallel_size: int
    rationale: str


_V6E_SLICE_SHAPES: dict[str, V6ESliceShape] = {
    "v6e-1": V6ESliceShape(
        tpu_type="v6e-1",
        topology="1x1",
        chip_count=1,
        host_layout="1/8 host",
        vm_count=1,
        scope="sub-host",
    ),
    "v6e-4": V6ESliceShape(
        tpu_type="v6e-4",
        topology="2x2",
        chip_count=4,
        host_layout="1/2 host",
        vm_count=1,
        scope="sub-host",
    ),
    "v6e-8": V6ESliceShape(
        tpu_type="v6e-8",
        topology="2x4",
        chip_count=8,
        host_layout="1 host",
        vm_count=1,
        scope="single-host",
    ),
    "v6e-16": V6ESliceShape(
        tpu_type="v6e-16",
        topology="4x4",
        chip_count=16,
        host_layout="2 hosts",
        vm_count=4,
        scope="multi-host",
    ),
    "v6e-32": V6ESliceShape(
        tpu_type="v6e-32",
        topology="4x8",
        chip_count=32,
        host_layout="4 hosts",
        vm_count=8,
        scope="multi-host",
    ),
    "v6e-64": V6ESliceShape(
        tpu_type="v6e-64",
        topology="8x8",
        chip_count=64,
        host_layout="8 hosts",
        vm_count=16,
        scope="multi-host",
    ),
    "v6e-128": V6ESliceShape(
        tpu_type="v6e-128",
        topology="8x16",
        chip_count=128,
        host_layout="16 hosts",
        vm_count=32,
        scope="multi-host",
    ),
    "v6e-256": V6ESliceShape(
        tpu_type="v6e-256",
        topology="16x16",
        chip_count=256,
        host_layout="32 hosts",
        vm_count=64,
        scope="multi-host",
    ),
}


def _v6e_slice_shape(tpu_type: str) -> V6ESliceShape:
    try:
        return _V6E_SLICE_SHAPES[tpu_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_V6E_SLICE_SHAPES))
        raise ValueError(f"Unsupported v6e TPU type {tpu_type!r}. Supported values: {supported}") from exc


def gpt_oss_120b_v6e_strategy(tpu_type: str) -> GptOss120BV6EStrategy:
    slice_shape = _v6e_slice_shape(tpu_type)
    recommended_tp = min(slice_shape.chip_count, V6E_CHIPS_PER_HOST, GPT_OSS_120B_NUM_KV_HEADS)
    rationale = (
        f"{tpu_type} spans {slice_shape.host_layout} ({slice_shape.scope}); "
        f"use host-local tp={recommended_tp} for GPT-OSS 120B because one v6e host tops out at "
        f"{V6E_CHIPS_PER_HOST} chips and the model has {GPT_OSS_120B_NUM_KV_HEADS} KV heads."
    )
    return GptOss120BV6EStrategy(
        slice_shape=slice_shape,
        recommended_tensor_parallel_size=recommended_tp,
        rationale=rationale,
    )


def gpt_oss_120b_v6e_tensor_parallel_size(tpu_type: str) -> int:
    return gpt_oss_120b_v6e_strategy(tpu_type).recommended_tensor_parallel_size


def gpt_oss_120b_v6e_vllm_config(
    *,
    tpu_type: str,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    cpu: int = 32,
    disk: str = "80g",
    ram: str = "256g",
    model_impl_type: str = "vllm",
    prefer_jax_for_bootstrap: bool = False,
) -> VLLMConfig:
    resolved_tp = tensor_parallel_size
    if resolved_tp is None:
        resolved_tp = gpt_oss_120b_v6e_tensor_parallel_size(tpu_type)
    if resolved_tp > _v6e_slice_shape(tpu_type).chip_count:
        raise ValueError(f"tensor_parallel_size={resolved_tp} exceeds {tpu_type} chip count")
    if GPT_OSS_120B_NUM_ATTENTION_HEADS % resolved_tp != 0:
        raise ValueError(
            f"tensor_parallel_size={resolved_tp} does not divide GPT-OSS 120B attention heads "
            f"({GPT_OSS_120B_NUM_ATTENTION_HEADS})"
        )
    return gpt_oss_120b_tpu_vllm_config(
        tpu_type=tpu_type,
        tensor_parallel_size=resolved_tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        cpu=cpu,
        disk=disk,
        ram=ram,
        model_impl_type=model_impl_type,
        prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
    )
