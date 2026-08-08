# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded right-resource-reuse SM90 program from Relation/Fold IR.

The emitted CUDA kernel is intentionally a small structural prototype rather
than a performance replacement for the query-major CuTe skeleton.  One CTA
stages a right-side K/V block in shared memory, reuses it for a bounded group
of relation-left query blocks, and writes the mergeable normalized-exponential
state directly back to its source query.  Selected-slot kernel boundaries
preserve source order and make every state write single-owner without atomics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from tile_lifetime.h100_streaming_lowering import LoweredScoreMap, lower_h100_streaming_program
from tile_lifetime.routed_attention_plan import (
    BoundedKVReusePlan,
    RoutedAttentionOrientation,
    RoutedAttentionPhysicalPlan,
    RoutedStreamingAttentionCompilation,
    bounded_kv_reuse_plan,
    compile_bounded_kv_major_candidate,
)


@dataclass(frozen=True)
class DeviceKVReuseWave:
    """Device task arrays for one selected-slot wave."""

    selected_slot: int
    key_value_block: torch.Tensor
    query_block: torch.Tensor
    query_count: torch.Tensor

    @property
    def task_count(self) -> int:
        """Number of right-resource tasks in the wave."""
        return int(self.key_value_block.numel())


@dataclass
class CompiledH100BoundedKVReuseProgram:
    """Generated CUDA executable and persistent query-state buffers."""

    compilation: RoutedStreamingAttentionCompilation
    physical_plan: RoutedAttentionPhysicalPlan
    reuse_plan: BoundedKVReusePlan
    score_map: LoweredScoreMap
    generated_source: str
    generated_source_sha256: str
    extension: Any
    waves: tuple[DeviceKVReuseWave, ...]
    row_max: torch.Tensor
    row_sum_exp: torch.Tensor
    weighted_value: torch.Tensor

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Execute deterministic selected-slot waves and finalize BF16 output."""
        _validate_runtime_tensor("query", query, self.compilation.program.qk.inputs[0].shape)
        _validate_runtime_tensor("key", key, self.compilation.program.qk.inputs[1].shape)
        _validate_runtime_tensor("value", value, self.compilation.program.pv.inputs[1].shape)
        _validate_runtime_tensor("output", output, self.compilation.program.finalize.output.shape)
        self.row_max.fill_(-torch.inf)
        self.row_sum_exp.zero_()
        self.weighted_value.zero_()
        for wave in self.waves:
            if wave.task_count == 0:
                continue
            self.extension.run_contract_fold_wave(
                query,
                key,
                value,
                wave.key_value_block,
                wave.query_block,
                wave.query_count,
                self.row_max,
                self.row_sum_exp,
                self.weighted_value,
                self.score_map.scale,
                self.physical_plan.config.query_block_size,
            )
        self.extension.finalize_normalized_fold(self.row_sum_exp, self.weighted_value, output)


def compile_h100_bounded_kv_reuse_program(
    compilation: RoutedStreamingAttentionCompilation,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_capacity_per_task: int = 2,
) -> CompiledH100BoundedKVReuseProgram:
    """Generate and compile one bounded KV-major physical program."""
    if not torch.cuda.is_available():
        raise RuntimeError("the bounded H100 KV-reuse emitter requires CUDA")
    for name, tensor, shape in (
        ("query", query, compilation.program.qk.inputs[0].shape),
        ("key", key, compilation.program.qk.inputs[1].shape),
        ("value", value, compilation.program.pv.inputs[1].shape),
        ("output", output, compilation.program.finalize.output.shape),
    ):
        _validate_runtime_tensor(name, tensor, shape)
    if query.shape[-1] != value.shape[-1]:
        raise ValueError("the first bounded KV-reuse skeleton requires equal Q/K and V feature dimensions")

    lowering = lower_h100_streaming_program(compilation.program)
    candidate = compile_bounded_kv_major_candidate(
        compilation.relation,
        compilation.candidates[0].config,
        query_capacity_per_task=query_capacity_per_task,
    )
    if candidate.orientation is not RoutedAttentionOrientation.KV_MAJOR_SLOT_WAVES:
        raise ValueError("bounded right-resource lowering produced the wrong relation orientation")
    reuse_plan = bounded_kv_reuse_plan(
        compilation.relation,
        query_capacity_per_task=query_capacity_per_task,
    )
    source = _generated_cuda_source(lowering.score_map)
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    extension = load_inline(
        name=f"shuttle_kv_reuse_{source_sha256[:16]}",
        cpp_sources="",
        cuda_sources=source,
        functions=None,
        extra_cflags=("-O3",),
        extra_cuda_cflags=("-O3", "--use_fast_math", "-lineinfo"),
        with_cuda=True,
        verbose=False,
    )
    waves = tuple(
        DeviceKVReuseWave(
            selected_slot=wave.selected_slot,
            key_value_block=torch.as_tensor(wave.key_value_block, device=query.device),
            query_block=torch.as_tensor(wave.query_block, device=query.device),
            query_count=torch.as_tensor(wave.query_count, device=query.device),
        )
        for wave in reuse_plan.waves
    )
    sequence = query.shape[1]
    query_heads = query.shape[2]
    value_dimension = value.shape[3]
    return CompiledH100BoundedKVReuseProgram(
        compilation=compilation,
        physical_plan=candidate,
        reuse_plan=reuse_plan,
        score_map=lowering.score_map,
        generated_source=source,
        generated_source_sha256=source_sha256,
        extension=extension,
        waves=waves,
        row_max=torch.empty((sequence, query_heads), dtype=torch.float32, device=query.device),
        row_sum_exp=torch.empty((sequence, query_heads), dtype=torch.float32, device=query.device),
        weighted_value=torch.empty(
            (sequence, query_heads, value_dimension),
            dtype=torch.float32,
            device=query.device,
        ),
    )


def _generated_cuda_source(score_map: LoweredScoreMap) -> str:
    score_transform = "score *= scale;"
    if score_map.softcap is not None:
        score_transform += f" score = {score_map.softcap:.17g}f * tanhf(score / {score_map.softcap:.17g}f);"
    domain_predicate = "key_token <= query_token" if score_map.causal else "true"
    return f"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {{

__device__ __forceinline__ float warp_sum(float value) {{
  for (int offset = 16; offset > 0; offset /= 2) {{
    value += __shfl_down_sync(0xffffffff, value, offset);
  }}
  return __shfl_sync(0xffffffff, value, 0);
}}

__global__ void contract_fold_wave_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const int* key_value_block,
    const int* query_block,
    const int* query_count,
    float* row_max,
    float* row_sum_exp,
    float* weighted_value,
    int task_count,
    int query_capacity,
    int sequence,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    int block_size,
    float scale) {{
  const int task = blockIdx.x;
  const int query_head = blockIdx.y;
  if (task >= task_count || query_head >= query_heads) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = blockDim.x >> 5;
  const int kv_head = query_head / (query_heads / key_value_heads);
  const int kv_block = key_value_block[task];
  const int tile_elements = block_size * head_dimension;

  extern __shared__ __nv_bfloat16 staged[];
  __nv_bfloat16* staged_key = staged;
  __nv_bfloat16* staged_value = staged + tile_elements;
  for (int index = threadIdx.x; index < tile_elements; index += blockDim.x) {{
    const int row = index / head_dimension;
    const int feature = index - row * head_dimension;
    const int token = kv_block * block_size + row;
    const int source = (token * key_value_heads + kv_head) * head_dimension + feature;
    staged_key[index] = key[source];
    staged_value[index] = value[source];
  }}
  __syncthreads();

  const int consumers = query_count[task];
  for (int consumer = 0; consumer < consumers; ++consumer) {{
    const int q_block = query_block[task * query_capacity + consumer];
    for (int row = warp; row < block_size; row += warp_count) {{
      const int query_token = q_block * block_size + row;
      if (query_token >= sequence) continue;
      float q_fragment[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      float output_fragment[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      #pragma unroll
      for (int item = 0; item < 4; ++item) {{
        const int feature = lane + item * 32;
        if (feature < head_dimension) {{
          const int q_index = (query_token * query_heads + query_head) * head_dimension + feature;
          q_fragment[item] = __bfloat162float(query[q_index]);
          output_fragment[item] = weighted_value[q_index];
        }}
      }}
      const int state_index = query_token * query_heads + query_head;
      float maximum = row_max[state_index];
      float denominator = row_sum_exp[state_index];
      for (int key_row = 0; key_row < block_size; ++key_row) {{
        const int key_token = kv_block * block_size + key_row;
        if (key_token >= sequence || !({domain_predicate})) continue;
        float dot = 0.0f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {{
          const int feature = lane + item * 32;
          if (feature < head_dimension) {{
            dot += q_fragment[item] * __bfloat162float(staged_key[key_row * head_dimension + feature]);
          }}
        }}
        float score = warp_sum(dot);
        {score_transform}
        const float new_maximum = fmaxf(maximum, score);
        const float old_scale = denominator > 0.0f ? expf(maximum - new_maximum) : 0.0f;
        const float probability = expf(score - new_maximum);
        denominator = denominator * old_scale + probability;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {{
          const int feature = lane + item * 32;
          if (feature < head_dimension) {{
            const float v = __bfloat162float(staged_value[key_row * head_dimension + feature]);
            output_fragment[item] = output_fragment[item] * old_scale + probability * v;
          }}
        }}
        maximum = new_maximum;
      }}
      if (lane == 0) {{
        row_max[state_index] = maximum;
        row_sum_exp[state_index] = denominator;
      }}
      #pragma unroll
      for (int item = 0; item < 4; ++item) {{
        const int feature = lane + item * 32;
        if (feature < head_dimension) {{
          const int output_index = (query_token * query_heads + query_head) * head_dimension + feature;
          weighted_value[output_index] = output_fragment[item];
        }}
      }}
    }}
  }}
}}

__global__ void finalize_normalized_fold_kernel(
    const float* row_sum_exp,
    const float* weighted_value,
    __nv_bfloat16* output,
    int element_count,
    int head_dimension) {{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < element_count;
       index += blockDim.x * gridDim.x) {{
    const int row_head = index / head_dimension;
    output[index] = __float2bfloat16(weighted_value[index] / row_sum_exp[row_head]);
  }}
}}

}}  // namespace

void run_contract_fold_wave(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_value_block,
    torch::Tensor query_block,
    torch::Tensor query_count,
    torch::Tensor row_max,
    torch::Tensor row_sum_exp,
    torch::Tensor weighted_value,
    double scale,
    int64_t block_size) {{
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(), "Q/K/V must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16, "Q must be BF16");
  TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16, "K/V must be BF16");
  TORCH_CHECK(key_value_block.scalar_type() == at::kInt, "KV task indices must be int32");
  TORCH_CHECK(query_block.scalar_type() == at::kInt && query_count.scalar_type() == at::kInt,
              "query task indices must be int32");
  const int task_count = key_value_block.numel();
  const int query_capacity = query_block.size(1);
  const int sequence = query.size(1);
  const int query_heads = query.size(2);
  const int key_value_heads = key.size(2);
  const int head_dimension = query.size(3);
  const int shared_bytes = 2 * block_size * head_dimension * sizeof(__nv_bfloat16);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      contract_fold_wave_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes));
  const dim3 grid(task_count, query_heads);
  const int threads = 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  contract_fold_wave_kernel<<<grid, threads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
      key_value_block.data_ptr<int>(),
      query_block.data_ptr<int>(),
      query_count.data_ptr<int>(),
      row_max.data_ptr<float>(),
      row_sum_exp.data_ptr<float>(),
      weighted_value.data_ptr<float>(),
      task_count,
      query_capacity,
      sequence,
      query_heads,
      key_value_heads,
      head_dimension,
      block_size,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

void finalize_normalized_fold(torch::Tensor row_sum_exp, torch::Tensor weighted_value, torch::Tensor output) {{
  const int64_t element_count = output.numel();
  const int head_dimension = output.size(3);
  const int threads = 256;
  const int blocks = std::min<int64_t>((element_count + threads - 1) / threads, 65535);
  auto stream = at::cuda::getCurrentCUDAStream();
  finalize_normalized_fold_kernel<<<blocks, threads, 0, stream>>>(
      row_sum_exp.data_ptr<float>(),
      weighted_value.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      element_count,
      head_dimension);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("run_contract_fold_wave", &run_contract_fold_wave);
  module.def("finalize_normalized_fold", &finalize_normalized_fold);
}}
"""


def _validate_runtime_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError(f"{name} must be BF16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA-resident")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must use contiguous BSHD storage")
