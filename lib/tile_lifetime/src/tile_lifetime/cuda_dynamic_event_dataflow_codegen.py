# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate bounded CUDA validation kernels from generic runtime event plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventTensorPlan,
    TaskFamily,
    event_tensor_runtime_inputs,
    phased_event_storage_binding,
    verify_event_dataflow_program,
    verify_event_tensor_plan,
)


class CudaDynamicEventLoweringError(ValueError):
    """A structured rejection from a bounded dynamic event lowering."""


@dataclass(frozen=True)
class CudaDynamicEventLowering:
    """Generated source and structural fingerprint for a dynamic event plan."""

    source: str
    source_sha256: str
    plan_fingerprint: str
    event_count: int
    source_count: int
    consumer_count: int


@dataclass(frozen=True)
class CudaPhasedPipelineLowering:
    """Generated source for a reusable phased Contract/Fold pipeline."""

    source: str
    source_sha256: str
    plan_fingerprint: str
    generation_count: int
    pipeline_depth: int


def generate_cuda_runtime_event_lowering(plan: EventTensorPlan) -> CudaDynamicEventLowering:
    """Generate a CTA counted-event kernel whose relation tables are runtime inputs."""
    verify_event_tensor_plan(plan)
    reasons: list[str] = []
    if plan.memory_scope is not EventMemoryScope.CTA:
        reasons.append("the first dynamic validation lowering requires CTA-scope readiness")
    if plan.generation_policy is not EventGenerationPolicy.PER_INVOCATION:
        reasons.append("the dynamic validation lowering accepts one fresh event generation")
    trigger_count = {coordinate: 0 for coordinate in plan.domain.coordinates}
    for pair in plan.trigger_relation.pairs:
        trigger_count[pair.source] += 1
    if any(count != 1 for count in trigger_count.values()):
        reasons.append("the dynamic validation lowering requires one consumer per event")
    notify_count_by_source: dict[tuple[int, ...], int] = {}
    for pair in plan.notify_relation.pairs:
        notify_count_by_source[pair.source] = notify_count_by_source.get(pair.source, 0) + 1
    if any(count != 1 for count in notify_count_by_source.values()):
        reasons.append("the dynamic validation lowering requires each producer to notify one event")
    if reasons:
        raise CudaDynamicEventLoweringError("; ".join(reasons))

    runtime = event_tensor_runtime_inputs(plan)
    payload = {
        "event_count": len(runtime.event_initial_counts),
        "source_count": len(runtime.source_event_offsets) - 1,
        "consumer_count": len(runtime.event_consumers),
        "scope": plan.memory_scope.value,
        "generation": plan.generation_policy.value,
        "runtime_interface": "counts/event_source_offsets/event_sources",
        "notify": [(pair.source, pair.target) for pair in plan.notify_relation.pairs],
        "trigger": [(pair.source, pair.target) for pair in plan.trigger_relation.pairs],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = _runtime_event_source()
    return CudaDynamicEventLowering(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        plan_fingerprint=fingerprint,
        event_count=len(runtime.event_initial_counts),
        source_count=len(runtime.source_event_offsets) - 1,
        consumer_count=len(runtime.event_consumers),
    )


def generate_cuda_phased_pipeline_lowering(program: EventDataflowProgram) -> CudaPhasedPipelineLowering:
    """Generate a multi-generation physical pipeline from four generic task stages."""
    verify_event_dataflow_program(program)
    if len(program.task_families) != 4 or len(program.dependences) != 4:
        raise CudaDynamicEventLoweringError("the bounded phased template requires four task stages and dependences")
    first, fold_update, second, finalize = program.task_families
    _validate_pipeline_domains(first, fold_update, second, finalize)
    _validate_pipeline_relations(program, first, fold_update, second, finalize)
    if any(plan.generation_policy is not EventGenerationPolicy.PHASED for plan in program.event_plans):
        raise CudaDynamicEventLoweringError("every pipeline readiness plan must use phased generation")

    generation_count = first.axes[0].extent
    pipeline_depth = first.axes[1].extent
    if pipeline_depth > 32:
        raise CudaDynamicEventLoweringError("the bounded phased CUDA template supports at most 32 slots")
    bindings = []
    for plan in program.event_plans:
        axis_names = tuple(axis.name for axis in plan.domain.axes)
        if "generation" not in axis_names:
            raise CudaDynamicEventLoweringError("every phased event domain must expose a generation axis")
        generation_axis = axis_names.index("generation")
        slot_axis = axis_names.index("pipeline_slot") if "pipeline_slot" in axis_names else None
        binding = phased_event_storage_binding(
            plan,
            slot=lambda coordinate, axis=slot_axis: coordinate[axis] if axis is not None else 0,
            generation=lambda coordinate, axis=generation_axis: coordinate[axis],
        )
        runtime = event_tensor_runtime_inputs(plan, storage_binding=binding)
        bindings.append(
            {
                "counts": runtime.event_initial_counts,
                "slots": runtime.event_storage_slots,
                "generations": runtime.event_generations,
            }
        )
    payload = {
        "generation_count": generation_count,
        "pipeline_depth": pipeline_depth,
        "bindings": bindings,
        "scope": EventMemoryScope.CTA.value,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = _phased_pipeline_source()
    return CudaPhasedPipelineLowering(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        plan_fingerprint=fingerprint,
        generation_count=generation_count,
        pipeline_depth=pipeline_depth,
    )


def _validate_pipeline_domains(
    first: TaskFamily,
    fold_update: TaskFamily,
    second: TaskFamily,
    finalize: TaskFamily,
) -> None:
    expected_axes = ("generation", "pipeline_slot")
    if tuple(axis.name for axis in first.axes) != expected_axes:
        raise CudaDynamicEventLoweringError("the first stage must use generation and pipeline-slot axes")
    if fold_update.axes != first.axes or second.axes != first.axes:
        raise CudaDynamicEventLoweringError("the three tiled stages must share one task domain")
    if tuple(axis.name for axis in finalize.axes) != ("generation",):
        raise CudaDynamicEventLoweringError("the finalizer must use the generation domain")
    if finalize.axes[0].extent != first.axes[0].extent:
        raise CudaDynamicEventLoweringError("pipeline generation extents must agree")


def _validate_pipeline_relations(
    program: EventDataflowProgram,
    first: TaskFamily,
    fold_update: TaskFamily,
    second: TaskFamily,
    finalize: TaskFamily,
) -> None:
    relation_by_endpoints = {
        (dependence.relation.source, dependence.relation.target): dependence.relation
        for dependence in program.dependences
    }
    required_endpoints = {
        (first, fold_update),
        (fold_update, second),
        (second, finalize),
        (finalize, first),
    }
    if set(relation_by_endpoints) != required_endpoints:
        raise CudaDynamicEventLoweringError("phased pipeline dependences do not match the four generic stages")
    generation_count = first.axes[0].extent
    pipeline_depth = first.axes[1].extent
    pointwise = {
        ((generation, slot), (generation, slot))
        for generation in range(generation_count)
        for slot in range(pipeline_depth)
    }
    if {(pair.source, pair.target) for pair in relation_by_endpoints[(first, fold_update)].pairs} != pointwise or {
        (pair.source, pair.target) for pair in relation_by_endpoints[(fold_update, second)].pairs
    } != pointwise:
        raise CudaDynamicEventLoweringError("tiled Contract/Fold stage dependences must be pointwise")
    expected_finalize = {
        ((generation, slot), (generation,)) for generation in range(generation_count) for slot in range(pipeline_depth)
    }
    if {(pair.source, pair.target) for pair in relation_by_endpoints[(second, finalize)].pairs} != expected_finalize:
        raise CudaDynamicEventLoweringError("final Fold must consume every second-Contract slot")
    expected_reuse = {
        ((generation - 1,), (generation, slot))
        for generation in range(1, generation_count)
        for slot in range(pipeline_depth)
    }
    if {(pair.source, pair.target) for pair in relation_by_endpoints[(finalize, first)].pairs} != expected_reuse:
        raise CudaDynamicEventLoweringError("slot reuse must follow prior-generation Fold finalization")


def _runtime_event_source() -> str:
    return r"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic EventTensorPlan runtime tables; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__global__ void shuttle_runtime_counted_event(
    const float* input,
    float* partials,
    float* output,
    const int* event_counts,
    const int* event_source_offsets,
    const int* event_sources,
    int event_count) {
  __shared__ int remaining;
  const int event_index = blockIdx.x;
  if (event_index >= event_count) return;
  const int producer_count = event_counts[event_index];
  const int source_begin = event_source_offsets[event_index];
  const int source_end = event_source_offsets[event_index + 1];
  if (producer_count < 0 || producer_count != source_end - source_begin || producer_count > blockDim.x) {
    if (threadIdx.x == 0) output[event_index] = NAN;
    return;
  }
  if (producer_count == 0) {
    if (threadIdx.x == 0) output[event_index] = 0.0f;
    return;
  }
  if (threadIdx.x == 0) remaining = producer_count;
  __syncthreads();
  if (threadIdx.x >= producer_count) return;

  const int source = event_sources[source_begin + threadIdx.x];
  partials[source] = input[source];
  __threadfence_block();
  const int prior_remaining = atomicSub(&remaining, 1);
  if (prior_remaining == 1) {
    float accumulator = 0.0f;
    for (int index = source_begin; index < source_end; ++index) {
      accumulator = __fadd_rn(accumulator, partials[event_sources[index]]);
    }
    output[event_index] = accumulator;
  }
}

void check_int32(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

}  // namespace

void run_runtime_counted_event_out(
    torch::Tensor input,
    torch::Tensor partials,
    torch::Tensor output,
    torch::Tensor event_counts,
    torch::Tensor event_source_offsets,
    torch::Tensor event_sources,
    int maximum_count) {
  TORCH_CHECK(input.is_cuda() && partials.is_cuda() && output.is_cuda(), "payload tensors must be CUDA tensors");
  TORCH_CHECK(input.is_contiguous() && partials.is_contiguous() && output.is_contiguous(),
              "payload tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 && partials.scalar_type() == torch::kFloat32 &&
              output.scalar_type() == torch::kFloat32, "payload tensors must be FP32");
  check_int32(event_counts, "event counts");
  check_int32(event_source_offsets, "event source offsets");
  check_int32(event_sources, "event sources");
  TORCH_CHECK(maximum_count > 0 && maximum_count <= 1024, "maximum count must be in [1, 1024]");
  TORCH_CHECK(event_source_offsets.numel() == event_counts.numel() + 1, "event offset extent mismatch");
  TORCH_CHECK(output.numel() == event_counts.numel(), "event output extent mismatch");
  TORCH_CHECK(input.numel() == partials.numel() && input.numel() == event_sources.numel(),
              "event source extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 32;
  while (threads < maximum_count) threads *= 2;
  shuttle_runtime_counted_event<<<event_counts.numel(), threads, 0, stream>>>(
      input.data_ptr<float>(), partials.data_ptr<float>(), output.data_ptr<float>(),
      event_counts.data_ptr<int>(), event_source_offsets.data_ptr<int>(), event_sources.data_ptr<int>(),
      event_counts.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run_runtime_counted_event_out", &run_runtime_counted_event_out);
}
""".strip()


def _phased_pipeline_source() -> str:
    return r"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from a generic phased Contract/Fold/Contract task graph; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kMaximumPipelineDepth = 32;

__device__ __forceinline__ void wait_for_generation(int* address, int generation) {
  while (atomicAdd(address, 0) != generation) __nanosleep(64);
}

__global__ void shuttle_phased_contract_fold_pipeline(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int generation_count,
    int pipeline_depth,
    int dimension) {
  __shared__ int first_ready[kMaximumPipelineDepth];
  __shared__ int state_ready[kMaximumPipelineDepth];
  __shared__ int slot_reusable[kMaximumPipelineDepth];
  __shared__ float score[kMaximumPipelineDepth];
  __shared__ float state_max[kMaximumPipelineDepth];
  __shared__ float state_sum[kMaximumPipelineDepth];
  __shared__ float state_weighted[kMaximumPipelineDepth];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (threadIdx.x < pipeline_depth) {
    first_ready[threadIdx.x] = -1;
    state_ready[threadIdx.x] = -1;
    slot_reusable[threadIdx.x] = 0;
  }
  __syncthreads();

  if (warp == 0 && lane < pipeline_depth) {
    const int slot = lane;
    for (int generation = 0; generation < generation_count; ++generation) {
      wait_for_generation(&slot_reusable[slot], generation);
      float accumulator = 0.0f;
      for (int index = 0; index < dimension; ++index) {
        accumulator = fmaf(
            query[generation * dimension + index],
            key[(generation * pipeline_depth + slot) * dimension + index],
            accumulator);
      }
      score[slot] = accumulator;
      __threadfence_block();
      atomicExch(&first_ready[slot], generation);
    }
  } else if (warp == 1 && lane < pipeline_depth) {
    const int slot = lane;
    for (int generation = 0; generation < generation_count; ++generation) {
      wait_for_generation(&first_ready[slot], generation);
      const float local_score = score[slot];
      state_max[slot] = local_score;
      state_sum[slot] = 1.0f;
      state_weighted[slot] = value[generation * pipeline_depth + slot];
      __threadfence_block();
      atomicExch(&state_ready[slot], generation);
    }
  } else if (warp == 2 && lane == 0) {
    for (int generation = 0; generation < generation_count; ++generation) {
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        wait_for_generation(&state_ready[slot], generation);
      }
      float running_max = -INFINITY;
      float running_sum = 0.0f;
      float running_weighted = 0.0f;
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        const float next_max = fmaxf(running_max, state_max[slot]);
        const float prior_scale = expf(running_max - next_max);
        const float next_scale = expf(state_max[slot] - next_max);
        running_sum = prior_scale * running_sum + next_scale * state_sum[slot];
        running_weighted = prior_scale * running_weighted + next_scale * state_weighted[slot];
        running_max = next_max;
      }
      output[generation] = running_weighted / running_sum;
      __threadfence_block();
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        atomicExch(&slot_reusable[slot], generation + 1);
      }
    }
  }
}

}  // namespace

void run_phased_contract_fold_pipeline_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && output.is_cuda(),
              "pipeline tensors must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && output.is_contiguous(),
              "pipeline tensors must be contiguous");
  TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
              value.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
              "pipeline tensors must be FP32");
  TORCH_CHECK(query.dim() == 2 && key.dim() == 3 && value.dim() == 2 && output.dim() == 1,
              "pipeline tensor ranks are invalid");
  const int generations = query.size(0);
  const int dimension = query.size(1);
  const int depth = key.size(1);
  TORCH_CHECK(depth > 0 && depth <= kMaximumPipelineDepth, "pipeline depth must be in [1, 32]");
  TORCH_CHECK(key.size(0) == generations && key.size(2) == dimension, "key extent mismatch");
  TORCH_CHECK(value.size(0) == generations && value.size(1) == depth, "value extent mismatch");
  TORCH_CHECK(output.numel() == generations, "output extent mismatch");
  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_phased_contract_fold_pipeline<<<1, 96, 0, stream>>>(
      query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(), output.data_ptr<float>(),
      generations, depth, dimension);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run_phased_contract_fold_pipeline_out", &run_phased_contract_fold_pipeline_out);
}
""".strip()
