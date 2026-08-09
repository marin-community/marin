# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a restricted EventTensorPlan to a real CTA readiness primitive."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from tile_lifetime.event_dataflow import (
    Coordinate,
    EventGenerationPolicy,
    EventMemoryScope,
    EventTensorPlan,
    TaskFamily,
    verify_event_tensor_plan,
)


class CudaEventLoweringError(ValueError):
    """A structured rejection from the bounded CUDA event lowering."""


@dataclass(frozen=True)
class CudaEventCounterLowering:
    """Generated CTA event-counter skeleton and its source-independent layout."""

    source: str
    source_sha256: str
    plan_fingerprint: str
    event_count: int
    source_count: int
    consumer_count: int
    threads_per_block: int
    event_initial_counts: tuple[int, ...]
    event_source_offsets: tuple[int, ...]
    event_sources: tuple[int, ...]
    event_consumers: tuple[int, ...]
    consumer_source_offsets: tuple[int, ...]
    consumer_sources: tuple[int, ...]
    memory_scope: EventMemoryScope
    generation_policy: EventGenerationPolicy


def generate_cuda_event_counter_lowering(plan: EventTensorPlan) -> CudaEventCounterLowering:
    """Generate one-CTA-per-event CUDA for an exact producer/finalizer relation.

    Each source task copies one FP32 contribution into a materialized partial
    slot and arrives on a CUDA block-scope barrier. The event's sole consumer
    waits on the barrier and sums the exact required source slots. The generated
    extension also contains a CTA-barrier control and a two-kernel control.

    This deliberately narrow physical skeleton validates EventTensorPlan
    readiness on hardware. It is independent of Fold, MoE, or attention names.
    """
    verify_event_tensor_plan(plan)
    _validate_cuda_event_plan(plan)
    source_family = plan.required_dependence.relation.source
    consumer_family = plan.required_dependence.relation.target
    assert isinstance(source_family, TaskFamily)
    assert isinstance(consumer_family, TaskFamily)

    source_linear = {coordinate: index for index, coordinate in enumerate(source_family.coordinates)}
    consumer_linear = {coordinate: index for index, coordinate in enumerate(consumer_family.coordinates)}
    event_linear = {coordinate: index for index, coordinate in enumerate(plan.domain.coordinates)}

    event_initial_counts = tuple(plan.initial_count.as_mapping()[coordinate] for coordinate in plan.domain.coordinates)
    event_source_offsets = [0]
    event_sources: list[int] = []
    for event_coordinate in plan.domain.coordinates:
        sources = sorted(
            source_linear[pair.source] for pair in plan.notify_relation.pairs if pair.target == event_coordinate
        )
        event_sources.extend(sources)
        event_source_offsets.append(len(event_sources))

    event_consumers = [-1] * len(plan.domain.coordinates)
    for pair in plan.trigger_relation.pairs:
        event_consumers[event_linear[pair.source]] = consumer_linear[pair.target]

    consumer_source_offsets = [0]
    consumer_sources: list[int] = []
    required = plan.required_dependence.relation
    for consumer_coordinate in consumer_family.coordinates:
        sources = sorted(source_linear[coordinate] for coordinate in required.sources_for(consumer_coordinate))
        consumer_sources.extend(sources)
        consumer_source_offsets.append(len(consumer_sources))

    maximum_count = max(event_initial_counts)
    threads_per_block = min(1024, max(32, 1 << math.ceil(math.log2(maximum_count))))
    payload = {
        "required": [{"source": list(pair.source), "target": list(pair.target)} for pair in required.pairs],
        "notify": [{"source": list(pair.source), "event": list(pair.target)} for pair in plan.notify_relation.pairs],
        "trigger": [{"event": list(pair.source), "target": list(pair.target)} for pair in plan.trigger_relation.pairs],
        "counts": event_initial_counts,
        "scope": plan.memory_scope.value,
        "generation": plan.generation_policy.value,
    }
    plan_fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = _render_cuda_source(
        event_count=len(plan.domain.coordinates),
        source_count=len(source_family.coordinates),
        consumer_count=len(consumer_family.coordinates),
        threads_per_block=threads_per_block,
        event_initial_counts=event_initial_counts,
        event_source_offsets=tuple(event_source_offsets),
        event_sources=tuple(event_sources),
        event_consumers=tuple(event_consumers),
        consumer_source_offsets=tuple(consumer_source_offsets),
        consumer_sources=tuple(consumer_sources),
        plan_fingerprint=plan_fingerprint,
    )
    return CudaEventCounterLowering(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        plan_fingerprint=plan_fingerprint,
        event_count=len(plan.domain.coordinates),
        source_count=len(source_family.coordinates),
        consumer_count=len(consumer_family.coordinates),
        threads_per_block=threads_per_block,
        event_initial_counts=event_initial_counts,
        event_source_offsets=tuple(event_source_offsets),
        event_sources=tuple(event_sources),
        event_consumers=tuple(event_consumers),
        consumer_source_offsets=tuple(consumer_source_offsets),
        consumer_sources=tuple(consumer_sources),
        memory_scope=plan.memory_scope,
        generation_policy=plan.generation_policy,
    )


def _validate_cuda_event_plan(plan: EventTensorPlan) -> None:
    reasons: list[str] = []
    if plan.memory_scope is not EventMemoryScope.CTA:
        reasons.append("the first CUDA event lowering requires CTA-scope visibility")
    if plan.generation_policy is not EventGenerationPolicy.PER_INVOCATION:
        reasons.append("the first CUDA event lowering initializes fresh event storage per kernel invocation")
    counts = plan.initial_count.as_mapping()
    if any(count <= 0 for count in counts.values()):
        reasons.append("CUDA block barriers require at least one producer arrival per event")
    if counts and max(counts.values()) > 1024:
        reasons.append("CUDA block barriers support at most 1024 producer tasks per event in this skeleton")

    trigger_count_by_event = {coordinate: 0 for coordinate in plan.domain.coordinates}
    for pair in plan.trigger_relation.pairs:
        trigger_count_by_event[pair.source] += 1
    if any(count != 1 for count in trigger_count_by_event.values()):
        reasons.append("the first CUDA event lowering requires exactly one consumer per event")

    event_count_by_source: dict[Coordinate, int] = {}
    for pair in plan.notify_relation.pairs:
        event_count_by_source[pair.source] = event_count_by_source.get(pair.source, 0) + 1
    if any(count != 1 for count in event_count_by_source.values()):
        reasons.append("the first CUDA event lowering requires each producer to notify exactly one event")
    if len(event_count_by_source) != len(plan.required_dependence.relation.source.coordinates):
        reasons.append("the first CUDA event lowering requires every producer task to notify an event")
    if reasons:
        raise CudaEventLoweringError("; ".join(reasons))


def _cuda_array(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values)


def _render_cuda_source(
    *,
    event_count: int,
    source_count: int,
    consumer_count: int,
    threads_per_block: int,
    event_initial_counts: tuple[int, ...],
    event_source_offsets: tuple[int, ...],
    event_sources: tuple[int, ...],
    event_consumers: tuple[int, ...],
    consumer_source_offsets: tuple[int, ...],
    consumer_sources: tuple[int, ...],
    plan_fingerprint: str,
) -> str:
    return f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from EventTensorPlan {plan_fingerprint}; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

namespace {{

constexpr int kEventCount = {event_count};
constexpr int kSourceCount = {source_count};
constexpr int kConsumerCount = {consumer_count};
constexpr int kThreads = {threads_per_block};
constexpr int kHostEventInitialCounts[kEventCount] = {{{_cuda_array(event_initial_counts)}}};
__device__ __constant__ int kEventInitialCounts[kEventCount] = {{{_cuda_array(event_initial_counts)}}};
__device__ __constant__ int kEventSourceOffsets[kEventCount + 1] = {{{_cuda_array(event_source_offsets)}}};
__device__ __constant__ int kEventSources[kSourceCount] = {{{_cuda_array(event_sources)}}};
__device__ __constant__ int kEventConsumers[kEventCount] = {{{_cuda_array(event_consumers)}}};
__device__ __constant__ int kConsumerSourceOffsets[kConsumerCount + 1] = {{{_cuda_array(consumer_source_offsets)}}};
__device__ __constant__ int kConsumerSources[kSourceCount] = {{{_cuda_array(consumer_sources)}}};

__device__ __forceinline__ int ordered_local_index(
    int local_index,
    int producer_count,
    int order_offset,
    int order_stride) {{
  return (local_index * order_stride + order_offset) % producer_count;
}}

__device__ __forceinline__ void delay_producer(int ordered_local, int delay_cycles) {{
  if (delay_cycles > 0) {{
    __nanosleep(static_cast<unsigned int>((ordered_local * 37 + 11) % delay_cycles));
  }}
}}

__device__ __forceinline__ void finalize_consumer(
    int consumer,
    const float* partials,
    float* output) {{
  float accumulator = 0.0f;
  for (int index = kConsumerSourceOffsets[consumer];
       index < kConsumerSourceOffsets[consumer + 1]; ++index) {{
    accumulator = __fadd_rn(accumulator, partials[kConsumerSources[index]]);
  }}
  output[consumer] = accumulator;
}}

__global__ __launch_bounds__(kThreads) void shuttle_counted_event_kernel(
    const float* input,
    float* partials,
    float* output,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  __shared__ cuda::barrier<cuda::thread_scope_block> event;
  const int event_index = blockIdx.x;
  const int producer_count = kEventInitialCounts[event_index];
  if (threadIdx.x == 0) {{
    init(&event, producer_count);
  }}
  __syncthreads();
  if (threadIdx.x >= producer_count) return;

  const int ordered_local = ordered_local_index(
      threadIdx.x, producer_count, order_offset % producer_count, order_stride);
  const int source = kEventSources[kEventSourceOffsets[event_index] + ordered_local];
  partials[source] = input[source];
  delay_producer(ordered_local, delay_cycles);
  auto token = event.arrive();
  if (threadIdx.x == 0) {{
    event.wait(cuda::std::move(token));
    finalize_consumer(kEventConsumers[event_index], partials, output);
  }}
}}

__global__ __launch_bounds__(kThreads) void shuttle_block_barrier_control_kernel(
    const float* input,
    float* partials,
    float* output,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  const int event_index = blockIdx.x;
  const int producer_count = kEventInitialCounts[event_index];
  if (threadIdx.x < producer_count) {{
    const int ordered_local = ordered_local_index(
        threadIdx.x, producer_count, order_offset % producer_count, order_stride);
    const int source = kEventSources[kEventSourceOffsets[event_index] + ordered_local];
    partials[source] = input[source];
    delay_producer(ordered_local, delay_cycles);
  }}
  __syncthreads();
  if (threadIdx.x == 0) {{
    finalize_consumer(kEventConsumers[event_index], partials, output);
  }}
}}

__global__ __launch_bounds__(kThreads) void shuttle_kernel_boundary_producer(
    const float* input,
    float* partials,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  const int event_index = blockIdx.x;
  const int producer_count = kEventInitialCounts[event_index];
  if (threadIdx.x >= producer_count) return;
  const int ordered_local = ordered_local_index(
      threadIdx.x, producer_count, order_offset % producer_count, order_stride);
  const int source = kEventSources[kEventSourceOffsets[event_index] + ordered_local];
  partials[source] = input[source];
  delay_producer(ordered_local, delay_cycles);
}}

__global__ void shuttle_kernel_boundary_consumer(const float* partials, float* output) {{
  const int consumer = blockIdx.x;
  if (threadIdx.x == 0) {{
    finalize_consumer(consumer, partials, output);
  }}
}}

void check_tensors(const torch::Tensor& input, const torch::Tensor& partials, const torch::Tensor& output) {{
  TORCH_CHECK(input.is_cuda() && partials.is_cuda() && output.is_cuda(), "event tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "event input must be FP32");
  TORCH_CHECK(partials.scalar_type() == torch::kFloat32, "event partials must be FP32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "event output must be FP32");
  TORCH_CHECK(input.is_contiguous() && partials.is_contiguous() && output.is_contiguous(),
              "event tensors must be contiguous");
  TORCH_CHECK(input.numel() == kSourceCount && partials.numel() == kSourceCount,
              "event source extent mismatch");
  TORCH_CHECK(output.numel() == kConsumerCount, "event consumer extent mismatch");
  TORCH_CHECK(input.device() == partials.device() && input.device() == output.device(),
              "event tensors must share one device");
}}

void check_order(int order_stride) {{
  TORCH_CHECK(order_stride > 0, "producer-order stride must be positive");
  for (int event_index = 0; event_index < kEventCount; ++event_index) {{
    TORCH_CHECK(std::gcd(order_stride, kHostEventInitialCounts[event_index]) == 1,
                "producer-order stride must be coprime with every event count");
  }}
}}

std::vector<int64_t> attributes(const void* kernel) {{
  cudaFuncAttributes value;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&value, kernel));
  return {{value.numRegs, value.sharedSizeBytes, value.localSizeBytes, value.maxThreadsPerBlock}};
}}

}}  // namespace

void run_counted_event_out(
    torch::Tensor input,
    torch::Tensor partials,
    torch::Tensor output,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  check_tensors(input, partials, output);
  check_order(order_stride);
  TORCH_CHECK(order_offset >= 0, "producer-order offset must be non-negative");
  TORCH_CHECK(delay_cycles >= 0, "producer delay must be non-negative");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_counted_event_kernel<<<kEventCount, kThreads, 0, stream>>>(
      input.data_ptr<float>(), partials.data_ptr<float>(), output.data_ptr<float>(),
      order_offset, order_stride, delay_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

void run_block_barrier_control_out(
    torch::Tensor input,
    torch::Tensor partials,
    torch::Tensor output,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  check_tensors(input, partials, output);
  check_order(order_stride);
  TORCH_CHECK(order_offset >= 0, "producer-order offset must be non-negative");
  TORCH_CHECK(delay_cycles >= 0, "producer delay must be non-negative");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_block_barrier_control_kernel<<<kEventCount, kThreads, 0, stream>>>(
      input.data_ptr<float>(), partials.data_ptr<float>(), output.data_ptr<float>(),
      order_offset, order_stride, delay_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

void run_kernel_boundary_control_out(
    torch::Tensor input,
    torch::Tensor partials,
    torch::Tensor output,
    int order_offset,
    int order_stride,
    int delay_cycles) {{
  check_tensors(input, partials, output);
  check_order(order_stride);
  TORCH_CHECK(order_offset >= 0, "producer-order offset must be non-negative");
  TORCH_CHECK(delay_cycles >= 0, "producer delay must be non-negative");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_kernel_boundary_producer<<<kEventCount, kThreads, 0, stream>>>(
      input.data_ptr<float>(), partials.data_ptr<float>(), order_offset, order_stride, delay_cycles);
  shuttle_kernel_boundary_consumer<<<kConsumerCount, 1, 0, stream>>>(
      partials.data_ptr<float>(), output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

std::vector<int64_t> counted_event_attributes() {{
  return attributes(reinterpret_cast<const void*>(shuttle_counted_event_kernel));
}}

std::vector<int64_t> block_barrier_attributes() {{
  return attributes(reinterpret_cast<const void*>(shuttle_block_barrier_control_kernel));
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("run_counted_event_out", &run_counted_event_out);
  module.def("run_block_barrier_control_out", &run_block_barrier_control_out);
  module.def("run_kernel_boundary_control_out", &run_kernel_boundary_control_out);
  module.def("counted_event_attributes", &counted_event_attributes);
  module.def("block_barrier_attributes", &block_barrier_attributes);
}}
""".strip()
