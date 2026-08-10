# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate concrete GPU payload kernels linked to derived Event Tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from shuttle.ir import DType
from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    CudaEventFfiBuffer,
    CudaEventFfiKind,
    CudaEventFfiLowering,
)
from tile_lifetime.event_buffering import (
    EventRealizationAudit,
    EventRealizationKind,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow_adapters import SegmentedContractTaskDataflow, StreamingFoldTaskDataflow
from tile_lifetime.relation import RelationPlan


@dataclass(frozen=True)
class EventLinkedCudaFfi:
    """Typed-FFI source plus an audit of logical-to-physical readiness."""

    ffi: CudaEventFfiLowering
    event_audit: EventRealizationAudit
    physical_schedule: tuple[str, ...]


def evaluate_segmented_contract_event(
    relation: RelationPlan,
    source: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Evaluate the generated segmented Contract boundary as a CPU oracle."""
    source = np.asarray(source, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if source.ndim != 2 or weight.ndim != 3:
        raise ValueError("segmented Contract expects source [item,K] and weight [segment,K,N]")
    if source.shape[0] != relation.source_item_count or weight.shape[0] != relation.destination_count:
        raise ValueError("segmented Contract payload domains disagree with the RelationPlan")
    if source.shape[1] != weight.shape[1]:
        raise ValueError("segmented Contract reduction dimensions disagree")
    output = np.empty((relation.route_count, weight.shape[2]), dtype=np.float32)
    offsets = relation.destination_edge_offsets
    grouped_sources = relation.grouped_source_item
    for segment in range(relation.destination_count):
        begin, end = (int(value) for value in offsets[segment : segment + 2])
        output[begin:end] = source[grouped_sources[begin:end]] @ weight[segment]
    return output


def evaluate_streaming_contract_fold_event(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    domain_valid: np.ndarray,
    *,
    score_scale: float,
) -> np.ndarray:
    """Evaluate packed QK/Fold/PV payloads with an independent online Fold."""
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    domain_valid = np.asarray(domain_valid, dtype=np.bool_)
    if query.ndim != 3 or key.ndim != 4 or value.ndim != 4 or domain_valid.ndim != 4:
        raise ValueError("streaming payload ranks must be query=3 and key/value/domain=4")
    rows, query_tile, reduction = query.shape
    if key.shape[:2] != value.shape[:2] or key.shape[:3] != domain_valid.shape[:1] + domain_valid.shape[2:]:
        raise ValueError("streaming partition domains disagree")
    if key.shape[0] != rows or key.shape[3] != reduction:
        raise ValueError("streaming QK dimensions disagree")
    if domain_valid.shape[:2] != (rows, query_tile):
        raise ValueError("streaming DomainRestriction row domain disagrees")
    output = np.empty((rows, query_tile, value.shape[3]), dtype=np.float32)
    for row in range(rows):
        row_max = np.full(query_tile, -np.inf, dtype=np.float32)
        row_sum = np.zeros(query_tile, dtype=np.float32)
        weighted = np.zeros((query_tile, value.shape[3]), dtype=np.float32)
        for partition in range(key.shape[1]):
            score = query[row] @ key[row, partition].T
            score *= np.float32(score_scale)
            score = np.where(domain_valid[row, :, partition], score, -np.inf)
            tile_max = np.max(score, axis=1)
            next_max = np.maximum(row_max, tile_max)
            prior_scale = np.where(row_sum > 0, np.exp(row_max - next_max), 0.0).astype(np.float32)
            probability = np.where(
                np.isfinite(score),
                np.exp(score - next_max[:, None]),
                0.0,
            ).astype(np.float32)
            weighted = prior_scale[:, None] * weighted + probability @ value[row, partition]
            row_sum = prior_scale * row_sum + np.sum(probability, axis=1)
            row_max = next_max
        output[row] = weighted / row_sum[:, None]
    return output


def generate_segmented_contract_event_ffi(
    dataflow: SegmentedContractTaskDataflow,
    relation: RelationPlan,
    *,
    reduction_dimension: int,
    output_dimension: int,
    target_name: str,
) -> EventLinkedCudaFfi:
    """Generate a real RelationPlan-driven segmented matrix contraction."""
    if reduction_dimension <= 0 or output_dimension <= 0:
        raise ValueError("segmented Contract dimensions must be positive")
    plan = dataflow.program.event_plans[0]
    counts = tuple(int(value) for value in relation.group_count)
    offsets = tuple(int(value) for value in relation.destination_edge_offsets)
    if plan.initial_count.as_mapping() != {
        (segment, output_tile): counts[segment]
        for segment in range(dataflow.segment_count)
        for output_tile in range(dataflow.output_tile_count)
    }:
        raise ValueError("segmented Contract event counts disagree with the RelationPlan")
    if dataflow.output_tile_count != 1:
        raise ValueError("the first payload lowering requires one output tile per segment")
    edge_sources = tuple(int(value) for value in relation.grouped_source_item)
    if len(edge_sources) != relation.route_count:
        raise ValueError("RelationPlan grouped sources do not cover every valid edge")

    realization = erased_event_realization(
        plan,
        kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
        mechanism="same CTA reads each routed row before its segmented Contract",
        reason="the generated body performs payload gather and contraction in one ordered task",
    )
    audit = verify_event_realizations(dataflow.program, (realization,))
    target_symbol = _handler_symbol(target_name)
    source_count = relation.route_count
    segment_count = relation.destination_count
    source_item_count = relation.source_item_count
    source = _segmented_contract_source(
        target_symbol=target_symbol,
        source_item_count=source_item_count,
        source_count=source_count,
        segment_count=segment_count,
        reduction_dimension=reduction_dimension,
        output_dimension=output_dimension,
    )
    fingerprint = _fingerprint(
        {
            "kind": "segmented_contract",
            "counts": counts,
            "offsets": offsets,
            "edge_sources": edge_sources,
            "reduction_dimension": reduction_dimension,
            "output_dimension": output_dimension,
            "event_realization": realization.kind.value,
        }
    )
    ffi = CudaEventFfiLowering(
        kind=CudaEventFfiKind.SEGMENTED_CONTRACT,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        device_source_sha256=hashlib.sha256(_segmented_contract_kernel_source().encode()).hexdigest(),
        plan_fingerprint=fingerprint,
        target_name=target_name,
        handler_symbol=target_symbol,
        inputs=(
            CudaEventFfiBuffer("source", DType.FP32, (source_item_count, reduction_dimension)),
            CudaEventFfiBuffer("weight", DType.FP32, (segment_count, reduction_dimension, output_dimension)),
            CudaEventFfiBuffer("event_counts", DType.INT32, (segment_count,)),
            CudaEventFfiBuffer("event_offsets", DType.INT32, (segment_count + 1,)),
            CudaEventFfiBuffer("edge_sources", DType.INT32, (source_count,)),
        ),
        outputs=(CudaEventFfiBuffer("output", DType.FP32, (source_count, output_dimension)),),
    )
    return EventLinkedCudaFfi(
        ffi=ffi,
        event_audit=audit,
        physical_schedule=(
            "one CTA per runtime segment",
            "RelationPlan CSR counts/offsets define the ragged task domain",
            "each output row loads its source payload then executes a generic Contract",
            "edge-ready Event Tensor erased by same-task program order",
        ),
    )


def generate_streaming_contract_fold_event_ffi(
    dataflow: StreamingFoldTaskDataflow,
    *,
    query_tile_size: int,
    key_value_tile_size: int,
    reduction_dimension: int,
    value_dimension: int,
    score_scale: float,
    target_name: str,
) -> EventLinkedCudaFfi:
    """Generate QK/Fold/PV execution with explicit staged-buffer reuse."""
    dimensions = (query_tile_size, key_value_tile_size, reduction_dimension, value_dimension)
    if any(value <= 0 for value in dimensions):
        raise ValueError("streaming physical dimensions must be positive")
    if query_tile_size > 32:
        raise ValueError("the first reference GPU body supports at most 32 query rows per tile")
    if reduction_dimension > 128 or value_dimension > 128:
        raise ValueError("the first reference GPU body supports dimensions up to 128")
    if dataflow.pipeline_depth <= 0 or dataflow.fold_partition_count <= 0:
        raise ValueError("streaming pipeline depth and partition count must be positive")

    realizations = []
    for plan in dataflow.program.event_plans:
        endpoints = (plan.notify_relation.source, plan.trigger_relation.target)
        if endpoints in {
            (dataflow.qk_contract, dataflow.fold_partial),
            (dataflow.fold_partial, dataflow.pv_contract),
        }:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="one query-row owner executes QK, Fold update, and PV in source order",
                    reason="the selected scalar reference body retains normalized-exp state in registers",
                )
            )
        elif plan.notify_relation.source == dataflow.key_value_stage:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="CTA barrier after shared K/V stage fill",
                    reason="query-row consumers must acquire staged K/V payload",
                )
            )
        elif plan.trigger_relation.target == dataflow.key_value_stage:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="CTA barrier plus circular-slot generation advance",
                    reason="the last PV consumer must finish before a slot is overwritten",
                )
            )
        else:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="one query-row owner finalizes after its ordered partition loop",
                    reason="all partial Fold updates complete in the same owner before finalization",
                )
            )
    audit = verify_event_realizations(dataflow.program, tuple(realizations))
    target_symbol = _handler_symbol(target_name)
    source = _streaming_contract_fold_source(
        target_symbol=target_symbol,
        row_tile_count=dataflow.row_tile_count,
        partition_count=dataflow.fold_partition_count,
        pipeline_depth=dataflow.pipeline_depth,
        query_tile_size=query_tile_size,
        key_value_tile_size=key_value_tile_size,
        reduction_dimension=reduction_dimension,
        value_dimension=value_dimension,
        score_scale=score_scale,
    )
    fingerprint = _fingerprint(
        {
            "kind": "streaming_contract_fold",
            "row_tile_count": dataflow.row_tile_count,
            "partition_count": dataflow.fold_partition_count,
            "pipeline_depth": dataflow.pipeline_depth,
            "query_tile_size": query_tile_size,
            "key_value_tile_size": key_value_tile_size,
            "reduction_dimension": reduction_dimension,
            "value_dimension": value_dimension,
            "score_scale": score_scale,
            "events": [(entry.plan_name, entry.kind.value, entry.mechanism) for entry in audit.entries],
        }
    )
    ffi = CudaEventFfiLowering(
        kind=CudaEventFfiKind.STREAMING_CONTRACT_FOLD,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        device_source_sha256=hashlib.sha256(_streaming_contract_fold_kernel_source().encode()).hexdigest(),
        plan_fingerprint=fingerprint,
        target_name=target_name,
        handler_symbol=target_symbol,
        inputs=(
            CudaEventFfiBuffer(
                "query",
                DType.FP32,
                (dataflow.row_tile_count, query_tile_size, reduction_dimension),
            ),
            CudaEventFfiBuffer(
                "key",
                DType.FP32,
                (
                    dataflow.row_tile_count,
                    dataflow.fold_partition_count,
                    key_value_tile_size,
                    reduction_dimension,
                ),
            ),
            CudaEventFfiBuffer(
                "value",
                DType.FP32,
                (
                    dataflow.row_tile_count,
                    dataflow.fold_partition_count,
                    key_value_tile_size,
                    value_dimension,
                ),
            ),
            CudaEventFfiBuffer(
                "domain_valid",
                DType.INT32,
                (
                    dataflow.row_tile_count,
                    query_tile_size,
                    dataflow.fold_partition_count,
                    key_value_tile_size,
                ),
            ),
        ),
        outputs=(
            CudaEventFfiBuffer(
                "output",
                DType.FP32,
                (dataflow.row_tile_count, query_tile_size, value_dimension),
            ),
        ),
    )
    return EventLinkedCudaFfi(
        ffi=ffi,
        event_audit=audit,
        physical_schedule=(
            f"{dataflow.pipeline_depth}-slot circular shared K/V buffer",
            "cooperative K/V stage fill then CTA acquire barrier",
            "one query-row owner performs QK Contract, online normalized-exp Fold, and PV Contract",
            "CTA release barrier after the last PV consumer before slot generation advances",
        ),
    )


def _handler_symbol(target_name: str) -> str:
    symbol = target_name.replace(".", "_")
    if not symbol.isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    return symbol


def _fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _segmented_contract_kernel_source() -> str:
    return """
__global__ void shuttle_segmented_contract(
    const float* source,
    const float* weight,
    const int* event_counts,
    const int* event_offsets,
    const int* edge_sources,
    float* output) {
  const int segment = blockIdx.x;
  const int count = event_counts[segment];
  const int begin = event_offsets[segment];
  const int end = event_offsets[segment + 1];
  if (count != end - begin) return;
  for (int item = threadIdx.x; item < count * kOutputDimension; item += blockDim.x) {
    const int local_edge = item / kOutputDimension;
    const int feature = item - local_edge * kOutputDimension;
    const int edge = begin + local_edge;
    const int source_row = edge_sources[edge];
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kReductionDimension; ++reduction) {
      accumulator = fmaf(
          source[source_row * kReductionDimension + reduction],
          weight[(segment * kReductionDimension + reduction) * kOutputDimension + feature],
          accumulator);
    }
    output[edge * kOutputDimension + feature] = accumulator;
  }
}
""".strip()


def _segmented_contract_source(
    *,
    target_symbol: str,
    source_item_count: int,
    source_count: int,
    segment_count: int,
    reduction_dimension: int,
    output_dimension: int,
) -> str:
    return f"""// Generated from RelationPlan + SegmentedContract + EventTensorPlan; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace {{
constexpr int kSourceItemCount = {source_item_count};
constexpr int kSourceCount = {source_count};
constexpr int kSegmentCount = {segment_count};
constexpr int kReductionDimension = {reduction_dimension};
constexpr int kOutputDimension = {output_dimension};
std::atomic<int> call_count{{0}};

{_segmented_contract_kernel_source()}

ffi::Error ShuttleSegmentedContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 2> source,
    ffi::Buffer<ffi::F32, 3> weight,
    ffi::Buffer<ffi::S32, 1> event_counts,
    ffi::Buffer<ffi::S32, 1> event_offsets,
    ffi::Buffer<ffi::S32, 1> edge_sources,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> output) {{
  shuttle_segmented_contract<<<kSegmentCount, 256, 0, stream>>>(
      source.typed_data(), weight.typed_data(), event_counts.typed_data(),
      event_offsets.typed_data(), edge_sources.typed_data(), output->typed_data());
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("shuttle_segmented_contract: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleSegmentedContractBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleSegmentedContract,
    ShuttleSegmentedContractBinding());

extern "C" int {target_symbol}_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
""".strip()


def _streaming_contract_fold_kernel_source() -> str:
    return """
__global__ void shuttle_streaming_contract_fold(
    const float* query,
    const float* key,
    const float* value,
    const int* domain_valid,
    float* output) {
  extern __shared__ float staged[];
  float* staged_key = staged;
  float* staged_value = staged + kPipelineDepth * kKeyValueTile * kReductionDimension;
  __shared__ int slot_generation[kPipelineDepth];
  __shared__ int generation_valid;
  const int row_tile = blockIdx.x;
  if (threadIdx.x < kPipelineDepth) slot_generation[threadIdx.x] = 0;
  __syncthreads();

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float weighted[kValueDimension];
  if (threadIdx.x < kQueryTile) {
    for (int feature = 0; feature < kValueDimension; ++feature) weighted[feature] = 0.0f;
  }

  for (int partition = 0; partition < kPartitionCount; ++partition) {
    const int slot = partition % kPipelineDepth;
    const int generation = partition / kPipelineDepth;
    if (threadIdx.x == 0) generation_valid = slot_generation[slot] == generation;
    __syncthreads();
    if (!generation_valid) return;
    const int key_items = kKeyValueTile * kReductionDimension;
    for (int index = threadIdx.x; index < key_items; index += blockDim.x) {
      staged_key[slot * key_items + index] =
          key[((row_tile * kPartitionCount + partition) * kKeyValueTile * kReductionDimension) + index];
    }
    const int value_items = kKeyValueTile * kValueDimension;
    for (int index = threadIdx.x; index < value_items; index += blockDim.x) {
      staged_value[slot * value_items + index] =
          value[((row_tile * kPartitionCount + partition) * kKeyValueTile * kValueDimension) + index];
    }
    // Physical realization of key_value_stage -> QK/PV acquire readiness.
    __syncthreads();

    if (threadIdx.x < kQueryTile) {
      const int query_row = threadIdx.x;
      float score[kKeyValueTile];
      float tile_max = -INFINITY;
      for (int key_row = 0; key_row < kKeyValueTile; ++key_row) {
        const int valid_index =
            ((row_tile * kQueryTile + query_row) * kPartitionCount + partition) * kKeyValueTile + key_row;
        float accumulator = 0.0f;
        if (domain_valid[valid_index] != 0) {
          for (int reduction = 0; reduction < kReductionDimension; ++reduction) {
            accumulator = fmaf(
                query[(row_tile * kQueryTile + query_row) * kReductionDimension + reduction],
                staged_key[(slot * kKeyValueTile + key_row) * kReductionDimension + reduction],
                accumulator);
          }
          accumulator *= kScoreScale;
        } else {
          accumulator = -INFINITY;
        }
        score[key_row] = accumulator;
        tile_max = fmaxf(tile_max, accumulator);
      }
      const float next_max = fmaxf(row_max, tile_max);
      const float prior_scale = row_sum > 0.0f ? expf(row_max - next_max) : 0.0f;
      float tile_sum = 0.0f;
      for (int feature = 0; feature < kValueDimension; ++feature) weighted[feature] *= prior_scale;
      for (int key_row = 0; key_row < kKeyValueTile; ++key_row) {
        const float probability = isfinite(score[key_row]) ? expf(score[key_row] - next_max) : 0.0f;
        tile_sum += probability;
        for (int feature = 0; feature < kValueDimension; ++feature) {
          weighted[feature] = fmaf(
              probability,
              staged_value[(slot * kKeyValueTile + key_row) * kValueDimension + feature],
              weighted[feature]);
        }
      }
      row_sum = prior_scale * row_sum + tile_sum;
      row_max = next_max;
    }

    // Physical last-consumer release before the circular slot is reused.
    __syncthreads();
    if (threadIdx.x == 0) slot_generation[slot] = generation + 1;
    __syncthreads();
  }
  if (threadIdx.x < kQueryTile) {
    const int query_row = threadIdx.x;
    for (int feature = 0; feature < kValueDimension; ++feature) {
      output[(row_tile * kQueryTile + query_row) * kValueDimension + feature] = weighted[feature] / row_sum;
    }
  }
}
""".strip()


def _streaming_contract_fold_source(
    *,
    target_symbol: str,
    row_tile_count: int,
    partition_count: int,
    pipeline_depth: int,
    query_tile_size: int,
    key_value_tile_size: int,
    reduction_dimension: int,
    value_dimension: int,
    score_scale: float,
) -> str:
    shared_bytes = pipeline_depth * key_value_tile_size * (reduction_dimension + value_dimension) * 4
    return f"""// Generated from Contract + Fold + DomainRestriction + EventTensorPlan; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cmath>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace {{
constexpr int kRowTileCount = {row_tile_count};
constexpr int kPartitionCount = {partition_count};
constexpr int kPipelineDepth = {pipeline_depth};
constexpr int kQueryTile = {query_tile_size};
constexpr int kKeyValueTile = {key_value_tile_size};
constexpr int kReductionDimension = {reduction_dimension};
constexpr int kValueDimension = {value_dimension};
constexpr float kScoreScale = {score_scale:.17g}f;
constexpr int kSharedBytes = {shared_bytes};
std::atomic<int> call_count{{0}};

{_streaming_contract_fold_kernel_source()}

ffi::Error ShuttleStreamingContractFold(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 3> query,
    ffi::Buffer<ffi::F32, 4> key,
    ffi::Buffer<ffi::F32, 4> value,
    ffi::Buffer<ffi::S32, 4> domain_valid,
    ffi::Result<ffi::Buffer<ffi::F32, 3>> output) {{
  shuttle_streaming_contract_fold<<<kRowTileCount, 128, kSharedBytes, stream>>>(
      query.typed_data(), key.typed_data(), value.typed_data(),
      domain_valid.typed_data(), output->typed_data());
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("shuttle_streaming_contract_fold: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleStreamingContractFoldBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<ffi::F32, 4>>()
      .Arg<ffi::Buffer<ffi::F32, 4>>()
      .Arg<ffi::Buffer<ffi::S32, 4>>()
      .Ret<ffi::Buffer<ffi::F32, 3>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleStreamingContractFold,
    ShuttleStreamingContractFoldBinding());

extern "C" int {target_symbol}_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
""".strip()
