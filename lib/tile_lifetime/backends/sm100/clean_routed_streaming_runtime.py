# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct runtime for the clean SM100 routed streaming skeleton.

The runtime deliberately bypasses MiniMax MSA's public attention entry point
and semantic combine. It imports the audited physical extraction, constructs a
generic right-major schedule from ``q2k`` relation metadata, compiles the
extracted Contract/Fold skeleton directly, and finalizes its partial state with
Shuttle-generated deterministic CUDA.

The first executable slice is intentionally narrow: one flat variable-length
batch, BF16 operands, FP32 partial state, head dimension 128, and block size
128. Those constraints describe the first physical template rather than a
named sparse-attention operator.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import linecache
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from clean_routed_streaming_emitter import (
    GENERATED_RELATION_BUILDER_CLASS,
    GENERATED_RELATION_SCHEDULER_MODULE,
    MINIMAX_MSA_CUTE_ROOT,
    ExtractedSM100Sources,
    PartialMergeScheduleKind,
    PartialStateMergeProgram,
    PartialValueDType,
    extract_clean_sm100_sources,
    import_extracted_python_sources,
    render_partial_merge_cuda,
)

from tile_lifetime.sm100_routed_lowering import SM100RoutedStreamingLowering

_RELATION_MODULE_NAME = "shuttle_sm100_extracted_relation_builder"
_MERGE_EXTENSION_CACHE: dict[str, ModuleType] = {}


@dataclass(frozen=True)
class SM100RelationRuntime:
    """Right-major CSR and bounded work schedule generated from a relation."""

    left_to_right_indices: Any
    right_to_left_offsets: Any
    right_to_left_sources: Any
    scheduler_metadata: Any
    work_count: Any
    partial_slot_sources: Any
    split_counts: Any
    work_capacity: int
    cu_seqlens_q: Any
    cu_seqlens_k: Any


@dataclass(frozen=True)
class SM100RoutedStreamingResult:
    """Final value plus exposed physical partial state for inspection."""

    output: Any
    log_normalizer_partials: Any
    normalized_value_partials: Any
    relation_runtime: SM100RelationRuntime


def _exec_module(name: str, source: str) -> ModuleType:
    filename = f"<{name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    specification = importlib.util.spec_from_loader(name, loader=None)
    if specification is None:
        raise RuntimeError(f"failed to create module specification for {name}")
    module = importlib.util.module_from_spec(specification)
    prior_module = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        if prior_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior_module
        raise
    return module


def _load_merge_extension(source: str, build_directory: Path | None) -> ModuleType:
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    cached = _MERGE_EXTENSION_CACHE.get(source_hash)
    if cached is not None:
        return cached

    # Torch/CUDA are optional for CPU-only compiler tests, so importing the
    # extension builder is deferred until the executable backend is requested.
    if build_directory is not None:
        build_directory.mkdir(parents=True, exist_ok=True)
    extension = importlib.import_module("torch.utils.cpp_extension")
    module = extension.load_inline(
        name=f"shuttle_sm100_partial_merge_{source_hash[:16]}",
        cpp_sources="",
        cuda_sources=source,
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        build_directory=None if build_directory is None else str(build_directory),
        verbose=False,
    )
    _MERGE_EXTENSION_CACHE[source_hash] = module
    return module


def compile_tiled_fold_finalize(
    program: PartialStateMergeProgram,
    *,
    build_directory: Path | None = None,
) -> ModuleType:
    """Compile one generated tiled Fold-finalization program."""
    return _load_merge_extension(render_partial_merge_cuda(program), build_directory)


class SM100RoutedStreamingCallable:
    """Compile and execute one generic relation-driven Contract/Fold plan."""

    def __init__(
        self,
        msa_root: Path,
        lowering: SM100RoutedStreamingLowering,
        *,
        partial_value_dtype: PartialValueDType = PartialValueDType.BF16,
        partial_merge_schedule: PartialMergeScheduleKind = PartialMergeScheduleKind.ROW_BLOCK,
        build_directory: Path | None = None,
    ) -> None:
        self._msa_root = msa_root.resolve()
        self._lowering = lowering
        self._partial_value_dtype = partial_value_dtype
        self._partial_merge_schedule = partial_merge_schedule
        self._build_directory = build_directory
        self._sources: ExtractedSM100Sources | None = None
        self._physical_module: ModuleType | None = None
        self._scheduler_module: ModuleType | None = None
        self._relation_module: ModuleType | None = None
        self._merge_module: ModuleType | None = None
        self._compiled_physical: dict[tuple[Any, ...], Any] = {}

    @property
    def generated_sources(self) -> ExtractedSM100Sources:
        """Return the audited sources used by this callable."""
        return self._ensure_sources()

    def _ensure_sources(self) -> ExtractedSM100Sources:
        if self._sources is None:
            self._sources = extract_clean_sm100_sources(
                self._msa_root,
                self._lowering,
                paged_key_value=False,
                partial_value_dtype=self._partial_value_dtype,
                partial_merge_schedule=self._partial_merge_schedule,
            )
        return self._sources

    def _ensure_modules(self) -> tuple[ModuleType, ModuleType, ModuleType]:
        sources = self._ensure_sources()
        cute_root = (self._msa_root / MINIMAX_MSA_CUTE_ROOT).resolve()
        if str(cute_root) not in sys.path:
            sys.path.insert(0, str(cute_root))
        if self._physical_module is None:
            self._physical_module = import_extracted_python_sources(
                sources,
                msa_root=self._msa_root,
                source_directory=(None if self._build_directory is None else self._build_directory / "extracted_python"),
            )
        if self._scheduler_module is None:
            self._scheduler_module = _exec_module(
                GENERATED_RELATION_SCHEDULER_MODULE,
                sources.scheduler_source,
            )
        if self._relation_module is None:
            self._relation_module = _exec_module(
                _RELATION_MODULE_NAME,
                sources.relation_builder_source,
            )
        if self._merge_module is None:
            self._merge_module = _load_merge_extension(
                sources.merge_cuda_source,
                self._build_directory,
            )
        return self._physical_module, self._relation_module, self._merge_module

    def marshal_relation(self, q2k_indices: Any, *, total_key_tokens: int) -> SM100RelationRuntime:
        """Build right-major CSR, work items, and split-state ownership."""
        _, relation_module, _ = self._ensure_modules()
        self._validate_q2k(q2k_indices)
        torch = sys.modules["torch"]
        device = q2k_indices.device
        cu_seqlens_q = torch.tensor(
            [0, self._lowering.query_length],
            dtype=torch.int32,
            device=device,
        )
        cu_seqlens_k = torch.tensor(
            [0, total_key_tokens],
            dtype=torch.int32,
            device=device,
        )
        builder_class = getattr(relation_module, GENERATED_RELATION_BUILDER_CLASS)
        builder = builder_class()
        row_ptr, query_indices, schedule = builder(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            right_payload_extent=total_key_tokens,
            right_item_width=self._lowering.schedule.right_block_size,
            maximum_right_payload_extent=total_key_tokens,
            maximum_left_item_count=self._lowering.query_length,
            right_item_count=self._lowering.right_block_count,
            left_lanes_per_group=self._lowering.head_group_size,
            return_schedule=True,
        )
        if not schedule.enabled:
            raise RuntimeError("the first physical template requires a non-empty prepared schedule")
        if schedule.scheduler_metadata is None or schedule.work_count is None:
            raise RuntimeError("the prepared relation schedule has no work metadata")
        if schedule.qsplit_indices is None or schedule.split_counts is None:
            raise RuntimeError("the prepared relation schedule has no partial-state ownership")
        return SM100RelationRuntime(
            left_to_right_indices=q2k_indices,
            right_to_left_offsets=row_ptr,
            right_to_left_sources=query_indices,
            scheduler_metadata=schedule.scheduler_metadata,
            work_count=schedule.work_count,
            partial_slot_sources=schedule.qsplit_indices,
            split_counts=schedule.split_counts,
            work_capacity=schedule.work_capacity,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )

    def __call__(self, q2k_indices: Any, q: Any, k: Any, v: Any) -> SM100RoutedStreamingResult:
        """Execute the generated path without an opaque attention or combine call."""
        physical_module, _, merge_module = self._ensure_modules()
        self._validate_operands(q, k, v)
        relation = self.marshal_relation(q2k_indices, total_key_tokens=int(k.shape[0]))
        torch = sys.modules["torch"]
        partial_count = self._lowering.selected_count
        query_count, query_heads, head_dim = (int(value) for value in q.shape)
        value_partial_torch_dtype = {
            PartialValueDType.BF16: torch.bfloat16,
            PartialValueDType.FP32: torch.float32,
        }[self._partial_value_dtype]
        value_partials = torch.empty(
            (partial_count, query_count, query_heads, head_dim),
            dtype=value_partial_torch_dtype,
            device=q.device,
        )
        log_normalizer_partials = torch.empty(
            (partial_count, query_count, query_heads),
            dtype=torch.float32,
            device=q.device,
        )
        q_flat = q.reshape(-1, head_dim).contiguous()
        value_partials_flat = value_partials.reshape(-1, head_dim)
        q_gather4_descriptor = self._q_gather4_descriptor(q_flat)
        compiled = self._compiled_kernel(
            physical_module,
            q=q,
            k=k,
            v=v,
            relation=relation,
            value_partials_flat=value_partials_flat,
            log_normalizer_partials=log_normalizer_partials,
            q_flat=q_flat,
            q_gather4_descriptor=q_gather4_descriptor,
        )
        compiled(
            k,
            v,
            relation.right_to_left_sources,
            relation.partial_slot_sources,
            relation.right_to_left_offsets,
            relation.scheduler_metadata,
            relation.work_count,
            value_partials_flat,
            log_normalizer_partials,
            None,
            q_flat,
            q_gather4_descriptor,
            None,
            None,
            relation.cu_seqlens_q,
            relation.cu_seqlens_k,
            self._lowering.score_map.scale,
            1.0,
            self._lowering.right_block_count,
            self._lowering.key_value_heads,
            self._lowering.query_length,
            relation.work_capacity,
        )
        output = merge_module.merge(
            log_normalizer_partials,
            value_partials,
            relation.split_counts,
            self._lowering.head_group_size,
        )
        return SM100RoutedStreamingResult(
            output=output,
            log_normalizer_partials=log_normalizer_partials,
            normalized_value_partials=value_partials,
            relation_runtime=relation,
        )

    def _compiled_kernel(
        self,
        physical_module: ModuleType,
        *,
        q: Any,
        k: Any,
        v: Any,
        relation: SM100RelationRuntime,
        value_partials_flat: Any,
        log_normalizer_partials: Any,
        q_flat: Any,
        q_gather4_descriptor: Any,
    ) -> Any:
        cutlass = importlib.import_module("cutlass")
        cute = importlib.import_module("cutlass.cute")
        tensor_helpers = importlib.import_module("src.common.cute_dsl_utils")

        key = (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.dtype,
            k.dtype,
            v.dtype,
            value_partials_flat.dtype,
            relation.work_capacity,
            self._lowering.score_map.causal,
        )
        cached = self._compiled_physical.get(key)
        if cached is not None:
            return cached
        constructor = dict(self._ensure_sources().emitter_plan.physical_constructor)
        constructor["qk_dtype"] = cutlass.BFloat16
        constructor["pv_dtype"] = cutlass.BFloat16
        kernel_class = getattr(physical_module, self._ensure_sources().emitter_plan.physical_class)
        kernel = kernel_class(**constructor)
        compiled = cute.compile(
            kernel,
            tensor_helpers.to_cute_tensor(k),
            tensor_helpers.to_cute_tensor(v),
            tensor_helpers.to_cute_tensor(relation.right_to_left_sources),
            tensor_helpers.to_cute_tensor(relation.partial_slot_sources),
            tensor_helpers.to_cute_tensor(relation.right_to_left_offsets),
            tensor_helpers.to_cute_tensor(relation.scheduler_metadata),
            tensor_helpers.to_cute_tensor(relation.work_count),
            tensor_helpers.to_cute_tensor(value_partials_flat),
            tensor_helpers.to_cute_tensor(log_normalizer_partials),
            None,
            tensor_helpers.to_cute_tensor(q_flat),
            None if q_gather4_descriptor is None else tensor_helpers.to_cute_tensor(q_gather4_descriptor),
            None,
            None,
            tensor_helpers.to_cute_tensor(relation.cu_seqlens_q),
            tensor_helpers.to_cute_tensor(relation.cu_seqlens_k),
            cutlass.Float32(self._lowering.score_map.scale),
            cutlass.Float32(1.0),
            cutlass.Int32(self._lowering.right_block_count),
            cutlass.Int32(self._lowering.key_value_heads),
            cutlass.Int32(self._lowering.query_length),
            relation.work_capacity,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        self._compiled_physical[key] = compiled
        return compiled

    def _q_gather4_descriptor(self, q_flat: Any) -> Any:
        if self._lowering.head_group_size not in (1, 2, 4):
            return None
        tma_helpers = importlib.import_module("src.common.tma_utils")
        return tma_helpers.create_q_gather4_tma_desc(q_flat, box_x=64)

    def _validate_q2k(self, q2k_indices: Any) -> None:
        torch = sys.modules["torch"]
        expected = (
            self._lowering.key_value_heads,
            self._lowering.query_length,
            self._lowering.selected_count,
        )
        if tuple(q2k_indices.shape) != expected:
            raise ValueError(f"q2k relation must have shape {expected}, got {tuple(q2k_indices.shape)}")
        if q2k_indices.dtype != torch.int32:
            raise TypeError("q2k relation must use int32 block indices")
        if not q2k_indices.is_cuda or not q2k_indices.is_contiguous():
            raise ValueError("q2k relation must be a contiguous CUDA tensor")

    def _validate_operands(self, q: Any, k: Any, v: Any) -> None:
        torch = sys.modules["torch"]
        query_heads = self._lowering.key_value_heads * self._lowering.head_group_size
        expected_q = (self._lowering.query_length, query_heads, 128)
        expected_kv = (self._lowering.key_length, self._lowering.key_value_heads, 128)
        if tuple(q.shape) != expected_q:
            raise ValueError(f"Q must have shape {expected_q}, got {tuple(q.shape)}")
        if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
            raise ValueError(f"K and V must have shape {expected_kv}")
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise TypeError("the first physical template requires BF16 Q/K/V")
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("Q/K/V must be CUDA tensors")
        if q.device != k.device or q.device != v.device:
            raise ValueError("Q/K/V must share one device")
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")


def compile_routed_streaming_callable(
    msa_root: Path,
    lowering: SM100RoutedStreamingLowering,
    *,
    partial_value_dtype: PartialValueDType = PartialValueDType.BF16,
    partial_merge_schedule: PartialMergeScheduleKind = PartialMergeScheduleKind.ROW_BLOCK,
    build_directory: Path | None = None,
) -> SM100RoutedStreamingCallable:
    """Construct the lazy callable for one recovered generic lowering."""
    return SM100RoutedStreamingCallable(
        msa_root,
        lowering,
        partial_value_dtype=partial_value_dtype,
        partial_merge_schedule=partial_merge_schedule,
        build_directory=build_directory,
    )
