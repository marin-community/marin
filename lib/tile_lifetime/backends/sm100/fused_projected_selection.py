# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic SM100 projected Contract, block Fold, and Selection runtime.

The retained physical primitives are MiniMax MSA's pinned score-only SM100
QK/block-maximum mainloop and its standalone top-k indexer. The accepted path
directly instantiates the physical template from compiler parameters. The
private MiniMax variant-manager adapter lives under ``benchmarks/backends`` and
is never imported by this accepted runtime.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fused_projected_selection_emitter import (
    DirectProjectedSelectionSources,
    compile_direct_projected_selection,
)

from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    SM100SelectionStrategy,
)

MINIMAX_MSA_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
PINNED_SOURCE_SHA256 = {
    "python/fmha_sm100/api.py": "ff2c75301868e65126fceb748dbc2b08f9960f559bf0c59f5370a155509649a3",
    "python/fmha_sm100/jit.py": "23ef1c849cf41daab1f01b5aee07ac4baf7e4e64bd2869c6426b88270192b4bd",
    "python/fmha_sm100/csrc/include/fmha_cutlass_sm100.cuh": (
        "784086cd9f979e4733fd2167c94de9d23be865889564cb520a133786032c0d4e"
    ),
    "python/fmha_sm100/csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp": (
        "49e8740260d91397e49fe0e4848d19540ad18bca94c30e3e3ea17c069038edd1"
    ),
    "python/fmha_sm100/csrc/include/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp": (
        "7dd3cb52dae1e34f6e239cc25909f6bd44e5d4a0fef887ca6efe19c6a3c67dc9"
    ),
    "python/fmha_sm100/csrc/include/sparse_topk_select.cuh": (
        "6c23b2b908fb8e54614866b0393a5d2b84fbe43a26773b25bfad38fa7a33ddf3"
    ),
    "python/fmha_sm100/csrc/sparse_topk_select.cu": "b1a703fcc5a91880f7793a9a867710f2adad1469cca5e3b0211de7c325f24398",
}

_FORBIDDEN_OR_OPAQUE_CALLS = frozenset(
    {
        "fmha_sm100",
        "get_fmha_variant",
        "sparse_atten_func",
        "sparse_fmha",
    }
)
_REQUIRED_LOW_LEVEL_CALLS = frozenset(
    {
        "_fmha_sm100_plan",
        "get_sparse_topk_module",
        "sparse_topk_select",
    }
)


@dataclass(frozen=True)
class FusedProjectedSelectionAdapterPlan:
    """Compiler-visible semantics and private physical variant parameters."""

    lowering: SM100ProjectedSelectionLowering
    query_tile_size: int
    key_value_tile_size: int
    sparse_mode: int
    page_size: int
    split_key_value: bool
    pack_factor: int
    maximum_key_tiles: int
    score_scale: float
    scale_rewrite: str
    numerical_policy: str
    query_position_offset: int
    key_position_offset: int
    local_block_by_query: tuple[int, ...]
    external_semantic_kernels: tuple[str, ...]


@dataclass(frozen=True)
class FusedProjectedSelectionSourceAudit:
    """Static reachable-call audit for the accepted generated runtime."""

    called_attributes: tuple[str, ...]
    forbidden_or_opaque_calls: tuple[str, ...]
    required_low_level_calls: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.forbidden_or_opaque_calls and set(self.required_low_level_calls) == _REQUIRED_LOW_LEVEL_CALLS


@dataclass
class PreparedFusedProjectedSelection:
    """Prebuilt low-level score-only and generic Selection execution state."""

    adapter_plan: FusedProjectedSelectionAdapterPlan
    torch: Any
    physical_plan: dict[str, Any]
    score_runner: Any
    selection_module: Any
    maximum_score: Any
    selection_workspace: Any
    output_indices: Any
    key_value_indices: Any
    group_index: Any
    query_index: Any
    local_block: Any
    valid_block_count: Any
    physical_source_classification: str
    generated_sources: DirectProjectedSelectionSources | None

    def __call__(self, left_index: Any, right_index: Any) -> Any:
        """Execute score-only Contract/Fold, forced-edge Map, and top-k Selection."""
        lowering = self.adapter_plan.lowering
        program = lowering.program
        expected_left = (program.source_count, program.group_count, program.relation_feature_count)
        expected_right = (program.resolved_right_count, program.relation_feature_count)
        if tuple(left_index.shape) != expected_left or tuple(right_index.shape) != expected_right:
            raise ValueError(
                f"projected index operands must have shapes {expected_left} and {expected_right}, "
                f"found {tuple(left_index.shape)} and {tuple(right_index.shape)}"
            )
        if left_index.dtype != self.torch.bfloat16 or right_index.dtype != self.torch.bfloat16:
            raise ValueError("the pinned score-only Contract requires BF16 operands")
        if left_index.device != right_index.device or left_index.device != self.maximum_score.device:
            raise ValueError("projected index operands and prepared buffers must share one device")

        key_pages = right_index.contiguous().reshape(
            lowering.right_block_count,
            1,
            program.right_block_size,
            program.relation_feature_count,
        )
        plan = self.physical_plan
        current_stream = self.torch.cuda.current_stream().cuda_stream
        self.score_runner(
            plan["cute_workspace_buffer"],
            left_index.contiguous(),
            key_pages,
            key_pages,
            plan["qo_segment_lens"],
            plan["kv_segment_lens"],
            plan["qo_segment_offsets"],
            plan["kv_segment_offsets"],
            plan["packed_work_range"],
            plan["packed_work_info"],
            None,
            float(program.score_scale),
            1.0,
            1.0,
            1.0,
            1.0,
            plan["max_qo_len"],
            plan["qo_offset"],
            plan["num_kv_splits"],
            plan["kv_tile_begin_indices"],
            plan["kv_tile_end_indices"],
            plan["kv_split_indices"],
            plan["workspace_o"],
            plan["workspace_lse"],
            plan["num_kv_splits_per_row"],
            self.adapter_plan.query_tile_size,
            self.key_value_indices,
            plan["kv_page_indptr"],
            self.maximum_score,
            self.adapter_plan.maximum_key_tiles,
            None,
            self.adapter_plan.pack_factor,
            bool(plan["qo_len_uniform"]),
            current_stream,
        )

        # The pinned score-only mainloop emits raw per-block maxima. For a
        # positive scale, max(scale * score) == scale * max(score), so the
        # compiler moves the Map across the maximum Fold and emits it here.
        self.maximum_score.mul_(float(program.score_scale))
        # The Selection contract always retains the query-local right block.
        # This generic indexed Map is one value per (group, query), not an
        # attention operation. It executes after the fused block-maximum Fold.
        self.maximum_score[self.group_index, self.local_block, self.query_index] = math.inf
        self.selection_module.sparse_topk_select(
            self.maximum_score,
            self.output_indices,
            self.selection_workspace,
            program.selected_count,
            lowering.right_block_count,
            0,
            0,
            current_stream,
        )
        # Keep the rectangular top-k result but make underfilled causal rows
        # explicit. This is a generic DomainRestriction-to-Selection
        # legalization: indices outside the row's legal right-domain prefix
        # become the declared invalid sentinel rather than duplicate edges.
        self.output_indices.masked_fill_(
            self.output_indices >= self.valid_block_count,
            program.selection_semantics.invalid_index,
        )
        return self.output_indices


def adapter_plan_from_lowering(
    lowering: SM100ProjectedSelectionLowering,
) -> FusedProjectedSelectionAdapterPlan:
    """Legalize generic projected Selection to the pinned physical primitives."""
    if lowering.schedule.strategy is not SM100SelectionStrategy.FUSED_STREAMING_TOP_K:
        raise ValueError("the score-only adapter requires the fused projected-selection candidate")
    program = lowering.program
    if program.accumulation_dtype != "fp32" or program.projection_output_dtype != "bf16":
        raise ValueError("the pinned score-only Contract requires BF16 inputs and FP32 accumulation")
    if program.relation_feature_count != 128 or program.right_block_size != 128:
        raise ValueError("the pinned score-only Contract requires feature and right-block size 128")
    if program.selected_count != 16:
        raise ValueError("the pinned generic Selection primitive currently supports top-k 16")
    if not math.isfinite(program.score_scale) or program.score_scale <= 0:
        raise ValueError("order-only Selection can elide only a finite positive score scale")
    if program.token_restriction.predicate != "left_greater_equal_right":
        raise ValueError("the score-only mainloop requires left-position >= right-position")
    if not program.force_local_block:
        raise ValueError("the physical Selection requires an explicit forced local right block")

    query_position = np.arange(program.source_count, dtype=np.int64) + program.left_position_offset
    local_block = (query_position - program.right_position_offset) // program.right_block_size
    if np.any(local_block < 0) or np.any(local_block >= lowering.right_block_count):
        raise ValueError("forced local blocks fall outside the right domain")
    query_tile_size = 128 if program.source_count <= 128 else 256
    maximum_key_tiles = math.ceil(lowering.right_block_count / 128) * 128
    return FusedProjectedSelectionAdapterPlan(
        lowering=lowering,
        query_tile_size=query_tile_size,
        key_value_tile_size=128 if query_tile_size == 256 else 256,
        sparse_mode=2,
        page_size=128,
        split_key_value=False,
        pack_factor=1,
        maximum_key_tiles=maximum_key_tiles,
        score_scale=program.score_scale,
        scale_rewrite="positive_scale_moved_after_maximum_fold",
        numerical_policy="real_algebra_equivalent",
        query_position_offset=program.left_position_offset,
        key_position_offset=program.right_position_offset,
        local_block_by_query=tuple(int(value) for value in local_block),
        external_semantic_kernels=(),
    )


def prepare_fused_projected_selection(
    lowering: SM100ProjectedSelectionLowering,
    *,
    msa_root: Path,
    device: Any,
) -> PreparedFusedProjectedSelection:
    """Generate the direct physical Contract/Fold entry and allocate state."""
    adapter_plan = adapter_plan_from_lowering(lowering)
    if adapter_plan.query_tile_size != 256 or adapter_plan.key_value_tile_size != 128:
        raise ValueError("the direct generated instance currently supports query tile 256 and key tile 128")
    score_runner, generated_sources = compile_direct_projected_selection(msa_root)
    return _prepare_projected_selection(
        lowering,
        msa_root=msa_root,
        device=device,
        score_runner=score_runner,
        physical_source_classification="direct_generated_contract_maximum_fold",
        generated_sources=generated_sources,
    )


def _prepare_projected_selection(
    lowering: SM100ProjectedSelectionLowering,
    *,
    msa_root: Path,
    device: Any,
    score_runner: Any,
    physical_source_classification: str,
    generated_sources: DirectProjectedSelectionSources | None,
) -> PreparedFusedProjectedSelection:
    """Allocate generic planner, Fold, and Selection state for one score runner."""
    verify_pinned_msa_sources(msa_root)
    adapter_plan = adapter_plan_from_lowering(lowering)
    python_root = str((msa_root / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    torch = importlib.import_module("torch")
    plan_api = importlib.import_module("fmha_sm100.api")
    jit = importlib.import_module("fmha_sm100.jit")
    planner_device = torch.device(device)
    if planner_device.index is None:
        planner_device = torch.cuda.current_device()
    program = lowering.program
    query_lengths = torch.tensor([program.source_count], dtype=torch.int32)
    key_lengths = torch.tensor([program.resolved_right_count], dtype=torch.int32)
    query_offset = torch.tensor([program.left_position_offset], dtype=torch.int32)
    physical_plan = plan_api._fmha_sm100_plan(
        query_lengths,
        key_lengths,
        program.group_count,
        num_kv_heads=1,
        qo_offset=query_offset,
        num_kv_splits=1,
        page_size=program.right_block_size,
        output_maxscore=True,
        causal=True,
        device=planner_device,
    )
    if physical_plan["pack_factor"] != adapter_plan.pack_factor or physical_plan["num_kv_splits"] != 1:
        raise ValueError("the private planner selected a physical variant outside the clean adapter contract")
    if physical_plan["max_k_tiles"] != adapter_plan.maximum_key_tiles:
        raise ValueError("the private planner and generic lowering disagree on the padded right-block domain")

    selection_module = jit.get_sparse_topk_module()
    maximum_score = torch.full(
        (program.group_count, adapter_plan.maximum_key_tiles, program.source_count),
        -math.inf,
        dtype=torch.float32,
        device=device,
    )
    selection_workspace = torch.empty(maximum_score.numel(), dtype=torch.int32, device=device)
    output_indices = torch.empty(
        (program.source_count, program.group_count, program.selected_count),
        dtype=torch.int32,
        device=device,
    )
    key_value_indices = torch.arange(lowering.right_block_count, dtype=torch.int32, device=device)
    group_index = torch.arange(program.group_count, dtype=torch.int64, device=device)[:, None]
    query_index = torch.arange(program.source_count, dtype=torch.int64, device=device)[None, :]
    local_block = torch.tensor(adapter_plan.local_block_by_query, dtype=torch.int64, device=device)[None, :]
    valid_block_count = torch.tensor(
        np.asarray(adapter_plan.local_block_by_query, dtype=np.int32) + 1,
        dtype=torch.int32,
        device=device,
    )[:, None, None]
    return PreparedFusedProjectedSelection(
        adapter_plan=adapter_plan,
        torch=torch,
        physical_plan=physical_plan,
        score_runner=score_runner,
        selection_module=selection_module,
        maximum_score=maximum_score,
        selection_workspace=selection_workspace,
        output_indices=output_indices,
        key_value_indices=key_value_indices,
        group_index=group_index,
        query_index=query_index,
        local_block=local_block,
        valid_block_count=valid_block_count,
        physical_source_classification=physical_source_classification,
        generated_sources=generated_sources,
    )


def audit_fused_projected_selection_source() -> FusedProjectedSelectionSourceAudit:
    """Verify accepted runtime reachability excludes opaque semantic dispatch."""
    tree = ast.parse(Path(__file__).read_text())
    attributes = tuple(
        sorted(
            {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
        )
    )
    forbidden = tuple(sorted(_FORBIDDEN_OR_OPAQUE_CALLS.intersection(attributes)))
    required = tuple(sorted(_REQUIRED_LOW_LEVEL_CALLS.intersection(attributes)))
    return FusedProjectedSelectionSourceAudit(
        called_attributes=attributes,
        forbidden_or_opaque_calls=forbidden,
        required_low_level_calls=required,
    )


def verify_pinned_msa_sources(msa_root: Path) -> dict[str, str]:
    """Verify every retained upstream source against the accepted revision."""
    actual = {}
    for relative_path, expected_sha256 in PINNED_SOURCE_SHA256.items():
        path = msa_root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"pinned MSA source mismatch for {relative_path}: {digest} != {expected_sha256}")
        actual[relative_path] = digest
    return actual
