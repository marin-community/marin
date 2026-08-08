# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle-only MiniMax projected-selection variant-manager adapter.

This module deliberately calls ``get_fmha_variant(...OnlyScore...)``. It is a
preassembled expert semantic kernel and therefore contaminated under Shuttle's
clean-synthesis rule. Accepted execution paths must not import this module.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_projected_selection import (
    PreparedFusedProjectedSelection,
    _prepare_projected_selection,
    adapter_plan_from_lowering,
    verify_pinned_msa_sources,
)

from tile_lifetime.sm100_selection_lowering import SM100ProjectedSelectionLowering


@dataclass
class PreparedPublicProjectedSelectionOracle:
    """Opaque official score/top-k path retained only as a comparison oracle."""

    torch: Any
    program: Any
    plan: Any
    key_value_indices: Any
    group_index: Any
    query_index: Any
    local_block: Any
    physical_source_classification: str = "official_public_msa_score_topk_oracle_contaminated"

    def __call__(self, left_index: Any, right_index: Any) -> Any:
        key_pages = right_index.contiguous().reshape(
            self.program.resolved_right_count // self.program.right_block_size,
            1,
            self.program.right_block_size,
            self.program.relation_feature_count,
        )
        fmha_sm100 = importlib.import_module("fmha_sm100")
        _, scores = fmha_sm100.fmha_sm100(
            left_index.contiguous(),
            key_pages,
            key_pages,
            self.plan,
            sm_scale=float(self.program.score_scale),
            kv_indices=self.key_value_indices,
            output_o=False,
            output_maxscore=True,
        )
        if scores is None:
            raise RuntimeError("official projected-selection oracle returned no block scores")
        scores[self.group_index, self.local_block, self.query_index] = math.inf
        return fmha_sm100.sparse_topk_select(
            scores,
            self.program.selected_count,
            num_valid_pages=self.program.resolved_right_count // self.program.right_block_size,
        )


def prepare_public_projected_selection_oracle(
    lowering: SM100ProjectedSelectionLowering,
    *,
    msa_root: Path,
    device: Any,
) -> PreparedPublicProjectedSelectionOracle:
    """Prepare the official public score/top-k path for oracle comparison."""
    verify_pinned_msa_sources(msa_root)
    python_root = str((msa_root / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    torch = importlib.import_module("torch")
    fmha_sm100 = importlib.import_module("fmha_sm100")
    program = lowering.program
    query_lengths = torch.tensor([program.source_count], dtype=torch.int32)
    key_lengths = torch.tensor([program.resolved_right_count], dtype=torch.int32)
    query_offset = torch.tensor([program.left_position_offset], dtype=torch.int32)
    plan = fmha_sm100.fmha_sm100_plan(
        query_lengths,
        key_lengths,
        program.group_count,
        causal=True,
        qo_offset=query_offset,
        page_size=program.right_block_size,
        output_maxscore=True,
        num_kv_heads=1,
    )
    right_blocks = program.resolved_right_count // program.right_block_size
    query_position = torch.arange(program.source_count, dtype=torch.int64, device=device)
    query_position += program.left_position_offset
    local_block = torch.div(query_position, program.right_block_size, rounding_mode="floor")
    return PreparedPublicProjectedSelectionOracle(
        torch=torch,
        program=program,
        plan=plan,
        key_value_indices=torch.arange(right_blocks, dtype=torch.int32, device=device),
        group_index=torch.arange(program.group_count, dtype=torch.int64, device=device)[:, None],
        query_index=torch.arange(program.source_count, dtype=torch.int64, device=device)[None, :],
        local_block=local_block[None, :],
    )


def prepare_oracle_derived_projected_selection(
    lowering: SM100ProjectedSelectionLowering,
    *,
    msa_root: Path,
    device: Any,
) -> PreparedFusedProjectedSelection:
    """Prepare MiniMax's private preassembled variant for comparison only."""
    verify_pinned_msa_sources(msa_root)
    adapter_plan = adapter_plan_from_lowering(lowering)
    python_root = str((msa_root / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    torch = importlib.import_module("torch")
    jit = importlib.import_module("fmha_sm100.jit")
    dtype_code = jit._dlpack_dtype_code(torch.bfloat16)
    score_variant = jit.get_fmha_variant(
        dtype_code,
        adapter_plan.query_tile_size,
        lowering.program.source_count <= 64,
        adapter_plan.sparse_mode,
        adapter_plan.page_size,
        adapter_plan.split_key_value,
        adapter_plan.pack_factor,
    )
    return _prepare_projected_selection(
        lowering,
        msa_root=msa_root,
        device=device,
        score_runner=score_variant.run,
        physical_source_classification="oracle_derived_private_variant_manager_contaminated",
        generated_sources=None,
    )
