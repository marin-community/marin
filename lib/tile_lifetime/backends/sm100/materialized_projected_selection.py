# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute projected block selection from generic physical primitives.

The first SM100 candidate intentionally materializes token scores. It composes
one generic dense Contract, a maximum Fold over fixed right blocks, an index
DomainRestriction, and a top-k Selection. This is a clean, simple baseline for
the fused streaming-selection candidate; it does not call an attention or MSA
indexing kernel.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    SM100SelectionStrategy,
)


def execute_materialized_projected_selection(
    lowering: SM100ProjectedSelectionLowering,
    left_index: Any,
    right_index: Any,
) -> Any:
    """Return canonical ``[group, left, selected]`` right-block indices.

    ``left_index`` has shape ``[left, group, feature]`` and ``right_index`` has
    shape ``[right, feature]``. The backend is deliberately expressed through
    ordinary Torch tensor primitives so the Contract/Fold/Selection boundary
    is explicit and independently measurable.
    """
    if lowering.schedule.strategy is not SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES:
        raise ValueError("the materialized executor requires the materialized block-score candidate")

    torch = importlib.import_module("torch")

    if left_index.ndim != 3 or right_index.ndim != 2:
        raise ValueError("projected relation operands must be [left,group,feature] and [right,feature]")
    left_count, group_count, feature_count = left_index.shape
    right_count, right_feature_count = right_index.shape
    program = lowering.program
    if (
        left_count != program.source_count
        or group_count != program.group_count
        or feature_count != program.relation_feature_count
        or right_count != program.resolved_right_count
        or right_feature_count != feature_count
    ):
        raise ValueError("runtime projected-relation tensors do not match the lowered semantic domains")

    score = torch.matmul(left_index.float(), right_index.float().transpose(0, 1))
    score = score * float(program.score_scale)
    query_position = torch.arange(left_count, device=score.device, dtype=torch.int64) + program.left_position_offset
    key_position = torch.arange(right_count, device=score.device, dtype=torch.int64) + program.right_position_offset
    allowed = key_position[None, None, :] <= query_position[:, None, None]
    score = torch.where(allowed, score, torch.full_like(score, -math.inf))

    block_score = score.reshape(
        left_count,
        group_count,
        lowering.right_block_count,
        program.right_block_size,
    ).amax(dim=-1)
    if program.force_local_block:
        local_block = torch.div(
            query_position - program.right_position_offset,
            program.right_block_size,
            rounding_mode="floor",
        )
        block_score.scatter_(2, local_block[:, None, None].expand(-1, group_count, 1), math.inf)

    selected_score, selected_block = torch.topk(
        block_score,
        k=program.selected_count,
        dim=-1,
        largest=True,
        sorted=False,
    )
    valid = torch.isfinite(selected_score) | torch.isposinf(selected_score)
    sentinel = torch.full_like(selected_block, lowering.right_block_count)
    canonical = torch.where(valid, selected_block, sentinel).sort(dim=-1).values
    canonical = torch.where(canonical == lowering.right_block_count, -1, canonical)
    return canonical.permute(1, 0, 2).contiguous().to(torch.int32)
