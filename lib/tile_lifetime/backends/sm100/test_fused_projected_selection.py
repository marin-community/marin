# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent
SOURCE = BACKEND.parent.parent / "src"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(SOURCE))

from fused_projected_selection import (  # noqa: E402
    adapter_plan_from_lowering,
    audit_fused_projected_selection_source,
)

from tile_lifetime.routed_attention import IndexDomainRestriction, ProjectedBlockSelectionProgram  # noqa: E402
from tile_lifetime.sm100_selection_lowering import (  # noqa: E402
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)


def _lowering(*, scale: float = 128**-0.5, offset: int = 16128, selected_count: int = 16):
    program = ProjectedBlockSelectionProgram(
        source_input="query_hidden",
        left_weight_input="left_index_weight",
        right_weight_input="right_index_weight",
        source_count=256,
        source_feature_count=128,
        group_count=8,
        relation_feature_count=128,
        right_block_size=128,
        selected_count=selected_count,
        score_scale=scale,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=True,
        right_source_input="key_value_hidden",
        right_source_feature_count=128,
        right_count=16384,
        left_position_offset=offset,
    )
    return lower_sm100_projected_selection(program, default_sm100_selection_schedules()[1])


def test_fused_adapter_erases_positive_scale_and_lowers_causal_positions() -> None:
    default = adapter_plan_from_lowering(_lowering())
    changed_scale = adapter_plan_from_lowering(_lowering(scale=0.125))
    changed_offset = adapter_plan_from_lowering(_lowering(offset=16000))

    assert default.score_scale != changed_scale.score_scale
    assert default.scale_rewrite == changed_scale.scale_rewrite
    assert default.numerical_policy == "real_algebra_equivalent"
    assert (
        default.query_tile_size,
        default.key_value_tile_size,
        default.sparse_mode,
        default.page_size,
        default.pack_factor,
    ) == (
        changed_scale.query_tile_size,
        changed_scale.key_value_tile_size,
        changed_scale.sparse_mode,
        changed_scale.page_size,
        changed_scale.pack_factor,
    )
    assert default.query_position_offset == 16128
    assert changed_offset.query_position_offset == 16000
    assert default.local_block_by_query[0] == 126
    assert default.local_block_by_query[-1] == 127
    assert changed_offset.local_block_by_query[0] == 125
    assert changed_offset.local_block_by_query[-1] == 126
    assert default.external_semantic_kernels == ()


def test_direct_runtime_audit_has_only_generic_low_level_calls() -> None:
    audit = audit_fused_projected_selection_source()

    assert audit.clean
    assert audit.forbidden_or_opaque_calls == ()
    assert set(audit.required_low_level_calls) == {
        "_fmha_sm100_plan",
        "get_sparse_topk_module",
        "sparse_topk_select",
    }


def test_fused_adapter_rejects_unsupported_top_k_variant() -> None:
    with pytest.raises(ValueError, match="top-k 16"):
        adapter_plan_from_lowering(_lowering(selected_count=8))
