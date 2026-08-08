# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime.routed_attention import IndexDomainRestriction, ProjectedBlockSelectionProgram
from tile_lifetime.sm100_selection_lowering import (
    SM100SelectionStrategy,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)


def _program(*, force_local_block: bool = True) -> ProjectedBlockSelectionProgram:
    return ProjectedBlockSelectionProgram(
        source_input="hidden",
        left_weight_input="index_q.weight",
        right_weight_input="index_k.weight",
        source_count=512,
        source_feature_count=3072,
        group_count=4,
        relation_feature_count=128,
        right_block_size=128,
        selected_count=4,
        score_scale=128**-0.5,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=force_local_block,
    )


def test_sm100_selection_candidates_make_score_materialization_explicit() -> None:
    program = _program()
    materialized, fused = tuple(
        lower_sm100_projected_selection(program, schedule) for schedule in default_sm100_selection_schedules()
    )

    assert materialized.schedule.strategy is SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES
    assert materialized.token_score_materialization_bytes == 512 * 4 * 512 * 4
    assert materialized.block_score_materialization_bytes == 512 * 4 * 4 * 4
    assert fused.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
    assert fused.token_score_materialization_bytes == 0
    assert fused.block_score_materialization_bytes == 512 * 4 * 4 * 4
    assert materialized.relation_index_bytes == fused.relation_index_bytes == 512 * 4 * 4 * 4
    assert "Fold -> top-k Selection -> Relation" in fused.dump()
    assert "external semantics: none" in fused.dump()


def test_sm100_selection_requires_the_semantic_local_edge_rule() -> None:
    with pytest.raises(ValueError, match="forced-local"):
        lower_sm100_projected_selection(_program(force_local_block=False), default_sm100_selection_schedules()[0])
