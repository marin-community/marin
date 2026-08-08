# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from materialized_projected_selection import execute_materialized_projected_selection  # noqa: E402

from tile_lifetime.routed_attention import (  # noqa: E402
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
)
from tile_lifetime.sm100_selection_lowering import SM100SelectionStrategy  # noqa: E402


def _lowering():
    program = ProjectedBlockSelectionProgram(
        source_input="query_hidden",
        left_weight_input="query_index_weight",
        right_weight_input="key_index_weight",
        source_count=4,
        source_feature_count=4,
        group_count=2,
        relation_feature_count=4,
        right_block_size=2,
        selected_count=2,
        score_scale=0.5,
        token_restriction=IndexDomainRestriction(
            predicate="left_greater_equal_right",
            left_axis="query_position",
            right_axis="key_position",
        ),
        force_local_block=True,
        right_source_input="key_value_hidden",
        right_source_feature_count=4,
        right_count=8,
        left_position_offset=4,
        right_position_offset=0,
    )
    # The production lowerer accepts only 128x128x128. The CPU behavior test
    # uses its backend contract at a smaller shape.
    return SimpleNamespace(
        program=program,
        schedule=SimpleNamespace(strategy=SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES),
        right_block_count=4,
    )


def test_materialized_selection_matches_independent_contract_fold_selection() -> None:
    lowering = _lowering()
    left = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    right = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 3.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    selected = execute_materialized_projected_selection(lowering, left, right)

    expected = np.array(
        [
            [[0, 2], [0, 2], [0, 3], [0, 3]],
            [[1, 2], [1, 2], [1, 3], [1, 3]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(selected.numpy(), expected)
