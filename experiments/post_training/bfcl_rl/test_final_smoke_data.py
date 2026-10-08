# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.bfcl_rl.final_smoke_data import fresh_smoke_rows


def pair(tokens: int, source_index: int) -> dict:
    return {
        "chosen_input_ids": [10, 20, tokens, 40],
        "chosen_assistant_masks": [0, 0, 1, 0],
        "rejected_input_ids": [10, 20, tokens + 1000, 50],
        "rejected_assistant_masks": [0, 0, 1, 0],
        "source_row_index": source_index,
    }


def test_fresh_canary_preserves_masks_and_avoids_repeating_a_frozen_pair():
    fresh = pair(30, -1)
    frozen = [pair(30, 123)] + [pair(value, value) for value in range(31, 95)]
    rows = fresh_smoke_rows([fresh], frozen)
    assert rows == [fresh, *frozen[1:64]]
    assert len({tuple(row["chosen_input_ids"]) for row in rows}) == 64
    assert all(row["chosen_assistant_masks"] == [0, 0, 1, 0] for row in rows)
    assert all(row["rejected_assistant_masks"] == [0, 0, 1, 0] for row in rows)
