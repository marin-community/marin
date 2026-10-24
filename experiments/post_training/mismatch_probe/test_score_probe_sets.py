# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.post_training.mismatch_probe.score_probe_sets import (
    compact_rows,
    forward_groups,
    load_reference_shard,
    padded_rows,
    save_reference_shard,
    unique_rows,
)
from experiments.post_training.mismatch_probe.scoring_pass import ProbeSequence


def real_rows(groups):
    return [index for group in groups for index in group[: unique_rows(group)]]


def test_groups_cover_every_sequence_once_in_dp_multiples_within_budget():
    rng = np.random.default_rng(0)
    lengths = [int(length) for length in rng.integers(100, 9_000, size=301)]
    groups = forward_groups(lengths, dp=8, rows_per_group=64, token_budget=64 * 4_096)
    assert sorted(real_rows(groups)) == list(range(301))
    for group in groups:
        assert len(group) % 8 == 0 and len(group) <= 64
        assert len(group) * max(lengths[index] for index in group) <= 64 * 4_096 or unique_rows(group) == 1


def test_no_forward_holds_more_than_four_long_sequences():
    lengths = [131_072] * 9 + [70_000] * 2 + [500] * 30
    groups = forward_groups(lengths, dp=8, rows_per_group=64, long_threshold=65_536, max_long=4)
    assert sorted(real_rows(groups)) == list(range(len(lengths)))
    for group in groups:
        long = sum(lengths[index] > 65_536 for index in group[: unique_rows(group)])
        if long:
            assert len(group) == 8 and long <= 4  # one row per rank: the group is a single forward


def test_a_reference_shard_round_trips_in_compact_form(tmp_path):
    sequences = [
        ProbeSequence(f"s{row}", list(range(10, 10 + width)), ["context"] * width, [-1] * width, "P_X")
        for row, width in enumerate((7, 4))
    ]
    padded_len, top_k = 7, 3
    rng = np.random.default_rng(1)
    results = {
        "topk_ids": rng.integers(0, 50, size=(2, padded_len, top_k)),
        "topk_logprobs": rng.normal(size=(2, padded_len, top_k)).astype(np.float32),
        "topk_tail_logmass": rng.normal(size=(2, padded_len)).astype(np.float32),
    }
    path = str(tmp_path / "reference-0000.npz")
    save_reference_shard(path, sequences, results)
    loaded = load_reference_shard(path, sequences)
    widths = [7, 4]
    for key in results:
        expected = compact_rows(results[key], widths)
        assert all(np.array_equal(a, b) for a, b in zip(compact_rows(loaded[key], widths), expected, strict=True))
    assert np.array_equal(padded_rows(expected, widths, padded_len)[1, :3], np.zeros((3,), dtype=np.float32))
    with pytest.raises(ValueError, match="different group"):
        load_reference_shard(path, sequences[::-1])
