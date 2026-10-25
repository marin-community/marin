# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.post_training.mismatch_probe.score_probe_sets import (
    compact_rows,
    exact_summary,
    forward_groups,
    load_reference_shard,
    padded_rows,
    save_reference_shard,
    shard_path,
    summarize_kl,
    unique_rows,
    write_rows_shard,
)
from experiments.post_training.mismatch_probe.scoring_pass import ProbeSequence
from experiments.post_training.mismatch_probe.sequence_kl import KLForm, ProbeBucket, SequenceKLRow


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


def _row(sequence_id, position, kl, bucket):
    return SequenceKLRow(
        sequence_id=sequence_id,
        position=position,
        token_id=1,
        kl=kl,
        reference_logprob=-1.0,
        candidate_logprob=-1.5,
        candidate_step=0,
        reference_id="E",
        probe_hash="h",
        top_k=256,
        reference_tail_mass=-5.0,
        candidate_tail_mass=-5.0,
        form=KLForm.FORWARD,
        token_type_tag=bucket,
        position_tag=0,
    )


def test_set_summaries_read_every_shard_and_pair_exact_rows_by_position(tmp_path):
    out = str(tmp_path / "scores")
    write_rows_shard(shard_path(out, "P_X", "kl-B", 0, "parquet"), [_row("a", 1, 0.2, ProbeBucket.REASONING)])
    write_rows_shard(
        shard_path(out, "P_X", "kl-B", 1, "parquet"),
        [_row("b", 1, 0.4, ProbeBucket.REASONING), _row("b", 2, 0.6, ProbeBucket.FINAL_MESSAGE)],
    )
    summary = summarize_kl(out, "P_X", "kl-B")
    assert (summary["positions"], summary["sequences"]) == (3, 2)
    assert summary["token_mean_kl"] == pytest.approx(0.4) and summary["sequence_mean_kl"] == pytest.approx(0.35)
    assert summary["buckets"]["reasoning"] == {"positions": 2, "token_mean_kl": pytest.approx(0.3)}
    exact = exact_summary([(_row("a", 1, 0.2, ProbeBucket.REASONING), 0.25)])
    assert exact["all"]["relative_error_of_mean"] == pytest.approx(-0.2) and set(exact["buckets"]) == {"reasoning"}
