# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.post_training.mismatch_probe.score_probe_sets import (
    compact_rows,
    exact_summary,
    flip_rows,
    forward_groups,
    load_reference_shard,
    padded_rows,
    resolve_reference_roots,
    save_reference_shard,
    shard_path,
    summarize_flips,
    summarize_kl,
    unique_rows,
    write_flip_rows_shard,
    write_rows_shard,
)
from experiments.post_training.mismatch_probe.scoring_pass import ProbeSequence
from experiments.post_training.mismatch_probe.sequence_kl import KLForm, ProbeBucket, SequenceFlipRow, SequenceKLRow


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


def _flip_row(sequence_id, position, reference_top1, candidate_top1, reference_logprob):
    return SequenceFlipRow(
        sequence_id=sequence_id,
        position=position,
        reference_top1=reference_top1,
        candidate_top1=candidate_top1,
        flipped=reference_top1 != candidate_top1,
        reference_top1_logprob=reference_logprob,
        candidate_top1_logprob=reference_logprob - 1.0,
        candidate_step=10,
        reference_id="E",
        candidate_id="theta",
        probe_hash="h",
        token_type_tag=ProbeBucket.REASONING,
        position_tag=0,
    )


def test_flip_rows_compare_top1_at_the_reference_logits_positions():
    sequences = [
        ProbeSequence(f"s{row}", list(range(10, 10 + width)), ["assistant_text"] * width, [0] * width, "P_X")
        for row, width in enumerate((7, 4))
    ]
    padded_len, top_k = 7, 2
    reference = {
        # Column 0 is each pass's per-position top-1; its value only needs to be consistent.
        "topk_ids": np.stack([np.arange(padded_len * top_k).reshape(padded_len, top_k)] * 2).astype(np.int64),
        "topk_logprobs": np.full((2, padded_len, top_k), -1.0, dtype=np.float32),
    }
    candidate = {
        "topk_ids": np.stack([np.arange(padded_len * top_k).reshape(padded_len, top_k)] * 2).astype(np.int64),
        "topk_logprobs": np.full((2, padded_len, top_k), -2.0, dtype=np.float32),
    }
    # Second sequence, token position 2 (logits position 4): the candidate's top-1 differs.
    candidate["topk_ids"][1, 4, 0] = 99
    rows = flip_rows(sequences, reference, candidate, real=2, reference_id="E", candidate_id="theta", candidate_step=10)
    widths = [7, 4]
    by_sequence: dict[str, list] = {}
    for row in rows:
        by_sequence.setdefault(row.sequence_id, []).append(row)
    for row_index, sequence in enumerate(sequences):
        scored = sequence.scored_positions()
        assert [row.position for row in by_sequence[sequence.sample_id]] == scored
        for row, position in zip(by_sequence[sequence.sample_id], scored, strict=True):
            logits_position = padded_len - widths[row_index] + position - 1
            assert row.reference_top1 == reference["topk_ids"][row_index, logits_position, 0]
            assert row.flipped == (candidate["topk_ids"][row_index, logits_position, 0] != row.reference_top1)
            assert row.reference_top1_logprob == pytest.approx(-1.0)
            assert row.candidate_top1_logprob == pytest.approx(-2.0)
            assert row.token_type_tag is ProbeBucket.ASSISTANT_TEXT
    flips = [row for row in rows if row.flipped]
    assert len(flips) == 1 and flips[0].sequence_id == "s1" and flips[0].position == 2
    assert flips[0].reference_top1 == 8 and flips[0].candidate_top1 == 99


def test_reference_roots_default_to_reference_out_and_override_per_set():
    sets = {"P_X": "s3://px.jsonl", "P_chat": "s3://chat.jsonl"}
    roots = resolve_reference_roots(sets, "s3://refs", ["P_X=s3://refs-e-x"])
    assert roots == {"P_X": "s3://refs-e-x", "P_chat": "s3://refs"}
    assert resolve_reference_roots(sets, "s3://refs", None) == {name: "s3://refs" for name in sets}
    with pytest.raises(ValueError, match="must be SET=DIR"):
        resolve_reference_roots(sets, "s3://refs", ["P_R=s3://refs-e-r"])
    with pytest.raises(ValueError, match="must be SET=DIR"):
        resolve_reference_roots(sets, "s3://refs", ["unknown-set=s3://refs"])


def test_flip_summaries_report_overall_bucket_and_margin_quartile_rates(tmp_path):
    out = str(tmp_path / "scores")
    write_flip_rows_shard(
        shard_path(out, "P_X", "flip-theta", 0, "parquet"),
        [
            _flip_row("a", 1, 5, 5, -0.1),
            _flip_row("a", 2, 6, 7, -3.0),
        ],
    )
    write_flip_rows_shard(
        shard_path(out, "P_X", "flip-theta", 1, "parquet"),
        [_flip_row("b", 1, 7, 9, -1.0), _flip_row("b", 2, 8, 8, -0.5)],
    )
    summary = summarize_flips(out, "P_X", "flip-theta")
    assert (summary["positions"], summary["sequences"]) == (4, 2)
    assert summary["flip_rate"] == pytest.approx(0.5)
    assert summary["sequence_flip_rate"] == pytest.approx(0.5)
    assert summary["buckets"]["reasoning"] == {"positions": 4, "flip_rate": pytest.approx(0.5)}
    # Margins sorted: -3.0, -1.0, -0.5, -0.1 with quartile edges about -1.5, -0.75, -0.4.
    quartiles = summary["reference_margin_quartiles"]
    assert set(quartiles) == {"q1", "q2", "q3", "q4"}
    assert (quartiles["q1"]["positions"], quartiles["q1"]["flip_rate"]) == (1, pytest.approx(1.0))
    assert (quartiles["q2"]["positions"], quartiles["q2"]["flip_rate"]) == (1, pytest.approx(1.0))
    assert (quartiles["q4"]["positions"], quartiles["q4"]["flip_rate"]) == (1, pytest.approx(0.0))
