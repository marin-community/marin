# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The scoring passes' batch building and KL reduction, against hand-computed values.

Run from ``vendor/skyrl`` (the root environment has no torch on macOS)::

    uv run --frozen --extra cpu pytest \\
        ../../experiments/post_training/mismatch_probe/test_scoring_pass.py -q -o addopts=""
"""

import hashlib
import math

import numpy as np
import pytest
import torch

from experiments.post_training.mismatch_probe.scoring_pass import (
    CONTEXT_TAG,
    ProbeSequence,
    build_score_batch,
    candidate_gather_ids,
    exact_rows_from_passes,
    kl_rows_from_passes,
    read_rows_parquet,
    sequence_probe_hash,
    write_rows_parquet,
)
from experiments.post_training.mismatch_probe.sequence_kl import KLForm, ProbeBucket


def sequence(sample_id: str, tokens: list[int], tags: list[str]) -> ProbeSequence:
    return ProbeSequence(
        sample_id=sample_id,
        token_ids=tokens,
        token_type_tags=tags,
        position_tags=[0] * len(tokens),
        bucket_group="P_think",
    )


def test_sequence_probe_hash_is_sha256_of_the_token_ids():
    tokens = [1, 2, 3]
    expected = hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()
    assert sequence_probe_hash(tokens) == expected
    assert sequence_probe_hash([2, 1, 3]) != expected


def test_scored_positions_skip_context_and_the_first_token():
    sequence_row = sequence(
        "s", [10, 11, 12, 13], [CONTEXT_TAG, ProbeBucket.REASONING, CONTEXT_TAG, ProbeBucket.FINAL_MESSAGE]
    )
    assert sequence_row.scored_positions() == [1, 3]


def test_build_score_batch_pads_left_and_selects_exact_positions():
    first = sequence("a", [1, 2, 3, 4], [CONTEXT_TAG] * 4)
    second = sequence("b", [5, 6], [CONTEXT_TAG, ProbeBucket.REASONING])
    batch = build_score_batch(
        [first, second], pad_token_id=99, gather_ids=None, exact_positions=True, exact_rate=1.0, master_seed=7
    )
    assert batch["sequences"].tolist() == [[1, 2, 3, 4], [99, 99, 5, 6]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    # Rate 1.0 selects every scored token; its row-plus-one tag sits at the logits
    # position before it, whose distribution scores it.
    assert batch["probe_score_positions"].tolist() == [[0, 0, 0, 0], [0, 0, 2, 0]]


def test_build_score_batch_exact_selection_is_seeded_and_reproducible():
    tokens = list(range(20))
    tags = [ProbeBucket.REASONING] * 20
    first = sequence("a", tokens, tags)
    second = sequence("a", tokens, tags)
    one = build_score_batch(
        [first], pad_token_id=0, gather_ids=None, exact_positions=True, exact_rate=0.5, master_seed=7
    )
    two = build_score_batch(
        [second], pad_token_id=0, gather_ids=None, exact_positions=True, exact_rate=0.5, master_seed=7
    )
    assert one["probe_score_positions"].tolist() == two["probe_score_positions"].tolist()
    selected = int(np.count_nonzero(one["probe_score_positions"]))
    assert 0 < selected < 19  # the seeded mask draws, rather than taking everything or nothing


def test_candidate_gather_ids_align_the_stored_support_and_the_realized_token():
    sequence_row = sequence("s", [10, 11, 12], [CONTEXT_TAG, ProbeBucket.REASONING, ProbeBucket.FINAL_MESSAGE])
    reference_ids = np.array([[100, 101], [102, 103]])  # logits positions 0 and 1, K = 2
    ids = candidate_gather_ids(sequence_row, reference_ids, padded_len=4)
    assert ids.shape == (4, 3)
    # Left padding, then per logits position: the reference's two ids, then the
    # token that position's distribution must score.
    assert ids.tolist() == [[0, 0, 0], [100, 101, 11], [102, 103, 12], [0, 0, 0]]


def test_kl_rows_reduce_binned_kl_per_scored_position():
    sequence_row = sequence("s", [10, 11, 12], [CONTEXT_TAG, ProbeBucket.REASONING, ProbeBucket.FINAL_MESSAGE])
    top_k = 2
    # One padded batch row of width 3; logits positions 0..2.
    reference_ids = np.array([[[11, 100], [100, 101], [102, 103]]])
    log_10 = math.log(0.5)
    reference_logprobs = np.full((1, 3, top_k), log_10)
    # Candidate: 0.3 on the reference's top id, 0.2 on its second, realized token's
    # logprob appended. The tail bin is 0.5 on both sides.
    gathered = np.zeros((1, 3, top_k + 1))
    # The distribution predicting token 1 sits at logits position 0, and so on.
    gathered[0, 0] = [math.log(0.3), math.log(0.2), math.log(0.4)]
    gathered[0, 1] = [math.log(0.3), math.log(0.2), math.log(0.4)]
    reference_tail = np.full((1, 3), -math.inf)  # the support holds all of the reference's mass
    candidate_tail = np.full((1, 3), math.log(0.5))
    rows = kl_rows_from_passes(
        [sequence_row],
        reference_ids,
        reference_logprobs,
        reference_tail,
        gathered,
        candidate_tail,
        reference_id="E",
        candidate_id="B",
        candidate_step=92,
    )
    assert [row.position for row in rows] == [1, 2]
    # Hand-computed three-bin KL at position 1: support [0.5, 0.5] plus a zero tail
    # against candidate [0.3, 0.2] plus its 0.5 tail.
    expected = 0.5 * math.log(0.5 / 0.3) + 0.5 * math.log(0.5 / 0.2)
    assert rows[0].kl == pytest.approx(expected, rel=1e-12)
    assert rows[0].token_id == 11
    assert rows[0].token_type_tag is ProbeBucket.REASONING
    assert rows[0].form is KLForm.FORWARD
    assert rows[0].reference_id == "E" and rows[0].candidate_step == 92
    assert rows[0].probe_hash == sequence_row.probe_hash
    # The realized token 11 sits in the support at log(0.5).
    assert rows[0].reference_logprob == pytest.approx(log_10)
    assert rows[0].candidate_logprob == pytest.approx(math.log(0.4))
    # Position 2's token 12 is outside the support: the reference's logprob is -inf.
    assert rows[1].reference_logprob == -math.inf
    assert rows[1].token_type_tag is ProbeBucket.FINAL_MESSAGE


def test_rows_parquet_round_trip(tmp_path):
    sequence_row = sequence("s", [10, 11], [CONTEXT_TAG, ProbeBucket.REASONING])
    reference_ids = np.array([[[100, 101], [100, 101]]])
    reference_logprobs = np.full((1, 2, 2), math.log(0.5))
    gathered = np.zeros((1, 2, 3))
    gathered[0, 0] = [math.log(0.3), -20.0, math.log(0.2)]
    reference_tail = np.full((1, 2), -math.inf)
    candidate_tail = np.full((1, 2), math.log(0.7 - math.exp(-20.0)))
    rows = kl_rows_from_passes(
        [sequence_row],
        reference_ids,
        reference_logprobs,
        reference_tail,
        gathered,
        candidate_tail,
        reference_id="E",
        candidate_id="B",
        candidate_step=1,
    )
    path = tmp_path / "rows" / "kl.parquet"
    count = write_rows_parquet(rows, path)
    assert count == 1
    restored = read_rows_parquet(path)
    assert restored == rows
    # StrEnum members compare equal to their strings; the reader must still restore the enums.
    assert isinstance(restored[0].token_type_tag, ProbeBucket) and isinstance(restored[0].form, KLForm)


def test_exact_rows_pair_the_two_passes_by_coordinate():
    sequence_row = sequence("s", [10, 11, 12], [CONTEXT_TAG] * 3)
    vocabulary = torch.arange(4, dtype=torch.float64)
    reference_rows = torch.log_softmax(torch.stack([vocabulary + 1, vocabulary + 2]), dim=-1).numpy()
    candidate_rows = torch.log_softmax(torch.stack([vocabulary, vocabulary]), dim=-1).numpy()
    # Row 0 is left-padded by one: logits positions 1 and 2 score tokens 1 and 2.
    coordinates = [(0, 1), (0, 2)]
    rows = exact_rows_from_passes([sequence_row], reference_rows, coordinates, candidate_rows, coordinates, padded_len=4)
    assert [sample_id for sample_id, _position, _hash, _kl in rows] == ["s", "s"]
    for index, ((_sample_id, position, probe_hash, kl), reference_row, candidate_row) in enumerate(
        zip(rows, reference_rows, candidate_rows, strict=True)
    ):
        p = np.exp(reference_row)
        expected = float((p * (reference_row - candidate_row)).sum())
        assert kl == pytest.approx(expected, rel=1e-12)
        assert position == index + 1
        assert probe_hash == sequence_row.probe_hash


def test_exact_rows_report_the_binned_rows_token_positions():
    """The positions the builder marks, read back as the worker reports them, are scored positions."""
    short = sequence("a", [1, 2, 3], [CONTEXT_TAG, ProbeBucket.REASONING, ProbeBucket.FINAL_MESSAGE])
    long = sequence("b", [4, 5, 6, 7, 8], [CONTEXT_TAG, CONTEXT_TAG] + [ProbeBucket.REASONING] * 3)
    sequences = [short, long]
    batch = build_score_batch(
        sequences, pad_token_id=0, gather_ids=None, exact_positions=True, exact_rate=1.0, master_seed=7
    )
    tags = batch["probe_score_positions"]
    # The worker reports (tag - 1, padded logits position) for every tagged position.
    coordinates = [(int(tags[row, column]) - 1, column) for row, column in zip(*np.nonzero(tags), strict=True)]
    rows_of_vocab = np.log(np.full((len(coordinates), 2), 0.5))
    rows = exact_rows_from_passes(
        sequences, rows_of_vocab, coordinates, rows_of_vocab, coordinates, padded_len=batch["sequences"].shape[1]
    )
    reported = sorted((sample_id, position) for sample_id, position, _hash, _kl in rows)
    expected = sorted((s.sample_id, position) for s in sequences for position in s.scored_positions())
    assert reported == expected
