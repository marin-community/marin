# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the probe score pass's compact-space logit scoring.

The pass computes the reference top-K, the candidate gathers, and the exact
full-vocabulary rows from left-unpadded (compact) logits and scatters them back
to the batch's padded layout. These tests hold it against a brute-force full
log-softmax reference on random inputs, including ragged left-padded batches.
"""

import pytest
import torch

from skyrl_train.mismatch_probe.score_math import score_compacted_logits

BATCH = 3
PADDED_LEN = 12
COMPACT_LEN = PADDED_LEN  # TP=1: no sequence-alignment rounding
VOCAB = 50
TOP_K = 7
LENGTHS = (PADDED_LEN, 7, 3)


def batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(compact logits, left-padded attention mask) for a ragged batch."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(BATCH, PADDED_LEN, dtype=torch.bool)
    for row, length in enumerate(LENGTHS):
        mask[row, PADDED_LEN - length :] = True
    logits = torch.randn(BATCH, COMPACT_LEN, VOCAB + 8, generator=generator)  # padded vocabulary columns
    return logits, mask


def reference_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits[..., :VOCAB].float(), dim=-1)


def test_topk_equals_torch_topk_of_the_full_log_softmax():
    logits, mask = batch(seed=0)
    scores = score_compacted_logits(
        logits, mask, top_k=TOP_K, gather_ids=None, probe_score_positions=None, vocabulary_size=VOCAB, chunk_size=5
    )
    reference_values, reference_ids = torch.topk(reference_log_softmax(logits), TOP_K, dim=-1)
    for row, length in enumerate(LENGTHS):
        torch.testing.assert_close(scores["topk_ids"][row, mask[row]], reference_ids[row, :length])
        torch.testing.assert_close(scores["topk_logprobs"][row, mask[row]], reference_values[row, :length])
        assert torch.count_nonzero(scores["topk_ids"][row, ~mask[row]]) == 0


def test_gather_reads_the_log_softmax_at_the_given_ids():
    logits, mask = batch(seed=1)
    generator = torch.Generator().manual_seed(2)
    gather_ids = torch.randint(0, VOCAB, (BATCH, PADDED_LEN, TOP_K), generator=generator)
    scores = score_compacted_logits(
        logits,
        mask,
        top_k=None,
        gather_ids=gather_ids,
        probe_score_positions=None,
        vocabulary_size=VOCAB,
        chunk_size=None,
    )
    reference = reference_log_softmax(logits)
    for row, length in enumerate(LENGTHS):
        expected = torch.gather(reference[row, :length], -1, gather_ids[row, mask[row]].long())
        torch.testing.assert_close(scores["gathered_logprobs"][row, mask[row]], expected)


def test_exact_rows_are_the_full_log_softmax_of_the_selected_positions():
    logits, mask = batch(seed=3)
    # This micro-batch holds rows 40.. of the full batch: each selected position is
    # tagged with its batch-wide row plus one, and the coordinates report that row.
    first_row = 40
    positions = torch.zeros(BATCH, PADDED_LEN, dtype=torch.long)
    positions[0] = mask[0].long() * (first_row + 1)  # every real position of the longest row
    positions[2, PADDED_LEN - 1] = first_row + 3  # one position of the shortest row
    scores = score_compacted_logits(
        logits, mask, top_k=None, gather_ids=None, probe_score_positions=positions, vocabulary_size=VOCAB, chunk_size=4
    )
    reference = reference_log_softmax(logits)
    # Values and coordinates are produced in the chunk sweep's order; the contract
    # is their pairing, not the order.
    assert sorted(scores["exact_positions"].tolist()) == sorted(
        [[first_row, position] for position in mask[0].nonzero(as_tuple=True)[0].tolist()]
        + [[first_row + 2, PADDED_LEN - 1]]
    )
    padded_to_compact = {
        row: {int(padded): compact for compact, padded in enumerate(mask[row].nonzero(as_tuple=True)[0].tolist())}
        for row in range(BATCH)
    }
    for value, (row, position) in zip(scores["exact_logprobs"], scores["exact_positions"].tolist(), strict=True):
        local = row - first_row
        torch.testing.assert_close(value, reference[local, padded_to_compact[local][position]])


def test_exactly_one_mode_must_be_set():
    logits, mask = batch(seed=4)
    with pytest.raises(ValueError, match="exactly one"):
        score_compacted_logits(
            logits,
            mask,
            top_k=None,
            gather_ids=None,
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )
    with pytest.raises(ValueError, match="exactly one"):
        score_compacted_logits(
            logits,
            mask,
            top_k=2,
            gather_ids=torch.zeros(BATCH, PADDED_LEN, 2, dtype=torch.long),
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )


def test_shorter_compact_logits_raise():
    logits, mask = batch(seed=5)
    with pytest.raises(ValueError, match="shorter than the longest sequence"):
        score_compacted_logits(
            logits[:, :-2],
            mask,
            top_k=TOP_K,
            gather_ids=None,
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )


def test_mismatched_gather_shape_raises():
    logits, mask = batch(seed=6)
    with pytest.raises(ValueError, match="do not match the batch"):
        score_compacted_logits(
            logits,
            mask,
            top_k=None,
            gather_ids=torch.zeros(BATCH, PADDED_LEN + 1, TOP_K, dtype=torch.long),
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )
